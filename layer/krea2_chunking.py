"""LayerStream - token-chunked execution of the Krea 2 single-stream DiT.

Same idea as the H3 node: weights already stream (ComfyUI dynamic VRAM), activations are what grows with the
image size (4 MP -> 16k tokens x hidden 6144, SwiGLU 16384, fp32 norm copies, 48-head K/V).

* norms / modulation / SwiGLU MLP / projections are token-wise -> run on slices, accumulate into the residual in place
* attention: pass 1 computes only K and V (GQA-expanded, as the stock block does) for all tokens; pass 2 recomputes
  q + gate per slice, attends the slice against the full K/V, projects and accumulates into the residual
* LastLayer (fp32 norm copy of the whole sequence): stock when it fits, slices otherwise

Residual updates use the stock expression `x + gate * out` (two roundings), not a fused addcmul.
Falls back to the stock block for: reference-image (timestep_zero_index) path, attention masks, attn patches.
"""
import logging

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.krea2.model as k2
import comfy.model_management
from comfy.ldm.flux.math import apply_rope, apply_rope1
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention_masked

_orig_block_forward = k2.SingleStreamBlock.forward
_orig_last_forward = k2.LastLayer.forward
_state = {"chunk": 0}


def _modulate(block, which, xs, scale, shift):
    norm = block.prenorm if which == "pre" else block.postnorm
    return (1 + scale) * norm(xs) + shift


def _attention_two_pass(block, x, prescale, preshift, pregate, freqs, transformer_options, chunk):
    attn = block.attn
    bs, n, _ = x.shape
    rep = attn.heads // attn.kvheads
    k_buf = v_buf = None

    # pass 1: K and V only
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        h = _modulate(block, "pre", x[:, a:b], prescale, preshift)
        k, v = attn.wk(h), attn.wv(h)
        del h
        k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
        k = attn.qknorm.knorm(k)
        k = apply_rope1(k, freqs[:, :, a:b])
        if rep != 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if k_buf is None:
            k_buf = torch.empty((bs, attn.heads, n, k.shape[-1]), dtype=k.dtype, device=k.device)
            v_buf = torch.empty((bs, attn.heads, n, v.shape[-1]), dtype=v.dtype, device=v.device)
        k_buf[:, :, a:b] = k
        v_buf[:, :, a:b] = v
        del k, v

    # pass 2: q + gate per slice, attend against full K/V, project, accumulate into the residual
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        xs = x[:, a:b]
        h = _modulate(block, "pre", xs, prescale, preshift)
        q, gate = attn.wq(h), attn.gate(h)
        del h
        q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
        q = attn.qknorm.qnorm(q)
        q = apply_rope1(q, freqs[:, :, a:b])
        out = optimized_attention_masked(AttentionTensorContainer(q), AttentionTensorContainer(k_buf),
                                         AttentionTensorContainer(v_buf), attn.heads, mask=None, skip_reshape=True,
                                         preferred_attention=attn.comfy_attention, transformer_options=transformer_options)
        o = attn.wo(out * F.sigmoid(gate))
        del out, gate, q
        xs.add_(pregate * o)
        del o
    del k_buf, v_buf


def _chunked_block_forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
    cfg = transformer_options.get("layerstream_krea2")
    patches = transformer_options.get("patches", {})
    if (not cfg or timestep_zero_index is not None or mask is not None or comfy.model_management.in_training
            or "attn1_patch" in patches or "attn1_output_patch" in patches):
        return _orig_block_forward(self, x, vec, freqs, mask=mask, timestep_zero_index=timestep_zero_index,
                                   transformer_options=transformer_options)

    mlp_chunk = int(cfg.get("mlp_chunk", 0))
    attn_chunk = int(cfg.get("attn_chunk", 0))
    _state["chunk"] = mlp_chunk or attn_chunk
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
    n = x.shape[1]

    if attn_chunk > 0:
        _attention_two_pass(self, x, prescale, preshift, pregate, freqs, transformer_options, attn_chunk)
    else:
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask, transformer_options=transformer_options)

    chunk = mlp_chunk or n
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        xs = x[:, a:b]
        h = _modulate(self, "post", xs, postscale, postshift)
        o = self.mlp(h)
        del h
        xs.add_(postgate * o)
        del o
    return x


def _chunked_last_forward(self, x, tvec):
    chunk = _state["chunk"]
    _state["chunk"] = 0
    if not chunk:
        return _orig_last_forward(self, x, tvec)
    try:
        return _orig_last_forward(self, x, tvec)
    except torch.OutOfMemoryError:
        logging.info("[LayerStream] Krea2 last layer does not fit unchunked, falling back to %d-token slices", chunk)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    scale, shift = self.modulation(tvec)
    outs = []
    for a in range(0, x.shape[1], chunk):
        h = (1 + scale) * self.norm(x[:, a:a + chunk]) + shift
        outs.append(self.linear(h))
        del h
    return torch.cat(outs, dim=1)


k2.SingleStreamBlock.forward = _chunked_block_forward
k2.LastLayer.forward = _chunked_last_forward
logging.info("[LayerStream] Krea2 block chunking available")


class LayerStreamKrea2Chunking:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mlp_chunk_tokens": ("INT", {"default": 4096, "min": 256, "max": 1_000_000, "step": 256,
                                         "tooltip": "Tokens per MLP / norm slice. Smaller = lower peak VRAM, slightly slower."}),
            "attn_chunk_tokens": ("INT", {"default": 4096, "min": 0, "max": 1_000_000, "step": 256,
                                          "tooltip": "Query tokens per attention slice (two-pass: only K/V stay resident). 0 = stock attention."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "osiworx/layer"
    DESCRIPTION = "Run Krea 2 transformer-block activations in token chunks to cut peak VRAM at large resolutions."

    def apply(self, model, mlp_chunk_tokens, attn_chunk_tokens):
        m = model.clone()
        topts = m.model_options.setdefault("transformer_options", {})
        topts["layerstream_krea2"] = {"mlp_chunk": mlp_chunk_tokens, "attn_chunk": attn_chunk_tokens}
        return (m,)


NODE_CLASS_MAPPINGS = {"LayerStreamKrea2Chunking": LayerStreamKrea2Chunking}
NODE_DISPLAY_NAME_MAPPINGS = {"LayerStreamKrea2Chunking": "LayerStream: Krea2 Chunking"}
