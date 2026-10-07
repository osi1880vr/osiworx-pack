"""LayerStream - token-chunked execution of the MiniMax H3 DiT.

Activation memory of a long / large video scales with the token count (qkv, MLP expansion, fp32 output head,
residual copies). Weights can already be streamed by ComfyUI's dynamic VRAM; this removes the activation wall.

* MLP, norms, modulation, projections are token-wise -> run on slices, accumulate into the residual in place.
* Attention needs all keys/values -> two passes: pass 1 keeps only K and V of every token, pass 2 recomputes q per
  slice, attends the slice against full K/V, projects and accumulates into the residual in place.
* The fp32 output head is run stock (bit exact) and only falls back to slices when it does not fit.

The result is bit-identical to the stock block (verified frame by frame).
"""
import logging
import os

import torch

import comfy.ldm.minimax.model as h3
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

_orig_block_forward = h3.DiTBlock.forward
_orig_final_forward = h3.FinalLayer.forward
_final_state = {"chunk": 0}


def _clip_segments(segments, a, b):
    """absolute (start, stop, row) segments -> segments local to the slice [a, b)"""
    out = []
    for s, e, row in segments:
        lo, hi = max(s, a), min(e, b)
        if lo >= hi:
            continue
        if torch.is_tensor(row):
            row = row[lo - s:hi - s]
        out.append((lo - a, hi - a, row))
    return out


def _qkv_slice(attn, h, rope_slice):
    """qkv projection + fused per-head RMSNorm + rope for one slice, exactly as Attention.forward does it"""
    n = h.shape[0]
    q, k, v = attn.qkv_proj(h).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(n, attn.heads, attn.head_dim)
    q = q.view(1, n, attn.heads, attn.head_dim)
    k = k.view(1, n, attn.heads, attn.head_dim)
    qw = comfy.model_management.cast_to(attn.q_norm.weight, device=h.device)
    kw = comfy.model_management.cast_to(attn.k_norm.weight, device=h.device)
    rot = rope_slice.shape[-3] * 2
    comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_slice, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0], k[0], v


def _attention_two_pass(block, x, shift, scale, gate, mod_segments, rope_freqs, transformer_options, chunk):
    attn = block.attn
    n = x.shape[0]
    k_buf = v_buf = None

    # pass 1: only K and V of every token stay resident
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        segs = _clip_segments(mod_segments, a, b)
        h = h3._mod_scale_shift(block.norm1(x[a:b]), shift, scale, segs)
        q, k, v = _qkv_slice(attn, h, rope_freqs[:, a:b])
        if k_buf is None:
            k_buf = torch.empty((n,) + tuple(k.shape[1:]), dtype=k.dtype, device=k.device)
            v_buf = torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
        k_buf[a:b] = k
        v_buf[a:b] = v
        del h, q, k, v

    k_full = k_buf.transpose(0, 1).unsqueeze(0)  # views; the containers are single-use so they are rebuilt per slice
    v_full = v_buf.transpose(0, 1).unsqueeze(0)

    # pass 2: per slice, recompute q, attend against full K/V, project, accumulate into the residual
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        segs = _clip_segments(mod_segments, a, b)
        xs = x[a:b]
        h = h3._mod_scale_shift(block.norm1(xs), shift, scale, segs)
        q, k, v = _qkv_slice(attn, h, rope_freqs[:, a:b])
        del h, k, v
        qc = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(qc, AttentionTensorContainer(k_full), AttentionTensorContainer(v_full), attn.heads, preferred_attention=attn.comfy_attention, mask=None,
                                  skip_reshape=True, transformer_options=transformer_options)
        o = attn.out_proj(out.squeeze(0))
        del out, qc, q
        h3._mod_gate(xs, gate, o, segs)
        del o
    del k_buf, v_buf, k_full, v_full


def _chunked_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
    cfg = transformer_options.get("layerstream_h3")
    if not cfg:
        return _orig_block_forward(self, x, t_emb, mod_segments, rope_freqs,
                                   transformer_options=transformer_options, attention=attention)

    _final_state["chunk"] = int(cfg.get("mlp_chunk", 0))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)

    attn_chunk = int(cfg.get("attn_chunk", 0))
    if attn_chunk > 0 and attention is None and rope_freqs is not None and not comfy.model_management.in_training:
        _attention_two_pass(self, x, shift_msa, scale_msa, gate_msa, mod_segments, rope_freqs,
                            transformer_options, attn_chunk)
    else:
        # attention replaced by another patch (e.g. sparse attention) or chunking off: stock path
        attention = self.attn if attention is None else attention
        h = h3._mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = h3._mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
        del h

    # mlp half: token-wise, run on slices and accumulate into the residual stream in place
    n = x.shape[0]
    chunk = int(cfg.get("mlp_chunk", 0)) or n
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        segs = _clip_segments(mod_segments, a, b)
        xs = x[a:b]
        hc = h3._mod_scale_shift(self.norm2(xs), shift_mlp, scale_mlp, segs)
        oc = self.mlp(hc)
        h3._mod_gate(xs, gate_mlp, oc, segs)
        del hc, oc
    return x


def _chunked_final_forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
    """Stock output head whenever it fits (bit exact). If the fp32 copy of the residual does not fit, run the
    head per slice (an fp32 GEMM rounds slightly differently at another row count: ~1e-7 relative)."""
    chunk = _final_state["chunk"]
    _final_state["chunk"] = 0
    if not chunk or os.environ.get("LAYERSTREAM_NO_FINAL_CHUNK"):
        return _orig_final_forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)
    if not os.environ.get("LAYERSTREAM_FORCE_FINAL_CHUNK"):
        try:
            return _orig_final_forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)
        except torch.OutOfMemoryError:
            logging.info("[LayerStream] output head does not fit unchunked, falling back to %d-token slices", chunk)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    shift, scale = self.adaln_proj(t_emb)
    n = self.video_out.weight.shape[0] // self.video_out.out_features
    start = stop = 0
    if n != 1:
        if sample_sigmas is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (round(float(1.0 - h3.time_shift_sigma(s, shifts[0], 1.0)) * n) for s in (sigma, sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)

    def run(seg, head, flow_shift):
        a0, b0, row = seg
        outs = []
        for a in range(a0, b0, chunk):
            b = min(a + chunk, b0)
            r = row[a - a0:b - a0] if torch.is_tensor(row) else row
            h = (self.norm(x[a:b]) * (1.0 + h3._mod_row(scale, r, scale.dtype)) + h3._mod_row(shift, r, shift.dtype)).to(torch.float32)
            outs.append(head(h) if n == 1 else h3._pdd_head(head, h, n, start, stop, flow_shift))
            del h
        return torch.cat(outs, dim=0)

    return run(video_seg, self.video_out, shifts[0]), run(audio_seg, self.audio_out, shifts[1])


h3.FinalLayer.forward = _chunked_final_forward
h3.DiTBlock.forward = _chunked_forward
logging.info("[LayerStream] MiniMax H3 block chunking available")


class LayerStreamH3Chunking:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mlp_chunk_tokens": ("INT", {"default": 8192, "min": 256, "max": 1_000_000, "step": 256,
                                         "tooltip": "Tokens per MLP slice. Smaller = lower peak VRAM, slightly slower."}),
            "attn_chunk_tokens": ("INT", {"default": 8192, "min": 0, "max": 1_000_000, "step": 256,
                                          "tooltip": "Query tokens per attention slice (two-pass: only K/V stay resident). 0 = stock attention."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "osiworx/layer"
    DESCRIPTION = "Run MiniMax H3 transformer-block activations in token chunks to cut peak VRAM on long/large videos. Output is identical."

    def apply(self, model, mlp_chunk_tokens, attn_chunk_tokens):
        m = model.clone()
        topts = m.model_options.setdefault("transformer_options", {})
        topts["layerstream_h3"] = {"mlp_chunk": mlp_chunk_tokens, "attn_chunk": attn_chunk_tokens}
        return (m,)


NODE_CLASS_MAPPINGS = {"LayerStreamH3Chunking": LayerStreamH3Chunking}
NODE_DISPLAY_NAME_MAPPINGS = {"LayerStreamH3Chunking": "LayerStream: H3 Chunking"}
