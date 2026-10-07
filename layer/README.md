# osiworx-pack / layer (LayerStream nodes)

> ## NO QUALITY LOSS (H3): bit-identical output
> **MiniMax H3 with `LayerStream: H3 Chunking` produces exactly the same video and audio as stock ComfyUI.** Not "close": identical.
> Every frame compared (PSNR = infinity) for 124 frames at 6 steps and for the full 15 s, 1 MP clip (362 frames, 2 and 6 steps),
> including under a hard 8 GB PyTorch allocation cap. The audio track was compared byte for byte on a short clip (56 frames, 6 steps):
> identical. A 15 s audio comparison has not been run yet.
> The node only changes *how much is computed at once*, not *what* is computed.
>
> **Krea 2 is different, and we say so plainly:** one chunk = bit-identical; several chunks on a bf16 model = numerically equivalent but
> **not** bit-identical (details below). Whether int8/fp8 Krea2 files are exact is **untested**.

Token-chunked execution for big diffusion transformers in ComfyUI, to lower **activation** memory on long videos and
large images. Two nodes:

| node | model | exactness vs stock ComfyUI |
|---|---|---|
| `LayerStream: H3 Chunking` | MiniMax H3 (image/text to video) | **bit-identical** (verified frame by frame) |
| `LayerStream: Krea2 Chunking` | Krea 2 | numerically equivalent, **not** bit-identical when more than one chunk is used |

This is an experiment that was tested on **one machine**. Please read "What this is not" and "Limits" before relying on it.

## The problem it targets

ComfyUI's Dynamic VRAM (comfy-aimdo) already streams **weights** layer by layer, so a 20-25 GB model does not need
20-25 GB of VRAM. What it does not shrink is **activation** memory, which grows with the number of tokens
(video length x resolution, or image area): qkv buffers, the MLP expansion, fp32 norm/output-head copies, the residual stream.
That is what runs out first on long clips / large images.

## What it does

- Norms, modulation, MLP and projections are token-wise, so they run on slices of the sequence and accumulate into the
  residual stream in place.
- Attention needs all keys/values. Two passes: pass 1 keeps only K and V for every token; pass 2 recomputes the queries per
  slice, attends each slice against the full K/V, projects, and accumulates into the residual.
- The fp32 output head (H3) / last layer (Krea2) runs the stock way first and only falls back to slices if it runs out of memory.
- No weights are touched, no extra files are written, the model is not retrained or approximated.

## Install

Install the whole `osiworx-pack` (this is its `layer` module): copy the pack folder into `ComfyUI/custom_nodes/`. Nodes: category
`osiworx/layer`. Chain it between your model (after any LoRA)
and the sampler/scheduler/guider. Each node is only registered if your ComfyUI has that model's code (`comfy.ldm.minimax`,
`comfy.ldm.krea2`); the other keeps working. Developed against ComfyUI **0.39.0**.

Inputs of both nodes: `mlp_chunk_tokens`, `attn_chunk_tokens` (0 = stock attention). Smaller chunk = lower peak VRAM,
slower. Defaults: H3 8192, Krea2 4096.

### Launch flags (weights) - needed for a small budget, optional on a big card

The node only handles activations. On a 24 GB card with default launch flags the node alone was enough in one production test (see
"Production report" below). For a **small** VRAM budget (well below the model size) also start ComfyUI with:

```
--vram-headroom N --disable-pinned-memory
```

`N` = GB of VRAM ComfyUI keeps free; `N = (your total VRAM) - (target budget)`. `--disable-pinned-memory` stops pinned host RAM from
being counted as "shared GPU memory" (otherwise Windows readings look like VRAM is being used).
Without `--vram-headroom`, default Dynamic VRAM fills the card with weights and pushes activations into shared system memory,
which is very slow (we hit this several times during testing).

## Measured results

Test machine: RTX 4090 24 GB, 128 GB RAM, Windows, ComfyUI 0.39.0, PyTorch 2.14.1 + CUDA 13.
"Hard cap" = `torch.cuda.set_per_process_memory_fraction`, i.e. a cap on **PyTorch's own allocations**, with the streamed weights
limited separately through `--vram-headroom`. This is **not** the same as the whole-GPU number; both are listed where we have them.

### MiniMax H3, image-to-video, 1024x1024 (1.05 MP), 357 frames (about 15 s), int8 hybrid model (19.5 GB), turbo LoRA

| setup | result |
|---|---|
| stock ComfyUI workflow, default flags (author's report on a 4090) | does not fit; spills to shared memory / OOM |
| stock, no node, `--vram-headroom 10 --disable-pinned-memory` | **works** (no code needed). 2 steps: 320 s |
| node (mlp 8192, attn 8192) + hard 8 GB PyTorch cap, headroom 10 | **works**, 2 steps: 352 s (about 10 % slower), 6 steps: 969 s |
| same, 124 frames, MLP+attention chunked (4096), 6 steps | bit-identical to stock, 124/124 frames |
| same, 15 s clip, 2 steps and 6 steps, under the 8 GB cap | bit-identical to stock, 362/362 frames |
| node with a hard 4 GB cap | **fails** (the 1.5 GiB K or V buffer does not fit). Floor of this design is about 4.2 GiB |
| node with a hard 6 GB cap | failed in an early version (fixed since, **not re-tested**) |

Important context:
- On a 24 GB card the **launch flag alone** already fixed the 15 s clip. The node's added value is a much lower activation
  peak (hard 8 GB allocator cap), which matters for smaller cards or longer/larger clips than we could test. We did not run
  H3 on a card smaller than 24 GB.
- The whole-GPU peak of the 8 GB-cap run was about +11.5 GB over idle, because we used `--vram-headroom 10` for the weights.
  **We have not shown that this fits an 8 GB card.** You would raise `--vram-headroom` for that and it is untested for H3.
- 1-2 step H3 clips look broken (e.g. distorted details) with and without the node; the turbo LoRA is meant for about 6 steps.

### Krea 2 (fp16 23.9 GiB and bf16 24.5 GiB files), 8 steps, cfg 1

1024x1024, **stock ComfyUI, no node**, flags `--vram-headroom (24 - cap) --disable-pinned-memory`:

| hard cap | result vs uncapped render |
|---|---|
| 4 GB, 2 GB | identical image (both files) |
| 1 GB | 61 dB PSNR (invisible difference); whole GPU about +1.7 GB over idle |

So **weights streaming alone already handles Krea2 at 1 MP**; the node is not needed there.

2048x2048 (4 MP), bf16, **with tiled VAE decode** (`VAEDecodeTiled`, tile 512):

| setup | 3 GB cap | 2 GB cap | 1 GB cap |
|---|---|---|---|
| stock | works | fails (RMSNorm fp32 copy) | not run |
| node (chunks 4096/4096) | works | works | fails (K/V buffers / residual) |

- The node therefore buys roughly **1 GB** of lower floor at 4 MP. Without the tiled VAE the plain VAE decode ran out of memory at
  3 GB even though sampling had finished. Tiled VAE is part of the recipe.
- Output with the node at 4 MP vs stock: 38.5 dB PSNR (visually the same image).
- Whole-GPU peak at the 2 GB cap: about +3.2 GB over idle.

### Production report (second machine, default launch flags)

Reported by the author from a separate production ComfyUI (24 GB card, **no special launch flags**), Krea 2 **bf16** (24.5 GB file):

- **with** `LayerStream: Krea2 Chunking`: runs, about 30 s.
- **without** the node: did not finish in a reasonable time ("runs forever"), although the dedicated VRAM showed about 24 GB in both cases.

Our explanation (not verified on that machine): with default flags Dynamic VRAM fills the card with weights; the unchunked activations then
spill into shared system memory, which is very slow. Chunking keeps the activations small so nothing spills. Not recorded for that run:
resolution, chunk sizes, ComfyUI version, shared-GPU-memory readings. It is a single anecdotal result, not a benchmark.

### Exactness details

- H3: bit-identical in every comparison we ran.
- Krea2: with a single chunk (chunk size larger than the sequence) it is bit-identical to stock, so the code path itself is exact.
  With several chunks the bf16 matmul rounds differently at a different row count: 40.7 dB after 1 step, about 29 dB after 8 steps at
  1024x1024 with 1024-token chunks, 38.5 dB at 4 MP with 4096-token chunks. The images look the same to the eye.
- The fp32 output head in H3 behaves the same way if it has to fall back to slices (about 1e-7 relative, visible as about 36 dB over
  6 steps), which is why the stock head is used whenever it fits.

### Speed

Chunking costs time, mostly because layer weights are fetched again for every slice when they are streamed.
H3 15 s: about 10 % slower at 8 GB cap (2 steps). Krea2 1024x1024 with 1024-token chunks: about 4x slower (128 s vs 30 s); 4 MP with
4096-token chunks: about 2.7x slower (147 s vs 52-56 s). Use the largest chunk that fits.

## What this is not

- Not a general "run any model on any GPU" tool. It is two model-specific nodes.
- Not a replacement for Dynamic VRAM; it depends on it for the weights.
- Not tested below 24 GB of physical VRAM. All "caps" are software caps on a 24 GB card.
- Not a speedup. It trades time for memory.

## Limits

- Tested on one machine and one ComfyUI version; no tests on other GPUs, on low system RAM, or on slow disks (128 GB RAM hides the
  cost of streaming weights from disk).
- H3 floor is about 4.2 GiB of activations (K and V of the whole sequence stay resident). Going lower would need K/V offload.
- Krea2 falls back to the stock block for reference-image editing (`timestep_zero_index`), attention masks and attention patches
  (no savings there). H3 falls back when attention is replaced by another patch (e.g. sparse attention).
- Only a plain VAE decode was measured for H3 (it fit inside the 8 GB cap). `VAEDecodeTiled` was only tested with the Krea2 VAE.
- Not tested: batch sizes above 1, LoRA stacks beyond the H3 turbo LoRA, controlnets, or samplers other than the ones used in the
  bundled workflows (euler for Krea2, res_multistep for H3).
- The nodes patch `DiTBlock.forward` / `FinalLayer.forward` (H3) and `SingleStreamBlock.forward` / `LastLayer.forward` (Krea2) at import
  time. They run the stock code unless the node is connected, but they depend on ComfyUI internals and can break when those change.
- Debug switches via environment variables: `LAYERSTREAM_FORCE_FINAL_CHUNK=1`, `LAYERSTREAM_NO_FINAL_CHUNK=1` (H3 output head).
