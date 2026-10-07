# osiworx-pack

> ## NO QUALITY LOSS: MiniMax H3 output is BIT-IDENTICAL with and without chunking
> The `LayerStream: H3 Chunking` node does not approximate anything. It runs the same math in slices.
> We compared the decoded video **frame by frame** against stock ComfyUI: **every frame identical (PSNR = infinity)**, verified for
> 124 frames at 6 steps and for the full 15 s / 1 MP clip (362 frames, 2 and 6 steps). The **audio track** was compared byte for byte
> on a short clip (56 frames, 6 steps): identical (a 15 s audio comparison is still to be run).
> If you see a difference in an H3 result, that is a bug: please report it.
>
> **Krea 2:** with a single chunk the result is also bit-identical (it is the same code path). With several chunks on a **bf16** model the
> matrix multiplications round differently at different slice sizes, so pixels are not bit-identical (38-41 dB against stock; the
> images look the same). We have **not** yet tested whether int8/fp8 Krea2 files behave like H3 (exact) or like bf16. Until then, for
> Krea2 treat the result as "numerically equivalent", not "bit-identical".

A ComfyUI custom node pack. Version 0.1.0, experimental. Each module is import-safe: if it cannot load in your ComfyUI, it is
skipped and the rest keeps working.

## Modules

| module | what | status |
|---|---|---|
| [`layer`](layer/README.md) | token-chunked execution of big diffusion transformers to lower **activation** memory (MiniMax H3 video, Krea 2 images). Nodes `LayerStream: H3 Chunking`, `LayerStream: Krea2 Chunking` | tested on one machine (RTX 4090, ComfyUI 0.39.0). Read its README for numbers and limits |
| [`wos`](wos/megapixel_size.py) | small utility nodes. `Megapixel Size`: takes an input image and a megapixel budget and returns `width`, `height` (aspect ratio kept, rounded to a multiple of 8/16/32/64) and the real megapixels. Handy to set the video size from the start image | pure arithmetic, no model code. Unit-tested (`tests/test_megapixel_size.py`) |

## Install

Copy this folder into `ComfyUI/custom_nodes/` (folder name `osiworx-pack`) and restart ComfyUI. Nodes appear under
`osiworx/layer` (search "LayerStream") and `osiworx/image` (search "Megapixel Size").

**Megapixel Size and H3:** use `multiple = 32` for MiniMax H3 (32-pixel tokens). The node's default is 16, which is fine for most other
video models but can give a size H3 does not divide evenly. 1.05 MP with 32 gives exactly 1024x1024 for a square image.

## Test workflows (`workflows/`)

Copy to `ComfyUI/user/default/workflows/` or drag into the UI. Pick your own model files in the loader nodes (the names are the
test machine's).

- `layerstream_krea2_lowvram.json`: Krea2 (24.5 GB bf16), 2048x2048, chunking node + tiled VAE decode.
- `layerstream_h3_15s_1mp.json`: MiniMax H3 image-to-video, about 1 MP (a square start image gives 1024x1024; width/height come from the
  `Megapixel Size` node), 357 frames (about 15 s), 6 steps, turbo LoRA, with audio. Bypass the chunking node to test the no-node route.
  To pin the size by hand instead, delete the `Megapixel Size` node and type width/height into the H3 node.

## Launch flags (`scripts/`)

The nodes handle activations; weights are streamed by ComfyUI's Dynamic VRAM. On a large card the flags are optional; to force a small
budget steer it with:

`--vram-headroom N --disable-pinned-memory` (`N` = total VRAM minus the budget you want). `scripts/run_lowvram.bat` and
`scripts/run_h3_15s.bat` are examples. Without `--vram-headroom` the card fills with weights and activations spill into shared system
memory, which is very slow.

## Honest summary of the first module

- Production report (one run, default launch flags, 24 GB card): Krea2 bf16 (24.5 GB) ran in about 30 s with the Krea2 node and did not
  finish without it. Details and caveats in [layer/README.md](layer/README.md).
- H3 15 s at 1 MP: on the 24 GB test card the launch flag alone already fixed it; the node additionally keeps PyTorch's own allocations
  under a hard 8 GB cap with **bit-identical** output, about 10 % slower. Not tested on cards below 24 GB.
- Krea2: at 1 MP the flags are enough (no node needed). At 4 MP the node lowers the floor by roughly 1 GB (stock 3 GB, node 2 GB, with tiled
  VAE); output is numerically equivalent, not bit-identical, and 2.7-4x slower.
- Full details, limits and what was not tested: [layer/README.md](layer/README.md).

## License

MIT, see [LICENSE](LICENSE).

## Layout

```
osiworx-pack/
  __init__.py          loads the modules
  pyproject.toml
  layer/               module: chunked execution (h3_chunking.py, krea2_chunking.py, README.md)
  wos/                 module: utility nodes (megapixel_size.py)
  tests/               plain-python tests (no ComfyUI needed)
  workflows/           test workflows
  scripts/             example launchers
```

Adding a module: create a folder with an `__init__.py` that exposes `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, then add its name
to `MODULES` in the top-level `__init__.py`.
