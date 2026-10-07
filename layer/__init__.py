"""osiworx-pack / layer: token-chunked execution for big diffusion transformers (lower activation memory).

Nodes: "LayerStream: H3 Chunking" (MiniMax H3), "LayerStream: Krea2 Chunking" (Krea 2).
Each model is import-safe on its own: if this ComfyUI does not have that model's code the matching node is not registered.
"""
import importlib
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in ("h3_chunking", "krea2_chunking"):
    try:
        _m = importlib.import_module(f".{_mod}", __name__)
        NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:  # missing model code, API change, ...
        logging.warning("[osiworx-pack] layer.%s not available in this ComfyUI: %s", _mod, e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
