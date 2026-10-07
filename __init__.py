"""osiworx-pack - ComfyUI custom node pack.

Modules (each one is import-safe: if it cannot load in this ComfyUI it is skipped, the others keep working):
  layer  - token-chunked execution for big DiTs (activation memory): MiniMax H3, Krea 2
  wos    - small utility nodes: Megapixel Size (width/height for a megapixel budget, from an input image)
"""
import importlib
import logging

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

MODULES = ("layer", "wos")

for _name in MODULES:
    try:
        _m = importlib.import_module(f".{_name}", __name__)
        NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception as e:
        logging.warning("[osiworx-pack] module '%s' not available in this ComfyUI: %s", _name, e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
