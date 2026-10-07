"""Width and height for a target megapixel count, from an input image.

ComfyUI IMAGE tensors are [batch, height, width, channels]. The node reads
that size, keeps the aspect ratio, and snaps both sides to a multiple video
models accept (16 by default).
"""

from __future__ import annotations


def snap_to_multiple(value: float, multiple: int) -> int:
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    snapped = int(value / multiple + 0.5) * multiple
    return max(multiple, snapped)


def size_for_megapixels(
    width: int,
    height: int,
    megapixels: float,
    multiple: int = 16,
) -> tuple[int, int]:
    """Return (width, height) whose area is near megapixels * 1_000_000.

    Aspect ratio follows the source. Each side is rounded to ``multiple``.
    """
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if megapixels <= 0:
        raise ValueError(f"megapixels must be > 0, got {megapixels}")

    target_pixels = float(megapixels) * 1_000_000.0
    scale = (target_pixels / (width * height)) ** 0.5
    return (
        snap_to_multiple(width * scale, multiple),
        snap_to_multiple(height * scale, multiple),
    )


def image_size(image) -> tuple[int, int]:
    """Height, width from a ComfyUI IMAGE batch (or a single HWC frame)."""
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) not in (3, 4):
        raise ValueError("expected a ComfyUI IMAGE tensor [batch, height, width, channels]")
    if len(shape) == 4:
        height, width = int(shape[1]), int(shape[2])
    else:
        height, width = int(shape[0]), int(shape[1])
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    return height, width


class MegapixelSize:
    """Fit an image's aspect ratio to a chosen megapixel budget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.05,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Target area in millions of pixels (width * height / 1,000,000).",
                    },
                ),
                "multiple": (
                    ["8", "16", "32", "64"],
                    {
                        "default": "16",
                        "tooltip": "Round width and height to this multiple. 16 is the usual video latent step.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("width", "height", "megapixels")
    FUNCTION = "fit"
    CATEGORY = "osiworx/image"

    def fit(self, image, megapixels, multiple):
        height, width = image_size(image)
        step = int(multiple)
        out_w, out_h = size_for_megapixels(width, height, float(megapixels), step)
        actual = (out_w * out_h) / 1_000_000.0
        return (out_w, out_h, round(actual, 4))


NODE_CLASS_MAPPINGS = {
    "MegapixelSize": MegapixelSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegapixelSize": "Megapixel Size",
}
