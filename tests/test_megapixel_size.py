"""Plain-assert tests for wos/megapixel_size.py (no torch / ComfyUI needed).  python tests/test_megapixel_size.py"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("megapixel_size", Path(__file__).resolve().parent.parent / "wos" / "megapixel_size.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Fake:  # stands in for an IMAGE tensor [B, H, W, C]
    def __init__(self, shape):
        self.shape = shape


# square image, 1.05 MP, multiple 32 -> 1024x1024 (what the H3 workflow uses)
assert m.size_for_megapixels(1024, 1024, 1.05, 32) == (1024, 1024)
# 1 MP square snaps to the nearest multiple of 16
assert m.size_for_megapixels(512, 512, 1.0, 16) == (1008, 1008)
# aspect ratio is kept and both sides are multiples
for (w, h) in [(1920, 1080), (1000, 1500), (777, 333), (64, 4000)]:
    for mult in (8, 16, 32, 64):
        ow, oh = m.size_for_megapixels(w, h, 1.0, mult)
        assert ow % mult == 0 and oh % mult == 0 and ow >= mult and oh >= mult
        if min(w, h) > 500:  # snapping error is small only for sane sizes
            assert abs((ow / oh) - (w / h)) / (w / h) < 0.08, (w, h, mult, ow, oh)
# image_size reads [B,H,W,C] and [H,W,C]
assert m.image_size(Fake((1, 720, 1280, 3))) == (720, 1280)
assert m.image_size(Fake((720, 1280, 3))) == (720, 1280)
# node wrapper returns width, height, actual megapixels
w, h, mp = m.MegapixelSize().fit(Fake((1, 1080, 1920, 3)), 1.0, "32")
assert (w, h) == (1344, 736) and abs(mp - w * h / 1e6) < 1e-3, (w, h, mp)  # 1333 -> 1344, 750 -> 736 (nearest multiple of 32)
# bad input
for bad in (lambda: m.size_for_megapixels(0, 10, 1.0), lambda: m.size_for_megapixels(10, 10, 0), lambda: m.image_size(Fake((3, 3)))):
    try:
        bad()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
print("megapixel_size tests OK")
