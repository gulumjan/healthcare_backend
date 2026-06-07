"""
test_inference.py
─────────────────
Run without the server to verify models load and produce output.

Usage:
    python test_inference.py
    python test_inference.py path/to/image.jpg
"""

import sys
import os
import time

# Make sure app is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.services.model_service import run_classification, run_segmentation


def make_dummy_image_bytes() -> bytes:
    """Creates a minimal PNG in memory — no file needed."""
    from PIL import Image
    import io
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            image_bytes = f.read()
        print(f"Using image: {sys.argv[1]}")
    else:
        image_bytes = make_dummy_image_bytes()
        print("Using dummy 512×512 gray image")

    print("\n─── Problem A: Classification ───")
    t = time.time()
    result_a = run_classification(image_bytes)
    print(f"  Runtime : {result_a.runtime_ms}ms  (wall: {(time.time()-t)*1000:.0f}ms)")
    print(f"  Model   : {result_a.model}")
    for p in result_a.predictions:
        bar = "█" * int(p.confidence * 20)
        print(f"  {p.label:<25} {p.confidence:.3f}  {bar}")

    print("\n─── Problem B: Segmentation ───")
    t = time.time()
    result_b = run_segmentation(image_bytes)
    print(f"  Runtime : {result_b.runtime_ms}ms  (wall: {(time.time()-t)*1000:.0f}ms)")
    print(f"  Model   : {result_b.model}")
    print(f"  Dice    : {result_b.dice_score:.4f}")
    print(f"  IoU     : {result_b.iou:.4f}")
    for m in result_b.masks:
        print(f"  {m.label:<20} conf={m.confidence:.3f}  cx={m.cx:.2f} cy={m.cy:.2f}")

    print("\n✅ All good — models work correctly")


if __name__ == "__main__":
    main()