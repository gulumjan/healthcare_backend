import io
import uuid
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple


# ─── Standard transforms ──────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_classification_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_segmentation_transform(size: int = 512) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Load image from bytes ────────────────────────────────────

def load_image_from_bytes(data: bytes) -> Tuple[Image.Image, str]:
    """Returns (PIL image in RGB, unique image_id)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img, str(uuid.uuid4())


# ─── Preprocess for model ─────────────────────────────────────

def preprocess_for_classification(img: Image.Image, size: int = 224) -> torch.Tensor:
    transform = get_classification_transform(size)
    return transform(img).unsqueeze(0)  # [1, C, H, W]


def preprocess_for_segmentation(img: Image.Image, size: int = 512) -> torch.Tensor:
    transform = get_segmentation_transform(size)
    return transform(img).unsqueeze(0)  # [1, C, H, W]


# ─── Convert segmentation mask → ellipse params ───────────────
# Used when ML model produces a pixel mask; converts to the
# ellipse-based format the frontend understands.

def mask_to_ellipse(binary_mask: np.ndarray) -> dict:
    """
    Fit a rough ellipse to a binary mask.
    Returns normalized (cx, cy, rx, ry, rotation).
    """
    h, w = binary_mask.shape
    ys, xs = np.where(binary_mask > 0.5)
    if len(xs) == 0:
        return {"cx": 0.5, "cy": 0.5, "rx": 0.1, "ry": 0.1, "rotation": 0.0}

    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h
    rx = float(xs.std()) / w * 2
    ry = float(ys.std()) / h * 2
    return {"cx": cx, "cy": cy, "rx": max(rx, 0.02), "ry": max(ry, 0.02), "rotation": 0.0}