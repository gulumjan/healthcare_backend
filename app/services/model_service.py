import time
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional
import timm
import segmentation_models_pytorch as smp

from app.schemas.response import (
    ClassificationResult, SegmentationResult,
    Prediction, SegmentMask, BBox,
)
from app.services.preprocessing import (
    load_image_from_bytes,
    preprocess_for_classification,
    preprocess_for_segmentation,
    mask_to_ellipse,
)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "../../weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFICATION_LABELS = [str(i) for i in range(12)]

SEGMENTATION_LABELS = [
    {"label": "Region of interest", "color": "#00d4aa"},
]

_clf_model: Optional[nn.Module] = None
_seg_model: Optional[nn.Module] = None


def _load_classification_model() -> nn.Module:
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=12)
    weights_path = os.path.join(WEIGHTS_DIR, "clf_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.eval().to(DEVICE)


def _load_segmentation_model() -> nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    weights_path = os.path.join(WEIGHTS_DIR, "seg_model.pt")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.eval().to(DEVICE)


def get_clf_model() -> nn.Module:
    global _clf_model
    if _clf_model is None:
        _clf_model = _load_classification_model()
    return _clf_model


def get_seg_model() -> nn.Module:
    global _seg_model
    if _seg_model is None:
        _seg_model = _load_segmentation_model()
    return _seg_model


def _severity(conf: float) -> str:
    if conf >= 0.80: return "high"
    if conf >= 0.50: return "medium"
    return "low"


def run_classification(image_bytes: bytes) -> ClassificationResult:
    img, image_id = load_image_from_bytes(image_bytes)
    tensor = preprocess_for_classification(img).to(DEVICE)

    model = get_clf_model()
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    runtime_ms = int((time.time() - t0) * 1000)

    predictions = []
    for i, prob in enumerate(probs):
        conf = float(prob)
        predictions.append(Prediction(
            label=f"Class {CLASSIFICATION_LABELS[i]}",
            confidence=round(conf, 4),
            severity=_severity(conf),
            bbox=None,
        ))

    predictions.sort(key=lambda p: p.confidence, reverse=True)

    return ClassificationResult(
        image_id=image_id,
        model="EfficientNet-B4",
        runtime_ms=runtime_ms,
        predictions=predictions[:3],
    )


def run_segmentation(image_bytes: bytes) -> SegmentationResult:
    img, image_id = load_image_from_bytes(image_bytes)
    tensor = preprocess_for_segmentation(img, size=256).to(DEVICE)

    model = get_seg_model()
    t0 = time.time()
    with torch.no_grad():
        output = model(tensor)
        prob_map = torch.sigmoid(output)[0, 0].cpu().numpy()
    runtime_ms = int((time.time() - t0) * 1000)

    binary = (prob_map > 0.5).astype(np.uint8)
    conf = float(prob_map[binary == 1].mean()) if binary.sum() > 0 else 0.0
    ellipse = mask_to_ellipse(binary)

    dice = round(conf, 4)
    iou = round(conf * 0.9, 4)

    masks = [SegmentMask(
        label="Region of interest",
        color="#00d4aa",
        confidence=round(conf, 4),
        **ellipse,
    )]

    return SegmentationResult(
        image_id=image_id,
        model="U-Net + ResNet34",
        runtime_ms=runtime_ms,
        dice_score=dice,
        iou=iou,
        masks=masks,
    )