from pydantic import BaseModel
from typing import Literal, Optional


# ─── Shared ───────────────────────────────────────────────────

class BBox(BaseModel):
    x: float  # normalized 0..1
    y: float
    w: float
    h: float


# ─── Problem A — Classification ───────────────────────────────

class Prediction(BaseModel):
    label: str
    confidence: float
    severity: Literal["high", "medium", "low"]
    bbox: Optional[BBox] = None


class ClassificationResult(BaseModel):
    image_id: str
    model: str
    runtime_ms: int
    predictions: list[Prediction]


# ─── Problem B — Segmentation ─────────────────────────────────

class SegmentMask(BaseModel):
    label: str
    color: str
    cx: float   # normalized center x
    cy: float
    rx: float   # normalized radius x
    ry: float
    rotation: float
    confidence: float


class SegmentationResult(BaseModel):
    image_id: str
    model: str
    runtime_ms: int
    dice_score: float
    iou: float
    masks: list[SegmentMask]


# ─── Unified response ─────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    task: Literal["A", "B"]
    classification: Optional[ClassificationResult] = None
    segmentation: Optional[SegmentationResult] = None
    error: Optional[str] = None