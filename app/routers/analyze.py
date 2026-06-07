from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Literal

from app.schemas.response import AnalyzeResponse
from app.services.model_service import run_classification, run_segmentation

router = APIRouter()

MAX_SIZE_MB = 20
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/webp", "image/bmp",
}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    task: Literal["A", "B"] = Form(...),
):
    # ── Validate ────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use PNG, JPG, or WEBP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) // 1024 // 1024}MB). Max {MAX_SIZE_MB}MB.",
        )

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")

    # ── Run inference (offload to thread so event loop stays free) ──
    try:
        if task == "A":
            result = await run_in_threadpool(run_classification, image_bytes)
            return AnalyzeResponse(task="A", classification=result)
        else:
            result = await run_in_threadpool(run_segmentation, image_bytes)
            return AnalyzeResponse(task="B", segmentation=result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")


@router.get("/health")
async def health():
    return {"status": "ok"}