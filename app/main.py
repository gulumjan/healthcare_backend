from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers.analyze import router as analyze_router


# ─── Startup: pre-load models so first request isn't slow ─────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up both models on startup
    from app.services.model_service import get_clf_model, get_seg_model
    print("⚡ Loading models...")
    get_clf_model()
    get_seg_model()
    print("✅ Models ready")
    yield


# ─── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Magnet AI — Medical Imaging API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)