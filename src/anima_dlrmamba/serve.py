from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from .config import load_config
from .infer import run_inference


app = FastAPI(title="DLRMamba Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "module": "dlrmamba"}


@app.get("/ready")
def ready() -> dict[str, str]:
    cfg_path = os.getenv("DLRMAMBA_CONFIG", "configs/default.toml")
    try:
        load_config(cfg_path)
        return {"status": "ready"}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"status": "not_ready", "error": str(exc)}


@app.post("/predict")
async def predict(rgb: UploadFile = File(...), ir: UploadFile = File(...)) -> dict[str, object]:
    cfg_path = os.getenv("DLRMAMBA_CONFIG", "configs/default.toml")
    checkpoint = os.getenv("DLRMAMBA_CHECKPOINT", "")

    # Persist temporary in-memory images for shared infer path.
    rgb_bytes = await rgb.read()
    ir_bytes = await ir.read()

    rgb_img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    ir_img = Image.open(io.BytesIO(ir_bytes)).convert("RGB")

    rgb_tmp = "/tmp/dlrmamba_rgb.jpg"
    ir_tmp = "/tmp/dlrmamba_ir.jpg"
    rgb_img.save(rgb_tmp)
    ir_img.save(ir_tmp)

    preds = run_inference(cfg_path, rgb_tmp, ir_tmp, checkpoint)
    return {"detections": preds}
