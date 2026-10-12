"""Loopback-only development interface for captioning and saved comparisons."""

from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from captionbench.data import load_manifest, read_jsonl
from captionbench.models import caption_image

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
STATIC = Path(__file__).parent / "static"
INFERENCE_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


def create_app(manifest_path: str | Path | None = None, predictions_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="CaptionBench", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {
            "version": "0.1.0",
            "models": ["blip-base", "blip-large", "generic"],
            "gallery_configured": manifest_path is not None,
            "mode": "local",
        }

    def gallery_data():
        if manifest_path is None:
            return [], []
        records = load_manifest(Path(manifest_path))
        predictions = []
        if predictions_path:
            predictions = read_jsonl(predictions_path)
            test_ids = {row["image_id"] for row in records if row["split"] == "test"}
            seen = set()
            for row in predictions:
                if not isinstance(row, dict):
                    raise ValueError("Prediction rows must be objects.")
                image_id, model, caption = (row.get(key) for key in ("image_id", "model", "caption"))
                if not all(isinstance(value, str) and value.strip() for value in (image_id, model, caption)):
                    raise ValueError("Prediction rows need an image ID, model, and caption.")
                if image_id not in test_ids or (image_id, model) in seen:
                    raise ValueError("Prediction IDs must be unique and belong to the test split.")
                seen.add((image_id, model))
        return records, predictions

    @app.get("/api/gallery")
    def gallery():
        try:
            records, predictions = gallery_data()
        except (ValueError, OSError) as exc:
            raise HTTPException(400, "The gallery files could not be read. Check the manifest and predictions.") from exc
        return [
            {
                "image_id": row["image_id"],
                "image_url": f"/api/images/{index}",
                "split": row["split"],
                "evidence_label": "Synthetic software fixture" if "synthetic" in row.get("source", "").lower()
                                  else "Configured test sample",
                "references": row["references"],
                "predictions": [
                    {key: pred.get(key) for key in ("model", "caption", "latency_seconds", "revision")}
                    for pred in predictions if pred["image_id"] == row["image_id"]
                ],
            }
            for index, row in enumerate(records) if row["split"] == "test"
        ]

    @app.get("/api/images/{index}")
    def gallery_image(index: int):
        try:
            records, _ = gallery_data()
        except (ValueError, OSError) as exc:
            raise HTTPException(400, "The gallery files could not be read. Check the manifest and predictions.") from exc
        if index < 0 or index >= len(records) or records[index]["split"] != "test":
            raise HTTPException(404, "Image not found.")
        return FileResponse(records[index]["image_path"])

    @app.post("/api/caption")
    async def caption(request: Request, model: str = "blip-base"):
        if model not in {"blip-base", "blip-large", "generic"}:
            raise HTTPException(400, "Choose BLIP base, BLIP large, or the generic control.")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BYTES:
                raise HTTPException(413, "Choose an image smaller than 10 MB.")
            chunks.append(chunk)
        try:
            with Image.open(io.BytesIO(b"".join(chunks))) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "Choose an image with fewer than 20 million pixels.")
                image.load()
                clean_image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "The file is not a readable image. Choose a PNG, JPEG, or WebP image.") from exc

        def infer():
            if not INFERENCE_LOCK.acquire(blocking=False):
                raise HTTPException(429, "A caption is in progress. Try again after it finishes.")
            try:
                with tempfile.TemporaryDirectory(prefix="captionbench-") as temporary:
                    path = Path(temporary) / "upload.png"
                    clean_image.save(path)
                    return caption_image(path, model_key=model)
            except (ImportError, RuntimeError, OSError, ValueError) as exc:
                LOGGER.exception("The selected caption model failed")
                raise HTTPException(503, "The model could not run. Check the server log and model setup instructions.") from exc
            finally:
                INFERENCE_LOCK.release()

        result = await run_in_threadpool(infer)
        return {**result, "model": model, "visual_input": model != "generic"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Run the CaptionBench local demo.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.manifest, args.predictions), host="127.0.0.1", port=args.port)


app = create_app(os.environ.get("CAPTIONBENCH_MANIFEST"), os.environ.get("CAPTIONBENCH_PREDICTIONS"))

if __name__ == "__main__":
    main()
