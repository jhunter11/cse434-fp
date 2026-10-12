import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from captionbench.app import create_app


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, "PNG")
    return stream.getvalue()


def test_demo_generic_caption_and_deleted_upload(monkeypatch):
    from captionbench import app as module
    seen = []

    def fake_caption(path, **kwargs):
        assert path.exists()
        seen.append(path)
        return {"caption": "a photograph of a scene", "model_id": "generic", "latency_seconds": 0}

    monkeypatch.setattr(module, "caption_image", fake_caption)
    client = TestClient(create_app())
    result = client.post("/api/caption?model=generic", content=png_bytes())
    assert result.status_code == 200
    assert result.json()["visual_input"] is False
    assert not seen[0].exists()


def test_bad_image_and_model():
    client = TestClient(create_app())
    assert client.post("/api/caption?model=generic", content=b"not an image").status_code == 400
    assert client.post("/api/caption?model=unknown", content=png_bytes()).status_code == 400
    assert client.post("/api/caption?model=generic", content=b"x" * (10 * 1024 * 1024 + 1)).status_code == 413


def test_empty_gallery_and_home():
    client = TestClient(create_app())
    assert client.get("/api/gallery").json() == []
    assert client.get("/").status_code == 200
    assert client.get("/api/images/-1").status_code == 404


def test_phone_image_orientation_reaches_model_upright(monkeypatch):
    from captionbench import app as module
    observed_sizes = []

    def fake_caption(path, **kwargs):
        with Image.open(path) as image:
            observed_sizes.append(image.size)
        return {"caption": "test", "latency_seconds": 0}

    monkeypatch.setattr(module, "caption_image", fake_caption)
    source = Image.new("RGB", (20, 30), "red")
    exif = source.getexif()
    exif[274] = 6
    stream = io.BytesIO()
    source.save(stream, "JPEG", exif=exif)
    response = TestClient(create_app()).post("/api/caption?model=generic", content=stream.getvalue())
    assert response.status_code == 200
    assert observed_sizes == [(30, 20)]


def test_gallery_rejects_unknown_and_duplicate_predictions(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(png_bytes())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image_id": "known", "image_path": str(image), "split": "test",
                                    "references": ["a red image"] * 5}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    for rows in ([{"model": "generic", "caption": "a scene"}],
                 [{"image_id": "unknown", "model": "generic", "caption": "a scene"}],
                 [{"image_id": "known", "model": "generic", "caption": "a scene"}] * 2):
        predictions.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        assert TestClient(create_app(manifest, predictions)).get("/api/gallery").status_code == 400
