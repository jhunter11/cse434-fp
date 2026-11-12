"""Reference-free CLIPScore (Hessel et al., 2021). Needs the optional models extra."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_PREFIX = "A photo depicting "
CLIP_WEIGHT = 2.5


def clipscore(image_paths, captions_by_model, device="auto", batch_size=64, revision=None):
    """Return ``{model: {image_id: score}}`` and a provenance record.

    The score is ``2.5 * max(cos(image, text), 0)`` with CLIP ViT-B/32 and the
    text prefix "A photo depicting ". These settings match the original paper.
    The function embeds each image once and reuses it for every model.
    CLIP truncates text at 77 tokens. The score uses no reference captions.
    """
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoConfig, CLIPModel, CLIPProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AutoConfig.from_pretrained(CLIP_MODEL_ID, revision=revision, trust_remote_code=False)
    from .models import resolve_revision
    resolved = resolve_revision(CLIP_MODEL_ID, revision, config) or revision
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, revision=resolved, trust_remote_code=False)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, revision=resolved, trust_remote_code=False).to(device).eval()

    def unit(output):
        tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
        return torch.nn.functional.normalize(tensor.float(), dim=-1)

    image_ids = list(image_paths)
    image_vectors = {}
    with torch.inference_mode():
        for start in range(0, len(image_ids), batch_size):
            batch = image_ids[start:start + batch_size]
            images = []
            for image_id in batch:
                with Image.open(Path(image_paths[image_id])) as source:
                    images.append(ImageOps.exif_transpose(source).convert("RGB"))
            pixels = processor(images=images, return_tensors="pt").to(device)
            vectors = unit(model.get_image_features(**pixels))
            image_vectors.update(zip(batch, vectors, strict=True))
        scores = {}
        for model_name, captions in captions_by_model.items():
            scores[model_name] = {}
            for start in range(0, len(image_ids), batch_size):
                batch = image_ids[start:start + batch_size]
                tokens = processor(text=[CLIP_PREFIX + captions[i] for i in batch], return_tensors="pt",
                                   padding=True, truncation=True, max_length=77).to(device)
                vectors = unit(model.get_text_features(**tokens))
                for image_id, text_vector in zip(batch, vectors, strict=True):
                    cosine = float((image_vectors[image_id] * text_vector).sum())
                    scores[model_name][image_id] = CLIP_WEIGHT * max(cosine, 0.0)
    provenance = {
        "model_id": CLIP_MODEL_ID, "revision": resolved, "prefix": CLIP_PREFIX, "weight": CLIP_WEIGHT,
        "formula": "2.5 * max(cosine(image, text), 0); reference-free", "device": device,
        "torch_version": version("torch"), "transformers_version": version("transformers"),
    }
    return scores, provenance
