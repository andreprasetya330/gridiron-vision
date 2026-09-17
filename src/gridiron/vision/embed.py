"""Learned appearance embeddings for player crops.

Hand-built colour features were the weakest link in identity. A median Lab
colour is one number per channel, so it cannot tell a white jersey in shadow
from a green one in sun - measured on Sugar Bowl film, clustering on all three
Lab channels split the field by *lighting* rather than by team. Dropping
luminance fixed that, but only by throwing away real information: two teams in
white and silver would still be indistinguishable.

SigLIP sees the crop instead of averaging it. The embedding responds to stripes,
collar shape, helmet colour, and number placement, none of which survive a
median. This follows Roboflow's sports pipeline, which solves the same
unsupervised problem: separate two teams on film you have never labelled,
without knowing what either uniform looks like.

Their pipeline reduces the embeddings with UMAP before clustering, which this
one does not. That step pays off over thousands of crops drawn from many games;
on the few dozen tracks in a single play it measurably hurt, distorting the
neighbourhood enough that clustering absorbed real players into the officials
(scripts/embedding_study.py). KMeans runs on the raw vectors instead.

The model is optional. It lives behind `available()` so the pipeline keeps
running on the colour features when the vision extra is not installed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SIGLIP_MODEL = "google/siglip-base-patch16-224"


def available() -> bool:
    """True when the embedding stack is installed and importable."""
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _load(model_name: str, device: str):
    # Only the vision tower. AutoProcessor would also drag in SigLIP's text
    # tokenizer, which needs sentencepiece and is pure dead weight here.
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor, torch


class SiglipEmbedder:
    """Turns BGR crops into L2-normalised appearance vectors."""

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """(N, 768) embeddings for N BGR crops, in input order."""
        if not crops:
            return np.zeros((0, 768), dtype=np.float32)

        from PIL import Image

        model, processor, torch = _load(self.model_name, self.device)
        images = [
            Image.fromarray(crop[:, :, ::-1]) if crop.ndim == 3 else Image.fromarray(crop)
            for crop in crops
        ]

        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(self.device)
                hidden = model(**inputs).last_hidden_state
                out.append(torch.mean(hidden, dim=1).float().cpu().numpy())

        features = np.concatenate(out).astype(np.float32)
        # Normalised so cosine similarity is a plain dot product, which is what
        # the downstream clustering and any track merging both want.
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.clip(norms, 1e-6, None)


def player_crop(frame: np.ndarray, detection) -> np.ndarray:
    """The whole player, padded a little, rather than a guessed torso box.

    The torso heuristic was tuned to dodge grass, which matters when you are
    averaging pixels. An embedding would rather see the entire player: stance,
    helmet and leg colour all carry team information, and officials are far
    easier to spot from a whole body than from a 18x25 patch of shirt.
    """
    h, w = frame.shape[:2]
    pad_x = 0.05 * (detection.x2 - detection.x1)
    pad_y = 0.05 * (detection.y2 - detection.y1)
    x1 = int(np.clip(detection.x1 - pad_x, 0, w - 1))
    x2 = int(np.clip(detection.x2 + pad_x, 1, w))
    y1 = int(np.clip(detection.y1 - pad_y, 0, h - 1))
    y2 = int(np.clip(detection.y2 + pad_y, 1, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((2, 2, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]
