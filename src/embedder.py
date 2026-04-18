"""Create text and image embeddings for datasets and charts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import numpy as np


class MultiModalEmbedder:
    """Text and image embedding helper with graceful fallback."""

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        enable_real_models: bool = False,
    ) -> None:
        self.text_model_name = text_model_name
        self.clip_model_name = clip_model_name
        self.enable_real_models = enable_real_models

        self._text_model = None
        self._clip_model = None
        self._clip_processor = None
        self._torch = None

        self.text_dim = 384
        self.image_dim = 512

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def _mock_vector(self, value: bytes, dim: int) -> np.ndarray:
        # Deterministic fallback embedding from hash bytes.
        digest = hashlib.sha256(value).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    def _load_text_model(self) -> None:
        if not self.enable_real_models:
            return
        if self._text_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self.text_model_name)
            self.text_dim = self._text_model.get_sentence_embedding_dimension()
        except Exception:
            self._text_model = None

    def _load_clip_model(self) -> None:
        if not self.enable_real_models:
            return
        if self._clip_model is not None and self._clip_processor is not None:
            return
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            self._torch = torch
            self._Image = Image
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.image_dim = int(self._clip_model.config.projection_dim)
        except Exception:
            self._torch = None
            self._clip_model = None
            self._clip_processor = None

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return normalized text embeddings as float32 matrix."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.text_dim), dtype=np.float32)

        self._load_text_model()
        if self._text_model is not None:
            vectors = self._text_model.encode(text_list, convert_to_numpy=True).astype(np.float32)
            return self._normalize(vectors)

        fallback = np.vstack([self._mock_vector(t.encode("utf-8"), self.text_dim) for t in text_list])
        return fallback.astype(np.float32)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Return normalized image embedding vector."""
        image_path = Path(image_path)
        self._load_clip_model()

        if self._clip_model is not None and self._clip_processor is not None and self._torch is not None:
            image = self._Image.open(image_path).convert("RGB")
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with self._torch.no_grad():
                image_features = self._clip_model.get_image_features(**inputs)
            vector = image_features.cpu().numpy().astype(np.float32)
            vector = self._normalize(vector)
            return vector[0]

        try:
            image_bytes = image_path.read_bytes()
        except Exception:
            image_bytes = str(image_path).encode("utf-8")
        return self._mock_vector(image_bytes, self.image_dim).astype(np.float32)


if __name__ == "__main__":
    embedder = MultiModalEmbedder()
    text_vec = embedder.embed_texts(["What depends on sales.csv?"])
    print(f"Text embedding shape: {text_vec.shape}")
