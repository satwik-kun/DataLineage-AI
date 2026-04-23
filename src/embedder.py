"""Create text and image embeddings for datasets and charts."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
from pathlib import Path
from typing import Iterable, List
import warnings

import numpy as np


class MultiModalEmbedder:
    """Text and image embedding helper with graceful fallback."""

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        enable_real_models: bool = True,
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

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @contextlib.contextmanager
    def _suppress_noisy_output(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

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
            with self._suppress_noisy_output():
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
            with self._suppress_noisy_output():
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

    def _extract_clip_tensor(self, clip_output) -> np.ndarray | None:
        if hasattr(clip_output, "cpu"):
            tensor = clip_output
        elif hasattr(clip_output, "text_embeds"):
            tensor = clip_output.text_embeds
        elif hasattr(clip_output, "image_embeds"):
            tensor = clip_output.image_embeds
        elif hasattr(clip_output, "pooler_output"):
            tensor = clip_output.pooler_output
        elif hasattr(clip_output, "last_hidden_state"):
            tensor = clip_output.last_hidden_state[:, 0, :]
        else:
            return None

        vector = tensor.cpu().numpy().astype(np.float32)
        if vector.shape[1] > self.image_dim:
            vector = vector[:, : self.image_dim]
        elif vector.shape[1] < self.image_dim:
            pad = self.image_dim - vector.shape[1]
            vector = np.pad(vector, ((0, 0), (0, pad)), mode="constant")
        return self._normalize(vector)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return normalized text embeddings as float32 matrix."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.text_dim), dtype=np.float32)

        self._load_text_model()
        if self._text_model is not None:
            with self._suppress_noisy_output():
                vectors = self._text_model.encode(text_list, convert_to_numpy=True).astype(np.float32)
            return self._normalize(vectors)

        fallback = np.vstack([self._mock_vector(t.encode("utf-8"), self.text_dim) for t in text_list])
        return fallback.astype(np.float32)

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Return normalized image embedding vector."""
        image_path = Path(image_path)
        self._load_clip_model()

        if self._clip_model is not None and self._clip_processor is not None and self._torch is not None:
            try:
                image = self._Image.open(image_path).convert("RGB")
                inputs = self._clip_processor(images=image, return_tensors="pt")
                with self._torch.no_grad():
                    image_features = self._clip_model.get_image_features(**inputs)

                vector = self._extract_clip_tensor(image_features)
                if vector is not None:
                    return vector[0]
            except Exception:
                pass

        try:
            image_bytes = image_path.read_bytes()
        except Exception:
            image_bytes = str(image_path).encode("utf-8")
        return self._mock_vector(image_bytes, self.image_dim).astype(np.float32)

    def embed_clip_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Return CLIP text-space embeddings for cross-modal retrieval."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.image_dim), dtype=np.float32)

        self._load_clip_model()
        if self._clip_model is not None and self._clip_processor is not None and self._torch is not None:
            try:
                inputs = self._clip_processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
                with self._torch.no_grad():
                    text_features = self._clip_model.get_text_features(**inputs)
                vector = self._extract_clip_tensor(text_features)
                if vector is not None:
                    return vector.astype(np.float32)
            except Exception:
                pass

        fallback = np.vstack([self._mock_vector(t.encode("utf-8"), self.image_dim) for t in text_list])
        return fallback.astype(np.float32)


if __name__ == "__main__":
    embedder = MultiModalEmbedder()
    text_vec = embedder.embed_texts(["What depends on sales.csv?"])
    print(f"Text embedding shape: {text_vec.shape}")
