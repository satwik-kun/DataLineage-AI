"""Retrieval layer for text/image to lineage assets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

from embedder import MultiModalEmbedder
from graph_builder import filter_nodes_by_type, get_downstream_nodes, get_upstream_nodes, simulate_impact
from vector_store import FaissVectorStore


DEFAULT_TOP_K = 5


class LineageRetriever:
    """Build text and image retrieval over lineage assets."""

    def __init__(self, graph: nx.DiGraph, embedder: MultiModalEmbedder, project_root: str | Path) -> None:
        self.graph = graph
        self.embedder = embedder
        self.project_root = Path(project_root)

        self.text_store = FaissVectorStore(dim=self.embedder.image_dim)
        self.image_store = FaissVectorStore(dim=self.embedder.image_dim)
        self.multimodal_store = FaissVectorStore(dim=self.embedder.image_dim)

    def _node_text(self, node_id: str, attrs: Dict) -> str:
        return " | ".join(
            [
                f"id: {node_id}",
                f"type: {attrs.get('type', 'unknown')}",
                f"description: {attrs.get('description', '')}",
                f"path: {attrs.get('path', '')}",
            ]
        )

    def _node_semantic_text(self, node_id: str, attrs: Dict) -> str:
        node_type = attrs.get("type", "unknown")
        description = attrs.get("description", "")
        return (
            f"Node {node_id} is a {node_type}. "
            f"Business meaning: {description}. "
            f"This node participates in analytics lineage and report dependencies."
        )

    def build_indices(self) -> None:
        """Create text and image vector indices from graph nodes."""
        text_records = []
        text_payloads = []

        for node_id, attrs in self.graph.nodes(data=True):
            text_payloads.append(self._node_text(node_id, attrs))
            text_records.append({"node_id": node_id, "modality": "text", "type": attrs.get("type", "unknown")})

        text_vectors = self.embedder.embed_clip_texts(text_payloads)
        self.text_store.add(text_vectors, text_records)

        image_vectors = []
        image_records = []
        multimodal_vectors = []
        multimodal_records = []

        semantic_payloads = []
        semantic_records = []

        for node_id, attrs in self.graph.nodes(data=True):
            semantic_payloads.append(self._node_semantic_text(node_id, attrs))
            semantic_records.append({"node_id": node_id, "modality": "clip_text", "type": attrs.get("type", "unknown")})

            if attrs.get("type") != "chart":
                continue
            if "path" not in attrs:
                continue

            chart_path = self.project_root / attrs["path"]
            if not chart_path.exists():
                continue

            image_vec = self.embedder.embed_image(chart_path)
            image_vectors.append(image_vec)
            image_records.append({"node_id": node_id, "path": str(chart_path), "modality": "image", "type": "chart"})
            multimodal_vectors.append(image_vec)
            multimodal_records.append({"node_id": node_id, "path": str(chart_path), "modality": "image", "type": "chart"})

        if semantic_payloads:
            clip_text_vectors = self.embedder.embed_clip_texts(semantic_payloads)
            multimodal_vectors.extend(list(clip_text_vectors))
            multimodal_records.extend(semantic_records)

        if image_vectors:
            self.image_store.add(vectors=np.vstack(image_vectors), metadata=image_records)

        if multimodal_vectors:
            self.multimodal_store.add(vectors=np.vstack(multimodal_vectors), metadata=multimodal_records)

    def query_text(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        query_vector = self.embedder.embed_clip_texts([query])[0]
        hits = self.text_store.search(query_vector, top_k=top_k)
        return sorted(hits, key=lambda h: (-h.get("score", 0.0), str(h.get("node_id", ""))))

    def find_dependents(self, node_id: str) -> List[str]:
        return get_downstream_nodes(self.graph, node_id)

    def impact_analysis(self, node_id: str) -> List[str]:
        return simulate_impact(self.graph, node_id)

    def related_datasets_for_chart(self, chart_node_id: str) -> List[str]:
        upstream = get_upstream_nodes(self.graph, chart_node_id)
        return filter_nodes_by_type(self.graph, upstream, "dataset")

    def query_image(self, image_path: str | Path, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        query_vector = self.embedder.embed_image(image_path)
        chart_hits = self.image_store.search(query_vector, top_k=top_k)

        enriched = []
        for hit in sorted(chart_hits, key=lambda h: (-h.get("score", 0.0), str(h.get("node_id", "")))):
            chart_id = hit["node_id"]
            enriched.append(
                {
                    **hit,
                    "related_datasets": self.related_datasets_for_chart(chart_id),
                }
            )
        return enriched

    def retrieve_from_image(self, image_path: str | Path, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Cross-modal retrieval from image to all lineage nodes in shared CLIP space."""
        query_vector = self.embedder.embed_image(image_path)
        hits = self.multimodal_store.search(query_vector, top_k=top_k)

        deduped: List[Dict] = []
        seen = set()
        for hit in sorted(hits, key=lambda h: (-h.get("score", 0.0), str(h.get("node_id", "")))):
            node_id = hit.get("node_id")
            if node_id in seen:
                continue
            seen.add(node_id)
            node_type = self.graph.nodes.get(node_id, {}).get("type", "unknown")
            deduped.append({**hit, "type": node_type})
        return deduped


if __name__ == "__main__":
    print("retriever module")
