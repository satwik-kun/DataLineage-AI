"""Build and query the data lineage graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx


def load_pipeline_metadata(metadata_path: str | Path) -> Dict:
    """Load pipeline metadata JSON."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lineage_graph(metadata: Dict) -> nx.DiGraph:
    """Build a directed lineage graph from metadata nodes and edges."""
    graph = nx.DiGraph()

    for node in metadata.get("nodes", []):
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        graph.add_node(node_id, **attrs)

    for edge in metadata.get("edges", []):
        graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "depends_on"))

    return graph


def get_downstream_nodes(graph: nx.DiGraph, node_id: str, include_source: bool = False) -> List[str]:
    """Return all downstream nodes affected by node_id."""
    if node_id not in graph:
        return []

    downstream = sorted(nx.descendants(graph, node_id))
    if include_source:
        return [node_id] + downstream
    return downstream


def get_upstream_nodes(graph: nx.DiGraph, node_id: str, include_source: bool = False) -> List[str]:
    """Return all upstream dependencies of node_id."""
    if node_id not in graph:
        return []

    upstream = sorted(nx.ancestors(graph, node_id))
    if include_source:
        return [node_id] + upstream
    return upstream


def impact_analysis(graph: nx.DiGraph, corrupted_dataset: str) -> List[str]:
    """Alias for downstream traversal used by impact analysis."""
    return get_downstream_nodes(graph, corrupted_dataset)


def get_node_type(graph: nx.DiGraph, node_id: str) -> str:
    """Get node type attribute, defaulting to unknown."""
    return graph.nodes.get(node_id, {}).get("type", "unknown")


def filter_nodes_by_type(graph: nx.DiGraph, node_ids: List[str], node_type: str) -> List[str]:
    """Filter node ids by type attribute."""
    return [node_id for node_id in node_ids if get_node_type(graph, node_id) == node_type]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    metadata = load_pipeline_metadata(project_root / "data" / "pipelines.json")
    lineage_graph = build_lineage_graph(metadata)
    print(f"Loaded graph with {lineage_graph.number_of_nodes()} nodes and {lineage_graph.number_of_edges()} edges")
