"""Build and query the data lineage graph."""

from __future__ import annotations

import json
from collections import deque
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


def simulate_impact(graph: nx.DiGraph, node_id: str) -> List[str]:
    """Traverse all downstream nodes with BFS order for multi-hop impact simulation."""
    if node_id not in graph:
        return []

    visited = {node_id}
    queue: deque[str] = deque([node_id])
    ordered: List[str] = []

    while queue:
        current = queue.popleft()
        neighbors = sorted(graph.successors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            ordered.append(neighbor)
            queue.append(neighbor)

    return ordered


def simulate_impact_with_depth(graph: nx.DiGraph, node_id: str) -> Dict[str, int]:
    """Return BFS depth per downstream node from source; safe for cyclic graphs via visited set."""
    if node_id not in graph:
        return {}

    visited = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    depths: Dict[str, int] = {}

    while queue:
        current, depth = queue.popleft()
        neighbors = sorted(graph.successors(current))
        for neighbor in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            next_depth = depth + 1
            depths[neighbor] = next_depth
            queue.append((neighbor, next_depth))

    return depths


def is_graph_dag(graph: nx.DiGraph) -> bool:
    """Check if lineage graph is a DAG."""
    return nx.is_directed_acyclic_graph(graph)


def build_impact_chains(graph: nx.DiGraph, source_node: str, impacted_nodes: List[str]) -> List[str]:
    """Create readable source->...->target chains using shortest directed paths."""
    chains: List[str] = []
    for target in impacted_nodes:
        try:
            path = nx.shortest_path(graph, source=source_node, target=target)
        except Exception:
            continue

        segments = []
        for i in range(len(path) - 1):
            left = path[i]
            right = path[i + 1]
            relation = graph.edges[left, right].get("relation", "depends_on")
            segments.append(f"{left} -[{relation}]-> {right}")
        if segments:
            chains.append(" ; ".join(segments))
    return chains


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
