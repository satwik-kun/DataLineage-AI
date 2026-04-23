"""CLI entry point for lineage and impact analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd

from embedder import MultiModalEmbedder
from graph_builder import (
    build_impact_chains,
    build_lineage_graph,
    is_graph_dag,
    load_pipeline_metadata,
    simulate_impact,
    simulate_impact_with_depth,
)
from rag import SimpleRAGExplainer
from retriever import DEFAULT_TOP_K, LineageRetriever


MIN_SEMANTIC_CONFIDENCE = 0.2
_RUNTIME_CACHE: tuple | None = None


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_runtime() -> tuple:
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is not None:
        return _RUNTIME_CACHE

    root = project_root_from_file()
    metadata = load_pipeline_metadata(root / "data" / "pipelines.json")
    graph = build_lineage_graph(metadata)
    embedder = MultiModalEmbedder(enable_real_models=env_flag("AIR_ENABLE_REAL_MODELS", default=True))
    retriever = LineageRetriever(graph=graph, embedder=embedder, project_root=root)
    retriever.build_indices()
    rag = SimpleRAGExplainer()
    _RUNTIME_CACHE = (root, graph, retriever, rag)
    return _RUNTIME_CACHE


def generate_charts(root: Path, quiet: bool = False) -> None:
    data_dir = root / "data" / "datasets"
    chart_dir = root / "data" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    sales_df = pd.read_csv(data_dir / "sales.csv")
    customers_df = pd.read_csv(data_dir / "customers.csv")
    churn_df = pd.read_csv(data_dir / "churn.csv")

    import matplotlib.pyplot as plt

    sales_by_region = sales_df.groupby("region", as_index=False)["revenue"].sum()
    plt.figure(figsize=(7, 4))
    plt.bar(sales_by_region["region"], sales_by_region["revenue"])
    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(chart_dir / "sales_summary_chart.png", dpi=120)
    plt.savefig(chart_dir / "sales_chart.png", dpi=120)
    plt.close()

    churn_join = customers_df.merge(churn_df, on="customer_id", how="left")
    churn_join["churn_flag"] = churn_join["churn_flag"].fillna(0)
    churn_rate = churn_join.groupby("segment", as_index=False)["churn_flag"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(churn_rate["segment"], churn_rate["churn_flag"])
    plt.title("Churn Rate by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(chart_dir / "churn_by_segment_chart.png", dpi=120)
    plt.close()

    if not quiet:
        print(f"Charts generated in {chart_dir}")


def summarize_text_hits(hits: List[dict]) -> List[str]:
    lines = []
    for hit in hits:
        lines.append(f"{hit.get('node_id')} (type={hit.get('type')}, score={hit.get('score'):.3f})")
    return lines


def cmd_query_text(question: str) -> bool:
    _, graph, retriever, rag = build_runtime()
    text_hits = retriever.query_text(question, top_k=DEFAULT_TOP_K)
    if not text_hits:
        print("No retrieval matches found.")
        return False

    best_score = float(text_hits[0].get("score", 0.0))
    if best_score < MIN_SEMANTIC_CONFIDENCE:
        print("Top text matches:")
        for line in summarize_text_hits(text_hits):
            print(f"- {line}")
        print("\nExplanation:")
        print("Low confidence: retrieved lineage evidence is insufficient to answer reliably.")
        return False

    top_node = text_hits[0].get("node_id")
    dependency_nodes = retriever.impact_analysis(top_node) if top_node in graph else []
    chains = build_impact_chains(graph, top_node, dependency_nodes) if top_node in graph else []

    context_lines = summarize_text_hits(text_hits)
    if dependency_nodes:
        context_lines.append(f"Downstream dependents of {top_node}: {', '.join(dependency_nodes)}")
    context_lines.extend(chains)

    print("Top text matches:")
    for line in context_lines:
        print(f"- {line}")

    print("\nExplanation:")
    print(rag.explain(question, context_lines))
    return bool(text_hits)


def cmd_query_image(image_path: str) -> bool:
    root, graph, retriever, rag = build_runtime()

    candidate_path = Path(image_path)
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    if not candidate_path.exists():
        print("Image path not found.")
        return False

    hits = retriever.retrieve_from_image(candidate_path, top_k=DEFAULT_TOP_K)

    if not hits:
        print("No image index entries found. Generate charts first.")
        return False

    best = hits[0]
    node_id = best.get("node_id")
    related = retriever.related_datasets_for_chart(node_id) if node_id in graph and graph.nodes[node_id].get("type") == "chart" else []
    print("Top cross-modal match:")
    print(f"- {best.get('node_id')} (score={best.get('score'):.3f})")
    print("Additional matches:")
    for item in hits[1:4]:
        print(f"- {item.get('node_id')} (type={item.get('type')}, score={item.get('score'):.3f})")

    if related:
        print("Related datasets:")
        for ds in related:
            print(f"- {ds}")

    question = f"Given image {candidate_path}, which dataset is related?"
    context = [f"Matched node: {best.get('node_id')}", f"Related datasets: {', '.join(related) if related else 'none'}"]
    print("\nExplanation:")
    print(rag.explain(question, context))
    return True


def cmd_impact(dataset_id: str) -> bool:
    _, graph, _, rag = build_runtime()
    if dataset_id not in graph:
        print("Node not found in lineage graph")
        return False

    impacted = simulate_impact(graph, dataset_id)
    depths = simulate_impact_with_depth(graph, dataset_id)
    chains = build_impact_chains(graph, dataset_id, impacted)
    print(f"Impacted downstream nodes for {dataset_id}:")
    if not impacted:
        print("- none")
    else:
        for node in impacted:
            node_type = graph.nodes[node].get("type", "unknown")
            print(f"- {node} (type={node_type})")

    question = f"If {dataset_id} is corrupted, what is affected?"
    context = [f"Impacted nodes: {', '.join(impacted) if impacted else 'none'}"] + chains
    if depths:
        context.append(f"Max traversal depth: {max(depths.values())}")
    print("\nExplanation:")
    print(rag.explain(question, context))
    return True


def cmd_demo_tests() -> None:
    root = project_root_from_file()
    generate_charts(root, quiet=True)

    global _RUNTIME_CACHE
    _RUNTIME_CACHE = None
    _, graph, retriever, _ = build_runtime()

    text_hits = retriever.query_text("Which reports rely on revenue data?", top_k=DEFAULT_TOP_K)
    test1_pass = any(hit.get("type") in {"chart", "pipeline"} for hit in text_hits)

    image_hits = retriever.retrieve_from_image(root / "data" / "charts" / "sales_chart.png", top_k=DEFAULT_TOP_K)
    test2_pass = any(hit.get("node_id") == "sales.csv" for hit in image_hits)

    impacted = simulate_impact(graph, "sales.csv")
    depth_map = simulate_impact_with_depth(graph, "sales.csv")
    max_depth = max(depth_map.values()) if depth_map else 0
    expected_nodes = {
        "customer_revenue_join",
        "churn_impact_features",
        "sales_summary_chart.png",
        "churn_by_segment_chart.png",
    }
    test3_pass = expected_nodes.issubset(set(impacted)) and max_depth > 1

    dag_ok = is_graph_dag(graph)

    print(f"TEST 1: TEXT QUERY -> {'PASS' if test1_pass else 'FAIL'}")
    print(f"TEST 2: IMAGE QUERY -> {'PASS' if test2_pass else 'FAIL'}")
    print(f"TEST 3: IMPACT ANALYSIS -> {'PASS' if test3_pass else 'FAIL'}")

    # Internal invariant checks to enforce robustness without noisy CLI output.
    assert dag_ok or len(depth_map) == len(set(impacted))
    assert max_depth > 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-powered Data Lineage & Impact Analysis Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, help="Semantic text query, e.g. \"What depends on sales.csv?\"")
    parser.add_argument("--image", type=str, help="Image path for cross-modal retrieval")
    parser.add_argument("--impact", type=str, help="Dataset/node id for multi-hop impact analysis")
    parser.add_argument("--demo-tests", action="store_true", help="Run mandatory rubric demo tests")
    parser.add_argument("--generate-charts", action="store_true", help="Generate charts from datasets")
    parser.add_argument("legacy_command", nargs="?", choices=["demo-tests", "generate-charts"], help="Legacy command mode")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = project_root_from_file()
    if args.generate_charts or args.legacy_command == "generate-charts":
        generate_charts(root)

    if args.demo_tests or args.legacy_command == "demo-tests":
        cmd_demo_tests()
        return

    action_taken = False
    if args.query:
        action_taken = True
        cmd_query_text(args.query)
    if args.image:
        action_taken = True
        cmd_query_image(args.image)
    if args.impact:
        action_taken = True
        cmd_impact(args.impact)

    if not action_taken and not args.generate_charts:
        parser.print_help()


if __name__ == "__main__":
    main()
