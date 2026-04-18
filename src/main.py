"""CLI entry point for lineage and impact analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd

from embedder import MultiModalEmbedder
from graph_builder import build_lineage_graph, get_downstream_nodes, load_pipeline_metadata
from rag import SimpleRAGExplainer
from retriever import LineageRetriever


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[1]


def build_runtime() -> tuple:
    root = project_root_from_file()
    metadata = load_pipeline_metadata(root / "data" / "pipelines.json")
    graph = build_lineage_graph(metadata)
    embedder = MultiModalEmbedder()
    retriever = LineageRetriever(graph=graph, embedder=embedder, project_root=root)
    retriever.build_indices()
    rag = SimpleRAGExplainer()
    return root, graph, retriever, rag


def generate_charts(root: Path) -> None:
    data_dir = root / "data" / "datasets"
    chart_dir = root / "data" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    sales_df = pd.read_csv(data_dir / "sales.csv")
    customers_df = pd.read_csv(data_dir / "customers.csv")
    churn_df = pd.read_csv(data_dir / "churn.csv")

    import matplotlib.pyplot as plt

    sales_by_region = sales_df.groupby("region", as_index=False)["total_amount"].sum()
    plt.figure(figsize=(7, 4))
    plt.bar(sales_by_region["region"], sales_by_region["total_amount"])
    plt.title("Total Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Amount")
    plt.tight_layout()
    plt.savefig(chart_dir / "sales_summary_chart.png", dpi=120)
    plt.close()

    churn_join = customers_df.merge(churn_df, on="customer_id", how="left")
    churn_join["churned"] = churn_join["churned"].fillna(0)
    churn_rate = churn_join.groupby("segment", as_index=False)["churned"].mean()
    plt.figure(figsize=(7, 4))
    plt.bar(churn_rate["segment"], churn_rate["churned"])
    plt.title("Churn Rate by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(chart_dir / "churn_by_segment_chart.png", dpi=120)
    plt.close()

    print(f"Charts generated in {chart_dir}")


def summarize_text_hits(hits: List[dict]) -> List[str]:
    lines = []
    for hit in hits:
        lines.append(f"{hit.get('node_id')} (type={hit.get('type')}, score={hit.get('score'):.3f})")
    return lines


def parse_dataset_in_question(question: str) -> str | None:
    match = re.search(r"([A-Za-z0-9_\-]+\.csv)", question)
    if match:
        return match.group(1)
    return None


def cmd_query_text(question: str) -> None:
    _, graph, retriever, rag = build_runtime()
    text_hits = retriever.query_text(question, top_k=5)

    dataset_candidate = parse_dataset_in_question(question)
    dependency_nodes = []
    if dataset_candidate and dataset_candidate in graph:
        dependency_nodes = get_downstream_nodes(graph, dataset_candidate)

    context_lines = summarize_text_hits(text_hits)
    if dependency_nodes:
        context_lines.append(f"Downstream dependents of {dataset_candidate}: {', '.join(dependency_nodes)}")

    print("Top text matches:")
    for line in context_lines:
        print(f"- {line}")

    print("\nExplanation:")
    print(rag.explain(question, context_lines))


def cmd_query_image(image_path: str) -> None:
    _, _, retriever, rag = build_runtime()
    hits = retriever.query_image(image_path, top_k=1)

    if not hits:
        print("No image index entries found. Generate charts first.")
        return

    best = hits[0]
    related = best.get("related_datasets", [])
    print("Best matched chart node:")
    print(f"- {best.get('node_id')} (score={best.get('score'):.3f})")
    print("Related datasets:")
    for ds in related:
        print(f"- {ds}")

    question = f"Given image {image_path}, which dataset is related?"
    context = [f"Matched chart: {best.get('node_id')}", f"Related datasets: {', '.join(related) if related else 'none'}"]
    print("\nExplanation:")
    print(rag.explain(question, context))


def cmd_impact(dataset_id: str) -> None:
    _, graph, _, rag = build_runtime()
    if dataset_id not in graph:
        print(f"Dataset node not found in graph: {dataset_id}")
        return

    impacted = get_downstream_nodes(graph, dataset_id)
    print(f"Impacted downstream nodes for {dataset_id}:")
    if not impacted:
        print("- none")
    else:
        for node in impacted:
            node_type = graph.nodes[node].get("type", "unknown")
            print(f"- {node} (type={node_type})")

    question = f"If {dataset_id} is corrupted, what is affected?"
    context = [f"Impacted nodes: {', '.join(impacted) if impacted else 'none'}"]
    print("\nExplanation:")
    print(rag.explain(question, context))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-powered Data Lineage & Impact Analysis Engine (MVP)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate-charts", help="Generate sample charts in data/charts")

    text_p = sub.add_parser("query-text", help="Run text query (ex: What depends on sales.csv?)")
    text_p.add_argument("--question", required=True, help="Natural language question")

    image_p = sub.add_parser("query-image", help="Match an input chart image to known lineage chart nodes")
    image_p.add_argument("--image-path", required=True, help="Path to query image")

    impact_p = sub.add_parser("impact", help="Run impact analysis for a dataset node")
    impact_p.add_argument("--dataset", required=True, help="Dataset id, e.g., sales.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = project_root_from_file()

    if args.command == "generate-charts":
        generate_charts(root)
    elif args.command == "query-text":
        cmd_query_text(args.question)
    elif args.command == "query-image":
        cmd_query_image(args.image_path)
    elif args.command == "impact":
        cmd_impact(args.dataset)


if __name__ == "__main__":
    main()
