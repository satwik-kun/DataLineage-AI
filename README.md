# AI-powered Data Lineage & Impact Analysis Engine (MVP)

Minimal Python MVP for lineage modeling, text/image retrieval, impact analysis, and simple RAG explanations.

## Install

```bash
pip install -r requirements.txt
```

## Run

Generate charts:

```bash
python src/main.py generate-charts
```

Text query:

```bash
python src/main.py query-text --question "What depends on sales.csv?"
```

Image query:

```bash
python src/main.py query-image --image-path data/charts/sales_summary_chart.png
```

Impact analysis:

```bash
python src/main.py impact --dataset sales.csv
```

## Notes

- If `OPENAI_API_KEY` is set, explanations can use OpenAI.
- Without OpenAI key or model downloads, the app falls back to deterministic mock behavior.
