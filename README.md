# AI-powered Data Lineage & Impact Analysis Engine (MVP)

Minimal Python MVP for lineage modeling, text/image retrieval, impact analysis, and simple RAG explanations.

## Install

```bash
pip install -r requirements.txt
```

## Run

Generate charts:

```bash
python src/main.py --generate-charts
```

Text query:

```bash
python src/main.py --query "What depends on sales.csv?"
```

Image query:

```bash
python src/main.py --image data/charts/sales_chart.png
```

Impact analysis:

```bash
python src/main.py --impact sales.csv
```

Run all mandatory demo tests:

```bash
python src/main.py --demo-tests
```

## Notes

- If `OPENAI_API_KEY` is set, explanations can use OpenAI.
- Embedding models are attempted by default; set `AIR_ENABLE_REAL_MODELS=0` to force deterministic fallback embeddings.
- Without OpenAI key or model downloads, the app falls back to deterministic mock behavior.
