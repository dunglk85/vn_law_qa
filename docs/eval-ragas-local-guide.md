# Running RAGAS Evaluation Locally

## Overview

`scripts/eval_ragas.py` measures your RAG pipeline quality using 4 metrics:

| Metric              | What it measures                                           |
|---------------------|------------------------------------------------------------|
| `faithfulness`      | Is the answer grounded in retrieved contexts?              |
| `answer_relevancy`  | How well does the answer address the question?             |
| `context_precision` | Were all retrieved contexts actually useful?               |
| `context_recall`    | Was the ground-truth answer covered by retrieved contexts? |

## Prerequisites

1. **Python virtual environment** with all dependencies installed:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **OpenAI API key** in `.env` (`OPENAI_API_KEY`) -- RAGAS uses an LLM-as-judge via `gpt-4o-mini`.

3. **PostgreSQL with pgvector** running (default: `localhost:5432`).

4. **Your app server running** (only for `evaluate` mode):

   ```powershell
   uvicorn app.api:app --reload
   ```

   The evaluate mode sends questions to `http://localhost:8000/ask`.

## Step 1 -- Generate Test Data

The `generate` mode creates Q&A pairs from your source Parquet documents:

```powershell
python scripts/eval_ragas.py generate --test-size 20
```

- Reads documents from `data/` (configured via `DATA_DIR` in `.env`).
- Uses RAGAS `TestsetGenerator` with `gpt-4o-mini` to produce question-answer pairs.
- Output: `seed/qna_test.jsonl` (JSONL format, one `{"question": ..., "answer": ...}` per line).

If you already have a test file, skip this step.

## Step 2 -- Run Evaluation

Make sure the server is running, then:

```powershell
python scripts/eval_ragas.py evaluate
```

What happens:

1. Reads test questions from `seed/qna_test.jsonl`.
2. POSTs each question to `http://localhost:8000/ask` (with auth token if `EVAL_API_TOKEN` is set).
3. Collects `answer` and `contexts` from the server response.
4. Runs RAGAS metrics against the reference answers.
5. Prints a table of per-question scores + averages.
6. Saves full results to `eval_results.json`.

### Using a custom test file

```powershell
python scripts/eval_ragas.py evaluate --test-path seed/my_custom_test.jsonl
```

## Quick One-Liner (full flow)

```powershell
python scripts/eval_ragas.py generate --test-size 20
python scripts/eval_ragas.py evaluate
```

## Understanding the Output

```
  Q | faithfulness | answer_relevancy | context_precision | context_recall
  1 | 0.95         | 0.88             | 0.78              | 0.82
  2 | 1.00         | 0.92             | 0.85              | 0.90
  ...
  Averages:
  - faithfulness: 0.97
  - answer_relevancy: 0.90
  - context_precision: 0.81
  - context_recall: 0.86
```

- **0.8+** is generally good for most metrics.
- **faithfulness < 0.7** → the model is hallucinating or using outside knowledge.
- **context_recall < 0.7** → retrieval is missing relevant chunks.
- **context_precision < 0.7** → too much noise in retrieved contexts.

## Environment Variables

| Variable           | Default               | Description                              |
|--------------------|-----------------------|------------------------------------------|
| `EVAL_API_TOKEN`   | (empty)               | Bearer token for authorized eval requests |
| `OPENAI_API_KEY`   | (from `.env`)         | Used by RAGAS for LLM-as-judge           |
| `LLM_MODEL`        | `gpt-4o-mini`         | Model RAGAS uses for scoring              |

Note: The evaluation hits the OpenAI API for every metric computation. Costs scale as `test_size x 4 metrics`.
