# Law Crawler — Vietnamese Law Data Ingestion

Standalone batch pipeline for crawling and structuring Vietnamese legal documents
into Parquet files ready for RAG embedding.

## Architecture

A **medallion pipeline** (Bronze → Silver → Gold) managed by [DVC](https://dvc.org/).
Independent of the main RAG application — produces Parquet files consumed by the seeder.

```
Bronze                     Silver                      Gold
──────────────────────     ────────────────────────    ──────────────────────────
phap_dien.py               phap_dien.py                documents.py
  HTML/JSON → Parquet  →     Clean & validate       →    Denormalize + enrich
                                                         (full_context field)
vbqppl.py                  vbqppl.py                   chunks.py
  vbpl.vn web crawl    →     Split chapters/articles →   Chunk for embedding
                           quality.py
                             DVC metric report
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python src/pipeline.py

# Run only a specific layer
python src/pipeline.py --stage bronze   # or silver / gold

# Run from a layer onward
python src/pipeline.py --from silver

# With DVC (caching + incremental runs)
dvc repro
```

## Resuming an interrupted crawl

The VBQPPL web crawl (`bronze/vbqppl.py`) is crash-resumable: on startup it reads
any existing `data/bronze/vbqppl/vbpl.parquet` and skips already-fetched item IDs.

The Pháp Điển HTML ingest (`bronze/phap_dien.py`) supports file-level checkpointing
via the `LAW_CHECKPOINT` env var (or `checkpoint` in `params.yaml`):

```bash
LAW_CHECKPOINT=0012.html python -m src.bronze.phap_dien
```

## Configuration

| Env var | `params.yaml` key | Default | Purpose |
|---------|-------------------|---------|---------|
| `LAW_CHECKPOINT` | `checkpoint` | `""` | Resume Pháp Điển ingest from this HTML filename |
| `LAW_CHUNK_SIZE` | `chunk_size` | `1000` | Characters per RAG chunk |
| `LAW_CHUNK_OVERLAP` | `chunk_overlap` | `200` | Overlap between adjacent chunks |
| `LAW_CRAWL_DELAY` | `crawl_delay` | `0.5` | Seconds between VBQPPL HTTP requests |

## Pipeline outputs

| File | Description |
|------|-------------|
| `data/gold/law_documents.parquet` | Denormalized articles with hierarchical context |
| `data/gold/law_document_chunks.parquet` | Chunked Pháp Điển articles for embedding |
| `data/gold/vbqppl_documents.parquet` | VBQPPL chapters/articles |
| `data/gold/vbqppl_chunks.parquet` | Chunked VBQPPL records for embedding |
| `metrics/quality.json` | DVC-tracked data quality report |

## Running tests

```bash
pytest tests/ -v
```
