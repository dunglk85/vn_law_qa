# ADR-0001: Law Crawler Integration with Main RAG Application

**Status:** Accepted  
**Date:** 2026-06-22  
**Author:** Winston (System Architect)  
**Reviewed by:** Admin  
**Supersedes:** Initial proposal (Parquet loader in main app)

---

## Summary

**Decision:** Hybrid approach — remove dead file-based loaders from the main app, add a minimal Parquet loader for gold chunks. The crawler stays independent and outputs Parquet files. The main app reads them via a simplified Parquet loader.

**Rationale:** Law data is currently the only source, so PDF/DOCX/MD/TXT loaders are dead code. But we keep the Parquet loader port/adapter to avoid coupling the crawler to the main app, preserve DVC caching, and leave room for future data sources.

---

## Context

The `law-crawler/` module is a standalone batch pipeline that ingests Vietnamese legal documents from two sources:

1. **Pháp Điển** (phap-dien HTML/JSON files) — structured law articles with hierarchical metadata (Chủ đề → Đề mục → Chương → Điều)
2. **VBQPPL** (vbpl.vn web crawl) — full-text legal documents linked from articles

The crawler uses a **Medallion architecture** (Bronze → Silver → Gold) orchestrated by DVC:

```
law-crawler/
├── src/bronze/     # Raw ingestion → Parquet
├── src/silver/     # Cleaning, validation, dedup → Parquet
├── src/gold/       # Flattened documents + chunks → Parquet
└── metrics/        # Quality checks (DVC metrics)
```

**The problem:** The gold layer produces RAG-ready chunks (`law_document_chunks.parquet`, `vbqppl_chunks.parquet`), but the main application (`app/`) has no code to consume them. The `ingest_service.py` only loads files via LangChain document loaders (PDF, DOCX, MD, TXT) — it does not read Parquet.

The law-crawler pipeline is currently a **dead-end**: data flows in, gets processed through bronze/silver/gold, then stops. The main application's pgvector store never receives it.

Additionally, several technical debt items have been identified:

| Issue | Severity | Location |
|-------|----------|----------|
| `params.yaml` not wired to code | High | `src/settings.py` reads env vars, not YAML |
| `sys.path.insert(0, ...)` in 7 files | High | All pipeline modules |
| Duplicate `setup_logging()` | Medium | `db.py` and `src/settings.py` |
| Legacy MySQL path orphaned | Medium | `main.py`, `document-crawler/`, `models/` |
| Gold layer `.apply()` performance | Medium | `src/gold/documents.py:75-84` |
| `.gitignore` stray markdown fence | Low | Line 172 |

---

## Alternatives Considered

### Option A: Move Embed into Crawler (Rejected)

Crawler imports `app.factory` to embed and store directly into pgvector.

| Pro | Con |
|-----|-----|
| Single pipeline, one command | **Coupling:** crawler imports `app.factory`, `app.config` |
| Removes dead code | **CI needs secrets:** `DATABASE_URL`, `OPENAI_API_KEY` in GitHub Actions |
| Atomic end-to-end | **DVC caching breaks:** embed writes to DB, not files |
| No schema contract needed | **Harder to test:** needs real pgvector in tests |
| | **No going back:** if a second source is needed, must rebuild |

### Option B: Parquet Loader Only (Rejected)

Add Parquet loader to main app, keep all existing PDF/DOCX/MD/TXT loaders.

| Pro | Con |
|-----|-----|
| Preserves flexibility | Keeps ~200 lines of dead loader code |
| Crawler stays independent | Two ingestion paths creates confusion |
| DVC caching works | More code to test and maintain |

### Option C: Hybrid (Accepted)

Remove dead file-based loaders, add minimal Parquet loader. Crawler stays independent.

| Pro | Con |
|-----|-----|
| Removes dead code (~200 lines) | Still has artifact handoff (Parquet → app) |
| Crawler stays independent | Slightly more code than Option A |
| DVC caching works fully | |
| Easy to add new sources later | |
| CI stays simple (no secrets for crawl) | |
| Hexagonal architecture preserved | |

---

## Decision

### Architecture

```
law-crawler/                        app/
├── src/bronze/                     ├── core/
├── src/silver/                     │   └── ingest_service.py  (simplified)
├── src/gold/                       ├── ports/
│   ├── law_document_chunks.parquet │   └── document_loader.py (kept)
│   └── vbqppl_chunks.parquet       ├── adapters/
└── metrics/quality.json            │   └── document_loaders/
                                    │       └── parquet_loader.py (new, ~80 lines)
                                    │
                                    │   REMOVED:
                                    │   ├── adapters/chunkers/ (dead code)
                                    │   ├── adapters/metadata_enrichers/ (dead code)
                                    │   ├── ports/chunking.py (dead code)
                                    │   └── ports/metadata_enrichment.py (dead code)
                                    └── config.py (simplified)
```

**Data flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        law-crawler/                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Bronze  │───▶│  Silver  │───▶│   Gold   │                   │
│  │          │    │          │    │          │                   │
│  │ HTML/JSON│    │ Clean &  │    │ Flatten  │                   │
│  │ → Parquet│    │ Validate │    │ & Chunk  │                   │
│  └──────────┘    └──────────┘    └────┬─────┘                   │
│                                       │                          │
└───────────────────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
                              data/gold/*.parquet
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                           app/                                   │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ ParquetLoader    │───▶│  Embeddings  │───▶│  pgvector    │  │
│  │ Adapter          │    │  (OpenAI)    │    │  (store)     │  │
│  └──────────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                      │          │
│  ┌──────────────────────────────────────────────────┐│          │
│  │  RAG Service / Agents / API  (query only)        ││          │
│  └──────────────────────────────────────────────────┘│          │
│                                                      │          │
└──────────────────────────────────────────────────────┼──────────┘
```

### Implementation Details

**1. New: `app/adapters/document_loaders/parquet_loader.py`**

Reads gold chunk Parquet files and converts to LangChain `Document` objects:

```python
class ParquetLoaderAdapter(DocumentLoaderPort):
    def load(self, path: str) -> list[Document]:
        df = pd.read_parquet(path)
        return [
            Document(
                page_content=row["text"],
                metadata={
                    "source": "law-crawler",
                    "article_id": row.get("article_id", ""),
                    "chude": row.get("chude", ""),
                    "demuc": row.get("demuc", ""),
                    "chuong": row.get("chuong", ""),
                    "chunk_index": row.get("chunk_index", 0),
                },
            )
            for _, row in df.iterrows()
        ]
```

**2. Simplified: `app/core/ingest_service.py`**

Remove PDF/DOCX/MD/TXT loaders. Keep only Parquet loading + embedding + storage.

**3. Removed from main app:**

| File/Dir | Reason |
|----------|--------|
| `app/adapters/chunkers/` | Chunking done by crawler gold layer |
| `app/adapters/metadata_enrichers/` | No metadata enrichment needed |
| `app/ports/chunking.py` | No chunking port needed |
| `app/ports/metadata_enrichment.py` | No enrichment port needed |
| `app/api.py` `/ingest` endpoint | Replaced by crawler batch job |

**4. Updated: `app/config.py`**

Remove: `chunker_type`, `metadata_enricher_type`  
Keep: `data_dir` (points to crawler gold output or artifact download location)

### Technical Debt Remediation

| Priority | Action | Effort | Rationale |
|----------|--------|--------|-----------|
| P1 | Wire `params.yaml` → code | Small | DVC param tracking is broken without this |
| P1 | Remove `sys.path.insert` — add `pyproject.toml` | Medium | Improves maintainability, IDE support |
| P2 | Vectorize `full_context` construction | Small | Performance on large datasets |
| P2 | Consolidate `setup_logging()` | Trivial | Single source of truth |
| P3 | Delete legacy MySQL path | Small | Reduces confusion once medallion is proven |
| P3 | Fix `.gitignore` stray fence | Trivial | Cleanup |

---

## Rationale

### Why Hybrid Over Full Crawler Integration?

1. **Crawler stays independent** — No imports from `app/`. Can be developed, tested, and deployed separately.
2. **DVC caching works** — All stages produce file outputs. No "write to database" stage that bypasses DVC.
3. **CI stays simple** — GitHub Actions only needs Python + pip. No `DATABASE_URL` or `OPENAI_API_KEY` secrets.
4. **Easy to add sources later** — If a second data source emerges (court decisions, company policies), just add a new loader adapter. The port/adapter pattern is already in place.
5. **Hexagonal architecture preserved** — The main app doesn't reach into the crawler. The crawler doesn't reach into the app. They communicate via files.
6. **Dead code removed** — PDF/DOCX/MD/TXT loaders, chunkers, and enrichers are gone. ~200 lines removed.

### Why Not Full Crawler Integration?

- **Coupling:** Crawler importing `app.factory` creates a hard dependency. Any refactor in the app breaks the crawler.
- **CI complexity:** Embedding in CI requires OpenAI API keys and database access. Forks can't run CI without secrets.
- **DVC caching:** Database writes can't be cached by DVC. Every `dvc repro` re-embeds everything.
- **Testing:** Crawler tests would need a real pgvector instance or complex mocking.
- **No going back:** If a second source is needed, you'd need to rebuild file-based ingestion from scratch.

### Why Not Keep All Existing Loaders?

- **Dead code:** Law data is the only source. PDF/DOCX loaders are never called.
- **Cognitive load:** Two ingestion paths creates confusion.
- **Maintenance burden:** More code to test, document, and keep working.

---

## Consequences

### Positive

- **Clean separation:** Crawler produces files, app consumes them. No coupling.
- **Simpler main app:** Removes ~200 lines of dead code, 2 ports, 2 adapter dirs.
- **DVC caching preserved:** All pipeline stages produce file outputs.
- **CI stays simple:** No secrets needed for crawl pipeline.
- **Extensible:** Adding a new data source = adding a new loader adapter.
- **Independent deployment:** Crawler and app can be updated separately.

### Negative

- **Artifact handoff:** Gold Parquet files must be transferred from crawler to app. In CI, this means downloading the artifact. In production, this means a shared volume or download step.
- **Not fully atomic:** There's a gap between "chunks produced" and "chunks embedded." If the embed step fails, you have orphaned chunks.
- **Schema contract needed:** The Parquet schema must be stable between crawler and loader. Changes to gold chunk columns require updating the loader.

### Risks

| Risk | Mitigation |
|------|------------|
| Parquet schema drift | Define chunk schema as a Pydantic model shared between crawler and loader |
| Gold chunks not found by app | Document the expected path; make it configurable via `DATA_DIR` |
| Embed step fails after successful crawl | Add retry logic; log clearly which chunks are embedded |
| Legacy MySQL path has external users | Announce deprecation; keep for one release cycle |
| Future source needs chunking in app | Re-add chunking port/adapter when needed (YAGNI until then) |

---

## Implementation Plan

### Phase 1: Add Parquet Loader to Main App (P0)

1. Create `app/adapters/document_loaders/parquet_loader.py` (~80 lines)
2. Simplify `app/core/ingest_service.py` — remove PDF/DOCX/MD/TXT loaders, use Parquet loader
3. Update `app/config.py` — `data_dir` points to gold chunk location

### Phase 2: Remove Dead Code from Main App (P0)

1. Delete `app/adapters/chunkers/`
2. Delete `app/adapters/metadata_enrichers/`
3. Delete `app/ports/chunking.py`
4. Delete `app/ports/metadata_enrichment.py`
5. Remove `/ingest` endpoint from `app/api.py`
6. Remove `chunker_type`, `metadata_enricher_type` from `app/config.py`
7. Update `app/factory.py` — remove chunker and enricher factory methods
8. Update `tests/test_architecture.py` — remove tests for deleted modules
9. Update `docs/architecture.md` — reflect simplified architecture

### Phase 3: Crawler Technical Debt (P1-P2)

1. Wire `params.yaml` → `src/settings.py` (read YAML directly)
2. Add `law-crawler/pyproject.toml` and remove all `sys.path.insert`
3. Vectorize `full_context` in `src/gold/documents.py`
4. Consolidate `setup_logging()` into `src/settings.py` only

### Phase 4: Crawler Cleanup (P3)

1. Delete `main.py`, `document-crawler/`, `models/`, `db.py`
2. Fix `.gitignore` stray fence
3. Update `law-crawler/README.md` to reflect medallion-only architecture

---

## See Also

- [`law-crawler/README.md`](../law-crawler/README.md) — Crawler documentation
- [`docs/architecture.md`](../architecture.md) — Main application architecture
- [`law-crawler/dvc.yaml`](../law-crawler/dvc.yaml) — Pipeline definition
- [`law-crawler/src/gold/chunks.py`](../law-crawler/src/gold/chunks.py) — Gold chunk schema
