# Law Crawler — Vietnamese Law Data Ingestion

Standalone batch tool for crawling and structuring Vietnamese legal documents into MySQL.

## What it does

| Crawler | Source | Output | Description |
|---------|--------|--------|-------------|
| `main.py` | `phap-dien/` HTML files & JSON | MySQL (peewee ORM) | Pháp Điển Việt Nam — chapters, articles, tables, cross-references |
| `document-crawler/main.py` | `vbpl.vn` (web) | MySQL (SQLAlchemy) | VBQPPL full-text documents linked from articles |
| `document-crawler/split_document.py` | `vbpl` table | `vb_chimuc` table | Splits full-text HTML into chapters/articles |

## Architecture

**Independent of the main RAG application.** The main app uses PostgreSQL + pgvector for vector search. This crawler uses MySQL for structured law data. They share no data store, no schema, and no runtime. Data must be exported/transformed if needed by the main app.

## Pipeline dependency

```
crawl_phap_dien → crawl_vbqppl → split_documents
```

- `crawl_phap_dien` must run first (creates `pddieu` table with links)
- `crawl_vbqppl` queries `pddieu` for `vbqppl_link` values
- `split_documents` queries `vbpl` for full-text content

## Usage

### Manual

```bash
# Start MySQL
docker compose up -d

# Install dependencies
pip install -r requirements.txt

# Run法Điển crawler (requires phap-dien/ data files)
python main.py

# Run VBQPPL crawler
python document-crawler/main.py

# Run document splitter
python document-crawler/split_document.py
```

### GitHub Actions (scheduled)

The workflow `.github/workflows/law-crawler.yml` runs weekly:

- **Schedule:** Sunday 02:00 UTC
- **Manual trigger:** Actions → Run workflow (optionally skip法Điển crawl)
- **MySQL:** Ephemeral service container (data in artifact)
- **Artifact:** `law-data-{run_id}` — SQL dump, retained 30 days

To load the crawled data into your own MySQL:

```bash
# Download the artifact from GitHub Actions
# Then import:
mysql -u root -p law < law_dump.sql
```

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `LAW_DB_HOST` | `localhost` | MySQL host |
| `LAW_DB_PORT` | `3306` | MySQL port |
| `LAW_DB_USER` | `root` | MySQL user |
| `LAW_DB_PASSWORD` | *(empty)* | MySQL password |
| `LAW_DB_NAME` | `law` | MySQL database |
| `LAW_CHECKPOINT` | *(none)* | Resume from specific HTML file |
