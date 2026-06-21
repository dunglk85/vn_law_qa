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

## Usage

```bash
# Start MySQL
docker compose up -d

# Install dependencies
pip install -r requirements.txt

# Run Pháp Điển crawler (requires phap-dien/ data files)
python main.py

# Run VBQPPL crawler
python document-crawler/main.py

# Run document splitter
python document-crawler/split_document.py
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
