"""Generate test Parquet from sample docs for QA pipeline testing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "gold"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def read_file(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return fp.read_text(encoding="latin-1")
        except Exception:
            return ""


def chunk_text(text: str, source: str, category: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text_content = text[start:end]
        chunk_id = hashlib.md5(f"{source}:{idx}".encode()).hexdigest()[:12]
        chunks.append({
            "chunk_id": chunk_id,
            "article_id": source.replace(".", "_"),
            "title": source,
            "chude": category,
            "demuc": "",
            "chuong": "",
            "chunk_index": idx,
            "total_chunks": 0,
            "text": chunk_text_content,
            "schema_version": "1.0.0",
        })
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    for c in chunks:
        c["total_chunks"] = idx
    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {DATA_DIR} for text files...")
    docs = []
    skipped = 0
    for fp in sorted(DATA_DIR.rglob("*")):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext in (".txt", ".md"):
            text = read_file(fp)
            if text.strip():
                docs.append((text, fp.relative_to(DATA_DIR).as_posix(), DATA_DIR.name))
            else:
                skipped += 1
        else:
            skipped += 1
    print(f"Loaded {len(docs)} text files, skipped {skipped} binary/other files")
    all_chunks = []
    for text, fname, cat in docs:
        all_chunks.extend(chunk_text(text, fname, cat))
    if not all_chunks:
        print("No chunks generated. Writing a single placeholder document.")
        all_chunks = [{
            "chunk_id": "placeholder_000001",
            "article_id": "company_overview",
            "title": "Company Overview",
            "chude": "general",
            "demuc": "",
            "chuong": "",
            "chunk_index": 0,
            "total_chunks": 1,
            "text": "This is a placeholder document for testing the RAG pipeline.",
            "schema_version": "1.0.0",
        }]
    df = pd.DataFrame(all_chunks)
    out_path = OUTPUT_DIR / "law_document_chunks.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Generated {len(df)} chunks -> {out_path}")
    print(f"  Sample chunk text: {df['text'].iloc[0][:80]}...")


if __name__ == "__main__":
    main()
