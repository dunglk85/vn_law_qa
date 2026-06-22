"""Gold layer: Text chunking — split documents for RAG embedding.

Reads flattened gold documents and splits into overlapping
text chunks suitable for vector embedding. Produces metadata-rich
chunks with source document tracing.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.settings import CHUNK_OVERLAP, CHUNK_SIZE, GOLD, setup_logging

logger = setup_logging(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of approximately chunk_size characters."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_chunks() -> pd.DataFrame:
    doc_path = GOLD / "law_documents.parquet"
    if not doc_path.exists():
        logger.warning("Gold law_documents not found at %s", doc_path)
        return pd.DataFrame()

    df = pd.read_parquet(doc_path)
    logger.info("Chunking %d documents", len(df))

    all_chunks: list[dict] = []
    for i in range(len(df)):
        row = df.iloc[i]
        article_id = row.get("mapc", f"doc_{i}")
        title = row.get("ten", "")
        text = row.get("full_context", row.get("noidung", ""))
        chude = row.get("chude_ten", "")
        demuc = row.get("demuc_ten", "")
        chuong = row.get("chuong_ten", "")

        chunks = chunk_text(text)
        for ci, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{article_id}_c{ci}",
                "article_id": article_id,
                "title": title,
                "chude": chude,
                "demuc": demuc,
                "chuong": chuong,
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "text": chunk,
            })

    logger.info("Produced %d chunks from %d articles", len(all_chunks), len(df))
    return pd.DataFrame(all_chunks)


def build_vbqppl_chunks() -> pd.DataFrame:
    vb_path = GOLD / "vbqppl_documents.parquet"
    if not vb_path.exists():
        logger.warning("Gold vbqppl_documents not found")
        return pd.DataFrame()

    df = pd.read_parquet(vb_path)
    logger.info("Chunking %d VBQPPL entries", len(df))

    all_chunks: list[dict] = []
    for i in range(len(df)):
        row = df.iloc[i]
        text = row.get("noi_dung", "")
        chunks = chunk_text(text)
        for ci, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"vb_{row['id']}_c{ci}",
                "source_id": row["id"],
                "source_type": "vbqppl",
                "parent_id": row.get("chi_muc_cha"),
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "text": chunk,
            })

    logger.info("Produced %d VBQPPL chunks", len(all_chunks))
    return pd.DataFrame(all_chunks)


def main() -> None:
    logger.info("=== Gold: Text chunking ===")

    df_chunks = build_chunks()
    if not df_chunks.empty:
        path = GOLD / "law_document_chunks.parquet"
        df_chunks.to_parquet(path, index=False)
        logger.info("Wrote %d chunks to %s", len(df_chunks), path)

    df_vb_chunks = build_vbqppl_chunks()
    if not df_vb_chunks.empty:
        path = GOLD / "vbqppl_chunks.parquet"
        df_vb_chunks.to_parquet(path, index=False)
        logger.info("Wrote %d VBQPPL chunks to %s", len(df_vb_chunks), path)

    logger.info("Gold chunking done")


if __name__ == "__main__":
    main()
