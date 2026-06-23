"""Gold layer: Text chunking — split documents for RAG embedding.

Reads flattened gold documents and splits into overlapping
text chunks suitable for vector embedding. Produces metadata-rich
chunks with source document tracing.

Chunking is paragraph-aware: it prefers to break at paragraph
boundaries rather than cutting mid-sentence.
"""
import re

import pandas as pd

from src.schema import LawDocumentChunk, VBQPPLChunk
from src.settings import CHUNK_OVERLAP, CHUNK_SIZE, GOLD, setup_logging

logger = setup_logging(__name__)

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_MIN_STEP_FACTOR = 10


def _prefer_boundary(text: str, pos: int, min_pos: int, max_pos: int) -> int:
    """Adjust a cut position to the nearest paragraph or sentence boundary.

    Scans forward from *pos* within *max_pos* for a paragraph break,
    then a sentence break. If none found within range, falls back to
    the original position.
    """
    # Try paragraph boundary first
    for m in _PARAGRAPH_RE.finditer(text, pos, max_pos):
        if m.start() >= min_pos:
            return m.start()
    # Then sentence boundary
    for m in _SENTENCE_RE.finditer(text, pos, max_pos):
        if m.start() >= min_pos:
            return m.start() + 1
    return pos


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks with boundary awareness.

    Each chunk is at most *chunk_size* characters. Chunks prefer
    paragraph breaks over sentence breaks over arbitrary character cuts.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
        )
    min_step = max(1, chunk_size // _MIN_STEP_FACTOR)
    step = chunk_size - overlap
    if step < min_step:
        raise ValueError(
            f"chunk_size ({chunk_size}) minus overlap ({overlap}) = {step}, "
            f"minimum step is {min_step} to avoid excessive chunk generation"
        )

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            adjusted = _prefer_boundary(text, end, start + step, min(len(text), end + chunk_size // 2))
            end = adjusted
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def build_chunks() -> pd.DataFrame:
    doc_path = GOLD / "law_documents.parquet"
    if not doc_path.exists():
        logger.warning("Gold law_documents not found at %s", doc_path)
        return pd.DataFrame()

    df = pd.read_parquet(doc_path)
    logger.info("Chunking %d documents", len(df))

    all_chunks: list[LawDocumentChunk] = []
    for i in range(len(df)):
        row = df.iloc[i]
        article_id = row.get("mapc", f"doc_{i}")
        title = row.get("ten", "")
        text = row.get("full_context", row.get("noidung", ""))
        chude = row.get("chude_ten", "")
        demuc = row.get("demuc_ten", "")
        chuong = row.get("chuong_ten", "")

        chunks = chunk_text(text)
        for ci, chunk_text_val in enumerate(chunks):
            all_chunks.append(LawDocumentChunk(
                chunk_id=f"{article_id}_c{ci}",
                article_id=article_id,
                title=title,
                chude=chude,
                demuc=demuc,
                chuong=chuong,
                chunk_index=ci,
                total_chunks=len(chunks),
                text=chunk_text_val,
            ))

    result = pd.DataFrame([m.model_dump() for m in all_chunks])
    logger.info("Produced %d chunks from %d articles", len(result), len(df))
    return result


def build_vbqppl_chunks() -> pd.DataFrame:
    vb_path = GOLD / "vbqppl_documents.parquet"
    if not vb_path.exists():
        logger.warning("Gold vbqppl_documents not found")
        return pd.DataFrame()

    df = pd.read_parquet(vb_path)
    logger.info("Chunking %d VBQPPL entries", len(df))

    all_chunks: list[VBQPPLChunk] = []
    for i in range(len(df)):
        row = df.iloc[i]
        text_val = row.get("noi_dung", "")
        chunks = chunk_text(text_val)
        for ci, chunk_text_val in enumerate(chunks):
            all_chunks.append(VBQPPLChunk(
                chunk_id=f"vb_{row['id']}_c{ci}",
                source_id=row["id"],
                chunk_index=ci,
                total_chunks=len(chunks),
                text=chunk_text_val,
                parent_id=row.get("chi_muc_cha"),
            ))

    result = pd.DataFrame([m.model_dump() for m in all_chunks])
    logger.info("Produced %d VBQPPL chunks", len(result))
    return result


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
