"""Ingest gold Parquet files into pgvector with batching."""
import asyncio
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "ingest-script-secret-not-for-production")
os.environ.setdefault("ADMIN_PASSWORD", "ingest-script-pwd-not-for-production")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

os.environ["DATA_DIR"] = str(project_root / "data" / "gold")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from app.config import config
from app.factory import create_document_loader, create_embeddings, create_vector_store, create_retriever

BATCH_SIZE = 500


async def main():
    print(f"DATA_DIR: {config.data_dir}")
    print(f"DATABASE_URL: {config.database_url}")

    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings=embeddings)
    retriever = create_retriever(vector_store=vector_store)
    loader = create_document_loader()

    print("Loading documents from Parquet...")
    docs = loader.load(config.data_dir)
    print(f"Loaded {len(docs)} documents total")

    if not docs:
        print("No documents loaded!")
        return

    total = len(docs)
    for i in range(0, total, BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        try:
            await vector_store.add_documents(batch)
            print(f"  Ingested batch {i//BATCH_SIZE + 1}/{(total-1)//BATCH_SIZE + 1} ({len(batch)} docs, {i+len(batch)}/{total})")
        except Exception as exc:
            print(f"  ERROR on batch {i//BATCH_SIZE + 1}: {exc}")
            raise

    print("Creating index...")
    try:
        await vector_store.create_index()
        retriever.build_index(docs)
    except Exception as exc:
        print(f"Index post-processing note: {exc}")

    print("Done - all documents ingested into pgvector!")


if __name__ == "__main__":
    asyncio.run(main())
