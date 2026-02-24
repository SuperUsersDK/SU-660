from __future__ import annotations

import argparse
import glob
import hashlib
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models as qdrant_models

SETTINGS = None


def _get_settings():
    global SETTINGS
    # Match demo_real behavior: let python-dotenv discover and load .env at runtime.
    # This happens before importing app.settings, so Settings captures the loaded values.
    load_dotenv()
    if SETTINGS is not None:
        return SETTINGS
    try:
        from app.settings import SETTINGS as loaded_settings
    except ModuleNotFoundError:
        from settings import SETTINGS as loaded_settings
    SETTINGS = loaded_settings
    return SETTINGS


def stable_id(text: str, source: str, page: int, idx: int) -> str:
    raw = f"{source}|{page}|{idx}|{text}".encode("utf-8")
    # Qdrant point IDs must be uint or UUID. Use deterministic UUIDv5 for idempotent re-runs.
    digest = hashlib.sha1(raw).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def load_pdfs(pdf_dir: str) -> list:
    docs = []
    for pdf_path in sorted(glob.glob(str(Path(pdf_dir) / "*.pdf"))):
        loader = PyPDFLoader(pdf_path)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = Path(pdf_path).name
        docs.extend(loaded)
    return docs


def ensure_collection(client: QdrantClient, collection_name: str, embeddings: OpenAIEmbeddings) -> None:
    if client.collection_exists(collection_name):
        return

    # Probe embedding size once so we can create the collection explicitly.
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def main(pdf_dir: str = "data/pdfs") -> None:
    settings = _get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY mangler")

    docs = load_pdfs(pdf_dir)
    if not docs:
        raise RuntimeError(f"Ingen PDF-filer fundet i {pdf_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    ids: list[str] = []
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = i
        chunk_id = stable_id(
            doc.page_content,
            str(doc.metadata.get("source_file", "unknown")),
            int(doc.metadata.get("page", -1)),
            int(doc.metadata.get("chunk_index", i)),
        )
        doc.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)

    # Quick connectivity check (raises early if Qdrant is not reachable).
    client = QdrantClient(url=settings.qdrant_url)
    client.get_collections()

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Avoid Qdrant.from_documents() here because some langchain/qdrant-client version
    # combinations break on `recreate_collection(init_from=...)`.
    ensure_collection(client, settings.collection_name, embeddings)
    vs = Qdrant(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
    )
    vs.add_documents(chunks, ids=ids)

    print(
        f"Ingested {len(chunks)} chunks from {len(docs)} pages "
        f"into collection={settings.collection_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Directory containing PDF files")
    args = parser.parse_args()
    main(args.pdf_dir)
