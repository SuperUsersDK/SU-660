from __future__ import annotations

import argparse
import glob
import hashlib
import sys
import uuid
import warnings
from pathlib import Path

# LangChain currently emits this warning under Python 3.14 due to upstream
# pydantic.v1 compatibility shims. Suppress the noisy warning for the demo CLI.
if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
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


def load_documents(document_dir: str) -> list:
    docs = []
    for pdf_path in sorted(glob.glob(str(Path(document_dir) / "*.pdf"))):
        loader = PyPDFLoader(pdf_path)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = Path(pdf_path).name
            doc.metadata["source_type"] = "pdf"
        docs.extend(loaded)
    for txt_path in sorted(glob.glob(str(Path(document_dir) / "*.txt"))):
        loader = TextLoader(txt_path, autodetect_encoding=True)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = Path(txt_path).name
            doc.metadata["source_type"] = "txt"
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


def existing_ids_in_collection(
    client: QdrantClient, collection_name: str, ids: list[str], batch_size: int = 256
) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        points = client.retrieve(
            collection_name=collection_name,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            found.add(str(point.id))
    return found


def _chunk_label(doc, chunk_id: str) -> str:
    src = str(doc.metadata.get("source_file", "unknown"))
    page = str(doc.metadata.get("page", "?"))
    idx = str(doc.metadata.get("chunk_index", "?"))
    return f"{src} page={page} chunk={idx} id={chunk_id}"


def main(document_dir: str = "data/pdfs", force: bool = False) -> None:
    settings = _get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY mangler")

    docs = load_documents(document_dir)
    if not docs:
        raise RuntimeError(f"Ingen PDF/TXT-filer fundet i {document_dir}")

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
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    client.get_collections()
    collection_exists = client.collection_exists(settings.collection_name)

    new_chunks = chunks
    new_ids = ids
    if collection_exists and not force:
        existing_ids = existing_ids_in_collection(client, settings.collection_name, ids)
        new_chunks = []
        new_ids = []
        for chunk, chunk_id in zip(chunks, ids):
            if chunk_id in existing_ids:
                print(f"SKIP    {_chunk_label(chunk, chunk_id)}")
                continue
            print(f"INSERT  {_chunk_label(chunk, chunk_id)}")
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

        if not new_ids:
            print(
                f"No new chunks to ingest ({len(chunks)} already present) "
                f"in collection={settings.collection_name}"
            )
            return
    elif force:
        for chunk, chunk_id in zip(chunks, ids):
            print(f"FORCE   {_chunk_label(chunk, chunk_id)}")
    else:
        for chunk, chunk_id in zip(chunks, ids):
            print(f"INSERT  {_chunk_label(chunk, chunk_id)}")

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Avoid Qdrant.from_documents() here because some langchain/qdrant-client version
    # combinations break on `recreate_collection(init_from=...)`.
    if not collection_exists:
        ensure_collection(client, settings.collection_name, embeddings)
    vs = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )
    vs.add_documents(new_chunks, ids=new_ids)

    print(
        f"Ingested {len(new_chunks)} new chunks (of {len(chunks)} total) from {len(docs)} source docs "
        f"into collection={settings.collection_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF/TXT files into Qdrant")
    parser.add_argument(
        "--document-dir",
        "--pdf-dir",
        dest="document_dir",
        default="data/pdfs",
        help="Directory containing PDF/TXT files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and upsert all chunks even if they already exist",
    )
    args = parser.parse_args()
    main(args.document_dir, force=args.force)
