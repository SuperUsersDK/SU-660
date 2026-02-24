from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

try:
    from app.settings import SETTINGS
except ModuleNotFoundError:
    from settings import SETTINGS


def stable_id(text: str, source: str, page: int, idx: int) -> str:
    raw = f"{source}|{page}|{idx}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_pdfs(pdf_dir: str) -> list:
    docs = []
    for pdf_path in sorted(glob.glob(str(Path(pdf_dir) / "*.pdf"))):
        loader = PyPDFLoader(pdf_path)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = Path(pdf_path).name
        docs.extend(loaded)
    return docs


def main(pdf_dir: str = "data/pdfs") -> None:
    if not SETTINGS.openai_api_key:
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
    client = QdrantClient(url=SETTINGS.qdrant_url)
    client.get_collections()

    embeddings = OpenAIEmbeddings(model=SETTINGS.embedding_model)

    # from_documents creates the collection if missing and supports ids for idempotent re-runs.
    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=SETTINGS.qdrant_url,
        collection_name=SETTINGS.collection_name,
        ids=ids,
    )

    print(
        f"Ingested {len(chunks)} chunks from {len(docs)} pages "
        f"into collection={SETTINGS.collection_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--pdf-dir", default="data/pdfs", help="Directory containing PDF files")
    args = parser.parse_args()
    main(args.pdf_dir)

