from __future__ import annotations

import argparse
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
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models as qdrant_models

SETTINGS = None
APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DOCUMENT_DIR = APP_DIR / "data"

if str(_REPO_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, YELLOW, colorize, print_label, print_section

PDF_CHUNK_SIZE = 1400
PDF_CHUNK_OVERLAP = 180
MARKDOWN_CHUNK_SIZE = 1200
MARKDOWN_CHUNK_OVERLAP = 160
CODE_CHUNK_SIZE = 1000
CODE_CHUNK_OVERLAP = 120


def _get_settings():
    global SETTINGS
    # Load the single supported .env file from the git repo root.
    load_dotenv(_REPO_ROOT_DIR / ".env", override=True)
    if SETTINGS is not None:
        return SETTINGS
    from settings import SETTINGS as loaded_settings
    SETTINGS = loaded_settings
    return SETTINGS


def stable_id(text: str, source: str, page: int, idx: int) -> str:
    raw = f"{source}|{page}|{idx}|{text}".encode("utf-8")
    # Qdrant point IDs must be uint or UUID. Use deterministic UUIDv5 for idempotent re-runs.
    digest = hashlib.sha1(raw).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def load_text_file(file_path: str) -> list[Document]:
    path = Path(file_path)

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        return [Document(page_content=text, metadata={"source": str(path), "encoding": encoding})]

    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(path), "encoding": "utf-8-replace"})]


def iter_document_paths(document_dir: str) -> list[Path]:
    base_dir = Path(document_dir)
    if not base_dir.is_absolute():
        base_dir = APP_DIR / base_dir
    pdf_paths = sorted(base_dir.glob("*.pdf"))
    markdown_paths = sorted(base_dir.glob("*.md"))
    python_paths = sorted(base_dir.glob("*.py"))
    csharp_paths = sorted(base_dir.glob("*.cs"))
    return pdf_paths + markdown_paths + python_paths + csharp_paths


def load_document_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = path.name
            doc.metadata["source_type"] = "pdf"
        return loaded

    if path.suffix.lower() in {".md", ".py", ".cs"}:
        loaded = load_text_file(str(path))
        source_type_map = {
            ".md": "markdown",
            ".py": "python",
            ".cs": "csharp",
        }
        for doc in loaded:
            doc.metadata["source_file"] = path.name
            doc.metadata["source_type"] = source_type_map[path.suffix.lower()]
        return loaded

    return []


def ensure_collection(client: QdrantClient, collection_name: str, embeddings: OpenAIEmbeddings) -> None:
    if client.collection_exists(collection_name):
        return

    # Probe embedding size once so we can create the collection explicitly.
    dim = len(embeddings.embed_query("dimension probe"))
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
    except Exception as exc:
        message = str(exc)
        if "Collection data already exists" in message and collection_name in message:
            print_status(
                "COLLECT",
                (
                    f"Qdrant rapporterer, at {collection_name} allerede findes på disk. "
                    "Fortsætter med eksisterende data."
                ),
                YELLOW,
            )
            return
        raise


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


def chunk_markdown_documents(docs: list[Document]) -> list[Document]:
    chunked_docs: list[Document] = []
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MARKDOWN_CHUNK_SIZE,
        chunk_overlap=MARKDOWN_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    for source_doc in docs:
        header_docs = header_splitter.split_text(source_doc.page_content)
        if not header_docs:
            header_docs = [Document(page_content=source_doc.page_content, metadata={})]

        for idx, header_doc in enumerate(header_docs):
            merged_metadata = dict(source_doc.metadata)
            merged_metadata.update(header_doc.metadata)
            split_docs = body_splitter.create_documents(
                [header_doc.page_content],
                metadatas=[merged_metadata],
            )
            for chunk_index, chunk_doc in enumerate(split_docs):
                chunk_doc.metadata["chunk_strategy"] = "markdown_header_recursive"
                chunk_doc.metadata["section_index"] = idx
                chunk_doc.metadata["chunk_in_section"] = chunk_index
            chunked_docs.extend(split_docs)

    return chunked_docs


def chunk_code_documents(docs: list[Document], *, language: Language, strategy_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=CODE_CHUNK_SIZE,
        chunk_overlap=CODE_CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["chunk_strategy"] = strategy_name
    return chunks


def chunk_documents(docs: list[Document], source_type: str) -> list[Document]:
    if source_type == "markdown":
        return chunk_markdown_documents(docs)

    if source_type == "python":
        return chunk_code_documents(docs, language=Language.PYTHON, strategy_name="python_syntax_aware")

    if source_type == "csharp":
        return chunk_code_documents(docs, language=Language.CSHARP, strategy_name="csharp_syntax_aware")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF_CHUNK_SIZE,
        chunk_overlap=PDF_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["chunk_strategy"] = "pdf_recursive_prose"
    return chunks


def print_status(label: str, message: str, color: str = CYAN) -> None:
    print(colorize(f"{label:<10}", BOLD, color) + " " + colorize(message, DIM))


def build_vector_store(
    client: QdrantClient,
    collection_name: str,
    embeddings: OpenAIEmbeddings,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def main(document_dir: str = str(DEFAULT_DOCUMENT_DIR), force: bool = False) -> None:
    settings = _get_settings()
    collection_name = settings.collection_name

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY mangler")

    document_paths = iter_document_paths(document_dir)
    if not document_paths:
        raise RuntimeError(f"Ingen .pdf, .md, .py eller .cs filer fundet i {document_dir}")

    print_section("Ingest Demo")
    print_label("CONFIG", CYAN)
    print(colorize(f"Document dir: {document_dir}", DIM))
    print(colorize(f"Collection:   {collection_name}", DIM))
    print(colorize(f"Embed model:  {settings.embedding_model}", DIM))
    print(colorize(f"Kildedokumenter fundet: {len(document_paths)}", DIM))
    print(colorize(f"Markdown chunking: header-aware + recursive ({MARKDOWN_CHUNK_SIZE} tegn, overlap {MARKDOWN_CHUNK_OVERLAP})", DIM))
    print(colorize(f"Python/C# chunking: syntax-aware ({CODE_CHUNK_SIZE} tegn, overlap {CODE_CHUNK_OVERLAP})", DIM))
    print(colorize(f"PDF chunking: prose-aware recursive ({PDF_CHUNK_SIZE} tegn, overlap {PDF_CHUNK_OVERLAP})", DIM))

    # Quick connectivity check (raises early if Qdrant is not reachable).
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    print_status("QDRANT", f"Forbinder til {settings.qdrant_url}")
    client.get_collections()
    print_status("QDRANT", "Forbindelse OK", GREEN)
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    if not client.collection_exists(collection_name):
        print_status("COLLECT", f"Opretter collection {collection_name}", YELLOW)
        ensure_collection(client, collection_name, embeddings)
        print_status("COLLECT", "Collection oprettet", GREEN)
    else:
        print_status("COLLECT", f"Bruger eksisterende collection {collection_name}", GREEN)
    vs = build_vector_store(client, collection_name, embeddings)

    total_source_docs = 0
    total_chunks = 0
    total_new_chunks = 0

    for path in document_paths:
        print_label("DOKUMENT", BLUE)
        print(colorize(path.name, BOLD, BLUE))
        print_status("LOAD", f"Indlæser {path.suffix.lower()}-fil")
        docs = load_document_file(path)
        if not docs:
            print_status("SKIP", f"Kunne ikke indlæse {path.name}", RED)
            continue

        total_source_docs += 1
        print_status("LOAD", f"Indlæst {len(docs)} dokumentdele", GREEN)
        print_status("CHUNK", "Splitter dokument i chunks")
        source_type = str(docs[0].metadata.get("source_type", "unknown"))
        chunks = chunk_documents(docs, source_type)
        strategy = str(chunks[0].metadata.get("chunk_strategy", "unknown")) if chunks else "unknown"
        print_status("CHUNK", f"Lavet {len(chunks)} chunks", GREEN)
        print_status("METHOD", strategy, CYAN)
        ids: list[str] = []
        for i, doc in enumerate(chunks):
            doc.metadata["chunk_index"] = i
            chunk_id = stable_id(
                doc.page_content,
                str(doc.metadata.get("source_file", path.name)),
                int(doc.metadata.get("page", -1)),
                int(doc.metadata.get("chunk_index", i)),
            )
            doc.metadata["chunk_id"] = chunk_id
            ids.append(chunk_id)

        total_chunks += len(chunks)
        new_chunks = chunks
        new_ids = ids

        if not force:
            print_status("CHECK", "Tjekker hvilke chunks der allerede findes i Qdrant")
            existing_ids = existing_ids_in_collection(client, collection_name, ids)
            new_chunks = []
            new_ids = []
            for chunk, chunk_id in zip(chunks, ids):
                if chunk_id in existing_ids:
                    print(colorize("SKIP      ", BOLD, YELLOW) + colorize(_chunk_label(chunk, chunk_id), DIM))
                    continue
                print(colorize("INSERT    ", BOLD, CYAN) + colorize(_chunk_label(chunk, chunk_id), DIM))
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

            if not new_ids:
                print_status("STORE", f"Ingen nye chunks for {path.name} ({len(chunks)} findes allerede)", YELLOW)
                continue
        else:
            print_status("FORCE", "Springer dublet-tjek over og upserter alle chunks", YELLOW)
            for chunk, chunk_id in zip(chunks, ids):
                print(colorize("FORCE     ", BOLD, MAGENTA) + colorize(_chunk_label(chunk, chunk_id), DIM))

        print_status("EMBED", f"Genererer embeddings for {len(new_chunks)} chunks")
        print_status("STORE", "Gemmer chunks i Qdrant")
        vs.add_documents(new_chunks, ids=new_ids)
        total_new_chunks += len(new_chunks)
        print_status(
            "DONE",
            f"Ingested {len(new_chunks)} nye chunks ud af {len(chunks)} fra {path.name}",
            GREEN,
        )

    print_section("Færdig")
    print(
        colorize(
            f"Finished ingest: {total_new_chunks} new chunks (of {total_chunks} total) "
            f"from {total_source_docs} source docs into collection={collection_name}",
            BOLD,
            GREEN,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF/Markdown/code files into Qdrant")
    parser.add_argument(
        "--document-dir",
        "--pdf-dir",
        dest="document_dir",
        default=str(DEFAULT_DOCUMENT_DIR),
        help="Directory containing .pdf, .md, .py and .cs files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and upsert all chunks even if they already exist",
    )
    args = parser.parse_args()
    main(args.document_dir, force=args.force)
