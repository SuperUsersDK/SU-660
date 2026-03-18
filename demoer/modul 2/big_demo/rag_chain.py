from __future__ import annotations

import sys
import warnings

if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from settings import SETTINGS


def build_vs() -> QdrantVectorStore:
    client = QdrantClient(
        url=SETTINGS.qdrant_url,
        api_key=SETTINGS.qdrant_api_key or None,
    )
    embeddings = OpenAIEmbeddings(model=SETTINGS.embedding_model)
    return QdrantVectorStore(
        client=client,
        collection_name=SETTINGS.collection_name,
        embedding=embeddings,
    )


def _doc_citation(doc: Document) -> str:
    src = doc.metadata.get("source_file", "unknown")
    page = doc.metadata.get("page")
    if page is None:
        return f"[source={src}]"
    return f"[source={src} page={page}]"


def build_context(docs: list[Document]) -> str:
    parts: list[str] = []
    for doc in docs:
        parts.append(f"{_doc_citation(doc)}\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieval_hits_with_scores(question: str, k: int = 8) -> list[tuple[Document, float]]:
    vs = build_vs()
    return vs.similarity_search_with_score(question, k=k)


def rag_answer(question: str, k: int = 8) -> str:
    vs = build_vs()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant for technical documentation, markdown notes, code files, and PDFs. "
                "Answer ONLY using the retrieved context. "
                "Cite sources as [source=...] and include page when available. "
                "If the context is insufficient, say so clearly.",
            ),
            ("human", "Question: {q}\n\nContext:\n{ctx}"),
        ]
    )

    model = ChatOpenAI(model=SETTINGS.chat_model, temperature=0)
    rendered_messages = prompt.format_messages(q=question, ctx=context)
    print("\nFULDT PROMPT TIL LLM:\n")
    for message in rendered_messages:
        message_type = getattr(message, "type", message.__class__.__name__)
        print(f"[{message_type.upper()}]")
        print(message.content)
        print()

    response = (prompt | model).invoke({"q": question, "ctx": context})
    return response.content


def format_retrieval_score_lines(question: str, k: int = 8) -> list[str]:
    hits = retrieval_hits_with_scores(question, k=k)
    lines: list[str] = []
    for i, (doc, score) in enumerate(hits, start=1):
        src = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page")
        page_part = f" page={page}" if page is not None else ""
        lines.append(f"{i}. score={score:.4f} source={src}{page_part}")
    return lines


def retrieval_hits_summary(question: str, k: int = 8) -> list[dict[str, Any]]:
    hits = retrieval_hits_with_scores(question, k=k)
    out: list[dict[str, Any]] = []
    for i, (doc, score) in enumerate(hits, start=1):
        out.append(
            {
                "rank": i,
                "score": score,
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )
    return out
