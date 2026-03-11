from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models as qdrant_models

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, CYAN, DIM, GREEN, MAGENTA, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

DATA_DIR = Path(__file__).resolve().parent / "data"
COLLECTION_NAME = "modul3_rag_pipeline_demo"
TOP_K = 3


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_chat_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def get_embed_model() -> str:
    return os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def get_qdrant_api_key() -> str:
    return os.getenv("QDRANT_API_KEY", "")


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name},
            )
        )
    return documents


def build_vectorstore(documents: list[Document]) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model=get_embed_model())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    client = QdrantClient(
        url=get_qdrant_url(),
        api_key=get_qdrant_api_key() or None,
    )

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def main() -> None:
    clear_screen()
    load_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY mangler i repo-roden .env")

    documents = load_documents()
    question = "Hvorfor er metadata-filtre nyttige i en klassisk RAG-pipeline?"

    print_section("Klassisk RAG Pipeline")
    print_label("PIPELINE", CYAN)
    print(colorize("User question", DIM))
    print(colorize("↓", DIM))
    print(colorize("Embedding", DIM))
    print(colorize("↓", DIM))
    print(colorize("Vector search", DIM))
    print(colorize("↓", DIM))
    print(colorize("Retrieve documents", DIM))
    print(colorize("↓", DIM))
    print(colorize("Prompt construction", DIM))
    print(colorize("↓", DIM))
    print(colorize("LLM", DIM))
    print(colorize("↓", DIM))
    print(colorize("Answer", DIM))

    print_label("LANGCHAIN-KOMPONENTER", BLUE)
    print(colorize("- OpenAIEmbeddings", BLUE))
    print(colorize("- QdrantVectorStore", BLUE))
    print(colorize("- Retriever", BLUE))
    print(colorize("- ChatPromptTemplate", BLUE))
    print(colorize("- ChatOpenAI", BLUE))
    prompt_to_continue("Tryk Enter for at starte pipeline-demoen...")

    print_section("1. Documents")
    print_label("KILDER", CYAN)
    for document in documents:
        print(colorize(f"- {document.metadata.get('source')}", CYAN))
    print(colorize(f"\nDokumenter indlæst: {len(documents)}", DIM))
    prompt_to_continue("Tryk Enter for at bygge chunks og vector store...")

    print_section("2. Chunking + Embeddings + Vector DB")
    vectorstore = build_vectorstore(documents)
    print(colorize("Dokumenterne er chunket, embedded og gemt i Qdrant.", GREEN))
    print(colorize(f"Collection: {COLLECTION_NAME}", DIM))
    print(colorize(f"Qdrant URL: {get_qdrant_url()}", DIM))
    print(colorize(f"Embedding-model: {get_embed_model()}", DIM))
    prompt_to_continue("Tryk Enter for at stille brugerens spørgsmål...")

    print_section("3. User Question")
    print_label("SPØRGSMÅL", YELLOW)
    print(colorize(question, YELLOW))
    prompt_to_continue("Tryk Enter for at køre vector search...")

    print_section("4. Vector Search + Retrieve Documents")
    hits = vectorstore.similarity_search_with_score(question, k=TOP_K)
    print_label("TOP HITS", MAGENTA)
    print(colorize(f"top_k={TOP_K}", DIM))
    for rank, (doc, score) in enumerate(hits, start=1):
        print(colorize(f"{rank}. score={score:.4f} source={doc.metadata.get('source')}", MAGENTA))
        print(colorize(doc.page_content[:220].replace("\n", " ") + "...", DIM))
        print()
    prompt_to_continue("Tryk Enter for at bygge prompten...")

    print_section("5. Prompt Construction")
    context = "\n\n".join(
        f"[{doc.metadata.get('source')}] {doc.page_content}"
        for doc, _ in hits
    )
    prompt = ChatPromptTemplate.from_template(
        "Du er en hjælpsom RAG-assistent.\n"
        "Svar kun ud fra konteksten.\n"
        "Hvis svaret ikke findes i konteksten, så sig det tydeligt.\n\n"
        "Kontekst:\n{context}\n\n"
        "Spørgsmål:\n{question}"
    )
    rendered_prompt = prompt.format(context=context, question=question)
    print_label("PROMPT", BLUE)
    print(colorize(rendered_prompt, DIM))
    prompt_to_continue("Tryk Enter for at sende prompten til LLM...")

    print_section("6. LLM + Answer")
    llm = ChatOpenAI(model=get_chat_model(), temperature=0)
    answer = (prompt | llm).invoke({"context": context, "question": question})
    print_label("SVAR", GREEN)
    print(colorize(answer.content, GREEN))


if __name__ == "__main__":
    main()
