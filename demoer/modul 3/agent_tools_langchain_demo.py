from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models as qdrant_models

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BLUE, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, YELLOW, clear_screen, colorize, print_label, print_section, prompt_to_continue

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = Path(__file__).resolve().parent / "data" / "agent_demo.db"
COLLECTION_NAME = "modul3_agent_tools_demo"


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


def print_status(label: str, message: str, color: str = CYAN) -> None:
    print(colorize(f"{label:<10}", BOLD, color) + " " + colorize(message, DIM))


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def build_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(
        url=get_qdrant_url(),
        api_key=get_qdrant_api_key() or None,
    )
    embeddings = OpenAIEmbeddings(model=get_embed_model())
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

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


def ensure_sqlite_db() -> None:
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS service_info (
                service_name TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                sla_hours INTEGER NOT NULL,
                note TEXT NOT NULL
            )
            """
        )
        cursor.execute("DELETE FROM service_info")
        cursor.executemany(
            "INSERT INTO service_info(service_name, owner, sla_hours, note) VALUES (?, ?, ?, ?)",
            [
                ("vector-search", "Platform Team", 4, "Supports retrieval over the Qdrant collections."),
                ("chat-api", "AI Applications", 8, "Handles answer generation and orchestration."),
                ("document-import", "Data Engineering", 24, "Responsible for ingest and chunking jobs."),
            ],
        )
        connection.commit()
    finally:
        connection.close()


VECTORSTORE: QdrantVectorStore | None = None


@tool
def search_documents(query: str) -> str:
    """Search the Qdrant document collection and return the most relevant chunks."""
    global VECTORSTORE
    if VECTORSTORE is None:
        return "Document search is not ready."

    print_status("TOOL docs", f"Søger i Qdrant efter: {query}", MAGENTA)
    hits = VECTORSTORE.similarity_search(query, k=3)
    if not hits:
        return "[docs] Ingen relevante dokumenter fundet."

    parts = []
    for hit in hits:
        parts.append(f"[docs source={hit.metadata.get('source')}] {hit.page_content}")
    return "\n\n".join(parts)


@tool
def lookup_database(service_name: str) -> str:
    """Look up internal service ownership and SLA information in the local SQLite database."""
    print_status("TOOL db", f"Slår op i SQLite for service: {service_name}", YELLOW)
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT service_name, owner, sla_hours, note FROM service_info WHERE service_name = ?",
            (service_name,),
        )
        row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        return f"[db] Ingen service fundet med navn {service_name}."
    service, owner, sla_hours, note = row
    return f"[db] service={service}, owner={owner}, sla_hours={sla_hours}, note={note}"


@tool
def web_search(topic: str) -> str:
    """Search the web via Wikipedia summary API and return a short summary."""
    print_status("TOOL web", f"Søger på web efter: {topic}", BLUE)
    encoded_topic = urllib.parse.quote(topic.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SU-660-demo-agent/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return f"[web] Web search failed: {exc}"

    extract = payload.get("extract")
    if not extract:
        return f"[web] Ingen kort opsummering fundet for {topic}."
    return f"[web] {extract}"


@tool
def github_repo_stats(repo: str) -> str:
    """Fetch public GitHub repository stats via GitHub's REST API, for example 'langchain-ai/langchain'."""
    print_status("TOOL api", f"Kalder GitHub API for repo: {repo}", GREEN)
    url = f"https://api.github.com/repos/{repo}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SU-660-demo-agent/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return f"[api] API-kald fejlede: {exc}"

    if "stargazers_count" not in payload:
        return f"[api] Ingen repo-data fundet for {repo}."
    return (
        f"[api] repo={payload['full_name']}, stars={payload['stargazers_count']}, "
        f"forks={payload['forks_count']}, open_issues={payload['open_issues_count']}"
    )


def build_agent() -> Any:
    llm = ChatOpenAI(model=get_chat_model(), temperature=0)
    tools = [search_documents, lookup_database, web_search, github_repo_stats]
    return create_agent(
        llm,
        tools=tools,
        system_prompt=(
            "Du er en AI-assistent med adgang til flere datakilder. "
            "Brug tools når brugeren spørger om dokumenter, interne serviceoplysninger, web-viden eller API-data. "
            "Når du svarer, så nævn tydeligt hvilke dele der kommer fra docs, db, web eller api. "
            "Hvis et tool fejler, så sig det i stedet for at gætte."
        ),
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content) if content is not None else str(message)


def main() -> None:
    global VECTORSTORE

    clear_screen()
    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY mangler i repo-roden .env")

    ensure_sqlite_db()
    VECTORSTORE = build_vectorstore()
    agent = build_agent()

    question = (
        "Forklar kort hvad metadata-filtre gør i RAG ud fra vores dokumenter, "
        "fortæl hvem der ejer service 'vector-search' i den interne database, "
        "giv en kort web-forklaring på hvad LangChain er, "
        "og hent også GitHub-statistik for repoet langchain-ai/langchain. "
        "Svar på dansk i 4 bullets."
    )

    print_section("Agent + Tools Workflow")
    print_label("WORKFLOW", CYAN)
    print(colorize("User", DIM))
    print(colorize("↓", DIM))
    print(colorize("Agent", DIM))
    print(colorize("↓", DIM))
    print(colorize("Tools", DIM))
    print(colorize("↓", DIM))
    print(colorize("LLM", DIM))
    print(colorize("↓", DIM))
    print(colorize("Answer", DIM))

    print_label("TOOLS", BLUE)
    print(colorize("- search_documents -> Qdrant", BLUE))
    print(colorize("- lookup_database -> SQLite", BLUE))
    print(colorize("- web_search -> Wikipedia API", BLUE))
    print(colorize("- github_repo_stats -> GitHub API", BLUE))
    prompt_to_continue("Tryk Enter for at køre agent-demoen...")

    print_section("User Question")
    print_label("SPØRGSMÅL", YELLOW)
    print(colorize(question, YELLOW))
    prompt_to_continue("Tryk Enter for at lade agenten bruge sine tools...")

    print_section("Agent Run")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final_text = _message_text(messages[-1]) if messages else str(result)

    print_label("SVAR", GREEN)
    print(colorize(final_text, GREEN))


if __name__ == "__main__":
    main()
