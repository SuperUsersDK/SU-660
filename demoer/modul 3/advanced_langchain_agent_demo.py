from __future__ import annotations

import os
import sqlite3
import sys
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
DB_PATH = DATA_DIR / "advanced_agent_demo.db"
COLLECTION_NAME = "modul3_advanced_agent_demo"

VECTORSTORE: QdrantVectorStore | None = None


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
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name},
            )
        )
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
            CREATE TABLE IF NOT EXISTS services (
                service_name TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                sla_hours REAL NOT NULL,
                escalation_contact TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS escalation_policy (
                severity TEXT PRIMARY KEY,
                action TEXT NOT NULL
            )
            """
        )
        cursor.execute("DELETE FROM services")
        cursor.execute("DELETE FROM escalation_policy")
        cursor.executemany(
            "INSERT INTO services(service_name, owner, sla_hours, escalation_contact) VALUES (?, ?, ?, ?)",
            [
                ("vector-search", "Platform Team", 4.0, "platform-oncall@example.com"),
                ("chat-api", "AI Applications", 8.0, "ai-apps-oncall@example.com"),
                ("document-import", "Data Engineering", 24.0, "data-eng-oncall@example.com"),
            ],
        )
        cursor.executemany(
            "INSERT INTO escalation_policy(severity, action) VALUES (?, ?)",
            [
                ("low", "Log issue, monitor, and review in next daily standup."),
                ("medium", "Notify service owner and create an incident timeline."),
                ("high", "Escalate to on-call immediately and update stakeholders every 30 minutes."),
            ],
        )
        connection.commit()
    finally:
        connection.close()


@tool
def search_documents(query: str) -> str:
    """Search relevant knowledge-base documents in Qdrant."""
    global VECTORSTORE
    if VECTORSTORE is None:
        return "[docs] Vector store er ikke initialiseret."

    print_status("TOOL docs", f"Søger i Qdrant efter: {query}", MAGENTA)
    hits = VECTORSTORE.similarity_search(query, k=3)
    if not hits:
        return "[docs] Ingen relevante dokumenter fundet."

    parts = []
    for hit in hits:
        parts.append(f"[docs source={hit.metadata.get('source')}] {hit.page_content}")
    return "\n\n".join(parts)


@tool
def lookup_service(service_name: str) -> str:
    """Look up service owner, SLA, and escalation contact in SQLite."""
    print_status("TOOL db", f"Slår op på service: {service_name}", YELLOW)
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT service_name, owner, sla_hours, escalation_contact FROM services WHERE service_name = ?",
            (service_name,),
        )
        row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        return f"[db] Ingen service fundet med navn {service_name}."
    service, owner, sla_hours, escalation_contact = row
    return (
        f"[db] service={service}, owner={owner}, "
        f"sla_hours={sla_hours}, escalation_contact={escalation_contact}"
    )


@tool
def lookup_policy(severity: str) -> str:
    """Look up escalation policy text by severity."""
    print_status("TOOL policy", f"Slår policy op for severity: {severity}", BLUE)
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT action FROM escalation_policy WHERE severity = ?",
            (severity.lower(),),
        )
        row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        return f"[policy] Ingen policy fundet for severity={severity}."
    return f"[policy] severity={severity.lower()}, action={row[0]}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression such as '6.5 - 4'."""
    print_status("TOOL calc", f"Beregner: {expression}", GREEN)
    allowed = set("0123456789+-*/(). %")
    if any(char not in allowed for char in expression):
        return "[calc] Unsupported characters in expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:  # pragma: no cover - demo fallback
        return f"[calc] Error: {exc}"
    return f"[calc] {expression} = {result}"


def build_agent() -> Any:
    llm = ChatOpenAI(model=get_chat_model(), temperature=0)
    tools = [search_documents, lookup_service, lookup_policy, calculate]
    return create_agent(
        llm,
        tools=tools,
        system_prompt=(
            "Du er en avanceret AI operations-assistent med adgang til flere tools. "
            "Du skal selv vælge de relevante tools og kombinere resultaterne i et præcist svar. "
            "Når spørgsmålet handler om vores knowledge base, brug search_documents. "
            "Når det handler om service-ejerskab eller SLA, brug lookup_service. "
            "Når det handler om eskalering, brug lookup_policy. "
            "Når der skal regnes på tal eller SLA-overskridelse, brug calculate. "
            "Svar på dansk med disse sektioner: Oversigt, Evidens, Risici, Næste handlinger. "
            "Nævn tydeligt hvilke facts der kommer fra docs, db, policy og calc. "
            "Hvis noget mangler, så sig det tydeligt i stedet for at gætte."
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
        "Vi har et high-severity problem på service 'vector-search'. "
        "Outage har varet 6.5 timer. "
        "Fortæl hvem der ejer servicen og hvad SLA er, "
        "brug dokumenterne til kort at forklare hvorfor observability og metadata-filtre er relevante i denne situation, "
        "beregn hvor mange timer vi er over SLA, "
        "og afslut med den rigtige eskaleringshandling. "
        "Svar struktureret på dansk."
    )

    print_section("Avanceret LangChain Agent")
    print_label("WORKFLOW", CYAN)
    print(colorize("User", DIM))
    print(colorize("↓", DIM))
    print(colorize("Agent", DIM))
    print(colorize("↓", DIM))
    print(colorize("Tool loop", DIM))
    print(colorize("↓", DIM))
    print(colorize("LLM synthesis", DIM))
    print(colorize("↓", DIM))
    print(colorize("Structured answer", DIM))

    print_label("TOOLS", BLUE)
    print(colorize("- search_documents -> Qdrant knowledge base", BLUE))
    print(colorize("- lookup_service -> SQLite service database", BLUE))
    print(colorize("- lookup_policy -> SQLite policy table", BLUE))
    print(colorize("- calculate -> arithmetic tool", BLUE))

    print_label("POINTE", MAGENTA)
    print(colorize("Dette demo viser en agent, der selv vælger flere tools og samler dem i ét svar.", DIM))
    print(colorize("Det er mere avanceret end en fast chain, fordi tool-brugen bestemmes dynamisk af agenten.", DIM))
    prompt_to_continue("Tryk Enter for at køre agent-demoen...")

    print_section("User Question")
    print_label("SPØRGSMÅL", YELLOW)
    print(colorize(question, YELLOW))
    prompt_to_continue("Tryk Enter for at lade agenten planlægge og bruge tools...")

    print_section("Agent Run")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    final_text = _message_text(messages[-1]) if messages else str(result)

    print_label("SVAR", GREEN)
    print(colorize(final_text, GREEN))


if __name__ == "__main__":
    main()
