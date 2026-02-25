# demo_full (RAG + Qdrant + tools + CLI + API)

Praktisk demo-setup til PDF-baseret RAG med:

- Qdrant (Docker)
- LangChain + OpenAI (retrieval + chat)
- Tool-calling agent (calculator, doc stats, context search)
- CLI og FastAPI endpoint

Bemærk:

- Agentdelen er implementeret til `langchain` 1.x (`create_agent`) for kompatibilitet med nyere miljøer.
- Kør app lokalt i venv for hurtigste demo. Docker API-container er optional.

## Struktur

```text
demo_full/
  docker-compose.yml
  .env
  requirements.txt
  app/
    ingest.py
    chat_cli.py
    api.py
    rag_chain.py
    tools.py
    settings.py
  data/
    pdfs/
  qdrant_storage/
```

## App Flow

```text
Documents (PDF/TXT) -> app/ingest.py -> Qdrant collection
                                    |
                                    v
                           app/rag_chain.py
                         (retrieval + agent + tools)
                            /                    \
                           v                      v
                  app/chat_cli.py            app/api.py
                   (terminal demo)         (POST /ask endpoint)
```

- `app/settings.py`: loader konfiguration fra `demo_full/.env` og repo-root `.env` til `SETTINGS`.
- `app/tools.py`: definerer agent-tools (`calculator`, `doc_stats`, `search_in_context`).
- `app/ingest.py`: læser PDF/TXT, chunker, laver embeddings og upserter til Qdrant med stabile IDs.
- `app/rag_chain.py`: bygger Qdrant vector store, retrieval-funktioner, RAG chain og tool-calling agent.
- `app/chat_cli.py`: interaktiv CLI; viser retrieval scores og kalder agenten.
- `app/api.py`: FastAPI endpoint der returnerer både `answer` og `retrieval_hits` (med scores).

## Kørsel (A-Z)

1. Start Qdrant

```bash
cd demo_full
docker compose up -d qdrant
```

2. Opsæt Python lokalt

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Sæt env (`demo_full/.env`)

```bash
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
```

4. Læg PDF'er i `data/pdfs/`

5. Ingest

```bash
python app/ingest.py
```

6. Chat (agent + tools)

```bash
python app/chat_cli.py
```

7. API (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
```

Vigtigt:

- Kør kommandoen fra `demo_full/` (ikke fra `demo_full/app/`), ellers virker `app.api:app` importen ikke.
- Swagger UI findes på `http://127.0.0.1:8000/docs`.

### API demo (`api.py`)

Start API'en:

```bash
cd demo_full
uvicorn app.api:app --reload --port 8000
```

Hvis du står i `demo_full/app`, brug i stedet:

```bash
uvicorn api:app --reload --port 8000
```

### `curl` eksempler

Kort spørgsmål med kilder:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Hvad siger dokumenterne om termination/opsigelse? Citer kilder og svar kort."}'
```

Vis pænt formatteret JSON (inkl. `retrieval_hits` scores):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Hvad handler dokumenterne om? Citer kilder."}' | jq
```

Calculator/tool-scenarie:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Ud fra dokumenterne: hvis vi har 23 enheder pr dag i 47 dage, hvad er totalen? Brug kilder hvis relevant."}'
```

Ekstra nyttige commands:

```bash
docker compose ps
docker compose logs -f qdrant
docker compose down
docker compose --profile api up -d rag_api   # starter optional API-container profile
```

## Demo-scenarier

- Ren RAG: "Hvad siger dokumenterne om X? Citer kilder."
- RAG + calculator: "Ud fra side 3 i doc A: hvis vi har 23 enheder pr dag og 47 dage, hvad er totalen?"
- RAG + search_in_context: "Find den præcise formulering omkring 'termination clause' og citer."
- Doc stats: "Hvilke kilder bliver brugt i dit svar?"
