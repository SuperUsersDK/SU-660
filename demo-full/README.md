# demo-full (RAG + Qdrant + tools + CLI + API)

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
demo-full/
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

## Kørsel (A-Z)

1. Start Qdrant

```bash
cd demo-full
docker compose up -d qdrant
# eller:
make up
```

2. Opsæt Python lokalt

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# eller:
make install
```

3. Sæt env (`demo-full/.env`)

```bash
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
```

4. Læg PDF'er i `data/pdfs/`

5. Ingest

```bash
python app/ingest.py
# eller:
make ingest
```

6. Chat (agent + tools)

```bash
python app/chat_cli.py
# eller:
make chat
```

7. API (FastAPI)

```bash
uvicorn app.api:app --reload --port 8000
# eller:
make api
```

Ekstra nyttige commands:

```bash
make ps
make logs
make down
make api-docker   # starter optional API-container profile
```

## Demo-scenarier

- Ren RAG: "Hvad siger dokumenterne om X? Citer kilder."
- RAG + calculator: "Ud fra side 3 i doc A: hvis vi har 23 enheder pr dag og 47 dage, hvad er totalen?"
- RAG + search_in_context: "Find den præcise formulering omkring 'termination clause' og citer."
- Doc stats: "Hvilke kilder bliver brugt i dit svar?"
