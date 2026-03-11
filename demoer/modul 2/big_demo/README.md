# big_demo (RAG + Qdrant + CLI + API)

Praktisk demo-setup til RAG over dokumentation, markdown, kode og PDF-filer.

Demoen bruger:

- Qdrant som vector database
- OpenAI embeddings og chat-model
- LangChain til retrieval, agent og tools
- både CLI og FastAPI endpoint

## Understøttede filtyper

`app/ingest.py` kan nu indlæse:

- `.pdf`
- `.md`
- `.py`
- `.cs`

Standardmappen er `app/data/`, og scriptet behandler én kildefil ad gangen:

1. load én fil
2. chunk den fil
3. embed dens chunks
4. gem dem i Qdrant
5. fortsæt til næste fil

## Chunking-strategi

Chunking er tilpasset dokumentation og kode, ikke romaner.

- Markdown:
  bruger `MarkdownHeaderTextSplitter` og derefter recursive chunking
  med ca. `1200` tegn og `160` tegn overlap
- Python og C#:
  bruger syntax-aware chunking via `RecursiveCharacterTextSplitter.from_language(...)`
  med ca. `1000` tegn og `120` tegn overlap
- PDF:
  bruger prose-orienteret recursive chunking
  med ca. `1400` tegn og `180` tegn overlap

Det giver mere meningsfulde chunks for:

- overskriftsbaseret dokumentation
- kodefiler med funktioner, klasser og metoder
- længere prose i PDF-dokumenter

## Struktur

```text
big_demo/
  docker-compose.yml
  app/
    ingest.py
    chat_cli.py
    api.py
    rag_chain.py
    tools.py
    settings.py
    data/
  qdrant_storage/
```

## Konfiguration

`.env` skal ligge i repo-roden.

Eksempel:

```env
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_data
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

Standard collection-navn er nu `rag_data`.

## Kørsel

1. Start Qdrant

```bash
cd "demoer/modul 2/big_demo"
docker compose up -d qdrant
```

2. Aktivér dit venv og installer dependencies

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Læg filer i `app/data/`

4. Kør ingest

```bash
python app/ingest.py
```

5. Kør CLI

```bash
python app/chat_cli.py
```

6. Start API

```bash
uvicorn app.api:app --reload --port 8000
```

## Hvad ingest viser

`ingest.py` skriver farvet status undervejs, bl.a.:

- Qdrant-forbindelse
- collection-oprettelse eller brug af eksisterende collection
- hvilken fil der behandles
- load, chunking, embed og store
- hvilken chunking-metode der bruges
- hvilke chunks der indsættes eller springes over

## Typiske spørgsmål til demoen

- Hvordan chunkes markdown anderledes end kode?
- Hvorfor er syntax-aware chunking bedre til `.py` og `.cs`?
- Hvorfor er header-aware chunking nyttigt til `.md`?
- Hvad betyder overlap for retrieval-kvalitet og tokenforbrug?
- Hvordan ser man i logs, hvad der faktisk bliver gemt i Qdrant?
