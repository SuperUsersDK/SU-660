# SU-660 Demo Repository

Samlet demo-repo til undervisning i:

- prompt engineering
- retrieval / RAG
- chunking-strategier
- embeddings-strategier
- tool-calling
- evaluering
- simple API/CLI-demoer

Repoet indeholder baade:

- **lokale mock-demoer** (ingen API-noegler noedvendige)
- **"real" demoer** med OpenAI/LangChain/Chroma
- **fuld RAG-demo** med Qdrant + CLI + API
- **undervisningsmateriale om chunking** (inkl. Python-eksempler og slides)
- **undervisningsmateriale om embeddings** (inkl. Python-eksempler og slides)

## Projektstruktur

```text
SU-660/
  demo/                 # lokale mock-demoer (ingen API-afhaengigheder)
  demo_real/            # OpenAI + LangChain + Chroma demoer
  demo_full/            # fuld RAG demo (Qdrant + ingest + CLI + API)
  chunking/             # chunking Python-eksempler (undervisning)
  embeddings/           # embeddings Python-eksempler (undervisning)
  chunking.md           # dyb undervisningsguide om chunking
  chunking_slides.md    # slide/manus-version om chunking
  embeddings.md         # dyb undervisningsguide om embeddings
  embeddings_slides.md  # slide/manus-version om embeddings
  slides/               # kursusslides
  requirements.txt      # root Python dependencies
```

## Hvad er hvad?

## `demo/` (mock / offline)

Små undervisningsvenlige Python-demoer, der kan køres uden API-noegler.

Eksempler:

- prompt templates
- context window / memory trimming
- retrieval-only
- simpel RAG
- eval loop
- A/B sammenligning

God til:

- klasseundervisning uden netværkskrav
- introduktion til begreber før "real" integrationsdemoer

Se: `demo/README.md`

## `demo_real/` (OpenAI + LangChain + Chroma)

Reelle integrationsdemoer med:

- OpenAI (chat + embeddings)
- LangChain
- Chroma

God til:

- live-demo af "rigtige" LLM-kald
- LangChain prompt/chains/tools/agent-eksempler

Se: `demo_real/README.md`

## `demo_full/` (Qdrant RAG app)

Mere komplet demo med:

- ingestion af PDF/TXT
- Qdrant vector store
- tool-calling agent
- CLI chat
- FastAPI endpoint (`/ask`)

God til:

- end-to-end RAG demo
- retrieval + tools + API i samme setup

Se: `demo_full/README.md`

## `chunking/` + `chunking.md`

Undervisningsmateriale og scripts om chunking:

- strategier (fixed, recursive, sentence, heading-aware, etc.)
- chunk size, overlap, metadata
- evaluering og tradeoffs
- sammenligning af built-in LangChain splitters

Se:

- `chunking.md`
- `chunking_slides.md`
- `chunking/README.md`

## `embeddings/` + `embeddings.md`

Undervisningsmateriale og scripts om embeddings i RAG:

- dense vs sparse retrieval
- similarity metrics (cosine/dot) og normalisering
- query-strategier (rewrite, multi-query)
- metadata-filtre og hybrid retrieval
- modelvalg, tradeoffs og evaluering

Se:

- `embeddings.md`
- `embeddings_slides.md`
- `embeddings/README.md`

## Krav / forudsætninger

Minimum:

- Python 3.10+ (3.11 anbefales)

For `demo_full`:

- Docker + Docker Compose (til Qdrant)

For `demo_real` / `demo_full`:

- OpenAI API-key i miljøvariabler eller `.env`

## Start Docker-services fra repo-roden

Root `docker-compose.yml` starter nu:

- Qdrant
- n8n

### Bash

```bash
docker compose up -d
docker compose ps
```

### PowerShell

```powershell
docker compose up -d
docker compose ps
```

Services vil typisk være tilgængelige på:

- Qdrant: `http://localhost:6333`
- n8n: `http://localhost:5678`

`n8n` får også repoet mountet read-only som `/files` inde i containeren. Det bruges af workflows, der skal læse lokale filer fra disk, fx:

- `/files/demoer/modul 3/data/*.md`

`n8n` er også sat op med `N8N_RESTRICT_FILE_ACCESS_TO=/files`, så `Read/Write Files from Disk` må læse fra den mountede repo-mappe.

For webhook- og chat-baserede workflows er `n8n` sat op med:

- `N8N_HOST=localhost`
- `WEBHOOK_URL=http://localhost:5678/`

Hvis du ændrer disse værdier i `docker-compose.yml`, så genstart containerne:

```powershell
docker compose down
docker compose up -d
```

For at stoppe dem igen:

### Bash

```bash
docker compose down
```

### PowerShell

```powershell
docker compose down
```

## Setup af Python virtual environment (venv)

Fra repo-roden:

### Bash (Linux/macOS/WSL)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### PowerShell (Windows)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Bemærk:

- Root `requirements.txt` indeholder pakker til flere demos (inkl. LangChain, Qdrant, Chroma, FastAPI, chunking splitters).
- Hvis du bruger system-Python i stedet for `.venv`, kan du ramme versionsforskelle mellem miljøer.

## Miljøvariabler (`.env`)

Repoet bruger `python-dotenv`, og flere scripts loader `.env`.

Typiske variable:

### Bash

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_data
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

### PowerShell

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:OPENAI_EMBED_MODEL="text-embedding-3-small"
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_COLLECTION="rag_data"
$env:EMBEDDING_MODEL="text-embedding-3-small"
$env:CHAT_MODEL="gpt-4o-mini"
```

Der findes også en root `.env.example`.

## Hurtig start (foerste demoer)

### 1) Offline mock-demoer (ingen API-key)

```bash
.venv/bin/python demo/01_prompt_template.py
.venv/bin/python demo/03_chunking_strategies.py
.venv/bin/python demo/05_rag_chat.py
.venv/bin/python demo/09_eval_loop.py
```

### 2) Chunking undervisningsdemoer

```bash
.venv/bin/python chunking/01_fixed_size_words.py
.venv/bin/python chunking/02_fixed_vs_overlap.py
.venv/bin/python chunking/04_recursive_separator.py
.venv/bin/python chunking/08_builtin_langchain_splitters.py
```

`08_builtin_langchain_splitters.py` kraever:

- `langchain_text_splitters`
- `tiktoken`

(begge er allerede i root `requirements.txt`)

### 3) Embeddings undervisningsdemoer

```bash
.venv/bin/python embeddings/01_lexical_vs_dense.py
.venv/bin/python embeddings/02_similarity_metrics.py
.venv/bin/python embeddings/06_hybrid_retrieval_rrf.py
.venv/bin/python embeddings/07_model_tradeoffs.py
```

Valgfri live API demo:

```bash
.venv/bin/python embeddings/08_openai_embeddings_demo.py
```

### 4) `demo_real` (OpenAI/LangChain/Chroma)

```bash
.venv/bin/python demo_real/01_prompt_template_openai.py
.venv/bin/python demo_real/02_pydantic_output_parser_langchain.py
.venv/bin/python demo_real/05_rag_chat_langchain_chroma.py
.venv/bin/python demo_real/11_agent_tool_calling_langchain.py
```

## `demo_full` (Qdrant + ingest + CLI + API)

### Start Qdrant (Docker)

#### Bash

```bash
cd demo_full
docker compose up -d qdrant
docker compose ps
```

#### PowerShell

```powershell
Set-Location demo_full
docker compose up -d qdrant
docker compose ps
```

### Ingest dokumenter

`demo_full/app/ingest.py` bruger flaget `--document-dir` (med `--pdf-dir` som alias).

Eksempler:

#### Bash

```bash
python3 app/ingest.py --document-dir ./data
```

eller hvis du bruger en undermappe:

```bash
python3 app/ingest.py --document-dir ./data/pdfs
```

#### PowerShell

```powershell
python .\app\ingest.py --document-dir .\data
python .\app\ingest.py --document-dir .\data\pdfs
```

### Start CLI chat

#### Bash

```bash
python3 app/chat_cli.py
```

#### PowerShell

```powershell
python .\app\chat_cli.py
```

CLI'en viser:

- retrieval hits med scores
- derefter agentens svar

### Start API

Start fra `demo_full/` (vigtigt):

#### Bash

```bash
uvicorn app.api:app --reload --port 8000
```

#### PowerShell

```powershell
uvicorn app.api:app --reload --port 8000
```

Swagger UI:

- `http://127.0.0.1:8000/docs`

Eksempel `curl`:

### Bash (`curl`)

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What do the documents say about termination? Cite sources briefly."}'
```

### PowerShell (`Invoke-RestMethod`)

```powershell
$body = @{
  question = "What do the documents say about termination? Cite sources briefly."
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/ask" `
  -ContentType "application/json" `
  -Body $body
```

API response indeholder typisk:

- `answer`
- `retrieval_hits` (rank, score, source_file, page, chunk_id)

## Fejlfinding (typiske problemer)

### `ModuleNotFoundError`

Oftest fordi:

- `.venv` ikke er aktiveret
- dependencies ikke er installeret
- script køres fra forkert mappe

Tjek:

```bash
which python3
python3 -V
pip show langchain langchain-text-splitters tiktoken
```

### `OPENAI_API_KEY mangler`

Sæt noeglen i shell eller `.env`:

### Bash

```bash
export OPENAI_API_KEY="..."
```

### PowerShell

```powershell
$env:OPENAI_API_KEY="..."
```

### `uvicorn app.api:app` fejler med `No module named 'app'`

Du staar sandsynligvis i `demo_full/app/`.

Løsning:

- koer fra `demo_full/`:
  - `uvicorn app.api:app --reload --port 8000`
- eller brug:
  - `uvicorn api:app --reload --port 8000` (fra `demo_full/app/`)

### Qdrant / Docker problemer

Tjek:

```bash
docker compose ps
docker compose logs -f qdrant
```

## Anbefalet undervisningsflow (kort)

1. Start med `demo/` (mock, begreber)
2. Brug `chunking.md` + `chunking/` scripts til chunking undervisning
3. Gaa videre til `demo_real/` for reelle modelkald
4. Slut med `demo_full/` som end-to-end RAG demo

## Relevante dokumenter

- `demo/README.md`
- `demo_real/README.md`
- `demo_full/README.md`
- `chunking.md`
- `chunking_slides.md`
- `slides/DemoSlides.md`
