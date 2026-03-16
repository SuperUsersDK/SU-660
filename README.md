# SU-660 Repository

Dette repository bruges til kursusdemoer og øvelser om blandt andet:

- prompt engineering
- RAG
- chunking
- embeddings
- Qdrant
- n8n
- agent-lignende workflows

Indholdet er delt op i få hovedmapper:

- `demoer/`: færdige demoer, primært opdelt pr. modul
- `øvelser/`: opgaver, templates og løsninger
- `chunking/`: ældre eller mere isolerede chunking-eksempler
- `templates/`: små boilerplates til kursister
- `tools/`: fælles hjælpefunktioner til terminal-output mv.

## Hvad kursister har brug for

For at kunne køre demoer og lave øvelserne skal kursister typisk have:

- Python 3.10 eller nyere
- `venv`
- Docker Desktop eller Docker Engine + Compose
- adgang til internettet for OpenAI-baserede demoer
- en `.env` fil i repo-roden med mindst en gyldig `OPENAI_API_KEY`

Nogle demoer bruger også:

- Qdrant i Docker
- n8n i Docker
- lokale filer i `demoer/.../data`
- billedmodeller via `torch` og `transformers`

## Opret et virtual environment

Fra repo-roden.

### Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Når miljøet er aktivt, kan demoer køres med `python ...`.

## Miljøvariabler

Der forventes en `.env` fil i repo-roden:

- `.env`

Typiske værdier:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=hemmelighed
QDRANT_COLLECTION=rag_data
```

Minimum for de fleste OpenAI-demoer er:

- `OPENAI_API_KEY`

Minimum for Qdrant-demoer er typisk:

- `QDRANT_URL`
- eventuelt `QDRANT_API_KEY`

## Installer requirements

Alle Python-afhængigheder installeres fra:

- `requirements.txt`

Det inkluderer bl.a.:

- OpenAI SDK
- LangChain
- Qdrant client
- FastAPI
- `python-dotenv`
- `tiktoken`
- `torch`
- `torchvision`
- `transformers`

Hvis installation af de tunge pakker fejler, er det ofte fordi:

- Python-versionen er forkert
- man ikke står i det aktive venv
- `pip` er for gammel

## Start Docker-containerne

Repoet har en root `docker-compose.yml`:

- `docker-compose.yml`

Den starter:

- Qdrant
- n8n

### Start

```bash
docker compose up -d
```

eller i PowerShell:

```powershell
docker compose up -d
```

### Stop

```bash
docker compose down
```

### Stop og nulstil volumes

```bash
docker compose down -v
```

Services bliver typisk tilgængelige på:

- Qdrant: `http://localhost:6333`
- n8n: `http://localhost:5678`

Qdrant er sat op med API key i Compose-filen. Hvis demoer bruger auth, skal samme værdi findes i `.env` som `QDRANT_API_KEY`.

`n8n` har også repoet mountet ind som:

- `/files`

Det bruges af workflows, der læser lokale filer fra disk.

## Typisk arbejdsgang for kursister

1. Klon eller åbn repository.
2. Opret og aktivér et venv.
3. Installer `requirements.txt`.
4. Opret `.env` i repo-roden.
5. Start Docker-services med `docker compose up -d`.
6. Kør demoer med `python ...`.
7. Løs øvelserne i `øvelser/`.
7. Løs øvelserne i `øvelser/`.

## Hvor starter man?

Et godt udgangspunkt er typisk:

- `demoer/modul 1/`: prompt engineering, hallucination, tokenbudget
- `demoer/modul 2/`: RAG, chunking, metadata, rensning af data
- `demoer/modul 3/`: mere avancerede workflows, agent-demoer, n8n og billeder
- `øvelser/modul 2/`: kursistopgaver og løsninger

## Eksempler på kørsel

### Kør en Python-demo

```bash
python "demoer/modul 1/advanced_prompt_engineering_demo.py"
```

### Kør en RAG-demo

```bash
python "demoer/modul 2/rag_openai_demo.py"
```

### Kør et foto-ingest-demo

```bash
python "demoer/modul 3/photos/ingest_photos.py"
```

### Importér et n8n-workflow

Workflow-filer til n8n ligger typisk i:

- `demoer/modul 3/`

Importér dem i n8n via:

1. `Workflows`
2. `Import from file`

## Typiske problemer

### `OPENAI_API_KEY mangler`

Din `.env` mangler en gyldig OpenAI-nøgle, eller scriptet kan ikke finde repo-roden.

### `Qdrant 401 Unauthorized`

`QDRANT_API_KEY` matcher ikke den key, som Qdrant-containeren er startet med.

### `python-multipart` mangler

Nogle FastAPI-upload-demoer kræver `python-multipart`. Den ligger i `requirements.txt`, så geninstaller dependencies i dit aktive venv.

### `torch` eller `transformers` mangler

Foto-demoer kræver de tunge ML-pakker. Sørg for, at hele `requirements.txt` er installeret i dit venv.

### `Not existing vector name` i Qdrant

Det er normalt en mismatch mellem collection-schema i Qdrant og det, som workflowet eller scriptet forventer.

## Noter til undervisning

Repoet indeholder både:

- meget små, pædagogiske demoer
- mere realistiske demoer med OpenAI, Qdrant og n8n
- templates til øvelser
- løsninger til udvalgte opgaver

Det er derfor normalt, at nogle demoer kræver mere setup end andre.
