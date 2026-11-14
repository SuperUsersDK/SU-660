# SU-660 undervisningsdemoer

Demoerne er små eksempler til undervisning i prompts, chunking, RAG og agents.

## Testet miljø

- Windows x64, Python 3.12.7
- Qdrant 1.20.0 og n8n 2.47.4; Docker Compose bruger de konkrete testede image-digests.
- Python-pakkeversioner er fastholdt i `constraints-classroom.txt`.

Kør fra repository-roden i PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints-classroom.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d
```

På macOS/Linux bruges `python3.12 -m venv .venv` og
`source .venv/bin/activate`. Det registrerede klassemiljø er Windows;
platformsspecifikke pakker installeres kun, hvor de kræves af dependencies.

Sæt `OPENAI_API_KEY` i `.env` til chat-demoerne. `OPENAI_MODEL` er som standard
`gpt-6-luna`. Qdrant-nøglen i `.env` skal matche Compose-konfigurationen.
Bevar en eksisterende `.env`, når miljøet opdateres.

Installer med både requirements og constraints for at få det samme sæt versioner.
`requirements.txt` beskriver dependencies; constraints-filen fastholder den testede
versionskombination. Opdater først constraints efter at have afprøvet demoerne.

## Forbered tokenizere før undervisningen

Kør én gang med internetadgang efter installation:

```powershell
python tools/warmup_tokenizers.py
```

Kommandoen henter `o200k_base` og `cl100k_base` og kontrollerer encode/decode.
Token-baseret chunking bruger derefter samme cache i `.cache/tiktoken/`, også
uden internet. Bevar mappen mellem undervisningsgange. På hver kursistmaskine
køres warmup før en offline-session. En eksisterende `TIKTOKEN_CACHE_DIR` eller
`DATA_GYM_CACHE_DIR` respekteres; brug samme miljøvariabel ved warmup og demo.
Kommandoen kræver ingen API-nøgle og foretager ingen modelkald. Chat-demoer og
OpenAI-embeddings kræver fortsat netadgang, og CLIP har sin egen modelcache.

### Modeller, der downloades første gang

To demoer bruger lokale modeller, der hentes fra Hugging Face ved første kørsel og
derefter virker offline. Kør dem én gang med internet før undervisningen:

- **Chunking-laboratoriet** (modul 2): `intfloat/multilingual-e5-small`, ca. 470 MB,
  hentes ved første indeksering i webappen.
- **Foto-demoen** (modul 3): CLIP, ca. 600 MB (og 1,7 GB for den store model), hentes
  af `ingest_photos.py`. Billederne hentes separat med `fetch_sample_photos.py`, fordi
  billedmappen ikke er i git.

## Indhold

| Mappe | Indhold | Vejledning |
|---|---|---|
| `demoer/modul 1/` | Prompt engineering, hallucination og token-budget | [README](demoer/modul%201/README.md) |
| `demoer/modul 2/` | Datarensning, filtre, chunking, RAG og chunking-laboratoriet (webapp) | [README](demoer/modul%202/README.md) |
| `demoer/modul 2/chunking_lab/` | Animeret webapp: chunking-strategier og deres effekt på søgning | [README](demoer/modul%202/chunking_lab/README.md) |
| `demoer/modul 2/big_demo/` | RAG med agent, Qdrant og chat i terminalen | [README](demoer/modul%202/big_demo/README.md) |
| `demoer/modul 3/` | Agents med tools, n8n-workflow og foto-agent | [README](demoer/modul%203/README.md) |
| `demoer/modul 3/photos/` | Foto-webapp med CLIP: søgning, 2D/3D-kort, tags og dubletter | [README](demoer/modul%203/photos/README.md) |
| `chunking/` | Små chunking-eksempler 01–08 med worksheet | [README](chunking/README.md) |
| `øvelser/modul 2/` | Øvelser i dokumentfiltrering og chunking med løsninger | [README](øvelser/modul%202/README.md) |
| `templates/` | `boilerplate.py`: minimalt OpenAI-kald som udgangspunkt for egne scripts | – |
| `tools/` | Fælles terminal-output og tokenizer-warmup | – |
| `tests/` | Offline regressionstests | – |

## Startpunkter

```powershell
python "øvelser/modul 2/chunking_template.py"
python "chunking/07_parameter_grid_eval.py"
python "demoer/modul 2/rag_openai_demo.py"
python -m unittest discover -s tests -v
```

Chunking-skabelonen indeholder bevidst TODO-opgaver til kursisterne.
Parameter-grid-demoet bruger både RAG-noterne og policy-dokumentet, så alle
benchmark-spørgsmål har evidens i det indekserede materiale.

De to webapps startes fra deres egen mappe (Qdrant skal køre):

```powershell
# Chunking-laboratoriet: http://127.0.0.1:8020
cd "demoer/modul 2/chunking_lab"; uvicorn app:app --port 8020

# Foto-demoen: hent og indlæs billeder først, derefter http://127.0.0.1:8010
python "demoer/modul 3/photos/fetch_sample_photos.py"
python "demoer/modul 3/photos/ingest_photos.py"
cd "demoer/modul 3/photos"; uvicorn app:app --port 8010
```

n8n åbnes på `http://localhost:5678`; opret den lokale brugerkonto, importér
workflowet fra modul 3, og vælg OpenAI- og Qdrant-credentials før brug.
Qdrant-adressen fra n8n-containeren er `http://qdrant:6333`.
