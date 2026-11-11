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

Se [Modul 3-guiden](demoer/modul%203/README.md) for undervisningsflow og agent-tools,
[foto-demoets opsætning](demoer/modul%203/photos/README.md) for dependencies,
billeder og modelcache, og [RAG-demoets vejledning](demoer/modul%202/big_demo/README.md)
for ingestion og chat. n8n åbnes på `http://localhost:5678`; opret den lokale
brugerkonto, importér workflowet og vælg OpenAI- og Qdrant-credentials før brug.
Qdrant-adressen fra n8n-containeren er `http://qdrant:6333`.
