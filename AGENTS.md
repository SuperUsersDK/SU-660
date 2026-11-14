# Instruktioner til AI-agenter

## RAG-data er ikke eksempler

`demoer/modul 2/big_demo/data/` indeholder kildedokumenter, som
`big_demo/ingest.py` indekserer i Qdrant. Filerne i mappen er ikke
kursuseksempler, heller ikke `.py`-filerne.

- Kør aldrig scripts fra mappen, når der bliver bedt om at køre eksemplerne.
- Hent aldrig miljøvariabler fra mappen, når `.env` eller `.env.example` opdateres.
- `ingest.py`, der læser mappen, er et almindeligt eksempel og må gerne køres.

## Kørsel af eksempler

- Brug Python 3.12 i `.venv`, og installér med
  `pip install -r requirements.txt -c constraints-classroom.txt`.
- OpenAI-demoerne kræver `OPENAI_API_KEY` i `.env`. Qdrant-demoerne kræver
  `docker compose up -d qdrant` og `QDRANT_API_KEY=hemmelighed`.
  Se `.env.example`.
- Mange demoer venter på Enter (`tools/terminal_ui.prompt_to_continue`). Når de
  køres uden terminal, skal der sendes linjeskift til stdin, ellers stopper de
  med `EOFError`.
- Kør `big_demo/ingest.py` før `big_demo/chat_cli.py`. Chat-CLI'en stoppes med `exit`.
- `demoer/modul 3/photos/ingest_photos.py` kræver mindst 5 `.jpg`-billeder i
  `demoer/modul 3/photos/data/`. Mappen er git-ignoreret og må aldrig committes.
  `photos/fetch_sample_photos.py` henter 20 billeder i hver af 10 kategorier fra
  Wikimedia Commons (kræver internet, ca. 10 min) og genoptager en afbrudt kørsel.
  Det rører kun billeder, det selv har hentet (`<kategori>_NNN_*.jpg`); andre billeder
  i mappen bevares, også med `--fresh`.
- Kør foto-ingest før `photos/tag_photos.py` og `photo_agent_demo.py`. Siden
  "Sammenlign modeller" kræver desuden `ingest_photos.py --model openai/clip-vit-large-patch14`.
- Foto-webappen er en server: start den med
  `cd "demoer/modul 3/photos" && uvicorn app:app --port 8010` i stedet for at køre `app.py`.
- Chunking-laboratoriet er også en server: start det med
  `cd "demoer/modul 2/chunking_lab" && uvicorn app:app --port 8020`.
  Filerne i `demoer/modul 2/chunking_lab/texts/` (også `lager.py`) er eksempeltekster
  til chunking, ikke eksempler, der skal køres.
- Testene køres fra repository-roden med `python -m unittest discover -s tests`.
  Testfilerne virker ikke, hvis man starter dem direkte.
- Agent-demoerne i modul 3 skriver til de versionerede filer
  `demoer/modul 3/data/*.db`. Gendan dem med `git checkout`, hvis ændringerne
  ikke skal committes.
