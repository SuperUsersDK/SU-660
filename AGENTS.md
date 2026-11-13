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
  `demoer/modul 3/photos/data/`.
- Testene køres fra repository-roden med `python -m unittest discover -s tests`.
  Testfilerne virker ikke, hvis man starter dem direkte.
- Agent-demoerne i modul 3 skriver til de versionerede filer
  `demoer/modul 3/data/*.db`. Gendan dem med `git checkout`, hvis ændringerne
  ikke skal committes.
