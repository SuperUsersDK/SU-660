# Photos Demo

Dette demo ingest'er billeder fra `data/` til Qdrant og giver en lille webside,
hvor man kan uploade et billede og få de 5 mest lignende billeder tilbage.
Man kan også søge med tekst, fx "a shipwreck" eller "solnedgang over havet".
CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren søges direkte
mod de gemte billedvektorer uden ny ingest. Tekst-til-billede scores ligger
typisk omkring 0,2–0,35, mens billede-til-billede ofte er over 0,8. Det er
rækkefølgen, der tæller. Engelsk virker bedst, men enkle danske sætninger virker
ofte også.

Bemærk:

- demoet bruger Qdrant på `localhost`
- billed-embeddings laves med CLIP via `transformers`
- standardmodellen er `openai/clip-vit-base-patch32`
- første kørsel downloader modelvægte

## Struktur

```text
photos/
  app.py
  ingest_photos.py
  README.md
  data/
```

## Opsætning

Kør fra repository-roden med Python 3.12 (den testede version er 3.12.7):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints-classroom.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d qdrant
New-Item -ItemType Directory -Force "demoer/modul 3/photos/data"
```

På macOS/Linux aktiveres miljøet med `source .venv/bin/activate`, og billedmappen
oprettes med `mkdir -p "demoer/modul 3/photos/data"`.

Pillow, PyTorch, torchvision, Transformers og python-multipart installeres med
kommandoen ovenfor. Foto-demoet bruger ikke en OpenAI API-nøgle: CLIP kører lokalt.
`QDRANT_API_KEY` i `.env` skal matche den lokale service i `docker-compose.yml`.

## Kørsel

1. Læg mindst 5 `.jpg` eller `.jpeg` billeder i `demoer/modul 3/photos/data/`.
   Brug eksempelvis egne billeder af forskellige motiver. Webappen opretter selv
   mappen, hvis den mangler, men der skal være billeder i den før ingest.
2. Kør ingest:

```bash
python "demoer/modul 3/photos/ingest_photos.py"
```

3. Start webappen:

```bash
cd "demoer/modul 3/photos"
uvicorn app:app --reload --port 8010
```

4. Åbn:

```text
http://127.0.0.1:8010
```

Varm modellen op før undervisningen ved at køre ingest én gang med internetadgang.
Første kørsel downloader CLIP-vægtene; efterfølgende kørsler bruger den lokale cache.
Hvis `IMAGE_EMBED_MODEL` ændres i `.env`, skal webappen genstartes og billederne
ingestes igen, så søgning og indeks bruger samme model.
