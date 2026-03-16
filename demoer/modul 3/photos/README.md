# Photos Demo

Dette demo ingest'er billeder fra `data/` til Qdrant og giver en lille webside,
hvor man kan uploade et billede og få de 5 mest lignende billeder tilbage.

Bemærk:

- demoet bruger Qdrant på `localhost`
- billed-embeddings laves med CLIP via `transformers`
- standardmodellen er `openai/clip-vit-base-patch32`
- første kørsel downloader modelvægte

## Struktur

```text
photos/
  app.py
  ingest.py
  README.md
  data/
```

## Kørsel

1. Læg `.jpg` eller `.jpeg` billeder i `data/`
2. Kør ingest:

```bash
python "demoer/modul 3/photos/ingest.py"
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
