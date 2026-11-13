# Photos Demo

Dette demo ingest'er billeder fra `data/` til Qdrant med CLIP og giver en lille
webapp, hvor man kan søge, se vektorrummet og lege med embeddings. CLIP kører
lokalt; foto-demoet bruger ikke en OpenAI API-nøgle. Kun agent-demoet
`../photo_agent_demo.py` kalder OpenAI.

Bemærk:

- demoet bruger Qdrant på `localhost`
- billed-embeddings laves med CLIP via `transformers`
- standardmodellen er `openai/clip-vit-base-patch32`
- første kørsel downloader modelvægte

## Sider i webappen

| Side | Hvad den viser | Pointe til holdet |
|---|---|---|
| **Søg** (`/`) | Søg med et uploadet billede eller med tekst, eventuelt filtreret på et tag | Tekst og billeder ligger i samme vektorrum; payload-filtre kombineres med vektorsøgning |
| **Kort** (`/map`) | Alle billeder som thumbnails på et 2D-kort (t-SNE). Søgetekst lyser de bedste hits op | Et embedding er en position; lignende billeder samler sig i klynger |
| **Bland billede + tekst** (`/mix`) | `billede + vægt × tekst`, fx et vrag + "with a scuba diver" | Regning med embeddings og "the modality gap" |
| **Hvad ser modellen?** (`/model-view`) | Det 224×224 udsnit og de 7×7 felter, som CLIP faktisk ser | Kanterne skæres væk; opløsningen er lav |
| **Dubletter** (`/duplicates`) | Billedpar med cosine-lighed over en tærskel | Alle par sammenlignes med ét matrixprodukt; serieoptagelser findes |
| **Sammenlign modeller** (`/compare`) | Samme søgning mod to CLIP-modeller side om side | Valget af embedding-model er en designbeslutning |

### Tekstsøgning

CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren søges direkte
mod de gemte billedvektorer uden ny ingest. Tekst-til-billede scores ligger typisk
omkring 0,2–0,35, mens billede-til-billede ofte er over 0,8. Det er rækkefølgen,
der tæller. Engelsk virker bedst, men enkle danske sætninger virker ofte også.

### Zero-shot tags

`ingest_photos.py` tagger til sidst hvert billede ved at sammenligne det med en
liste engelske beskrivelser i `tagging.py` (fx `"a photo of a sunken shipwreck
underwater"` → `vrag`). Der trænes ingenting. Tag og sikkerhed gemmes i payload, så
søgningen kan filtrere på dem.

Ret labels eller prompts i `tagging.py`, og kør `python "demoer/modul 3/photos/tag_photos.py"`.
Det genbruger vektorerne i Qdrant og tager få sekunder. God øvelse: prompten
`"a photo of a scuba diver underwater"` fanger næsten alt under vandet, mens
`"a photo of a scuba diver wearing a mask and an air tank"` kun fanger dykkere.

### Bland billede og tekst

Billede-billede-ligheder spreder sig 4–7 gange mere end tekst-billede-ligheder,
fordi tekst og billeder ligger forskudt i CLIP-rummet ("the modality gap"). Uden
justering ville teksten næsten intet flytte. Begge led skaleres derfor efter deres
spredning over samlingen, så vægt 1 betyder, at billede og tekst trækker lige meget.
Negativ vægt trækker væk fra teksten.

### Sammenlign modeller

Hver model får sin egen collection, fordi vektorer fra forskellige modeller ikke kan
blandes. Standardmodellen bruger `modul3_photos_demo`; sammenligningsmodellen
(`COMPARE_IMAGE_EMBED_MODEL`, standard `openai/clip-vit-large-patch14`) bruger
`modul3_photos_demo__openai_clip_vit_large_patch14`. Den store model fylder ca. 1,7 GB
og er flere gange langsommere at ingeste.

## Struktur

```text
photos/
  app.py               webapp og sider
  ingest_photos.py     embed billeder og gem dem i Qdrant (+ zero-shot tags)
  tag_photos.py        tag igen uden at embedde forfra
  image_embeddings.py  CLIP: billede/tekst -> vektor, og modellens udsnit
  photo_store.py       fælles Qdrant-opslag
  tagging.py           labels og zero-shot klassifikation
  analysis.py          t-SNE-kort, dubletter og blanding af vektorer
  README.md
  data/                billederne (ignoreres af git)
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

Pillow, PyTorch, torchvision, Transformers, scikit-learn og python-multipart
installeres med kommandoen ovenfor. `QDRANT_API_KEY` i `.env` skal matche den
lokale service i `docker-compose.yml`.

## Kørsel

1. Læg mindst 5 `.jpg` eller `.jpeg` billeder i `demoer/modul 3/photos/data/`.
   Mappen er i `.gitignore`, så billederne skal lægges ind på hver maskine.
2. Kør ingest (embedder og tagger):

```bash
python "demoer/modul 3/photos/ingest_photos.py"
```

3. Valgfrit, til **Sammenlign modeller**:

```bash
python "demoer/modul 3/photos/ingest_photos.py" --model openai/clip-vit-large-patch14
```

4. Start webappen:

```bash
cd "demoer/modul 3/photos"
uvicorn app:app --reload --port 8010
```

5. Åbn `http://127.0.0.1:8010`.

Varm modellerne op før undervisningen ved at køre ingest én gang med internetadgang.
Første kørsel downloader CLIP-vægtene; efterfølgende kørsler bruger den lokale cache.
Hvis `IMAGE_EMBED_MODEL` ændres i `.env`, bruger webappen den collection, der hører
til modellen, så ingest skal køres for den nye model.

## Forslag til forløb i undervisningen

1. **Søg** med et billede og med tekst. Spørg: hvorfor er tekstscores så lave?
2. **Kort**: find klyngerne, og søg "a shipwreck" og "sunset" for at se dem lyse op.
3. **Tags og filtre**: filtrér en tekstsøgning på `dykker`. Ret en prompt i
   `tagging.py`, kør `tag_photos.py`, og se fordelingen ændre sig.
4. **Hvad ser modellen?** på et bredt billede: hvad sker der med motiver ude i siden?
5. **Bland** et vragbillede med "with a scuba diver" og derefter "at sunset".
6. **Dubletter** og **Sammenlign modeller** som afrunding.
7. Kør `photo_agent_demo.py` for at vise de samme vektorer brugt som agent-tool.
