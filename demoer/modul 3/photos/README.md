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
| **Kort** (`/map`) | Alle billeder som thumbnails i 2D eller 3D (t-SNE eller PCA). 3D kan roteres og zoomes. Søgetekst lyser de bedste hits op | Et embedding er en position; lignende billeder samler sig i klynger |
| **Hvad ser modellen?** (`/model-view`) | Det 224×224 udsnit og de 7×7 felter, som CLIP faktisk ser | Kanterne skæres væk; opløsningen er lav |
| **Dubletter** (`/duplicates`) | Billedpar med cosine-lighed over en tærskel | Alle par sammenlignes med ét matrixprodukt; serieoptagelser findes |
| **Sammenlign modeller** (`/compare`) | Samme søgning mod to CLIP-modeller side om side | Valget af embedding-model er en designbeslutning |

### Tekstsøgning

CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren søges direkte
mod de gemte billedvektorer uden ny ingest. Tekst-til-billede scores ligger typisk
omkring 0,2–0,35, mens billede-til-billede ofte er over 0,8. Det er rækkefølgen,
der tæller. Engelsk virker bedst, men enkle danske sætninger virker ofte også.

### Kort i 2D og 3D

Knapperne over kortet skifter mellem **2D t-SNE**, **3D t-SNE** og **3D PCA**. I 3D
trækker man for at rotere, scroller for at zoome og højreklikker eller shift-trækker
for at flytte. Kortet roterer langsomt af sig selv, indtil man rører det.

- **t-SNE** bevarer naboskaber. Klynger betyder noget, men akser og afstande mellem
  klynger gør ikke. En søgetekst placeres ved sine bedste hits, fordi t-SNE ikke kan
  placere nye punkter. Perplexity (hvor mange naboer hvert punkt tager hensyn til)
  sættes til antal billeder / 20, mellem 5 og 30. Den skal være mindre end en typisk
  klynge: med 20 billeder pr. kategori giver 10 tydeligt skarpere klynger end 30.
- **PCA** er en lineær projektion på de tre retninger med mest variation. Statuslinjen
  viser, hvor stor en del af variationen akserne fanger (ca. 26% for eksempelbillederne).
  En søgetekst placeres præcist, og gule streger viser afstanden til hits. Den
  afstand er "the modality gap": tekst og billeder ligger forskudt i CLIP-rummet.

Three.js ligger i `static/three/` (MIT-licens), så 3D-kortet virker uden internet.

### Zero-shot tags

`ingest_photos.py` tagger til sidst hvert billede ved at sammenligne det med en
liste engelske beskrivelser (fx `"a photo of a mushroom"` → `svamp`). Der trænes
ingenting. Tag og sikkerhed gemmes i payload, så søgningen kan filtrere på dem.
Dropdowns i søgningen viser de tags, der findes i Qdrant.

`PHOTO_LABEL_SET` i `.env` vælger label-sæt:

- `eksempler` (standard): de 10 kategorier fra `sample_categories.py`, som passer til
  `fetch_sample_photos.py`.
- `dykning`: vrag, fisk, koralrev, dykker, smådyr, båd, mennesker og solnedgang til en
  samling dykkerbilleder.

Ret labels eller prompts, og kør `python "demoer/modul 3/photos/tag_photos.py"`. Det
genbruger vektorerne i Qdrant og tager få sekunder. God øvelse: zero-shot kan kun
vælge mellem de labels, den får, og siger aldrig "ved ikke". Fjern fx `svamp` fra
listen, tag igen, og se, hvor svampene havner.

### Sammenlign modeller

Hver model får sin egen collection, fordi vektorer fra forskellige modeller ikke kan
blandes. Standardmodellen bruger `modul3_photos_demo`; sammenligningsmodellen
(`COMPARE_IMAGE_EMBED_MODEL`, standard `openai/clip-vit-large-patch14`) bruger
`modul3_photos_demo__openai_clip_vit_large_patch14`. Den store model fylder ca. 1,7 GB
og er flere gange langsommere at ingeste.

## Struktur

```text
photos/
  app.py                  webapp og sider
  fetch_sample_photos.py  hent eksempelbilleder (10 kategorier) fra Wikimedia Commons
  sample_categories.py    kategorierne: tag, CLIP-prompt, Commons-kategori og farve
  ingest_photos.py        embed billeder og gem dem i Qdrant (+ zero-shot tags)
  tag_photos.py           tag igen uden at embedde forfra
  image_embeddings.py     CLIP: billede/tekst -> vektor, og modellens udsnit
  photo_store.py          fælles Qdrant-opslag
  tagging.py              labels og zero-shot klassifikation
  analysis.py             t-SNE/PCA-kort og dubletter
  static/three/           Three.js til 3D-kortet (lokalt, virker offline)
  README.md
  data/                   billederne (ignoreres af git)
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

1. Hent eksempelbilleder, eller læg mindst 5 egne `.jpg`/`.jpeg` billeder i
   `demoer/modul 3/photos/data/`. Mappen er i `.gitignore`, så billederne skal
   hentes eller lægges ind på hver maskine.

```bash
python "demoer/modul 3/photos/fetch_sample_photos.py"
```

   Scriptet henter 20 billeder i hver af 10 visuelt forskellige kategorier (bil, fly,
   skib, fugl, insekt, blomst, svamp, kirke, mad og bjerg) fra Wikimedias "Quality
   images". Et CLIP-filter beholder kun billeder, som CLIP sikkert placerer i deres
   egen kategori, så klyngerne bliver tydelige i 2D og 3D. Det tager ca. 10 minutter,
   fordi Wikimedia begrænser tempoet. `--per-category 50` henter flere. En afbrudt
   kørsel genoptages ved at køre scriptet igen; `--fresh` starter forfra. Billederne
   er CC BY, CC BY-SA eller CC0; fotograf og licens står på hvert billedes Commons-side.

2. Kør ingest (embedder og tagger):

```bash
python "demoer/modul 3/photos/ingest_photos.py"
```

   `--incremental` beholder collectionen, tilføjer kun nye billeder og fjerner
   billeder, der er slettet fra mappen. Det kan køres, mens hentningen stadig er i gang.

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

Eksemplerne passer til billederne fra `fetch_sample_photos.py`.

1. **Søg** med tekst, fx "a red sports car", "a butterfly" og "snowy mountains", og
   med et uploadet billede. Spørg: hvorfor er tekstscores (ca. 0,3) så meget lavere
   end billedscores?
2. **Kort** i 2D: de 10 kategorier ligger som hver sin klynge, og beslægtede klynger
   ligger tæt (bil, fly og skib; blomst og insekt). Søg "a mushroom in the forest",
   og se svampene lyse op. Skift til 3D t-SNE, og rotér.
3. **3D PCA**: første akse skiller natur fra menneskeskabte ting. Søg igen, og vis
   afstanden mellem søgeteksten og billederne (the modality gap). Akserne viser kun ca.
   26% af variationen.
4. **Tags og filtre**: "an owl" giver mest fugle; filtrér på `fugl`. Fjern `svamp` fra
   labels i `sample_categories.py`, kør `tag_photos.py`, og se, hvor svampene havner,
   når modellen ikke kan sige "ved ikke".
5. **Hvad ser modellen?** på et bredt landskabsbillede: hvad sker der ude i siderne?
6. **Dubletter** og **Sammenlign modeller** som afrunding.
7. Kør `photo_agent_demo.py` for at vise de samme vektorer brugt som agent-tool.
