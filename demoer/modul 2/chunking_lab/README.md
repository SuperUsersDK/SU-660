# Chunking-laboratoriet

En webapp, der viser hvordan forskellige chunking-strategier klipper den samme
tekst, og hvad det betyder for søgningen i en vektor-database.

Når man trykker **✂️ Chunk og indeksér**, animeres hele forløbet for hver chunk:
en saks klipper chunken ud af dokumentet, chunken flyver ind i embedding-modellen,
bliver til en vektor (vist som søjler) og flyver videre til sin plads i
vektor-databasen. En søgning animeres på samme måde: spørgsmålet bliver til en
vektor, lander som en stjerne i databasen, og der trækkes linjer til de nærmeste
chunks, som fremhæves i dokumentet.

## Start

Kræver Qdrant (`docker compose up -d qdrant`) og `QDRANT_API_KEY` i `.env`. Fra
repository-roden med `.venv` aktiveret:

```bash
cd "demoer/modul 2/chunking_lab"
uvicorn app:app --port 8020
```

Åbn `http://127.0.0.1:8020`. Første indeksering downloader embedding-modellen
(ca. 470 MB); derefter virker appen uden internet.

## Strategier

| Strategi | Klipper ved | Typisk styrke / svaghed |
|---|---|---|
| Fast antal tegn | hver N tegn | Simpel, men skærer midt i ord og sætninger |
| Tegn + overlap | hver N tegn, gentager slutningen af forrige chunk | Redder svar ved snittene, men giver flere og overlappende chunks |
| Sætninger | hele sætninger op til en maks. længde | Klipper aldrig i en sætning |
| Afsnit | tomme linjer | Holder afsnit samlet; overskrifter bliver ofte egne, indholdsløse chunks |
| Rekursiv (LangChain) | afsnit → linjer → ord → tegn | God standard; `RecursiveCharacterTextSplitter` |
| Markdown-overskrifter | `#`, `##`, `###` | Ét emne pr. chunk; virker kun på Markdown |
| Kildekode (Python) | funktioner og klasser (syntakstræ) | Hele funktioner; virker kun på Python |

Parametrene (størrelse, overlap) kan ændres med skydere, og opdelingen vises med det
samme. Overlap vises som stribede områder i teksten.

## Tekster og spørgsmål

| Tekst | Viser |
|---|---|
| Rejsepolitik (Markdown) | Overskrifter; fast antal tegn klipper "Business class … over 8 timer" over |
| Lagerstyring (Python) | Kun kildekode-strategien holder hele funktioner samlet |
| Kundeservice-FAQ | Afsnit = ét spørgsmål og svar pr. chunk |
| Øresundsbroen (prosa) | Lange afsnit uden struktur |

Hver tekst har eksempelspørgsmål med et kendt svar. Efter en søgning vises, om
svaret ligger i den øverste chunk (✅), længere nede (🟡), slet ikke i top 3 (❌),
eller om det er klippet over, så ingen chunk indeholder hele svaret (✂️).

**⚖️ Sammenlign alle strategier** søger det samme spørgsmål mod alle strategier og
viser resultatet i en tabel. Med standardindstillingerne finder fast antal tegn den
rigtige chunk øverst i 9 af 16 eksempelspørgsmål, overlap i 12 og sætninger, afsnit
og rekursiv i 13. Markdown- og kode-strategierne rammer 4 af 4 og 3 af 4 på deres
egne tekster.

Under **✏️ Egen tekst** kan man indsætte sin egen tekst (op til 20.000 tegn).

## Embeddings og database

- Standard er den lokale, flersprogede model `intfloat/multilingual-e5-small`
  (384 dimensioner, op til 512 tokens). Chunks gemmes som `passage: …` og spørgsmål
  som `query: …`, som modellen er trænet til. Chunks over 512 tokens klippes af
  modellen, og appen advarer om det.
- `CHUNK_LAB_EMBEDDINGS=openai` i `.env` bruger `OPENAI_EMBED_MODEL` i stedet.
- Alle kørsler ligger i Qdrant-collectionen `chunk_lab__<model>`. Hver kombination
  af tekst, strategi og parametre har sit eget `run_id` i payload, og søgningen
  filtrerer på det. En kørsel, der allerede er embeddet, genbruges.
- Kortet i databasen er vektorerne lagt ned i 2D med PCA. Spørgsmålet placeres med
  samme projektion.

## Filer

```text
chunking_lab/
  app.py        FastAPI: API til chunking, indeksering, søgning og sammenligning
  chunkers.py   strategierne; hver chunk husker sin præcise placering i teksten
  embedder.py   lokal e5-model eller OpenAI
  store.py      Qdrant: gem, genbrug og søg pr. kørsel
  texts/        eksempeltekster og spørgsmål (texts.json)
  static/       index.html, app.js (animationer) og style.css
```

`texts/lager.py` er en eksempeltekst til kode-chunking, ikke et program, der skal køres.
