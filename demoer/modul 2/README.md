# Modul 2: Data, chunking og RAG

Demoerne viser vejen fra rå dokumenter til søgning: filtrering og rensning af data,
chunking, metadata og RAG. Alle kommandoer køres fra repository-roden med `.venv`
aktiveret (se [fælles opsætning](../../README.md)).

| Demo | Viser | Kræver |
|---|---|---|
| `chunking_lab/` (webapp) | Animeret sammenligning af chunking-strategier og deres effekt på søgning | Qdrant |
| `chunking_demo.py` | Seks chunking-strategier på samme Markdown-fil | Tokenizer-warmup |
| `data_rensning_demo.py` | En webside før og efter rensning for HTML-støj | – |
| `filter1.py`, `filter2.py`, `filter3.py` | Tre små filtre: find dokumenter, frasortér ubrugelig tekst, filtrér på metadata | – |
| `metadata_filter_demo.py` | Metadata-filtrering af et dokumentsæt | – |
| `rag_openai_demo.py` | Samme spørgsmål med og uden (simuleret) RAG | `OPENAI_API_KEY` |
| `big_demo/` | RAG med agent: ingest af filer til Qdrant og chat i terminalen | `OPENAI_API_KEY`, Qdrant |

Demoer med pauser venter på Enter mellem delene, så underviseren kan forklare.

## Chunking-laboratoriet (webapp)

Interaktiv og animeret demo: vælg en tekst og en chunking-strategi (fast antal tegn,
overlap, sætninger, afsnit, rekursiv, Markdown eller kildekode), og se chunks blive
klippet ud, embeddet og gemt i Qdrant. Derefter kan man søge og sammenligne alle
strategier på samme spørgsmål. Se [vejledningen](chunking_lab/README.md).

```bash
docker compose up -d qdrant
cd "demoer/modul 2/chunking_lab"
uvicorn app:app --port 8020
```

Åbn `http://127.0.0.1:8020`. Første indeksering downloader en lokal embedding-model
(ca. 470 MB); derefter virker appen uden internet.

## Chunking i terminalen

```bash
python "demoer/modul 2/chunking_demo.py"
```

Chunker `data/knowledge_base/chunking_workshop.md` med token-chunking, token-chunking
med overlap, sætninger, struktur (overskrifter), "semantisk" gruppering og LangChains
Markdown-splitter. Den semantiske del er en didaktisk tilnærmelse baseret på tekstlig
lighed, ikke rigtige embeddings. Kør `python tools/warmup_tokenizers.py` én gang med
internet først.

## Datarensning og filtre

```bash
python "demoer/modul 2/data_rensning_demo.py"
python "demoer/modul 2/filter1.py"
python "demoer/modul 2/filter2.py"
python "demoer/modul 2/filter3.py"
python "demoer/modul 2/metadata_filter_demo.py"
```

- `data_rensning_demo.py` fremhæver HTML-støj i en rå webside (`<script>`, `<style>`,
  header, footer, aside og tags) og viser den rensede tekst.
- `filter1.py` finder `.md`-, `.txt`- og `.pdf`-filer under `data/`, men kun i mappen
  `knowledge_base`. `dummy.jpg` sorteres fra, men `requirements.txt` slipper igennem,
  fordi den ender på `.txt`. God diskussion: filtyper alene er ikke et kvalitetsfilter.
- `filter2.py` afgør, om en tekst er brugbar (fx er `"OK"` og `"Home | About | Contact"` det ikke).
- `filter3.py` beholder kun dokumenter, hvis metadata matcher.
- `metadata_filter_demo.py` viser alle dokumenter og derefter kun dem med emnet RAG og
  sproget dansk.

Pointen er, at RAG-kvalitet starter før embeddings. Øvelsen i
[`øvelser/modul 2`](../../øvelser/modul%202/README.md) bygger videre på filtrene.

## RAG med og uden kontekst

```bash
python "demoer/modul 2/rag_openai_demo.py"
```

Viser forskellen mellem en prompt uden RAG og den samme prompt med lokal, simuleret
RAG. Retrieval sker med ordoverlap direkte i scriptet, uden embeddings og uden
vektor-database; kun svaret kommer fra en rigtig OpenAI-model. Pointen:

- uden RAG svarer modellen generelt og løst
- med RAG bliver svaret forankret i konkret kontekst

## RAG med agent og Qdrant

```bash
python "demoer/modul 2/big_demo/ingest.py"
python "demoer/modul 2/big_demo/chat_cli.py"
```

Se [big_demo/README.md](big_demo/README.md).

## Data

`data/knowledge_base/` indeholder `chunking_workshop.md` (bruges af `chunking_demo.py`
og chunking-øvelsen) samt `dummy.jpg` og `requirements.txt`, som ikke hører hjemme i en
vidensbase og viser filtrenes begrænsninger.
