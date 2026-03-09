# Chunking (Slides / undervisningsversion)

Denne version er lavet til slides/undervisning og er kortere end `chunking.md`.
Brug den som manus + slideoutline.

## Slide 1: Titel

**Chunking til RAG**

- Hvorfor chunking betyder saa meget
- Strategier, stoerrelser, overlap og tuning
- Praktiske eksempler i Python

Talepunkt:

- "Chunking er ofte det vigtigste retrieval-designvalg efter valg af embeddings."

## Slide 2: Laeringsmaal

Efter sessionen kan deltagerne:

- forklare hvorfor chunking paavirker retrieval-kvalitet
- sammenligne chunking-strategier
- vaelge chunk size og overlap med rationale
- bruge metadata og IDs korrekt
- tune chunking med simple maalinger

## Slide 3: Hvad er chunking?

Chunking = opdeling af dokumenter i mindre enheder, der kan:

- embeddes
- lagres i vector DB
- retrieved
- sendes som kontekst til LLM

Hvis chunks er for store:

- mere stoej
- dyrere prompts
- daarligere precision

Hvis chunks er for smaa:

- mangler kontekst
- svar bliver svage
- retrieval finder "halv information"

## Slide 4: Hvorfor det er vigtigt i RAG

RAG-kvalitet afhænger ofte af:

1. chunking
2. retrieval
3. prompt
4. model

Typiske fejl ved daarlig chunking:

- forkert eller manglende citation
- relevante passager findes ikke i top-k
- modellen svarer vagt trods relevant dokument findes
- top-k skaleres op som symptombehandling

## Slide 5: Grundbegreber

Tre centrale parametre:

- **Chunk size** (hvor meget tekst per chunk)
- **Overlap** (gentagelse mellem nabo-chunks)
- **Chunk boundary strategy** (hvor vi splitter)

Supplerende vigtige valg:

- metadata
- top-k
- retriever/reranker
- dokumentrensning

## Slide 6: Chunk size (tegn, ord, tokens)

Maalemetoder:

- tegn (chars): simpelt, men groft
- ord: didaktisk og let at forstaa
- tokens: bedst til LLM-budgetter
- semantiske enheder: headings/saetninger/afsnit

Praktisk anbefaling:

- Undervisning/prototype: ord/tegn er fint
- Produktion: styr efter tokens (eller mindst token-estimat)

## Slide 7: Overlap (hvad og hvorfor)

Overlap = lidt tekst gentages i naeste chunk.

Fordele:

- beskytter mod graense-tab
- bedre recall omkring chunk boundaries
- mere robust retrieval

Ulemper:

- flere chunks
- flere embeddings (dyrere ingest)
- risiko for dubletter i top-k

Startpunkt:

- 10-20% overlap

## Slide 8: Visuelt eksempel (forklaring)

Eksempel:

- Chunk size = 800 tokens
- Overlap = 120 tokens

Chunk A:

- token 1-800

Chunk B:

- token 681-1480

Talepunkt:

- "Overlap er en forsikring mod daarlige graenser - men det er ikke gratis."

## Slide 9: Metadata er en del af chunking

Chunking uden metadata er kun halvvejs.

Minimum metadata:

- `source_file`
- `page` (PDF)
- `chunk_id`
- `chunk_index`

Ekstra metadata (ofte nyttigt):

- `section` / heading path
- dokumenttype
- kategori
- dato/version

Hvorfor:

- bedre citationer
- metadata-filtre
- debugging
- evaluering

## Slide 10: Deterministiske IDs (idempotent ingest)

Formaal:

- samme tekst -> samme ID
- re-run uden dubletter
- lettere sammenligning før/efter tuning

God praksis:

- ID baseret paa kilde + side + chunk-index + tekst
- brug UUID (fx UUIDv5) hvis vector DB kræver UUID

## Slide 11: Strategi 1 - Fixed-size chunking

Beskrivelse:

- split i faste vinduer (tegn/ord/tokens)

Fordele:

- simpelt
- hurtigt
- godt som baseline

Ulemper:

- bryder ofte midt i saetninger
- semantisk "grim" split

Godt til:

- baseline
- fallback
- simple demos

## Slide 12: Strategi 2 - Sentence-based chunking

Beskrivelse:

- split paa saetninger
- gruppering i 2-5 saetninger/chunk

Fordele:

- naturlige graenser
- bedre læsbarhed og grounding

Ulemper:

- ujævn chunk-størrelse
- sentence splitting kan fejle (forkortelser, PDF-stoej)

Godt til:

- guides, artikler, FAQ

## Slide 13: Strategi 3 - Paragraph-based chunking

Beskrivelse:

- split paa afsnit (og evt. merge korte afsnit)

Fordele:

- bevarer struktur
- god semantik

Ulemper:

- ujævn størrelse
- PDF/OCR kan ødelægge afsnit

Godt til:

- rapporter, notater, handboeger

## Slide 14: Strategi 4 - Recursive chunking (praktisk default)

Ide:

- split fra groft til fint:
  - afsnit
  - linjer
  - saetninger
  - ord
  - fallback

Fordele:

- god balance mellem struktur og størrelse
- robust pa tværs af dokumenttyper

Ulemper:

- lidt mere kompleks logik

Anbefaling:

- ofte bedste standardvalg til generel RAG

## Slide 15: Strategi 5 - Heading-aware chunking

Beskrivelse:

- brug headings/sektioner som primære graenser
- chunk inde i sektioner med recursive/fixed

Fordele:

- stærke citationer
- nemt at filtrere/diagnosticere
- godt til "hvad siger sektion X?"

Ulemper:

- kraever strukturbevarelse i ingestion

Godt til:

- policies
- kontrakter
- docs/manuals

## Slide 16: Strategi 6 - Semantic chunking

Beskrivelse:

- chunk-graenser baseres på semantiske skift (embedding similarity)

Fordele:

- kan følge emneskift bedre
- potentielt bedre retrieval i komplekse tekster

Ulemper:

- dyrere ingest
- mere tuning/kompleksitet
- svaerere at debugge

Talepunkt:

- "Semantic chunking er ikke automatisk bedre end god recursive chunking."

## Slide 17: Hybrid chunking (realistisk produktion)

Eksempel pipeline:

1. heading-aware split
2. recursive split inde i sektion
3. overlap
4. metadata enrichment

Hvorfor hybrid ofte vinder:

- kombinerer struktur + robust størrelse + retrieval-kvalitet

## Slide 18: Startvaerdier (praktisk)

Generel RAG QA (tekst/PDF):

- Chunk size: 400-1000 tokens
- Overlap: 10-20%
- top-k: 4-8

Korte FAQ/docs:

- Chunk size: 150-400 tokens
- Overlap: 5-15%

Lange rapporter/kontrakter:

- Chunk size: 600-1200 tokens
- Overlap: 10-20%
- heading/page metadata

## Slide 19: Saerlige dokumenttyper

PDF:

- gem altid `page`
- ryd op i headers/footers hvis muligt
- forvent OCR/layout-stoej

Kontrakter:

- bevar sektioner/clauses
- citations er kritiske

Kode:

- split paa funktion/klasse/blokke
- undgaa at bryde midt i code fences

Transskriptioner:

- split paa speaker turns + timestamps

## Slide 20: Chunking skal tunes sammen med retrieval

Chunking kan ikke vurderes isoleret.

Tuning hænger sammen med:

- embedding model
- retriever type
- top-k
- reranker
- metadata-filtre
- prompting

Talepunkt:

- "Et chunking-setup er kun godt i den samlede pipeline."

## Slide 21: Evalueringsmetode (enkelt og effektiv)

Lav et benchmark med 10-30 realistiske spørgsmål.

Maal fx:

- hit-rate@k
- keyword coverage i retrieved chunks
- citation correctness
- groundedness
- latency/cost

Proces:

1. vælg baseline
2. ændr 1 parameter
3. mål igen
4. beslut med data

## Slide 22: Anti-patterns

Undgaa:

- "1000 chars paa alt"
- 0 overlap uden test
- 50% overlap uden grund
- ingen metadata
- ingen evaluering
- chunking før basale tekst-rensninger

## Slide 23: Demo-flow (praktisk)

Brug scripts i `chunking/`:

1. `01_fixed_size_words.py`
2. `02_fixed_vs_overlap.py`
3. `03_sentence_and_paragraph.py`
4. `04_recursive_separator.py`
5. `05_heading_aware_markdown.py`
6. `06_semanticish_grouping.py`
7. `07_parameter_grid_eval.py`

Forslag:

- vis chunks
- vis top-k hits
- diskuter tradeoffs
- vis parameter-grid til sidst

## Slide 24: 60-min plan (built-in chunking modul)

Maal:

- sammenligne built-in splitters i `langchain_text_splitters`
- vise hvorfor built-ins stadig kraever tuning og evaluering

60-min forslag:

1. Intro (5 min)
2. Splitter-overblik (10 min)
3. Live demo (15 min)
4. Tolkning af output (10 min)
5. Mini-oevelse (10 min)
6. Opsummering (10 min)

## Slide 25: Built-in splitters vi viser (LangChain)

- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter`
- `TokenTextSplitter`
- `MarkdownHeaderTextSplitter`

Talepunkt:

- "Samme tekst, samme query, samme retrieval-logik - kun chunking aendres."

## Slide 26: Live demo runbook (60-min modul)

Koer:

```bash
python3 chunking/08_builtin_langchain_splitters.py
```

Vis for hver strategi:

- antal chunks
- 2-6 chunk previews
- top hits for samme query
- hvordan resultaterne skal tolkes (top-1 vs top-k, struktur, redundans)

Underviser-tip:

- Del worksheet ud: `chunking/08_builtin_langchain_splitters_worksheet.md`
- Lad kursisterne udfylde hypoteser foer demoen og sammenligne bagefter
## Slide 27: Strategi-tabel (fordele/ulemper)

| Strategi | Fordele | Ulemper | Godt til |
|---|---|---|---|
| Fixed-size | Simpelt, hurtigt, baseline | Bryder semantik | Baseline/fallback |
| Sentence-based | Naturlige graenser | Ujævn størrelse | Artikler/FAQ |
| Paragraph-based | Bevarer struktur | PDF/OCR kan ødelægge afsnit | Rapporter/notater |
| Recursive | Robust default | Mere kompleks | Generel RAG |
| Heading-aware | Stærke citationer | Kræver struktur | Policies/manuals |
| Semantic | Kan følge emneskift | Dyrt/komplekst | Høj-kvalitets use cases |
| Hybrid | Ofte bedst samlet | Mere engineering | Produktion |

## Slide 28: Parameter-tabel (værdier og tradeoffs)

| Parameter | Startpunkt | Hvis for lav | Hvis for hoej |
|---|---:|---|---|
| Chunk size | 400-1000 tokens | Mangel på kontekst | Stoej/dyre prompts |
| Overlap | 10-20% | Graense-tab | Dubletter/cost |
| top-k | 4-8 | Misser relevant info | Stoej i kontekst |
| Metadata | source/page/chunk_id | Daarrlige citationer | (næsten ingen reel ulempe) |

## Slide 29: Opsummering (5 takeaways)

1. Chunking er retrieval-design, ikke kun tekstsplit
2. Recursive + metadata + overlap er et stærkt startpunkt
3. Tokens > ord/tegn i produktion
4. Metadata og stabile IDs gør drift/debugging muligt
5. Tuning skal være evalueringsdrevet

## Slide 30: Q&A / Oevelser

Forslag til diskussion:

- Hvilken strategi ville I vælge til kontrakter?
- Hvornår er overlap ikke det rigtige svar?
- Hvordan ville I evaluere chunking i jeres egne docs?
