# Chunking til RAG og LLM-systemer (undervisningsmateriale)

Dette dokument er lavet til undervisning og er bevidst mere dybdegaende end en hurtig cheat sheet.
Det er designet til at kunne gennemgaas over ca. 2-3 timer med demos, diskussioner og oefelser.

## Laeringsmaal

Efter gennemgang boer deltagerne kunne:

- Forklare hvorfor chunking er centralt i RAG (retrieval quality, latency, cost, grounding)
- Vurdere tradeoffs mellem forskellige chunking-strategier
- Vaelge chunk-stoerrelse og overlap til forskellige dokumenttyper
- Forstaa hvorfor metadata, grænser og formatbevarelse er lige saa vigtigt som selve split-logikken
- Kende en praktisk evalueringsmetode til at tune chunking i stedet for at gaette

## Forslag til undervisningsforloeb (2-3 timer)

1. Intro og motivation (15-20 min)
2. Grundbegreber: chunk size, overlap, token vs tegn vs ord (20 min)
3. Chunking-strategier (45-60 min)
4. Praktiske demoer i `chunking/` (30-45 min)
5. Tuning og evaluering (20-30 min)
6. Faldgruber, anti-patterns og opsamling (15-20 min)

## 60-minutters plan: Indbygget chunking (LangChain splitters)

Denne plan er lavet til et fokuseret modul i et kursus, hvor deltagerne skal se praktisk vaerdi i built-in splitters.

### Modulmaal (60 min)

- Forstaa hvad "indbygget chunking" betyder i `langchain_text_splitters`
- Sammenligne 4 built-in strategier paa samme tekst og query
- Kunne vaelge en praktisk default til forskellige dokumenttyper
- Forstaa hvorfor top-1 score ikke alene afgør kvalitet

### Tidsplan (60 min)

1. Intro: hvorfor built-ins er relevante (5 min)
- Hurtig framing: chunking er stadig et designvalg, selv med built-ins

2. Overblik over splitters (10 min)
- `CharacterTextSplitter`
- `RecursiveCharacterTextSplitter`
- `TokenTextSplitter`
- `MarkdownHeaderTextSplitter`

3. Live demo (15 min)
- Koer `python3 chunking/08_builtin_langchain_splitters.py`
- Vis chunk count, preview og top hits for samme query

4. Tolkning af resultater (10 min)
- Top-1 vs top-k
- precision vs recall
- struktur vs enkelhed
- redundans/cost

5. Mini-oevelse (10 min)
- Kursisterne vaelger strategi + chunk size/overlap for:
  - policy/manual
  - FAQ/KB
  - stor PDF-rapport

6. Opsummering (10 min)
- Praktisk default
- Hvornår heading-aware/token-based giver ekstra vaerdi
- Hvad man evaluerer bagefter

### Underviser runbook (foreslaaet)

Koer:

```bash
python3 chunking/08_builtin_langchain_splitters.py
```

Gennemgaa i denne raekkefolge:

1. `CharacterTextSplitter`
- simpelt baseline, ofte semantisk grove graenser

2. `RecursiveCharacterTextSplitter`
- stærk default, bedre separator-bevarelse

3. `TokenTextSplitter`
- bedre alignment med LLM-budgetter

4. `MarkdownHeaderTextSplitter`
- strukturmetadata og bedre forklarlighed/citations-egnethed

### Vigtige undervisningspointer

- Built-in splitters er ofte nok til et godt første produktionssetup
- Recursive er et stærkt standardvalg
- Heading-aware er særligt godt til strukturerede dokumenter
- Token-baseret chunking er vigtigt, når cost/kontekstbudget betyder noget
- Sammenlign configs paa samme data/query - skift kun én ting ad gangen

## Hvad er chunking?

Chunking er processen med at opdele dokumenter i mindre stykker ("chunks"), som kan:

- embeddes (vektoriseres)
- lagres i vector database
- findes via retrieval
- sendes som kontekst til en LLM

Uden chunking faar man typisk enten:

- for store tekstblokke (daarlig retrieval-precision, dyr kontekst)
- for smaa tekstblokke (manglende kontekst, daarlig svar-kvalitet)

Chunking er derfor et kompromis mellem:

- relevans (retrieval precision)
- kontekstbevarelse (retrieval recall / svarbarhed)
- latency
- storage
- embedding-cost

## Hvorfor chunking betyder saa meget i RAG

Selv med en god model kan et RAG-system fejle, hvis retrieval henter:

- tekst uden den noedvendige kontekst
- for brede chunks med meget stoej
- chunks der bryder midt i definitioner/tabeller/afsnit

Typiske symptomer paa daarlig chunking:

- modellen citerer forkert side/sektion
- modellen svarer vagt trods relevant dokument findes
- retrieval finder "naesten rigtige" chunks
- top-k skal saettes kunstigt hoejt for at ramme rigtigt
- token-forbrug eksploderer uden kvalitetsloeft

## Grundbegreber

## Chunk-stoerrelse

Chunk-stoerrelse er hvor meget tekst et chunk indeholder.
Det kan maales i:

- tegn (chars)
- ord
- tokens (bedst til LLM-praksis)
- semantiske enheder (saetninger, afsnit, sektioner)

### Tommelfingerregler

- Chars er nemt og hurtigt, men upræcist ift. modelens tokenbudget
- Ord er bedre til didaktiske demoer, men stadig upræcist
- Tokens er bedst til produktion, men kraever tokenizer
- Strukturbaserede chunks (saetning/afsnit/heading) giver ofte bedre kvalitet end ren fixed-size

## Overlap

Overlap betyder at noget tekst gentages mellem nabo-chunks.

Formaal:

- undgaa at vigtig information falder paa graensen
- bevare lokal kontekst (f.eks. definition + forklaring)
- forbedre retrieval ved graense-tilfaelde

Ulemper:

- flere chunks (mere storage)
- flere embeddings (dyrere ingest)
- stoerre risiko for redundans i retrieval-resultater

### Praktisk tommelfingerregel

- Start med 10-20% overlap
- Skru op hvis dokumenter har taette sammenhaenge
- Skru ned hvis du faar mange dubletter i top-k

Eksempel:

- chunk size = 800 tokens
- overlap = 120 tokens
- overlap ratio = 15%

## Token vs ord vs tegn (vigtigt i undervisning)

En klassisk fejl er at designe chunking i tegn eller ord og glemme, at LLM-kontekst styres af tokens.

Konsekvens:

- Et "800 ord" chunk kan blive meget stoerre end forventet i tokens
- Kode, tabeller og specialtegn kan skubbe token-forbrug op
- Dansk/engelsk blandet indhold kan opfoere sig forskelligt

Anbefaling:

- Demo: brug ord/tegn for enkelhed
- Produktion: maal altid paa tokens (eller valider token-estimat)

## Metadata er en del af chunking (ikke kun en bonus)

Chunking handler ikke kun om split; det handler ogsaa om payload/metadata.

Minimum metadata pr. chunk (typisk):

- `source_file`
- `page` (for PDF hvis muligt)
- `chunk_id`
- `chunk_index`
- sektion/heading (hvis tilgaengelig)
- dokumenttype/kategori

Hvorfor det er vigtigt:

- citationer bliver bedre
- filtrering bliver mulig (`source_file`, kategori, dato)
- debugging bliver markant nemmere
- evaluering kan kobles tilbage til kilde

## Deterministiske IDs (idempotent ingest)

Hvis du ingest'er flere gange, vil du typisk undgaa dubletter.
Det kraever stabile IDs baseret paa indhold + kildeinfo.

Eksempel (god praksis):

- hash/UUID af: `source_file + page + chunk_index + text`

Fordele:

- re-run uden dubletter
- let at sammenligne foer/efter chunking-aendringer
- nemmere migrations/versionsstyring

## Chunking-strategier (overblik)

Nedenfor er de mest brugte strategier i praksis.

## 1. Fixed-size chunking (tegn/ord/tokens)

Tekst deles i faste vinduer, evt. med overlap.

Eksempel:

- 800 tokens per chunk
- 120 tokens overlap

Fordele:

- simpelt
- hurtigt
- stabilt outputformat
- let at benchmarke

Ulemper:

- bryder ofte midt i saetninger/afsnit
- semantisk "grimt"
- kan splitte tabeller/kodeblokke forkert

Hvornar bruge:

- baseline
- store tekstmasser uden tydelig struktur
- som fallback i recursive chunking

## 2. Sentence-based chunking

Tekst deles i saetninger og samles i grupper (f.eks. 2-5 saetninger per chunk).

Fordele:

- mere naturlige graenser
- bedre læsbarhed og grounding
- ofte bedre retrieval for QA

Ulemper:

- saetningssplitting kan vaere fejlbehæftet (forkortelser, punktummer i tabeller)
- chunk-stoerrelse bliver variabel
- lange saetninger kan blive problematiske

Hvornar bruge:

- policy docs, guides, artikler
- FAQ-lignende materiale

## 3. Paragraph-based chunking

Chunking paa afsnitsniveau (ofte med merge af korte afsnit).

Fordele:

- bevarer forfatterens struktur
- god til rapporter/notater
- bedre semantik end fixed-size

Ulemper:

- afsnit kan vaere meget ujævne i laengde
- OCR/PDF-ekstraktion kan give daarlige afsnitsbrud

Hvornar bruge:

- rapporter, artikler, manuals
- dokumenter med fornuftige blanklines

## 4. Recursive chunking (separator-hierarki)

Tekst splittes "fra groft til fint", fx:

1. sektion / heading
2. afsnit
3. saetning
4. mellemrum
5. fallback fixed split

Maal:

- bevare naturlige graenser saa langt som muligt
- garantere max chunk-stoerrelse

Fordele:

- meget god generel default
- robust paa forskellige dokumenter
- kombinerer struktur og stoerrelseskontrol

Ulemper:

- mere kompleks logik
- kan stadig fejle paa tabeller/kode/OCR-stoej

Hvornar bruge:

- standardvalg i mange RAG-systemer
- blandet dokumentportfolio

## 5. Heading-aware chunking (strukturbaseret)

Bruger dokumentets struktur (Markdown headings, HTML headings, DOCX styles, etc.).

Typisk flow:

- split per heading/section
- chunk inde i hver section med recursive/fixed strategy
- gem heading path som metadata

Fordele:

- meget stærke citationer
- bedre retrieval ved "find sektion X"
- god explainability

Ulemper:

- kraever strukturbevarelse i ingestion
- virker daarligt hvis PDF-parser mister headings

Hvornar bruge:

- handboeger
- dokumentation
- kontrakter/policies med nummererede sektioner

## 6. Semantic chunking (embeddings / similarity-baseret)

Chunkgraenser bestemmes ud fra semantiske skift snarere end faste laengder.

Typiske varianter:

- split mellem saetninger med lav similarity
- clustering af saetninger
- boundary detection med embeddings

Fordele:

- kan give mere meningsfulde chunks
- god til blandet tekst med emneskift

Ulemper:

- dyrere ved ingest
- mere kompleks tuning
- svaerere at debugge
- ikke altid bedre end god recursive chunking

Hvornar bruge:

- høj-kvalitets use cases
- komplekse dokumenter med utydelig layoutstruktur

## 7. Hybrid chunking

Kombinerer flere strategier.

Eksempler:

- Heading-aware -> recursive -> overlap
- Paragraph-first -> fallback fixed-size for lange afsnit
- Semantic boundaries + max token cap

I praksis er hybrid ofte det bedste produktionsvalg.

## Saerlige dokumenttyper og chunking-hensyn

## PDF'er

Problemer:

- OCR-stoej
- tabeller bliver flad tekst
- sidefod/sidehoved gentages
- kunstige linjeskift

Anbefalinger:

- fjern sidehoved/sidefod hvis muligt
- gem `page` metadata altid
- overvej side-niveau + subsektion chunking
- test retrieval paa rigtige sporgsmaal, ikke kun "det ser pænt ud"

## Kontrakter / juridiske dokumenter

Anbefalinger:

- bevar sektioner/undersektioner
- gem clause-id / section-number som metadata
- brug overlap moderat (for meget overlap giver dublet-citationer)
- overvej citation enforcement i prompt

## Kode / teknisk dokumentation

Anbefalinger:

- split paa funktion/klasse/fil-blokke
- bevar headings + code fences
- undgaa at bryde midt i kodeblokke
- gem sti/modulnavn som metadata

## Transskriptioner / dialog

Anbefalinger:

- split paa speaker-turns
- gem speaker + timestamp range
- chunk paa emneskift eller max token-budget

## Chunk-stoerrelse: hvordan vaelger man?

Der findes ikke en universel "rigtig" vaerdi.
Vaelg ud fra dokumenttype, sporgsmaalstype og retrieval-setup.

## Startvaerdier (praktiske defaults)

For generel RAG QA (tekst/PDF):

- chunk size: 400-1000 tokens
- overlap: 10-20%
- top-k: 4-8

For korte FAQ-lignende docs:

- chunk size: 150-400 tokens
- overlap: 5-15%

For lange rapporter/kontrakter:

- chunk size: 600-1200 tokens
- overlap: 10-20%
- plus heading metadata

For kode og dokumentation:

- chunk size: ofte struktur-baseret foerst
- fallback token-cap: 300-800 tokens
- overlap: lav-moderat (5-15%) med respekt for blokgraenser

## Overlap: hvordan vælger man?

Start med 10-15%.

Skru op hvis:

- retrieval mister info omkring chunkgraenser
- spørgsmål kræver lokal sammenhaeng (definition + betingelser)
- dokumentet har mange korte, tætte pointer

Skru ned hvis:

- top-k bliver fyldt med dubletter
- storage/embedding-cost er for hoej
- svar gentager samme kildepassage

## Viktig tradeoff: overlap og duplikation

Overlaps øger antal chunks.
En enkel intuition:

- Jo hoejere overlap, jo mere "duplikeret tekst" embeddes
- Det kan forbedre recall, men skade precision og cost

I undervisning er det nyttigt at vise samme tekst med:

- 0% overlap
- 10% overlap
- 25% overlap

...og sammenligne retrieval hits.

## Chunking og retrieval skal tunes sammen

Chunking kan ikke vurderes isoleret.
Det skal ses sammen med:

- embedding-model
- retriever-type (dense, hybrid, BM25 + dense)
- top-k
- reranker (hvis brugt)
- metadata-filtre
- prompt-format

Et chunking-setup der ser "daarligt" ud alene, kan fungere godt med reranking.
Omvendt kan pæne chunks give daarlig retrieval uden metadata/tuning.

## Evalueringsmetode (praktisk og undervisningsvenlig)

Brug en lille benchmark med 10-30 realistiske spørgsmål:

- spørgsmål
- forventet kilde/sektion (eller keywords)
- evt. forventet svarformat

Maal minimum:

- hit-rate@k (finder retrieval relevant chunk?)
- citation correctness (er kilder plausible/korrekte?)
- answer groundedness (svarer modellen kun ud fra kontekst?)
- latency
- token-forbrug / cost

Praktisk proces:

1. Vaelg baseline (fx recursive 800/120)
2. Varier 1 parameter ad gangen
3. Mål retrieval + svar
4. Vaelg data-baseret, ikke "det foeles bedre"

## Anti-patterns (klassiske fejl)

1. "Vi bruger bare 1000 tegn pa alt"
- Virker maaske paa nogle docs, men fejler paa mange formater

2. Intet overlap overhovedet
- Graenseproblemer bliver tydelige ved QA

3. For meget overlap (fx 50%)
- Dyrt, redundant, og retrieval bliver fyldt med dubletter

4. Ingen metadata
- Svaert at citere, debugge og filtrere

5. Ingen evaluering
- Man tuner paa mavefornemmelse

6. Chunking foer tekst-rensning
- Sidehoveder, OCR-stoej og linjeskift skader kvalitet

## Undervisningsoevelser (kan bruges i klassen)

### Oevelse 1: Sammenlign 2 strategier

Brug samme dokument og spørgsmål.
Sammenlign:

- fixed-size (med overlap)
- sentence-based

Diskuter:

- hvilke hits kommer i top-k?
- virker citationer bedre?
- hvad er tradeoffs?

### Oevelse 2: Tune overlap

Test samme chunk size med overlap:

- 0%
- 10%
- 20%

Diskuter:

- recall vs dubletter
- cost vs kvalitet

### Oevelse 3: Heading-aware vs ikke-heading-aware

Brug et struktureret dokument (policy/manual).
Se om spørgsmål som "Hvad står der i sektion X?" forbedres.

### Oevelse 4: Evaluering i mini-grid

Lav et lille grid:

- chunk size: 300 / 600 / 900
- overlap: 0 / 10% / 20%

Vurder hit-rate@k og kvalitet.

## Praktiske anbefalinger (god default til mange teams)

Hvis I starter fra scratch:

- Start med recursive chunking
- Bevar headings/metadata hvis muligt
- Brug 400-1000 tokens og 10-20% overlap
- Gem `source_file`, `page`, `chunk_id`
- Evaluer på rigtige spørgsmål
- Tilføj reranker senere hvis nødvendigt

## Tabel: Strategier (fordele/ulemper/anbefalet brug)

| Strategi | Kort beskrivelse | Fordele | Ulemper | Gode use cases | Typiske startvaerdier |
|---|---|---|---|---|---|
| Fixed-size (ord/tokens) | Faste vinduer, evt. overlap | Simpelt, hurtigt, stabilt | Bryder semantik, grimme graenser | Baseline, fallback | 400-1000 tokens, 10-20% overlap |
| Sentence-based | Grupper af saetninger | Naturlige graenser, god QA | Ujævn størrelse, saetningssplit kan fejle | Artikler, guides, FAQ | 2-5 saetninger/chunk, 0-1 saetnings overlap |
| Paragraph-based | Afsnitsniveau | Bevarer struktur, god læsbarhed | Ujævn størrelse, PDF/OCR kan bryde afsnit | Rapporter, notater | 1-3 afsnit/chunk, max token cap |
| Recursive | Separator-hierarki + fallback | Robust default, god balance | Mere kompleks logik | Generel RAG, blandede docs | Max 600-1000 tokens, overlap 10-20% |
| Heading-aware | Split efter sektioner/headings | Stærke citationer, god filtrering | Kræver strukturbevarelse | Manuals, policies, contracts | Heading split + recursive inde i sektion |
| Semantic | Embedding/similarity-baseret graenser | Kan give meningsfulde chunks | Dyrere, kompleks tuning/debug | High-quality use cases | Sentence units + similarity threshold + token cap |
| Hybrid | Kombinerer ovenstaaende | Ofte bedst i produktion | Mere engineering/tuning | Produktionssystemer | Heading-aware + recursive + metadata |

## Tabel: Parametre (stoerrelse, overlap, top-k, mm.)

| Parameter | Hvad styrer den? | For lavt | For hoejt | Praktisk startpunkt | Noter |
|---|---|---|---|---|---|
| Chunk size | Informationsmaengde per chunk | Mangler kontekst | Meget stoej og dyre prompts | 400-1000 tokens | Afhaenger af doc-type og spørgsmål |
| Overlap | Lokal kontekst mellem chunks | Graense-tab | Dubletter og ekstra cost | 10-20% | Start lavere ved korte docs |
| top-k | Antal retrieved chunks | Misser relevant info | For meget stoej | 4-8 | Tuning skal ske sammen med chunking |
| Metadata-felter | Filtrering/citation/debug | Daarrlig explainability | Mere payload (mindre issue) | `source_file`, `page`, `chunk_id` | Ogsaa section/category hvis muligt |
| Chunk unit | Tegn/ord/tokens/semantisk | Uhensigtsmæssig kontrol | Overengineering | Tokens/recursive | Brug chars/ord til undervisning og prototyper |
| Rensning før chunking | Tekstkvalitet | OCR-stoej og artefakter | Ekstra pipeline-kompleksitet | Basis-rensning | Stor effekt paa retrieval |

## Tabel: Hurtige valg efter dokumenttype

| Dokumenttype | Bedste startstrategi | Chunk size | Overlap | Metadata ekstra | Kommentar |
|---|---|---:|---:|---|---|
| PDF-rapporter | Recursive + page metadata | 600-1000 tokens | 10-20% | `page`, `source_file` | Rens sidehoveder hvis muligt |
| Kontrakter/policies | Heading-aware + recursive | 500-900 tokens | 10-15% | `section`, `page` | Citationer er kritiske |
| FAQ/KB artikler | Sentence/paragraph | 150-500 tokens | 5-15% | `article_id` | Kortere chunks virker ofte bedre |
| Kode/docs | Struktur-baseret + token cap | 300-800 tokens | 5-15% | `path`, `symbol` | Bryd ikke midt i kodeblokke |
| Transskriptioner | Turn/speaker-based | 300-700 tokens | 10-20% | `speaker`, `time_range` | Emneskift er vigtigere end tegn |

## Python-eksempler i dette repo

Se mappen `chunking/` for kørbare eksempler:

- `chunking/01_fixed_size_words.py`
- `chunking/02_fixed_vs_overlap.py`
- `chunking/03_sentence_and_paragraph.py`
- `chunking/04_recursive_separator.py`
- `chunking/05_heading_aware_markdown.py`
- `chunking/06_semanticish_grouping.py`
- `chunking/07_parameter_grid_eval.py`
- `chunking/08_builtin_langchain_splitters.py` (LangChain built-in splitters)

Eksemplerne bruger en simpel mock-retrievalscore (lexical similarity), saa de er lette at forstå i undervisning uden ekstern infrastruktur.

## Opsummering (kort version)

Hvis du kun husker 5 ting:

1. Chunking er en retrieval-beslutning, ikke kun en split-funktion
2. Start simpelt (recursive + metadata + overlap 10-20%)
3. Evaluer pa rigtige spørgsmål, ikke kun "pæne chunks"
4. Metadata er afgørende for citations og debugging
5. Chunking skal tunes sammen med top-k, retriever og prompt
