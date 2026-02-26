# Embeddings til RAG og LLM-systemer (undervisningsmateriale)

Dette dokument er lavet til et 3-dages kursus i LLM + RAG og er designet som et undervisningsmodul,
der kan bruges baade som selvstaendig session og som bro mellem chunking og retrieval/reranking.

Anbefalet placering i et 3-dages forloeb:

- Dag 1: grundbegreber, prompts, simple retrieval-demoer
- Dag 2: chunking + embeddings (hvordan data bliver "soegbar")
- Dag 3: RAG pipelines, evaluering, tools, API og drift

## Laeringsmaal

Efter modulet boer deltagerne kunne:

- forklare hvad embeddings er, og hvorfor de bruges i retrieval
- forklare forskellen paa lexical (sparse) og dense retrieval
- vaelge similarity metric og forstaa normaliseringens betydning
- beskrive tradeoffs i modelvalg (quality, latency, cost, dimension)
- kombinere embeddings med metadata-filtre og evt. hybrid retrieval
- evaluere embedding-strategier paa egne queries i stedet for at gaette

## Hvad er embeddings?

Et embedding er en numerisk vektor, der repræsenterer tekst saaledes, at semantisk lignende tekster
typisk ligger taettere paa hinanden i vektor-rummet.

I RAG bruges embeddings til at:

- embedde dokument-chunks ved ingest
- embedde brugerens query ved retrieval
- sammenligne query-vektor med dokument-vektorer (fx cosine similarity)

Kort sagt:

- chunking bestemmer *hvad* der kan findes
- embeddings bestemmer *hvordan lignende tekst maales*
- retrieval pipeline bestemmer *hvordan hits bliver valgt og kombineret*

## Embeddings er ikke magi (vigtige kursuspointer)

Embeddings loeser ikke alene:

- daarlig chunking
- manglende metadata
- for brede/noisy dokumenter
- governance/permissions
- hallucinationer i genereringsleddet

Embeddings forbedrer retrieval, men kun som del af en samlet pipeline.

## Forslag til undervisningsforloeb (2-3 timer)

1. Intro: hvad embeddings er og hvorfor RAG bruger dem (15 min)
2. Dense vs sparse retrieval (20 min)
3. Similarity metrics og normalisering (20 min)
4. Chunk-level vs document-level embeddings (15 min)
5. Metadata-filtrering og hybrid retrieval (20 min)
6. Modelvalg, cost/latency/quality tradeoffs (20 min)
7. Praktiske demoer i `embeddings/` (30-45 min)
8. Faldgruber og evalueringsstrategi (15 min)

## 60-minutters modulplan (kompakt)

Hvis du kun har 60 min:

1. `01_lexical_vs_dense.py` (10 min)
2. `02_similarity_metrics.py` (10 min)
3. `04_chunk_vs_document_embeddings.py` (10 min)
4. `05_metadata_filtering.py` + `06_hybrid_retrieval_rrf.py` (15 min)
5. `07_model_tradeoffs.py` + opsamling (15 min)

Hvis internet/API er tilgaengeligt:

- afslut med `08_openai_embeddings_demo.py` (5-10 min ekstra)

## Dense vs sparse retrieval

### Sparse / lexical (BM25/keyword-lignende)

Styrker:

- meget god ved exact terms, IDs, produktkoder, juridiske ordlydsmatch
- forklarlig
- ofte billig og hurtig

Svagheder:

- rammer daarligt ved synonymi/parafraser
- sprogvariation (fx dansk/engelsk blanding) kan reducere recall

### Dense / embedding retrieval

Styrker:

- bedre semantisk matching
- robust mod parafraser
- ofte stærk til naturligt sprog-spoergsmaal

Svagheder:

- kan miste exact-match signal (fx versionsnumre, SKU, paragrafnumre)
- kræver modelvalg, embedding-cost og vektorinfrastruktur
- kan være svaerere at debugge uden gode metadata og evalueringscases

## Similarity metrics og normalisering

Typiske metrics:

- cosine similarity (mest almindelig i undervisning og praksis)
- dot product
- euclidean distance (mindre brugt som direkte retrieval-score i RAG)

Vigtig pointe:

- Dot product er foelsom over for vektorlaengde
- Cosine maaler vinkel/similaritet i retning
- Mange pipelines L2-normaliserer embeddings, saa dot product ~= cosine

Undervisningsfejl der ofte optræder:

- skifte metric mellem ingest/retrieval uden at vide det
- antage at alle vector stores bruger samme default
- sammenligne scores paa tværs af modeller som om de er direkte kompatible

## Chunk-level vs document-level embeddings

I RAG vil man normalt embedde *chunks*, ikke hele dokumenter.

Hvorfor:

- dokument-vektorer bliver for brede og "udjævner" signaler
- query matcher ofte kun en lille del af dokumentet
- chunk-level retrieval giver bedre grounding og citationer

Men:

- chunking og embeddings skal designes sammen
- daarlige chunk boundaries kan stadig skade retrieval, selv med god embedding model

## Query embedding strategier

Det er ikke kun dokumentsiden der kan optimeres. Query-siden er ogsaa vigtig.

Strategier:

1. Baseline query (embed user query som den er)
2. Query rewrite (normaliser ordvalg, stavefejl, domaene-termer)
3. Multi-query retrieval (flere omskrivninger/subqueries)
4. Query routing (vaelg collection eller metadatafilter foerst)

Tradeoffs:

- bedre recall vs mere latency/cost
- mere kompleksitet i observability/debugging

Anbefaling til kursister:

- start simpelt (baseline + metadata)
- tilfoej rewrite/multi-query kun hvis benchmark viser behov

## Metadata + embeddings (meget vigtigt)

Embeddings boer sjældent staa alene.

Typiske metadata-filtre:

- tenant/kunde
- dokumenttype
- produkt/version
- sprog
- dato/periode
- adgangsniveau/team

Hvorfor metadata-filter foer retrieval?

- bedre precision
- mindre soegerum => lavere latency
- governance/adgangskontrol
- bedre forklarlighed i demoer og debugging

## Hybrid retrieval (dense + sparse)

Hybrid retrieval kombinerer typisk:

- sparse (keyword/BM25-lignende)
- dense (embeddings)

Fusion kan ske med fx:

- score blending (kraever score-kalibrering)
- rank fusion (fx Reciprocal Rank Fusion / RRF)

Hvorfor hybrid ofte virker godt:

- sparse redder exact terms
- dense redder parafraser/synonymer
- samlet retrieval bliver mere robust

## Modelvalg: hvad skal man vurdere?

Vurder embedding-modeller paa mere end "benchmark score":

- retrieval quality paa egne data/querys
- latency (ingest + query)
- cost (særligt ved re-embedding)
- dimension (storage + vector DB performance)
- multilingual support
- domæne-match (fx juridisk, support, teknisk)
- stabilitet/versionering
- hosting og data/privacy-krav

### Praktisk strategi til kursister

1. Vælg en solid general-purpose baseline model
2. Evaluer paa 20-50 realistiske spørgsmål
3. Maal top-k hit quality (ikke kun final answer)
4. Test metadata-filtre og hybrid retrieval
5. Skift model kun hvis data viser tydelig gevinst

## Vigtige faldgruber

1. Re-embedding uden versionsstyring
- Gem modelnavn/version i metadata eller collection-navn

2. Blande embeddings fra forskellige modeller i samme collection
- Undgaa dette, medmindre du bevidst migrerer og markerer versioner

3. Ingen evalueringscases
- "Det foeles bedre" er ikke nok

4. For aggressiv chunking
- Embeddings kan ikke redde chunks uden noedvendig kontekst

5. Overfokus paa modelvalg
- Metadata, chunking og retrieval-konfiguration giver ofte stoerre gevinst

6. Ignorere sprog og domæne
- En model kan se god ud paa engelsk men falde paa dansk fagsprog

## Evalueringsmetode (praktisk og undervisningsvenlig)

Lav et lille eval-set med:

- 20-50 realistiske spørgsmål
- forventede dokumenter/chunks (eller acceptable kilder)
- eventuelle metadatafiltre

Maal mindst:

- hit-rate@k (kommer korrekt kilde i top-k?)
- precision@k / relevansvurdering
- latency
- cost per query (groft estimat er fint i undervisning)

Udvid senere med:

- answer-level evaluering (grounding, korrekthed, citationer)
- failure-mode kategorisering

## Sammenhaeng med resten af 3-dages kurset

Embeddings-modulet boer bindes til:

- chunking-modulet (`chunking.md`)
- retrieval tuning (top-k, filters, hybrid, rerank)
- prompting/grounding (hvordan retrieved kontekst bruges)
- observability/evaluering (sporbarhed af retrieval hits)
- demo_full (Qdrant ingest + chat + API)

## Praktiske demoer i denne repo

Se `embeddings/README.md` og koer fx:

```bash
python3 embeddings/01_lexical_vs_dense.py
python3 embeddings/02_similarity_metrics.py
python3 embeddings/04_chunk_vs_document_embeddings.py
python3 embeddings/06_hybrid_retrieval_rrf.py
python3 embeddings/07_model_tradeoffs.py
```

Hvis API/noegle er tilgaengelig:

```bash
python3 embeddings/08_openai_embeddings_demo.py
```

## Opsummering (de vigtigste pointer)

- Embeddings er centralt for semantic retrieval i RAG
- Embeddings skal designes sammen med chunking og metadata
- Hybrid + metadata-filtre er ofte mere robust end "dense-only"
- Modelvalg er et tradeoff (quality/latency/cost/dimension), ikke en absolut sandhed
- Evaluer paa egne data og spørgsmål - ikke kun generelle benchmarks

