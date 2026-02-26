# Embeddings (Slides / undervisningsversion)

Kort slide/manus-version til kursusmodul om embeddings i RAG.

## Slide 1: Titel

**Embeddings til RAG**

- hvad embeddings er
- retrieval-strategier
- modelvalg og tradeoffs
- praktiske demoer

Talepunkt:

- "Chunking bestemmer hvad der kan findes. Embeddings bestemmer hvordan lighed maales."

## Slide 2: Laeringsmaal

Efter sessionen kan deltagerne:

- forklare dense vs sparse retrieval
- forstaa cosine/dot/normalisering
- bruge metadata sammen med embeddings
- forklare hybrid retrieval
- diskutere modelvalg (quality/cost/latency)

## Slide 3: Hvad er embeddings?

Embeddings = vektor-repraesentation af tekst

Bruges til:

- embed query
- embed chunks
- sammenligne lighed i vector space

Ikke nok alene:

- chunking
- metadata
- retrieval config
- prompting

## Slide 4: Dense vs sparse retrieval

Sparse (keyword/BM25-lignende):

- stærk paa exact match
- god til IDs / termer

Dense (embeddings):

- stærk paa semantik og parafraser
- ofte bedre til naturlige spoergsmaal

Praktisk pointe:

- dense erstatter ikke altid sparse

## Slide 5: Similarity metrics

Typiske valg:

- cosine similarity
- dot product

Vigtig pointe:

- uden normalisering kan dot favorisere "lange" vektorer
- med L2-normalisering bliver dot ofte ~ cosine

Demo:

- `embeddings/02_similarity_metrics.py`

## Slide 6: Chunk-level vs document-level embeddings

Hvorfor chunk-level i RAG:

- mindre stoej
- mere specifik retrieval
- bedre grounding/citation

Risiko ved document-level:

- relevante signaler drukner i resten af dokumentet

Demo:

- `embeddings/04_chunk_vs_document_embeddings.py`

## Slide 7: Query-side strategier

Muligheder:

- baseline query
- query rewrite
- multi-query retrieval
- routing til korrekt collection/filter

Tradeoff:

- recall op
- latency/cost/kompleksitet op

Demo:

- `embeddings/03_query_embedding_strategies.py`

## Slide 8: Metadata + embeddings

Embeddings alene er sjældent nok.

Metadata-filtre:

- tenant
- type
- lang
- version
- dato

Effekt:

- bedre precision
- governance/adgangskontrol
- lavere latency

Demo:

- `embeddings/05_metadata_filtering.py`

## Slide 9: Hybrid retrieval

Kombiner:

- sparse retrieval
- dense retrieval

Fusion:

- score blending
- RRF (Reciprocal Rank Fusion)

Hvorfor:

- exact terms + semantik i samme pipeline

Demo:

- `embeddings/06_hybrid_retrieval_rrf.py`

## Slide 10: Modelvalg (ingen universel vinder)

Vurder:

- quality paa egne queries
- latency
- cost
- embedding dimension
- multilingual/domain fit
- privacy/hosting

Demo:

- `embeddings/07_model_tradeoffs.py`

## Slide 11: Production pointers

- versioner embedding-model i metadata/collection-navn
- bland ikke flere modeller i samme collection uden plan
- maal retrieval@k foer du tuner prompts
- behold logs over retrieval hits og scores

## Slide 12: Evalueringsstrategi

Lav et lille eval-set:

- realistiske spørgsmål
- forventede kilder/chunks
- metadatafiltre hvis relevant

Maal:

- hit-rate@k
- precision/relevance
- latency
- cost (groft estimat)

## Slide 13: 3-dages kursusplacering

Dag 1:

- prompts + simple retrieval begreber

Dag 2:

- chunking + embeddings (denne session)

Dag 3:

- RAG pipeline tuning, evaluering, agents, API, drift

## Slide 14: Live demo-forloeb (foreslaaet)

1. `01_lexical_vs_dense.py`
2. `02_similarity_metrics.py`
3. `04_chunk_vs_document_embeddings.py`
4. `05_metadata_filtering.py`
5. `06_hybrid_retrieval_rrf.py`
6. `07_model_tradeoffs.py`
7. (valgfri) `08_openai_embeddings_demo.py`

## Slide 15: Opsummering

- embeddings er centrale, men virker bedst sammen med chunking + metadata
- hybrid retrieval er ofte en stærk praktisk default
- modelvalg er et tradeoff, ikke et modevalg
- evaluer retrieval først, sa prompt og svar derefter

