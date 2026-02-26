# Demo Slides - Generativ AI og LLM (Python)

## Slide 0 - Kursusagenda (3 dage)

### Dag 1: Fundament

- prompt design og prompt templates
- kontekstvindue / memory trimming
- retrieval-begreber og simpel RAG
- introduktion til evaluering

### Dag 2: RAG datalag og retrieval-kvalitet

- chunking-strategier (size, overlap, boundaries)
- embeddings (dense vs sparse, similarity, metadata, hybrid)
- retrieval-tuning og benchmark-tankegang
- grounding og citations

### Dag 3: Systemer, drift og integration

- tools/chains/agents
- workflows og orchestration
- eval-loops og A/B-sammenligning
- demo_full (ingest, Qdrant, CLI, API) og praktisk opsamling

Speaker notes:
- Brug denne slide til at sætte forventninger: vi bygger lagvist op.
- Peg på at chunking + embeddings er dag 2-kernen for god RAG-kvalitet.
- Kobl dag 3 til "fra demo til system" (reproducérbarhed, evaluering, integration).

---

## Slide 1 - Demo 1: Prompt template

- Mål: Vis forskellen mellem løs prompt og robust prompt-template
- Fokus: rolle, task, constraints, few-shot, outputformat
- Demo-fil: `demo/01_prompt_template.py`

Speaker notes:
- Start med at vise samme opgave i to varianter.
- Peg på at template gør adfærden mere stabil og lettere at genbruge.
- Kobl direkte til øvelse 1: deltagere omskriver egen prompt.

---

## Slide 2 - Demo 2: Kontrolleret kontekstvindue

- Mål: Forklare memory-styring og token-budgettering
- Fokus: trimming/truncation af historik
- Demo-fil: `demo/02_context_window_chat.py`

Speaker notes:
- Vis hvilke beskeder der bliver beholdt, og hvilke der ryger ud.
- Ændr `max_context_tokens` live for at vise tradeoff.
- Pointen er designvalg, ikke bare modelgrænser.

---

## Slide 3 - Demo 3: Chunking-strategier

- Mål: Vise hvordan chunking påvirker retrieval
- Fokus: fixed chunks vs sætningsbaseret chunking
- Demo-fil: `demo/03_chunking_strategies.py`

Speaker notes:
- Sammenlign retrieval-resultater for samme query.
- Forklar hvorfor chunk size og overlap er domæneafhængigt.
- Brug outputtet som afsæt til øvelse 3.

---

## Slide 4 - Demo 4: Retrieval uden generering

- Mål: Isolere retrieval-laget før RAG
- Fokus: top-k hits, scores, inspection
- Demo-fil: `demo/04_retrieval_only.py`

Speaker notes:
- Dette er den hurtigste måde at debugge RAG på.
- Fremhæv at retrieval-fejl ofte ligner model-fejl.
- Bed deltagerne tænke i "hvorfor kom dette hit med?".

---

## Slide 5 - Demo 5: Simpel RAG-chat

- Mål: End-to-end mini-RAG (retrieval + grounded svar)
- Fokus: kildehenvisning og kontekstforankring
- Demo-fil: `demo/05_rag_chat.py`

Speaker notes:
- Vis top hits før modelsvaret.
- Forklar forskellen på "grounded" og "sand/faktisk korrekt".
- Brug et godt og et mindre godt spørgsmål for kontrast.

---

## Slide 6 - Demo 6: Retrieval-sammenligning

- Mål: Introducere retrieval-tuning med simple målinger
- Fokus: chunking, metadata-filter, benchmark
- Demo-fil: `demo/06_retrieval_comparison.py`

Speaker notes:
- Gå direkte til tabellen og forklar kolonnerne.
- Fremhæv at små benchmark-tests er nok til tidlig læring.
- Kobl til øvelse 6 (eksperiment og måling).

---

## Slide 6A - Embeddings modul: Dense vs sparse retrieval

- Mål: Forklare hvad embeddings bidrager med i retrieval-laget
- Fokus: dense (embedding) vs sparse (keyword) retrieval
- Demo-fil: `embeddings/01_lexical_vs_dense.py`

Speaker notes:
- Brug denne slide som bro fra chunking til retrieval-kvalitet.
- Peg på at embeddings ikke erstatter chunking eller metadata.
- Vis hvordan parafraser kan rammes bedre med dense retrieval.

---

## Slide 6B - Embeddings modul: Similarity, metadata og hybrid

- Mål: Vise praktiske designvalg i embedding-baseret retrieval
- Fokus: cosine vs dot, metadata-filter, hybrid retrieval (RRF)
- Demo-filer:
  - `embeddings/02_similarity_metrics.py`
  - `embeddings/05_metadata_filtering.py`
  - `embeddings/06_hybrid_retrieval_rrf.py`

Speaker notes:
- Start med metric/normalisering, ellers bliver scores svære at tolke.
- Vis metadata-filter før vector search som precision/governance-greb.
- Slut med hybrid retrieval som en stærk praktisk default i mange RAG-systemer.

---

## Slide 6C - Embeddings modul: Modelvalg og kursusforankring

- Mål: Forberede deltagerne på rigtige modelvalg i dag 2/3 RAG-arbejde
- Fokus: quality, latency, cost, dimension, evaluering
- Demo-filer:
  - `embeddings/07_model_tradeoffs.py`
  - (valgfri live) `embeddings/08_openai_embeddings_demo.py`
- Materialer:
  - `embeddings.md`
  - `embeddings_slides.md`
  - `embeddings/README.md`

Speaker notes:
- Betoningen skal være "evaluer på egne queries", ikke model-branding.
- Kobl tilbage til chunking: embeddings og chunking skal tunes sammen.
- Placer modulet som dag 2-kernen før mere komplette RAG pipelines (`demo_full`).

---

## Slide 7 - Demo 7: Chain + tool-kald

- Mål: Vise opdeling mellem retrieval, tool, og sammensat svar
- Fokus: deterministiske tools vs generativt output
- Demo-fil: `demo/07_chain_with_tool.py`

Speaker notes:
- Vis outputtet step-by-step, ikke kun slutsvaret.
- Forklar hvorfor beregninger/opslag bør være tool-kald.
- Brug som intro til LangChain tools/chains.

---

## Slide 8 - Demo 8: Workflow trigger (n8n-bro)

- Mål: Vise event-baseret workflow-orkestrering
- Fokus: payloads til ingestion og evaluering
- Demo-fil: `demo/08_workflow_trigger.py`

Speaker notes:
- Vis JSON payload og forklar felter.
- Kobl til n8n/webhook-koncept uden at kræve live n8n setup.
- Fremhæv logging/audit-værdi af event-streams.

---

## Slide 9 - Demo 9: Eval-loop

- Mål: Introducere eval-loop med simple målinger + manuelle ratings
- Fokus: repeatable evaluering og logging
- Demo-fil: `demo/09_eval_loop.py`

Speaker notes:
- Forklar `keyword_score` som pædagogisk startmåling, ikke endelig sandhed.
- Vis JSONL output som artefakt til senere analyse.
- Kobl til deltagernes egne eval-cases i øvelse 9/10.

---

## Slide 10 - Demo 10: A/B-sammenligning

- Mål: Vise kontrolleret sammenligning af prompts/retrieval-settings
- Fokus: én variabel ad gangen
- Demo-fil: `demo/10_ab_compare.py`

Speaker notes:
- Spørg holdet hvilken hypotese de vil teste som næste variant.
- Fremhæv at A/B handler om læring og retning, ikke perfekte stats.
- Brug som overgang til drift/monitorering og løbende forbedring.

---

## Slide 11 - Real integrations (valgfri live)

- `demo_real/05_rag_chat_langchain_chroma.py`
- `demo_real/07_chain_tool_langchain.py`
- `demo_real/11_agent_tool_calling_langchain.py`
- `demo_real/09_eval_loop_openai.py`

Speaker notes:
- Markér tydeligt forskel på mock-demoer (`demo/`) og API-afhængige demoer (`demo_real/`).
- Kør kun real demoer hvis netværk/API-nøgler er klar.
- Brug direkte `python`/`uvicorn` kommandoer for hurtig opsætning og demo-kørsel.
