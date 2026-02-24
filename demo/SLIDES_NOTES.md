# Slide-notes til demoer (KursusPlan)

Dette dokument er talepunkter og live-demo-noter til hver demo i `demo/`.

## Generel demo-struktur (bruges pa alle demoer)

- Start med "hvad viser denne demo?" (1 sætning)
- Kør demoen
- Stop efter output og peg på 1-2 konkrete observationer
- Forklar hvad deltagerne skal ændre i øvelsen bagefter
- Hav en fallback: vis kode og forklar forventet output hvis noget fejler

## Demo 1 - `demo/01_prompt_template.py`

### Formål

- Vise forskellen mellem en løs prompt og en robust prompt-template.

### Talepunkter

- Samme opgave kan give meget forskellig kvalitet afhængig af struktur.
- Rolle + task + constraints + few-shot reducerer varians.
- JSON/outputformat er ofte vigtigere end "smart formulering".

### Live-fokus

- Sammenlign de to prompts visuelt.
- Peg på at robust prompten er længere, men lettere at vedligeholde.

### Øvelse-bridge

- Bed deltagerne omskrive en eksisterende prompt til template med 2-3 faste felter.

## Demo 2 - `demo/02_context_window_chat.py`

### Formål

- Forklare memory-styring, token-budget og truncation i samtaler.

### Talepunkter

- Mere historik er ikke altid bedre.
- Vælg hvad der skal bevares: instruktioner, beslutninger, åbne spørgsmål.
- Token-budget er en designbeslutning, ikke kun en modelgrænse.

### Live-fokus

- Vis hvilke beskeder der beholdes/fjernes.
- Ændr `max_context_tokens` live for at vise adfærdsændring.

### Øvelse-bridge

- Deltagere implementerer egen strategi: "seneste N" vs "summarized memory".

## Demo 3 - `demo/03_chunking_strategies.py`

### Formål

- Vise hvordan chunking-strategi ændrer retrieval-resultater.

### Talepunkter

- Chunk-størrelse og overlap er domæneafhængigt.
- For store chunks giver støj; for små chunks mister sammenhæng.
- Chunking skal testes empirisk, ikke gættes.

### Live-fokus

- Sammenlign top-k hits for samme query.
- Vis at relevans-score kan ændre sig selv ved små chunking-ændringer.

### Øvelse-bridge

- Lad teams definere chunking-regler til deres egne dokumenttyper.

## Demo 4 - `demo/04_retrieval_only.py`

### Formål

- Isolere retrieval-trinnet før generering.

### Talepunkter

- Debug retrieval først, ellers fejlfortolkes som modelproblem.
- Top-k og metadata-felter bør inspiceres manuelt.
- Retrieval-output er et produktionsartefakt, ikke kun intern teknik.

### Live-fokus

- Vis query og top-k resultater uden modeloutput.
- Forklar hvorfor dette er den hurtigste fejlfindingsmetode.

### Øvelse-bridge

- Deltagerne skal dokumentere 2 retrieval-fejl og foreslå forbedring.

## Demo 5 - `demo/05_rag_chat.py`

### Formål

- Vise et minimalt end-to-end RAG-flow.

### Talepunkter

- RAG = retrieval + prompting + generation + grounding.
- "Grounded" svar betyder ikke nødvendigvis "korrekt", kun "forankret i kontekst".
- Kildehenvisning hjælper debugging og tillid.

### Live-fokus

- Vis top hits før svaret.
- Sammenlign et godt spørgsmål og et svagere spørgsmål.

### Øvelse-bridge

- Deltagerne justerer top-k eller filter og evaluerer forskel i svar.

## Demo 6 - `demo/06_retrieval_comparison.py`

### Formål

- Introducere retrieval-tuning med simple målinger.

### Talepunkter

- Brug små benchmarks tidligt; vent ikke på "perfekt eval setup".
- Sammenlign kun få parametre ad gangen.
- Metadata-filtre kan være lige så vigtige som embeddings.

### Live-fokus

- Gå direkte til tabellen og forklar kolonnerne.
- Peg på tradeoff mellem score og kompleksitet.

### Øvelse-bridge

- Deltagere designer deres egen mini-måling (precision-lignende eller coverage).

## Demo 7 - `demo/07_chain_with_tool.py`

### Formål

- Vise opdeling i retrieval-trin, tool-trin og sammensat svar.

### Talepunkter

- Ikke alt skal løses af LLM'en.
- Tool-kald giver deterministiske svar til beregning/opslag.
- Chains gør systemet mere forklarligt og testbart.

### Live-fokus

- Vis output fra hvert trin separat.
- Fremhæv hvorfor tool-resultatet ikke bør "hallucineres".

### Øvelse-bridge

- Deltagere tilføjer et ekstra tool (fx prisberegning eller datovalidering).

## Demo 8 - `demo/08_workflow_trigger.py`

### Formål

- Forklare workflow-orkestrering og event-baserede triggere.

### Talepunkter

- Python-komponenter og workflow-værktøjer bør kobles via simple payloads/events.
- Ingestion og evaluering kan være asynkrone flows.
- God event-logging gør drift og audit nemmere.

### Live-fokus

- Vis payload JSON og forklar felter.
- Peg på hvordan samme payload kan sendes til n8n/webhook senere.

### Øvelse-bridge

- Deltagere designer et workflow med mindst 3 events/steps.

## Demo 9 - `demo/09_eval_loop.py`

### Formål

- Introducere eval-loop med automatiske og manuelle ratings.

### Talepunkter

- Start simpelt: lille eval-sæt > ingen evaluering.
- Kombinér målinger med manuel vurdering.
- Eval-resultater skal logges og kunne gentages.

### Live-fokus

- Gennemgå tabellen: hvad betyder `keyword_score` og `manual_rating`.
- Vis outputfilen og forklar hvorfor JSONL er praktisk.

### Øvelse-bridge

- Deltagere skriver 3 egne eval-cases og scorer dem.

## Demo 10 - `demo/10_ab_compare.py`

### Formål

- Vise A/B-testtankegang for prompts og retrieval-indstillinger.

### Talepunkter

- Skift kun én variabel ad gangen.
- A/B-resultater uden faste testcases er svære at stole på.
- Målet er læring og retning, ikke statistisk perfektion i starten.

### Live-fokus

- Vis hvilke parametre der ændres mellem varianter.
- Spørg holdet: "Hvad ville I teste som næste B-variant?"

### Øvelse-bridge

- Deltagere vælger en hypotese og opstiller A/B-testplan.

## Praktiske live-demo tips

- Hav terminalvindue med stor font og "clean" output.
- Kør demoerne én gang før undervisning samme dag.
- Gem kendte outputs/screenshots som backup ved netværksproblemer.
- Marker tydeligt hvad der er mock (`demo/`) og hvad der er real integrations (`demo_real/`).
