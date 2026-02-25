# Demoer til `KursusPlan.md`

Denne mappe indeholder Python-demoer og korte beskrivelser til alle demo-punkter, der er nævnt i `KursusPlan.md`.

## Kørsel

Anbefalet (via `venv`):

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 demo/01_prompt_template.py
python3 demo/05_rag_chat.py
python3 demo/07_chain_with_tool.py
```

Eller kør en enkelt demo direkte fra repo-roden:

```bash
.venv/bin/python demo/01_prompt_template.py
```

Alle demoer bruger kun Python standardbibliotek og en lille mock-LLM/retrieval-implementering i `demo/common.py`, så de kan køres uden API-nøgler.

## Oversigt over demoer

### 01 - Fra simpel prompt til robust prompt-template i Python

- Fil: `demo/01_prompt_template.py`
- Formål: Viser forskellen mellem en løs prompt og en struktureret prompt-template med rolle, task, few-shot og JSON-outputkrav.
- Brug i undervisning: Introduktion til prompt engineering og hvorfor templates giver mere stabil adfærd.

### 02 - Samtaleflow med kontrolleret kontekstvindue

- Fil: `demo/02_context_window_chat.py`
- Formål: Simulerer memory-styring og token-budgettering ved at trimme historik ned til et max token-estimat.
- Brug i undervisning: Forklar truncation, context windows og hvorfor "mere kontekst" ikke altid er bedre.

### 03 - Chunking-strategier på samme dokument og effekt på retrieval

- Fil: `demo/03_chunking_strategies.py`
- Formål: Sammenligner fast chunking vs. sætningsbaseret chunking på samme dokument og viser retrieval-forskelle.
- Brug i undervisning: God til at diskutere chunk-størrelse, overlap og relevans.

### 04 - Første retrieval uden generering (kun søgning/top-k)

- Fil: `demo/04_retrieval_only.py`
- Formål: Demonstrerer retrieval som separat trin og viser top-k resultater med scores.
- Brug i undervisning: Vigtigt for at skille retrieval-fejl fra generations-fejl.

### 05 - Simpel RAG-chat i Python

- Fil: `demo/05_rag_chat.py`
- Formål: End-to-end mini-RAG: retrieval + grounded svar med kildehenvisning.
- Brug i undervisning: Første samlede RAG-flow før mere avanceret tuning.

### 06 - Sammenligning af retrieval-resultater ved forskellige indstillinger

- Fil: `demo/06_retrieval_comparison.py`
- Formål: Sammenligner chunking og metadata-filtering på et lille benchmark og opsummerer målinger.
- Brug i undervisning: Viser tuning og eksperimentel tilgang til retrieval.

### 07 - Chain der kombinerer retrieval og et eksternt tool-kald

- Fil: `demo/07_chain_with_tool.py`
- Formål: Simulerer et chain-flow med retrieval-trin, grounded svar og et separat tool-kald (mock valutakurs).
- Brug i undervisning: Introduktion til tool integration og opdeling i deltrin.

### 08 - Simpelt workflow for dokumentbehandling eller evalueringstrigger

- Fil: `demo/08_workflow_trigger.py`
- Formål: Viser event-payloads for document ingest og evaluation trigger (kan bruges som input til n8n/webhook).
- Brug i undervisning: Bro mellem Python-komponenter og workflow-orkestrering.

### 09 - Simpel eval-loop i Python (manuelle ratings + basale målinger)

- Fil: `demo/09_eval_loop.py`
- Formål: Kører et lille eval-sæt, beregner simple keyword-baserede målinger og logger manuelle ratings.
- Brug i undervisning: Introducerer eval-loop, logging og kvalitetssikring.

### 10 - Sammenlign to promptversioner eller to retrieval-opsaetninger

- Fil: `demo/10_ab_compare.py`
- Formål: En simpel A/B-teststruktur, hvor promptvariant/top-k kan sammenlignes på samme eval-sæt.
- Brug i undervisning: Viser principper for A/B-test og kontrollerede ændringer.

## Forslag til udvidelser (til live-kursus)

- Erstat mock-LLM i `demo/common.py` med rigtig modelklient (fx OpenAI/Azure/OpenRouter).
- Tilføj embeddings/vector store (FAISS/Chroma) bag samme interface som `retrieve(...)`.
- Erstat mock tool-kald i `demo/07_chain_with_tool.py` med rigtige API-kald.
- Send events fra `demo/08_workflow_trigger.py` til en rigtig webhook/n8n-instans.
