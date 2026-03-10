# Real integrations demoer (OpenAI + LangChain + Chroma)

Denne mappe er en "rigtig" version af udvalgte demoer fra `demo/`, hvor mock-komponenterne er erstattet med:

- OpenAI (chat + embeddings)
- LangChain (prompt templates, chains, tools)
- Chroma (vector store / retrieval)

## Forudsætninger

1. Installer dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Bemærk om versioner:

- Demoerne er nu skrevet til `langchain` 1.x API (agent-demoen bruger `create_agent`, ikke `AgentExecutor/create_tool_calling_agent`).
- Anbefalet er at bruge versions-ranges i `requirements.txt` (nu sat til `langchain*` 1.x).
- Testet i dette repo-miljø med bl.a. `langchain==1.2.10`, `langchain-openai==1.1.10`, `langchain-chroma==1.1.0`, `chromadb==1.5.1`, `openai==2.23.0`.

2. Opret miljøvariabler i repo-roden (kopiér fra `.env.example`):

```bash
cp .env.example .env
```

3. Kør demoer fra repo-roden:

```bash
.venv/bin/python demo_real/01_prompt_template_openai.py
.venv/bin/python demo_real/02_pydantic_output_parser_langchain.py
.venv/bin/python demo_real/05_rag_chat_langchain_chroma.py
.venv/bin/python demo_real/07_chain_tool_langchain.py
.venv/bin/python demo_real/11_agent_tool_calling_langchain.py
.venv/bin/python demo_real/09_eval_loop_openai.py
```

Direkte fil-kørsel (`python3 demo_real/...py`) er understøttet i demoerne.

## Dækkede kursus-demoer

- Demo 1: robust prompt template (`01_prompt_template_openai.py`)
- Demo 2: struktureret output med Pydantic parser (`02_pydantic_output_parser_langchain.py`)
- Demo 5: simpel RAG-chat (`05_rag_chat_langchain_chroma.py`)
- Demo 7: chain + tool-kald (`07_chain_tool_langchain.py`)
- Demo 7 (udvidet): komplet agent med tool-calling (`11_agent_tool_calling_langchain.py`)
- Demo 9/10: eval-loop og A/B-lignende sammenligning (`09_eval_loop_openai.py`)

## Noter til undervisning

- Demoerne er små og undervisningsvenlige, ikke produktionsklar kode.
- Der bruges lokal Chroma persist i `demo_real/.chroma/`.
- API-kald koster penge; brug korte prompts og små demo-datasæt live.
- Hvis I opgraderer `langchain` major-version igen, forvent især ændringer omkring agents/tool-calling APIs.
