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

2. Opret miljøvariabler (kopiér fra `.env.example`):

```bash
cp demo_real/.env.example .env
```

3. Kør demoer fra repo-roden:

```bash
.venv/bin/python demo_real/01_prompt_template_openai.py
.venv/bin/python demo_real/05_rag_chat_langchain_chroma.py
.venv/bin/python demo_real/07_chain_tool_langchain.py
.venv/bin/python demo_real/11_agent_tool_calling_langchain.py
.venv/bin/python demo_real/09_eval_loop_openai.py
```

Alternativt:

```bash
make demo-real
make demo-real-rag
make demo-real-agent
```

## Dækkede kursus-demoer

- Demo 1: robust prompt template (`01_prompt_template_openai.py`)
- Demo 5: simpel RAG-chat (`05_rag_chat_langchain_chroma.py`)
- Demo 7: chain + tool-kald (`07_chain_tool_langchain.py`)
- Demo 7 (udvidet): komplet agent med tool-calling (`11_agent_tool_calling_langchain.py`)
- Demo 9/10: eval-loop og A/B-lignende sammenligning (`09_eval_loop_openai.py`)

## Noter til undervisning

- Demoerne er små og undervisningsvenlige, ikke produktionsklar kode.
- Der bruges lokal Chroma persist i `demo_real/.chroma/`.
- API-kald koster penge; brug korte prompts og små demo-datasæt live.
