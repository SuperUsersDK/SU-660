PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install check demo demo-all demo-real demo-real-check demo-real-rag demo-real-agent demo-full demo-full-ingest demo-full-chat slides clean

help:
	@echo "Targets:"
	@echo "  make venv            - create virtual environment"
	@echo "  make install         - install requirements in $(VENV)"
	@echo "  make check           - syntax-check demo and demo_real Python files"
	@echo "  make demo            - run a small local mock demo set"
	@echo "  make demo-all        - run all local mock demos"
	@echo "  make demo-real       - show available real demo commands"
	@echo "  make demo-real-check - verify LangChain/OpenAI deps are installed in $(VENV)"
	@echo "  make demo-real-rag   - run LangChain+Chroma RAG demo (needs API key)"
	@echo "  make demo-real-agent - run LangChain tool-calling agent demo (needs API key)"
	@echo "  make demo-full       - show available demo_full commands"
	@echo "  make demo-full-ingest - ingest PDFs into Qdrant using demo_full/app/ingest.py"
	@echo "  make demo-full-chat  - start demo_full CLI chat"
	@echo "  make slides          - print slides file path"
	@echo "  make clean           - remove caches and generated outputs"

$(PY):
	$(PYTHON) -m venv $(VENV)

venv: $(PY)

install: $(PY)
	$(PIP) install -r requirements.txt

check: $(PY)
	$(PY) -m py_compile demo/*.py
	$(PY) -m py_compile demo_real/*.py

demo: $(PY)
	$(PY) demo/01_prompt_template.py
	$(PY) demo/05_rag_chat.py
	$(PY) demo/07_chain_with_tool.py

demo-all: $(PY)
	$(PY) demo/01_prompt_template.py
	$(PY) demo/02_context_window_chat.py
	$(PY) demo/03_chunking_strategies.py
	$(PY) demo/04_retrieval_only.py
	$(PY) demo/05_rag_chat.py
	$(PY) demo/06_retrieval_comparison.py
	$(PY) demo/07_chain_with_tool.py
	$(PY) demo/08_workflow_trigger.py
	$(PY) demo/09_eval_loop.py
	$(PY) demo/10_ab_compare.py

demo-real:
	@echo "Run one of these after setting OPENAI_API_KEY (.env supported):"
	@echo "If this is a fresh venv, run: make install"
	@echo "  $(PY) demo_real/01_prompt_template_openai.py"
	@echo "  $(PY) demo_real/02_pydantic_output_parser_langchain.py"
	@echo "  $(PY) demo_real/05_rag_chat_langchain_chroma.py"
	@echo "  $(PY) demo_real/07_chain_tool_langchain.py"
	@echo "  $(PY) demo_real/11_agent_tool_calling_langchain.py"
	@echo "  $(PY) demo_real/09_eval_loop_openai.py"

demo-real-check: $(PY)
	@$(PY) -c 'import importlib.util as u; mods=["openai","langchain","langchain_openai","langchain_chroma","chromadb"]; missing=[m for m in mods if u.find_spec(m) is None]; print("demo-real deps OK" if not missing else "Missing modules: " + ", ".join(missing)); raise SystemExit(0 if not missing else 1)' \
	|| { echo ""; echo "Missing Python packages in $(VENV). Run: make install"; exit 1; }

demo-real-rag: demo-real-check
	$(PY) demo_real/05_rag_chat_langchain_chroma.py

demo-real-agent: demo-real-check
	$(PY) demo_real/11_agent_tool_calling_langchain.py

demo-full:
	@echo "demo_full commands (run from repo root):"
	@echo "  cd demo_full && docker compose up -d qdrant"
	@echo "  cd demo_full && .venv/bin/pip install -r ../requirements.txt"
	@echo "  make demo-full-ingest"
	@echo "  make demo-full-chat"

demo-full-ingest:
	cd demo_full && .venv/bin/python app/ingest.py

demo-full-chat:
	cd demo_full && .venv/bin/python app/chat_cli.py

slides:
	@echo "slides/DemoSlides.md"

clean:
	find demo demo_real -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -f demo/output/*.jsonl demo_real/output/*.jsonl
