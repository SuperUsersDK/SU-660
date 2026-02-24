PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install check demo demo-all demo-real demo-real-rag demo-real-agent slides clean

help:
	@echo "Targets:"
	@echo "  make venv            - create virtual environment"
	@echo "  make install         - install requirements in $(VENV)"
	@echo "  make check           - syntax-check demo and demo_real Python files"
	@echo "  make demo            - run a small local mock demo set"
	@echo "  make demo-all        - run all local mock demos"
	@echo "  make demo-real       - show available real demo commands"
	@echo "  make demo-real-rag   - run LangChain+Chroma RAG demo (needs API key)"
	@echo "  make demo-real-agent - run LangChain tool-calling agent demo (needs API key)"
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
	@echo "  $(PY) demo_real/01_prompt_template_openai.py"
	@echo "  $(PY) demo_real/05_rag_chat_langchain_chroma.py"
	@echo "  $(PY) demo_real/07_chain_tool_langchain.py"
	@echo "  $(PY) demo_real/11_agent_tool_calling_langchain.py"
	@echo "  $(PY) demo_real/09_eval_loop_openai.py"

demo-real-rag: $(PY)
	$(PY) demo_real/05_rag_chat_langchain_chroma.py

demo-real-agent: $(PY)
	$(PY) demo_real/11_agent_tool_calling_langchain.py

slides:
	@echo "slides/DemoSlides.md"

clean:
	find demo demo_real -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -f demo/output/*.jsonl demo_real/output/*.jsonl
