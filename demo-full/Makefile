PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
COMPOSE ?= docker compose

.PHONY: help venv install deps-check up down logs ps ingest chat api api-docker check

help:
	@echo "Targets (run inside demo-full/):"
	@echo "  make venv      - create local virtualenv"
	@echo "  make install   - install Python deps into $(VENV)"
	@echo "  make deps-check - verify Python deps are installed in $(VENV)"
	@echo "  make up        - start Qdrant (Docker)"
	@echo "  make down      - stop containers"
	@echo "  make logs      - tail Qdrant logs"
	@echo "  make ps        - show compose services"
	@echo "  make check     - syntax-check app/*.py"
	@echo "  make ingest    - ingest PDFs from data/pdfs into Qdrant"
	@echo "  make chat      - start local CLI chat (agent + tools)"
	@echo "  make api       - start local FastAPI on :8000"
	@echo "  make api-docker - start optional API container profile"

$(PY):
	$(PYTHON) -m venv $(VENV)

venv: $(PY)

install: $(PY)
	$(PIP) install -r requirements.txt

deps-check: $(PY)
	@$(PY) -c 'import importlib.util as u; mods=["fastapi","uvicorn","qdrant_client","langchain","langchain_community","langchain_openai","langchain_text_splitters","pypdf"]; missing=[m for m in mods if u.find_spec(m) is None]; print("demo-full deps OK" if not missing else "Missing modules: " + ", ".join(missing)); raise SystemExit(0 if not missing else 1)' \
	|| { echo ""; echo "Missing Python packages in $(VENV). Run: make install"; exit 1; }

up:
	$(COMPOSE) up -d qdrant

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f qdrant

ps:
	$(COMPOSE) ps

check:
	$(PYTHON) -m py_compile app/*.py

ingest: deps-check
	$(PY) app/ingest.py

chat: deps-check
	$(PY) app/chat_cli.py

api: deps-check
	$(PY) -m uvicorn app.api:app --reload --port 8000

api-docker:
	$(COMPOSE) --profile api up -d rag_api
