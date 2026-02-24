PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
COMPOSE ?= docker compose

.PHONY: help venv install up down logs ps ingest chat api api-docker check

help:
	@echo "Targets (run inside demo-full/):"
	@echo "  make venv      - create local virtualenv"
	@echo "  make install   - install Python deps into $(VENV)"
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

ingest: $(PY)
	$(PY) app/ingest.py

chat: $(PY)
	$(PY) app/chat_cli.py

api: $(PY)
	$(PY) -m uvicorn app.api:app --reload --port 8000

api-docker:
	$(COMPOSE) --profile api up -d rag_api
