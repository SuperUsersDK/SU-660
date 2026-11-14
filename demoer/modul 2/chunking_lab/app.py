"""Chunking-laboratoriet: se hvordan chunking-strategier klipper tekst, og hvad det betyder for søgning.

Start fra denne mappe:

    uvicorn app:app --port 8020
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import embedder
import store
from chunkers import STRATEGIES, STRATEGY_BY_ID, Chunk, run_strategy

load_dotenv(ROOT_DIR / ".env", override=True)

TEXT_DIR = HERE / "texts"
MAX_CUSTOM_CHARS = 20_000
PREVIEW_VALUES = 32  # antal tal fra vektoren, der vises i animationen

app = FastAPI(title="Chunking-laboratoriet")
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


# --------------------------------------------------------------------------- #
# Tekster
# --------------------------------------------------------------------------- #

def load_texts() -> list[dict]:
    texts = json.loads((TEXT_DIR / "texts.json").read_text(encoding="utf-8"))
    for item in texts:
        item["text"] = (TEXT_DIR / item["file"]).read_text(encoding="utf-8")
    return texts


TEXTS = {item["id"]: item for item in load_texts()}


def resolve_text(text_id: str | None, custom: str | None) -> tuple[str, str]:
    if custom is not None:
        # Teksten bruges uændret, så chunk-positionerne passer med browserens tekst.
        if not custom.strip():
            raise HTTPException(400, "Skriv eller indsæt en tekst først.")
        if len(custom) > MAX_CUSTOM_CHARS:
            raise HTTPException(400, f"Teksten er for lang (maks. {MAX_CUSTOM_CHARS:,} tegn).")
        return "egen tekst", custom
    if text_id not in TEXTS:
        raise HTTPException(404, "Ukendt tekst.")
    return text_id, TEXTS[text_id]["text"]


# --------------------------------------------------------------------------- #
# Kørsler: chunk + embed + gem
# --------------------------------------------------------------------------- #

@dataclass
class Projection:
    """2D-PCA af en kørsels vektorer, så både chunks og spørgsmål kan tegnes i samme kort."""

    mean: np.ndarray
    components: np.ndarray
    low: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, vectors: list[list[float]]) -> "Projection":
        data = np.asarray(vectors, dtype=np.float64)
        mean = data.mean(axis=0)
        if len(data) < 2:
            components = np.zeros((2, data.shape[1]))
        else:
            _, _, vt = np.linalg.svd(data - mean, full_matrices=False)
            components = np.zeros((2, data.shape[1]))
            components[: min(2, len(vt))] = vt[:2]
        raw = (data - mean) @ components.T
        low = raw.min(axis=0)
        span = np.maximum(raw.max(axis=0) - low, 1e-9)
        return cls(mean, components, low, span)

    def transform(self, vector: list[float]) -> list[float]:
        raw = (np.asarray(vector, dtype=np.float64) - self.mean) @ self.components.T
        return [float(v) for v in (raw - self.low) / self.span]


@dataclass
class Run:
    run_id: str
    strategy: str
    source: str
    text: str
    chunks: list[Chunk]
    vectors: list[list[float]]
    projection: Projection
    cached: bool
    embed_ms: int
    token_counts: list[int] | None


RUNS: dict[str, Run] = {}


def make_run_id(text: str, strategy: str, params: dict) -> str:
    key = json.dumps(
        {"model": embedder.info().model, "strategy": strategy, "params": params, "text": hashlib.sha1(text.encode()).hexdigest()},
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def clean_params(strategy_id: str, params: dict | None) -> dict:
    strategy = STRATEGY_BY_ID[strategy_id]
    return {
        p.name: max(p.minimum, min(p.maximum, int((params or {}).get(p.name, p.default))))
        for p in strategy.params
    }


def index_run(strategy_id: str, source: str, text: str, params: dict | None) -> Run:
    """Chunk teksten, og embed + gem chunks i Qdrant, medmindre kørslen allerede findes."""
    if strategy_id not in STRATEGY_BY_ID:
        raise HTTPException(404, "Ukendt strategi.")
    params = clean_params(strategy_id, params)
    run_id = make_run_id(text, strategy_id, params)
    if run_id in RUNS:
        return RUNS[run_id]

    chunks = run_strategy(strategy_id, text, params)  # ValueError: strategien passer ikke til teksten
    if not chunks:
        raise ValueError("Strategien gav ingen chunks.")

    info = embedder.info()
    client = store.get_client()
    collection = store.collection_name(info.model)
    started = time.perf_counter()
    vectors = store.stored_vectors(client, collection, run_id, len(chunks))
    cached = vectors is not None
    if vectors is None:
        vectors = embedder.embed_passages([chunk.text for chunk in chunks])
        store.ensure_collection(client, collection, len(vectors[0]))
        store.save_run(client, collection, run_id, chunks=chunks, vectors=vectors, strategy=strategy_id, source=source)
    run = Run(
        run_id=run_id,
        strategy=strategy_id,
        source=source,
        text=text,
        chunks=chunks,
        vectors=vectors,
        projection=Projection.fit(vectors),
        cached=cached,
        embed_ms=int((time.perf_counter() - started) * 1000),
        token_counts=embedder.token_counts([chunk.text for chunk in chunks]),
    )
    RUNS[run_id] = run
    return run


def preview(vector: list[float]) -> list[float]:
    """De første tal i vektoren, skaleret til -1..1, så animationen kan tegne dem som søjler."""
    head = np.asarray(vector[:PREVIEW_VALUES], dtype=np.float64)
    peak = float(np.abs(head).max()) or 1.0
    return [round(float(v) / peak, 3) for v in head]


def run_payload(run: Run) -> dict:
    info = embedder.info()
    lengths = [len(chunk.text) for chunk in run.chunks]
    truncated = [count > info.max_tokens for count in run.token_counts] if run.token_counts and info.max_tokens else []
    return {
        "run_id": run.run_id,
        "strategy": run.strategy,
        "cached": run.cached,
        "embed_ms": run.embed_ms,
        "collection": store.collection_name(info.model),
        "model": {"provider": info.provider, "name": info.model, "dims": len(run.vectors[0]), "max_tokens": info.max_tokens},
        "stats": {
            "chunks": len(run.chunks),
            "avg": round(sum(lengths) / len(lengths)),
            "min": min(lengths),
            "max": max(lengths),
            "truncated": sum(truncated),
        },
        "chunks": [
            {
                "index": index,
                "start": chunk.start,
                "end": chunk.end,
                "length": len(chunk.text),
                "tokens": run.token_counts[index] if run.token_counts else None,
                "meta": chunk.meta,
                "pos": run.projection.transform(vector),
                "preview": preview(vector),
            }
            for index, (chunk, vector) in enumerate(zip(run.chunks, run.vectors))
        ],
    }


def search_run(run: Run, question: str, limit: int) -> dict:
    vector = embedder.embed_query(question)
    info = embedder.info()
    hits = store.search(store.get_client(), store.collection_name(info.model), run.run_id, vector, limit)
    return {
        "hits": [{"index": index, "score": round(score, 4)} for index, score in hits],
        "pos": run.projection.transform(vector),
        "preview": preview(vector),
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

class IndexRequest(BaseModel):
    strategy: str
    text_id: str | None = None
    text: str | None = None
    params: dict = Field(default_factory=dict)


class SearchRequest(BaseModel):
    run_id: str
    question: str
    limit: int = 3


class CompareRequest(BaseModel):
    question: str
    expected: str | None = None
    text_id: str | None = None
    text: str | None = None
    params: dict[str, dict] = Field(default_factory=dict)


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/setup")
def setup() -> dict:
    info = embedder.info()
    return {
        "model": {"provider": info.provider, "name": info.model, "dims": info.dims, "max_tokens": info.max_tokens},
        "collection": store.collection_name(info.model),
        "texts": [
            {key: item[key] for key in ("id", "title", "kind", "text", "questions")}
            for item in TEXTS.values()
        ],
        "strategies": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "params": [p.__dict__ for p in s.params],
            }
            for s in STRATEGIES
        ],
    }


@app.post("/api/chunk")
def api_chunk(request: IndexRequest) -> JSONResponse:
    """Kun chunking (ingen embedding), så grænserne kan vises, mens man trækker i en parameter."""
    _, text = resolve_text(request.text_id, request.text)
    if request.strategy not in STRATEGY_BY_ID:
        raise HTTPException(404, "Ukendt strategi.")
    try:
        chunks = run_strategy(request.strategy, text, clean_params(request.strategy, request.params))
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "not_applicable": True}, status_code=422)
    lengths = [len(chunk.text) for chunk in chunks] or [0]
    return JSONResponse({
        "stats": {"chunks": len(chunks), "avg": round(sum(lengths) / len(lengths)), "min": min(lengths), "max": max(lengths)},
        "chunks": [
            {"index": i, "start": c.start, "end": c.end, "length": len(c.text), "meta": c.meta}
            for i, c in enumerate(chunks)
        ],
    })


@app.post("/api/index")
def api_index(request: IndexRequest) -> JSONResponse:
    source, text = resolve_text(request.text_id, request.text)
    try:
        run = index_run(request.strategy, source, text, request.params)
    except ValueError as exc:
        return JSONResponse({"error": str(exc), "not_applicable": True}, status_code=422)
    except Exception as exc:  # fx Qdrant nede eller manglende nøgle
        return JSONResponse({"error": f"Indeksering fejlede: {exc}"}, status_code=500)
    return JSONResponse(run_payload(run))


@app.post("/api/search")
def api_search(request: SearchRequest) -> JSONResponse:
    run = RUNS.get(request.run_id)
    if run is None:
        return JSONResponse({"error": "Kørslen findes ikke længere. Tryk på Chunk og indeksér igen."}, status_code=409)
    if not request.question.strip():
        return JSONResponse({"error": "Skriv et spørgsmål."}, status_code=400)
    try:
        return JSONResponse(search_run(run, request.question.strip(), max(1, min(request.limit, 10))))
    except Exception as exc:
        return JSONResponse({"error": f"Søgning fejlede: {exc}"}, status_code=500)


@app.post("/api/compare")
def api_compare(request: CompareRequest) -> JSONResponse:
    """Søg det samme spørgsmål mod alle strategier og vis, hvilke der finder svaret."""
    source, text = resolve_text(request.text_id, request.text)
    expected = (request.expected or "").strip()
    rows = []
    for strategy in STRATEGIES:
        row = {"strategy": strategy.id, "name": strategy.name}
        try:
            run = index_run(strategy.id, source, text, request.params.get(strategy.id))
            result = search_run(run, request.question, 3)
        except ValueError as exc:
            rows.append({**row, "not_applicable": str(exc)})
            continue
        except Exception as exc:
            rows.append({**row, "error": str(exc)})
            continue
        hits = []
        for rank, hit in enumerate(result["hits"], 1):
            chunk = run.chunks[hit["index"]]
            hits.append({
                "rank": rank,
                "index": hit["index"],
                "score": hit["score"],
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "contains_answer": bool(expected) and expected in chunk.text,
            })
        answer_rank = next((h["rank"] for h in hits if h["contains_answer"]), None)
        lengths = [len(chunk.text) for chunk in run.chunks]
        rows.append({
            **row,
            "run_id": run.run_id,
            "chunks": len(run.chunks),
            "avg": round(sum(lengths) / len(lengths)),
            "hits": hits,
            "answer_rank": answer_rank,
            # Findes svaret overhovedet helt inde i én chunk? Ellers kan ingen søgning finde det.
            "answer_intact": bool(expected) and any(expected in chunk.text for chunk in run.chunks),
        })
    return JSONResponse({"question": request.question, "expected": expected or None, "rows": rows})
