from __future__ import annotations

import base64
import html
import io
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis import find_duplicates, mix_vectors, project_2d, weighted_position
from image_embeddings import (
    get_compare_embed_model,
    get_image_embed_model,
    image_to_vector,
    model_input_preview,
    patch_size,
    text_to_vector,
)
import photo_store
from tagging import LABELS

DATA_DIR = Path(__file__).resolve().parent / "data"
COLLECTION_NAME = photo_store.COLLECTION_NAME
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}

TAG_COLORS = {
    "vrag": "#8c5a2b",
    "fisk": "#d96c06",
    "koralrev": "#c2185b",
    "dykker": "#116466",
    "smådyr": "#7b3fa0",
    "båd": "#2f6db3",
    "mennesker": "#4d7c0f",
    "solnedgang": "#c99a06",
}


def load_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=True)


# Configure the model before the first request can load and cache its weights.
load_env()
DATA_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Photo Similarity Demo")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


def get_qdrant_api_key() -> str:
    return os.getenv("QDRANT_API_KEY", "")


def get_client() -> QdrantClient:
    return QdrantClient(
        url=get_qdrant_url(),
        api_key=get_qdrant_api_key() or None,
    )


def default_collection() -> str:
    return photo_store.collection_for_model(get_image_embed_model())


# --------------------------------------------------------------------------- #
# HTML-hjælpere
# --------------------------------------------------------------------------- #

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def image_url(filename: str) -> str:
    return "/data/" + quote(filename)


def thumb_url(filename: str) -> str:
    return "/thumb/" + quote(filename)


def tag_badge(tag: object) -> str:
    if not tag:
        return ""
    color = TAG_COLORS.get(str(tag), "#5b6366")
    return f"<span class='tag' style='background:{color}'>{esc(tag)}</span>"


def result_cards(results: list[dict], *, links: bool = True) -> str:
    if not results:
        return "<p class='hint'>Ingen resultater.</p>"
    cards = []
    for item in results:
        filename = str(item["filename"])
        actions = ""
        if links:
            actions = (
                "<div class='actions'>"
                f"<a href='/mix?{urlencode({'base': filename})}'>Bland med tekst</a>"
                f"<a href='/model-view?{urlencode({'file': filename})}'>Hvad ser modellen?</a>"
                "</div>"
            )
        cards.append(
            f"""
            <div class="card">
              <a href="{image_url(filename)}" target="_blank"><img src="{thumb_url(filename)}" alt="{esc(filename)}" loading="lazy" /></a>
              <div class="meta">
                <div class="name">{esc(filename)} {tag_badge(item.get('tag'))}</div>
                <div class="score">score: {float(item['score']):.4f}</div>
                {actions}
              </div>
            </div>
            """
        )
    return "<div class='grid'>" + "".join(cards) + "</div>"


def tag_options(selected: str = "") -> str:
    options = ["<option value=''>Alle tags</option>"]
    for tag in LABELS:
        chosen = " selected" if tag == selected else ""
        options.append(f"<option value='{esc(tag)}'{chosen}>{esc(tag)}</option>")
    return "<select name='tag'>" + "".join(options) + "</select>"


def model_view_html(image: Image.Image, *, title: str, model_name: str | None = None) -> str:
    crop, kept_fraction = model_input_preview(image)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=90)
    crop_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    patch = patch_size(model_name)
    cells = crop.size[0] // patch
    cut = round((1 - kept_fraction) * 100)
    return f"""
    <div class="modelview">
      <h2>{esc(title)}</h2>
      <div class="crop" style="--cells:{cells}">
        <img src="{crop_url}" alt="modellens input" />
      </div>
      <p class="hint">Modellen ser kun dette udsnit på {crop.size[0]}×{crop.size[1]} px, delt i
      {cells}×{cells} = {cells * cells} felter på {patch}×{patch} px. {cut}% af det originale
      billede er skåret væk ved kanterne.</p>
    </div>
    """


STYLE = """
:root {
  --bg: #f3efe7; --panel: #fffaf2; --ink: #1f2a2e; --muted: #5b6366;
  --accent: #116466; --accent-2: #d96c06; --line: #d8cfc0;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0; font-family: Georgia, "Times New Roman", serif;
  background: radial-gradient(circle at top, #fff8ec, var(--bg)); color: var(--ink); }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
nav { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; }
nav a { text-decoration: none; color: var(--ink); padding: 8px 14px; border-radius: 999px;
  border: 1px solid var(--line); background: var(--panel); }
nav a.active { background: var(--accent); color: white; border-color: var(--accent); }
h1 { margin: 0 0 8px; font-size: 36px; }
p { line-height: 1.5; }
a { color: var(--accent); }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
  padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,.06); }
form { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
form + form { margin-top: 12px; }
input[type="file"] { font-size: 16px; max-width: 100%; }
input[type="text"], input[type="number"], select { font-size: 16px; padding: 10px 14px;
  border: 1px solid var(--line); border-radius: 999px; background: white; }
input[type="text"] { flex: 1; min-width: 200px; }
button { border: 0; background: linear-gradient(135deg, var(--accent), #1d7874); color: white;
  border-radius: 999px; padding: 12px 18px; cursor: pointer; font-size: 16px; }
.hint { color: var(--muted); font-size: 14px; }
.error { color: #b42318; font-weight: 700; }
.preview img { max-width: 280px; border-radius: 12px; border: 1px solid var(--line); }
.grid { margin-top: 20px; display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }
.card { background: white; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
.card img { width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; }
.meta { padding: 10px 12px; }
.name { font-weight: 700; margin-bottom: 4px; word-break: break-all; }
.score { color: var(--accent-2); }
.actions { display: flex; flex-direction: column; gap: 2px; margin-top: 6px; font-size: 14px; }
.tag { display: inline-block; color: white; font-size: 12px; font-weight: 400; padding: 1px 8px;
  border-radius: 999px; vertical-align: middle; }
.tags { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0 16px; }
.row { display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start; }
.modelview .crop { position: relative; width: 224px; height: 224px; }
.modelview .crop img { width: 224px; height: 224px; display: block; border-radius: 4px; }
.modelview .crop::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(to right, rgba(255,255,255,.7) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,.7) 1px, transparent 1px);
  background-size: calc(224px / var(--cells)) calc(224px / var(--cells)); }
.original img { max-width: min(520px, 100%); border-radius: 8px; border: 1px solid var(--line); }
.pair { display: grid; grid-template-columns: 1fr 1fr auto; gap: 12px; align-items: center;
  padding: 10px 0; border-bottom: 1px solid var(--line); }
.pair > * { min-width: 0; }
.pair img { width: 100%; max-height: 180px; object-fit: cover; border-radius: 8px; }
.columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
.columns .grid { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
@media (max-width: 600px) {
  h1 { font-size: 28px; }
  .pair { grid-template-columns: 1fr 1fr; }
  .pair > div:last-child { grid-column: 1 / -1; }
  .hint { word-break: break-word; }
}
"""

NAV = [
    ("/", "Søg"),
    ("/map", "Kort"),
    ("/mix", "Bland billede + tekst"),
    ("/duplicates", "Dubletter"),
    ("/compare", "Sammenlign modeller"),
]


def layout(title: str, body: str, *, active: str = "/", extra_head: str = "") -> str:
    nav = "".join(
        f"<a href='{href}' class='{'active' if href == active else ''}'>{label}</a>"
        for href, label in NAV
    )
    return f"""<!doctype html>
<html lang="da">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{esc(title)} · Photo Similarity Demo</title>
    <style>{STYLE}</style>
    {extra_head}
  </head>
  <body>
    <main>
      <nav>{nav}</nav>
      {body}
    </main>
  </body>
</html>"""


def tag_counts() -> dict[str, int]:
    try:
        photos = photo_store.fetch_all(get_client(), collection=default_collection(), with_vectors=False)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for photo in photos:
        if photo.tag:
            counts[photo.tag] = counts.get(photo.tag, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Søgning
# --------------------------------------------------------------------------- #

def render_page(
    *,
    uploaded_image_data_url: str | None = None,
    upload_model_view: str = "",
    query_text: str | None = None,
    tag: str = "",
    results: list[dict] | None = None,
    error: str | None = None,
) -> str:
    preview_html = ""
    if uploaded_image_data_url:
        preview_html = (
            "<div class='row'>"
            "<div class='preview'><h2>Uploadet billede</h2>"
            f"<img src='{uploaded_image_data_url}' alt='upload' /></div>"
            f"{upload_model_view}"
            "</div>"
        )
    elif query_text:
        preview_html = (
            "<div class='preview'>"
            f"<h2>Tekstsøgning: “{esc(query_text)}”</h2>"
            "<p class='hint'>Tekst-til-billede scores ligger typisk omkring 0,2–0,35; "
            "rækkefølgen er det vigtige.</p>"
            "</div>"
        )

    counts = tag_counts()
    counts_html = ""
    if counts:
        chips = "".join(
            f"{tag_badge(name)} <span class='hint'>{count}</span>"
            for name, count in sorted(counts.items(), key=lambda item: -item[1])
        )
        counts_html = (
            "<p class='hint'>Zero-shot tags fra CLIP (ingen træning). "
            "Vælg et tag for at filtrere søgningen:</p>"
            f"<div class='tags'>{chips}</div>"
        )

    error_html = f"<p class='error'>{esc(error)}</p>" if error else ""
    if tag and results is not None:
        error_html += f"<p class='hint'>Filtreret på tag: {tag_badge(tag)}</p>"

    body = f"""
      <h1>Photo Similarity Demo</h1>
      <p>Upload et JPG-billede, eller beskriv et motiv med tekst, og se de 5 mest
      lignende billeder fra Qdrant med score.</p>
      <p class="hint">Model: {esc(get_image_embed_model())}</p>
      <div class="panel">
        {counts_html}
        <form action="/search" method="post" enctype="multipart/form-data">
          <input type="file" name="image" accept=".jpg,.jpeg" required />
          {tag_options(tag)}
          <button type="submit">Find lignende billeder</button>
        </form>
        <form action="/search-text" method="post">
          <input type="text" name="query" value="{esc(query_text or '')}"
                 placeholder="fx 'a shipwreck' eller 'solnedgang over havet'" required />
          {tag_options(tag)}
          <button type="submit">Søg med tekst</button>
        </form>
        {error_html}
        {preview_html}
        {result_cards(results) if results is not None else ""}
      </div>
    """
    return layout("Søg", body, active="/")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_page()


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, image: UploadFile = File(...), tag: str = Form("")) -> str:
    if Path(image.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS:
        return render_page(error="Upload kun .jpg eller .jpeg filer.")

    raw = await image.read()
    try:
        with Image.open(io.BytesIO(raw)) as pil_image:
            vector = image_to_vector(pil_image)
            upload_view = model_view_html(pil_image, title="Det ser modellen")
    except UnidentifiedImageError:
        return render_page(error="Filen kunne ikke læses som et billede.")

    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    try:
        results = photo_store.search(get_client(), vector, collection=default_collection(), tag=tag or None)
    except Exception as exc:
        return render_page(
            uploaded_image_data_url=data_url,
            upload_model_view=upload_view,
            tag=tag,
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(uploaded_image_data_url=data_url, upload_model_view=upload_view, tag=tag, results=results)


@app.post("/search-text", response_class=HTMLResponse)
def search_text(query: str = Form(...), tag: str = Form("")) -> str:
    query = query.strip()
    if not query:
        return render_page(error="Skriv en tekst at søge efter.")

    # CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren kan
    # søges direkte mod de billedvektorer, som ingest_photos.py har gemt.
    vector = text_to_vector(query)
    try:
        results = photo_store.search(get_client(), vector, collection=default_collection(), tag=tag or None)
    except Exception as exc:
        return render_page(
            query_text=query,
            tag=tag,
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(query_text=query, tag=tag, results=results)


# --------------------------------------------------------------------------- #
# Billeder og thumbnails
# --------------------------------------------------------------------------- #

def library_path(filename: str) -> Path:
    """Slå et filnavn op i data/ uden at tillade stier uden for mappen."""
    candidate = (DATA_DIR / Path(filename).name).resolve()
    if candidate.parent != DATA_DIR.resolve() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Billedet findes ikke")
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Billedet findes ikke")
    return candidate


@lru_cache(maxsize=1024)
def _thumbnail_bytes(path: str, mtime: float) -> bytes:
    with Image.open(path) as image:
        thumb = ImageOps.exif_transpose(image).convert("RGB")
        thumb.thumbnail((200, 200))
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()


@app.get("/thumb/{filename}")
def thumbnail(filename: str) -> Response:
    path = library_path(filename)
    data = _thumbnail_bytes(str(path), path.stat().st_mtime)
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/model-view", response_class=HTMLResponse)
def model_view(file: str) -> str:
    path = library_path(file)
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
        view = model_view_html(image, title="Det ser modellen")
    body = f"""
      <h1>Hvad ser modellen?</h1>
      <p>CLIP-processoren skalerer den korteste side til 224 px og skærer et kvadrat ud
      fra midten. Det er alt, hvad modellen nogensinde ser af billedet.</p>
      <div class="panel row">
        <div class="original"><h2>Original ({width}×{height})</h2>
          <img src="{image_url(path.name)}" alt="{esc(path.name)}" /></div>
        {view}
      </div>
      <p><a href="/mix?{urlencode({'base': path.name})}">Bland dette billede med tekst</a></p>
    """
    return layout("Hvad ser modellen?", body, active="")


# --------------------------------------------------------------------------- #
# Regning med embeddings
# --------------------------------------------------------------------------- #

@app.get("/mix", response_class=HTMLResponse)
def mix(base: str = "", text: str = "", weight: float = 1.0) -> str:
    weight = max(-2.0, min(2.0, weight))
    intro = """
      <h1>Bland billede og tekst</h1>
      <p>Regning med embeddings: <code>billede + vægt × tekst</code>. Positiv vægt trækker
      resultatet mod teksten, negativ vægt trækker væk fra den.</p>
      <p class="hint">Tekst og billeder ligger forskudt i CLIP-rummet, så tekstens ligheder
      spreder sig langt mindre end billedernes. Begge led skaleres derfor efter deres spredning
      over samlingen; vægt 1 betyder, at billede og tekst trækker lige meget.</p>
    """
    client = get_client()
    collection = default_collection()

    if not base:
        try:
            photos = photo_store.fetch_all(client, collection=collection, with_vectors=False)
        except Exception as exc:
            return layout("Bland", intro + f"<p class='error'>Kunne ikke hente billeder: {esc(exc)}</p>", active="/mix")
        step = max(1, len(photos) // 24)
        picks = photos[::step][:24]
        cards = "".join(
            f"<a class='card' href='/mix?{urlencode({'base': p.filename})}'>"
            f"<img src='{thumb_url(p.filename)}' alt='{esc(p.filename)}' loading='lazy' /></a>"
            for p in picks
        )
        body = intro + f"""
          <div class="panel">
            <p>Vælg et billede at tage udgangspunkt i, eller brug <em>Bland med tekst</em>
            på et søgeresultat.</p>
            <div class="grid">{cards}</div>
          </div>
        """
        return layout("Bland", body, active="/mix")

    base_path = library_path(base)
    results_html = ""
    error_html = ""
    base_vector = photo_store.get_vector(client, base_path.name, collection=collection)
    if base_vector is None:
        error_html = "<p class='error'>Billedet findes ikke i Qdrant. Kør ingest_photos.py.</p>"
    elif text.strip():
        reference = [p.vector for p in photo_store.fetch_all(client, collection=collection)]
        mixed = mix_vectors(base_vector, text_to_vector(text.strip()), weight, reference=reference)
        results = photo_store.search(client, mixed, collection=collection, limit=10, exclude_filename=base_path.name)
        sign = "+" if weight >= 0 else "−"
        results_html = (
            f"<h2>billede {sign} {abs(weight):.2f} × “{esc(text.strip())}”</h2>"
            + result_cards(results)
        )
    else:
        results = photo_store.search(client, base_vector, collection=collection, limit=10, exclude_filename=base_path.name)
        results_html = "<h2>Uden tekst: billedets nærmeste naboer</h2>" + result_cards(results)

    body = intro + f"""
      <div class="panel">
        <div class="row">
          <div class="preview"><h2>Udgangspunkt</h2>
            <img src="{thumb_url(base_path.name)}" alt="{esc(base_path.name)}" />
            <p class="hint">{esc(base_path.name)}</p></div>
          <form action="/mix" method="get" style="flex:1">
            <input type="hidden" name="base" value="{esc(base_path.name)}" />
            <input type="text" name="text" value="{esc(text)}" placeholder="fx 'with a scuba diver' eller 'at sunset'" />
            <label class="hint">vægt
              <input type="number" name="weight" value="{weight}" step="0.25" min="-2" max="2" style="width:90px" />
            </label>
            <button type="submit">Bland</button>
          </form>
        </div>
        {error_html}
        {results_html}
      </div>
    """
    return layout("Bland", body, active="/mix")


# --------------------------------------------------------------------------- #
# Kort over vektorrummet
# --------------------------------------------------------------------------- #

_MAP_CACHE: dict[tuple[str, int], dict] = {}


def map_data() -> dict:
    """Hent alle vektorer og projicér dem til 2D. Resultatet caches pr. collection og antal."""
    collection = default_collection()
    photos = photo_store.fetch_all(get_client(), collection=collection)
    key = (collection, len(photos))
    if key not in _MAP_CACHE:
        coords = project_2d([photo.vector for photo in photos])
        _MAP_CACHE.clear()
        _MAP_CACHE[key] = {
            "points": [
                {"filename": photo.filename, "x": x, "y": y, "tag": photo.tag}
                for photo, (x, y) in zip(photos, coords)
            ],
        }
    return _MAP_CACHE[key]


@app.get("/api/map")
def api_map() -> JSONResponse:
    try:
        return JSONResponse(map_data())
    except Exception as exc:
        return JSONResponse({"error": f"Kunne ikke lave kortet: {exc}"}, status_code=500)


@app.get("/api/map/search")
def api_map_search(q: str, k: int = 10) -> JSONResponse:
    data = map_data()
    positions = {point["filename"]: (point["x"], point["y"]) for point in data["points"]}
    hits = photo_store.search(get_client(), text_to_vector(q), collection=default_collection(), limit=max(1, min(k, 50)))
    # Skarpe vægte (som CLIPs egen softmax), så det bedste hit trækker mest.
    top = max((hit["score"] for hit in hits), default=0.0)
    weights = [math.exp(100 * (hit["score"] - top)) for hit in hits]
    known = [(positions[hit["filename"]], weight) for hit, weight in zip(hits, weights) if hit["filename"] in positions]
    position = weighted_position([pos for pos, _ in known], [w for _, w in known]) if known else (0.5, 0.5)
    return JSONResponse({"hits": hits, "position": position})


MAP_SCRIPT = """
const stage = document.getElementById('stage');
const world = document.getElementById('world');
const status = document.getElementById('status');
const legend = document.getElementById('legend');
const COLORS = %COLORS%;
const THUMB = 44;
let scale = 1, tx = 0, ty = 0, nodes = {}, activeTag = null;

function apply() { world.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; }

stage.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const next = Math.min(10, Math.max(0.5, scale * Math.exp(-e.deltaY * 0.0015)));
  tx = mx - (mx - tx) * next / scale; ty = my - (my - ty) * next / scale; scale = next; apply();
}, { passive: false });

let drag = null;
stage.addEventListener('pointerdown', (e) => { drag = { x: e.clientX - tx, y: e.clientY - ty, moved: false }; });
window.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const nx = e.clientX - drag.x, ny = e.clientY - drag.y;
  if (Math.abs(nx - tx) + Math.abs(ny - ty) > 2) drag.moved = true;
  tx = nx; ty = ny; apply();
});
window.addEventListener('pointerup', () => { setTimeout(() => { drag = null; }, 0); });

function place(x, y) {
  const w = stage.clientWidth - THUMB, h = stage.clientHeight - THUMB;
  return [x * w, y * h];
}

function setDim(predicate) {
  for (const [name, el] of Object.entries(nodes)) el.classList.toggle('dim', predicate ? !predicate(name, el) : false);
}

async function load() {
  status.textContent = 'Beregner t-SNE over alle vektorer …';
  const res = await fetch('/api/map');
  const data = await res.json();
  if (data.error) { status.textContent = data.error; return; }
  const counts = {};
  for (const p of data.points) {
    const [left, top] = place(p.x, p.y);
    const a = document.createElement('a');
    a.href = '/model-view?file=' + encodeURIComponent(p.filename);
    a.className = 'pt';
    a.dataset.tag = p.tag || '';
    a.style.left = left + 'px'; a.style.top = top + 'px';
    a.style.borderColor = COLORS[p.tag] || '#888';
    a.title = p.filename + (p.tag ? ' · ' + p.tag : '');
    a.addEventListener('click', (e) => { if (drag && drag.moved) e.preventDefault(); });
    const img = document.createElement('img');
    img.src = '/thumb/' + encodeURIComponent(p.filename); img.loading = 'lazy'; img.alt = p.filename;
    a.appendChild(img); world.appendChild(a); nodes[p.filename] = a;
    counts[p.tag] = (counts[p.tag] || 0) + 1;
  }
  for (const [tag, color] of Object.entries(COLORS)) {
    if (!counts[tag]) continue;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'legend'; b.style.background = color;
    b.textContent = `${tag} (${counts[tag]})`;
    b.onclick = () => {
      activeTag = activeTag === tag ? null : tag;
      clearSearch();
      setDim(activeTag ? (name, el) => el.dataset.tag === activeTag : null);
    };
    legend.appendChild(b);
  }
  status.textContent = `${data.points.length} billeder. Scroll for at zoome, træk for at flytte, klik på et billede.`;
}

function clearSearch() {
  document.querySelectorAll('.pt.hit').forEach((el) => el.classList.remove('hit'));
  const old = document.getElementById('marker'); if (old) old.remove();
}

document.getElementById('mapsearch').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  status.textContent = 'Søger …';
  const res = await fetch('/api/map/search?' + new URLSearchParams({ q, k: 10 }));
  const data = await res.json();
  clearSearch(); activeTag = null;
  const hitNames = new Set(data.hits.map((h) => h.filename));
  setDim((name) => hitNames.has(name));
  for (const h of data.hits) if (nodes[h.filename]) nodes[h.filename].classList.add('hit');
  const [left, top] = place(data.position[0], data.position[1]);
  const m = document.createElement('div');
  m.id = 'marker'; m.textContent = '“' + q + '”';
  m.style.left = (left + THUMB / 2) + 'px'; m.style.top = (top + THUMB / 2) + 'px';
  world.appendChild(m);
  status.textContent = `Top ${data.hits.length} for “${q}” er fremhævet. Bedste score ${data.hits[0] ? data.hits[0].score.toFixed(3) : '-'}.`;
});

document.getElementById('reset').onclick = () => { clearSearch(); activeTag = null; setDim(null); scale = 1; tx = 0; ty = 0; apply(); };
load();
"""

MAP_STYLE = """
<style>
#stage { position: relative; height: 72vh; min-height: 420px; overflow: hidden; border-radius: 14px;
  background: #0d2b33; cursor: grab; touch-action: none; }
#stage:active { cursor: grabbing; }
#world { position: absolute; inset: 0; transform-origin: 0 0; }
.pt { position: absolute; width: 44px; height: 44px; border: 2px solid #888; border-radius: 6px;
  overflow: hidden; transition: opacity .2s, transform .2s; background: #123; }
.pt img { width: 100%; height: 100%; object-fit: cover; display: block; pointer-events: none; }
.pt:hover { transform: scale(2.6); z-index: 10; }
.pt.dim { opacity: .12; }
.pt.hit { opacity: 1; z-index: 5; outline: 3px solid #ffd54a; transform: scale(1.6); }
#marker { position: absolute; transform: translate(-50%, -50%); z-index: 20; background: #ffd54a;
  color: #1f2a2e; padding: 4px 10px; border-radius: 999px; font-weight: 700; white-space: nowrap;
  box-shadow: 0 0 0 6px rgba(255,213,74,.3); pointer-events: none; }
.legend { padding: 4px 12px; font-size: 14px; border-radius: 999px; }
#legend { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
</style>
"""


@app.get("/map", response_class=HTMLResponse)
def map_page() -> str:
    import json

    script = MAP_SCRIPT.replace("%COLORS%", json.dumps(TAG_COLORS, ensure_ascii=False))
    body = f"""
      <h1>Kort over vektorrummet</h1>
      <p>Hvert billede er en vektor med hundredvis af tal ({esc(get_image_embed_model())}).
      t-SNE lægger dem ned i 2D, så billeder med lignende vektorer lander tæt på hinanden.
      Klynger betyder noget; akserne og afstande mellem klynger gør ikke.</p>
      <div class="panel">
        <form id="mapsearch">
          <input type="text" id="q" placeholder="Søg med tekst, fx 'a shipwreck' eller 'sunset'" />
          <button type="submit">Vis på kortet</button>
          <button type="button" id="reset">Nulstil</button>
        </form>
        <div id="legend"></div>
        <p class="hint" id="status"></p>
        <div id="stage"><div id="world"></div></div>
        <p class="hint">Søgeteksten placeres som et vægtet gennemsnit af dens bedste hits,
        fordi t-SNE ikke kan placere nye punkter præcist.</p>
      </div>
      <script>{script}</script>
    """
    return layout("Kort", body, active="/map", extra_head=MAP_STYLE)


# --------------------------------------------------------------------------- #
# Næsten-dubletter
# --------------------------------------------------------------------------- #

@app.get("/duplicates", response_class=HTMLResponse)
def duplicates(threshold: float = 0.95) -> str:
    threshold = max(0.5, min(0.999, threshold))
    try:
        photos = photo_store.fetch_all(get_client(), collection=default_collection())
    except Exception as exc:
        return layout("Dubletter", f"<p class='error'>Kunne ikke hente vektorer: {esc(exc)}</p>", active="/duplicates")
    pairs = find_duplicates([p.filename for p in photos], [p.vector for p in photos], threshold=threshold)

    rows = "".join(
        f"""
        <div class="pair">
          <a href="{image_url(a)}" target="_blank"><img src="{thumb_url(a)}" alt="{esc(a)}" loading="lazy" /></a>
          <a href="{image_url(b)}" target="_blank"><img src="{thumb_url(b)}" alt="{esc(b)}" loading="lazy" /></a>
          <div><div class="score">{score:.4f}</div><div class="hint">{esc(a)}<br>{esc(b)}</div></div>
        </div>
        """
        for a, b, score in pairs
    )
    body = f"""
      <h1>Næsten-dubletter</h1>
      <p>Alle {len(photos)} vektorer sammenlignes med hinanden
      ({len(photos) * (len(photos) - 1) // 2:,} par) med ét matrixprodukt. Par over tærsklen
      er typisk serieoptagelser af samme motiv.</p>
      <div class="panel">
        <form action="/duplicates" method="get">
          <label class="hint">Tærskel
            <input type="number" name="threshold" value="{threshold}" step="0.01" min="0.5" max="0.999" style="width:110px" />
          </label>
          <button type="submit">Find par</button>
        </form>
        <p class="hint">{len(pairs)} par over {threshold:.2f} (højst 60 vises).</p>
        {rows or "<p class='hint'>Ingen par over tærsklen. Prøv en lavere værdi.</p>"}
      </div>
    """
    return layout("Dubletter", body, active="/duplicates")


# --------------------------------------------------------------------------- #
# Sammenlign modeller
# --------------------------------------------------------------------------- #

@app.get("/compare", response_class=HTMLResponse)
def compare(q: str = "", file: str = "") -> str:
    client = get_client()
    models = [get_image_embed_model(), get_compare_embed_model()]
    q = q.strip()
    base_name = library_path(file).name if file else ""

    columns = []
    for model_name in models:
        collection = photo_store.collection_for_model(model_name)
        header = f"<h2>{esc(model_name)}</h2><p class='hint'>collection: {esc(collection)}</p>"
        if not photo_store.collection_exists(client, collection):
            columns.append(
                f"<div>{header}<p class='error'>Collectionen findes ikke endnu.</p>"
                f"<p>Kør fra repository-roden:</p><pre>python \"demoer/modul 3/photos/ingest_photos.py\" --model {esc(model_name)}</pre></div>"
            )
            continue
        if not (q or base_name):
            columns.append(f"<div>{header}<p class='hint'>Klar.</p></div>")
            continue
        try:
            if base_name:
                vector = photo_store.get_vector(client, base_name, collection=collection)
                if vector is None:
                    raise RuntimeError("billedet findes ikke i denne collection")
                results = photo_store.search(client, vector, collection=collection, exclude_filename=base_name)
            else:
                vector = text_to_vector(q, model_name)
                results = photo_store.search(client, vector, collection=collection)
            columns.append(
                f"<div>{header}<p class='hint'>{len(vector)} dimensioner</p>"
                f"{result_cards(results, links=False)}</div>"
            )
        except Exception as exc:
            columns.append(f"<div>{header}<p class='error'>Søgning fejlede: {esc(exc)}</p></div>")

    base_html = ""
    if base_name:
        base_html = (
            f"<p>Nærmeste naboer til <strong>{esc(base_name)}</strong>:</p>"
            f"<img src='{thumb_url(base_name)}' alt='{esc(base_name)}' style='max-width:160px;border-radius:8px' />"
        )

    body = f"""
      <h1>Sammenlign modeller</h1>
      <p>Samme søgning mod to CLIP-modeller, som hver har sin egen collection.
      Valget af embedding-model er en designbeslutning: større modeller ser flere detaljer,
      men er langsommere og kræver mere hukommelse.</p>
      <div class="panel">
        <form action="/compare" method="get">
          <input type="text" name="q" value="{esc(q)}" placeholder="fx 'a lionfish' eller 'two divers on a wreck'" />
          <button type="submit">Sammenlign</button>
        </form>
        {base_html}
        <div class="columns">{''.join(columns)}</div>
      </div>
    """
    return layout("Sammenlign modeller", body, active="/compare")
