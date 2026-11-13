from __future__ import annotations

import base64
import hashlib
import html
import io
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
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

from analysis import find_duplicates, mix_vectors, project, weighted_position
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
STATIC_DIR = Path(__file__).resolve().parent / "static"
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
# Three.js ligger lokalt, så 3D-kortet også virker uden internet i klasselokalet.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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


def tag_options(selected: str = "", *, autosubmit: bool = False) -> str:
    options = ["<option value=''>Alle tags</option>"]
    for tag in LABELS:
        chosen = " selected" if tag == selected else ""
        options.append(f"<option value='{esc(tag)}'{chosen}>{esc(tag)}</option>")
    auto = " data-autosubmit" if autosubmit else ""
    return f"<select name='tag'{auto}>" + "".join(options) + "</select>"


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
img.mini { width: 44px; height: 44px; object-fit: cover; border-radius: 8px; border: 1px solid var(--line); }
form.current { outline: 2px solid rgba(17,100,102,.35); outline-offset: 6px; border-radius: 12px; }
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

@dataclass
class UploadedImage:
    raw: bytes
    filename: str
    vector: list[float]


# Uploadede billeder huskes i hukommelsen, så et skift af tag kan søge igen uden at
# brugeren skal uploade (en browser kan ikke genudfylde et filfelt). Kun de seneste gemmes.
_UPLOADS: OrderedDict[str, UploadedImage] = OrderedDict()
MAX_UPLOADS = 32


def remember_upload(raw: bytes, filename: str, vector: list[float]) -> str:
    upload_id = hashlib.sha256(get_image_embed_model().encode() + raw).hexdigest()[:20]
    _UPLOADS[upload_id] = UploadedImage(raw=raw, filename=filename, vector=vector)
    _UPLOADS.move_to_end(upload_id)
    while len(_UPLOADS) > MAX_UPLOADS:
        _UPLOADS.popitem(last=False)
    return upload_id


def recall_upload(upload_id: str) -> UploadedImage | None:
    upload = _UPLOADS.get(upload_id)
    if upload is not None:
        _UPLOADS.move_to_end(upload_id)
    return upload


@dataclass
class SearchState:
    """Alt hvad søgesiden skal huske mellem to søgninger."""

    query: str = ""
    text_tag: str = ""
    upload_id: str = ""
    image_tag: str = ""


def render_page(
    *,
    state: SearchState | None = None,
    mode: str = "",
    results: list[dict] | None = None,
    error: str | None = None,
) -> str:
    state = state or SearchState()
    upload = recall_upload(state.upload_id) if state.upload_id else None
    if upload is None:
        state.upload_id = ""

    preview_html = ""
    if mode == "image" and upload is not None:
        with Image.open(io.BytesIO(upload.raw)) as pil_image:
            upload_view = model_view_html(pil_image, title="Det ser modellen")
        preview_html = (
            "<div class='row'>"
            "<div class='preview'><h2>Uploadet billede</h2>"
            f"<img src='/upload/{state.upload_id}' alt='upload' /></div>"
            f"{upload_view}"
            "</div>"
        )
    elif mode == "text" and state.query:
        preview_html = (
            "<div class='preview'>"
            f"<h2>Tekstsøgning: “{esc(state.query)}”</h2>"
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
            "<p class='hint'>Zero-shot tags fra CLIP (ingen træning). Vælg et tag i en af "
            "søgningerne for at filtrere den; resultatet opdateres med det samme.</p>"
            f"<div class='tags'>{chips}</div>"
        )

    active_tag = state.image_tag if mode == "image" else state.text_tag if mode == "text" else ""
    error_html = f"<p class='error'>{esc(error)}</p>" if error else ""
    if active_tag and results is not None:
        error_html += f"<p class='hint'>Filtreret på tag: {tag_badge(active_tag)}</p>"

    remembered_html = ""
    if upload is not None:
        remembered_html = (
            f"<img class='mini' src='/upload/{state.upload_id}' alt='husket billede' />"
            f"<span class='hint'>Husket: {esc(upload.filename)}</span>"
        )

    body = f"""
      <h1>Photo Similarity Demo</h1>
      <p>Upload et JPG-billede, eller beskriv et motiv med tekst, og se de 5 mest
      lignende billeder fra Qdrant med score.</p>
      <p class="hint">Model: {esc(get_image_embed_model())}</p>
      <div class="panel">
        {counts_html}
        <form action="/search" method="post" enctype="multipart/form-data" class="{'current' if mode == 'image' else ''}">
          {remembered_html}
          <input type="file" name="image" accept=".jpg,.jpeg" {'' if upload else 'required'} />
          <input type="hidden" name="upload_id" value="{esc(state.upload_id)}" />
          <input type="hidden" name="query" value="{esc(state.query)}" />
          <input type="hidden" name="text_tag" value="{esc(state.text_tag)}" />
          {tag_options(state.image_tag, autosubmit=True)}
          <button type="submit">Find lignende billeder</button>
        </form>
        <form action="/search-text" method="post" class="{'current' if mode == 'text' else ''}">
          <input type="text" name="query" value="{esc(state.query)}"
                 placeholder="fx 'a shipwreck' eller 'solnedgang over havet'" required />
          <input type="hidden" name="upload_id" value="{esc(state.upload_id)}" />
          <input type="hidden" name="image_tag" value="{esc(state.image_tag)}" />
          {tag_options(state.text_tag, autosubmit=True)}
          <button type="submit">Søg med tekst</button>
        </form>
        {error_html}
        {preview_html}
        {result_cards(results) if results is not None else ""}
      </div>
      <script>{AUTOSUBMIT_SCRIPT}</script>
    """
    return layout("Søg", body, active="/")


# Skift af tag søger igen, når formularen har noget at søge med.
AUTOSUBMIT_SCRIPT = """
document.querySelectorAll('select[data-autosubmit]').forEach((select) => {
  select.addEventListener('change', () => {
    const form = select.form;
    const file = form.querySelector('input[type=file]');
    const uploadId = form.querySelector('input[name=upload_id]');
    const query = form.querySelector('input[type=text][name=query]');
    const ready = file
      ? (file.files.length > 0 || (uploadId && uploadId.value))
      : (query && query.value.trim());
    if (ready) form.requestSubmit();
  });
});
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_page()


@app.get("/upload/{upload_id}")
def uploaded_image(upload_id: str) -> Response:
    upload = recall_upload(upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Billedet er ikke længere husket")
    return Response(content=upload.raw, media_type="image/jpeg")


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    image: UploadFile | None = File(None),
    upload_id: str = Form(""),
    tag: str = Form(""),
    query: str = Form(""),
    text_tag: str = Form(""),
) -> str:
    state = SearchState(query=query.strip(), text_tag=text_tag, upload_id=upload_id, image_tag=tag)

    if image is not None and image.filename:
        # Nyt billede: embed det og husk det til senere tag-skift.
        if Path(image.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return render_page(state=state, error="Upload kun .jpg eller .jpeg filer.")
        raw = await image.read()
        try:
            with Image.open(io.BytesIO(raw)) as pil_image:
                vector = image_to_vector(pil_image)
        except UnidentifiedImageError:
            return render_page(state=state, error="Filen kunne ikke læses som et billede.")
        state.upload_id = remember_upload(raw, Path(image.filename).name, vector)
    else:
        # Intet nyt billede: genbrug det huskede (fx når tagget skiftes).
        upload = recall_upload(upload_id) if upload_id else None
        if upload is None:
            return render_page(state=state, error="Vælg et billede at uploade.")
        vector = upload.vector

    try:
        results = photo_store.search(get_client(), vector, collection=default_collection(), tag=tag or None)
    except Exception as exc:
        return render_page(
            state=state,
            mode="image",
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(state=state, mode="image", results=results)


@app.post("/search-text", response_class=HTMLResponse)
def search_text(
    query: str = Form(...),
    tag: str = Form(""),
    upload_id: str = Form(""),
    image_tag: str = Form(""),
) -> str:
    state = SearchState(query=query.strip(), text_tag=tag, upload_id=upload_id, image_tag=image_tag)
    if not state.query:
        return render_page(state=state, error="Skriv en tekst at søge efter.")

    # CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren kan
    # søges direkte mod de billedvektorer, som ingest_photos.py har gemt.
    vector = text_to_vector(state.query)
    try:
        results = photo_store.search(get_client(), vector, collection=default_collection(), tag=tag or None)
    except Exception as exc:
        return render_page(
            state=state,
            mode="text",
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(state=state, mode="text", results=results)


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

_MAP_CACHE: dict[tuple, dict] = {}
MAP_MODES = {"2d-tsne": (2, "tsne"), "3d-tsne": (3, "tsne"), "3d-pca": (3, "pca")}


def map_data(dims: int = 2, method: str = "tsne") -> dict:
    """Hent alle vektorer og projicér dem. Resultatet caches pr. collection, antal og metode."""
    collection = default_collection()
    photos = photo_store.fetch_all(get_client(), collection=collection)
    key = (collection, len(photos), dims, method)
    if key not in _MAP_CACHE:
        if any(cached[:2] != key[:2] for cached in _MAP_CACHE):
            _MAP_CACHE.clear()
        projection = project([photo.vector for photo in photos], dims=dims, method=method)
        _MAP_CACHE[key] = {
            "projection": projection,
            "payload": {
                "dims": dims,
                "method": method,
                "explained": projection.explained_variance,
                "points": [
                    {"filename": photo.filename, "tag": photo.tag, "pos": list(pos)}
                    for photo, pos in zip(photos, projection.coords)
                ],
            },
        }
    return _MAP_CACHE[key]


def map_mode(mode: str) -> tuple[int, str]:
    if mode not in MAP_MODES:
        raise HTTPException(status_code=400, detail=f"Ukendt visning: {mode}")
    return MAP_MODES[mode]


@app.get("/api/map")
def api_map(mode: str = "2d-tsne") -> JSONResponse:
    dims, method = map_mode(mode)
    try:
        return JSONResponse(map_data(dims, method)["payload"])
    except Exception as exc:
        return JSONResponse({"error": f"Kunne ikke lave kortet: {exc}"}, status_code=500)


@app.get("/api/map/search")
def api_map_search(q: str, k: int = 10, mode: str = "2d-tsne") -> JSONResponse:
    dims, method = map_mode(mode)
    data = map_data(dims, method)
    positions = {point["filename"]: tuple(point["pos"]) for point in data["payload"]["points"]}
    query_vector = text_to_vector(q)
    hits = photo_store.search(get_client(), query_vector, collection=default_collection(), limit=max(1, min(k, 50)))

    # PCA er lineær, så søgeteksten kan placeres præcist. t-SNE kan ikke placere nye
    # punkter; der bruges et vægtet gennemsnit af de bedste hits (skarpe vægte, som
    # CLIPs egen softmax, så det bedste hit trækker mest).
    exact = data["projection"].transform(query_vector)
    if exact is not None:
        position = exact
    else:
        top = max((hit["score"] for hit in hits), default=0.0)
        weights = [math.exp(100 * (hit["score"] - top)) for hit in hits]
        known = [(positions[hit["filename"]], w) for hit, w in zip(hits, weights) if hit["filename"] in positions]
        center = (0.5,) * dims if dims == 2 else (0.0,) * dims
        position = weighted_position([p for p, _ in known], [w for _, w in known]) if known else center
    return JSONResponse({"hits": hits, "position": list(position), "exact": exact is not None})


MAP_SCRIPT = r"""
const COLORS = %COLORS%;
const statusEl = document.getElementById('status');
const legendEl = document.getElementById('legend');
const stage2d = document.getElementById('stage2d');
const stage3d = document.getElementById('stage3d');
const autoRow = document.getElementById('autorow');
const autoBox = document.getElementById('autorotate');
const views = {};
let mode = '2d-tsne', query = null, activeTag = null, info = {};

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || data.detail || res.statusText);
  return data;
}

function baseStatus(data) {
  let text = `${data.points.length} billeder.`;
  if (data.method === 'pca' && data.explained) {
    const parts = data.explained.map((v) => Math.round(v * 100) + '%');
    const total = Math.round(data.explained.reduce((a, b) => a + b, 0) * 100);
    text += ` PCA-akserne viser ${parts.join(' + ')} = ${total}% af variationen i de oprindelige vektorer.`;
  }
  text += data.dims === 3
    ? ' Træk for at rotere, scroll for at zoome, højreklik/shift-træk for at flytte. Klik på et billede.'
    : ' Scroll for at zoome, træk for at flytte, klik på et billede.';
  return text;
}

function buildLegend(points) {
  if (legendEl.childElementCount) return;
  const counts = {};
  for (const p of points) counts[p.tag] = (counts[p.tag] || 0) + 1;
  for (const [tag, color] of Object.entries(COLORS)) {
    if (!counts[tag]) continue;
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'legend'; b.style.background = color;
    b.textContent = `${tag} (${counts[tag]})`;
    b.onclick = () => { activeTag = activeTag === tag ? null : tag; query = null; refresh(); };
    legendEl.appendChild(b);
  }
}

// ---------------------------------------------------------------- 2D
function create2D(el, data) {
  const world = document.createElement('div');
  world.className = 'world';
  el.appendChild(world);
  const THUMB = 44;
  let scale = 1, tx = 0, ty = 0, drag = null;
  const nodes = {};
  const apply = () => { world.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`; };
  const place = (x, y) => [x * (el.clientWidth - THUMB), y * (el.clientHeight - THUMB)];

  el.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = el.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const next = Math.min(10, Math.max(0.5, scale * Math.exp(-e.deltaY * 0.0015)));
    tx = mx - (mx - tx) * next / scale; ty = my - (my - ty) * next / scale; scale = next; apply();
  }, { passive: false });
  el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX - tx, y: e.clientY - ty, moved: false }; });
  window.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const nx = e.clientX - drag.x, ny = e.clientY - drag.y;
    if (Math.abs(nx - tx) + Math.abs(ny - ty) > 2) drag.moved = true;
    tx = nx; ty = ny; apply();
  });
  window.addEventListener('pointerup', () => { setTimeout(() => { drag = null; }, 0); });

  for (const p of data.points) {
    const [left, top] = place(p.pos[0], p.pos[1]);
    const a = document.createElement('a');
    a.href = '/model-view?file=' + encodeURIComponent(p.filename);
    a.className = 'pt'; a.dataset.tag = p.tag || '';
    a.style.left = left + 'px'; a.style.top = top + 'px';
    a.style.borderColor = COLORS[p.tag] || '#888';
    a.title = p.filename + (p.tag ? ' · ' + p.tag : '');
    a.addEventListener('click', (e) => { if (drag && drag.moved) e.preventDefault(); });
    const img = document.createElement('img');
    img.src = '/thumb/' + encodeURIComponent(p.filename); img.loading = 'lazy'; img.alt = p.filename;
    a.appendChild(img); world.appendChild(a); nodes[p.filename] = a;
  }

  return {
    show() { el.hidden = false; }, hide() { el.hidden = true; },
    reset() { scale = 1; tx = 0; ty = 0; apply(); },
    highlight({ hits, tag, marker }) {
      for (const [name, node] of Object.entries(nodes)) {
        const isHit = hits ? hits.has(name) : false;
        const dim = (hits && !isHit) || (tag && node.dataset.tag !== tag);
        node.classList.toggle('dim', Boolean(dim));
        node.classList.toggle('hit', isHit);
      }
      const old = world.querySelector('.marker'); if (old) old.remove();
      if (marker) {
        const [left, top] = place(marker.pos[0], marker.pos[1]);
        const m = document.createElement('div');
        m.className = 'marker'; m.textContent = '“' + marker.label + '”';
        m.style.left = (left + THUMB / 2) + 'px'; m.style.top = (top + THUMB / 2) + 'px';
        world.appendChild(m);
      }
    },
  };
}

// ---------------------------------------------------------------- 3D
async function create3D(el, data) {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  el.appendChild(renderer.domElement);
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip'; tooltip.hidden = true; el.appendChild(tooltip);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#0d2b33');
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  const HOME = new THREE.Vector3(0, 0.4, 2.6);
  camera.position.copy(HOME);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = autoBox.checked;
  controls.autoRotateSpeed = 0.7;
  controls.minDistance = 0.2; controls.maxDistance = 10;
  controls.addEventListener('start', () => { controls.autoRotate = false; autoBox.checked = false; });
  let targetGoal = new THREE.Vector3();

  // Et svagt omrids af rummet giver dybdefornemmelse, når kortet roteres.
  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
    new THREE.LineBasicMaterial({ color: 0x3a6670, transparent: true, opacity: 0.5 }),
  );
  scene.add(box);

  if (data.method === 'pca') {
    const axes = [[1, 0, 0, '#e57373'], [0, 1, 0, '#81c784'], [0, 0, 1, '#64b5f6']];
    axes.forEach(([x, y, z, color], i) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-x, -y, -z), new THREE.Vector3(x, y, z)]);
      scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color })));
      const pct = data.explained ? ` (${Math.round(data.explained[i] * 100)}%)` : '';
      const label = textSprite(THREE, `PC${i + 1}${pct}`, color, '#0d2b33');
      label.position.set(x * 1.15, y * 1.15, z * 1.15);
      label.scale.multiplyScalar(0.6);
      scene.add(label);
    });
  }

  const SIZE = 0.085;
  const sprites = [];
  const byName = {};
  for (const p of data.points) {
    const material = new THREE.SpriteMaterial({ color: COLORS[p.tag] || '#888888', transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(p.pos[0], p.pos[1], p.pos[2]);
    sprite.scale.set(SIZE, SIZE, 1);
    sprite.userData = p;
    scene.add(sprite); sprites.push(sprite); byName[p.filename] = sprite;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = c.height = 96;
      const ctx = c.getContext('2d');
      ctx.fillStyle = COLORS[p.tag] || '#888888'; ctx.fillRect(0, 0, 96, 96);
      const s = Math.min(img.width, img.height);
      ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 6, 6, 84, 84);
      const texture = new THREE.CanvasTexture(c);
      texture.colorSpace = THREE.SRGBColorSpace;
      material.map = texture; material.color.set('#ffffff'); material.needsUpdate = true;
    };
    img.src = '/thumb/' + encodeURIComponent(p.filename);
  }

  const markerGroup = new THREE.Group();
  scene.add(markerGroup);

  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(el);
  resize();

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hovered = null, downAt = null;
  renderer.domElement.addEventListener('pointermove', (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(sprites).find((h) => h.object.material.opacity > 0.5);
    hovered = hit ? hit.object : null;
    renderer.domElement.style.cursor = hovered ? 'pointer' : 'grab';
    if (hovered) {
      const p = hovered.userData;
      tooltip.hidden = false;
      tooltip.innerHTML = '';
      const img = document.createElement('img'); img.src = '/thumb/' + encodeURIComponent(p.filename);
      const label = document.createElement('div'); label.textContent = p.filename + (p.tag ? ' · ' + p.tag : '');
      tooltip.append(img, label);
      tooltip.style.left = Math.min(e.clientX - r.left + 14, r.width - 180) + 'px';
      tooltip.style.top = Math.min(e.clientY - r.top + 14, r.height - 200) + 'px';
    } else {
      tooltip.hidden = true;
    }
  });
  renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!downAt || !hovered) return;
    const moved = Math.abs(e.clientX - downAt[0]) + Math.abs(e.clientY - downAt[1]);
    if (moved < 5) window.location.href = '/model-view?file=' + encodeURIComponent(hovered.userData.filename);
  });
  renderer.domElement.addEventListener('pointerleave', () => { tooltip.hidden = true; hovered = null; });

  let visible = true;
  (function animate() {
    requestAnimationFrame(animate);
    if (!visible) return;
    controls.target.lerp(targetGoal, 0.08);
    controls.update();
    renderer.render(scene, camera);
  })();

  return {
    show() {
      el.hidden = false; visible = true; autoRow.hidden = false;
      autoBox.onchange = () => { controls.autoRotate = autoBox.checked; };
      controls.autoRotate = autoBox.checked;
      resize();
    },
    hide() { el.hidden = true; visible = false; autoRow.hidden = true; },
    reset() { camera.position.copy(HOME); targetGoal.set(0, 0, 0); },
    highlight({ hits, tag, marker }) {
      for (const sprite of sprites) {
        const name = sprite.userData.filename;
        const isHit = hits ? hits.has(name) : false;
        const dim = (hits && !isHit) || (tag && sprite.userData.tag !== tag);
        sprite.material.opacity = dim ? 0.1 : 1;
        sprite.material.depthWrite = !dim;
        const s = isHit ? SIZE * 2 : SIZE;
        sprite.scale.set(s, s, 1);
        sprite.renderOrder = isHit ? 10 : 0;
      }
      markerGroup.clear();
      targetGoal.set(0, 0, 0);
      if (marker) {
        const pos = new THREE.Vector3(marker.pos[0], marker.pos[1], marker.pos[2]);
        const label = textSprite(THREE, '“' + marker.label + '”', '#ffd54a', '#1f2a2e');
        label.position.copy(pos);
        markerGroup.add(label);
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffd54a, transparent: true, opacity: 0.7 });
        for (const name of hits || []) {
          if (!byName[name]) continue;
          const geometry = new THREE.BufferGeometry().setFromPoints([pos, byName[name].position]);
          markerGroup.add(new THREE.Line(geometry, lineMaterial));
        }
        targetGoal.copy(pos);
      }
    },
  };
}

function textSprite(THREE, text, background, color) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = 'bold 44px Georgia, serif';
  const w = Math.ceil(ctx.measureText(text).width) + 48;
  c.width = w; c.height = 72;
  ctx.font = 'bold 44px Georgia, serif';
  ctx.fillStyle = background; ctx.beginPath(); ctx.roundRect(0, 0, w, 72, 36); ctx.fill();
  ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 24, 38);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.renderOrder = 999;
  const height = 0.11;
  sprite.scale.set(height * w / 72, height, 1);
  return sprite;
}

// ---------------------------------------------------------------- styring
async function refresh() {
  const view = views[mode];
  if (!view) return;
  if (!query) {
    view.highlight({ hits: null, tag: activeTag, marker: null });
    statusEl.textContent = baseStatus(info[mode]);
    return;
  }
  statusEl.textContent = 'Søger …';
  try {
    const data = await getJSON('/api/map/search?' + new URLSearchParams({ q: query, k: 10, mode }));
    const hits = new Set(data.hits.map((h) => h.filename));
    view.highlight({ hits, tag: null, marker: { pos: data.position, label: query } });
    const best = data.hits[0] ? data.hits[0].score.toFixed(3) : '-';
    const where = data.exact
      ? 'Teksten er placeret præcist med PCA – læg mærke til afstanden til billederne (the modality gap).'
      : 'Teksten er placeret ved sine bedste hits, fordi t-SNE ikke kan placere nye punkter.';
    statusEl.textContent = `Top ${data.hits.length} for “${query}” er fremhævet (bedste score ${best}). ${where}`;
  } catch (err) {
    statusEl.textContent = 'Søgning fejlede: ' + err.message;
  }
}

async function setMode(next) {
  document.querySelectorAll('.modes button').forEach((b) => b.classList.toggle('active', b.dataset.mode === next));
  if (views[mode]) views[mode].hide();
  mode = next;
  if (!views[mode]) {
    statusEl.textContent = next.endsWith('tsne') ? 'Beregner t-SNE over alle vektorer …' : 'Beregner PCA …';
    try {
      const data = await getJSON('/api/map?mode=' + mode);
      info[mode] = data;
      buildLegend(data.points);
      const el = mode.startsWith('3d') ? stage3d.cloneNode(false) : stage2d;
      if (mode.startsWith('3d')) { el.id = 'stage-' + mode; stage3d.after(el); }
      views[mode] = mode.startsWith('3d') ? await create3D(el, data) : create2D(el, data);
    } catch (err) {
      statusEl.textContent = 'Kunne ikke lave kortet: ' + err.message;
      return;
    }
    if (next !== mode) return;
  }
  views[mode].show();
  refresh();
}

document.querySelectorAll('.modes button').forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
document.getElementById('mapsearch').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = document.getElementById('q').value.trim();
  query = q || null; activeTag = null; refresh();
});
document.getElementById('reset').onclick = () => {
  query = null; activeTag = null; document.getElementById('q').value = '';
  if (views[mode]) views[mode].reset();
  refresh();
};
setMode('2d-tsne');
"""

MAP_STYLE = """
<script type="importmap">
{ "imports": { "three": "/static/three/three.module.min.js", "three/addons/": "/static/three/addons/" } }
</script>
<style>
.stage { position: relative; height: 72vh; min-height: 420px; overflow: hidden; border-radius: 14px;
  background: #0d2b33; cursor: grab; touch-action: none; }
.stage[hidden] { display: none; }
.stage canvas { display: block; width: 100%; height: 100%; }
.world { position: absolute; inset: 0; transform-origin: 0 0; }
.pt { position: absolute; width: 44px; height: 44px; border: 2px solid #888; border-radius: 6px;
  overflow: hidden; transition: opacity .2s, transform .2s; background: #123; }
.pt img { width: 100%; height: 100%; object-fit: cover; display: block; pointer-events: none; }
.pt:hover { transform: scale(2.6); z-index: 10; }
.pt.dim { opacity: .12; }
.pt.hit { opacity: 1; z-index: 5; outline: 3px solid #ffd54a; transform: scale(1.6); }
.marker { position: absolute; transform: translate(-50%, -50%); z-index: 20; background: #ffd54a;
  color: #1f2a2e; padding: 4px 10px; border-radius: 999px; font-weight: 700; white-space: nowrap;
  box-shadow: 0 0 0 6px rgba(255,213,74,.3); pointer-events: none; }
.tooltip { position: absolute; z-index: 30; background: white; border-radius: 10px; padding: 6px;
  font-size: 13px; box-shadow: 0 6px 20px rgba(0,0,0,.3); pointer-events: none; max-width: 170px; }
.tooltip img { width: 158px; height: 158px; object-fit: cover; display: block; border-radius: 6px; margin-bottom: 4px; }
.legend { padding: 4px 12px; font-size: 14px; border-radius: 999px; }
#legend { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; }
.modes { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin-bottom: 12px; }
.modes button { border-radius: 0; background: white; color: var(--ink); padding: 8px 16px; }
.modes button.active { background: var(--accent); color: white; }
#autorow { margin: 0 0 8px; }
</style>
"""


@app.get("/map", response_class=HTMLResponse)
def map_page() -> str:
    import json

    script = MAP_SCRIPT.replace("%COLORS%", json.dumps(TAG_COLORS, ensure_ascii=False))
    body = f"""
      <h1>Kort over vektorrummet</h1>
      <p>Hvert billede er en vektor med hundredvis af tal ({esc(get_image_embed_model())}).
      Kortet lægger dem ned i 2 eller 3 dimensioner, så billeder med lignende vektorer lander
      tæt på hinanden.</p>
      <p class="hint"><strong>t-SNE</strong> bevarer naboskaber: klynger betyder noget, akser og
      afstande mellem klynger gør ikke. <strong>PCA</strong> er en lineær projektion: akserne er
      retningerne med mest variation, og afstande betyder noget, men klyngerne smelter mere sammen.</p>
      <div class="panel">
        <div class="modes">
          <button type="button" data-mode="2d-tsne">2D t-SNE</button>
          <button type="button" data-mode="3d-tsne">3D t-SNE</button>
          <button type="button" data-mode="3d-pca">3D PCA</button>
        </div>
        <form id="mapsearch">
          <input type="text" id="q" placeholder="Søg med tekst, fx 'a shipwreck' eller 'sunset'" />
          <button type="submit">Vis på kortet</button>
          <button type="button" id="reset">Nulstil</button>
        </form>
        <div id="legend"></div>
        <label id="autorow" class="hint" hidden><input type="checkbox" id="autorotate" checked /> Rotér automatisk</label>
        <p class="hint" id="status"></p>
        <div class="stage" id="stage2d"></div>
        <div class="stage" id="stage3d" hidden></div>
      </div>
      <script type="module">{script}</script>
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
