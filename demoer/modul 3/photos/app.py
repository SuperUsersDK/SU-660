from __future__ import annotations

import base64
import html
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from image_embeddings import get_image_embed_model, image_to_vector, text_to_vector

DATA_DIR = Path(__file__).resolve().parent / "data"
COLLECTION_NAME = "modul3_photos_demo"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


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


def render_page(
    *,
    uploaded_image_data_url: str | None = None,
    query_text: str | None = None,
    results: list[dict[str, str | float]] | None = None,
    error: str | None = None,
) -> str:
    results_html = ""
    if results:
        cards = []
        for item in results:
            cards.append(
                f"""
                <div class="card">
                  <img src="/data/{item['filename']}" alt="{item['filename']}" />
                  <div class="meta">
                    <div class="name">{item['filename']}</div>
                    <div class="score">score: {item['score']:.4f}</div>
                  </div>
                </div>
                """
            )
        results_html = "<div class='grid'>" + "".join(cards) + "</div>"

    preview_html = ""
    if uploaded_image_data_url:
        preview_html = (
            "<div class='preview'>"
            "<h2>Uploadet billede</h2>"
            f"<img src='{uploaded_image_data_url}' alt='upload' />"
            "</div>"
        )
    elif query_text:
        preview_html = (
            "<div class='preview'>"
            f"<h2>Tekstsøgning: “{html.escape(query_text)}”</h2>"
            "<p class='hint'>Tekst-til-billede scores ligger typisk omkring 0,2–0,35; "
            "rækkefølgen er det vigtige.</p>"
            "</div>"
        )

    error_html = f"<p class='error'>{html.escape(error)}</p>" if error else ""
    query_value = html.escape(query_text or "", quote=True)

    return f"""
    <!doctype html>
    <html lang="da">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Photo Similarity Demo</title>
        <style>
          :root {{
            --bg: #f3efe7;
            --panel: #fffaf2;
            --ink: #1f2a2e;
            --accent: #116466;
            --accent-2: #d96c06;
            --line: #d8cfc0;
          }}
          body {{
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            background: radial-gradient(circle at top, #fff8ec, var(--bg));
            color: var(--ink);
          }}
          main {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 32px 20px 48px;
          }}
          h1 {{
            margin-bottom: 8px;
            font-size: 40px;
          }}
          p {{
            line-height: 1.5;
          }}
          .panel {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
          }}
          form {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
          }}
          form + form {{
            margin-top: 12px;
          }}
          input[type="file"] {{
            font-size: 16px;
          }}
          input[type="text"] {{
            flex: 1;
            min-width: 220px;
            font-size: 16px;
            padding: 10px 14px;
            border: 1px solid var(--line);
            border-radius: 999px;
          }}
          .hint {{
            color: #5b6366;
            font-size: 14px;
          }}
          button {{
            border: 0;
            background: linear-gradient(135deg, var(--accent), #1d7874);
            color: white;
            border-radius: 999px;
            padding: 12px 18px;
            cursor: pointer;
            font-size: 16px;
          }}
          .error {{
            color: #b42318;
            font-weight: 700;
          }}
          .preview img {{
            max-width: 280px;
            border-radius: 12px;
            border: 1px solid var(--line);
          }}
          .grid {{
            margin-top: 24px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
          }}
          .card {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
          }}
          .card img {{
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            display: block;
          }}
          .meta {{
            padding: 12px;
          }}
          .name {{
            font-weight: 700;
            margin-bottom: 6px;
          }}
          .score {{
            color: var(--accent-2);
          }}
        </style>
      </head>
      <body>
        <main>
          <h1>Photo Similarity Demo</h1>
          <p>Upload et JPG-billede, eller beskriv et motiv med tekst, og se de 5 mest
          lignende billeder fra Qdrant med score.</p>
          <p>Model: {get_image_embed_model()}</p>
          <div class="panel">
            <form action="/search" method="post" enctype="multipart/form-data">
              <input type="file" name="image" accept=".jpg,.jpeg" required />
              <button type="submit">Find lignende billeder</button>
            </form>
            <form action="/search-text" method="post">
              <input type="text" name="query" value="{query_value}"
                     placeholder="fx 'a dog on the beach' eller 'solnedgang over havet'" required />
              <button type="submit">Søg med tekst</button>
            </form>
            {error_html}
            {preview_html}
            {results_html}
          </div>
        </main>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return render_page()


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, image: UploadFile = File(...)) -> str:
    if Path(image.filename or "").suffix.lower() not in SUPPORTED_EXTENSIONS:
        return render_page(error="Upload kun .jpg eller .jpeg filer.")

    raw = await image.read()
    try:
        with Image.open(io.BytesIO(raw)) as pil_image:
            vector = image_to_vector(pil_image)
    except UnidentifiedImageError:
        return render_page(error="Filen kunne ikke læses som et billede.")

    data_url = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    try:
        results = find_similar(vector)
    except Exception as exc:
        return render_page(
            uploaded_image_data_url=data_url,
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(uploaded_image_data_url=data_url, results=results)


@app.post("/search-text", response_class=HTMLResponse)
def search_text(query: str = Form(...)) -> str:
    query = query.strip()
    if not query:
        return render_page(error="Skriv en tekst at søge efter.")

    # CLIP lægger tekst og billeder i samme vektorrum, så tekstvektoren kan
    # søges direkte mod de billedvektorer, som ingest_photos.py har gemt.
    vector = text_to_vector(query)
    try:
        results = find_similar(vector)
    except Exception as exc:
        return render_page(
            query_text=query,
            error=f"Søgning fejlede. Har du kørt ingest_photos.py først? Detalje: {exc}",
        )

    return render_page(query_text=query, results=results)


def find_similar(vector: list[float]) -> list[dict[str, str | float]]:
    response = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True,
    )

    results: list[dict[str, str | float]] = []
    for hit in response.points:
        payload = hit.payload or {}
        filename = str(payload.get("filename", "ukendt.jpg"))
        results.append(
            {
                "filename": filename,
                "score": float(hit.score),
            }
        )
    return results
