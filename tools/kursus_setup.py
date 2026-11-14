"""Klargør en maskine til kurset: find ud af, hvad der mangler, og hent det.

Kør fra repository-roden med .venv aktiveret:

    python tools/kursus_setup.py          # menu
    python tools/kursus_setup.py scan     # kun scanning
    python tools/kursus_setup.py alt      # hent og indlæs alt, der mangler

Menuen kan også få et nummer direkte, fx `python tools/kursus_setup.py 3`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import BOLD, CYAN, DIM, GREEN, RED, YELLOW, colorize

PHOTOS_DIR = ROOT_DIR / "demoer" / "modul 3" / "photos"
PHOTO_DATA_DIR = PHOTOS_DIR / "data"
CHUNK_LAB_DIR = ROOT_DIR / "demoer" / "modul 2" / "chunking_lab"
BIG_DEMO_DIR = ROOT_DIR / "demoer" / "modul 2" / "big_demo"
ENV_FILE = ROOT_DIR / ".env"
ENV_EXAMPLE = ROOT_DIR / ".env.example"

E5_MODEL = "intfloat/multilingual-e5-small"
CLIP_BASE = "openai/clip-vit-base-patch32"
CLIP_LARGE = "openai/clip-vit-large-patch14"
TIKTOKEN_FILES = {
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
}
REQUIRED_PACKAGES = {
    "dotenv": "python-dotenv", "openai": "openai", "langchain": "langchain", "qdrant_client": "qdrant-client",
    "transformers": "transformers", "torch": "torch", "sklearn": "scikit-learn", "tiktoken": "tiktoken",
    "fastapi": "fastapi", "uvicorn": "uvicorn", "PIL": "pillow", "multipart": "python-multipart",
}
SAMPLE_PHOTOS_EXPECTED = 200  # 20 billeder i 10 kategorier

OK, MISSING, OPTIONAL = "ok", "mangler", "valgfri"


# --------------------------------------------------------------------------- #
# Hjælpere
# --------------------------------------------------------------------------- #

def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        # Pakkerne er ikke installeret endnu; læs de simple KEY=VALUE-linjer selv,
        # så scanningen stadig kan se nøglerne.
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() and not key.lstrip().startswith("#"):
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        return
    load_dotenv(ENV_FILE, override=True)


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")


def qdrant_get(path: str, timeout: float = 3) -> dict | None:
    """GET mod Qdrants REST-API. None hvis Qdrant ikke svarer."""
    import json

    request = urllib.request.Request(qdrant_url() + path, headers={"api-key": os.getenv("QDRANT_API_KEY", "")})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"status": exc.code}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def collection_points(name: str) -> int | None:
    """Antal punkter i en collection; 0 hvis den ikke findes, None hvis Qdrant er nede."""
    data = qdrant_get(f"/collections/{name}")
    if data is None:
        return None
    return int(data.get("result", {}).get("points_count") or 0) if "result" in data else 0


def hf_cached(repo_id: str) -> bool:
    """Er modellen hentet til Hugging Face-cachen (config + vægte)?"""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ModuleNotFoundError:
        return False

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(repo_id, filename), str)

    return cached("config.json") and (cached("model.safetensors") or cached("pytorch_model.bin"))


def tiktoken_cache_dir() -> Path:
    from chunking.tokenizer_cache import configure_tokenizer_cache

    return Path(configure_tokenizer_cache())


def tiktoken_missing() -> list[str]:
    """tiktoken gemmer hver fil under sha1 af dens URL."""
    cache = tiktoken_cache_dir()
    return [name for name, url in TIKTOKEN_FILES.items() if not (cache / hashlib.sha1(url.encode()).hexdigest()).exists()]


def photo_files() -> list[Path]:
    if not PHOTO_DATA_DIR.is_dir():
        return []
    return [p for p in PHOTO_DATA_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]


def sample_photo_count() -> int:
    tags = ("bil", "fly", "skib", "fugl", "insekt", "blomst", "svamp", "kirke", "mad", "bjerg")
    return sum(1 for p in photo_files() if p.name.split("_", 1)[0] in tags)


def env_value_set(key: str) -> bool:
    value = os.getenv(key, "").strip()
    return bool(value) and value.lower() not in {"din-nøgle", "your-key", "sk-..."}


def run(cmd: list[str], cwd: Path = ROOT_DIR) -> bool:
    """Kør en kommando med live output. Returnér True ved succes."""
    print(colorize("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), DIM), flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd).returncode == 0
    except FileNotFoundError as exc:
        print(colorize(f"Kunne ikke starte: {exc}", RED))
        return False
    except KeyboardInterrupt:
        print(colorize("\nAfbrudt.", YELLOW))
        return False


def python_script(path: Path, *args: str) -> bool:
    return run([sys.executable, str(path), *args])


def header(title: str) -> None:
    print()
    print(colorize(f"── {title} " + "─" * max(0, 60 - len(title)), BOLD, CYAN))


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""  # menupunkt eller kommando, der løser det


def scan(*, check_openai: bool = True) -> list[Check]:
    load_env()
    checks: list[Check] = []

    # Python og pakker
    version = sys.version_info
    checks.append(Check(
        "Python 3.12", OK if version[:2] == (3, 12) else OPTIONAL,
        f"{version.major}.{version.minor}.{version.micro}" + ("" if version[:2] == (3, 12) else " (kurset er testet med 3.12)"),
    ))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checks.append(Check("Virtuelt miljø (.venv)", OK if in_venv else MISSING,
                        "aktivt" if in_venv else "kører med systemets Python",
                        "aktivér .venv (se README)"))
    missing_packages = [pip for module, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
    checks.append(Check("Python-pakker", OK if not missing_packages else MISSING,
                        "alle installeret" if not missing_packages else "mangler: " + ", ".join(missing_packages),
                        "pip install -r requirements.txt -c constraints-classroom.txt"))

    # .env
    checks.append(Check(".env", OK if ENV_FILE.exists() else MISSING,
                        "findes" if ENV_FILE.exists() else "findes ikke", "menupunkt 9 opretter den fra .env.example"))
    checks.append(Check("OPENAI_API_KEY", OK if env_value_set("OPENAI_API_KEY") else MISSING,
                        "sat" if env_value_set("OPENAI_API_KEY") else "ikke sat i .env", "skriv nøglen i .env"))
    checks.append(Check("QDRANT_API_KEY", OK if env_value_set("QDRANT_API_KEY") else MISSING,
                        "sat" if env_value_set("QDRANT_API_KEY") else "ikke sat i .env (skal være 'hemmelighed')",
                        "skriv QDRANT_API_KEY=hemmelighed i .env"))

    # Qdrant
    collections = qdrant_get("/collections")
    if collections is None:
        checks.append(Check("Qdrant", MISSING, f"svarer ikke på {qdrant_url()}", "2"))
    elif collections.get("status") == 401 or collections.get("status") == 403:
        checks.append(Check("Qdrant", MISSING, "afviser nøglen (401)", "ret QDRANT_API_KEY i .env"))
    else:
        checks.append(Check("Qdrant", OK, f"kører på {qdrant_url()}"))

    # OpenAI (gratis kald: slå modellen op)
    if check_openai and env_value_set("OPENAI_API_KEY") and importlib.util.find_spec("openai"):
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
        try:
            from openai import OpenAI

            OpenAI(timeout=15).models.retrieve(model)
            checks.append(Check("OpenAI-adgang", OK, f"nøglen virker, og {model} er tilgængelig"))
        except Exception as exc:
            reason = getattr(exc, "message", None) or str(exc)
            checks.append(Check("OpenAI-adgang", MISSING, f"{model}: {reason[:90]}", "tjek nøglen og modeladgang"))

    # Tokenizere og lokale modeller
    if importlib.util.find_spec("tiktoken"):
        missing_tok = tiktoken_missing()
        checks.append(Check("Tokenizere (tiktoken)", OK if not missing_tok else MISSING,
                            "i cache" if not missing_tok else "mangler: " + ", ".join(missing_tok), "3"))
    checks.append(Check("Embedding-model (chunking-lab)", OK if hf_cached(E5_MODEL) else MISSING,
                        f"{E5_MODEL}, ca. 470 MB", "3"))
    checks.append(Check("CLIP-model (foto-demo)", OK if hf_cached(CLIP_BASE) else MISSING,
                        f"{CLIP_BASE}, ca. 1,1 GB", "3"))
    checks.append(Check("Stor CLIP-model", OK if hf_cached(CLIP_LARGE) else OPTIONAL,
                        f"{CLIP_LARGE}, ca. 1,7 GB (kun 'Sammenlign modeller')", "7"))

    # Billeder og data i Qdrant
    photos = len(photo_files())
    samples = sample_photo_count()
    if photos >= 5 and (samples == 0 or samples >= SAMPLE_PHOTOS_EXPECTED):
        checks.append(Check("Foto-billeder", OK, f"{photos} billeder i photos/data"))
    else:
        checks.append(Check("Foto-billeder", MISSING,
                            f"{photos} billeder (eksempelsættet har {SAMPLE_PHOTOS_EXPECTED})", "4"))

    if collections is not None and collections.get("status") not in (401, 403):
        from_large = "modul3_photos_demo__" + CLIP_LARGE.replace("/", "_").replace("-", "_")
        indexed = collection_points("modul3_photos_demo") or 0
        checks.append(Check("Billeder i Qdrant", OK if photos and indexed == photos else MISSING,
                            f"{indexed} af {photos} indlæst", "5"))
        large = collection_points(from_large) or 0
        checks.append(Check("Billeder i Qdrant (stor CLIP)", OK if photos and large == photos else OPTIONAL,
                            f"{large} af {photos} indlæst", "7"))
        rag_name = os.getenv("QDRANT_COLLECTION", "rag_data")
        rag = collection_points(rag_name) or 0
        checks.append(Check("RAG-dokumenter (modul 2)", OK if rag else MISSING, f"{rag} chunks i '{rag_name}'", "6"))
    return checks


def print_scan(checks: list[Check]) -> None:
    header("Status")
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        mark = {OK: colorize("[OK]      ", GREEN), MISSING: colorize("[MANGLER] ", BOLD, RED),
                OPTIONAL: colorize("[VALGFRI] ", YELLOW)}[c.status]
        fix = ""
        if c.status != OK and c.fix:
            fix = colorize(f"  → menupunkt {c.fix}" if c.fix.isdigit() else f"  → {c.fix}", CYAN)
        print(f"  {mark}{c.name:<{width}}{colorize(c.detail, DIM)}{fix}")
    missing = [c for c in checks if c.status == MISSING]
    print()
    if missing:
        print(colorize(f"  {len(missing)} ting mangler. Menupunkt 9 henter og indlæser det, der kan hentes.", BOLD, YELLOW))
    else:
        print(colorize("  Alt nødvendigt er klar til kurset.", BOLD, GREEN))


# --------------------------------------------------------------------------- #
# Handlinger
# --------------------------------------------------------------------------- #

def create_env() -> bool:
    if ENV_FILE.exists():
        print(colorize(".env findes allerede; den ændres ikke.", DIM))
        return True
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    print(colorize(".env er oprettet fra .env.example. Skriv din OPENAI_API_KEY i filen.", GREEN))
    return True


def start_qdrant() -> bool:
    header("Start Qdrant")
    load_env()
    if qdrant_get("/collections") is not None:
        print(colorize("Qdrant kører allerede.", GREEN))
        return True
    if not shutil.which("docker"):
        print(colorize("Docker er ikke installeret eller ikke på PATH. Installér og start Docker Desktop.", RED))
        return False
    if not run(["docker", "compose", "up", "-d", "qdrant"]):
        print(colorize("Kunne ikke starte Qdrant. Kører Docker Desktop?", RED))
        return False
    for _ in range(30):
        if qdrant_get("/collections") is not None:
            print(colorize("Qdrant kører.", GREEN))
            return True
        time.sleep(1)
    print(colorize("Qdrant startede ikke inden for 30 sekunder.", RED))
    return False


def download_model(repo_id: str, kind: str) -> bool:
    """Hent med samme transformers-klasser som demoerne, så præcis de rigtige filer caches."""
    if hf_cached(repo_id):
        print(colorize(f"Klar: {repo_id} (allerede hentet)", GREEN))
        return True
    print(colorize(f"Henter {repo_id} …", CYAN), flush=True)
    try:
        if kind == "clip":
            from transformers import AutoProcessor, CLIPModel

            AutoProcessor.from_pretrained(repo_id)
            CLIPModel.from_pretrained(repo_id)
        else:
            from transformers import AutoModel, AutoTokenizer

            AutoTokenizer.from_pretrained(repo_id)
            AutoModel.from_pretrained(repo_id)
    except Exception as exc:
        print(colorize(f"Kunne ikke hente {repo_id}: {exc}", RED))
        return False
    print(colorize(f"Klar: {repo_id}", GREEN))
    return True


def download_models(include_large: bool = False) -> bool:
    header("Tokenizere og modeller")
    ok = python_script(ROOT_DIR / "tools" / "warmup_tokenizers.py")
    ok = download_model(E5_MODEL, "text") and ok
    ok = download_model(CLIP_BASE, "clip") and ok
    if include_large:
        ok = download_model(CLIP_LARGE, "clip") and ok
    return ok


def fetch_photos() -> bool:
    header("Hent eksempelbilleder (Wikimedia Commons)")
    print(colorize("Tager ca. 10 minutter. En afbrudt hentning fortsætter, når den startes igen.", DIM))
    return python_script(PHOTOS_DIR / "fetch_sample_photos.py")


def ingest_photos(model: str | None = None) -> bool:
    header("Indlæs billeder i Qdrant" + (f" ({model})" if model else ""))
    if len(photo_files()) < 5:
        print(colorize("Der skal være mindst 5 billeder i photos/data. Hent dem først (menupunkt 4).", RED))
        return False
    return python_script(PHOTOS_DIR / "ingest_photos.py", *(["--model", model] if model else []))


def ingest_rag() -> bool:
    header("Indlæs RAG-dokumenter (modul 2, big_demo)")
    load_env()
    if not env_value_set("OPENAI_API_KEY"):
        print(colorize("Kræver OPENAI_API_KEY i .env (embeddings laves hos OpenAI).", RED))
        return False
    return python_script(BIG_DEMO_DIR / "ingest.py")


def large_clip() -> bool:
    return download_models(include_large=True) and ingest_photos(CLIP_LARGE)


def run_tests() -> bool:
    header("Tests")
    return run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])


def do_everything() -> bool:
    """Hent og indlæs alt, der mangler, i den rækkefølge det afhænger af hinanden."""
    header("Klargør alt, der mangler")
    if not ENV_FILE.exists():
        create_env()
    checks = {c.name: c for c in scan(check_openai=False)}

    def missing(name: str) -> bool:
        return name in checks and checks[name].status == MISSING

    if checks["Python-pakker"].status == MISSING:
        print(colorize("Installér først pakkerne: pip install -r requirements.txt -c constraints-classroom.txt", RED))
        return False
    results = []
    if missing("Qdrant"):
        results.append(start_qdrant())
    if any(missing(n) for n in ("Tokenizere (tiktoken)", "Embedding-model (chunking-lab)", "CLIP-model (foto-demo)")):
        results.append(download_models())
    if missing("Foto-billeder"):
        results.append(fetch_photos())
    checks = {c.name: c for c in scan(check_openai=False)}
    if missing("Billeder i Qdrant"):
        results.append(ingest_photos())
    if missing("RAG-dokumenter (modul 2)"):
        results.append(ingest_rag())
    print_scan(scan())
    return all(results)


MENU = [
    ("1", "Scan: hvad mangler?", lambda: (print_scan(scan()), True)[1]),
    ("2", "Start Qdrant (Docker)", start_qdrant),
    ("3", "Hent tokenizere og lokale modeller (ca. 1,6 GB)", download_models),
    ("4", "Hent eksempelbilleder til foto-demoen (ca. 10 min)", fetch_photos),
    ("5", "Indlæs billeder i Qdrant", ingest_photos),
    ("6", "Indlæs RAG-dokumenter i Qdrant (modul 2, bruger OpenAI)", ingest_rag),
    ("7", "Stor CLIP-model til 'Sammenlign modeller' (1,7 GB, valgfri)", large_clip),
    ("8", "Kør tests", run_tests),
    ("9", "Gør alt, der mangler (punkt 2-6)", do_everything),
]


def show_menu() -> None:
    print()
    print(colorize("  SU-660 · klargøring til kurset", BOLD, CYAN))
    print(colorize("  " + "═" * 40, CYAN))
    for key, label, _ in MENU:
        print(f"   {colorize(key, BOLD, YELLOW)}  {label}")
    print(f"   {colorize('0', BOLD, YELLOW)}  Afslut")
    print()


def run_choice(choice: str) -> bool:
    for key, _, action in MENU:
        if choice == key:
            return bool(action())
    print(colorize(f"Ukendt valg: {choice}", RED))
    return False


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # æøå i Windows-konsollen
    os.chdir(ROOT_DIR)
    aliases = {"scan": "1", "alt": "9"}
    if argv:
        return 0 if run_choice(aliases.get(argv[0], argv[0])) else 1

    while True:
        show_menu()
        try:
            choice = input(colorize("  Vælg: ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in {"0", "q", "afslut"}:
            return 0
        ok = run_choice(choice)
        print(colorize("\n  Færdig." if ok else "\n  Der opstod et problem; se beskeden ovenfor.", GREEN if ok else YELLOW))
        try:
            input(colorize("  Tryk Enter for at vende tilbage til menuen …", DIM))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
