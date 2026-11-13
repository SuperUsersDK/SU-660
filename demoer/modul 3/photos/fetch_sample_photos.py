"""Hent eksempelbilleder i 10 tydelige kategorier fra Wikimedia Commons.

Billederne kommer fra Commons' "Quality images"-kategorier og er frit licenserede
(CC BY, CC BY-SA eller CC0). Fotograf og licens står på hvert billedes Commons-side.

CLIP-filteret sikrer tydelige klynger på kortet: hvert billede sammenlignes med
alle 10 kategoribeskrivelser og beholdes kun, hvis CLIP sikkert placerer det i
sin egen kategori. Fx sorteres et nærbillede af en bildør eller et frø i
blomsterkategorien fra, og der hentes en ny kandidat i stedet.

Kør fra repository-roden (kræver internet; 20 billeder pr. kategori tager ca. 10 minutter,
fordi Wikimedia begrænser antallet af forespørgsler). Et afbrudt eller tidligere hentet
sæt genbruges: kør scriptet igen for at fortsætte, eller med --fresh for at starte forfra.

    python "demoer/modul 3/photos/fetch_sample_photos.py"
    python "demoer/modul 3/photos/ingest_photos.py"
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.terminal_ui import CYAN, DIM, GREEN, RED, YELLOW, colorize, print_label, print_section
from sample_categories import SAMPLE_CATEGORIES, SampleCategory

DATA_DIR = Path(__file__).resolve().parent / "data"
API = "https://commons.wikimedia.org/w/api.php"
# Generisk klientnavn; Wikimedia kræver en User-Agent.
HEADERS = {"User-Agent": "photo-demo-downloader/1.0 (educational use)"}
# Wikimedia leverer kun skalerede billeder i faste standardbredder.
THUMB_WIDTH = 960
MIN_SIDE = 400
# Underkategorier med afbildninger frem for det rigtige motiv springes over.
SKIP_SUBCATEGORY_WORDS = ("sculpture", "statue", "drawing", "illustration", "painting", "stamp",
                          "model", "toy", "museum", "fossil", "logo", "sign", "interior")


# --------------------------------------------------------------------------- #
# Wikimedia Commons
# --------------------------------------------------------------------------- #

def http_get(url: str) -> bytes:
    """Hent en URL og vent pænt, hvis Wikimedia svarer 429 (for mange forespørgsler)."""
    request = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or 5 * (attempt + 1)
            print(colorize(f"  Wikimedia beder os vente {wait}s ...", DIM), flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def api(params: dict[str, str]) -> dict:
    data = json.loads(http_get(API + "?" + urllib.parse.urlencode({"format": "json", **params})))
    time.sleep(0.5)
    return data


def list_files(category: str, *, want: int, max_depth: int = 2) -> list[str]:
    """Saml JPG-filnavne fra kategorien og dens underkategorier (bredde først)."""
    titles: list[str] = []
    queue = [(category, 0)]
    seen = {category}
    while queue and len(titles) < want:
        current, depth = queue.pop(0)
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{current}",
            "cmtype": "file|subcat",
            "cmlimit": "500",
        }
        while True:
            data = api(params)
            for member in data.get("query", {}).get("categorymembers", []):
                title = member["title"]
                if title.startswith("Category:"):
                    name = title.removeprefix("Category:")
                    if any(word in name.lower() for word in SKIP_SUBCATEGORY_WORDS):
                        continue
                    if depth < max_depth and name not in seen:
                        seen.add(name)
                        queue.append((name, depth + 1))
                elif title.lower().endswith((".jpg", ".jpeg")):
                    titles.append(title)
            if "continue" not in data or len(titles) >= want:
                break
            params.update(data["continue"])
    return list(dict.fromkeys(titles))


def thumb_urls(titles: list[str]) -> dict[str, str]:
    """Hent URL til en skaleret udgave af op til 10 filer pr. kald."""
    urls: dict[str, str] = {}
    for start in range(0, len(titles), 10):
        data = api({
            "action": "query",
            "titles": "|".join(titles[start:start + 10]),
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": str(THUMB_WIDTH),
        })
        for page in data.get("query", {}).get("pages", {}).values():
            item = (page.get("imageinfo") or [{}])[0]
            if item.get("mime") == "image/jpeg" and item.get("thumburl"):
                urls[page["title"]] = item["thumburl"]
    return urls


# --------------------------------------------------------------------------- #
# CLIP-filter
# --------------------------------------------------------------------------- #

def accept(probabilities: list[float], own_index: int, min_confidence: float) -> bool:
    """Behold kun billeder, CLIP sikkert placerer i deres egen kategori."""
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return best == own_index and probabilities[own_index] >= min_confidence


class ClipFilter:
    def __init__(self, categories: tuple[SampleCategory, ...]):
        from image_embeddings import texts_to_vectors

        self.label_vectors = texts_to_vectors([category.prompt for category in categories])

    def probabilities(self, image: Image.Image) -> list[float]:
        from image_embeddings import image_to_vector
        from tagging import LOGIT_SCALE, softmax

        vector = image_to_vector(image)
        cosines = [sum(a * b for a, b in zip(vector, label)) for label in self.label_vectors]
        return softmax([LOGIT_SCALE * cosine for cosine in cosines])


# --------------------------------------------------------------------------- #
# Hovedprogram
# --------------------------------------------------------------------------- #

def file_stem(title: str) -> str:
    stem = Path(title.removeprefix("File:")).stem
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:40] or "billede"


def sample_file_pattern() -> re.Pattern[str]:
    tags = "|".join(re.escape(category.tag) for category in SAMPLE_CATEGORIES)
    return re.compile(rf"^({tags})_(\d{{3}})_(.*)\.jpg$")


def remove_previous_samples(data_dir: Path) -> int:
    """Slet kun filer, som dette script selv har lavet (fx 'bil_007_....jpg')."""
    pattern = sample_file_pattern()
    removed = 0
    for path in data_dir.glob("*.jpg"):
        if pattern.match(path.name):
            path.unlink()
            removed += 1
    return removed


def existing_samples(data_dir: Path, tag: str) -> list[Path]:
    """Billeder, en tidligere kørsel har hentet til kategorien, sorteret efter nummer."""
    pattern = sample_file_pattern()
    found = []
    for path in data_dir.glob(f"{tag}_*.jpg"):
        match = pattern.match(path.name)
        if match and match.group(1) == tag:
            found.append((int(match.group(2)), path))
    return [path for _, path in sorted(found)]


def keep_first(paths: list[Path], count: int) -> list[Path]:
    """Behold de første `count` billeder, slet resten, og nummerér fortløbende."""
    for path in paths[count:]:
        path.unlink()
    kept = []
    for number, path in enumerate(paths[:count], 1):
        tag, _, rest = path.name.split("_", 2)
        target = path.with_name(f"{tag}_{number:03d}_{rest}")
        if target != path:
            path.rename(target)
        kept.append(target)
    return kept


def fetch_category(
    category: SampleCategory,
    index: int,
    *,
    count: int,
    data_dir: Path,
    clip: ClipFilter | None,
    min_confidence: float,
    used_stems: set[str],
    seed: int,
) -> int:
    print_label(f"{category.tag.upper()}  ({', '.join(category.commons_categories)})", CYAN)

    before = existing_samples(data_dir, category.tag)
    kept = keep_first(before, count)
    if len(before) > len(kept):
        print(colorize(f"  Slettede {len(before) - len(kept)} overskydende billeder", DIM))
    if len(kept) >= count:
        print(colorize(f"  {len(kept)}/{count} findes allerede", GREEN))
        return len(kept)
    if kept:
        print(colorize(f"  {len(kept)} billeder fra en tidligere kørsel genbruges", DIM))

    pool: list[str] = []
    for commons in category.commons_categories:
        pool += list_files(commons, want=max(300, count * 6))
    candidates = [title for title in dict.fromkeys(pool) if file_stem(title) not in used_stems]
    random.Random(f"{seed}-{category.tag}").shuffle(candidates)
    print(colorize(f"  {len(candidates)} kandidater", DIM))

    rejected = 0
    for start in range(0, len(candidates), 10):
        if len(kept) >= count:
            break
        batch = candidates[start:start + 10]
        urls = thumb_urls(batch)
        for title in batch:
            if len(kept) >= count or title not in urls:
                continue
            try:
                raw = http_get(urls[title])
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if min(image.size) < MIN_SIDE:
                        continue
                    if clip is not None and not accept(clip.probabilities(image), index, min_confidence):
                        rejected += 1
                        continue
            except (UnidentifiedImageError, OSError, urllib.error.URLError) as exc:
                print(colorize(f"  spring over {title}: {exc}", DIM))
                continue
            finally:
                time.sleep(0.3)

            stem = file_stem(title)
            path = data_dir / f"{category.tag}_{len(kept) + 1:03d}_{stem}.jpg"
            path.write_bytes(raw)
            used_stems.add(stem)
            kept.append(path)
        print(colorize(f"  {len(kept)}/{count} gemt, {rejected} frasorteret af CLIP", GREEN), flush=True)

    if len(kept) < count:
        print(colorize(f"  Kun {len(kept)} billeder fundet til {category.tag}", YELLOW))
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category", type=int, default=20, help="Billeder pr. kategori (standard 20)")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="CLIP-sikkerhed for egen kategori, 0-1 (standard 0.6)")
    parser.add_argument("--no-clip-filter", action="store_true", help="Behold alle billeder uden CLIP-tjek")
    parser.add_argument("--fresh", action="store_true",
                        help="Slet tidligere hentede eksempelbilleder og start forfra")
    parser.add_argument("--seed", type=int, default=660, help="Tilfældigt frø, så samme billeder vælges igen")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print_section("Hent eksempelbilleder")
    print(colorize(f"Mappe: {DATA_DIR}", DIM))
    print(colorize(f"Kategorier: {', '.join(c.tag for c in SAMPLE_CATEGORIES)}", DIM))
    print(colorize(f"Pr. kategori: {args.per_category}", DIM))

    if args.fresh:
        removed = remove_previous_samples(DATA_DIR)
        if removed:
            print(colorize(f"Slettede {removed} tidligere eksempelbilleder", DIM))

    clip = None
    if not args.no_clip_filter:
        print(colorize(f"CLIP-filter: til (min. sikkerhed {args.min_confidence:.2f})", DIM))
        clip = ClipFilter(SAMPLE_CATEGORIES)

    # Samme Commons-billede må ikke ende i to kategorier.
    pattern = sample_file_pattern()
    used_stems = {m.group(3) for p in DATA_DIR.glob("*.jpg") if (m := pattern.match(p.name))}

    total = 0
    for index, category in enumerate(SAMPLE_CATEGORIES):
        total += fetch_category(
            category,
            index,
            count=args.per_category,
            data_dir=DATA_DIR,
            clip=clip,
            min_confidence=args.min_confidence,
            used_stems=used_stems,
            seed=args.seed,
        )

    if not total:
        print(colorize("Ingen billeder hentet.", RED))
        sys.exit(1)

    print_label("RESULTAT", GREEN)
    print(colorize(f"{total} billeder i {DATA_DIR}", GREEN))
    print(colorize('Næste skridt: python "demoer/modul 3/photos/ingest_photos.py"', DIM))


if __name__ == "__main__":
    main()
