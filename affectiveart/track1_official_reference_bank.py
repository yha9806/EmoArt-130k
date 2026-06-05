from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BANK_VERSION = "track1_official_reference_bank_v1"

STYLE_ALIASES = (
    ("socialist realism", "Socialist Realism"),
    ("social realism", "Social Realism"),
    ("ink and wash painting", "Ink and wash painting"),
    ("ink and wash", "Ink and wash painting"),
    ("gongbi", "Gongbi"),
    ("ukiyo-e", "Ukiyo-e"),
    ("ukiyo e", "Ukiyo-e"),
    ("early renaissance", "Early Renaissance"),
    ("high renaissance", "High Renaissance"),
    ("mannerism", "Mannerism (Late Renaissance)"),
    ("late renaissance", "Mannerism (Late Renaissance)"),
    ("impressionism", "Impressionism"),
    ("baroque", "Baroque"),
    ("pop art", "Pop Art"),
    ("abstract art", "Abstract Art"),
    ("abstract", "Abstract Art"),
)

POSTER_TERMS = {
    "poster",
    "propaganda",
    "tass",
    "frontpage",
    "newspaper",
    "victory",
    "liberate",
    "soviet",
    "cyrillic",
    "typography",
    "flag",
    "flags",
}

GENERIC_STYLE_TOKENS = {
    "art",
    "artwork",
    "bold",
    "cyrillic",
    "lettering",
    "painting",
    "poster",
    "propaganda",
    "realism",
    "realist",
    "socialism",
    "socialist",
    "soviet",
    "style",
    "typography",
}

LOW_SIGNAL_QUERY_TOKENS = {
    "above",
    "accents",
    "before",
    "behind",
    "black",
    "blue",
    "bold",
    "celebratory",
    "composition",
    "dramatic",
    "framed",
    "golden",
    "large",
    "patriotic",
    "red",
    "solemn",
    "white",
}

ANCHOR_TOKEN_GROUPS = (
    frozenset({"naval", "sailor", "fleet", "warship", "ship", "sea", "battle", "battles"}),
    frozenset({"kremlin", "moscow", "square", "searchlight", "tower"}),
    frozenset({"train", "railway", "passenger", "border", "frontier"}),
    frozenset({"tank", "soldier", "soldiers", "banner", "banners"}),
    frozenset({"pilot", "airplane", "aircraft", "plane", "planes", "flight"}),
    frozenset({"wheat", "harvest", "harvesting", "peasant", "peasants", "barbed", "wire"}),
    frozenset({"medal", "ribbon", "firework", "fireworks", "searchlight", "kremlin"}),
    frozenset({"horse", "horses", "cavalry", "mounted"}),
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "with",
    "of",
    "in",
    "on",
    "by",
    "to",
    "for",
    "from",
    "using",
    "use",
    "artwork",
    "painting",
    "composition",
    "style",
    "rendered",
    "palette",
}


@dataclass(frozen=True)
class OfficialArtwork:
    style: str
    filename: str
    member_path: str
    image_path: str
    searchable_text: str


def build_official_reference_bank(
    routes: Iterable[dict[str, Any]],
    *,
    emoart_root: str | Path,
    out_dir: str | Path,
    max_candidates_per_route: int = 8,
) -> dict[str, Any]:
    if max_candidates_per_route <= 0:
        raise ValueError("max_candidates_per_route must be positive")
    emoart_root = Path(emoart_root)
    out_dir = Path(out_dir)
    asset_dir = out_dir / "reference_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    records_by_style = _load_official_artworks(emoart_root / "Annotation.json")
    selected_by_route: dict[str, list[OfficialArtwork]] = {}
    selected_scores: dict[tuple[str, str], float] = {}
    selected_pool_sizes: dict[str, int] = {}
    usage_counts: Counter[str] = Counter()

    for route in routes:
        sample_id = str(route.get("sample_id") or "")
        if not sample_id:
            continue
        style = _style_for_caption(str(route.get("caption") or ""), records_by_style.keys())
        candidates = records_by_style.get(style, [])
        selected_pool_sizes[sample_id] = len(candidates)
        ranked = _rank_candidates(route, candidates, usage_counts)
        selected = [artwork for artwork, _score in ranked[:max_candidates_per_route]]
        for artwork, score in ranked[:max_candidates_per_route]:
            selected_scores[(sample_id, artwork.member_path)] = score
            usage_counts.update([artwork.member_path])
        selected_by_route[sample_id] = selected

    extracted = _extract_selected_artworks(emoart_root, asset_dir, selected_by_route)
    route_references: dict[str, list[dict[str, str]]] = {}
    for sample_id, selected in selected_by_route.items():
        rows: list[dict[str, str]] = []
        for artwork in selected:
            file_value = extracted.get(artwork.member_path)
            if not file_value:
                continue
            score = selected_scores.get((sample_id, artwork.member_path), 0.0)
            rows.append(
                {
                    "file": file_value,
                    "note": (
                        "official EmoArt-130k retrieval; "
                        f"style={artwork.style}; score={score:.3f}; "
                        f"candidate_pool={selected_pool_sizes.get(sample_id, 0)}; "
                        f"source={artwork.image_path}"
                    ),
                }
            )
        if rows:
            route_references[sample_id] = rows

    payload = {
        "version": BANK_VERSION,
        "asset_root": str(asset_dir),
        "route_references": route_references,
        "references": {},
        "family_references": {},
        "caption_style_references": {},
        "metadata": {
            "emoart_root": str(emoart_root),
            "max_candidates_per_route": max_candidates_per_route,
            "official_candidate_pool_by_style": {
                style: len(records) for style, records in sorted(records_by_style.items())
            },
            "routes_with_official_references": len(route_references),
            "extracted_reference_assets": len(set(extracted.values())),
        },
    }
    summary = dict(payload["metadata"])
    summary["version"] = BANK_VERSION
    return {"index": payload, "summary": summary}


def write_official_reference_bank(
    routes: Iterable[dict[str, Any]],
    *,
    emoart_root: str | Path,
    out_dir: str | Path,
    index_json: str | Path,
    summary_json: str | Path,
    max_candidates_per_route: int = 8,
) -> dict[str, Any]:
    result = build_official_reference_bank(
        routes,
        emoart_root=emoart_root,
        out_dir=out_dir,
        max_candidates_per_route=max_candidates_per_route,
    )
    index_path = Path(index_json)
    summary_path = Path(summary_json)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(result["index"], indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result["summary"]


def _load_official_artworks(annotation_path: Path) -> dict[str, list[OfficialArtwork]]:
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Annotation.json must contain a list")
    output: dict[str, list[OfficialArtwork]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_path = str(row.get("image_path") or "")
        parts = re.split(r"[\\/]+", image_path)
        if len(parts) < 3 or parts[0] != "Images":
            continue
        style = parts[1]
        filename = parts[-1]
        searchable_text = " ".join(
            [
                style,
                filename,
                _annotation_text(row.get("description")),
            ]
        )
        output[style].append(
            OfficialArtwork(
                style=style,
                filename=filename,
                member_path=f"{style}/{filename}",
                image_path=image_path,
                searchable_text=searchable_text,
            )
        )
    return dict(output)


def _annotation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_annotation_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_annotation_text(item) for item in value)
    return ""


def _style_for_caption(caption: str, official_styles: Iterable[str] = ()) -> str:
    official_style_set = set(official_styles)
    caption_lower = caption.lower()
    caption_normalised = _normalise_style_text(caption)
    for needle, style in STYLE_ALIASES:
        if needle in caption_lower and (not official_style_set or style in official_style_set):
            return style
    for style in sorted(official_style_set, key=lambda item: len(_normalise_style_text(item)), reverse=True):
        for variant in _style_name_variants(style):
            variant_normalised = _normalise_style_text(variant)
            if variant_normalised and variant_normalised in caption_normalised:
                return style
    return "Abstract Art"


def _style_name_variants(style: str) -> list[str]:
    variants = [style]
    base = re.sub(r"\s*\([^)]*\)", "", style).strip()
    if base and base != style:
        variants.append(base)
    variants.extend(match.strip() for match in re.findall(r"\(([^)]*)\)", style) if match.strip())
    return variants


def _normalise_style_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def _rank_candidates(
    route: dict[str, Any],
    candidates: list[OfficialArtwork],
    usage_counts: Counter[str],
) -> list[tuple[OfficialArtwork, float]]:
    caption = str(route.get("caption") or "")
    query_tokens = _tokens(caption)
    specific_query_tokens = _specific_query_tokens(caption)
    poster_requested = _poster_requested(caption)
    ranked: list[tuple[OfficialArtwork, float]] = []
    for artwork in candidates:
        text = artwork.searchable_text.lower()
        candidate_tokens = _tokens(text)
        generic_overlap = len(query_tokens & candidate_tokens)
        specific_overlap = len(specific_query_tokens & candidate_tokens)
        specific_coverage = specific_overlap / max(1, len(specific_query_tokens))
        score = (0.65 * generic_overlap) + (3.25 * specific_overlap) + (6.0 * specific_coverage)
        score += _anchor_group_score(specific_query_tokens, candidate_tokens)
        if poster_requested:
            score += min(3.0, 0.75 * len(POSTER_TERMS & candidate_tokens))
            if any(term in Path(artwork.filename).stem.lower() for term in POSTER_TERMS):
                score += 1.0
        score -= 0.15 * usage_counts[artwork.member_path]
        score += _stable_fraction(str(route.get("sample_id") or ""), artwork.member_path)
        ranked.append((artwork, score))
    ranked.sort(key=lambda item: (-item[1], item[0].member_path))
    return ranked


def _anchor_group_score(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    score = 0.0
    for group in ANCHOR_TOKEN_GROUPS:
        requested = group & query_tokens
        if not requested:
            continue
        matched = group & candidate_tokens
        if matched:
            score += (3.0 * len(matched)) + (4.0 * (len(matched) / max(1, len(requested))))
        else:
            score -= 4.0
    return score


def _specific_query_tokens(caption: str) -> set[str]:
    return _tokens(caption) - GENERIC_STYLE_TOKENS - POSTER_TERMS - LOW_SIGNAL_QUERY_TOKENS


def _tokens(value: str) -> set[str]:
    prepared = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    prepared = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", prepared)
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", prepared.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            expanded.add(token[:-1])
    return expanded


def _poster_requested(caption: str) -> bool:
    text = caption.lower()
    return "poster" in text or "propaganda" in text or "typography" in text or "cyrillic" in text


def _stable_fraction(sample_id: str, member_path: str) -> float:
    value = f"{sample_id}:{member_path}"
    total = sum((index + 1) * ord(character) for index, character in enumerate(value))
    return (total % 1000) / 10000.0


def _extract_selected_artworks(
    emoart_root: Path,
    asset_dir: Path,
    selected_by_route: dict[str, list[OfficialArtwork]],
) -> dict[str, str]:
    by_style: dict[str, dict[str, OfficialArtwork]] = defaultdict(dict)
    for selected in selected_by_route.values():
        for artwork in selected:
            by_style[artwork.style][artwork.member_path] = artwork
    extracted: dict[str, str] = {}
    for style, artworks in by_style.items():
        archive_path = emoart_root / f"{style}.tar.gz"
        if not archive_path.exists():
            continue
        pending: dict[str, OfficialArtwork] = {}
        for member_path, artwork in artworks.items():
            output_name = _safe_asset_name(style, artwork.filename, artwork.member_path)
            output_path = asset_dir / output_name
            if output_path.exists() and output_path.is_file():
                extracted[member_path] = output_name
            else:
                pending[member_path] = artwork
        if not pending:
            continue
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                artwork = pending.get(member.name)
                if artwork is None or not member.isfile():
                    continue
                output_name = _safe_asset_name(style, artwork.filename, artwork.member_path)
                output_path = asset_dir / output_name
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, output_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted[member.name] = output_name
                del pending[member.name]
                if not pending:
                    break
    return extracted


def _safe_asset_name(style: str, filename: str, member_path: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower() or ".jpg"
    value = f"official_{style}_{stem}"
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    safe = "".join(character if character.isalnum() else "_" for character in value)
    safe = re.sub(r"_+", "_", safe).strip("_")
    digest = hashlib.sha1(member_path.encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{digest}{suffix}"
