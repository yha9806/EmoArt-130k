from __future__ import annotations

import html
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


CaptionPredicate = Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True)
class ReferenceRole:
    name: str
    label_zh: str
    why_zh: str
    required_if: CaptionPredicate
    evidence_terms: tuple[str, ...]


def build_reference_role_gate(
    routes: Iterable[dict[str, Any]],
    *,
    candidate_manifest: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [
        _gate_row(dict(route), fallback=_fallback_by_sample(candidate_manifest).get(str(route.get("sample_id") or "")))
        for route in routes
        if isinstance(route, dict)
    ]
    return {
        "summary": _summary(rows),
        "rows": rows,
        "missing_role_summary": _missing_role_summary(rows),
        "fallback_rows": [row for row in rows if "provider_reference_fallback" in row["risk_flags"]],
    }


def load_routes_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("routes"), list):
        rows = payload["routes"]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        raise ValueError("routes JSON must be a list or an object with routes/rows")
    return [dict(row) for row in rows if isinstance(row, dict)]


def load_candidate_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_reference_role_gate_artifacts(
    report: dict[str, Any],
    *,
    out_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "track1_reference_role_gate.json"
    md_path = out_dir / "track1_reference_role_gate_zh.md"
    html_path = out_dir / "track1_reference_role_gate_zh.html"
    payload = json.loads(json.dumps(report, ensure_ascii=False))
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_reference_role_gate_md(payload), encoding="utf-8")
    html_path.write_text(
        render_reference_role_gate_html(payload, out_html=html_path, repo_root=repo_root),
        encoding="utf-8",
    )
    return {"json": str(json_path), "md": str(md_path), "html": str(html_path)}


def render_reference_role_gate_md(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Track1 Reference Role Gate",
        "",
        "这份报告检查每个 reference board 是否覆盖官方 caption 需要的关键角色：媒介、主体、符号、真实锚点、动作关系和空间逻辑。",
        "",
        "## 摘要",
        "",
        f"- Total: {summary.get('total', 0)}",
        f"- Pass: {summary.get('pass', 0)}",
        f"- Fail: {summary.get('fail', 0)}",
        f"- Provider fallback: {summary.get('provider_reference_fallback', 0)}",
        "",
        "## 缺失角色统计",
        "",
    ]
    for row in report.get("missing_role_summary", []):
        lines.append(f"- `{row.get('role')}`: {row.get('count')}")
    lines.extend(["", "## Fail Rows", ""])
    for row in report.get("rows", []):
        if row.get("status") != "fail":
            continue
        missing = ", ".join(str(item) for item in row.get("missing_required_roles", []))
        flags = ", ".join(str(item) for item in row.get("risk_flags", []))
        lines.append(f"- `{row.get('sample_id')}`: missing=[{missing}] flags=[{flags}]")
    return "\n".join(lines).rstrip() + "\n"


def render_reference_role_gate_html(
    report: dict[str, Any],
    *,
    out_html: str | Path,
    repo_root: str | Path | None = None,
) -> str:
    out_html = Path(out_html)
    summary = report.get("summary", {})
    fail_rows = [row for row in report.get("rows", []) if row.get("status") == "fail"]
    pass_rows = [row for row in report.get("rows", []) if row.get("status") == "pass"]
    fail_cards = "\n".join(_render_row_card(row, out_html=out_html, repo_root=repo_root) for row in fail_rows[:240])
    pass_cards = "\n".join(_render_row_card(row, out_html=out_html, repo_root=repo_root) for row in pass_rows[:24])
    missing_items = "\n".join(
        f"<li><code>{_escape(row.get('role'))}</code>: {_escape(row.get('count'))}</li>"
        for row in report.get("missing_role_summary", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Track1 Reference Role Gate</title>
  <style>
    :root {{
      --bg: #f5f3ee;
      --panel: #fff;
      --line: #d7d1c5;
      --text: #202225;
      --muted: #64707d;
      --bad: #b42318;
      --good: #0f766e;
      --warn: #a15c07;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(245, 243, 238, 0.96);
    }}
    h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
    .intro {{ margin: 0; color: var(--muted); max-width: 1120px; }}
    main {{ padding: 20px 24px 56px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 20px;
    }}
    .metric {{
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .metric strong {{ display: block; font-size: 24px; color: var(--good); }}
    .metric.bad strong {{ color: var(--bad); }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    h2 {{ margin: 28px 0 12px; font-size: 19px; letter-spacing: 0; }}
    .card {{
      margin: 0 0 18px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .head {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    .sid {{ font-weight: 700; font-size: 17px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f9fafb;
    }}
    .badge.fail {{ color: var(--bad); border-color: #f0b8b2; background: #fff4f2; }}
    .badge.pass {{ color: var(--good); border-color: #a7d8d2; background: #effaf8; }}
    .caption {{ margin: 8px 0 12px; color: var(--muted); }}
    .checks {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 8px;
      margin: 10px 0 14px;
    }}
    .check {{
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfaf7;
      font-size: 13px;
    }}
    .check strong {{ display: block; margin-bottom: 3px; }}
    .missing {{ color: var(--bad); font-weight: 650; }}
    .ok {{ color: var(--good); font-weight: 650; }}
    .thumbs {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    figure {{ margin: 0; }}
    figure img {{
      width: 100%;
      height: 150px;
      object-fit: contain;
      background: #eee9df;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    figcaption {{
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
  <header>
    <h1>Track1 Reference Role Gate</h1>
    <p class="intro">角色质量门：检查 reference board 是否真的支撑 caption 的媒介、主体、符号、地标和空间关系。这个报告用于重检索和重生成前的拦截，不是最终人评。</p>
  </header>
  <main>
    <section class="summary">
      {_metric('Total', summary.get('total', 0))}
      {_metric('Pass', summary.get('pass', 0))}
      {_metric('Fail', summary.get('fail', 0), bad=True)}
      {_metric('Fallback', summary.get('provider_reference_fallback', 0), bad=True)}
    </section>
    <h2>一、缺失角色统计</h2>
    <div class="card"><ul>{missing_items}</ul></div>
    <h2>二、Fail 样本</h2>
    {fail_cards or '<p class="caption">没有 fail 样本。</p>'}
    <h2>三、Pass 抽样</h2>
    {pass_cards or '<p class="caption">没有 pass 样本。</p>'}
  </main>
</body>
</html>"""


def _gate_row(route: dict[str, Any], *, fallback: dict[str, Any] | None) -> dict[str, Any]:
    sample_id = str(route.get("sample_id") or "")
    caption = str(route.get("caption") or "")
    asset_texts = _asset_texts(route)
    checks = [_role_check(role, caption, route, asset_texts) for role in REFERENCE_ROLES if role.required_if(caption, route)]
    missing = [check["role"] for check in checks if check["status"] == "missing"]
    risk_flags: list[str] = []
    fallback_reason = ""
    if fallback:
        risk_flags.append("provider_reference_fallback")
        fallback_reason = str(fallback.get("reference_image_fallback_reason") or "")
    status = "fail" if missing or risk_flags else "pass"
    return {
        "sample_id": sample_id,
        "caption": caption,
        "status": status,
        "required_roles": [check["role"] for check in checks],
        "missing_required_roles": missing,
        "risk_flags": risk_flags,
        "fallback_reason": fallback_reason,
        "reference_assets": [str(item) for item in _as_list(route.get("reference_assets"))],
        "reference_asset_notes": [str(item) for item in _as_list(route.get("reference_asset_notes"))],
        "role_checks": checks,
    }


def _role_check(
    role: ReferenceRole,
    caption: str,
    route: dict[str, Any],
    asset_texts: list[dict[str, str]],
) -> dict[str, Any]:
    matches = [
        item
        for item in asset_texts
        if _role_matches_text(item["text"], role)
    ]
    return {
        "role": role.name,
        "label_zh": role.label_zh,
        "why_zh": role.why_zh,
        "status": "pass" if matches else "missing",
        "evidence_assets": [item["path"] for item in matches[:4]],
        "evidence_terms": list(role.evidence_terms),
    }


def _fallback_by_sample(candidate_manifest: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]]
    if candidate_manifest is None:
        rows = []
    elif isinstance(candidate_manifest, list):
        rows = [row for row in candidate_manifest if isinstance(row, dict)]
    elif isinstance(candidate_manifest, dict) and isinstance(candidate_manifest.get("rows"), list):
        rows = [row for row in candidate_manifest["rows"] if isinstance(row, dict)]
    else:
        rows = []
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("reference_image_fallback_without_reference"):
            continue
        sample_id = str(row.get("sample_id") or "")
        if sample_id:
            output[sample_id] = row
    return output


def _asset_texts(route: dict[str, Any]) -> list[dict[str, str]]:
    assets = [str(item) for item in _as_list(route.get("reference_assets"))]
    notes = [str(item) for item in _as_list(route.get("reference_asset_notes"))]
    output: list[dict[str, str]] = []
    for index, asset in enumerate(assets):
        note = notes[index] if index < len(notes) else ""
        output.append({"path": asset, "text": _normalise_text(f"{asset} {note}")})
    return output


def _normalise_text(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


_COMPACT_ROLE_TERMS = {
    "redsquare",
}


def _role_matches_text(normalised_text: str, role: ReferenceRole) -> bool:
    if role.name == "allied_flags":
        return _text_has_any_role_term(normalised_text, ("flag", "flags")) and _text_has_any_role_term(
            normalised_text,
            ("allied", "american", "british", "united", "soviet"),
        )
    if role.name == "mounted_civilian_relation":
        return _text_has_any_role_term(normalised_text, ("mounted", "cavalry", "horse", "horses")) and _text_has_any_role_term(
            normalised_text,
            ("civilian", "civilians", "refugee", "village", "evacuation"),
        )
    for raw_term in role.evidence_terms:
        if _text_has_role_term(normalised_text, raw_term):
            return True
    return False


def _text_has_any_role_term(normalised_text: str, raw_terms: tuple[str, ...]) -> bool:
    return any(_text_has_role_term(normalised_text, term) for term in raw_terms)


def _text_has_role_term(normalised_text: str, raw_term: str) -> bool:
    term = _normalise_text(raw_term)
    if not term:
        return False
    padded = f" {normalised_text} "
    tokens = set(normalised_text.split())
    compact = normalised_text.replace(" ", "")
    if " " in term:
        return f" {term} " in padded or term.replace(" ", "") in compact
    return term in tokens or (term in _COMPACT_ROLE_TERMS and term in compact)


def _caption_has_any(caption: str, *needles: str) -> bool:
    text = caption.lower()
    return any(needle in text for needle in needles)


def _caption_has_all(caption: str, *needles: str) -> bool:
    text = caption.lower()
    return all(needle in text for needle in needles)


def _needs_train_window(caption: str, route: dict[str, Any]) -> bool:
    return _caption_has_any(caption, "train", "railway", "locomotive") and _caption_has_any(
        caption, "window", "passenger", "frontier", "border guard", "border"
    )


def _needs_mounted_relation(caption: str, route: dict[str, Any]) -> bool:
    return _caption_has_any(caption, "mounted", "horse", "cavalry") and _caption_has_any(
        caption, "civilian", "civilians", "fleeing", "burning village", "village ruins"
    )


def _needs_allied_flags(caption: str, route: dict[str, Any]) -> bool:
    return _caption_has_all(caption, "soviet", "american", "british") and _caption_has_any(caption, "flag", "flags")


def _needs_naval_battle(caption: str, route: dict[str, Any]) -> bool:
    return _caption_has_any(caption, "naval", "sailor", "sailors", "warship", "warships", "cannon") and _caption_has_any(
        caption, "sea", "ship", "ships", "battle", "maritime", "fleet"
    )


REFERENCE_ROLES: tuple[ReferenceRole, ...] = (
    ReferenceRole(
        name="poster_print",
        label_zh="海报/印刷媒介",
        why_zh="caption 要求 poster/propaganda/Cyrillic typography 时，reference 需要至少有海报或宣传印刷媒介信号。",
        required_if=lambda caption, route: _caption_has_any(
            caption, "poster", "propaganda", "cyrillic", "typography", "slogan"
        ),
        evidence_terms=(
            "poster",
            "tass",
            "frontpage",
            "allforthefront",
            "allforvictory",
            "liberate",
            "tovictory",
            "kukryniksy",
            "propaganda",
        ),
    ),
    ReferenceRole(
        name="kremlin_landmark",
        label_zh="克林姆林/红场地标",
        why_zh="caption 点名 Kremlin/Red Square 时，reference 需要支撑塔楼、城墙或莫斯科红场视觉锚点。",
        required_if=lambda caption, route: _caption_has_any(caption, "kremlin", "red square", "moscow"),
        evidence_terms=("kremlin", "moscow", "redsquare", "red army parade", "military parade", "trinity gates"),
    ),
    ReferenceRole(
        name="allied_flags",
        label_zh="盟国旗帜符号",
        why_zh="caption 同时要求 Soviet/American/British flags 时，reference 需要支持多国旗帜主次关系。",
        required_if=_needs_allied_flags,
        evidence_terms=("allied", "american", "british", "flags", "flag", "united"),
    ),
    ReferenceRole(
        name="medal_symbol",
        label_zh="勋章/丝带符号",
        why_zh="caption 要求 medal/ribbon 时，reference 需要支撑勋章、丝带或徽章结构。",
        required_if=lambda caption, route: _caption_has_any(caption, "medal", "ribbon", "badge", "emblem"),
        evidence_terms=("medal", "ribbon", "badge", "emblem", "order"),
    ),
    ReferenceRole(
        name="naval_battle",
        label_zh="海军/舰船/海战主体",
        why_zh="caption 要求 naval/sailor/warships/cannon/maritime battle 时，reference 需要有舰船、海军或海战主体。",
        required_if=_needs_naval_battle,
        evidence_terms=(
            "naval",
            "navy",
            "sailor",
            "sailors",
            "marine",
            "marines",
            "fleet",
            "ship",
            "ships",
            "warship",
            "harbor",
            "harbour",
            "flotilla",
            "sea",
            "potemkin",
        ),
    ),
    ReferenceRole(
        name="cannon_artillery",
        label_zh="大炮/火炮结构",
        why_zh="caption 点名 cannon/artillery/shells 时，reference 需要支撑火炮机械结构。",
        required_if=lambda caption, route: _caption_has_any(caption, "cannon", "artillery", "shells", "howitzer"),
        evidence_terms=("cannon", "artillery", "shell", "howitzer", "gun"),
    ),
    ReferenceRole(
        name="train_window",
        label_zh="火车窗/乘客空间关系",
        why_zh="caption 要求 train window/passengers/frontier post 时，reference 需要支撑火车、窗洞和人物互动关系。",
        required_if=_needs_train_window,
        evidence_terms=("train", "railway", "rail", "locomotive", "echelon", "passenger", "frontier", "border"),
    ),
    ReferenceRole(
        name="mounted_civilian_relation",
        label_zh="骑兵与平民关系",
        why_zh="caption 同时要求 mounted soldiers 和 fleeing civilians 时，reference 需要支撑骑兵、平民、村庄/撤离方向等关系，而不只是军人风格。",
        required_if=_needs_mounted_relation,
        evidence_terms=("mounted", "cavalry", "horse", "horses", "civilian", "civilians", "refugee", "village", "evacuation"),
    ),
    ReferenceRole(
        name="aircraft",
        label_zh="飞机/空中主体",
        why_zh="caption 点名 aircraft/airplane/plane 时，reference 需要支撑空中飞机主体，避免模型凭模板乱补。",
        required_if=lambda caption, route: _caption_has_any(caption, "aircraft", "airplane", "airplanes", "plane", "planes"),
        evidence_terms=("aircraft", "airplane", "airplanes", "plane", "planes", "airmen", "aviation", "bomber"),
    ),
)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("status") or "") for row in rows)
    return {
        "total": len(rows),
        "pass": counts.get("pass", 0),
        "fail": counts.get("fail", 0),
        "provider_reference_fallback": sum(1 for row in rows if "provider_reference_fallback" in row["risk_flags"]),
    }


def _missing_role_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(str(item) for item in row.get("missing_required_roles", []))
    return [{"role": role, "count": count} for role, count in counts.most_common()]


def _render_row_card(
    row: dict[str, Any],
    *,
    out_html: Path,
    repo_root: str | Path | None,
) -> str:
    status = str(row.get("status") or "")
    missing = ", ".join(str(item) for item in row.get("missing_required_roles", [])) or "无"
    flags = ", ".join(str(item) for item in row.get("risk_flags", [])) or "无"
    checks = "\n".join(_render_check(check) for check in row.get("role_checks", []))
    thumbs = "\n".join(
        _render_thumb(path, out_html=out_html, repo_root=repo_root)
        for path in row.get("reference_assets", [])
    )
    return f"""
<article class="card">
  <div class="head">
    <span class="sid">{_escape(row.get('sample_id'))}</span>
    <span class="badge {status}">{_escape(status)}</span>
    <span class="badge">缺失: {_escape(missing)}</span>
    <span class="badge">风险: {_escape(flags)}</span>
  </div>
  <p class="caption">{_escape(row.get('caption'))}</p>
  <div class="checks">{checks}</div>
  <div class="thumbs">{thumbs}</div>
</article>"""


def _render_check(check: dict[str, Any]) -> str:
    css = "ok" if check.get("status") == "pass" else "missing"
    evidence = ", ".join(Path(str(item)).name for item in check.get("evidence_assets", [])) or "无"
    return f"""
<div class="check">
  <strong>{_escape(check.get('label_zh'))} <span class="{css}">{_escape(check.get('status'))}</span></strong>
  <div>{_escape(check.get('why_zh'))}</div>
  <div>证据: {_escape(evidence)}</div>
</div>"""


def _render_thumb(path_value: Any, *, out_html: Path, repo_root: str | Path | None) -> str:
    path = _resolve_asset_path(path_value, repo_root=repo_root)
    label = Path(str(path_value)).name
    if not path.exists() or not path.is_file():
        return f"<figure><figcaption>{_escape(label)} missing</figcaption></figure>"
    src = os.path.relpath(path, out_html.parent)
    return f"""
<figure>
  <img src="{_escape(src)}" alt="{_escape(label)}">
  <figcaption>{_escape(label)}</figcaption>
</figure>"""


def _resolve_asset_path(path_value: Any, *, repo_root: str | Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (Path(repo_root) if repo_root is not None else Path.cwd()) / path


def _metric(label: str, value: Any, *, bad: bool = False) -> str:
    cls = "metric bad" if bad else "metric"
    return f'<div class="{cls}"><strong>{_escape(value)}</strong><span>{_escape(label)}</span></div>'


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
