from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from PIL import Image

from affectiveart.track1_p1_light_proxy import (
    evaluate_package,
    evaluate_packages,
    load_known_placeholder_sha256s,
    load_replacement_sample_ids,
    load_route_index,
    parse_package_spec,
    write_p1_light_reports,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, quality=95)


def _submission_rows() -> list[dict[str, str]]:
    return [
        {"sample_id": "track1_0001", "path": "images/track1_0001.jpg"},
        {"sample_id": "track1_0002", "path": "images/track1_0002.jpg"},
    ]


def test_evaluate_package_hard_rejects_known_placeholder(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_image(baseline_dir / "track1_0001.jpg", (50, 60, 70))
    _write_image(baseline_dir / "track1_0002.jpg", (80, 90, 100))
    _write_image(candidate_dir / "track1_0001.jpg", (120, 10, 10))
    _write_image(candidate_dir / "track1_0002.jpg", (80, 90, 100))
    placeholder_digest = sha256((candidate_dir / "track1_0001.jpg").read_bytes()).hexdigest()

    report = evaluate_package(
        "candidate",
        _submission_rows(),
        candidate_dir,
        baseline_rows=_submission_rows(),
        baseline_image_dir=baseline_dir,
        route_index={
            "track1_0001": {"reference_style": "Socialist Realism", "caption": "poster"},
            "track1_0002": {"reference_style": "Ink and wash painting", "caption": "scroll"},
        },
        known_placeholder_sha256s={placeholder_digest: "provider_json_placeholder"},
    )

    assert report["summary"]["status"] == "reject_known_placeholder"
    assert report["summary"]["changed_sample_count"] == 1
    assert report["summary"]["known_placeholder_hit_count"] == 1
    assert report["summary"]["known_placeholder_samples"] == ["track1_0001"]
    changed = [row for row in report["rows"] if row["changed"]]
    assert changed[0]["known_placeholder_type"] == "provider_json_placeholder"


def test_evaluate_package_reports_changed_style_concentration(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_image(baseline_dir / "track1_0001.jpg", (10, 20, 30))
    _write_image(baseline_dir / "track1_0002.jpg", (70, 80, 90))
    _write_image(candidate_dir / "track1_0001.jpg", (120, 130, 140))
    _write_image(candidate_dir / "track1_0002.jpg", (70, 80, 90))

    report = evaluate_package(
        "candidate",
        _submission_rows(),
        candidate_dir,
        baseline_rows=_submission_rows(),
        baseline_image_dir=baseline_dir,
        route_index={
            "track1_0001": {
                "reference_style": "Socialist Realism",
                "caption": "A Soviet propaganda poster with bold typography.",
                "hard_constraints": ["caption_content", "requested_text_policy"],
                "reference_assets": ["ref_a.jpg", "ref_b.jpg"],
            },
            "track1_0002": {
                "reference_style": "Ink and wash painting",
                "caption": "A calm ink wash landscape scroll.",
            },
        },
        known_placeholder_sha256s={},
    )

    summary = report["summary"]
    assert summary["status"] == "candidate_ok_for_next_gate"
    assert summary["changed_sample_count"] == 1
    assert summary["changed_style_counts"] == {"Socialist Realism": 1}
    assert summary["changed_poster_like_count"] == 1
    assert summary["max_changed_style_concentration"] == 1.0
    changed = [row for row in report["rows"] if row["changed"]][0]
    assert changed["reference_style"] == "Socialist Realism"
    assert changed["hard_constraint_count"] == 2
    assert changed["reference_asset_count"] == 2


def test_evaluate_package_uses_manifest_replacement_ids_over_hash_differences(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_image(baseline_dir / "track1_0001.jpg", (10, 20, 30))
    _write_image(baseline_dir / "track1_0002.jpg", (70, 80, 90))
    _write_image(candidate_dir / "track1_0001.jpg", (120, 130, 140))
    _write_image(candidate_dir / "track1_0002.jpg", (75, 85, 95))

    report = evaluate_package(
        "candidate",
        _submission_rows(),
        candidate_dir,
        baseline_rows=_submission_rows(),
        baseline_image_dir=baseline_dir,
        route_index={
            "track1_0001": {"reference_style": "Socialist Realism"},
            "track1_0002": {"reference_style": "Ink and wash painting"},
        },
        known_placeholder_sha256s={},
        replacement_sample_ids={"track1_0001"},
    )

    assert report["summary"]["changed_sample_count"] == 1
    changed_ids = [row["sample_id"] for row in report["rows"] if row["changed"]]
    assert changed_ids == ["track1_0001"]


def test_load_route_index_and_write_reports(tmp_path: Path) -> None:
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "sample_id": "track1_0001",
                        "caption": "caption",
                        "reference_asset_notes": [
                            "official EmoArt-130k retrieval; style=Socialist Realism; score=12.0"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    route_index = load_route_index(routes_json)
    assert route_index["track1_0001"]["reference_style"] == "Socialist Realism"

    report = {
        "version": "track1_p1_light_proxy_v1",
        "summary": {
            "package_count": 1,
            "recommended_package": "candidate",
        },
        "packages": [
            {
                "package": "candidate",
                "summary": {
                    "status": "candidate_ok_for_next_gate",
                    "changed_sample_count": 1,
                    "known_placeholder_hit_count": 0,
                    "max_changed_style_concentration": 1.0,
                },
            }
        ],
    }
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    write_p1_light_reports(report, out_json, out_md)
    assert json.loads(out_json.read_text(encoding="utf-8"))["summary"]["recommended_package"] == "candidate"
    assert "Track1 P1-Light Proxy" in out_md.read_text(encoding="utf-8")


def test_parse_package_spec_requires_name_submission_and_image_dir() -> None:
    name, submission_json, image_dir = parse_package_spec(
        "b80=submissions/b80/submission.json=submissions/b80/images"
    )
    assert name == "b80"
    assert str(submission_json) == "submissions/b80/submission.json"
    assert str(image_dir) == "submissions/b80/images"


def test_load_replacement_ids_and_blocked_placeholder_hashes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "accepted_replacements": [
                    {"sample_id": "track1_0002"},
                    {"sample_id": "track1_0001"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_replacement_sample_ids(manifest) == {"track1_0001", "track1_0002"}

    placeholders = tmp_path / "placeholders.json"
    placeholders.write_text(
        json.dumps({"blocked_hashes": {"abc": "bad_placeholder"}}),
        encoding="utf-8",
    )
    assert load_known_placeholder_sha256s(placeholders) == {"abc": "bad_placeholder"}


def test_evaluate_packages_recommends_clean_changed_candidate_over_noop_baseline(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_image(baseline_dir / "track1_0001.jpg", (10, 20, 30))
    _write_image(baseline_dir / "track1_0002.jpg", (70, 80, 90))
    _write_image(candidate_dir / "track1_0001.jpg", (120, 130, 140))
    _write_image(candidate_dir / "track1_0002.jpg", (70, 80, 90))

    report = evaluate_packages(
        [
            ("current", _submission_rows(), baseline_dir),
            ("candidate", _submission_rows(), candidate_dir, {"track1_0001"}),
        ],
        baseline_rows=_submission_rows(),
        baseline_image_dir=baseline_dir,
        route_index={"track1_0001": {"reference_style": "Socialist Realism"}},
        known_placeholder_sha256s={},
    )

    assert report["summary"]["recommended_package"] == "candidate"
    assert report["packages"][0]["package"] == "candidate"
