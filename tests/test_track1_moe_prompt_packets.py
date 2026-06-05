import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from affectiveart.track1_moe_prompt_packets import (
    build_moe_prompt_packet,
    build_moe_prompt_packets,
    load_contracts,
    load_distribution_routes,
    load_routes,
    render_provider_text_section,
    write_moe_prompt_packet_reports,
)
from affectiveart.track1_prompt_lint import lint_prompt_batch
from affectiveart.track1_reference_text_bank import build_reference_text_packet


def _contract(**overrides):
    row = {
        "sample_id": "track1_0803",
        "caption": (
            "A Socialist Realism propaganda poster with the Soviet, American, and British "
            "flags flying above a Kremlin tower crowned by a red star, framed by dramatic "
            "blue searchlight beams, bold Russian text, and a solemn wartime color palette."
        ),
        "aspect_plan": {
            "label": "portrait_poster",
            "width": 768,
            "height": 1024,
            "prompt_directive": "Use a portrait poster canvas; keep the full printed poster front-facing.",
        },
        "reference_contract": {"required": True, "categories": ["landmark", "flags", "architecture"]},
        "text_contract": {"required": True, "modes": ["cyrillic"]},
        "relation_contract": {"required": True, "checks": ["movement_direction"]},
        "fid_risk": ["poster_distribution_sensitive", "reference_style_drift"],
    }
    row.update(overrides)
    return row


def _route(**overrides):
    row = {
        "sample_id": "track1_0803",
        "primary_expert": "poster_expert",
        "support_experts": ["reference_expert", "text_symbol_expert", "logic_expert", "fid_guard_expert"],
        "recommended_model": "gemini-3-pro-image",
        "fallback_model": "gemini-3.1-flash-image",
        "candidate_count": 4,
        "priority_bucket": "tier1_pro_reference",
        "queue_score": 12.9,
        "reference_level": "image_reference_required",
    }
    row.update(overrides)
    return row


def _distribution_route(**overrides):
    row = {
        "sample_id": "track1_0803",
        "caption": _contract()["caption"],
        "family_id": "kremlin_red_square",
        "hard_constraints": ["caption_content", "landmark_reference", "requested_text_policy"],
        "style_freedom": [
            "aspect_variation_allowed",
            "brushwork_variation_allowed",
            "viewpoint_variation_allowed",
        ],
        "composition_hints": ["red brick tower", "night scene", "searchlight beams"],
        "medium_options": ["aged lithograph", "oil-paint realism", "poster paint"],
        "aspect_hints": [{"label": "landscape", "count": 3}],
        "reference_assets": ["/tmp/raw/reference/path.jpg"],
        "candidate_strategies": [
            {"strategy": "aas_safe", "constraint_strength": "high", "freedom_strength": "low"},
            {"strategy": "reference_style", "constraint_strength": "medium_high", "freedom_strength": "medium"},
            {"strategy": "fid_diverse", "constraint_strength": "medium", "freedom_strength": "high"},
        ],
    }
    row.update(overrides)
    return row


class Track1MoePromptPacketsTest(unittest.TestCase):
    def test_provider_prompt_excludes_routing_metadata_and_reference_queries(self):
        contract = _contract()
        route = _route()
        text_packet = build_reference_text_packet(contract["sample_id"], contract["caption"])

        packet = build_moe_prompt_packet(rank=1, route=route, contract=contract, reference_text_packet=text_packet)
        prompt = packet["provider_prompt"]

        self.assertNotIn("Generate 4 candidates", prompt)
        self.assertNotIn("EXPERT ROUTING CONSTRAINTS", prompt)
        self.assertNotIn("Recommended generation model", prompt)
        self.assertNotIn("gemini-3-pro-image", prompt)
        self.assertNotIn("Reference queries for future grounding", prompt)
        self.assertNotIn("French flags", prompt)
        self.assertNotIn("track1_0803", prompt)
        self.assertNotIn("Variant:", prompt)
        self.assertNotIn("moe_", prompt)
        self.assertEqual(packet["review_metadata"]["candidate_count"], 4)
        self.assertEqual(packet["review_metadata"]["recommended_model"], "gemini-3-pro-image")

    def test_provider_prompt_adds_neutral_historical_context_for_block_prone_symbols(self):
        contract = _contract()
        route = _route()

        packet = build_moe_prompt_packet(rank=1, route=route, contract=contract)
        prompt = packet["provider_prompt"]

        self.assertIn("NEUTRAL HISTORICAL ART CONTEXT", prompt)
        self.assertIn("not political advocacy", prompt)
        self.assertIn("caption-required visual attributes", prompt)

    def test_document_text_section_prefers_document_terms_not_generic_victory_slogans(self):
        caption = (
            "A Socialist Realism propaganda poster with bold Cyrillic typography, a central "
            "surrender declaration document, and four military portraits framed by Soviet, "
            "American, British, and French flags on a mint-green vintage paper background."
        )
        text_packet = build_reference_text_packet("track1_0721", caption)

        section = render_provider_text_section(text_packet, caption=caption)

        self.assertIn("АКТ О КАПИТУЛЯЦИИ", section)
        self.assertIn("document title", section)
        self.assertNotIn("ПОБЕДА БУДЕТ ЗА НАМИ!", section)
        self.assertNotIn("Reference queries", section)

    def test_build_packets_writes_provider_prompt_files_separate_from_metadata(self):
        contract = _contract()
        route = _route()
        text_packet = build_reference_text_packet(contract["sample_id"], contract["caption"])
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={contract["sample_id"]: text_packet},
                out_dir=Path(tmp),
            )

            prompt_path = Path(rows[0]["provider_prompt_path"])
            prompt_text = prompt_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 1)
        self.assertTrue(prompt_path.name.endswith("_track1_0803.txt"))
        self.assertNotIn("candidate_count", prompt_text)
        self.assertIn("review_metadata", rows[0])

    def test_write_reports_and_cli(self):
        contract = _contract()
        route = _route()
        text_packet = build_reference_text_packet(contract["sample_id"], contract["caption"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route_json = root / "routes.json"
            contract_json = root / "contracts.json"
            bank_jsonl = root / "bank.jsonl"
            out_dir = root / "out"
            route_json.write_text(json.dumps({"rows": [route]}), encoding="utf-8")
            contract_json.write_text(json.dumps({"rows": [contract]}), encoding="utf-8")
            bank_jsonl.write_text(json.dumps(text_packet, ensure_ascii=False) + "\n", encoding="utf-8")

            rows = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={contract["sample_id"]: text_packet},
                out_dir=out_dir,
            )
            write_moe_prompt_packet_reports(rows, out_dir=out_dir)
            self.assertTrue((out_dir / "track1_moe_prompt_packets.json").exists())

            path = Path(__file__).resolve().parents[1] / "scripts" / "track1_moe_prompt_packets.py"
            spec = importlib.util.spec_from_file_location("track1_moe_prompt_packets_cli", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--routes-json",
                    str(route_json),
                    "--contract-json",
                    str(contract_json),
                    "--reference-text-bank-jsonl",
                    str(bank_jsonl),
                    "--out-dir",
                    str(out_dir),
                    "--limit",
                    "3",
                ]
            )
            payload = json.loads((out_dir / "track1_moe_prompt_packets.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["total"], 1)

    def test_distribution_route_expands_three_candidate_strategies(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = _distribution_route()

        with tempfile.TemporaryDirectory() as tmp:
            rows = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={},
                distribution_routes={contract["sample_id"]: distribution_route},
                out_dir=Path(tmp),
                max_strategies_per_sample=3,
            )
            prompt_names = sorted(path.name for path in (Path(tmp) / "provider_prompts_top").glob("*.txt"))

        self.assertEqual([row["candidate_strategy"] for row in rows], ["aas_safe", "reference_style", "fid_diverse"])
        self.assertEqual(
            prompt_names,
            [
                "01_track1_0803_aas_safe.txt",
                "02_track1_0803_reference_style.txt",
                "03_track1_0803_fid_diverse.txt",
            ],
        )
        self.assertIn("red brick tower", rows[1]["provider_prompt"])
        self.assertNotIn("portrait poster canvas", rows[2]["provider_prompt"].lower())
        self.assertEqual(rows[0]["review_metadata"]["family_id"], "kremlin_red_square")
        self.assertEqual(rows[0]["review_metadata"]["candidate_strategy"], "aas_safe")

    def test_default_behavior_remains_one_packet_without_distribution_routes(self):
        contract = _contract()
        route = _route(candidate_count=4)
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={},
                out_dir=Path(tmp),
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_strategy"], "legacy")

    def test_max_strategies_per_sample_truncates_candidate_strategy_order(self):
        contract = _contract()
        route = _route(candidate_count=3)
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={},
                distribution_routes={contract["sample_id"]: _distribution_route()},
                out_dir=Path(tmp),
                max_strategies_per_sample=2,
            )

        self.assertEqual([row["candidate_strategy"] for row in rows], ["aas_safe", "reference_style"])

    def test_fid_diverse_softens_generic_template_phrases_but_keeps_caption_and_family_hints(self):
        contract = _contract()
        route = _route(candidate_count=3)

        packet = build_moe_prompt_packet(
            rank=1,
            route=route,
            contract=contract,
            distribution_route=_distribution_route(),
            candidate_strategy="fid_diverse",
        )
        prompt = packet["provider_prompt"].lower()

        for phrase in [
            "portrait poster canvas",
            "front-facing",
            "flat printed poster",
            "medium-distance figures",
            "fewer, larger",
            "internal poster margins",
        ]:
            self.assertNotIn(phrase, prompt)
        self.assertIn("kremlin tower", prompt)
        self.assertIn("red brick tower", prompt)
        self.assertIn("aged lithograph", prompt)

    def test_distribution_strategy_prompt_excludes_internal_route_metadata(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = _distribution_route(
            sample_id="track1_0803",
            recommended_model="gemini-3-pro-image",
            candidate_count=9,
            primary_expert="poster_expert",
            priority_bucket="private_bucket",
            reference_path="/tmp/reference/private.jpg",
        )

        packet = build_moe_prompt_packet(
            rank=1,
            route=route,
            contract=contract,
            distribution_route=distribution_route,
            candidate_strategy="reference_style",
        )
        prompt = packet["provider_prompt"]

        for forbidden in [
            "track1_0803",
            "gemini-3-pro-image",
            "candidate_count",
            "primary_expert",
            "priority_bucket",
            "reference_path",
            "/tmp/reference/private.jpg",
            "candidate_strategies",
        ]:
            self.assertNotIn(forbidden, prompt)

    def test_distribution_reference_assets_are_kept_as_generation_metadata_only(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = _distribution_route(
            reference_assets=[
                "/tmp/emoart/reference/kremlin_a.jpg",
                "/tmp/emoart/reference/kremlin_b.jpg",
            ]
        )

        packet = build_moe_prompt_packet(
            rank=1,
            route=route,
            contract=contract,
            distribution_route=distribution_route,
            candidate_strategy="reference_style",
        )

        self.assertEqual(packet["reference_assets"], distribution_route["reference_assets"])
        self.assertEqual(packet["review_metadata"]["reference_asset_count"], 2)
        self.assertIn("attached reference board", packet["provider_prompt"])
        self.assertNotIn("/tmp/emoart/reference", packet["provider_prompt"])

    def test_load_distribution_routes_accepts_routes_rows_and_list_payloads(self):
        route = _distribution_route()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes_json = root / "routes.json"
            rows_json = root / "rows.json"
            list_json = root / "list.json"
            routes_json.write_text(json.dumps({"routes": [route], "summary": {"total": 1}}), encoding="utf-8")
            rows_json.write_text(json.dumps({"rows": [route]}), encoding="utf-8")
            list_json.write_text(json.dumps([route]), encoding="utf-8")

            self.assertEqual(load_distribution_routes(routes_json)[route["sample_id"]]["family_id"], "kremlin_red_square")
            self.assertEqual(load_distribution_routes(rows_json)[route["sample_id"]]["family_id"], "kremlin_red_square")
            self.assertEqual(load_distribution_routes(list_json)[route["sample_id"]]["family_id"], "kremlin_red_square")

    def test_cli_accepts_distribution_routes_json_and_writes_expanded_reports(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = _distribution_route()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route_json = root / "routes.json"
            contract_json = root / "contracts.json"
            distribution_json = root / "distribution.json"
            out_dir = root / "out"
            route_json.write_text(json.dumps({"rows": [route]}), encoding="utf-8")
            contract_json.write_text(json.dumps({"rows": [contract]}), encoding="utf-8")
            distribution_json.write_text(json.dumps({"routes": [distribution_route]}), encoding="utf-8")

            path = Path(__file__).resolve().parents[1] / "scripts" / "track1_moe_prompt_packets.py"
            spec = importlib.util.spec_from_file_location("track1_moe_prompt_packets_cli_dist", path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            code = module.main(
                [
                    "--routes-json",
                    str(route_json),
                    "--contract-json",
                    str(contract_json),
                    "--distribution-routes-json",
                    str(distribution_json),
                    "--max-strategies-per-sample",
                    "3",
                    "--out-dir",
                    str(out_dir),
                    "--limit",
                    "3",
                ]
            )
            payload = json.loads((out_dir / "track1_moe_prompt_packets.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["summary"]["total"], 3)
        self.assertEqual([row["candidate_strategy"] for row in payload["packets"]], ["aas_safe", "reference_style", "fid_diverse"])

    def test_cli_strategy_mode_controls_distribution_expansion(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_route = _distribution_route()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route_json = root / "routes.json"
            contract_json = root / "contracts.json"
            distribution_json = root / "distribution.json"
            legacy_out = root / "legacy"
            distribution_out = root / "distribution"
            route_json.write_text(json.dumps({"rows": [route]}), encoding="utf-8")
            contract_json.write_text(json.dumps({"rows": [contract]}), encoding="utf-8")
            distribution_json.write_text(json.dumps({"routes": [distribution_route]}), encoding="utf-8")

            module = self._load_cli_module("track1_moe_prompt_packets_cli_strategy")
            self.assertIn("--strategy-mode", module.build_parser().format_help())
            legacy_code = module.main(
                [
                    "--routes-json",
                    str(route_json),
                    "--contract-json",
                    str(contract_json),
                    "--distribution-routes-json",
                    str(distribution_json),
                    "--strategy-mode",
                    "legacy",
                    "--out-dir",
                    str(legacy_out),
                    "--limit",
                    "3",
                ]
            )
            distribution_code = module.main(
                [
                    "--routes-json",
                    str(route_json),
                    "--contract-json",
                    str(contract_json),
                    "--distribution-routes-json",
                    str(distribution_json),
                    "--strategy-mode",
                    "distribution",
                    "--out-dir",
                    str(distribution_out),
                    "--limit",
                    "3",
                ]
            )
            legacy_payload = json.loads((legacy_out / "track1_moe_prompt_packets.json").read_text(encoding="utf-8"))
            distribution_payload = json.loads((distribution_out / "track1_moe_prompt_packets.json").read_text(encoding="utf-8"))

        self.assertEqual(legacy_code, 0)
        self.assertEqual(distribution_code, 0)
        self.assertEqual([row["candidate_strategy"] for row in legacy_payload["packets"]], ["legacy"])
        self.assertEqual(
            [row["candidate_strategy"] for row in distribution_payload["packets"]],
            ["aas_safe", "reference_style", "fid_diverse"],
        )

    def test_distribution_expanded_fixture_does_not_fail_lint_from_global_template_phrases(self):
        contracts = [_contract(sample_id=f"track1_080{index}") for index in range(3)]
        routes = [_route(sample_id=contract["sample_id"], candidate_count=3) for contract in contracts]
        distribution_routes = {
            contract["sample_id"]: _distribution_route(sample_id=contract["sample_id"])
            for contract in contracts
        }

        with tempfile.TemporaryDirectory() as tmp:
            rows = build_moe_prompt_packets(
                routes,
                contracts={contract["sample_id"]: contract for contract in contracts},
                reference_text_bank={},
                distribution_routes=distribution_routes,
                out_dir=Path(tmp),
                max_strategies_per_sample=3,
            )

        report = lint_prompt_batch(rows, threshold=0.6)

        self.assertEqual(report["summary"]["failed_phrases"], [])

    def test_limit_caps_final_emitted_packets_after_strategy_expansion(self):
        contract = _contract()
        route = _route(candidate_count=3)
        distribution_routes = {contract["sample_id"]: _distribution_route()}

        with tempfile.TemporaryDirectory() as tmp:
            one = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={},
                distribution_routes=distribution_routes,
                out_dir=Path(tmp) / "one",
                limit=1,
            )
            two = build_moe_prompt_packets(
                [route],
                contracts={contract["sample_id"]: contract},
                reference_text_bank={},
                distribution_routes=distribution_routes,
                out_dir=Path(tmp) / "two",
                limit=2,
            )

        self.assertEqual([row["candidate_strategy"] for row in one], ["aas_safe"])
        self.assertEqual([row["candidate_strategy"] for row in two], ["aas_safe", "reference_style"])

    def test_summary_candidate_budget_matches_emitted_packets_not_repeated_source_budget(self):
        rows = [
            {
                "rank": index,
                "sample_id": "track1_0803",
                "caption": "caption",
                "candidate_strategy": strategy,
                "provider_prompt": "prompt",
                "provider_prompt_path": "",
                "review_metadata": {
                    "recommended_model": "model",
                    "candidate_count": 4,
                    "primary_expert": "poster_expert",
                    "support_experts": [],
                },
            }
            for index, strategy in enumerate(["aas_safe", "reference_style", "fid_diverse"], start=1)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_moe_prompt_packet_reports(rows, out_dir=out_dir)
            payload = json.loads((out_dir / "track1_moe_prompt_packets.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["summary"]["total"], 3)
        self.assertEqual(payload["summary"]["candidate_budget"], 3)
        self.assertEqual(payload["summary"]["source_candidate_budget"], 4)

    def test_protected_submission_outputs_are_rejected_before_creating_files(self):
        repo_root = Path(__file__).resolve().parents[1]
        protected_out = repo_root / "submissions" / "track1" / "images" / "task4_guard_tmp"
        self.assertFalse(protected_out.exists())

        with self.assertRaisesRegex(ValueError, "protected"):
            build_moe_prompt_packets(
                [_route()],
                contracts={_contract()["sample_id"]: _contract()},
                reference_text_bank={},
                out_dir=protected_out,
            )
        with self.assertRaisesRegex(ValueError, "protected"):
            write_moe_prompt_packet_reports([], out_dir=protected_out)

        self.assertFalse(protected_out.exists())

    def test_load_routes_and_contracts_accept_list_payloads(self):
        route = _route()
        contract = _contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes_json = root / "routes.json"
            contracts_json = root / "contracts.json"
            routes_json.write_text(json.dumps([route]), encoding="utf-8")
            contracts_json.write_text(json.dumps([contract]), encoding="utf-8")

            self.assertEqual(load_routes(routes_json), [route])
            self.assertEqual(load_contracts(contracts_json)[contract["sample_id"]]["caption"], contract["caption"])

    def test_load_routes_rejects_malformed_payloads_with_indexed_errors(self):
        malformed_payloads = [
            ({"routes": [_route()]}, "routes JSON"),
            (["not an object"], "route row 0"),
            ([{"candidate_count": 1}], "route row 0.*sample_id"),
            ([_route(candidate_count="many")], "route row 0.*candidate_count"),
            ([_route(candidate_count=None)], "route row 0.*candidate_count"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (payload, pattern) in enumerate(malformed_payloads):
                path = root / f"bad_routes_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=index):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_routes(path)

    def test_load_contracts_rejects_malformed_payloads_with_indexed_errors(self):
        malformed_payloads = [
            ({"contracts": [_contract()]}, "contracts JSON"),
            (["not an object"], "contract row 0"),
            ([{"caption": "caption"}], "contract row 0.*sample_id"),
            ([_contract(caption="")], "contract row 0.*caption"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (payload, pattern) in enumerate(malformed_payloads):
                path = root / f"bad_contracts_{index}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=index):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_contracts(path)

    def test_build_packets_rejects_missing_contract_before_creating_prompt_dir(self):
        route = _route(sample_id="track1_missing", candidate_count=1)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "track1_missing"):
                build_moe_prompt_packets(
                    [route],
                    contracts={},
                    reference_text_bank={},
                    out_dir=out_dir,
                )

            self.assertFalse((out_dir / "provider_prompts_top").exists())

    def test_cli_returns_two_for_malformed_inputs_without_prompt_dir(self):
        contract = _contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route_json = root / "bad_routes.json"
            contract_json = root / "contracts.json"
            out_dir = root / "out"
            route_json.write_text(json.dumps([{"candidate_count": 1}]), encoding="utf-8")
            contract_json.write_text(json.dumps([contract]), encoding="utf-8")

            module = self._load_cli_module("track1_moe_prompt_packets_cli_bad_inputs")
            code = module.main(
                [
                    "--routes-json",
                    str(route_json),
                    "--contract-json",
                    str(contract_json),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(code, 2)
            self.assertFalse((out_dir / "provider_prompts_top").exists())

    def _load_cli_module(self, name: str):
        path = Path(__file__).resolve().parents[1] / "scripts" / "track1_moe_prompt_packets.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module


if __name__ == "__main__":
    unittest.main()
