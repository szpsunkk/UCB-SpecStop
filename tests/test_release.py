"""Release invariants: preserve inputs, fail on damage, and bind plans to code."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.check_results import compare
from scripts.prepare_hardware import prepare, validate_url
from scripts.verify_data import verify


def test_archive_integrity_detects_modified_measurements(tmp_path):
    (tmp_path / "data").mkdir()
    sample = tmp_path / "data/sample.csv"
    sample.write_text("cost,tokens\n10,2\n")
    manifest = {"files": {"data/sample.csv": hashlib.sha256(sample.read_bytes()).hexdigest()}}
    (tmp_path / "data/SHA256SUMS.json").write_text(json.dumps(manifest))
    assert verify(tmp_path) == 1
    sample.write_text("cost,tokens\n1,2\n")
    with pytest.raises(ValueError, match="modified"):
        verify(tmp_path)


def test_fresh_plan_preserves_statistics_and_hashes(tmp_path):
    out = prepare(tmp_path / "campaign", "http://cloud.example.invalid:8000", "http://127.0.0.1:18082")
    plan = json.loads((out / "protocol/capability-01-plan.json").read_text())
    assert plan["generation_token_cap"] == 512 and plan["calibration_cells"] == 27
    assert plan["conditions"][0]["server"] == "http://cloud.example.invalid:8000"
    assert plan["conditions"][1]["server"] == "http://127.0.0.1:18082"
    root = Path(__file__).resolve().parents[1]
    for name, expected in plan["code_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert not json.loads((out / "campaign.json").read_text())["completed"]
    with pytest.raises(FileExistsError):
        prepare(out, "http://cloud.example.invalid", "http://127.0.0.1:18082")


def test_numerical_regression_check_rejects_changed_result():
    compare({"value": "1.234"}, {"value": 1.234})
    with pytest.raises(AssertionError):
        compare({"value": 1.235}, {"value": 1.234})


def test_verifier_url_rejects_embedded_credentials():
    with pytest.raises(Exception, match="credentials"):
        validate_url("http://" + "user:example" + "@cloud.example.invalid:8000")
