"""Create a new deployment-specific frozen plan; never edit the archived plan."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_url(value):
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("Use an http(s) verifier URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Keep credentials and query strings out of experiment URLs")
    return value.rstrip("/")


def prepare(destination, server, proxy_server):
    archive = ROOT / "data/table4"
    destination.mkdir(parents=True, exist_ok=False)
    protocol_dir = destination / "protocol"
    protocol_dir.mkdir()
    for name in ("gold_answers.json", "gsm8k_test.jsonl"):
        shutil.copy2(archive / name, destination / name)
    shutil.copy2(archive / "protocol/calibration-27.json", protocol_dir / "calibration-27.json")
    code_hashes = {str(p.relative_to(ROOT)): digest(p)
                   for directory in ("src", "hardware") for p in sorted((ROOT / directory).glob("*.py"))}
    for source in sorted((archive / "protocol").glob("capability-*-prompts.jsonl")):
        shutil.copy2(source, protocol_dir / source.name)
    for source in [archive / "protocol/protocol.json", *sorted((archive / "protocol").glob("capability-*-plan.json"))]:
        plan = json.loads(source.read_text())
        plan["archive_code_sha256"] = plan.pop("code_sha256")
        plan["code_sha256"] = code_hashes
        plan["runtime_launcher_sha256"] = digest(ROOT / "scripts/launch_target.py")
        plan["calibration_sha256"] = digest(protocol_dir / "calibration-27.json")
        plan["public_reproduction"] = "new deployment; archived parameters and calibration retained"
        for condition in plan["conditions"]:
            condition["server"] = server if condition["condition_id"] == "lan_direct" else proxy_server
        if "prompts_file" in plan:
            plan["prompts_sha256"] = digest(protocol_dir / plan["prompts_file"])
        (protocol_dir / source.name).write_text(json.dumps(plan, indent=2) + "\n")
    frozen = {"scope": "new public-code hardware campaign", "scoring_sha256": digest(ROOT / "experiments/table4/score.py"),
              "files": {str(p.relative_to(destination)): digest(p) for p in sorted(protocol_dir.iterdir())}}
    (destination / "frozen_manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (destination / "campaign.json").write_text(json.dumps({"completed": False, "sessions": []}, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=validate_url)
    parser.add_argument("--proxy-server", required=True, type=validate_url)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.out_dir.resolve(), args.server, args.proxy_server))


if __name__ == "__main__":
    main()
