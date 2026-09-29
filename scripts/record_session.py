"""Register a completed hardware session after checking its execution manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--session", required=True, choices=["capability-%02d" % i for i in range(1, 9)])
    args = parser.parse_args()
    folder = args.campaign / "hardware_run" / args.session
    manifest = json.loads((folder / "run_manifest.json").read_text())
    plan = args.campaign / "protocol" / (args.session + "-plan.json")
    if not manifest["completed"] or manifest["failure_rows"] or manifest["output_mismatches"]:
        raise ValueError("Session did not pass completion and output equality checks")
    if manifest["request_count"] != 72 or manifest["plan_sha256"] != hashlib.sha256(plan.read_bytes()).hexdigest():
        raise ValueError("Request count or plan digest differs")
    path = args.campaign / "campaign.json"
    campaign = json.loads(path.read_text())
    if any(s["name"] == args.session for s in campaign["sessions"]):
        raise ValueError("Session already registered")
    campaign["sessions"].append({"name": args.session, "requests": manifest["request_count"]})
    campaign["completed"] = len(campaign["sessions"]) == 8
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(campaign, indent=2) + "\n")
    temporary.replace(path)
    print("Registered " + args.session)


if __name__ == "__main__":
    main()
