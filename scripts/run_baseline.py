#!/usr/bin/env python3
"""Run the saved task selection with the recorded MasDrift pilot settings."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/masdrift"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without API calls.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/healthcare_baseline_20")
    parser.add_argument("--case-id", help="Run just one of the 20 selected tasks.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/healthcare_baseline_20.json").read_text())
    task_ids = json.loads((ROOT / config["task_ids"]).read_text())
    if args.case_id:
        if args.case_id not in task_ids:
            parser.error("--case-id must belong to data/selected_task_ids.json")
        task_ids = [args.case_id]
    output = args.out_dir.resolve()
    if output == (ROOT / "results").resolve() or (ROOT / "results").resolve() in output.parents:
        parser.error("Use outputs/ for new runs; results/ contains the published pilot.")
    # Prevent inherited endpoint pools or role overrides from changing the baseline.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MAS_LEAD_", "MAS_DELEGATE_", "MAS_LLM_", "OPENAI_"))}
    env["MAS_LLM_PROVIDER"] = config["provider"]
    env["OPENAI_MODEL"] = config["model"]
    if os.environ.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if not args.dry_run and not env.get("OPENAI_API_KEY"):
        parser.error("Export OPENAI_API_KEY before running the API-hosted baseline.")
    if not args.dry_run and (UPSTREAM / ".env").exists():
        parser.error("Remove vendor/masdrift/.env and use exported credentials to avoid hidden overrides.")
    for case_id in task_ids:
        cmd = [sys.executable, str(UPSTREAM / "main.py"),
               "--dataset", str(ROOT / config["dataset"]), "--case-id", case_id,
               "--structure", config["structure"], "--layers", str(config["layers"]),
               "--width", str(config["width"]), "--personas", config["personas"],
               "--visibility", config["visibility"], "--defense", config["defense"],
               "--llm-provider", config["provider"], "--llm-model", config["model"],
               "--llm-api-keys-env", "OPENAI_API_KEY", "--out-dir", str(output)]
        if args.resume:
            cmd.append("--resume")
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=UPSTREAM, env=env, check=True)


if __name__ == "__main__":
    main()
