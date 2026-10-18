#!/usr/bin/env python3
"""Monitor and resume mas-eval judging for the two 600-task experiment grids.

Default repository targets:
  results/raw
  results/source

Each condition folder is judged by evaluate.py with the mas-eval backend. The
monitor treats a job as complete only when the output JSON exists, has the
expected number of runs, and every run has non-error LLM verdicts for C1/C2a/C2b.
Missing or invalid jobs are rerun until all jobs complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_EXPERIMENT_ROOTS = [
    REPO / "results" / "raw",
    REPO / "results" / "source",
]
DEFAULT_CONDITIONS = ["C1", "C2", "C2-t", "C3", "N2", "N4", "N4-t", "N8"]
REQUIRED_LLM_KEYS = [
    "c1_unauthorized_action",
    "c2a_over_disclosure",
    "c2b_constraint_lost",
]


@dataclass(frozen=True)
class Job:
    root: Path
    condition: str
    results_dir: Path
    out_file: Path
    expected_runs: int

    @property
    def label(self) -> str:
        return f"{self.root.name}/{self.condition}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def count_trace_files(results_dir: Path) -> int:
    count = 0
    for path in results_dir.glob("*.json"):
        if path.name.lower().startswith("eval"):
            continue
        count += 1
    return count


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def llm_error_count(summary: dict[str, Any]) -> int:
    return sum(1 for run in summary.get("runs") or [] if run_llm_incomplete(run))


def run_llm_incomplete(run: dict[str, Any]) -> bool:
    llm = run.get("llm")
    if not isinstance(llm, dict) or llm.get("error"):
        return True
    for key in REQUIRED_LLM_KEYS:
        verdict = llm.get(key)
        if not isinstance(verdict, dict) or not isinstance(
            verdict.get("violation"), bool
        ):
            return True
    if run.get("structure") == "single":
        c2b = llm["c2b_constraint_lost"]
        if (
            c2b.get("applicable") is not False
            or c2b.get("status") != "structural_not_applicable"
            or c2b.get("violation") is not False
            or c2b.get("lost_at_hop") is not None
            or c2b.get("trajectory") != []
        ):
            return True
    return False


def bad_run_ids(summary: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for run in summary.get("runs") or []:
        if not isinstance(run, dict) or not run_llm_incomplete(run):
            continue
        result_file = str(run.get("result_file") or "").strip()
        case_id = str(run.get("case_id") or "").strip()
        if result_file:
            ids.append(result_file)
        elif case_id:
            ids.append(case_id)
    return sorted({item for item in ids if item})


def repairable_summary(job: Job) -> tuple[dict[str, Any], list[str]]:
    if not job.out_file.exists():
        return {}, []
    try:
        summary = load_json(job.out_file)
    except Exception:
        return {}, []
    runs = summary.get("runs")
    if not isinstance(runs, list):
        return summary, []
    if len(runs) != job.expected_runs:
        return summary, []
    if int(summary.get("skipped") or 0):
        return summary, []
    return summary, bad_run_ids(summary)


def merge_repair_summary(job: Job, repair_out: Path) -> tuple[int, int]:
    from evaluate import _atomic_write_json, aggregate

    base = load_json(job.out_file)
    repair = load_json(repair_out)
    base_runs = base.get("runs")
    repair_runs = repair.get("runs")
    if not isinstance(base_runs, list):
        raise ValueError(f"base output has no runs list: {job.out_file}")
    if not isinstance(repair_runs, list):
        raise ValueError(f"repair output has no runs list: {repair_out}")

    repaired_by_file = {
        str(run.get("result_file")): run
        for run in repair_runs
        if isinstance(run, dict) and run.get("result_file")
    }
    merged_runs: list[dict[str, Any]] = []
    replaced = 0
    for run in base_runs:
        key = str(run.get("result_file")) if isinstance(run, dict) else ""
        if key and key in repaired_by_file:
            merged_runs.append(repaired_by_file[key])
            replaced += 1
        else:
            merged_runs.append(run)

    base_keys = {
        str(run.get("result_file"))
        for run in base_runs
        if isinstance(run, dict) and run.get("result_file")
    }
    for key, run in repaired_by_file.items():
        if key not in base_keys:
            merged_runs.append(run)
            replaced += 1

    merged = aggregate(merged_runs, True)
    merged["runs"] = merged_runs
    merged["skipped"] = int(base.get("skipped") or 0)
    if isinstance(base.get("checkpoint"), dict):
        merged["checkpoint"] = base["checkpoint"]
    _atomic_write_json(job.out_file, merged)
    return replaced, len(repair_runs)


def check_job(job: Job) -> tuple[bool, str, dict[str, Any]]:
    if not job.out_file.exists():
        return False, "missing output", {}
    try:
        summary = load_json(job.out_file)
    except Exception as exc:
        return False, f"bad output JSON: {type(exc).__name__}: {exc}", {}

    runs = summary.get("runs")
    if not isinstance(runs, list):
        return False, "output has no runs list", summary
    if len(runs) != job.expected_runs:
        return False, f"run count {len(runs)} != expected {job.expected_runs}", summary

    skipped = int(summary.get("skipped") or 0)
    if skipped:
        return False, f"skipped traces present: {skipped}", summary

    errors = llm_error_count(summary)
    if errors:
        return False, f"LLM errors/missing verdicts: {errors}", summary

    return True, "complete", summary


def build_jobs(args: argparse.Namespace) -> list[Job]:
    roots = [Path(p) for p in args.roots]
    jobs: list[Job] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"experiment root not found: {root}")
        for condition in args.conditions:
            results_dir = root / condition
            if not results_dir.exists():
                raise FileNotFoundError(f"condition folder not found: {results_dir}")
            expected = count_trace_files(results_dir)
            if expected <= 0:
                raise ValueError(f"no trace JSON files in {results_dir}")
            out_dir = Path(args.eval_root) / root.name / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"eval_{condition}{args.output_suffix}.json"
            jobs.append(Job(
                root=root,
                condition=condition,
                results_dir=results_dir,
                out_file=out_file,
                expected_runs=expected,
            ))
    return jobs


def run_job(job: Job, args: argparse.Namespace, attempt: int) -> int:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root_part = safe_name(job.root.name)
    log_path = log_dir / f"{root_part}_{safe_name(job.condition)}_attempt{attempt:03d}.log"
    _, repair_ids = repairable_summary(job)
    canonical_checkpoint = Path(f"{job.out_file}.checkpoint.jsonl")
    resume_mode = bool(repair_ids) and canonical_checkpoint.exists()
    # New evaluator outputs can repair themselves from the canonical per-trace
    # checkpoint. Legacy outputs (including a run started before checkpoint support)
    # still use the small include-list + merge path.
    repair_mode = bool(repair_ids) and not resume_mode
    repair_out: Path | None = None
    input_max_chars = int(args.mas_eval_input_max_chars)
    use_short_input = (
        repair_mode
        and 0 < len(repair_ids) <= int(args.fallback_repair_max_runs)
        and attempt >= int(args.fallback_after_attempts)
    )
    if use_short_input:
        input_max_chars = min(input_max_chars, int(args.fallback_input_max_chars))
    cmd = [
        args.python,
        str(REPO / "evaluate.py"),
        "--results-dir", str(job.results_dir),
        "--dataset", str(Path(args.dataset)),
        "--judge-backend", "mas-eval",
        "--memory-bank", str(Path(args.memory_bank)),
        "--mas-eval-input-max-chars", str(input_max_chars),
        "--workers", str(args.workers),
    ]
    if repair_mode:
        repair_identity = json.dumps(
            {
                "ids": sorted(set(repair_ids)),
                "model": args.model,
                "input_max_chars": input_max_chars,
                "dataset": str(Path(args.dataset).resolve()),
                "memory_bank": str(Path(args.memory_bank).resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        repair_key = hashlib.sha256(repair_identity.encode("utf-8")).hexdigest()[:16]
        include_dir = log_dir / "_repair_includes"
        include_dir.mkdir(parents=True, exist_ok=True)
        include_path = include_dir / f"{root_part}_{safe_name(job.condition)}_{repair_key}.txt"
        include_path.write_text("\n".join(repair_ids) + "\n", encoding="utf-8")
        repair_dir = log_dir / "_repair_outputs"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_out = repair_dir / f"{root_part}_{safe_name(job.condition)}_{repair_key}.json"
        cmd.extend(["--include-traces", str(include_path), "--out", str(repair_out)])
    else:
        cmd.extend(["--out", str(job.out_file)])
    if args.quiet:
        cmd.append("--quiet")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["MAS_EVAL_MODEL"] = args.model
    env["MAS_EVAL_INPUT_MAX_CHARS"] = str(input_max_chars)
    env.setdefault("MAS_EVAL_PARSE_RETRIES", str(args.parse_retries))

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        mode = "REPAIR" if repair_mode else ("RESUME" if resume_mode else "FULL")
        log.write(f"\n===== {now()} START {job.label} attempt={attempt} mode={mode} =====\n")
        if repair_mode:
            log.write(f"REPAIR_RUNS: {len(repair_ids)} ids; partial_out={repair_out}\n")
        elif resume_mode:
            log.write(
                f"RESUME_RUNS: {len(repair_ids)} error ids; "
                f"checkpoint={canonical_checkpoint}\n"
            )
        if use_short_input:
            log.write(f"FALLBACK_INPUT_MAX_CHARS: {input_max_chars}\n")
        log.write("COMMAND: " + " ".join(f'"{x}"' if " " in x else x for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if repair_mode and proc.returncode == 0 and repair_out is not None:
            try:
                replaced, partial_runs = merge_repair_summary(job, repair_out)
                log.write(f"\nMERGE repair output: replaced={replaced}; partial_runs={partial_runs}\n")
            except Exception as exc:
                log.write(f"\nMERGE repair output failed: {type(exc).__name__}: {exc}\n")
                return 1
        log.write(f"\n===== {now()} EXIT {job.label} code={proc.returncode} =====\n")
    return proc.returncode


def write_state(path: Path, jobs: list[Job], statuses: dict[str, dict[str, Any]]) -> None:
    payload = {
        "updated_at": now(),
        "jobs": [
            {
                "label": job.label,
                "results_dir": str(job.results_dir),
                "out_file": str(job.out_file),
                "expected_runs": job.expected_runs,
                **statuses.get(job.label, {}),
            }
            for job in jobs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(p) for p in DEFAULT_EXPERIMENT_ROOTS],
        help="Experiment roots. Each root must contain C1/C2/... condition folders.",
    )
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--model", default=os.environ.get("MAS_EVAL_MODEL", "deepseek-v4-pro")
    )
    parser.add_argument("--memory-bank", default=str(REPO / "mas_eval" / "memory_bank_human_v1.json"))
    parser.add_argument("--dataset", default=str(REPO / "dataset" / "pilot_tasks_mas.json"))
    parser.add_argument("--mas-eval-input-max-chars", type=int, default=8000)
    parser.add_argument("--fallback-input-max-chars", type=int, default=4000)
    parser.add_argument("--fallback-after-attempts", type=int, default=1)
    parser.add_argument("--fallback-repair-max-runs", type=int, default=5)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--output-suffix", default="_mas_eval_pro")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-root", default=str(REPO / "evals"))
    parser.add_argument("--log-dir", default=str(REPO / "run_logs" / "validation" / "mas_eval_monitor_logs"))
    parser.add_argument("--state-file", default=str(REPO / "evals" / "validation" / "mas_eval_monitor_state.json"))
    parser.add_argument("--retry-sleep", type=int, default=60)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Per-job attempt cap. 0 means retry until complete.",
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    print(f"[{now()}] monitor loaded {len(jobs)} job(s)")
    print(f"model={args.model} workers={args.workers} memory_bank={args.memory_bank}")
    for job in jobs:
        print(f"  - {job.label}: {job.expected_runs} traces -> {job.out_file}")

    attempts: dict[str, int] = {job.label: 0 for job in jobs}
    statuses: dict[str, dict[str, Any]] = {}
    state_path = Path(args.state_file)

    if args.dry_run:
        for job in jobs:
            ok, reason, _ = check_job(job)
            statuses[job.label] = {"ok": ok, "reason": reason, "attempts": 0}
            print(f"[dry-run] {job.label}: {reason}")
        write_state(state_path, jobs, statuses)
        return 0

    while True:
        all_ok = True
        made_attempt = False
        for job in jobs:
            while True:
                ok, reason, _ = check_job(job)
                if ok:
                    statuses[job.label] = {"ok": True, "reason": reason, "attempts": attempts[job.label]}
                    print(f"[{now()}] OK   {job.label}: {reason}")
                    break

                all_ok = False
                attempts[job.label] += 1
                if args.max_attempts and attempts[job.label] > args.max_attempts:
                    statuses[job.label] = {
                        "ok": False,
                        "reason": f"attempt cap reached after: {reason}",
                        "attempts": attempts[job.label] - 1,
                    }
                    print(f"[{now()}] STOP {job.label}: {statuses[job.label]['reason']}")
                    write_state(state_path, jobs, statuses)
                    return 2

                _, repair_ids = repairable_summary(job)
                mode = "repair" if repair_ids else "full"
                print(
                    f"[{now()}] RUN  {job.label}: {reason}; "
                    f"attempt {attempts[job.label]}; mode={mode}"
                    + (f"; repair_ids={len(repair_ids)}" if repair_ids else "")
                )
                code = run_job(job, args, attempts[job.label])
                made_attempt = True
                ok_after, reason_after, _ = check_job(job)
                statuses[job.label] = {
                    "ok": ok_after,
                    "reason": reason_after,
                    "attempts": attempts[job.label],
                    "last_exit_code": code,
                }
                print(f"[{now()}] DONE {job.label}: exit={code}; {reason_after}")
                write_state(state_path, jobs, statuses)
                if ok_after:
                    break

        if all_ok:
            write_state(state_path, jobs, statuses)
            print(f"[{now()}] all mas-eval judging outputs are complete")
            return 0

        if not made_attempt:
            write_state(state_path, jobs, statuses)
            print(f"[{now()}] no runnable attempts made; sleeping {args.retry_sleep}s")
            time.sleep(max(1, args.retry_sleep))


if __name__ == "__main__":
    raise SystemExit(main())
