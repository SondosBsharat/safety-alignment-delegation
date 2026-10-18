from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate import (
    _PROGRESS_SCHEMA,
    _append_trace_checkpoint,
    _atomic_write_json,
    _load_trace_checkpoints,
    _write_checkpoint_progress,
)


def _successful_llm_run(result_file: str) -> dict:
    return {
        "result_file": result_file,
        "c1_deterministic": {"valid": True},
        "llm": {
            "c1_unauthorized_action": {"violation": False, "status": "valid"},
            "c2a_over_disclosure": {"violation": True, "status": "valid"},
            "c2b_constraint_lost": {"violation": False, "status": "valid"},
        },
    }


class EvaluateCheckpointTests(unittest.TestCase):
    def test_append_and_load_reuses_success_and_skipped_but_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "evaluation.json.checkpoint.jsonl"
            fingerprint = "fingerprint-a"
            trace_keys = ["trace-key-0", "trace-key-1", "trace-key-2"]
            result_files = ["trace-0.json", "trace-1.json", "trace-2.json"]
            success = _successful_llm_run(result_files[0])
            error = {
                "result_file": result_files[2],
                "llm": {"error": "temporary gateway failure"},
            }

            _append_trace_checkpoint(
                checkpoint,
                fingerprint=fingerprint,
                trace_key=trace_keys[0],
                index=0,
                result_file=result_files[0],
                status="success",
                run=success,
            )
            _append_trace_checkpoint(
                checkpoint,
                fingerprint=fingerprint,
                trace_key=trace_keys[1],
                index=1,
                result_file=result_files[1],
                status="skipped",
                run=None,
            )
            _append_trace_checkpoint(
                checkpoint,
                fingerprint=fingerprint,
                trace_key=trace_keys[2],
                index=2,
                result_file=result_files[2],
                status="error",
                run=error,
            )

            raw = checkpoint.read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertEqual(len([json.loads(line) for line in raw.splitlines()]), 3)

            reusable, statuses = _load_trace_checkpoints(
                checkpoint,
                fingerprint=fingerprint,
                trace_keys=trace_keys,
                result_files=result_files,
                use_llm=True,
            )

        self.assertEqual(reusable, {0: success, 1: None})
        self.assertEqual(statuses, {0: "success", 1: "skipped"})
        self.assertNotIn(2, reusable)

    def test_load_ignores_only_a_torn_final_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "evaluation.json.checkpoint.jsonl"
            success = _successful_llm_run("trace-0.json")
            _append_trace_checkpoint(
                checkpoint,
                fingerprint="fingerprint-a",
                trace_key="trace-key-0",
                index=0,
                result_file="trace-0.json",
                status="success",
                run=success,
            )
            with checkpoint.open("ab") as stream:
                stream.write(b'{"schema":"evaluation_trace_checkpoint_v1"')

            reusable, statuses = _load_trace_checkpoints(
                checkpoint,
                fingerprint="fingerprint-a",
                trace_keys=["trace-key-0"],
                result_files=["trace-0.json"],
                use_llm=True,
            )

            # Loading must physically remove the torn tail; otherwise this append
            # would concatenate valid JSON onto the fragment and poison all future
            # resumes.
            success_1 = _successful_llm_run("trace-1.json")
            _append_trace_checkpoint(
                checkpoint,
                fingerprint="fingerprint-a",
                trace_key="trace-key-1",
                index=1,
                result_file="trace-1.json",
                status="success",
                run=success_1,
            )
            reusable_after_append, statuses_after_append = _load_trace_checkpoints(
                checkpoint,
                fingerprint="fingerprint-a",
                trace_keys=["trace-key-0", "trace-key-1"],
                result_files=["trace-0.json", "trace-1.json"],
                use_llm=True,
            )

        self.assertEqual(reusable, {0: success})
        self.assertEqual(statuses, {0: "success"})
        self.assertEqual(reusable_after_append, {0: success, 1: success_1})
        self.assertEqual(statuses_after_append, {0: "success", 1: "success"})

    def test_load_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "evaluation.json.checkpoint.jsonl"
            _append_trace_checkpoint(
                checkpoint,
                fingerprint="fingerprint-a",
                trace_key="trace-key-0",
                index=0,
                result_file="trace-0.json",
                status="success",
                run=_successful_llm_run("trace-0.json"),
            )

            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                _load_trace_checkpoints(
                    checkpoint,
                    fingerprint="fingerprint-b",
                    trace_keys=["trace-key-0"],
                    result_files=["trace-0.json"],
                    use_llm=True,
                )

    def test_atomic_json_and_progress_files_are_complete_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_path = root / "nested" / "atomic.json"
            _atomic_write_json(atomic_path, {"generation": 1, "text": "初稿"})
            _atomic_write_json(atomic_path, {"generation": 2, "text": "完成"})

            self.assertEqual(
                json.loads(atomic_path.read_text(encoding="utf-8")),
                {"generation": 2, "text": "完成"},
            )
            self.assertEqual(list(atomic_path.parent.glob("*.tmp")), [])

            checkpoint = root / "evaluation.json.checkpoint.jsonl"
            progress = Path(f"{checkpoint}.progress.json")
            output = root / "evaluation.json"
            _write_checkpoint_progress(
                progress,
                fingerprint="fingerprint-a",
                fingerprint_payload={"dataset": "fixture"},
                checkpoint_path=checkpoint,
                out_path=output,
                statuses={0: "success", 1: "skipped", 2: "error"},
                expected_runs=4,
                resumed_runs=1,
                final_summary_written=False,
            )

            payload = json.loads(progress.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], _PROGRESS_SCHEMA)
        self.assertEqual(payload["completed_runs"], 3)
        self.assertEqual(payload["succeeded_runs"], 1)
        self.assertEqual(payload["skipped_runs"], 1)
        self.assertEqual(payload["failed_runs"], 1)
        self.assertEqual(payload["pending_runs"], 1)
        self.assertEqual(payload["resumed_runs"], 1)
        self.assertFalse(payload["final_summary_written"])
        self.assertFalse(payload["complete"])


if __name__ == "__main__":
    unittest.main()
