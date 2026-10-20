# Safety Alignment Across Trust Boundaries in Multi-Agent Systems

Supplementary code and results for Sondos Bsharat's project proposal. The project studies how original user safety constraints are communicated and followed during agent delegation.

**Status:** The 20-task MasDrift healthcare baseline is completed and included here. It measures **internal delegation only**. Experiments with independently operated external agents, persistent-constraint access, and budget-constrained verification remain planned; this release does not establish an external trust-boundary effect.

## Repository contents

```text
configs/                         Pilot settings, source hashes, validation environment
data/selected_task_ids.json       Exact 20 benchmark task IDs
scripts/run_baseline.py           Run the saved task selection
scripts/evaluate_baseline.py      Re-evaluate traces without an LLM judge
scripts/summarize_results.py      Recompute the proposal's summary numbers
scripts/verify_release.py         Check source hashes and task coverage
results/healthcare_baseline_20/   20 original execution traces
results/healthcare_baseline_20_eval.json
                                 Original rule + LLM evaluation, local paths normalized
results/healthcare_baseline_20_summary.json
                                 Derived proposal metrics
figures/                         Existing worker-action figure (PDF and PNG)
vendor/masdrift/                  Unmodified upstream source, benchmark, and tests
```


## Preliminary experiment

| Setting | Recorded value |
| --- | --- |
| Domain | Healthcare (synthetic benchmark scenarios) |
| Tasks | 20, identified in `data/selected_task_ids.json` |
| Repetitions | One per task |
| Supervisor and worker model | `gpt-4o-mini`, OpenAI |
| Architecture | Centralized: one supervisor and two workers |
| Layers / width | `1` / `2` |
| Personas | `generic` |
| Visibility / context | `scoped` / `branch-local` |
| Defense | `none` |

The saved task list is authoritative. The original sampling procedure and selection seed have not been verified, so we do not claim that a seed alone reconstructs the sample.

| Measurement | Observed result |
| --- | ---: |
| Task completion rate (TCR) | 20/20 (100%) |
| Runs with an executed unauthorized action (UAR) | 20/20 (100%) |
| Executed unauthorized actions | 96 |
| Supervisor | 10 |
| Worker 1 | 47 |
| Worker 2 | 39 |
| Execution tokens, excluding evaluation | 564,552 |

Completion means satisfaction of the evaluator's required benchmark steps; it does not imply safe execution. Unauthorized actions are executed simulated tool effects classified against benchmark authorization constraints. These are not real healthcare actions.

The figure shows each worker's share of the **86 worker-executed unauthorized actions**: 54.7% and 45.3%. These are action shares, not per-worker violation probabilities. The small, single-run pilot does not establish causality or comparative defense effectiveness.

![Worker shares of unauthorized actions](figures/unauthorized_actions_by_worker.png)

## Inspect results without installing dependencies

From the repository root, using Python 3.10 or newer:

```bash
python scripts/verify_release.py
python scripts/summarize_results.py --check
python scripts/run_baseline.py --dry-run
```

These commands use only the standard library and make no API calls. The last prints the 20 reconstructed execution commands. The wrapper was prepared from recorded settings; it is not a recovered copy of the original shell command.

## Install and re-evaluate the saved traces

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/evaluate_baseline.py
python scripts/summarize_results.py \
  --report outputs/healthcare_baseline_20_rule_eval.json --check
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Re-evaluation uses the included traces and benchmark with MasDrift's `--rule-only` mode. It makes no model calls and writes to `outputs/`, preserving published results. It recomputes TCR, executed-action UAR, action counts, and recorded token totals. It does not regenerate semantic handoff annotations.

 Upstream Python requirements contain version ranges rather than a complete lock. `configs/validation_environment.json` records the environment used to verify this release, not a guaranteed reconstruction of the original execution environment.

## Run new baseline executions

Export `OPENAI_API_KEY` in your shell or supply it through your secret manager. The wrapper reads the exported key and uses the standard OpenAI endpoint. Do not put a `.env` file inside `vendor/masdrift`, because the upstream loader could apply hidden overrides.

Run one selected task first:

```bash
python scripts/run_baseline.py --case-id pilot_hc_records_request_003
```

Run the full selection, or resume an interrupted run:

```bash
python scripts/run_baseline.py --resume
python scripts/evaluate_baseline.py \
  --results-dir outputs/healthcare_baseline_20 \
  --out outputs/new_baseline_rule_eval.json
python scripts/summarize_results.py --report outputs/new_baseline_rule_eval.json
```
## Evaluation provenance and limitations

- The original evaluation report preserves rule and LLM judgments. Absolute local paths were replaced with repository-relative paths; measurements and judgments were not changed. The original checkpoint fingerprint remains historical metadata and is not intended for resuming a relocated evaluation.
- The report records the native judge backend and a DeepSeek provider setting, but no resolved judge model. Exact reproduction of its semantic annotations cannot be claimed. Fresh semantic evaluation requires explicitly recording the chosen model, provider, and settings and may incur API costs.
- Possible constraint-propagation failures, especially judge-generated handoff indices, require manual validation against traces. They are not validated locations of first constraint loss.
- All 20 executions completed and all had unauthorized actions. This saturation motivates complementary action-level and handoff-level measures.
- Existing figures are included.

## Planned study

The proposed work separates **constraint propagation** (whether requirements reach a downstream agent) from **behavioral adherence** (whether the agent follows them). Planned experiments include:

- Internal-to-external and multihop external delegation with matched internal controls.
- Handoff annotations for preservation, weakening, omission, contradiction, and recovery.
- Comparisons with MasDrift's source-authorization, chain-propagation, and combined defenses, plus persistent original constraints.
- Budget-constrained authorization verification and adaptive external-agent attacks in a simulated environment.

The upstream snapshot includes existing MasDrift defenses, but this pilot ran only `none`. This project's external-agent extensions and verification strategy are not implemented in this release.

## Attribution

Xu, Z., Zhang, X., Luo, H., Jin, Y., Dong, Y., and Salam, H. (2026). *MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures*. [arXiv:2608.07556](https://arxiv.org/abs/2608.07556).

```bibtex
@article{xu2026masdrift,
  title={MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures},
  author={Xu, Zhuoning and Zhang, Xiucheng and Luo, Hanjun and Jin, Yingbin and Dong, Yinpeng and Salam, Hanan},
  journal={arXiv preprint arXiv:2608.07556},
  year={2026}
}
```
