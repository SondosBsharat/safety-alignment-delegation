# Safety Alignment Across Trust Boundaries in Multi-Agent Systems

Code and experimental materials for studying how safety requirements from the original user request are preserved and followed during multi-agent task delegation.

**Status:** Preliminary MasDrift internal-delegation baseline completed. Experiments involving independently operated external agents, persistent constraints, and verification are planned and **not implemented in this release**.

## Research objective

We study safety alignment at agent handoffs, separating:

- **Constraint propagation:** whether an original safety requirement is correctly communicated at a handoff.
- **Behavioral adherence:** whether an agent's observable actions respect applicable safety requirements.

The planned study examines internal-to-external (I–E) and internal-to-external-to-external (I–E–E) delegation, including appropriate internal controls, and evaluates whether carrying forward original user constraints and verifying consequential actions improve safety without sacrificing task completion.

## Preliminary experiment (completed)

Our pilot uses the **MasDrift** benchmark's healthcare tasks and its centralized supervisor–worker architecture. These results concern **internal delegation only** and are **not** evidence of an external trust-boundary effect.

| Setting | Value |
| --- | --- |
| Domain | Healthcare |
| Task count | 20 selected benchmark tasks |
| Task-selection seed | 42 |
| Repetitions | 1 per task |
| Model | `gpt-4o-mini` |
| Architecture | Centralized; 1 supervisor and 2 workers |
| Layers / width | `1` / `2` |
| Personas | `generic` |
| Visibility | `scoped` |
| Defense | `none` |

### Preliminary findings

| Measurement | Observed result |
| --- | ---: |
| Task completion rate (TCR) | 20/20 (100%) |
| Unauthorized action rate (UAR) | 20/20 (100%) |
| Total executed unauthorized actions | 96 |
| Supervisor-executed unauthorized actions | 10 |
| Worker 1 unauthorized actions | 47 |
| Worker 2 unauthorized actions | 39 |

Worker 1 and Worker 2 account for 47/86 (54.7%) and 39/86 (45.3%) of **worker-executed unauthorized actions**, respectively. These percentages are **shares of actions**, not probabilities of violating a requirement. The pilot used a small task sample and does not establish why violations occurred.

The evaluation also flagged possible constraint-propagation failures; handoff positions produced by the evaluator have not yet been manually validated against the raw traces.

## Repository contents

The following materials should be committed **once checked and prepared**. Paths describe the intended layout, not a claim that files already exist:

```text
safety-alignment-delegation/
├── README.md
├── .gitignore
├── configs/
│   └── healthcare_baseline_20.json       # exact settings and upstream revision
├── data/
│   └── selected_task_ids.json            # 20 public benchmark task identifiers
├── scripts/
│   ├── select_tasks.py                   # your actual task-selection script
│   ├── run_baseline.sh                   # verified exact baseline command
│   └── plot_worker_actions.py            # figure-generation source
├── results/
│   ├── healthcare_baseline_20_eval.json  # sanitized evaluator output
│   └── healthcare_baseline_20_summary.json
└── figures/
    └── unauthorized_actions_by_worker.pdf
```

**Do not create placeholder scripts just to match the tree.** Only commit code and results that correspond to the experiment you actually ran. If task-selection or plotting was done interactively, export the relevant notebook/cells as a script and test it.

## Reproducibility

1. Obtain the original MasDrift implementation from its official repository (see the paper *MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures*, Xu et al., 2026, arXiv:2608.07556). Record the exact upstream commit and any local modifications.
2. Create a Python environment and install the dependencies specified by that revision of MasDrift. Configure API credentials **outside the repository**.
3. Reconstruct the 20 healthcare task selection using the committed task IDs or the task-selection script (seed 42).
4. Run MasDrift with the configuration documented above. Save the exact tested execution command in `scripts/run_baseline.sh` after confirming it against the original experiment.
5. Evaluate the stored execution traces using MasDrift's evaluation entry point:

```bash
python evaluate.py \
  --results-dir results/healthcare_baseline_20 \
  --out results/healthcare_baseline_20_eval.json
```

6. Generate the worker-violation figure using your committed plotting script.

**Reproducibility note:** We have not yet included a verified end-to-end run command or a pinned upstream commit in this README. Add both from the original run before claiming fully reproducible results. Evaluation may require additional API tokens and its output can vary across evaluator/model versions.

## Planned experiments (not yet released)

- Internal-to-external and multihop external-to-external delegation using independently controlled agent services.
- Handoff-level annotations of preserved, weakened, omitted, or contradicted constraints; behavioral adherence; and recovery.
- Persistent original-constraint document and source-anchored authorization baselines.
- Budget-constrained verification evaluated against random checking and verify-all baselines, if implemented.
- Adaptive external-agent tests in a sandboxed environment.

## Safety and data handling

Use synthetic or appropriately sanitized benchmark data and sandboxed actions. Do not commit API keys, `.env` files, credentials, sensitive messages, identifying patient data, or raw external-service logs without screening them first. Report only actions that were actually executed in the experimental tool environment.

## Reference

Xu et al. (2026). *MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures*. https://arxiv.org/abs/2608.07556
