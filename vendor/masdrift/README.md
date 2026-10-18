# MasDrift

[![arXiv](https://img.shields.io/badge/arXiv-2608.07556-b31b1b.svg)](https://arxiv.org/abs/2608.07556)

MasDrift is a research codebase for studying instruction drift in multi-agent
LLM systems. It evaluates how coordination structure, system scale, agent
personas, context visibility, and boundary-state defenses affect task
completion and policy adherence.

This repository accompanies the paper
[MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures](https://arxiv.org/abs/2608.07556).
It contains no experiment outputs or API credentials.

## Experimental conditions

The maintained experiment matrix contains:

| ID | Structure | Size | Personas |
|---|---|---:|---|
| N2 | decentralized | 2 peers | generic |
| N4 | decentralized | 4 peers | generic |
| N8 | decentralized | 8 peers | generic |
| C1 | centralized | 1 management layer, width 2 | generic |
| C2 | centralized | 2 management layers, width 2 | generic |
| C3 | centralized | 3 management layers, width 2 | generic |
| N4-t | decentralized | 4 peers | typed |
| C2-t | centralized | 2 management layers, width 2 | typed |

Single-agent runs use the ordinary single-agent implementation. There is no
special single-agent ReAct-round branch.

The maintained defense treatments are:

- `none`: raw/no-defense condition.
- `authorizer`: Source defense, implemented by an authorization agent over the
  user request and public tool descriptions.
- `chain`: Chain defense, which propagates Tool-Scope-Stage policy state through
  handoffs.
- `both`: Source and Chain enabled together.

`none` does not inject Source- or Chain-specific instructions. Generic agent
behavior shared by all treatments—such as asking for missing information,
respecting tool preconditions, and confirming genuinely consequential
actions—remains enabled.

## Repository layout

```text
.
├── main.py                    # experiment entry point
├── single.py                  # single-agent baseline
├── centralized.py             # hierarchical coordination
├── decentralized.py           # peer coordination
├── agent_react.py             # agent loop and handoff logic
├── defense.py                 # defense configuration and prompts
├── defense_chain.py           # Chain state propagation
├── authorization_policy.py    # Source authorization policy
├── llm.py                     # provider and transport adapters
├── evaluate.py                # evaluation pipeline
├── dataset.py                 # dataset loader
├── dataset/                   # benchmark tasks and environments
├── mas_eval/                  # evaluation backend code (no selected gold data)
├── scripts/                   # maintained run and analysis scripts
├── tests/                     # regression tests
└── LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add only the credentials required by your
provider. `.env` is ignored by Git. Never commit real API keys.

## Running experiments

The default dataset is `dataset/pilot_tasks_mas.json`. Run the commands below
from the repository root; all documented paths are repository-relative and do
not depend on a particular checkout location.

Run one case:

```bash
python main.py \
  --case-id pilot_hc_referral_letter_001 \
  --structure centralized \
  --layers 2 \
  --width 2 \
  --personas generic \
  --visibility scoped \
  --defense none \
  --out-dir results/smoke
```

Run the maintained matrix on Windows PowerShell:

```powershell
# Raw/no defense
.\scripts\run_matrix.ps1 -Root results\raw -Defense none -Resume

# Source defense
.\scripts\run_matrix.ps1 -Root results\source -Defense authorizer -Resume

# Chain defense
.\scripts\run_matrix.ps1 -Root results\chain -Defense chain -Resume
```

Use `-Only N4,C2` for selected conditions and `-Workers N` to control
concurrency. The scripts default to the `python` available on `PATH`; another
interpreter can be supplied explicitly with `-Python`.

Provider settings can be supplied through `.env` or the command line:

```powershell
.\scripts\run_matrix.ps1 `
  -Root results\raw `
  -Defense none `
  -LlmProvider openai `
  -LlmModel gpt-5.4-mini `
  -LlmApiKeysEnv OPENAI_API_KEYS `
  -Resume
```

For OpenAI-compatible relays, pass the API root rather than a
`/chat/completions` endpoint.

## Evaluation

Evaluate one result directory:

```bash
python evaluate.py --results-dir results/raw/N4
```

Additional maintained evaluation and reporting utilities are under `scripts/`.
Generated results, reports, caches, and local secrets are excluded by
`.gitignore`.

## Provider response compatibility

For GPT models using the OpenAI Responses API, server-owned item IDs are
removed before message history is relayed to another endpoint. This avoids
invalid cross-request or cross-provider references and does not alter task
content.

Different model providers and OpenAI-compatible relays may return different
JSON structures, metadata fields, and identifier formats. Review and adjust
the response normalization in `llm.py` for the provider used by your
deployment. No GPT-specific handoff repair or extra model call is applied.

## Testing

Run the regression suite without requiring `pytest`:

```bash
python -m unittest discover -s tests
```

The tests cover dataset loading, structure construction, defense selection,
handoff state, provider transport, evaluation, and resume behavior.

## Data and anonymity

The benchmark contains synthetic people, organizations, emails, identifiers,
and scenarios required by the tasks. They are fictional benchmark content and
must not be interpreted as project authorship or affiliation.

Before release, scan the repository for credentials and local paths in addition
to relying on `.gitignore`. Experiment outputs may contain model-generated
personal data and should be reviewed separately before publication.

## Citation

If you use MasDrift in your research, please cite:

```bibtex
@article{xu2026masdrift,
  title={MasDrift: Benchmarking Authorization Preservation Across Multi-Agent Architectures},
  author={Xu, Zhuoning and Zhang, Xiucheng and Luo, Hanjun and Jin, Yingbin and Dong, Yinpeng and Salam, Hanan},
  journal={arXiv preprint arXiv:2608.07556},
  year={2026}
}
```

## License

The MasDrift code and benchmark are available under the MIT License. See
`LICENSE`.
