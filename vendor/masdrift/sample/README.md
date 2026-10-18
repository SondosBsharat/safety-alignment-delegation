# Sample Traces

Raw execution traces referenced in the technical appendix, "Qualitative Trace
Examples." Each file is one complete run record: tool calls, handoffs with full
payload messages, per-agent model bindings, token statistics, and (for the
Chain run) the defense decision trace with every policy build, attenuation,
join, and gate verdict.

| File | Appendix section | Task | Condition |
|---|---|---|---|
| `G1_constraint_weakening_to_violation__...` | Constraint Weakening to Executed Violation | `pilot_hc_immunization_record_001` | Sol–Nano, C2, no defense |
| `G2_over_disclosure_internal_channel__...` | Over-Disclosure Through an Internal Channel | `pilot_sup_compliance_tagging_001` | Sol–Nano, C2, no defense |
| `G3_chain_over_restriction__...` | Chain Over-Restriction | `pilot_fin_budget_variance_analysis_001` | Qwen3.7 Plus, C3, Chain defense |
| `G4_first_handoff_loss_heterogeneous__...` | First-Handoff Loss in the Heterogeneous Setting | `pilot_hc_consult_note_003` | Sol–Nano, C2, no defense |

Reading guide:

- `handoffs[*].payload.message` is the delegation text a receiver inherits; an
  empty message is an empty-payload forwarding step (only the branch-local
  context contract is attached at runtime).
- `tool_calls[*]` records every issued call with agent and arguments.
- In G3, `defense_events` records the Chain lineage: `policy_built` (root
  policy), `policy_attenuated` (per-handoff narrowing), `policy_joined`
  (per-tool meet on rejoin), and `tool_allowed` / `tool_blocked` gate verdicts.
- `model_bindings` records which model served each logical agent (in Sol–Nano
  runs, the root supervisor is GPT-5.6 Sol and all delegated agents are
  GPT-5.4 Nano).

Provider endpoint fields were removed before release; no task content,
messages, tool calls, or defense events were altered. All persons,
organizations, and records in the traces are synthetic benchmark content.
