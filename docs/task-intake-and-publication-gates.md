# Task Intake and Publication Gates

This document converts the related-work conclusions into an auditable intake
process. A task may execute before every item is complete, but its provenance
must prevent it from being presented above its evidence level.

## Evidence levels

| `publication_status` | Intended use | Minimum evidence |
|---|---|---|
| `intake` | Selected external source and task construction | Pinned source revision, license, contamination cutoff, and explicit blockers; no result claim |
| `fixture` | Unit tests, protocol demos, agent debugging | Deterministic baseline replay and explicit limitations |
| `pilot` | Internal model/task iteration | External source revision, license, validator, repeated measurements; calibration may still be incomplete but must be disclosed |
| `candidate` | Paper main table or public leaderboard | External source, independent/maintainer validation, calibrated or real-hardware final evaluation, all gates below |

Schema validation rejects a hand-authored task whose status is not `fixture`,
and rejects a `candidate` whose calibration is only `uncalibrated` or
`proxy_only`. A `calibrated` candidate must also reference an auditable
`calibration_bundle`; a `real_hardware` candidate must reference
`real_hardware_evidence`. The status string alone is not evidence.

`intake` is reserved for externally grounded work in progress. It allows the
runnable package to exist before a validator has reviewed it, but it cannot be
used in a paper table or leaderboard and it does not waive any pilot/candidate
gate.

## Required intake record

Add a schema-v3 `provenance` object to `task.json`:

```json
{
  "source_type": "upstream_pr",
  "source_url": "https://github.com/ORG/REPO/pull/NUMBER",
  "source_revision": "full-parent-commit-sha",
  "license": "Apache-2.0",
  "task_authors": ["task curator"],
  "validators": ["upstream maintainer or independent reviewer"],
  "contamination_cutoff": "2026-07-16",
  "publication_status": "pilot",
  "calibration_status": "partially_calibrated",
  "notes": "Transformation and exclusions"
}
```

For `real_trace`, additionally record in the task README:

- exact artifact revision and original schema;
- license and redistribution conditions;
- checksums of raw inputs kept outside the public agent workspace;
- deterministic cleaning/sessionization code;
- public/final time or domain split and leakage audit;
- whether request content was removed and only metadata retained.

For `maintainer_blind`, keep the sealed issue and reference patch in an
evaluator-owned store. Publish their hashes and the date of sealing before
running agents.

## Task construction

1. Check out the exact source revision and pin dependencies/container digest.
2. Reproduce the slow baseline at least three times.
3. Reproduce the expert patch or method at least three times and verify a
   statistically meaningful improvement.
4. Add correctness, model-quality, resource, integrity and performance gates.
5. Define public development and hidden final conditions with different seeds
   and hashes.
6. Include at least one transfer counterexample: the public optimum or expert
   patch must not be a universally safe constant recipe.
7. Replay the baseline and expert artifact from clean state.
8. Have a validator review task intent, available information, expected human
   time, and whether the hidden shift is operationally plausible.

Reject tasks whose baseline is unstable, whose expert gain disappears under
repetition, whose answer is one leaked flag, or whose hidden behavior is
determined mainly by an uncalibrated hand-written equation.

## Baseline ladder

Every candidate reports:

- `naive`: valid deliberately untuned implementation;
- `framework_default`: upstream default at the pinned revision;
- `expert_recipe`: original patch/method, including its provenance;
- `matched_search`: Random, Grid where finite, TPE, SMAC, and multi-fidelity BO
  where applicable, under the same query/cost/wall-time budget;
- human expert results on a representative subset.

The expert patch is never exposed as the hidden answer. Reports retain raw
ratios, per-profile metrics, failures, and costs even if they also present a
normalized aggregate.

## Simulator calibration bundle

For every simulator-backed candidate, evaluate identical configurations in
simulation and on the target hardware. The checked-in summary must include:

- hardware, model, framework and workload revisions;
- repeated measurements and variance;
- absolute/relative error;
- Spearman and Kendall rank correlation;
- top-k overlap;
- pairwise decision agreement;
- supported and unsupported decision regions.

`hardware_proxy` exists only to test multi-fidelity accounting. It must never
be labeled as a GPU measurement. A candidate becomes `calibrated` only when the
relevant decision boundary—not merely one baseline point—has been checked.
The task validator requires at least three paired configurations, at least
three hardware repeats per configuration, SHA-256 references to raw artifacts,
and explicit supported/unsupported decision regions.

## Anti-contamination split

Maintain three result sets:

1. historical reproducible tasks for development;
2. time-split recent tasks for periodic evaluation;
3. private maintainer-contributed tasks for headline contamination-resistant
   results.

Remove PR numbers and identifying text from the agent prompt, define the
internet policy before evaluation, and record the model knowledge cutoff when
available. Hidden workload/hardware counterexamples ensure that reproducing a
memorized historical patch is insufficient.
