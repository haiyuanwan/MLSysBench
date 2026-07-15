# MLSysBench Documentation

This directory contains canonical user documentation, implementation runbooks,
research design notes, and background surveys. The labels below distinguish
what describes the current executable repository from historical or proposed
scope.

## Start here

| Document | Status | Use it for |
|---|---|---|
| [Repository README](../README.md) | Canonical | Quick start, task readiness, command map |
| [Architecture](architecture.md) | Canonical | Module ownership, data flow, trust boundaries |
| [Benchmark protocol](benchmark-protocol.md) | Normative | Scenario families, phases, baselines, gates, and ordered design |
| [Status and roadmap](status-and-roadmap.md) | Canonical | Verified capabilities, staged deliverables, and acceptance criteria |
| [Implementation and CUDA runbook](simai-benchmark-code.md) | Current but host-specific | Real Vidur+AICB setup and execution |
| [Task design lessons](task-design-lessons.md) | Current methodology | Invariants learned from CLI-agent runs |
| [Task catalog](../tasks/README.md) | Canonical | Task maturity and publication requirements |
| [Result catalog](../benchmarks/README.md) | Canonical | Evidence levels and checked-in summaries |

## Research design and background

These documents support task selection and paper development. They are useful
context, but they may describe a broader future benchmark than the code
currently implements.

| Document | Scope |
|---|---|
| [Design proposal](design-proposal.md) | Historical full-stack proposal |
| [SimAI benchmark proposals](simai-benchmark-proposals.md) | Configuration, diagnosis, and scale-transfer concepts |
| [Task taxonomy](task-taxonomy.md) | Candidate inference-optimization task space |
| [Data sources](data-sources.md) | Task construction and baseline methodology |
| [Existing benchmarks](existing-benchmarks.md) | Related benchmark survey |
| [Competitions](competitions.md) | Competition and evaluation-format survey |
| [SimAI analysis](simai-analysis.md) | Detailed analysis of the vendored simulator stack |
| [Environment](environment.md) | Broader deployment and measurement options |

## Documentation rules

- The benchmark protocol is the normative target design. The root README,
  architecture guide, and status document distinguish implemented behavior
  from that target and should change with the code.
- A task README must state whether the task is synthetic, smoke-only,
  experimental, or publication-ready.
- Generated output belongs under ignored `runs/`; only compact result summaries
  with provenance and caveats belong under `benchmarks/`.
- Host-specific commands belong in runbooks, not in the quick start.
- Proposed features must be labeled as proposed or pending rather than written
  as current capability.

Before updating a task or its documentation, run:

```bash
python3 -m mlsysbench.simai_bench validate-task --task TASK_DIRECTORY
scripts/check_repo.sh
```
