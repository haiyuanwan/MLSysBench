# Balanced Scenario Fixture

This dependency-free fixture validates the canonical `balanced` family, its
mixed prompt/output and concurrency profiles, held-out workload metadata,
maximize-direction scoring, and complete fail-closed mock surfaces.

The public surface favors TP2 while the shifted concurrency surface favors
TP1, providing a counterexample to a single parallelism shortcut.

All numbers are synthetic. This is a protocol fixture, not benchmark evidence.
