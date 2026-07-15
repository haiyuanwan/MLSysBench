# Decode-Heavy Scenario Fixture

This dependency-free fixture validates the canonical `decode_heavy` family,
its short/long output profiles, held-out workload metadata, minimize-direction
scoring, and complete fail-closed mock surfaces.

The public surface favors TP2 while the final long-output surface favors TP1,
so copying the public optimum is intentionally not always correct.

All numbers are synthetic. This is a protocol fixture, not benchmark evidence.
