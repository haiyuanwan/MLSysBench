# Adaptive Chunk Patch — Protocol Fixture

Patch `solution/scheduler.py` to improve SLO-aware goodput on the public mixed
workload without baking in its exact request sequence. Final evaluation replays
the patch from a clean starter tree on burst and long-prompt profiles.

This is a hand-authored deterministic simulator fixture. It validates the
`PatchTransfer` artifact and evaluation protocol; it is not admissible as a
publication result until replaced by an upstream-derived, maintainer-validated
task and calibrated against hardware.

The fixture deliberately rejects the shortcut “select the public best fixed
chunk”: the public timing proxy prefers a 256-token chunk to a 16-token chunk,
while the hidden mixed profiles reverse that ordering. A robust patch must use
online observations rather than hard-code the public optimum. This reversal is
unit-tested but remains a synthetic protocol property.
