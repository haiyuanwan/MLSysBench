# Scheduler Probe Allocation — Multi-Fidelity Protocol Fixture

Development offers two evaluator fidelities:

- `simulator`: cost 1, up to 12 calls; deliberately optimistic about large
  prefills and mixed batches.
- `hardware_probe`: cost 4, up to 2 calls; closer to the final timing surface,
  but still a deterministic simulation.

The total development budget is 12 cost units. Put a fidelity name beside the
changes in a development request, for example:

```json
{"changes": {}, "fidelity": "hardware_probe"}
```

The research question is whether an agent chooses informative expensive probes
and corrects simulator-induced beliefs. Despite its name, `hardware_probe` is
not a physical GPU result; the provenance and evaluator expose it as
`hardware_proxy`.

The fixture contains a validator-tested counterexample: among otherwise
identical static policies, the cheap simulator prefers a 256-token prefill
chunk over a 16-token chunk, while the hardware proxy and hidden target reverse
that ordering. This is a protocol property of the hand-authored timing models,
not evidence that real hardware has the same boundary.
