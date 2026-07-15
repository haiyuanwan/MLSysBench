# Mock Scale-Transfer Task

The development phase exposes an 8-replica workload. The final evaluator uses
32 replicas, higher request pressure, and a different hidden performance
surface. Agents may run up to five development experiments and optionally
submit a distinct `final_changes` configuration for the hidden scale.

This dependency-free task validates the benchmark protocol. Its metrics are
synthetic and must not be reported as a model capability result.
