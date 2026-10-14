# AutoShield Phase 7 — GNN Baseline Results

## Status
Experimental GNN foundation implemented. Attack-path reconstruction
performance has not been validated.

## Implementation
- Deterministic attack graph construction.
- Relation-aware, two-layer Graph Convolutional Network (GCN).
- Typed edge scorer trained with binary cross-entropy.
- Scenario-based train, validation, and test split metadata.
- Dataset structural validation and automated tests.
- Saved GCN and edge-scorer weights.

## Dataset
- 36 scenario artifact triples: scenario_003 through scenario_038.
- 12 scenarios per split: train, validation, and test.
- 12 binary edge assessments used by the baseline.
- 24 additional assessments marked unknown and excluded from binary training.
- Raw incident evidence and ground-truth artifacts remain separate from
  graph construction and model inference.

## Baseline configuration
- Random seed: 7
- Epochs: 50
- Learning rate: 0.01

## Recorded metrics

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.75 | 0.75 |
| Recall | 1.00 | 1.00 |
| F1 | 0.8571 | 0.8571 |
| Accuracy | 0.75 | 0.75 |

There are four examples in each evaluation split. All four test predictions
are positive, with probability 0.7536 for every example. The model therefore
does not demonstrate discrimination between positive and negative examples.
These metrics must not be interpreted as evidence of generalization.

## Limitations
1. The labeled dataset is extremely small: four examples per split.
2. Current graph topology is largely static across scenarios.
3. HTTP evidence is preserved on graph edges, but the GNN encoder does not
   encode those request attributes as model features.
4. A forwarding edge demonstrates observed routing, not successful exploitation.
5. Previously observed perfect path scores are not meaningful evidence of
   attack-path reconstruction because the candidate paths are structurally
   trivial and the expected paths are derived from intended scenario targets.
6. The current baseline does not establish performance on unseen environments.

## Artifacts
- evaluation/results/gnn_baseline_results.json
- evaluation/results/models/attack_graph_gcn.pt
- evaluation/results/models/edge_scorer.pt
- evaluation/scenarios/
- evaluation/scenarios/gnn_splits.json

## Verification
- Automated test suite: 159 passed at the recorded checkpoint.
- Structural dataset validation: 36/36 scenario triples, zero structural
  errors, 12 scenarios per split.
- This structural validation does not establish semantic label correctness.

## Conclusion
Phase 7's experimental GNN implementation and baseline training pipeline
are available for further development. Meaningful attack-path reconstruction
remains future work and must be validated after incorporating scenario-specific
evidence features and independently reviewed labels.
