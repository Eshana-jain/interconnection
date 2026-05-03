# Final Presentation Outline

## Slide 1: Title

**ML Interconnection Queue Intelligence for the U.S. Electric Grid**  
CEE 498 MLC, Spring 2026  
Eshana Jain

## Slide 2: Motivation

- More than 2,300 GW of generation capacity is waiting in U.S. interconnection queues.
- Most projects are ultimately withdrawn, often after years of study costs and uncertainty.
- Developers need earlier estimates of withdrawal risk, likely upgrade-cost severity, and study duration.

## Slide 3: Project Goal

Predict three site-evaluation outcomes before queue filing:

- Probability that a project is withdrawn.
- Network upgrade cost bucket: $0-1M, $1-5M, $5-20M, $20M+.
- Timeline in months to complete the study process.

## Slide 4: Data and Features

- Backbone: LBNL Queued Up 2025 project-level queue data.
- Supplementary sources: MISO, PJM, and CAISO queue exports where available.
- Key engineered features: log MW capacity, ISO/RTO, fuel type, county, filing year, voltage, queue position, substation congestion over the prior 24 months, county queue density, ISO-year cohort size, deliverability/service type, distance to transmission, and renewable penetration.
- Cost modeling is scoped to the subset with upgrade-cost labels.

## Slide 5: Research Paper Reproduced

- Raissi, Perdikaris, and Karniadakis (2019) introduced physics-informed neural networks.
- Core idea: train with `L_data + lambda L_physics`.
- Their physics term enforces PDE residuals at collocation points.
- This project adapts the same structure to interconnection queue cost monotonicity.

## Slide 6: Physics-Informed Cost Model

The cost model is trained with:

```text
L = cross_entropy(cost_bucket) + lambda * monotonic_penalty
```

The monotonic penalty compares each project with a collocation copy where MW is increased while other inputs are held fixed. The model is penalized if expected cost bucket decreases.

## Slide 7: Baselines and Model Suite

- Logistic regression for withdrawal prediction.
- Ridge regression for timeline prediction.
- MLP classifier/regressor for nonlinear interactions.
- Gaussian process regression for cost uncertainty intervals.
- Unconstrained MLP vs monotonic PINN MLP for the main ablation.

## Slide 8: Experimental Setup

- 80/10/10 train/validation/test split.
- Stratification by ISO region and filing-year cohort where feasible.
- Class-weight balancing for withdrawal baseline.
- Metrics: AUC-ROC/F1, weighted F1/accuracy, MAE/R2, 90% prediction interval coverage, and monotonic implausibility rate.

## Slide 9: Results Summary

Use the latest generated `outputs/model_comparison.csv`, `outputs/metrics.json`, and `outputs/*.png`.

- Report the best withdrawal AUC/F1 from the latest run.
- Report timeline MAE/R2 from the latest run.
- Compare unconstrained cost MLP against PINN monotonic cost MLP.
- Report GPR 90% interval coverage from `gpr_uncertainty.picp_90`.

## Slide 10: PINN Ablation

Show `outputs/pinn_ablation_cost.png`.

Main message:

- The physics-informed model should be compared directly against the unconstrained cost MLP from the latest generated metrics.
- The monotonic penalty encodes project-scale domain knowledge in the same spirit as Raissi et al.'s PDE residual.

## Slide 11: Interpretation

- Congestion, queue cohort size, county density, filing year, deliverability, distance to transmission, renewable penetration, and MW capacity carry the main project story.
- Timeline prediction is more stable than cost prediction because timeline labels are more complete.
- GPR intervals widen uncertainty around log cost and provide a practical confidence measure.

## Slide 12: Limitations

- Public queue data has incomplete upgrade-cost labels.
- Substation joins can be noisy because identifiers differ across ISO/RTO data exports.
- The monotonicity rule is useful but simplified; actual upgrade cost depends on topology, deliverability, thermal constraints, and study assumptions.
- Synthetic fallback results should be replaced by cleaned public-source results when final data files are available.

## Slide 13: Conclusions and Next Steps

- The project implements a complete queue-risk modeling platform aligned with the proposal.
- The PINN-style monotonic loss transfers the key idea from the research paper into a civil infrastructure prediction task.
- Future work: richer grid-topology features, better calibrated cost uncertainty, and project-similarity retrieval if time permits.

## Backup: Q&A Notes

- If asked why this is physics-informed: the model is constrained by a domain law evaluated at collocation points, not only by labels.
- If asked why not a direct PDE: queue-level public data does not expose grid-state variables needed for power-flow PDE/DAE solving, so the project adapts the PINN training framework rather than the exact original equations.
- If asked about data leakage: preprocessing is fit on the training split only, and the split is stratified by ISO/year cohort.
