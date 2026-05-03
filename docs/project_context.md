# Project Context and Deliverable Checklist

## Course Expectations

The final presentation is 13 minutes plus 5 minutes of Q&A. The recommended structure is:

- Introduction, about 2 minutes: motivate the grid interconnection bottleneck and why queue prediction matters.
- Methodology, about 4 minutes: explain features, baselines, neural networks, GPR uncertainty, and the physics-informed monotonic loss.
- Results, about 5 minutes: show evaluation setup, metrics, ablation results, and example plots.
- Discussion and conclusions, about 2 minutes: summarize insights, limitations, and future work.

The final report should be at least six single-spaced pages in research-paper style and include:

- Title, authors, and abstract.
- Introduction and related background.
- Methodology with equations or pseudocode.
- Experimental setup, metrics, and results.
- Discussion and conclusions.
- Individual contributions.
- LLM use statement.
- References and code/data links.

## Proposal Commitments Implemented

- Withdrawal prediction: logistic regression baseline and neural-network classifier.
- Timeline prediction: linear/Ridge baseline and neural-network regressor.
- Cost bucket prediction: neural-network classifier on the cost-labeled subset.
- PINN reproduction: composite objective `L = L_data + lambda L_physics`, where `L_physics` penalizes non-monotonic cost predictions when MW is increased at collocation points.
- GPR uncertainty: Gaussian process regression on log network-upgrade cost with 90% prediction-interval coverage.
- Progress-update refinements: cost modeling is restricted to labeled rows, and the autoencoder is deprioritized relative to the PINN ablation.

## Research Paper Translation

Raissi et al. train neural networks using a supervised data loss plus a residual term evaluated at collocation points. In their paper, the residual comes from a PDE such as Burgers, Navier-Stokes, or KdV. This project keeps the same training structure but changes the physical residual to a power-systems monotonicity rule:

```text
L(theta) = L_data(theta) + lambda * L_monotonic(theta)
L_monotonic = mean(max(0, E[cost_bucket | x] - E[cost_bucket | x with larger MW]))
```

This is a civil-infrastructure adaptation rather than a direct PDE solver: the model is still physics-informed because the training objective constrains predictions using known domain structure instead of fitting labels alone.

## Current Demo Results

The checked-in `outputs/metrics.json` was generated with:

```bash
PYTHONPATH=src python3 -m interconnection_ml.run_pipeline --output outputs
```

Use the generated `outputs/metrics.json` and `outputs/model_comparison.csv` for the exact numbers in the final paper and slides. The documentation intentionally avoids fixed metric claims so the written story stays tied to the latest run rather than to stale hand-entered values.

The synthetic dataset is intentionally used as a submission-safe fallback. For the final report, rerun the same pipeline with a cleaned CSV from the LBNL/MISO/PJM/CAISO sources if those files are available.
