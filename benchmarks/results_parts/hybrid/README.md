# Hybrid detector on IBM AML (model + motifs)

Produced by `python benchmarks/run_hybrid.py benchmarks/results_parts/hybrid/hybrid.json HI-Small:rf,hgb,sage LI-Small:rf,hgb,sage`
(run from the repository root, with the IBM files in `data/` and the saved models in `runs/<model>-<dataset>`).

The base models are the detectors trained by the private Kaggle kernels (seed 0, same temporal
60/20/20 split; not retrained). The motif scanner uses `configs/patterns.yaml`; its calibration and
the stacker (logistic regression on the model logit, the motif score and a motif-fired flag) are
fitted on the validation slice; every number below is a test PR-AUC computed on the same scores.

| dataset | base model | model alone | motifs alone | hybrid | stacker coefficients |
|---|---|---:|---:|---:|---|
| HI-Small | random forest | 0.519 | 0.340 | **0.562** | model_logit +0.63, motif_score +8.60, motif_fired +1.57 |
| HI-Small | HistGradientBoosting | 0.484 | 0.340 | **0.545** | model_logit +1.26, motif_score +0.75, motif_fired +0.64 |
| HI-Small | GraphSAGE | 0.504 | 0.340 | **0.527** | model_logit +0.96, motif_score +0.55, motif_fired +0.27 |
| LI-Small | random forest | 0.191 | 0.147 | 0.186 | model_logit +0.43, motif_score +0.41, motif_fired +1.37 |
| LI-Small | HistGradientBoosting | 0.106 | 0.147 | 0.113 | model_logit +1.12, motif_score +0.21, motif_fired +0.13 |
| LI-Small | GraphSAGE | 0.175 | 0.147 | 0.168 | model_logit +1.21, motif_score +0.06, motif_fired -0.53 |

Single seed: compare with the 3-seed means of `benchmarks/results/REPORT.md` with care (the
models alone differ from those means by their seed; GraphSAGE alone is 0.504 here vs. 0.508 stored
in its `config.json` because inference samples neighbourhoods).
