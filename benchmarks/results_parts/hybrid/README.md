# Hybrid detector on IBM AML (model + motifs)

Produced by `python benchmarks/run_hybrid.py benchmarks/results_parts/hybrid/hybrid.json HI-Small:rf,hgb,sage LI-Small:rf,hgb,sage`
(run from the repository root, with the IBM files in `data/` and the saved models in `runs/<model>-<dataset>`).

The base models are the detectors trained by the private Kaggle kernels (seed 0, same temporal
60/20/20 split; not retrained). The motif scanner uses `configs/patterns.yaml` (relay detector,
8 calibration bands); its calibration and the stacker (logistic regression on the model logit,
the motif score and a motif-fired flag) are fitted on the validation slice; every number below
is a test PR-AUC computed on the same scores. Bold: the hybrid beats both of its parts.

| dataset | base model | model alone | motifs alone | hybrid | stacker coefficients |
|---|---|---:|---:|---:|---|
| HI-Small | random forest | 0.519 | 0.503 | **0.614** | model_logit +0.55, motif_score +3.36, motif_fired +2.46 |
| HI-Small | HistGradientBoosting | 0.484 | 0.503 | **0.641** | model_logit +1.21, motif_score +1.84, motif_fired +2.16 |
| HI-Small | GraphSAGE | 0.504 | 0.503 | **0.641** | model_logit +0.85, motif_score +1.84, motif_fired +1.82 |
| LI-Small | random forest | 0.191 | 0.189 | **0.236** | model_logit +0.44, motif_score +0.64, motif_fired +1.15 |
| LI-Small | HistGradientBoosting | 0.106 | 0.189 | **0.189** | model_logit +1.21, motif_score +0.54, motif_fired +0.93 |
| LI-Small | GraphSAGE | 0.175 | 0.189 | 0.184 | model_logit +1.14, motif_score +0.15, motif_fired +0.24 |

Single seed: compare with the 3-seed means of `benchmarks/results/REPORT.md` with care (the
models alone differ from those means by their seed; GraphSAGE alone is 0.504 here vs. 0.508 stored
in its `config.json` because inference samples neighbourhoods).
