# Limitations and responsible use

**Data.** The IBM datasets are synthetic: injected laundering motifs are easier to recover than
real laundering, and the labels are exact, which real labels never are (real labels come from
investigated cases and are biased towards what past systems flagged). Elliptic is a Bitcoin graph
with anonymised features; conclusions do not transfer automatically to bank data.

**Detectors.** Every setting of `configs/patterns.yaml` was chosen on the validation splits of
IBM AML HI-Small and LI-Small (the package defaults were chosen for the synthetic generator). The
tuned scanner reaches a test PR-AUC of **0.503** on HI-Small (95 % bootstrap interval 0.479–0.526)
and 0.189 on LI-Small, without any learned model (`benchmarks/results/REPORT.md`): the objective
of 0.5 is reached, narrowly. It was 0.005 / 0.001 with the original detectors. The steps, each
measured on the HI-Small validation slice: the IBM motifs change currency at almost every hop, so
amounts are compared in US dollars (validation PR-AUC 0.053 → 0.054); they last 27–80 h, longer
than the ≈ 51 h validation slice, so each slice is scanned with the 4 unlabelled days before it
(→ 0.085); every laundering attempt annotated in the training slice is an ACH transfer, so the
search is restricted to ACH (→ 0.162, full retuning 0.258; the same retuning over all payment
formats stops at 0.164, and at 0.029 instead of 0.093 on LI-Small); fans must spread over at
least a day with a mean amount of 5 000 USD and internal / cross-currency payments are excluded
(→ 0.288). The last two steps were judged out of sample, calibrating on one half of the
validation slice and measuring on the other: a precision per band of alert score instead of one
per detector (0.177 → 0.281) and the `relay` detector for the IBM *stack* motif (→ 0.387). The
test slice was only used to report the results (three times along the way: 0.340, 0.399, 0.503),
never to choose a setting.

Three caveats. The ACH restriction and the cross-currency exclusion are properties of the IBM
simulator chosen on its validation data — on other data, leave `transaction_types` and `exclude`
empty or re-examine them. The calibration bands use the labels of the validation slice (as the
per-detector calibration always did), so the scanner is rule-based, not label-free. And 36 % of
the laundering of the HI-Small validation slice belongs to no annotated motif: no motif detector
can see it (bipartite motifs, also without structure, stay undetected), which caps what motifs
alone can reach. Stacked with a model (hybrid detector), the motifs raise the HI-Small test
PR-AUC of the random forest, HistGradientBoosting and GraphSAGE models to 0.614, 0.641 and 0.641
(0.519, 0.484, 0.504 alone; one seed) and LI-Small to 0.236 with the random forest
(`benchmarks/results_parts/hybrid/README.md`).
Learned models inherit the biases of their labels. The cycle search is bounded by
budgets and can be incomplete on dense graphs (a warning is logged and `truncated` is set). The
risk-community detector ignores busy accounts (`max_member_degree`) and only sees groups whose
internal flows fit in `max_span`; schemes that reuse very active accounts are left to the other
detectors. The unsupervised auto-encoder flags what is *unusual*, which is not the same as what
is illicit: unusual but legitimate behaviour (a one-off large purchase) scores high too, and the
auto-encoder learns to reconstruct anomalies if it is trained for many epochs.

**Interfaces.** The web interface keeps the analysed tables in server memory (5 investigations,
2 million transactions each) and has no user accounts (only an optional shared API key); it is
meant for a single analyst on a workstation or behind an authenticating proxy. GNN scores use a fixed sampling seed (or full
neighbourhoods with `inference: full`), so they are reproducible, but the score of a transaction
still depends on the transactions sent with it as context.

**Explanations.** Local and approximate: occlusion measures a *score change*, not a cause;
GNNExplainer, PGExplainer, Captum attributions and attention are known to disagree with each
other. Check fidelity before relying on them (see {doc}`explainability`).

**Optional components.** The code that needs an extra (GNN training and Lightning, the
GNNExplainer / attention adapters, the FastAPI wrapper, Parquet / Polars input, MLflow, the
Hugging Face push) is tested only when the extra is installed; without it the corresponding tests
are skipped, so a plain `pytest` run on a minimal install does not cover them. The continuous
integration runs them in dedicated jobs (see `CONTRIBUTING.md`). The notebooks `01`–`03` run on
synthetic data; `04` needs a GPU runtime and the IBM files. Report problems together with the
exact versions of PyTorch and PyG.

**Responsible use.**

* Scores prioritise human review; they are not accusations. Keep analysts in the loop and
  document the decision, not just the score.
* Check error rates across customer groups before any deployment; the toolkit does not do
  fairness auditing for you.
* Transaction data are personal data: apply the retention, minimisation and access rules of your
  jurisdiction, do not log request bodies, and do not upload real customer data to third-party
  services (including public notebooks and Hub repositories).
* The package is research software under the MIT licence **without warranty**; it is not a
  compliance product.
