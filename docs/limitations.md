# Limitations and responsible use

**Data.** The IBM datasets are synthetic: injected laundering motifs are easier to recover than
real laundering, and the labels are exact, which real labels never are (real labels come from
investigated cases and are biased towards what past systems flagged). Elliptic is a Bitcoin graph
with anonymised features; conclusions do not transfer automatically to bank data.

**Detectors.** The motif settings of `configs/patterns.yaml` were tuned on the validation
splits of IBM AML HI-Small and LI-Small (the package defaults were chosen for the synthetic
generator). The tuned scanner reaches a test PR-AUC of 0.340 (HI-Small) and 0.147 (LI-Small),
without any training (`benchmarks/results/REPORT.md`), still below the objective of 0.5 and below
the best learned models on HI-Small. Three changes made this possible, each measured on the
HI-Small validation slice: the IBM motifs change currency at almost every hop, so amounts are now
compared in US dollars (validation PR-AUC 0.053 → 0.054); they last 27–80 h, longer than the
≈ 51 h validation slice, so each slice is scanned with the 4 unlabelled days before it (→ 0.085);
every laundering attempt annotated in the training slice is an ACH transfer, so the search is
restricted to ACH (→ 0.162; the full retuning reaches 0.258). Before these changes the cycle
detector recovered none of the annotated test cycles; it now recovers 81 % of them on HI-Small.
The ACH restriction is a property of the IBM simulator, chosen on validation data: on other data,
leave `transaction_types` empty or re-examine it. Only 62 % (HI) and 29 % (LI) of the laundering
transactions belong to an annotated motif, and bipartite / stack / random motifs have no dedicated
detector. The hybrid detector (model + motifs) improves the HI-Small test PR-AUC of the random
forest, HistGradientBoosting and GraphSAGE models (0.519 → 0.562, 0.484 → 0.545, 0.504 → 0.527;
one seed) but not on LI-Small (`benchmarks/results_parts/hybrid/README.md`).
Learned models inherit the biases of their labels. The cycle search is bounded by
budgets and can be incomplete on dense graphs (a warning is logged and `truncated` is set). The
risk-community detector ignores busy accounts (`max_member_degree`) and only sees groups whose
internal flows fit in `max_span`; schemes that reuse very active accounts are left to the other
detectors. The unsupervised auto-encoder flags what is *unusual*, which is not the same as what
is illicit: unusual but legitimate behaviour (a one-off large purchase) scores high too, and the
auto-encoder learns to reconstruct anomalies if it is trained for many epochs.

**Interfaces.** The web interface keeps the analysed tables in server memory (5 investigations,
2 million transactions each) and has no user accounts; it is meant for a single analyst on a
workstation or behind an authenticating proxy. GNN scores use a fixed sampling seed (or full
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
