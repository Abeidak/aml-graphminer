# Security policy

## Reporting a vulnerability

Please do **not** open a public issue. Report privately through the repository's *Security →
Report a vulnerability* page (GitHub private vulnerability reporting) or by e-mail to the
maintainers listed in `CITATION.cff` / `pyproject.toml`. Include the version, a minimal
reproduction and the impact. We aim to acknowledge reports within a week.

## Scope and known risks

* **Model files are code-adjacent.** Detector directories are read with `joblib` (tabular) and
  `torch.load` (GNN). Only load models you trained or obtained from a source you trust.
* **The REST API has no authentication, rate limiting or TLS.** Run it behind your own gateway;
  the compose file binds it to `127.0.0.1`.
* **Data protection.** Transaction data are personal data. Do not commit them, do not put them in
  public notebooks, issues or Hub repositories, and do not log request bodies.
