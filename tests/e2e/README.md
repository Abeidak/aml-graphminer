# End-to-end checks of a running service

Two acceptance checks that use a running server like a client would (local or deployed):

```bash
# REST API, investigation API and web assets (33 checks with a GNN model, 25-26 otherwise)
python scripts/check_service.py https://aml-graphminer-demo.onrender.com
python scripts/check_service.py http://127.0.0.1:8000 --upload-limit-mb 1 --report api.json

# web interface in a real headless browser (Edge or Chrome already installed; nothing downloaded)
cd tests/e2e && npm install
python -c "import aml_graphminer as aml; aml.generate_transactions(200, 1500, seed=21).to_csv('upload.csv', index=False)"
node ui_check.js http://127.0.0.1:8000 shots upload.csv
```

They are not part of `pytest` (they need a server, and the second one a browser). The datasets
they create on the server are deleted at the end.
