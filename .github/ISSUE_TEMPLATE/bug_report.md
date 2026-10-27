---
name: Bug report
about: Something does not work as documented
labels: bug
---

**What happened / what you expected**

**Minimal reproduction** (synthetic data from `amlgm generate` if possible — please do not paste real customer data)

**Environment**

```text
python -c "import aml_graphminer, platform, sys; print(aml_graphminer.__version__, platform.platform(), sys.version)"
pip list | grep -i -E "torch|geometric|lightning|pandas|numpy|scikit"
```
