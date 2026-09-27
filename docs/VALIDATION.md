# Validation record

Review date: 27 September 2026. The 4 code cells in the included notebook were executed in order
in a fresh IPython process launched from `notebooks/`, on Linux with Python 3.12 and CPU execution.
The saved notebook contains the resulting text and figure outputs, with no saved execution errors.

Three regression tests passed: returned assignments match final centroids after one update;
empty clusters do not produce NaN centers; invalid parameters/data are rejected.
The notebook executed both implementations and rendered the PCA and silhouette plots. Diagnosis
agreement uses the same observations used for clustering and is explicitly labeled in-sample.

Core package versions match the pins in `requirements.txt`. The notebook web interface and installation
on Windows/macOS were not separately exercised. Stochastic results can vary across platforms and
library builds. This validation covers the supplied example and focused regression cases, not every
possible input or production deployment.

Re-run regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```
