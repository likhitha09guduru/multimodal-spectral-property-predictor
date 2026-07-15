# Multimodal Spectral Property Predictor (CNN + GNN)

Predicts a molecule's **molecular weight** from two fused modalities:

- **GNN branch** — the molecule's SMILES is turned into a graph (atoms = nodes, bonds = edges) and passed through 2 GCN layers + global mean pooling.
- **CNN branch** — the molecule's IR spectrum (binned to a fixed-length grid) is passed through 1D-conv layers + adaptive pooling.

The two resulting embeddings are concatenated and passed through a small MLP head to predict `molecular_weight`.

## Architecture

```
SMILES                                    IR Spectrum
   │                                            │
   ▼                                            ▼
2× GCNConv + global mean pool          2× Conv1d + adaptive avg pool
   │                                            │
   ▼                                            ▼
Graph embedding (gnn_hidden_dim)     Spectral embedding (cnn_hidden_dim)
   │                                            │
   └───────────────┬────────────────────────────┘
                    ▼
              concat -> Linear -> ReLU -> Linear
                    │
                    ▼
           Predicted molecular weight
```

Implementation: `GNNBranch` and `CNNBranch` in `src/components/model_training.py`, fused by `CNNGNNRegressor`.

Data source: the real *"IR-NMR Multimodal Computational Spectra Dataset for 177K Patent-Extracted Organic Molecules"* (Zipoli, Alberts, Laino; IBM Research; [Zenodo record 16417648](https://doi.org/10.5281/zenodo.16417648); CDLA-Permissive-2.0).

## Setup

```bash
pip install -r requirements.txt
```

`torch-geometric` sometimes needs extra install flags depending on your OS/CUDA version — see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html if `pip install torch-geometric` fails outright.

## 1. Build the dataset

```bash
# real data (downloads ~8GB from Zenodo, slow)
python notebook/build_multimodal_dataset.py

# OR a small synthetic stand-in for quickly testing the pipeline
python notebook/build_multimodal_dataset.py --demo --n 500
```

This writes `notebook/data/multimodal_spectra_dataset.csv`.

## 2. Train

```bash
python src/pipeline/train_pipeline.py
```

Writes `artifacts/model.pt` (CNN+GNN weights) and `artifacts/preprocessor.pkl` (IR-spectrum scaler).

## 3. Serve predictions (local dev)

```bash
python app.py
```

Then visit `http://localhost:5000/predictdata` and submit a SMILES string plus a comma-separated, 256-point binned IR spectrum. `GET /health` reports whether the trained model loaded successfully.

Training writes three files to `artifacts/`: `model.pt` (weights), `model_config.json` (the exact hidden-dim hyperparameters used plus final test-set R²/MAE/RMSE, so inference always reconstructs the matching architecture), and `preprocessor.pkl` (the IR-spectrum scaler). If `matplotlib` is installed, it also writes `loss_curve.png` and `pred_vs_actual.png` (see Model evaluation below).

## Model evaluation

`model_config.json` (written after every training run) reports:

- **R²** — variance in `molecular_weight` explained by the model
- **MAE** — mean absolute error, in g/mol
- **RMSE** — root mean squared error, in g/mol (penalizes large misses more than MAE)

Two plots are also generated in `artifacts/`:

- **`loss_curve.png`** — training MSE loss per epoch, to check for convergence/overfitting
- **`pred_vs_actual.png`** — predicted vs. actual molecular weight on the held-out test set, with a `y = x` reference line; tighter clustering around that line means better fit

These are regenerated on every `train_pipeline.py` run and aren't checked into the repo (they depend on your trained model), so run training once to produce them for your own README/report.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

- `tests/test_utils.py` — SMILES validation, graph featurization, spectrum parsing
- `tests/test_model.py` — CNN+GNN forward-pass shapes
- `tests/test_data.py` — dataset builder schema + data transformation, using the fast synthetic `--demo` path (no download, no trained model needed)
- `tests/test_prediction.py` — `PredictPipeline` input validation (bad SMILES, wrong-length spectrum, missing artifacts) and a full predict() round trip against a freshly-initialized (untrained) model

None of these require the real ~8GB dataset or a fully trained model — they're fast enough to run in CI on every push.

## Production deployment

Don't use `python app.py` in production — it runs Flask's single-threaded dev server. Instead:

```bash
pip install -r requirements.txt
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 wsgi:application
```

Each gunicorn worker loads its own copy of the model into memory (loaded once, cached, not reloaded per-request), so size `-w` against available RAM rather than CPU count alone.

### Docker

```bash
docker build -t spectral-predictor .
docker run -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts spectral-predictor
```

The image doesn't bake in `artifacts/` (trained weights) by default — mount it as a volume, or `COPY artifacts ./artifacts` in the Dockerfile if you want it baked into the image instead.

### Config via environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ARTIFACTS_DIR` | `artifacts` | Where `model.pt` / `model_config.json` / `preprocessor.pkl` live |
| `PORT` | `5000` | Port for the dev server (`python app.py`) |
| `HOST` | `0.0.0.0` | Host for the dev server |
| `FLASK_DEBUG` | `false` | Never set `true` in production |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LOG_TO_FILE` | `false` | Also write logs to `logs/` (off by default; container logs should go to stdout) |

### What's still out of scope

This covers correctness, config/model coupling, input validation, and running under a real WSGI server/container. It does **not** include auth, rate limiting, TLS termination, horizontal-scaling/load-balancer config, or a model-monitoring/retraining pipeline — add those at the layer appropriate to where you're actually deploying it (e.g. a reverse proxy or your cloud provider's ingress for TLS and rate limiting).

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs dependencies, checks formatting with `black --check`, lints with `flake8`, then runs `pytest`. Format locally before pushing with:

```bash
black src app.py wsgi.py tests notebook
```

## Project layout

```
notebook/build_multimodal_dataset.py   # builds the fused SMILES + IR spectrum dataset
src/config.py                          # shared constants (IR_BINS, artifact paths)
src/components/data_ingestion.py       # reads + train/test splits the fused dataset
src/components/data_transformation.py  # SMILES -> graph, IR spectrum -> scaled tensor
src/components/model_training.py       # CNNGNNRegressor model + training loop + metrics/plots
src/components/model_trainer.py        # re-exports model_training.py (kept for structure parity)
src/pipeline/train_pipeline.py         # orchestrates ingestion -> transformation -> training
src/pipeline/predict_pipeline.py       # loads + caches saved model/preprocessor, validates input, runs inference
app.py                                 # Flask UI for single-sample predictions
wsgi.py                                # gunicorn entry point
Dockerfile                             # production container image
tests/                                 # pytest suite (no dataset/training required)
.github/workflows/ci.yml               # lint + test on every push/PR
```
