import os

# Fixed-length grid every IR spectrum is binned/interpolated onto. Shared by
# the dataset builder (notebook/build_multimodal_dataset.py), the Flask UI
# (app.py), and DataTransformation, so they can never silently disagree.
IR_BINS = 256

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")
MODEL_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "model_config.json")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
