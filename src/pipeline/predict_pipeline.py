import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

from src.components.model_training import CNNGNNRegressor
from src.config import ARTIFACTS_DIR
from src.exception import CustomException
from src.utils import is_valid_smiles, load_json, load_model, load_object, smiles_to_graph


class PredictionInputError(Exception):
    """
    Raised for bad *user* input (invalid SMILES, wrong-length or non-numeric
    IR spectrum) as opposed to CustomException, which wraps unexpected
    internal errors. Kept separate so app.py can map this to HTTP 400
    instead of HTTP 500.
    """

    pass


class CustomData:
    """
    Wraps a single raw prediction request -- a SMILES string (GNN modality
    input) and a raw IR spectrum (CNN modality input) -- before any
    validation or scaling has been applied.
    """

    def __init__(self, smiles: str, ir_spectrum):
        self.smiles = smiles
        self.ir_spectrum = list(ir_spectrum)


class PredictPipeline:
    """
    Loads the trained CNN+GNN fusion model, its architecture config, and the
    fitted IR-spectrum scaler, then exposes .predict(CustomData) -> a length-1
    np.ndarray with the predicted molecular weight.

    Artifacts are loaded lazily (on first .predict()/._ensure_loaded() call)
    and cached on the instance, so construction itself never touches disk --
    that's what lets app.py's /health check call _ensure_loaded() explicitly
    to report whether the model is ready without side effects at import time.
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or ARTIFACTS_DIR
        self.model_path = os.path.join(self.model_dir, "model.pt")
        self.model_config_path = os.path.join(self.model_dir, "model_config.json")
        self.preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")

        self._model = None
        self._preprocessor = None

    def _ensure_loaded(self):
        """
        Loads the model + preprocessor from disk if not already cached.
        Raises FileNotFoundError (not CustomException) when artifacts are
        missing, so callers -- and tests -- can distinguish "not trained
        yet" from a genuine internal error.
        """
        if self._model is not None and self._preprocessor is not None:
            return

        for path in (self.model_path, self.model_config_path, self.preprocessor_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model artifact not found: {path}. Train the model first with "
                    "`python src/pipeline/train_pipeline.py`."
                )

        try:
            config = load_json(self.model_config_path)
            model = CNNGNNRegressor(
                node_feature_dim=config["node_feature_dim"],
                gnn_hidden_dim=config["gnn_hidden_dim"],
                cnn_hidden_dim=config["cnn_hidden_dim"],
                fusion_hidden_dim=config["fusion_hidden_dim"],
            )
            model = load_model(self.model_path, model)
            preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

        self._model = model
        self._preprocessor = preprocessor

    def predict(self, data: CustomData) -> np.ndarray:
        # 1. Validate user input BEFORE touching disk, so bad input fails
        #    fast without requiring a trained model to be present.
        if not is_valid_smiles(data.smiles):
            raise PredictionInputError(f"Invalid SMILES string: {data.smiles!r}")

        if not data.ir_spectrum or any(not isinstance(v, (int, float)) for v in data.ir_spectrum):
            raise PredictionInputError("IR spectrum must be a non-empty list of numbers.")

        # 2. Load model + preprocessor (may raise FileNotFoundError).
        self._ensure_loaded()

        # 3. Validate the spectrum length against what the preprocessor was
        #    actually fit on, now that we know that length.
        expected_len = getattr(self._preprocessor, "n_features_in_", None)
        if expected_len is not None and len(data.ir_spectrum) != expected_len:
            raise PredictionInputError(
                f"IR spectrum has {len(data.ir_spectrum)} values; expected {expected_len}."
            )

        try:
            spectrum_scaled = self._preprocessor.transform(
                np.array(data.ir_spectrum, dtype=np.float32).reshape(1, -1)
            )
            spectrum_tensor = torch.tensor(spectrum_scaled, dtype=torch.float32)

            graph = smiles_to_graph(data.smiles)
            graph_batch = Batch.from_data_list([graph])

            self._model.eval()
            with torch.no_grad():
                preds = self._model(graph_batch, spectrum_tensor)

            return preds.numpy()
        except Exception as e:
            raise CustomException(e, sys)
