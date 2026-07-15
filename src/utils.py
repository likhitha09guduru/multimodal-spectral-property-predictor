import os
import sys
import json

import numpy as np
import pandas as pd
import pickle
import torch
from torch_geometric.data import Data
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_json(obj: dict, file_path: str):
    """Saves a plain dict as JSON, e.g. the model's architecture hyperparameters."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as file_obj:
            json.dump(obj, file_obj, indent=2)

    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file_obj:
            return json.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def save_model(file_path, model):
    """Saves a PyTorch model's state_dict, mirroring save_object's role for torch models."""
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        torch.save(model.state_dict(), file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_model(file_path, model):
    """Loads a PyTorch model's state_dict into an already-constructed model instance."""
    try:
        model.load_state_dict(torch.load(file_path, map_location="cpu"))
        model.eval()
        return model

    except Exception as e:
        raise CustomException(e, sys)


# ---------------------------------------------------------------------------
# Atom/bond featurization for the GNN branch (molecular graph modality)
# ---------------------------------------------------------------------------

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "H", "Si", "B"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]


def _one_hot(value, choices):
    encoding = [0] * (len(choices) + 1)  # +1 slot for "other"
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def _atom_features(atom):
    features = []
    features += _one_hot(atom.GetSymbol(), ATOM_TYPES)
    features += _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())
    return features


def is_valid_smiles(smiles: str) -> bool:
    """Cheap validity check used to reject bad input before it reaches the model."""
    if not smiles or not isinstance(smiles, str):
        return False
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles) is not None


def smiles_to_graph(smiles: str) -> Data:
    """
    Converts a SMILES string into a torch_geometric.data.Data graph: atoms
    become nodes (with a feature vector each), bonds become undirected edges.
    This is the input format the GNN branch of the model consumes.
    """
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")

        node_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(node_feats, dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        if len(edge_index) == 0:
            # single-atom molecule: self-loop so the graph isn't empty
            edge_index = [[0, 0]]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    except Exception as e:
        raise CustomException(e, sys)


def num_atom_features() -> int:
    """Feature vector length produced by smiles_to_graph, needed to size the GNN's input layer."""
    return len(ATOM_TYPES) + 1 + len(HYBRIDIZATIONS) + 1 + 4


def parse_spectrum_column(series: pd.Series) -> np.ndarray:
    """Parses a column of JSON-encoded IR spectrum arrays into a 2D numpy array (N, IR_BINS)."""
    try:
        return np.array([json.loads(s) for s in series], dtype=np.float32)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_regression(y_true, y_pred) -> float:
    try:
        return r2_score(y_true, y_pred)
    except Exception as e:
        raise CustomException(e, sys)


def regression_metrics(y_true, y_pred) -> dict:
    """R², MAE, and RMSE together, for logging/model_config/README reporting."""
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        }
    except Exception as e:
        raise CustomException(e, sys)
