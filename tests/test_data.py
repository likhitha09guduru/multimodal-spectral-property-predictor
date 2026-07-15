"""
Uses the synthetic --demo dataset (no network, no 8GB download) to exercise
build_multimodal_dataset -> data_transformation end-to-end at small scale.
"""

import json
import os

from notebook.build_multimodal_dataset import build_demo_dataset
from src.config import IR_BINS
from src.components.data_transformation import DataTransformation


def test_build_demo_dataset_schema():
    df = build_demo_dataset(n=10, seed=1)

    assert len(df) == 10
    assert list(df.columns) == ["id", "smiles", "ir_spectrum_binned", "molecular_weight"]
    assert (df["molecular_weight"] > 0).all()

    spectrum = json.loads(df["ir_spectrum_binned"].iloc[0])
    assert len(spectrum) == IR_BINS


def test_data_transformation_produces_matching_shapes(tmp_path):
    df = build_demo_dataset(n=12, seed=2)
    train_df = df.iloc[:8]
    test_df = df.iloc[8:]

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    dt = DataTransformation()
    dt.data_transformation_config.preprocessor_obj_file_path = str(tmp_path / "preprocessor.pkl")

    train_dataset, test_dataset, preprocessor_path = dt.initiate_data_transformation(
        str(train_path), str(test_path)
    )

    assert len(train_dataset) == 8
    assert len(test_dataset) == 4
    assert os.path.exists(preprocessor_path)

    graph, spectrum, target = train_dataset[0]
    assert spectrum.shape[0] == IR_BINS
    assert target.ndim == 0  # scalar molecular_weight
    assert graph.x.shape[0] > 0  # at least one atom
