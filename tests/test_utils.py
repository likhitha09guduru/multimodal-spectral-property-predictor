import numpy as np

from src.utils import (
    smiles_to_graph,
    num_atom_features,
    is_valid_smiles,
    parse_spectrum_column,
)


def test_valid_smiles_accepted():
    assert is_valid_smiles("CCO") is True


def test_invalid_smiles_rejected():
    assert is_valid_smiles("not_a_smiles!!") is False
    assert is_valid_smiles("") is False
    assert is_valid_smiles(None) is False


def test_smiles_to_graph_feature_dim_matches_num_atom_features():
    graph = smiles_to_graph("CCO")
    assert graph.x.shape[1] == num_atom_features()


def test_smiles_to_graph_single_atom_gets_self_loop():
    graph = smiles_to_graph("[He]")
    assert graph.edge_index.shape[1] >= 1


def test_parse_spectrum_column_roundtrip():
    import json
    import pandas as pd

    series = pd.Series([json.dumps([1.0, 2.0, 3.0]), json.dumps([4.0, 5.0, 6.0])])
    arr = parse_spectrum_column(series)
    assert arr.shape == (2, 3)
    assert np.allclose(arr[0], [1.0, 2.0, 3.0])
