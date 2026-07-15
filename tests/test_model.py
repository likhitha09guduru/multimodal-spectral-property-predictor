import torch
from torch_geometric.data import Batch

from src.components.model_training import CNNGNNRegressor
from src.utils import smiles_to_graph, num_atom_features
from src.config import IR_BINS


def _build_model(gnn_hidden=8, cnn_hidden=8, fusion_hidden=8):
    return CNNGNNRegressor(
        node_feature_dim=num_atom_features(),
        gnn_hidden_dim=gnn_hidden,
        cnn_hidden_dim=cnn_hidden,
        fusion_hidden_dim=fusion_hidden,
    )


def test_forward_pass_shapes():
    model = _build_model()
    graphs = [smiles_to_graph(s) for s in ["CCO", "c1ccccc1"]]
    graph_batch = Batch.from_data_list(graphs)
    spectrum_batch = torch.randn(2, IR_BINS)

    out = model(graph_batch, spectrum_batch)
    assert out.shape == (2,)


def test_single_sample_forward_pass():
    model = _build_model()
    graph_batch = Batch.from_data_list([smiles_to_graph("CC(=O)O")])
    spectrum_batch = torch.randn(1, IR_BINS)

    out = model(graph_batch, spectrum_batch)
    assert out.shape == (1,)
