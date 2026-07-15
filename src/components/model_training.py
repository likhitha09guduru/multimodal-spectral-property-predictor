import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from src.exception import CustomException
from src.logger import logging

from src.utils import save_model, save_json, regression_metrics, num_atom_features
from src.config import MODEL_PATH, MODEL_CONFIG_PATH, ARTIFACTS_DIR


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_PATH
    gnn_hidden_dim: int = 64
    cnn_hidden_dim: int = 64
    fusion_hidden_dim: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32


def _collate_multimodal(batch):
    """
    Custom collate_fn for MultimodalGraphSpectrumDataset: batches the
    variable-sized molecular graphs with torch_geometric's Batch, and stacks
    the fixed-length IR spectra and targets as ordinary tensors.
    """
    graphs, spectra, targets = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    spectra_batch = torch.stack(spectra)
    target_batch = torch.stack(targets)
    return graph_batch, spectra_batch, target_batch


class GNNBranch(nn.Module):
    """Molecular-graph branch: two GCN layers + global mean pooling -> graph embedding."""

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch_index)
        return x


class CNNBranch(nn.Module):
    """IR-spectrum branch: two 1D-conv blocks + adaptive pooling -> spectral embedding."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, hidden_dim)

    def forward(self, spectrum):
        x = spectrum.unsqueeze(1)  # (batch, 1, IR_BINS)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 32)
        x = F.relu(self.fc(x))
        return x


class CNNGNNRegressor(nn.Module):
    """
    Fuses the GNN molecular-graph embedding with the CNN IR-spectrum
    embedding via concatenation, then a small MLP head predicts
    molecular_weight.
    """

    def __init__(self, node_feature_dim, gnn_hidden_dim, cnn_hidden_dim, fusion_hidden_dim):
        super().__init__()
        self.gnn_branch = GNNBranch(node_feature_dim, gnn_hidden_dim)
        self.cnn_branch = CNNBranch(cnn_hidden_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim + cnn_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, graph_batch, spectrum_batch):
        graph_embedding = self.gnn_branch(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        spectrum_embedding = self.cnn_branch(spectrum_batch)
        fused = torch.cat([graph_embedding, spectrum_embedding], dim=1)
        return self.fusion_head(fused).squeeze(-1)


class ModelTrainer:
    """
    Trains the CNN+GNN fusion model on the fused multimodal (molecular graph
    + IR spectrum) samples to predict molecular_weight, then evaluates it on
    the held-out test split.
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_dataset, test_dataset):
        try:
            logging.info("Building DataLoaders for the fused (graph + spectrum) multimodal samples")

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.model_trainer_config.batch_size,
                shuffle=True,
                collate_fn=_collate_multimodal,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.model_trainer_config.batch_size,
                shuffle=False,
                collate_fn=_collate_multimodal,
            )

            model = CNNGNNRegressor(
                node_feature_dim=num_atom_features(),
                gnn_hidden_dim=self.model_trainer_config.gnn_hidden_dim,
                cnn_hidden_dim=self.model_trainer_config.cnn_hidden_dim,
                fusion_hidden_dim=self.model_trainer_config.fusion_hidden_dim,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.model_trainer_config.learning_rate)
            loss_fn = nn.MSELoss()

            logging.info("Training the CNN+GNN fusion model")
            loss_history = []
            for epoch in range(self.model_trainer_config.epochs):
                model.train()
                epoch_loss = 0.0
                for graph_batch, spectrum_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    preds = model(graph_batch, spectrum_batch)
                    loss = loss_fn(preds, target_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * target_batch.size(0)

                mean_epoch_loss = epoch_loss / len(train_dataset)
                loss_history.append(mean_epoch_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logging.info(
                        f"Epoch {epoch + 1}/{self.model_trainer_config.epochs} "
                        f"- train MSE: {mean_epoch_loss:.4f}"
                    )

            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for graph_batch, spectrum_batch, target_batch in test_loader:
                    preds = model(graph_batch, spectrum_batch)
                    all_preds.append(preds)
                    all_targets.append(target_batch)

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_targets).numpy()
            metrics = regression_metrics(y_true, y_pred)
            r2_square = metrics["r2"]

            if r2_square < 0.6:
                logging.info(f"CNN+GNN model R2 on test set was only {r2_square:.4f} (below 0.6 threshold)")

            logging.info(
                f"CNN+GNN fusion model trained. Test R2: {metrics['r2']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}"
            )

            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                model=model,
            )

            # Persist the architecture hyperparameters used for THIS trained
            # model, so predict_pipeline.py can reconstruct the exact same
            # network at inference time instead of assuming ModelTrainerConfig
            # defaults haven't changed since training.
            save_json(
                obj={
                    "node_feature_dim": num_atom_features(),
                    "gnn_hidden_dim": self.model_trainer_config.gnn_hidden_dim,
                    "cnn_hidden_dim": self.model_trainer_config.cnn_hidden_dim,
                    "fusion_hidden_dim": self.model_trainer_config.fusion_hidden_dim,
                    **metrics,
                },
                file_path=MODEL_CONFIG_PATH,
            )
            logging.info(f"Saved model architecture config to {MODEL_CONFIG_PATH}")

            self._save_plots(loss_history, y_true, y_pred)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _save_plots(loss_history, y_true, y_pred):
        """
        Saves a training-loss curve and a predicted-vs-actual scatter plot to
        artifacts/, for including in README/reports. Skipped gracefully if
        matplotlib isn't installed rather than failing the whole training run.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless, no display needed
            import matplotlib.pyplot as plt

            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

            fig, ax = plt.subplots()
            ax.plot(range(1, len(loss_history) + 1), loss_history)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train MSE loss")
            ax.set_title("Training loss curve")
            fig.savefig(os.path.join(ARTIFACTS_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(y_true, y_pred, alpha=0.5, s=12)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", linewidth=1)  # y = x reference line
            ax.set_xlabel("Actual molecular weight")
            ax.set_ylabel("Predicted molecular weight")
            ax.set_title("Predicted vs. actual (test set)")
            fig.savefig(os.path.join(ARTIFACTS_DIR, "pred_vs_actual.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            logging.info(f"Saved loss_curve.png and pred_vs_actual.png to {ARTIFACTS_DIR}/")
        except ImportError:
            logging.info("matplotlib not installed; skipping loss_curve.png / pred_vs_actual.png")
