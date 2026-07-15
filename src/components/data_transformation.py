import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, smiles_to_graph, parse_spectrum_column
from src.config import PREPROCESSOR_PATH


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = PREPROCESSOR_PATH


class MultimodalGraphSpectrumDataset(Dataset):
    """
    Holds the fused multimodal samples the CNN+GNN model consumes: one
    torch_geometric molecular graph (GNN branch) and one normalized IR
    spectrum vector (CNN branch) per molecule, paired with its
    molecular_weight target.
    """

    def __init__(self, graphs, spectra, targets):
        self.graphs = graphs
        self.spectra = spectra
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.spectra[idx], self.targets[idx]


class DataTransformation:
    """
    Builds the fusion preprocessing pipeline that turns each row's SMILES
    into a molecular graph (GNN modality) and each row's raw IR spectrum
    into a normalized fixed-length vector (CNN modality), so the downstream
    CNN+GNN regressor can consume both.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for the numeric (IR spectrum) modality
        transformation: a StandardScaler fitted bin-wise across the dataset.
        The graph modality needs no fitted transformer -- smiles_to_graph is
        a deterministic RDKit featurization applied identically at train and
        inference time.
        """
        try:
            scaler = StandardScaler()
            logging.info("Built StandardScaler for the IR spectrum (CNN) modality")
            return scaler

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test multimodal data completed")

            logging.info("Obtaining preprocessing (multimodal fusion) object")

            preprocessing_obj = self.get_data_transformer_object()

            # Real target: molecular weight (g/mol), computed with RDKit from
            # each molecule's actual SMILES structure - the task is to
            # predict it purely from the fused molecular graph + IR spectrum.
            target_column_name = "molecular_weight"

            train_spectra_raw = parse_spectrum_column(train_df["ir_spectrum_binned"])
            test_spectra_raw = parse_spectrum_column(test_df["ir_spectrum_binned"])

            logging.info(
                "Applying multimodal fusion preprocessing object on training dataframe and testing dataframe."
            )

            train_spectra = preprocessing_obj.fit_transform(train_spectra_raw)
            test_spectra = preprocessing_obj.transform(test_spectra_raw)

            train_graphs = [smiles_to_graph(smi) for smi in train_df["smiles"]]
            test_graphs = [smiles_to_graph(smi) for smi in test_df["smiles"]]

            train_targets = train_df[target_column_name].to_numpy(dtype=np.float32)
            test_targets = test_df[target_column_name].to_numpy(dtype=np.float32)

            train_dataset = MultimodalGraphSpectrumDataset(
                train_graphs,
                torch.tensor(train_spectra, dtype=torch.float32),
                torch.tensor(train_targets, dtype=torch.float32),
            )
            test_dataset = MultimodalGraphSpectrumDataset(
                test_graphs,
                torch.tensor(test_spectra, dtype=torch.float32),
                torch.tensor(test_targets, dtype=torch.float32),
            )

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj
            )

            return (
                train_dataset,
                test_dataset,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
