import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config import ARTIFACTS_DIR


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reads the fused multimodal dataset built by
            # notebook/build_multimodal_dataset.py from the REAL, published
            # "IR-NMR Multimodal Computational Spectra Dataset for 177K
            # Patent-Extracted Organic Molecules" (Zipoli, Alberts, Laino;
            # IBM Research; Zenodo record 16417648; CDLA-Permissive-2.0).
            # Run that script first (requires internet + pyarrow + rdkit)
            # to produce notebook/data/multimodal_spectra_dataset.csv.
            # Each row holds: smiles (-> GNN molecular graph), a binned raw
            # IR spectrum (-> 1D-CNN input), and the molecular_weight target.
            data_path = "notebook/data/multimodal_spectra_dataset.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"{data_path} not found. Run "
                    "`python notebook/build_multimodal_dataset.py` (or add "
                    "`--demo` for a quick synthetic run) first to build the "
                    "fused SMILES + IR spectrum dataset."
                )
            df = pd.read_csv(data_path)
            logging.info("Read the real, fused IR + NMR multimodal dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_dataset, test_dataset, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_dataset, test_dataset))
