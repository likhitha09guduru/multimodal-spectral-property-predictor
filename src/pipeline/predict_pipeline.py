import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Wraps a single real multimodal sample: IR modality band-intensity
    features, NMR modality shift-summary features, and the categorical
    composition-descriptor modality (all engineered by
    notebook/build_multimodal_dataset.py from the real Zenodo IR-NMR
    dataset), and turns them into the single-row dataframe the fusion
    preprocessor expects.
    """

    def __init__(self,
                 contains_nitrogen: str,
                 contains_oxygen: str,
                 contains_halogen: str,
                 contains_sulfur: str,
                 ir_band_ohnh_stretch_3200_3550: float,
                 ir_band_ch_stretch_2850_3000: float,
                 ir_band_carbonyl_1650_1750: float,
                 ir_band_aromatic_1450_1600: float,
                 ir_band_fingerprint_500_1500: float,
                 h_nmr_shift_mean: float,
                 h_nmr_shift_std: float,
                 h_nmr_shift_max: float,
                 h_nmr_peak_count: int,
                 c_nmr_shift_mean: float,
                 c_nmr_shift_std: float,
                 c_nmr_shift_max: float,
                 c_nmr_peak_count: int):

        self.contains_nitrogen = contains_nitrogen
        self.contains_oxygen = contains_oxygen
        self.contains_halogen = contains_halogen
        self.contains_sulfur = contains_sulfur

        self.ir_band_ohnh_stretch_3200_3550 = ir_band_ohnh_stretch_3200_3550
        self.ir_band_ch_stretch_2850_3000 = ir_band_ch_stretch_2850_3000
        self.ir_band_carbonyl_1650_1750 = ir_band_carbonyl_1650_1750
        self.ir_band_aromatic_1450_1600 = ir_band_aromatic_1450_1600
        self.ir_band_fingerprint_500_1500 = ir_band_fingerprint_500_1500

        self.h_nmr_shift_mean = h_nmr_shift_mean
        self.h_nmr_shift_std = h_nmr_shift_std
        self.h_nmr_shift_max = h_nmr_shift_max
        self.h_nmr_peak_count = h_nmr_peak_count

        self.c_nmr_shift_mean = c_nmr_shift_mean
        self.c_nmr_shift_std = c_nmr_shift_std
        self.c_nmr_shift_max = c_nmr_shift_max
        self.c_nmr_peak_count = c_nmr_peak_count

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "contains_nitrogen": [self.contains_nitrogen],
                "contains_oxygen": [self.contains_oxygen],
                "contains_halogen": [self.contains_halogen],
                "contains_sulfur": [self.contains_sulfur],
                "ir_band_ohnh_stretch_3200_3550": [self.ir_band_ohnh_stretch_3200_3550],
                "ir_band_ch_stretch_2850_3000": [self.ir_band_ch_stretch_2850_3000],
                "ir_band_carbonyl_1650_1750": [self.ir_band_carbonyl_1650_1750],
                "ir_band_aromatic_1450_1600": [self.ir_band_aromatic_1450_1600],
                "ir_band_fingerprint_500_1500": [self.ir_band_fingerprint_500_1500],
                "h_nmr_shift_mean": [self.h_nmr_shift_mean],
                "h_nmr_shift_std": [self.h_nmr_shift_std],
                "h_nmr_shift_max": [self.h_nmr_shift_max],
                "h_nmr_peak_count": [self.h_nmr_peak_count],
                "c_nmr_shift_mean": [self.c_nmr_shift_mean],
                "c_nmr_shift_std": [self.c_nmr_shift_std],
                "c_nmr_shift_max": [self.c_nmr_shift_max],
                "c_nmr_peak_count": [self.c_nmr_peak_count],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
