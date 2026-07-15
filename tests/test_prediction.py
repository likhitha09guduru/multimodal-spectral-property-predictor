import pytest

from src.pipeline.predict_pipeline import CustomData, PredictPipeline, PredictionInputError


def test_invalid_smiles_raises_before_touching_disk(tmp_path):
    # Points at an empty dir with no model artifacts. If SMILES validation
    # runs first (as it should), this raises PredictionInputError, never
    # FileNotFoundError -- proving bad input fails fast without needing a
    # trained model on disk.
    pipeline = PredictPipeline(model_dir=str(tmp_path))
    data = CustomData(smiles="not_a_real_smiles!!", ir_spectrum=[0.0] * 256)

    with pytest.raises(PredictionInputError):
        pipeline.predict(data)


def test_missing_artifacts_raises_file_not_found_for_valid_input(tmp_path):
    pipeline = PredictPipeline(model_dir=str(tmp_path))
    data = CustomData(smiles="CCO", ir_spectrum=[0.0] * 256)

    with pytest.raises(FileNotFoundError):
        pipeline.predict(data)


def test_wrong_length_spectrum_raises_after_model_loads(tmp_path):
    # Build a tiny real model + preprocessor into tmp_path so we can test the
    # spectrum-length check specifically, independent of SMILES validity.
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from src.components.model_training import CNNGNNRegressor
    from src.utils import save_model, save_json, save_object, num_atom_features

    expected_len = 32
    model = CNNGNNRegressor(
        node_feature_dim=num_atom_features(),
        gnn_hidden_dim=4,
        cnn_hidden_dim=4,
        fusion_hidden_dim=4,
    )
    save_model(str(tmp_path / "model.pt"), model)
    save_json(
        {
            "node_feature_dim": num_atom_features(),
            "gnn_hidden_dim": 4,
            "cnn_hidden_dim": 4,
            "fusion_hidden_dim": 4,
        },
        str(tmp_path / "model_config.json"),
    )
    scaler = StandardScaler().fit(np.random.rand(10, expected_len))
    save_object(str(tmp_path / "preprocessor.pkl"), scaler)

    pipeline = PredictPipeline(model_dir=str(tmp_path))
    data = CustomData(smiles="CCO", ir_spectrum=[0.0] * (expected_len - 1))  # wrong length

    with pytest.raises(PredictionInputError):
        pipeline.predict(data)


def test_correct_input_returns_a_prediction(tmp_path):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from src.components.model_training import CNNGNNRegressor
    from src.utils import save_model, save_json, save_object, num_atom_features

    expected_len = 32
    model = CNNGNNRegressor(
        node_feature_dim=num_atom_features(),
        gnn_hidden_dim=4,
        cnn_hidden_dim=4,
        fusion_hidden_dim=4,
    )
    save_model(str(tmp_path / "model.pt"), model)
    save_json(
        {
            "node_feature_dim": num_atom_features(),
            "gnn_hidden_dim": 4,
            "cnn_hidden_dim": 4,
            "fusion_hidden_dim": 4,
        },
        str(tmp_path / "model_config.json"),
    )
    scaler = StandardScaler().fit(np.random.rand(10, expected_len))
    save_object(str(tmp_path / "preprocessor.pkl"), scaler)

    pipeline = PredictPipeline(model_dir=str(tmp_path))
    data = CustomData(smiles="CCO", ir_spectrum=[0.1] * expected_len)

    result = pipeline.predict(data)
    assert result.shape == (1,)
