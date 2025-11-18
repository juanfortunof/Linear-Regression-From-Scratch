import pytest
import pandas as pd
import numpy as np
from LinReg_Prod import (
    Validator, Preprocessor, CorrChecker,
    LinearRegressor, Metrics
)

# ----------------------------------------------------------
# Validator Tests
# ----------------------------------------------------------

def test_validator_type_error_X():
    X = [1, 2, 3]
    y = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        Validator._check_type(X, y)


def test_validator_type_error_y():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = [1, 2, 3]
    with pytest.raises(TypeError):
        Validator._check_type(X, y)


def test_validator_shape_mismatch():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 2])
    with pytest.raises(ValueError):
        Validator._check_type(X, y)


def test_validator_nan_values():
    X = pd.DataFrame({"a": [1, np.nan, 3]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        Validator._check_values(X, y)


def test_validator_inf_values():
    X = pd.DataFrame({"a": [1, np.inf, 3]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        Validator._check_values(X, y)


def test_validator_check_sizes_too_many_features():
    X = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    with pytest.raises(ValueError):
        Validator._check_sizes(X)


def test_validator_feature_type_error():
    X = pd.DataFrame({"a":[1, 2, 3], "b": ["x", "y", "z"]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(TypeError):
        Validator._check_features(X, y)


def test_validator_duplicate_values():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    with pytest.raises(ValueError):
        Validator._check_duplicates(X)


def test_validator_colinearity():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    with pytest.raises(ValueError):
        Validator._check_colinearity(X)

# ----------------------------------------------------------
# Preprocessor tests
# ----------------------------------------------------------

def test_preprocessor_series_to_array():
    X = pd.Series([1, 2, 3])
    y = pd.Series([2, 4, 6])
    Xr, yr = Preprocessor._convert_data(X, y)
    assert isinstance(Xr, np.ndarray)
    assert isinstance(yr, np.ndarray)
    assert Xr.shape == (3, 1)
    assert yr.shape == (3, 1)

# ----------------------------------------------------------
# CorrChecker tests
# ----------------------------------------------------------

def test_corrchecker_low_corr(capsys):
    X = pd.DataFrame({"a":[1,1,1]})
    y = pd.Series([1,2,3])
    CorrChecker._check_corr(X, y)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out

# ----------------------------------------------------------
# LinearRegressor tests
# ----------------------------------------------------------

def test_linear_regressor_fit_and_predict():
    X = pd.DataFrame({"x":[1,2,3,4,5]})
    y = pd.Series([2,4,6,8,10])

    model = LinearRegressor()
    model.fit(X,y)

    preds = model.predict(pd.DataFrame({"x":[6,7]}))
    assert preds.shape == (2,1)
    assert np.allclose(preds.flatten(), [12,14], atol=1e-3)


def test_linear_regressor_singular_matrix():
    X = pd.DataFrame({"a":[1,2,3], "b":[1,2,3]})
    y = pd.Series([2,4,6])

    model = LinearRegressor()
    with pytest.raises(ValueError):
        model.fit(X,y)

# ----------------------------------------------------------
# Metrics tests
# ----------------------------------------------------------

def test_mse():
    pred = np.array([[2],[4],[6]])
    y    = np.array([[2],[4],[6]])
    assert Metrics.get_MSE(pred,y) == 0


def test_r2():
    pred = np.array([[1],[2],[3]])
    y    = np.array([[1],[2],[3]])
    assert Metrics.get_R2(pred,y) == 1.0
