import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from LinReg_Prod import (
    Validator, Preprocessor, CorrChecker,
    LinearRegressor, Metrics, Report
)


# -------------------------------
# Input type and structure tests
# -------------------------------

def test_validator_accepts_series_X():
    X = pd.Series([1,2,3])
    y = pd.Series([2,4,6])
    # should not raise
    Validator._check_type(X, y)


def test_validator_rejects_empty_dataframe():
    X = pd.DataFrame()
    y = pd.Series([1,2,3])

    with pytest.raises(ValueError):
        Validator._check_sizes(X)


# -------------------------------
# Value and content tests
# -------------------------------


def test_validator_rejects_mixed_types():
    X = pd.DataFrame({"a":[1,2,3], "b":[1.1,2.2,"x"]})
    y = pd.Series([1,2,3])

    with pytest.raises(TypeError):
        Validator._check_features(X, y)


# -------------------------------
# Size related tests
# -------------------------------

def test_validator_rejects_high_dimensional_data():
    # 2 rows, 5 columns → should fail
    X = pd.DataFrame(np.random.randn(2, 5))
    with pytest.raises(ValueError):
        Validator._check_sizes(X)


# -------------------------------
# Duplicate-related tests
# -------------------------------

def test_validator_detects_duplicate_nonadjacent_columns():
    X = pd.DataFrame({
        "a": [1,2,3],
        "b": [4,5,6],
        "c": [1,2,3]   # duplicate of column a
    })

    with pytest.raises(ValueError):
        Validator._check_duplicates(X)


# -------------------------------
# Colinearity tests
# -------------------------------

def test_validator_warns_about_correlation(recwarn):
    X = pd.DataFrame({
        "x1": [1,2,3,4,5],
        "x2": [2,4,6,8,10] # corr = 1
    })

    with pytest.raises(ValueError):
        Validator._check_colinearity(X)


# ------------------------------
#  Preprocessor Tests
# ------------------------------

def test_preprocessor_dataframe_multiple_columns():
    X = pd.DataFrame({"a":[1,2,3], "b":[4,5,6]})
    y = pd.Series([1,2,3])
    Xr, yr = Preprocessor._convert_data(X, y)
    assert isinstance(Xr, np.ndarray)
    assert Xr.shape == (3, 2)


def test_preprocessor_predict_mode_keeps_shape():
    X = pd.DataFrame({"a":[1,2]})
    out = Preprocessor._convert_data(X, np.array([1]), train=False)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,1)

# ------------------------------
#  Correlation Tests
# ------------------------------

def test_corrchecker_detects_no_low_corr(recwarn, capsys):
    # Everything is strongly correlated → NO warnings
    X = pd.DataFrame({"a":[1,2,3,4], "b":[2,4,6,8]})
    y = pd.Series([3,6,9,12])

    CorrChecker._check_corr(X, y)
    assert len(recwarn) == 0


def test_corrchecker_works_with_series_X(recwarn):
    X = pd.Series([10, 11, 10, 11, 10])
    y = pd.Series([1, 2, 3, 4, 5])
    CorrChecker._check_corr(X, y)
    assert len(recwarn) >= 1

# -------------------------------
# Fit + Predict pipeline
# -------------------------------

def test_regressor_fit_predict_identity():
    X = pd.DataFrame({"x": np.arange(1, 11)})
    y = pd.Series(3 * X["x"] + 5)

    model = LinearRegressor()
    model.fit(X, y)

    preds = model.predict(X)
    assert np.allclose(preds.flatten(), y.values, atol=1e-6)


def test_regressor_fails_on_singular_matrix():
    X = pd.DataFrame({"x1":[1,2,3], "x2":[1,2,3]})
    y = pd.Series([2,4,6])

    model = LinearRegressor()

    with pytest.raises(ValueError):
        model.fit(X, y)

# -------------------------------
# Assumption checking tests
# -------------------------------

def test_regressor_detects_heteroscedasticity(recwarn):
    X = pd.DataFrame({"x": np.arange(1, 101)})
    y = pd.Series(X["x"] + np.random.randn(100) * X["x"])  # heteroscedastic

    model = LinearRegressor()
    model.fit(X, y)

    # Should trigger BP warning
    messages = [str(w.message) for w in recwarn]
    assert any("heterocedaskicity" in m for m in messages)


def test_regressor_detects_non_normal_residuals(recwarn):
    X = pd.DataFrame({"x": np.arange(1, 101)})
    y = pd.Series(np.random.exponential(size=100))  # skewed

    model = LinearRegressor()
    model.fit(X, y)

    messages = [str(w.message) for w in recwarn]
    assert any("Jarque Bera" in m for m in messages)

# -------------------------------
# Assumption checking tests
# -------------------------------

def test_metrics_mae_zero():
    pred = np.array([[1],[2],[3]])
    y = np.array([[1],[2],[3]])
    assert Metrics.get_MAE(pred, y) == 0


def test_metrics_r2_negative():
    pred = np.array([[100],[100],[100]])
    y = np.array([[1],[2],[3]])

    R2 = Metrics.get_R2(pred, y)
    assert R2 < 0  # A model can have negative R2

# -------------------------------
# Report tests
# -------------------------------

def test_regplot_runs_without_error():
    X = np.array([[1],[2],[3]])
    y = np.array([[2],[4],[6]])
    pred = y.copy()

    with patch("matplotlib.pyplot.show"):
        Report._get_regplot(X, pred, y)


def test_residual_plot_runs():
    pred = np.array([[1],[2],[3]])
    residuals = np.array([[0.1],[-0.2],[0.3]])

    with patch("matplotlib.pyplot.show"):
        Report._get_residual_plot(pred, residuals)