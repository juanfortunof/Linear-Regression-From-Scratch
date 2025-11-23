import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from typing import Tuple
import warnings


class Validator:
    """Provides static methods to validate input data for regression models."""

    @staticmethod
    def _check_type(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        """Checks if X and y are pandas Series or DataFrames and have matching observation counts."""
        if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
            raise TypeError('The target feature should be a pandas Series or DataFrame.')
        if not isinstance(X, pd.Series) and not isinstance(X, pd.DataFrame):
            raise TypeError('X should be a pandas Series or DataFrame.')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y should have the same amount of observations.')

    @staticmethod
    def _check_values(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        """Checks for NaN or infinite values in X and y."""
        if X.isna().any().any() or np.isinf(X).any().any():
            raise ValueError('Check the values of your data, there could be NaNs, inf or empty values.')
        if y.isna().any().any() or np.isinf(y).any().any():
            raise ValueError('Check the values of your data, there could be NaNs, inf or empty values.')

    @staticmethod
    def _check_sizes(X: pd.DataFrame | pd.Series):
        """Validates that the dataset is not empty and has more rows than columns."""
        if X.values.shape == (0, 0):
            raise ValueError("Your Data can't be empty")
        if len(X.shape) == 1:
            return
        cols = X.shape[1]
        rows = X.shape[0]
        if cols > rows:
            raise ValueError('Your dataset is too small, add more rows or more observations.')

    @staticmethod
    def _check_features(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        """Ensures all features in X are numeric."""
        not_admited_types = ['object', 'datetime64', 'bool']
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        if len(set(X.select_dtypes(include=not_admited_types).columns)) > 0:
            raise TypeError('Features must be numeric.')

    @staticmethod
    def _check_colinearity(X: pd.DataFrame):
        """Detects strong multicollinearity in the feature matrix X."""
        corr = X.corr()
        for val in corr.values.flatten():
            if abs(val) > 0.8 and abs(val) < 1:
                warnings.warn("There are clear signs of colinearity on your data, try using features that ain't correlated between them.")
        XTX = X.T @ X
        det = np.linalg.det(XTX)
        if det == 0:
            raise ValueError('Your feature matrix is singular, which means there are severe colinearity problems.')

    @staticmethod
    def _validate(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
        """Performs a comprehensive set of validations on X and y."""
        Validator._check_type(X, y)
        Validator._check_values(X, y)
        Validator._check_sizes(X)
        Validator._check_features(X, y)
        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            Validator._check_colinearity(X)


class Preprocessor:
    """Handles data preprocessing steps for model input."""

    @staticmethod
    def _convert_data(X: pd.DataFrame | pd.Series, y: pd.Series, train=True) -> Tuple[np.ndarray, np.ndarray]:
        """Converts pandas Series/DataFrames to NumPy arrays for model compatibility."""
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X = X.values
        if train:
            if isinstance(y, pd.Series):
                y = y.values.reshape(-1, 1)
            elif isinstance(y, pd.DataFrame):
                y = y.values
            return X, y
        return X


class CorrChecker:
    """Checks the correlation between features and the target variable."""

    @staticmethod
    def _check_corr(X: pd.Series | pd.DataFrame, y: pd.Series) -> None:
        """Warns if feature-target correlations are below a certain threshold (0.3)."""
        df = pd.concat([X, y], axis=1)
        corrs = df.corr().iloc[:, -1]
        for col, corr in zip(corrs.index, corrs):
            corr = np.round(corr, 2)
            if abs(corr) < 0.3:
                warnings.warn('WARNING. Check the correlation between your features and the target variable.')


class LinearRegressor:
    """Implements a Linear Regression model with diagnostic checks."""

    def __init__(self, TimeSeries=False):
        """Initializes the regressor. Weights (w) and bias (b) are set during fitting."""
        self.w = None
        self.b = None
        self.TimeSeries = TimeSeries

    @staticmethod
    def _get_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates optimal weights and bias using the normal equation."""
        b = np.ones(shape=(X.shape[0], 1))
        input_b = np.hstack((X, b))
        XTX = input_b.T @ input_b
        XTy = input_b.T @ y
        try:
            weights = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix is singular. Your data still has multicollinearity issues.") from None
        return weights

    def fit(self, X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> None:
        """Trains the model and performs diagnostic assumption checks."""
        if X_train.shape[0] > 100000:
            print('Calculating advanced diagnostics. This could take long, we suggest you use a GPU or TPU.\n')
        Validator._validate(X_train, y_train)
        CorrChecker._check_corr(X_train, y_train)
        X_train_r, y_train_r = Preprocessor._convert_data(X_train, y_train)
        w_b = self._get_weights(X_train_r, y_train_r)
        self.w = w_b[:-1]
        self.b = w_b[-1]
        print()
        AssumpChecker._check_assumptions(X_train_r, y_train_r, self.w, self.b, self.TimeSeries)
        if X_train_r.shape[1] > 1 and len(X_train_r.shape) > 1:
            AssumpChecker._check_multicol(X_train)

    def predict(self, X: pd.DataFrame | pd.Series) -> np.ndarray:
        """Generates predictions for new data using learned weights and bias."""
        Validator._check_sizes(X)
        X = Preprocessor._convert_data(X, None, train=False)
        pred = X @ self.w + self.b
        return pred

    def get_metrics_report(self,
                           X: pd.DataFrame | pd.Series,
                           pred: np.ndarray,
                           y: np.ndarray,
                           charts=True) -> None:
        """Generates and prints a comprehensive metrics report (MSE, RMSE, MAE, R2) and plots."""
        # Initial validation checks
        Validator._check_type(X, y)
        Validator._check_features(X, y)
        Validator._check_values(X, y)

        # Convert data to numpy arrays
        X_res, y_res = Preprocessor._convert_data(X, y)

        # Ensure pred is also a NumPy array and has the correct shape for consistency with y_res
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        if pred.shape != y_res.shape:
             # Try to reshape pred to match y_res if it's a 1D array
            if len(pred.shape) == 1 and pred.shape[0] == y_res.shape[0]:
                pred = pred.reshape(-1, 1)
            else:
                # This case indicates a more fundamental problem with prediction output shape
                raise ValueError(f"Shape of predictions {pred.shape} does not match actual values {y_res.shape}")

        # --- RAM optimization for very large datasets ---
        # If the number of samples is extremely large, sample down for performance/memory
        MAX_SAMPLES_FOR_METRICS = 1_000_000 # Define a threshold for large datasets (e.g., 1 million rows)
        if X_res.shape[0] > MAX_SAMPLES_FOR_METRICS:
            print(f"Warning: Dataset size ({X_res.shape[0]} rows) exceeds {MAX_SAMPLES_FOR_METRICS} samples. "
                  f"Sampling down to {MAX_SAMPLES_FOR_METRICS} for metrics and charts to prevent RAM issues. "
                  f"Metrics will be approximate.")
            # Get random indices
            indices = np.random.choice(X_res.shape[0], MAX_SAMPLES_FOR_METRICS, replace=False)
            # Apply sampling
            X_res = X_res[indices]
            y_res = y_res[indices]
            pred = pred[indices]
        # --- End RAM optimization ---


        # Calculate metrics (removed time.sleep calls)
        RMSE = Metrics.get_RMSE(pred, y_res)
        MSE = Metrics.get_MSE(pred, y_res)
        MAE = Metrics.get_MAE(pred, y_res)
        R2 = Metrics.get_R2(pred, y_res)
        residuals = y_res - pred

        print('------------------METRICS REPORT-----------------\n')
        print(f'MSE: {np.round(MSE, 3)}\nRMSE: {np.round(RMSE, 3)}\nMAE: {np.round(MAE, 2)}\nR2: {np.round(R2, 3)}\n')

        # This existing sampling for charts is now applied to potentially already sampled data
        if charts:
            current_X_res = X_res
            current_y_res = y_res
            current_pred = pred
            current_residuals = residuals

            # Further sample for chart rendering if the data is still too large for good visualization
            # and it hasn't already been sampled down to an appropriate charting size by the MAX_SAMPLES_FOR_METRICS block.
            # This ensures that charts are always generated with a reasonable number of points (max 10,000)
            if current_X_res.shape[0] > 10000:
                print(f"Warning: Dataset for charts still large ({current_X_res.shape[0]} rows). "
                      f"Sampling down to 10,000 for chart rendering to improve performance and readability.")
                ind = np.random.choice(current_X_res.shape[0], 10000, replace=False)

                current_X_res = current_X_res[ind]
                current_y_res = current_y_res[ind]
                current_pred = current_pred[ind]
                current_residuals = current_residuals[ind]

            if current_X_res.shape[1] == 1:
                Report._get_regplot(current_X_res, current_pred, current_y_res)
                print()

            Report._get_residual_plot(current_pred, current_residuals)


class AssumpChecker:
    """Provides static methods to check linear regression assumptions."""

    @staticmethod
    def _check_Ramsey(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
        """Performs the Ramsey RESET test for model specification (non-linearity)."""
        w1, b1 = w, b
        pred1 = X @ w1 + b1
        pred1_sq = pred1 ** 2
        pred1_cu = pred1 ** 3
        ssrr = np.sum((y - pred1)**2, axis=0)

        X2 = np.hstack([X, pred1_sq, pred1_cu])
        all_w_R = LinearRegressor._get_weights(X2, y)
        w2, b2 = all_w_R[0:-1], all_w_R[-1]
        pred2 = X2 @ w2 + b2
        ssra = np.sum((y - pred2)**2, axis=0)

        q = 2
        n = X.shape[0]
        k = X.shape[1]

        F = ((ssrr-ssra) / q) / (ssra / (n - k - q - 1))
        F = F[0]
        F_critic = scipy.stats.f.ppf(1-0.05, k, n-k-q-1)

        if F > F_critic:
            warnings.warn('Ramsey Test failed, your model has non-linear relations, try using polynomial or logarithmic convertions.')

    @staticmethod
    def _check_dw(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
        """Performs the Durbin-Watson test for autocorrelation (Time Series only)."""
        pred = X @ w + b
        res = y - pred
        d = np.sum((res[:-1] - res[1:])**2) / np.sum(res **2)

        if d < 1.8:
            warnings.warn('Durbin Watson test shows that the model has positive correlation problems, try adding lags of one of the dependent variables as another dependent variable.')
        elif d > 2.2:
            warnings.warn('Durbin Watson test shows that the model has negative correlation problems, try adding lags of one of the dependent variables as another dependent variable.')

    @staticmethod
    def _check_ht(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
        """Performs the Breusch-Pagan test for heteroscedasticity."""
        pred = X @ w + b
        res = y - pred
        res_sq = res**2
        n = X.shape[0]
        k = X.shape[1]
        ssr = np.sum(res_sq)

        var_res = ssr / (n-k-1)
        g = res_sq / var_res

        all_w2 = LinearRegressor._get_weights(X, g)
        w2, b2 = all_w2[:-1], all_w2[-1]
        pred2 = X @ w2 + b2

        R2 = Metrics.get_R2(pred2, g)
        LM = n * R2
        chi_square_value = scipy.stats.chi2.ppf(1-0.05, k)

        if LM > chi_square_value:
            warnings.warn('The BP test shows that your model has heterocedaskicity problems, try transforming the dependent and independent variables.')

    @staticmethod
    def _check_jb(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
        """Performs the Jarque-Bera test for normality of residuals."""
        pred = X @ w + b
        res = y - pred
        n = X.shape[0]
        res_mean = np.mean(res)

        M2 = 1/n * np.sum((res-res_mean)**2)
        M3 = 1/n * np.sum((res-res_mean)**3)
        M4 = 1/n * np.sum((res-res_mean)**4)
        std_dev = np.sqrt(M2)

        S = M3 / std_dev**3
        K = M4 / std_dev**4

        JB = n/6 * ((S**2) + ((K - 3)**2 / 4))

        JB_critic = scipy.stats.chi2.ppf(1-0.05, 2)

        if JB > JB_critic:
            warnings.warn('Jarque Bera test failed. The residuals are not normally distributed.')

    @staticmethod
    def _check_multicol(X: pd.DataFrame) -> None:
        """Calculates VIF to detect multicollinearity among features."""
        corr = X.corr()
        try:
            inv_corr_matrix = np.linalg.inv(corr.values)
        except np.linalg.LinAlgError as E:
            raise ValueError('Your data has a severe multicolinearity problem, there is no way to find a solution to it.')
        VIF = np.diag(inv_corr_matrix)
        if np.any(VIF > 5):
            warnings.warn('Multicolinearity test failed, This is a serious issue, check the correlation between your features, they should not be correlated.')

    @staticmethod
    def _check_assumptions(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, TimeSeries=False) -> None:
        """Runs a suite of assumption checks (Ramsey, Jarque-Bera, Breusch-Pagan, Durbin-Watson)."""
        AssumpChecker._check_Ramsey(X, y, w, b)
        AssumpChecker._check_jb(X, y, w, b)
        AssumpChecker._check_ht(X, y, w, b)
        if TimeSeries:
            AssumpChecker._check_dw(X, y, w, b)


class Metrics:
    """Provides static methods to calculate common regression metrics."""

    @staticmethod
    def get_MSE(pred: np.ndarray, y: np.ndarray) -> float:
        """Calculates the Mean Squared Error (MSE)."""
        m = pred.shape[0]
        MSE = (1/m) * np.sum((y - pred) ** 2)
        return MSE

    @staticmethod
    def get_RMSE(pred: np.ndarray, y: np.ndarray) -> float:
        """Calculates the Root Mean Squared Error (RMSE)."""
        m = pred.shape[0]
        RMSE = ((1/m) * np.sum((y - pred) ** 2)) ** 0.5
        return RMSE

    @staticmethod
    def get_MAE(pred: np.ndarray, y: np.ndarray) -> float:
        """Calculates the Mean Absolute Error (MAE)."""
        m = pred.shape[0]
        MAE = 1/m * (np.sum(abs(y - pred)))
        return MAE

    @staticmethod
    def get_R2(pred: np.ndarray, y: np.ndarray) -> float:
        """Calculates the R-squared (Coefficient of Determination)."""
        num = np.sum((y - pred) ** 2)
        den = np.sum((y - y.mean()) ** 2)
        R2 = 1 - (num / den)
        return R2


class Report:
    """Provides static methods for generating regression plots."""

    @staticmethod
    def _get_regplot(X: np.ndarray, pred: np.ndarray, y: np.ndarray) -> None:
        """Generates a regression plot comparing actual vs. predicted values."""
        plt.scatter(y, X, alpha=0.3)
        plt.plot(pred, X, c='r')
        plt.title('Regresion plot')
        plt.xlabel('Input')
        plt.ylabel('Actual Values')
        plt.show()

    @staticmethod
    def _get_residual_plot(pred: np.ndarray, residuals: np.ndarray) -> None:
        """Generates a residual plot to check for patterns in errors."""
        plt.scatter(pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()