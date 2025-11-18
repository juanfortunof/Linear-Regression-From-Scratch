import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from typing import Tuple
import warnings


class Validator:

    @staticmethod
    def _check_type(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):

        if not isinstance(y, pd.Series) and not isinstance(y, pd.DataFrame):
            raise TypeError('The target feature should be a pandas Series or DataFrame.')

        if not isinstance(X, pd.Series) and not isinstance(X, pd.DataFrame):
            raise TypeError('X should be a pandas Series or DataFrame.')

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y should have the same amount of observations.')

    @staticmethod
    def _check_values(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):

        complete_df = pd.concat([X, y], axis=1)

        for val in complete_df.values.flatten():
            if np.isnan(val) or np.isinf(val):
                raise ValueError('Check the values of your data, there could be NaNs, inf or empty values.')

    @staticmethod
    def _check_sizes(X: pd.DataFrame | pd.Series):

        if X.values.shape == (0, 0):
            raise ValueError("Your Data can't be empty")

        cols = X.shape[1]
        rows = X.shape[0]

        if cols > rows:
            raise ValueError('Your dataset is too small, add more rows or more observations.')

    @staticmethod
    def _check_features(X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):

        not_admited_types = ['object', 'datetime64', 'bool']
        merged = pd.concat([X, y], axis=1)
        dtypes = merged.dtypes

        for data_type in dtypes:
            if str(data_type) in not_admited_types:
                raise TypeError('One of your features has a different type than numeric.')

    @staticmethod
    def _check_duplicates(X: pd.DataFrame):

        is_duplicated = X.T.duplicated()

        for val in is_duplicated:
            if val:
                raise ValueError('It looks like there is one or more features duplicated on the dataset.')

    @staticmethod
    def _check_colinearity(X: pd.DataFrame):

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

        Validator._check_type(X, y)
        Validator._check_values(X, y)
        Validator._check_sizes(X)
        Validator._check_features(X, y)

        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            Validator._check_duplicates(X)
            Validator._check_colinearity(X)


class Preprocessor:

    @staticmethod
    def _convert_data(X: pd.DataFrame | pd.Series, y: pd.Series, train=True) -> Tuple[np.ndarray, np.ndarray]:

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

    @staticmethod
    def _check_corr(X: pd.Series | pd.DataFrame, y: pd.Series) -> None:

        df = pd.concat([X, y], axis=1)
        corrs = df.corr().iloc[:, -1]

        for col, corr in zip(corrs.index, corrs):
            corr = np.round(corr, 2)
            if abs(corr) < 0.3:
                warnings.warn('WARNING. Check the correlation between your features and the target variable.')


class LinearRegressor:

    def __init__(self, w=None, b=None, TimeSeries=False):
        self.w = w
        self.b = b
        self.TimeSeries = TimeSeries
    
    @staticmethod
    def _get_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:

        b = np.ones(shape=(X.shape[0], 1))
        input_b = np.hstack((X, b))

        try:
            weights = (np.linalg.inv(input_b.T @ input_b) @ input_b.T) @ y
        except np.linalg.LinAlgError as e:
            raise ValueError("Matrix is singular. Your data still has multicollinearity issues.") from None 

        return weights

    def fit(self, X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> None:

        '''
            This method acts like training the model, it will also tell you if your
            model has heterocedaskicity, autocorrelation, multicolinearity, is it
            bad specified and if the data is normally distributed.
        '''

        Validator._validate(X_train, y_train)
        CorrChecker._check_corr(X_train, y_train)
        X_train_r, y_train_r = Preprocessor._convert_data(X_train, y_train)
        w_b = self._get_weights(X_train_r, y_train_r)
        self.w = w_b[:-1]
        self.b = w_b[-1]
        print()
        
        AssumpChecker._check_assumptions(X_train_r, y_train_r, self.w, self.b, self.TimeSeries)

        if X_train.shape[1] > 1 and len(X_train.shape) > 1:
            AssumpChecker._check_multicol(X_train)

    def predict(self, X: pd.DataFrame | pd.Series) -> np.ndarray:

        '''
            This method predicts the test set using the weights and the Bias.
        '''
        X = Preprocessor._convert_data(X, None, train=False)

        pred = X @ self.w + self.b
        return pred

    def get_metrics_report(self,
                           X: pd.DataFrame | pd.Series,
                           pred: np.ndarray,
                           y: np.ndarray,
                           charts=True) -> None:

        '''
            This method will give you a quick report of you regression, showing you metrics
            like RMSE, MSE, MAE, R2, a regression plot and a residual plot.
        '''

        X_res, y_res = Preprocessor._convert_data(X, y)

        RMSE = Metrics.get_RMSE(pred, y_res)
        MSE = Metrics.get_MSE(pred, y_res)
        MAE = Metrics.get_MAE(pred, y_res)
        R2 = Metrics.get_R2(pred, y_res)
        residuals = y_res - pred

        print('------------------METRICS REPORT-----------------\n')
        print(f'MSE: {np.round(MSE, 3)}\nRMSE: {np.round(RMSE, 3)}\nMAE: {np.round(MAE, 2)}\nR2: {np.round(R2, 3)}\n')

        if charts:

            if X_res.shape[1] == 1:
                Report._get_regplot(X_res, pred, y_res)
                print()

            Report._get_residual_plot(pred, residuals)
    
    
class AssumpChecker:

    @staticmethod
    def _check_Ramsey(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:

        ''' 
            This method does all the Ramsey process from start to finish, it
            can be hard to understand, so I suggest supporting the explanations
            with AI.
        '''

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

        ''' This method just applies for Time Series data only, it does the Durbin Watson test. '''

        pred = X @ w + b
        res = y - pred
        d = np.sum((res[:-1] - res[1:])**2) / np.sum(res **2)

        if d < 1.8:
            warnings.warn('Durbin Watson test shows that the model has positive correlation problems, try adding lags of one of the dependent variables as another dependent variable.')
        elif d > 2.2:
            warnings.warn('Durbin Watson test shows that the model has negative correlation problems, try adding lags of one of the dependent variables as another dependent variable.')

    @staticmethod
    def _check_ht(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:

        ''' This test checks for Heterocedaskicity on your model.'''

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

        ''' This is a private method '''

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

        ''' This is a private method '''

        VIFs = {col: 0 for col in X.columns}

        for col in X.columns:
            X_aux = X.drop(col, axis=1)
            y = X[col]

            all_w = LinearRegressor._get_weights(X_aux.values, y.values)
            w, b = all_w[:-1], all_w[-1]
            pred = X_aux @ w + b
            R2_aux = Metrics.get_R2(pred, y)

            VIFs[col] = 1 / (1 - R2_aux)

        for vif in VIFs.values():

            if vif > 5:
                warnings.warn('Multicolinearity test failed, This is a serious issue, check the correlation between your features, they should not be correlated.')
                break

    @staticmethod
    def _check_assumptions(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, TimeSeries=False) -> None:

        ''' This is a private method '''

        AssumpChecker._check_Ramsey(X, y, w, b)
        AssumpChecker._check_jb(X, y, w, b)
        AssumpChecker._check_ht(X, y, w, b)

        if TimeSeries:
            AssumpChecker._check_dw(X, y, w, b)


class Metrics:

    @staticmethod
    def get_MSE(pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the MSE '''

        m = pred.shape[0]

        MSE = (1/m) * np.sum((y - pred) ** 2)
        return MSE

    @staticmethod
    def get_RMSE(pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the RMSE '''

        m = pred.shape[0]

        RMSE = ((1/m) * np.sum((y - pred) ** 2)) ** 0.5
        return RMSE

    @staticmethod
    def get_MAE(pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the MAE '''

        m = pred.shape[0]

        MAE = 1/m * (np.sum(abs(y - pred)))
        return MAE

    @staticmethod
    def get_R2(pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the R2 '''

        num = np.sum((y - pred) ** 2)
        den = np.sum((y - y.mean()) ** 2)
        R2 = 1 - (num / den)

        return R2


class Report:

    @staticmethod
    def _get_regplot(X: np.ndarray, pred: np.ndarray, y: np.ndarray) -> None:

        ''' This is a private method '''

        plt.scatter(y, X, alpha=0.3)
        plt.plot(pred, X, c='r')
        plt.title('Regresion plot')
        plt.xlabel('Input')
        plt.ylabel('Actual Values')
        plt.show()

    @staticmethod
    def _get_residual_plot(pred: np.ndarray, residuals: np.ndarray) -> None:

        ''' This is a private method '''

        plt.scatter(pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()