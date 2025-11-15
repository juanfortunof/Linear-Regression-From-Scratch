import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from typing import Tuple
import warnings

class LinearRegressor:

    def __init__(self, w=None, b=None, pred=None):
        self.__w = w
        self.__b = b
        self.__pred = pred

    @staticmethod
    def __convert_training_data(X: pd.DataFrame | pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:

        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X = X.values

        y = y.values.reshape(-1, 1)
        
        return X, y

    @staticmethod
    def __convert_test_data(X: pd.DataFrame | pd.Series) -> np.ndarray:

        ''' This is a private method '''

        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        elif isinstance(X, pd.DataFrame):
            X = X.values

        return X

    @staticmethod
    def __get_residuals(pred: np.ndarray, y: np.ndarray) -> np.ndarray:

        ''' This is a private method '''

        residuals = pred - y
        return residuals

    @staticmethod
    def __check_corr(X: pd.DataFrame | pd.Series, y: pd.Series) -> None:

        df = pd.concat([X, y], axis=1)
        corrs = df.corr().iloc[:, -1]

        for col, corr in zip(corrs.index, corrs):
            if abs(corr) < 0.3:
                print(f'WARNING! {col} has a low correlation with the target variable ({np.round(corr, 2)}).')


    @staticmethod
    def __check_Ramsey(p_value: float) -> None:

        ''' This is a private method '''

        if p_value < 0.05:
            print('WARNING! The Ramsey Test shows that the model could have non-linear relations, ' +
                  'try using a polynomic or logarithmic transformation')
        else:
            print('Ramsey Test passed, The model meets the assumptions of linearity.')

    @staticmethod
    def __check_dw(dw_test: float) -> None:

        ''' This is a private method '''

        if dw_test >= 1.8 and dw_test <= 2.2:
            print('Durbin Watson test passed, the model meets the independence assumptions')
        elif dw_test < 1.8:
            print('WARNING! Durbin Watson test shows that the model has positive correlation ' +
                  'problems, try adding lags of one of the dependent variables as another dependent variable.')
        else:
            print('Durbin Watson test shows that the model has negative correlation problems, ' +
                  'try adding lags of one of the dependent variables as another dependent variable.')

    @staticmethod
    def __check_ht(p_value:float) -> None:

        ''' This is a private method '''

        if p_value < 0.05:
            print('WARNING! The Homocedaskicity test shows that your model has ' +
                   'heterocedaskicity problems, try transforming the dependent and independent variables.')
        else:
            print('The Breusch Pagan test was passed, there is no heterocedaskicity problems')

    @staticmethod
    def __check_jb(p_value: float) -> None:

        ''' This is a private method '''

        if p_value < 0.5:
            print('WARNING! The data does not has a normal ' +
                  'distribution try normalizing, square root, ' +
                  'logarithmic or inverse transformations.')
        else:
            print('Jarque Bera passed, the data is normally distributed')

    @staticmethod
    def __check_multicol(X: pd.DataFrame) -> None:

        ''' This is a private method '''

        vif_data = pd.DataFrame()
        vif_data['feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) \
                               for i in range(X.shape[1])]

        for idx, row in vif_data.iterrows():
            if row['VIF'] >= 5:
                print(f'WARNING! Check the correlation between {row['feature']} and the other independent variables. VIF: {np.round(row['VIF'], 2)}')
                

    @staticmethod
    def __check_assumptions(X: np.ndarray, y: np.ndarray) -> None:

        ''' This is a private method '''

        X_c = sm.add_constant(X)
        model = sm.OLS(y, X_c).fit()
        residuals = model.resid
        dw_test = durbin_watson(residuals)
        jb = jarque_bera(residuals)
        reset_test = smd.linear_reset(model,
                                      use_f=True,
                                      test_type='fitted')

        ht_test = smd.het_breuschpagan(residuals, model.model.exog)
        p_value_ramsey = reset_test.pvalue
        LinearRegressor.__check_Ramsey(p_value_ramsey)
        LinearRegressor.__check_jb(jb[1])
        LinearRegressor.__check_dw(dw_test)
        LinearRegressor.__check_ht(ht_test[1])

    @staticmethod
    def __get_regplot(X: np.ndarray, pred: np.ndarray, y: np.ndarray) -> None:

        ''' This is a private method '''

        plt.scatter(x=y, y=X, alpha=0.3)
        plt.plot(pred, X, c='r')
        plt.title('Regresion plot')
        plt.xlabel('Input')
        plt.ylabel('Actual Values')
        plt.show()

    @staticmethod
    def __get_residual_plot(pred: np.ndarray, residuals: np.ndarray) -> None:

        ''' This is a private method '''

        plt.scatter(pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()

    def _get_weights(self, input: np.ndarray, y: np.ndarray) -> np.ndarray:

        ''' This is a private method '''

        b = np.ones(shape=(input.shape[0], 1))
        input_b = np.hstack((input, b))

        weights = (np.linalg.inv(input_b.T @ input_b) @ input_b.T) @ y

        return weights

    def fit(self, X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> None:

        '''
            This method acts like training the model, it will also tell you if your 
            model has heterocedaskicity, autocorrelation, multicolinearity, is it 
            bad specified and if the data is normally distributed.
        '''

        LinearRegressor.__check_corr(X_train, y_train)
        X_train_r, y_train_r = LinearRegressor.__convert_training_data(X_train, y_train)
        print()
        LinearRegressor.__check_assumptions(X_train_r, y_train_r)

        if len(X_train.shape) > 1:
            LinearRegressor.__check_multicol(X_train)

        w_b = self._get_weights(X_train_r, y_train_r)
        self.__w = w_b[:-1]
        self.__b = w_b[-1]

    def predict(self, X: pd.DataFrame | pd.Series) -> np.ndarray:

        ''' 
            This method predicts the test set using the weights and the Bias. 
        '''

        X = LinearRegressor.__convert_test_data(X)
        
        self.__pred = X @ self.__w + self.__b
        return self.__pred

    
    def get_MSE(self, pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the MSE '''

        m = pred.shape[0]

        MSE = (1/m) * np.sum((pred - y) ** 2)
        return MSE
    

    def get_RMSE(self, pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the RMSE '''

        m = pred.shape[0]

        RMSE = ((1/m) * np.sum((pred - y) ** 2)) ** 0.5
        return RMSE


    def get_MAE(self, pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the MAE '''

        m = pred.shape[0]

        MAE = 1/m * (np.sum(abs(pred - y)))
        return MAE
    
    
    def get_R2(self, pred: np.ndarray, y: np.ndarray) -> float:

        ''' This method calculates the R2 '''

        num = np.sum((pred - y.mean()) ** 2)
        den = np.sum((y - y.mean()) ** 2)
        R2 = 1 - (num / den)

        return R2


    def get_metrics_report(self, 
                           X: pd.DataFrame | pd.Series,  
                           pred: np.ndarray,
                           y: np.ndarray,
                           charts=True) -> None:

        ''' 
            This method will give you a quick report of you regression, showing you metrics
            like RMSE, MSE, MAE, R2, a regression plot and a residual plot.
        '''

        X_res, y_res = LinearRegressor.__convert_training_data(X, y)

        RMSE = self.get_RMSE(pred, y_res)
        MSE = self.get_MSE(pred, y_res)
        MAE = self.get_MAE(pred, y_res)
        R2 = self.get_R2(pred, y_res)
        residuals = LinearRegressor.__get_residuals(pred, y_res)

        print()
        print('------------------METRICS REPORT-----------------\n')
        print(f'MSE: {MSE}\nRMSE: {RMSE}\nMAE: {MAE}\nR2: {R2}\n')

        if charts:

            if X_res.shape[1] == 1:
                LinearRegressor.__get_regplot(X_res, pred, y_res)
                print()

            LinearRegressor.__get_residual_plot(pred, residuals)


class RidgeRegressor(LinearRegressor):

    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def _get_weights(self, input: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        b = np.ones(shape=(input.shape[0], 1))
        input_b = np.hstack((input, b))

        weights = (np.linalg.inv((input_b.T @ input_b) + self.alpha * np.eye(input_b.shape[1])) @ input_b.T) @ y

        return weights
