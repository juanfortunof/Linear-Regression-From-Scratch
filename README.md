 Linear Regression from Scratch

**[English]** A robust, custom implementation of Linear Regression in Python. Unlike standard libraries, this project builds the mathematical logic from the ground up using the **Normal Equation** and includes a comprehensive suite of automatic statistical diagnostics to validate regression assumptions.

## 🇺🇸 English Documentation

### Key Features
* **Pure Math Implementation:** Calculates weights ($w$) and bias ($b$) using the Normal Equation method $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ rather than "black box" solvers.
* **Automatic Data Validation:** The `Validator` class ensures input integrity, checking for missing values (NaNs), infinite values, data types, and dimensional consistency before training.
* **Statistical Assumption Checks:** The model automatically runs diagnostic tests after fitting to warn you of potential statistical violations:
    * **Linearity:** Ramsey RESET test.
    * **Normality of Residuals:** Jarque-Bera test.
    * **Homoscedasticity:** Breusch-Pagan test.
    * **Multicollinearity:** VIF and Correlation checks.
    * **Autocorrelation:** Durbin-Watson test (optional for Time Series).
* **Detailed Reporting:** Generates a full metrics report (MSE, RMSE, MAE, $R^2$) and visualization plots (Regression Line and Residuals).
* **Memory Safe:** Includes logic to downsample extremely large datasets for plotting and metrics to prevent RAM overflow.

### Requirements
Ensure you have the following libraries installed:

pip install numpy pandas matplotlib seaborn scipy
Usage ExampleThe API is designed to be intuitive, similar to Scikit-Learn.Pythonimport pandas as pd
from linear_regressor import LinearRegressor # Assuming your file is named linear_regressor.py

# 1. Load your data
df = pd.read_csv('your_data.csv')
X = df[['feature1', 'feature2']]
y = df['target']

# 2. Initialize the model
# Set TimeSeries=True if your data is time-dependent (enables Durbin-Watson test)
model = LinearRegressor(TimeSeries=False)

# 3. Train the model
# This will print warnings if statistical assumptions (like normality) are violated
model.fit(X, y)

# 4. Make predictions
predictions = model.predict(X)

# 5. Generate a performance report
# Calculates metrics and displays regression/residual plots
model.get_metrics_report(X, predictions, y.values, charts=True)
🇪🇸 Documentación en EspañolCaracterísticas PrincipalesImplementación Matemática Pura: Calcula los pesos ($w$) y el sesgo ($b$) usando la Ecuación Normal $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ en lugar de solucionadores de caja negra.Validación Automática de Datos: La clase Validator asegura la integridad de los datos, verificando valores faltantes (NaNs), infinitos, tipos de datos y consistencia dimensional antes del entrenamiento.Verificación de Supuestos Estadísticos: El modelo ejecuta pruebas de diagnóstico automáticamente después del entrenamiento para advertir sobre posibles violaciones estadísticas:Linealidad: Test de Ramsey RESET.Normalidad de los Residuos: Test de Jarque-Bera.Homocedasticidad: Test de Breusch-Pagan.Multicolinealidad: Verificaciones de VIF y Correlación.Autocorrelación: Test de Durbin-Watson (opcional para Series Temporales).Reportes Detallados: Genera un reporte completo de métricas (MSE, RMSE, MAE, $R^2$) y gráficos de visualización (Línea de Regresión y Residuos).Optimización de Memoria: Incluye lógica para reducir la muestra (downsampling) en datasets extremadamente grandes para los gráficos y métricas, evitando el desbordamiento de RAM.RequisitosAsegúrate de tener instaladas las siguientes librerías:Bashpip install numpy pandas matplotlib seaborn scipy
Ejemplo de UsoLa API está diseñada para ser intuitiva, similar a Scikit-Learn.Pythonimport pandas as pd
from linear_regressor import LinearRegressor # Asumiendo que tu archivo se llama linear_regressor.py

-------------

Regresión Lineal desde Cero

**[Español]** Una implementación robusta y personalizada de Regresión Lineal en Python. A diferencia de las librerías estándar, este proyecto construye la lógica matemática desde cero utilizando la **Ecuación Normal** e incluye un conjunto completo de diagnósticos estadísticos automáticos.

### Componentes Clave
* **Implementación Matemática Pura:** Calcula coeficientes ($w$) e intercepto ($b$) usando el método de la ecuación Normal $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ en vez de los "black box" solvers.
* **Validación de Datos Automática:** La clase `Validator` se asegura de la integridad del input, chequea valores faltantes (NaNs), infinitos, tipos de datos, y consistencia dimensional antes de entrenar.
* **Chequeo de Tests Estadísticos:** El modelo corre test de diagnosticos automaticamente luego de entrenar para advertirte de posibles violaciones estadísticas:
    * **Linealidad:** Ramsey RESET test.
    * **Normalidad de Residuos:** Jarque-Bera test.
    * **Homocedasticidad:** Breusch-Pagan test.
    * **Multicolinealidad:** VIF y chequeos de correlación.
    * **Autocorrelación:** Test de Durbin-Watson (opcional para Seires de Tiempo).
* **Reportes detallados:** Genera un reporte con todas las métricas (MSE, RMSE, MAE, $R^2$) y gráficos (Línea de Regressión y Residuos).
* **Optimizado en Memoria:** Incluye lógica para disminuir el número de observaciones para gráficos y métricas para prevenir desbordamiento en el RAM.

  ### Requerimentos
Asegurate de tener las siguientes librerias instaladas:

pip install numpy pandas matplotlib seaborn scipy

# 1. Cargar tus datos
df = pd.read_csv('tu_data.csv')
X = df[['feature1', 'feature2']]
y = df['target']

# 2. Inicializar el modelo
# Usa TimeSeries=True si tus datos dependen del tiempo (activa el test Durbin-Watson)
model = LinearRegressor(TimeSeries=False)

# 3. Entrenar el modelo
# Esto imprimirá advertencias si se violan supuestos estadísticos (como la normalidad)
model.fit(X, y)

# 4. Hacer predicciones
predictions = model.predict(X)

# 5. Generar reporte de rendimiento
# Calcula métricas y muestra gráficos de regresión y residuos
model.get_metrics_report(X, predictions, y.values, charts=True)
