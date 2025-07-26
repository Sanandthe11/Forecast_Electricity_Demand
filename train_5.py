from arch.unitroot import ADF, KPSS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the .csv into a dataframe
df = pd.read_csv(r'C:\Users\SHEEJA\Downloads\Electricity_1.csv')
df['date']=pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)
df.index.freq = 'D'

plt.figure(figsize=(12, 6))
plt.plot_date(df.index, df['demand'],'o-')
plt.title('Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.xticks(df.index[::85])
plt.gcf().autofmt_xdate()
plt.show()

# Run ADF and KPSS test to find stationary or not
adf_result = ADF(df['demand'].dropna(), lags=1)
print(adf_result.summary())
kpss_result = KPSS(df['demand'].dropna(), lags=1)
print(kpss_result.summary())

# Split the data into train-test
split_point = int(len(df['demand']) * 0.8)
split_point_1 = int(len(df['demand']) * 0.2)
train = df['demand'].iloc[:split_point]
test = df['demand'].iloc[split_point:]

# ACF and PACF plots to find (pdq) [after diffrencing]
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(df['demand'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('ACF (Differenced Series)')
plot_pacf(df['demand'].dropna(), lags=40, ax=axes[1], method='ywm')
axes[1].set_title('PACF (Differenced Series)')
plt.tight_layout()
plt.show()

# Diffrence the data
df['demand_diff'] = df['demand'].diff()

# ACF and PACF plots to find (pdq) [after diffrencing]
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(df['demand_diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('ACF (Differenced Series)')
plot_pacf(df['demand_diff'].dropna(), lags=40, ax=axes[1], method='ywm')
axes[1].set_title('PACF (Differenced Series)')
plt.tight_layout()
plt.show()

# Creating bands of trend, seasonality and residuals
stl = STL(df['demand'], period=12, robust=True)
result = stl.fit()
trend = result.trend
seasonal = result.seasonal
resid = result.resid.dropna()

# Running auto-ARIMA
model = auto_arima(resid, seasonal=True, stepwise=True, suppress_warnings=True, trace=True)
n_periods = split_point_1  # number of steps ahead to forecast
forecast_resid = model.predict(n_periods=n_periods)

# Extend trend and seasonal components
last_trend = trend[-n_periods:].values
last_seasonal = seasonal[-n_periods:].values

# Add ARIMA-forecasted residuals
reconstructed_forecast = forecast_resid + last_trend + last_seasonal
forecast_index = test.index
min_len = min(len(forecast_index), len(reconstructed_forecast))
print(reconstructed_forecast)

# Plot the final forecast
plt.plot(train.index, train, label='Training', color='blue')
plt.plot(test.index[:min_len], test[:min_len], label='Test', color='green')
plt.plot(forecast_index[:min_len], reconstructed_forecast, label='Reconstructed Forecast', color='red')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.title('Forecast with STL + ARIMA Residuals')
plt.show()

# Model Evaluation
mae = mean_absolute_error(test[:min_len], reconstructed_forecast[:min_len])
mse = mean_squared_error(test[:min_len], reconstructed_forecast[:min_len])
rmse = np.sqrt(mse) # Calculate RMSE from MSE
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")