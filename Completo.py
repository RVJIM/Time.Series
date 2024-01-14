import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
import fun

# Read data from Excel files
equities_d = pd.read_excel('equities.xlsx', 'equities_d', index_col=None, na_values=['NA '])
equities_m = pd.read_excel('equities.xlsx', 'equities_m', index_col=None, na_values=['NA '])
economics = pd.read_excel('economics.xlsx', 'Foglio 1', index_col=None, na_values=['NA '])

labels_equity = equities_d.columns[1:]
labels_economics = economics.columns[1:]
P_e=pd.DataFrame(economics,columns=labels_economics)

# Time
t_d = equities_d["Date"]
t_m = equities_m["Date"]
t_e = economics["data"]

# Create folders Daily, Monthly and Economics
os.makedirs('Daily', exist_ok=True)
os.makedirs('Monthly', exist_ok=True)
os.makedirs('Economics', exist_ok=True)

# Calculate logarithmic returns and save in Excel
log_equities_d = np.log(equities_d.iloc[:, 1:])
log_equities_d.to_excel(os.path.join('Daily', 'Return Logarithmic.xlsx'))
log_equities_m = np.log(equities_m.iloc[:, 1:])
log_equities_m.to_excel(os.path.join('Monthly', 'Return Logarithmic.xlsx'))

# Create Lineplot 
'''fun.create_lineplot(log_equities_d, equities_d["Date"], 'Daily')
fun.create_lineplot(log_equities_m, equities_m["Date"], 'Monthly')
fun.create_lineplot(economics.iloc[:, 1:], economics["data"], 'Economics') 

# Calculate percentage returns
r_d = 100 * (log_equities_d - log_equities_d.shift(1))
r_d[1:].to_excel(os.path.join('Daily', 'Percentage Return.xlsx'))
r_m = 100 * (log_equities_m - log_equities_m.shift(1))
r_m[1:].to_excel(os.path.join('Monthly', 'Percentage Return.xlsx'))
r_e = 100 * ((economics.iloc[:, 1:] - economics.iloc[:, 1:].shift(1)) / economics.iloc[:, 1:].shift(1))
r_e[1:].to_excel(os.path.join('Economics', 'Percentage Return.xlsx'))

# Generate descriptive statistics, skewness, kurtosis and save results to Excel Files 
fun.save_return_description(r_d, labels_equity, 'Daily', 'daily_return_description.xlsx')
fun.save_return_description(r_m, labels_equity, 'Monthly', 'monthly_return_description.xlsx')
fun.save_return_description(r_e, labels_economics, 'Economics', 'economics_return_description.xlsx')

# Calculate Jarque-Bera test statistics and p-values 
jarque_bera_r_d = fun.calculate_jarque_bera(log_equities_d)
jarque_bera_r_m = fun.calculate_jarque_bera(log_equities_m)
jarque_bera_r_e = fun.calculate_jarque_bera(r_e)

# Create directory if it doesn't exist
os.makedirs('Jarque Bera', exist_ok=True)

# Merge and save results to Excel files
financial_jb = pd.concat([jarque_bera_r_d, jarque_bera_r_m], axis=1, keys=['Daily', 'Monthly'])
financial_jb.to_excel(os.path.join('Jarque Bera','Financial.xlsx'))
jarque_bera_r_e.to_excel(os.path.join('Jarque Bera', 'Economics.xlsx'))

# Returns Hist
fun.hist_plot(r_d, labels_equity, 50, 'Daily')
fun.hist_plot(r_m, labels_equity, 25, 'Monthly')
fun.hist_plot(r_e, labels_economics, 25, 'Economics') 

# Plot returns for equities_d, equities_m, and economics
fun.plot_returns(equities_d["Date"], r_d,'Daily')
fun.plot_returns(equities_m["Date"], r_m, 'Monthly')
fun.plot_returns(economics["data"], r_e, 'Economics')

# Variables with reduced time
r_m_short = r_m.iloc[:108].copy()
p_m_short = log_equities_m.iloc[:108].copy()
t_m_short = t_m[:108] #removed last 2 years

r_d_short = r_d.iloc[:2585].copy()
p_d_short = log_equities_d.iloc[:2585].copy()
t_d_short = t_d[:2585]

r_e_short = r_e.iloc[:101].copy()
p_e_short = P_e.iloc[:101].copy()
t_e_short = t_e[:101]

# Compute BIC
BIC_d_df = fun.calculate_BIC(p_d_short, labels_equity, 22, 'Daily')
BIC_m_df = fun.calculate_BIC(p_m_short, labels_equity, 22, 'Monthly')
BIC_e_df = fun.calculate_BIC(p_e_short, labels_economics, 12, 'Economics')

# Augmented Dickey-Fuller 
adf_result_d, p_d_unit_root = fun.perform_adf_test(p_d_short, BIC_d_df, labels_equity, 0.05, 22, 'Daily') 
adf_result_m, p_m_unit_root = fun.perform_adf_test(p_m_short, BIC_m_df, labels_equity, 0.05, 12, 'Monthly')
adf_result_e, p_e_unit_root = fun.perform_adf_test(p_e_short, BIC_e_df, labels_economics, 0.05, 12, 'Economics')

# ADF - FIRST DIFFERENCE
p_d_unit_root_BIC_df = fun.calculate_BIC(r_d_short[1:], p_d_unit_root, maxlag=22, folder_name='Daily', type='First Difference')
p_m_unit_root_BIC_df = fun.calculate_BIC(r_m_short[1:], p_m_unit_root, maxlag=12, folder_name='Monthly', type='First Difference')
p_e_unit_root_BIC_df = fun.calculate_BIC(r_e_short[1:], p_e_unit_root, maxlag=12, folder_name='Economics', type='First Difference')

adf_result_d_first, _ = fun.perform_adf_test(r_d_short[1:], p_d_unit_root_BIC_df, p_d_unit_root, 0.05, 22, 'Daily', type='First Difference')
adf_result_m_first, _ = fun.perform_adf_test(r_m_short[1:], p_m_unit_root_BIC_df, p_m_unit_root, 0.05, 12, 'Monthly', type='First Difference')
adf_result_e_first, _ = fun.perform_adf_test(r_e_short[1:], p_e_unit_root_BIC_df, p_e_unit_root, 0.05, 12, 'Economics', type='First Difference')

labels_check_d_p, labels_not_d = fun.stationarity_not(adf_result_d)
labels_check_m_p, labels_not_m = fun.stationarity_not(adf_result_m)
labels_check_e_p, labels_not_e = fun.stationarity_not(adf_result_e)
labels_check_d_fd, labels_not_d_fd = fun.stationarity_not(adf_result_d_first)
labels_check_m_fd, labels_not_m_fd = fun.stationarity_not(adf_result_m_first)
labels_check_e_fd, labels_not_e_fd = fun.stationarity_not(adf_result_e_first)

# ARMA Model, ACF, PACF, Ljung Box
model_results_d, best_order_d = fun.estimate_arma_model_new(p_d_short, labels_check_d_p, 20, 'Daily', 'Log-Prices')
model_results_m, best_order_m = fun.estimate_arma_model_new(p_m_short, labels_check_m_p, 20, 'Monthly', 'Log-Prices')
model_results_e, best_order_e = fun.estimate_arma_model_new(p_e_short, labels_check_e_p, 20, 'Economics', 'Log-Prices')
model_results_d_fd, best_order_d_fd = fun.estimate_arma_model_new(r_d_short[1:], labels_check_d_fd, 20, 'Daily', 'First Difference')
model_results_m_fd, best_order_m_fd = fun.estimate_arma_model_new(r_m_short[1:], labels_check_m_fd, 20, 'Monthly', 'First Difference')
model_results_e_fd, best_order_e_fd = fun.estimate_arma_model_new(r_e_short[1:], labels_check_e_fd, 20, 'Economics', 'First Difference')

# Compute Forecast rolling, with return a dictionary with all dataframe of Time, Forecast's value, Lower CI, Upper CI and True Value
# Period of forecast
forecast_periods_daily = 125 # 125,Exactly this, start from 06/18/2023 until 12/18/2023
forecast_periods_monthly = 25 # 25 not 24, because we eliminated December 2023 from the calculation

# All Forecast
forecast_result_daily = fun.forecast_time_series(log_equities_d, labels_check_d_p, forecast_periods_daily, best_order_d, 'Daily', 'Log-Prices')
forecast_result_monthly = fun.forecast_time_series(log_equities_m, labels_check_m_p, forecast_periods_monthly, best_order_m, 'Monthly', 'Log-Prices')
forecast_result_economics = fun.forecast_time_series(P_e[1:], labels_check_e_p, forecast_periods_monthly, best_order_e, 'Economics', 'Price')
forecast_result_daily_fd = fun.forecast_time_series(r_d[1:], labels_check_d_fd, forecast_periods_daily, best_order_d_fd, 'Daily', 'Percentage Return')
forecast_result_monthly_fd = fun.forecast_time_series(r_m[1:], labels_check_m_fd, forecast_periods_monthly, best_order_m_fd, 'Monthly', 'Percentage Return')
forecast_result_economics_fd = fun.forecast_time_series(r_e[1:], labels_check_e_fd, forecast_periods_monthly, best_order_e_fd,  'Economics', 'Percentage Return')

# ARMA Model, ACF, PACF, Ljung Box for non-stationarity
model_results_d_not, best_order_d_not = fun.estimate_arma_model_new(p_d_short, labels_not_d, 20, 'Daily', 'Log-Prices', 'Non-Stationarity')
model_results_m_not, best_order_m_not = fun.estimate_arma_model_new(p_m_short, labels_not_m, 20, 'Monthly', 'Log-Prices', 'Non-Stationarity')
model_results_e_not, best_order_e_not = fun.estimate_arma_model_new(p_e_short, labels_not_e, 20, 'Economics', 'Log-Prices', 'Non-Stationarity')

# Forecast for non-stationarity level
forecast_result_daily_not = fun.forecast_time_series(log_equities_d, labels_not_d, forecast_periods_daily, best_order_d_not, 'Daily', 'Log-Prices', 'Non-Stationarity')
forecast_result_monthly_not = fun.forecast_time_series(log_equities_m, labels_not_m, forecast_periods_monthly, best_order_m_not, 'Monthly', 'Log-Prices', 'Non-Stationarity')
forecast_result_economics_not = fun.forecast_time_series(r_e[1:], labels_not_e, forecast_periods_monthly, best_order_e_not, 'Economics', 'Price', 'Non-Stationarity')

# Compare forecast with Random Walk
fun.compare_forecast_rw(labels_not_d, forecast_result_daily_not, forecast_periods_daily, 123, 'Daily', 'Comparing', 'Log-Prices')
fun.compare_forecast_rw(labels_not_m, forecast_result_monthly_not, forecast_periods_monthly, 123, 'Monthly', 'Comparing', 'Log-Prices')
fun.compare_forecast_rw(labels_not_e, forecast_result_economics_not, forecast_periods_monthly, 123, 'Economics', 'Comparing', 'Log-Prices')
'''
# Square log returns, estimate arma and make forecast
# log_equities_d_square = log_equities_d ** 2
r_d_square = r_d ** 2
# r_d_square = 100 * (log_equities_d_square - log_equities_d_square.shift(1))
r_d_square[1:].to_excel(os.path.join('Daily', 'Percentage Return - Square.xlsx'))
r_d_short_square = r_d_square.iloc[:2585].copy()
# p_d_short_square = log_equities_d_square.iloc[:2585].copy()

BIC_d_df_square = fun.calculate_BIC(r_d_short_square[1:], labels_equity, 22, 'Daily')
adf_result_d_square, r_d_unit_root_square = fun.perform_adf_test(r_d_short_square[1:], BIC_d_df_square, labels_equity, 0.05, 22, 'Daily')
# p_d_unit_root_BIC_df_square = fun.calculate_BIC(r_d_short_square[1:], p_d_unit_root_square, maxlag=22, folder_name='Daily', type='First Difference')
# adf_result_d_first_square, _ = fun.perform_adf_test(r_d_short_square[1:], p_d_unit_root_BIC_df_square, p_d_unit_root_square, 0.05, 22, 'Daily', type='First Difference')

# labels_check_d_p_square, labels_not_d_square = fun.stationarity_not(adf_result_d_square)
labels_check_d_fd_square, labels_not_d_fd_square = fun.stationarity_not(adf_result_d_square)

# model_results_d_r_square, best_order_r_square = fun.estimate_arma_model_new(p_d_short_square, labels_check_d_p_square, 20, 'Daily', 'Square Log-Prices')
model_results_d_fd_square, best_order_d_fd_square = fun.estimate_arma_model_new(r_d_short_square[1:], labels_check_d_fd_square, 20, 'Daily', 'Square First Difference')
#model_results_d_not_square, best_order_d_not_square = fun.estimate_arma_model_new(p_d_short_square, labels_not_d_square, 20, 'Daily', 'Log-Prices', 'Non-Stationarity')
forecast_periods_daily = 125
# forecast_result_daily_square = fun.forecast_time_series(log_equities_d_square, labels_check_d_p_square, forecast_periods_daily, best_order_d_square, 'Daily', 'Square Log-Price')
forecast_result_daily_fd_square = fun.forecast_time_series(r_d_square[1:], labels_check_d_fd_square, forecast_periods_daily, best_order_d_fd_square, 'Daily', 'Square Percentage Return')
#forecast_result_daily_not_square = fun.forecast_time_series(log_equities_d_square, labels_not_d_square, forecast_periods_daily, best_order_d_not_square, 'Daily', 'Square Log-Price', 'Non-Stationarity') 
