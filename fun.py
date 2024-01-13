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

# Create Lineplot
def create_lineplot(data, time, folder_name):
    lineplot_folder = os.path.join(folder_name, 'LinePlot')
    os.makedirs(lineplot_folder, exist_ok=True)
    
    for label in data.columns:
        plt.plot(time, data[label], label=f'{label} Stock Price')
        plt.xlabel('Time')
        plt.ylabel('log prices')
        plt.title(label)
        plt.savefig(os.path.join(lineplot_folder, f'{label}.png'), dpi=300)
        plt.close()

# Generate descriptive statistics, skewness, kurtosis and save results to Excel Files
def save_return_description(data, labels, folder_name, filename):
    sd = os.path.join(folder_name, 'Statistics Description')
    os.makedirs(sd, exist_ok=True)
    
    return_description = pd.DataFrame(index=labels)
    return_description['skew'] = data.skew()
    return_description['kurt'] = data.kurt()
    return_description = pd.merge(return_description, data.describe().T, left_index=True, right_index=True)
    return_description.to_excel(os.path.join(sd, filename))


def calculate_jarque_bera(data):
    array = []
    for equity in data.columns:
        ret = sp.stats.jarque_bera(data[equity][1:])
        array.append([ret[0], ret[1]])
    return pd.DataFrame(array, index=data.columns, columns=['Stat', 'pvalue'])


def hist_plot(data, labels, bins, folder_name):
    hist_folder = os.path.join(folder_name, 'HistPlot - Returns')
    os.makedirs(hist_folder, exist_ok=True)

    for equity in labels:
        plt.hist(data[equity], bins=bins, density=True )
        plt.title(equity)
        plt.xlabel('Returns')
        plt.ylabel('Frequencies')
        plt.savefig(os.path.join(hist_folder, f'{equity}.png'), dpi=300)
        plt.close()


def plot_returns(time, returns, folder_name):
    returns_folder = os.path.join(folder_name, 'LinePlot - Returns')
    os.makedirs(returns_folder, exist_ok=True)

    for equity in returns.columns:
        plt.plot(time[1:], returns[equity][1:], label=f'{equity} Returns')
        plt.axhline(np.mean(returns[equity]), color='r', label='Mean')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.title(equity)
        plt.savefig(os.path.join(returns_folder, f'{equity}.png'), dpi=300)
        plt.close()


def calculate_BIC(data, labels, maxlag, folder_name, type = 'Normal'):
    ADF_BIC_folder = os.path.join(folder_name, 'ADF - BIC')
    os.makedirs(ADF_BIC_folder, exist_ok=True)

    BIC = []
    for equity in labels:
        ret = adfuller(data[equity], maxlag=maxlag, regression='c', autolag='BIC')
        ret1 = adfuller(data[equity], maxlag=maxlag, regression='ct', autolag='BIC')
        ret2 = adfuller(data[equity], maxlag=maxlag, regression='n', autolag='BIC')
        BIC.append([ret[5], ret1[5], ret2[5]])
    BIC_df = pd.DataFrame(BIC, index=labels, columns=['BIC constant', 'BIC constant and trend', 'BIC no constant no trend'])
    BIC_df['deterministic part'] = BIC_df.idxmin(axis=1)
    
    file_path = os.path.join(ADF_BIC_folder, f'{type}.xlsx')
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            BIC_df.to_excel(writer, sheet_name='BIC')

    return BIC_df


# Augmented Dickey-Fuller
def adf_test(data, maxlag, regression):
    ret = adfuller(data, maxlag=maxlag, regression=regression)
    return [ret[0], ret[1], ret[2], ret[3]]
 
def perform_adf_test(p_short, BIC_df, labels_equity, conf, freq, folder_name, type = 'Normal'):
    ADF_BIC_folder = os.path.join(folder_name, 'ADF - BIC')

    adf_result = []
    solutions = []
    p_unit_root = []
    
    for equity in BIC_df.index:
        if BIC_df.at[equity, BIC_df.columns[3]] == BIC_df.columns[2]:
            ret = adf_test(p_short[equity], maxlag=freq, regression='n')
            adf_result.append([ret[0], ret[1], ret[2], ret[3]])
        elif BIC_df.at[equity, BIC_df.columns[3]] == BIC_df.columns[1]:
            ret = adf_test(p_short[equity], maxlag=freq, regression='ct')
            adf_result.append([ret[0], ret[1], ret[2], ret[3]])
        else:
            ret = adf_test(p_short[equity], maxlag=freq, regression='c')
            adf_result.append([ret[0], ret[1], ret[2], ret[3]])
        
        if adf_result[-1][1] < conf:
            solutions.append('Stationarity')
        else:
            solutions.append('Unit root')
            p_unit_root.append(equity)
    
    adf_result_df = pd.DataFrame(adf_result, index=labels_equity, columns=['stat', 'pvalue', 'lags', 'obs'])
    adf_result_df['check'] = solutions

    file_path = os.path.join(ADF_BIC_folder, f'{type}.xlsx')
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
            adf_result_df.to_excel(writer, sheet_name='ADF')
    
    return adf_result_df, p_unit_root

def stationarity_not(df_check):
    labels_check = df_check[df_check['check'] == 'Stationarity'].index.tolist()
    labels_not = df_check[df_check['check'] != 'Stationarity'].index.tolist()
    return labels_check, labels_not

# ARMA Model, ACF, PACF, Ljung Box
def estimate_arma_model_new(data, labels_check, lags_acf_pacf, folder_name, type, check = 'Stationaritiy'):
    import itertools
    acf_pacf_folder = os.path.join(folder_name, 'ACF - PACF',check, type)
    os.makedirs(acf_pacf_folder, exist_ok=True)

    lb_folder = os.path.join(folder_name, 'Ljung_Box', type)
    os.makedirs(lb_folder, exist_ok=True)

    order_label = ['AR', 'd', 'MA']
    arma_order_d = {}
    all_results = []

    for equity in labels_check:
        # Compute ACF and PACF
        acf = sm.tsa.acf(data[equity], nlags=lags_acf_pacf)
        pacf = sm.tsa.pacf(data[equity], nlags=lags_acf_pacf)
        
        # Create table of ACF and PACF of log returns and save it in Excel File
        acf_pacf = pd.DataFrame({'ACF': acf, 'PACF': pacf})
        acf_pacf.to_excel(os.path.join(acf_pacf_folder, f'{equity} {type} - values.xlsx'))
        
        # Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(2, 1)
        plot_acf(data[equity], lags=lags_acf_pacf, ax=ax1)
        plot_pacf(data[equity], lags=lags_acf_pacf, ax=ax2)
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(acf_pacf_folder, f'{equity}.png'))
        plt.close()

        # For ARMA
        p_values = range(0, 5)
        d_values = [0]  # Set d_values to 0 by default
        q_values = range(0, 5)

        # Create a list of all possible combinations of p, d, and q
        param_combinations = list(itertools.product(p_values, d_values , q_values))

        best_bic = float('inf')
        best_order = ()

        # Iterate through all combinations and fit ARIMA models
        for order in param_combinations:
            try:
                model = sm.tsa.ARIMA(data[equity], order=order)
                results = model.fit()
                bic = results.bic

        # Update the best order if the current model has a lower BIC
                if bic < best_bic:
                    best_bic = bic
                    best_order = order
            except:
                continue

        # Print the best order found
        print(f'Best ARIMA Order: {best_order} with BIC: {best_bic}')

        # Fit the best model
        best_model = sm.tsa.ARIMA(data[equity], order=best_order)
        best_results = best_model.fit()
        
        all_results.append(best_results)

        # Find residuals and compute Ljung Box
        residuals = results.resid
        lb_residuals = acorr_ljungbox(residuals, lags=lags_acf_pacf)

        # Compute ACF and PACF of residuals
        pacf_residuals = sm.tsa.pacf(residuals, nlags=lags_acf_pacf)
        acf_residuals = sm.tsa.acf(residuals, nlags=lags_acf_pacf)

        # Create table of ACF and PACF of residuals and save it in Excel File
        acf_pacf_residuals = pd.DataFrame({'ACF': acf_residuals, 'PACF': pacf_residuals})
        
        # Take Standard Error and t-statistics and save it in Excel File
        std_e = results.bse
        t_stat = results.tvalues
        parameters = pd.concat([std_e, t_stat], axis=1, keys=['Standard Error', 't-statistics'])

        # Store ARMA model results
        arma_order_d[equity] = best_order
        arma_order = pd.DataFrame(arma_order_d, index=order_label).T

        # Plot ACF and PACF residuals
        fig, (ax3, ax4) = plt.subplots(2, 1)
        plot_acf(residuals, lags=lags_acf_pacf, ax=ax3)
        plot_pacf(residuals, lags=lags_acf_pacf, ax=ax4)
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(acf_pacf_folder, f'{equity}_residuals.png'))
        plt.close()

        # Save LB in Excel File
        parameters.to_excel(os.path.join(acf_pacf_folder, f'{equity}_parameters.xlsx'))
        acf_pacf_residuals.to_excel(os.path.join(acf_pacf_folder, f'{equity} residuals - values.xlsx'))
        lb_residuals.to_excel(os.path.join(lb_folder, f'{equity}.xlsx'))
        arma_order.to_excel(os.path.join(acf_pacf_folder, 'Best order ARMA.xlsx'), index=True)

    return all_results, arma_order_d


def forecast_time_series(data: pd.DataFrame, labels_check, forecast_periods: int, order, folder_name: str, type: str, check = 'Stationaritiy'):
    forecast_folder = os.path.join(folder_name, 'Forecast', check, type)
    os.makedirs(forecast_folder, exist_ok=True)

    # Checking if the forecast_periods is valid
    if forecast_periods < 1:
        raise ValueError("forecast_periods should be greater than or equal to 1.")

    df_forecasts = {}

    for equity in labels_check:
        forecast = np.zeros([forecast_periods,1])
        mod = sm.tsa.ARIMA(data[equity][:-forecast_periods-1], order=order[equity], trend='n')
        result = mod.fit()
        forecast[0,0] = result.forecast(steps=1)
        std_e = np.zeros([forecast_periods,1])
        residual_variance = result.sse/np.size(result.resid)
        std_e[0,0] = np.sqrt(residual_variance)
        
        periods = len(data) - forecast_periods

    # We begin from 0 because periods variable is total - periods of forecast and we need to start from here
        for ii in range(0, forecast_periods):
            m = periods + ii
            mod_roll = sm.tsa.ARIMA(data[equity][:m], order=order[equity], trend='n')
            result_roll = mod_roll.fit()
            forecast[ii,0] = result_roll.forecast(steps=1)
            residual_variance_roll = result_roll.sse/np.size(result_roll.resid)
            std_e[ii,0] = np.sqrt(residual_variance_roll)

        t = np.arange(periods, periods+forecast_periods)
        upper_ci = forecast + 1.96 * std_e
        lower_ci = forecast - 1.96 * std_e

    # Create dataframe for with values and save it to Excel File
        forecast_df = pd.DataFrame({
                "Time": t.flatten(),
                "Forecast": forecast.flatten(),
                "Lower CI": lower_ci.flatten(),
                "Upper CI": upper_ci.flatten(),
                "True Value": data[equity][periods:]
            })

        forecast_df.to_excel(os.path.join(forecast_folder, f'{equity}.xlsx'))

        # Plot Forecast and save it
        forecast_df.set_index("Time", inplace=True)
    
        plt.plot(forecast_df.index, forecast_df["Forecast"], 'b--', label="Forecast")
        plt.plot(forecast_df.index, forecast_df["True Value"], 'k-', label="True Value")
        plt.plot(forecast_df.index, forecast_df["Upper CI"], 'r--', label="Upper CI")
        plt.plot(forecast_df.index, forecast_df["Lower CI"], 'r--', label="Lower CI")

        plt.legend()
        plt.xlabel("Time")
        plt.ylabel(type)
        plt.title(f"{equity} Series Forecast")
        plt.savefig(os.path.join(forecast_folder, f'{equity}.png'), dpi=300)
        plt.close()

        # Store DataFrame with own name
        df_forecasts[equity] = forecast_df
    return df_forecasts

def RW(n, seed):
    x = 0
    y = 0  
    position_x = [0] #Start from the origin
    position_y = [0]
    np.random.seed(seed)
    for i in range (1,n):
        step = np.random.uniform(0,1) #Choose steps randombly 0 or 1
        if step > 0.5:
            x += 1
            y += 1
            position_x.append(x)
            position_y.append(y)
        elif step < 0.5:
            x += 1
            y -= 1
            position_x.append(x)
            position_y.append(y)
    return [position_x,position_y]


# Compare forecast with Random Walk
def compare_forecast_rw(labels_not, forecast_df_not, forecast_periods, seed: int, folder_name: int, type: str, check = 'Non-Stationarity'):
    forecast_folder = os.path.join(folder_name, 'Forecast', check, type)
    os.makedirs(forecast_folder, exist_ok=True)
    
    for equity in labels_not:
        '''for i in range(forecast_periods):
            plt.plot(RW(forecast_periods)[0],RW(forecast_periods)[1],"b",label= "Random Walk")
            plt.plot(t, x_diff[1:31], 'k-', label='Actual Data')
            plt.plot(t, for2, 'r--', label='Dynamic Forecast')'''

        plt.plot(forecast_df_not[equity].index, forecast_df_not[equity]['Forecast'], 'r--', label=f'Forecast - {equity}')
        plt.plot(forecast_df_not[equity].index, RW(forecast_periods,seed)[1], 'b', label= "Random Walk")
        plt.plot(forecast_df_not[equity].index, forecast_df_not[equity]['True Value'], 'k-', label='True Value')

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.title(f"{equity}")
        plt.savefig(os.path.join(forecast_folder, f'{equity}.png'), dpi=300)
        plt.close()