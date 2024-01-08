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

'''# Create Lineplot
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
 
# Calculate Jarque-Bera test statistics and p-values
def calculate_jarque_bera(data):
    array = []
    for equity in data.columns:
        ret = sp.stats.jarque_bera(data[equity][1:])
        array.append([ret[0], ret[1]])
    return pd.DataFrame(array, index=data.columns, columns=['Stat', 'pvalue'])
 
# Returns Hist
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

# Define a function to plot returns
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
 
# Compute BIC
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
 
# ARMA Model, ACF, PACF, Ljung Box
def estimate_arma_model(data, df_check, lags_acf_pacf, ar, ma, folder_name, type='Log-Returns'):
    acf_pacf_folder = os.path.join(folder_name, 'ACF - PACF', type)
    os.makedirs(acf_pacf_folder, exist_ok=True)

    lb_folder = os.path.join(folder_name, 'Ljung_Box', type)
    os.makedirs(lb_folder, exist_ok=True)

    labels_check = df_check[df_check['check'] == 'Stationarity'].index.tolist()
    all_results =[]

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

        # ARIMA MODEL
        order = [ar, 0, ma]
        model = sm.tsa.ARIMA(data[equity], order=order)
        results = model.fit()

        # Find residuals and compute Ljung Box
        residuals = results.resid
        lb_residuals = acorr_ljungbox(residuals, lags=lags_acf_pacf, model_df=ar + ma)

        # Compute ACF and PACF of residuals
        pacf_residuals = sm.tsa.pacf(residuals, nlags=lags_acf_pacf)
        acf_residuals = sm.tsa.acf(residuals, nlags=lags_acf_pacf)

        # Create table of ACF and PACF of residuals and save it in Excel File
        acf_pacf_residuals = pd.DataFrame({'ACF': acf_residuals, 'PACF': pacf_residuals})
        acf_pacf_residuals.to_excel(os.path.join(acf_pacf_folder, f'{equity} residuals - values.xlsx'))
        
        # Plot ACF and PACF residuals
        fig, (ax3, ax4) = plt.subplots(2, 1)
        plot_acf(residuals, lags=lags_acf_pacf, ax=ax3)
        plot_pacf(residuals, lags=lags_acf_pacf, ax=ax4)
        fig.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(acf_pacf_folder, f'{equity}_residuals.png'))
        plt.close()

        # Take Standard Error and t-statistics and save it in Excel File
        std_e = results.bse
        t_stat = results.tvalues
        parameters = pd.concat([std_e, t_stat], axis=1, keys=['Standard Error', 't-statistics'])
        parameters.to_excel(os.path.join(acf_pacf_folder, f'{equity}_parameters.xlsx'))

        # Save LB in Excel File
        lb_residuals.to_excel(os.path.join(lb_folder, f'{equity}.xlsx'))
        print(results.summary())
        all_results.append(results)
    return all_results

def forecast_time_series(data: pd.DataFrame, model, forecast_periods: int, folder_name):
    forecast_folder = os.path.join(folder_name, 'Forecast')
    os.makedirs(forecast_folder, exist_ok=True)

    # Checking if the forecast_periods is valid
    if forecast_periods < 1:
        raise ValueError("forecast_periods should be greater than or equal to 1.")
 
    # Creating an empty DataFrame to store the forecasted values, confidence intervals, and true values
    forecast_df = pd.DataFrame(columns=["Forecast", "Lower CI", "Upper CI", "True Value"])
    
    for equity in data:
        print(equity)
        # Rolling procedure for forecasting
        for i in range(len(equity) - forecast_periods + 1):
            # Splitting the data into training and testing sets
            train_equity = equity.iloc[:i + forecast_periods]
            test_equity = equity.iloc[i + forecast_periods]
 
        # Fitting the model on the training data
            #model_fit = model.fit(train_equity)
 
        # Forecasting the next period
            forecast, stderr, conf_int = model[i].forecast(steps=1)
 
            # Appending the forecasted values, confidence intervals, and true values to the DataFrame
            forecast_df = forecast_df.append({
                "Forecast": forecast[0],
                "Lower CI": conf_int[0][0],
                "Upper CI": conf_int[0][1],
                "True Value": test_equity
            }, ignore_index=True)

            plt.plot(forecast_df.index, forecast_df["Forecast"], label="Forecast")
            plt.fill_between(forecast_df.index, forecast_df["Lower CI"], forecast_df["Upper CI"], alpha=0.3, label="Confidence Interval")
            plt.plot(forecast_df.index, forecast_df["True Value"], label="True Value")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title(f"{equity} Series Forecast")
            plt.savefig(os.path.join(forecast_folder, f'{equity}.xlsx'))
            plt.close()'''

def forecast(data, forecast_periods):
    for equity in data:
        mod1 = sm.tsa.ARIMA(data[equity][:forecast_periods], order=(1,0,0), trend='n')
        out1 = mod1.fit()
        for1 = np.zeros([30,1])
        for1[0 ,0] = out1.forecast(steps=1)
        se1 = np.zeros([30,1])
        resvar = out1.sse/np.size(out1.resid)
        se1[0 ,0] = np.sqrt(resvar)
        for ii in range(1,30):
            m = forecast_periods + ii
            mod1 = sm.tsa.ARIMA(data[equity][0:m], order=(1,0,0), trend='n')
            out1 = mod1.fit()
            for1[ii,0] = out1.forecast(steps=1)
            resvar = out1.sse/np.size(out1.resid)
            se1[ii,0] = np.sqrt(resvar)
        t = np.arange(forecast_periods,forecast_periods+30)
        plt.plot(t, for1, 'b--', t, data[equity][forecast_periods:forecast_periods+30], 'k-', t, for1+1.96 *se1, 'r--', t, for1
        -1.96 *se1 ,'r--')
        plt.savefig('Fore2.png', dpi = 300)