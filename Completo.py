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
import f

# Read data from Excel files
equities_d = pd.read_excel('equities.xlsx', 'equities_d', index_col=None, na_values=['NA '])
equities_m = pd.read_excel('equities.xlsx', 'equities_m', index_col=None, na_values=['NA '])
economics = pd.read_excel('economics.xlsx', 'Foglio 1', index_col=None, na_values=['NA '])

labels_equity = equities_d.columns[1:]
labels_economics = economics.columns[1:]
P_e=pd.DataFrame(economics,columns=labels_economics)

# Time
t_d= equities_d["Date"]
t_m= equities_m["Date"]
t_e= economics["data"]

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
'''def create_lineplot(data, time, folder_name):
    lineplot_folder = os.path.join(folder_name, 'LinePlot')
    os.makedirs(lineplot_folder, exist_ok=True)
    
    for label in data.columns:
        plt.plot(time, data[label], label=f'{label} Stock Price')
        plt.xlabel('Time')
        plt.ylabel('log prices')
        plt.title(label)
        plt.savefig(os.path.join(lineplot_folder, f'{label}.png'), dpi=300)
        plt.close()
 
create_lineplot(log_equities_d, equities_d["Date"], 'Daily')
create_lineplot(log_equities_m, equities_m["Date"], 'Monthly')
create_lineplot(economics.iloc[:, 1:], economics["data"], 'Economics')'''

# Calculate percentage returns
r_d = 100 * (log_equities_d - log_equities_d.shift(1))
r_d[1:].to_excel(os.path.join('Daily', 'Percentage Return.xlsx'))
r_m = 100 * (log_equities_m - log_equities_m.shift(1))
r_m[1:].to_excel(os.path.join('Monthly', 'Percentage Return.xlsx'))
r_e = 100 * ((economics.iloc[:, 1:] - economics.iloc[:, 1:].shift(1)) / economics.iloc[:, 1:].shift(1))
r_e[1:].to_excel(os.path.join('Economics', 'Percentage Return.xlsx'))

# Generate descriptive statistics, skewness, kurtosis and save results to Excel Files
'''def save_return_description(data, labels, folder_name, filename):
    sd = os.path.join(folder_name, 'Statistics Description')
    os.makedirs(sd, exist_ok=True)
    
    return_description = pd.DataFrame(index=labels)
    return_description['skew'] = data.skew()
    return_description['kurt'] = data.kurt()
    return_description = pd.merge(return_description, data.describe().T, left_index=True, right_index=True)
    return_description.to_excel(os.path.join(sd, filename))
 
save_return_description(r_d, labels_equity, 'Daily', 'daily_return_description.xlsx')
save_return_description(r_m, labels_equity, 'Monthly', 'monthly_return_description.xlsx')
save_return_description(r_e, labels_economics, 'Economics', 'economics_return_description.xlsx')

# Calculate Jarque-Bera test statistics and p-values
def calculate_jarque_bera(data):
    array = []
    for equity in data.columns:
        ret = sp.stats.jarque_bera(data[equity][1:])
        array.append([ret[0], ret[1]])
    return pd.DataFrame(array, index=data.columns, columns=['Stat', 'pvalue'])
 
jarque_bera_r_d = calculate_jarque_bera(log_equities_d)
jarque_bera_r_m = calculate_jarque_bera(log_equities_m)
jarque_bera_r_e = calculate_jarque_bera(r_e)

# Create directory if it doesn't exist
os.makedirs('Jarque Bera', exist_ok=True)

# Merge and save results to Excel files
financial_jb = pd.concat([jarque_bera_r_d, jarque_bera_r_m], axis=1, keys=['Daily', 'Monthly'])
financial_jb.to_excel(os.path.join('Jarque Bera','Financial.xlsx'))
jarque_bera_r_e.to_excel(os.path.join('Jarque Bera', 'Economics.xlsx'))

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

hist_plot(r_d, labels_equity, 50, 'Daily')
hist_plot(r_m, labels_equity, 25, 'Monthly')
hist_plot(r_e, labels_economics, 25, 'Economics')'''

# Define a function to plot returns
'''def plot_returns(time, returns, folder_name):
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
 
# Plot returns for equities_d, equities_m, and economics
plot_returns(equities_d["Date"], r_d,'Daily')
plot_returns(equities_m["Date"], r_m, 'Monthly')
plot_returns(economics["data"], r_e, 'Economics')'''

# Variables with reduced time
r_m_short=r_m.iloc[:108].copy()
p_m_short=log_equities_m.iloc[:108].copy()
t_m_short = t_m[:108] #removed last 2 years

r_d_short=r_d.iloc[:2585].copy()
p_d_short=log_equities_d.iloc[:2585].copy()
t_d_short = t_d[:2585]

r_e_short=r_e.iloc[:101].copy()
p_e_short=P_e.iloc[:101].copy()
t_e_short = t_e[:101]

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

BIC_d_df = calculate_BIC(p_d_short, labels_equity, 22, 'Daily')
BIC_m_df = calculate_BIC(p_m_short, labels_equity, 22, 'Monthly')
BIC_e_df = calculate_BIC(p_e_short, labels_economics, 12, 'Economics')

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
 
adf_result_d, p_d_unit_root = perform_adf_test(p_d_short, BIC_d_df, labels_equity, 0.05, 22, 'Daily') 
adf_result_m, p_m_unit_root = perform_adf_test(p_m_short, BIC_m_df, labels_equity, 0.05, 12, 'Monthly')
adf_result_e, p_e_unit_root = perform_adf_test(p_e_short, BIC_e_df, labels_economics, 0.05, 12, 'Economics')

# ADF - FIRST DIFFERENCE
p_d_unit_root_BIC_df = calculate_BIC(r_d_short[1:], p_d_unit_root, maxlag=22, folder_name='Daily', type='First Difference')
p_m_unit_root_BIC_df = calculate_BIC(r_m_short[1:], p_m_unit_root, maxlag=12, folder_name='Monthly', type='First Difference')
p_e_unit_root_BIC_df = calculate_BIC(r_e_short[1:], p_e_unit_root, maxlag=12, folder_name='Economics', type='First Difference')

adf_result_d_first, _ = perform_adf_test(r_d_short[1:], p_d_unit_root_BIC_df, p_d_unit_root, 
                                         0.05, 22, 'Daily', type='First Difference')
 
adf_result_m_first, _ = perform_adf_test(r_m_short[1:], p_m_unit_root_BIC_df, p_m_unit_root, 
                                         0.05, 12, 'Monthly', type='First Difference')

adf_result_e_first, _ = perform_adf_test(r_e_short[1:], p_e_unit_root_BIC_df, p_e_unit_root, 
                                         0.05, 12, 'Economics', type='First Difference')

# ARMA Model, ACF, PACF, Ljung Box
def estimate_arma_model(data, df_check, lags_acf_pacf, ar, ma, folder_name, type='Log-Returns'):
    acf_pacf_folder = os.path.join(folder_name, 'ACF - PACF', type)
    os.makedirs(acf_pacf_folder, exist_ok=True)

    lb_folder = os.path.join(folder_name, 'Ljung_Box', type)
    os.makedirs(lb_folder, exist_ok=True)

    labels_check = df_check[df_check['check'] == 'Stationarity'].index.tolist()

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

estimate_arma_model(r_d_short[1:], adf_result_d_first, 20, 1, 1, 'Daily', 'First Difference')
estimate_arma_model(r_m_short[1:], adf_result_m_first, 20, 1, 1, 'Monthly', 'First Difference')
estimate_arma_model(r_e_short[1:], adf_result_e_first, 20, 1, 1, 'Economics', 'First Difference')
estimate_arma_model(p_d_short, adf_result_d, 20, 1, 1, 'Daily', 'Log-Prices')
estimate_arma_model(p_m_short, adf_result_m, 20, 1, 1, 'Monthly', 'Log-Prices')
estimate_arma_model(p_e_short, adf_result_e, 20, 1, 1, 'Economics', 'Log-Prices')

