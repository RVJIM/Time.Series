import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

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
        

def forecast():
    # simulate AR (1)
    phi1 = 0.8
    ma= [1]
    ar=[1, -phi1 ]
    x=sm. tsa . arma_generate_sample (ar ,ma , nsample =230 , burnin = 100)
    # estimate - leave last 30 for forecast
    mod1 =sm. tsa . ARIMA (x [0:199] , order =(1 ,0 ,0) ,trend ='n')
    out1 = mod1 . fit ()
    # dynamic forecast
    for1 = out1 . forecast ( steps = 30)
    # static forecast
    for2 =np. zeros ([30 ,1])
    for2 [0 ,0]= out1 . forecast ( steps =1)
    se1 =np. zeros ([30 ,1])
    resvar = out1 . sse /np. size ( out1 . resid )
    se1 [0 ,0]=np. sqrt ( resvar )

    for ii in range (1 ,30) :
        m= 199+ ii
        mod1 =sm. tsa . ARIMA (x[0: m], order =(1 ,0 ,0) ,trend ='n')
        out1 = mod1 . fit ()
        for2 [ii ,0] = out1 . forecast ( steps =1)
        resvar = out1 . sse /np. size ( out1 . resid )
        se1 [ii ,0] =np. sqrt ( resvar )
    
    ax1 = plt . subplot (121)
    plt . plot (np. arange (200 ,230) ,for1 ,'b--',np. arange (200 ,230) ,x [200:230] , 'k-')
    ax2 = plt . subplot (122)
    plt . plot (np. arange (200 ,230) ,for2 ,'b--',np. arange (200 ,230) ,x [200:230] , 'k-')
    plt . savefig ('Fore1 . png ',dpi = 300)

    # plot s of dynamic forecasts and confidence interval
    t=np. arange (200 ,230)
    plt . plot (t,for2 ,'b--',t,x [200:230] , 'k-',t, for2 +1.96 *se1 ,'r--',t,for2 -1.96 *se1 ,'r--')
    plt . savefig ('Fore2 . png ',dpi = 300)


def rolling_arima():
    phi1 = 0.6
    phi2 = -0.7
    phi3 = -0.3
    ar=[1, -phi1 , -phi2 , -phi3 ]
    ma= [1]
    x=sm. tsa . arma_generate_sample (ar ,ma , nsample =350 , burnin = 100)
    # fit different models : AR (3) , AR (1) , MA (3)
    # store 1- step a head forecasts
    foreall =np. zeros ([50 ,3])
    mod1 =sm. tsa . ARIMA (x [0:299] , order =(3 ,0 ,0))
    mod2 =sm. tsa . ARIMA (x [0:299] , order =(1 ,0 ,0))
    mod3 =sm. tsa . ARIMA (x [0:299] , order =(0 ,0 ,2))
    out1 = mod1 . fit ()
    out2 = mod2 . fit ()
    out3 = mod3 . fit ()
    foreall [0 ,0]= out1 . forecast ( steps =1)
    foreall [0 ,1]= out2 . forecast ( steps =1)
    foreall [0 ,2]= out3 . forecast ( steps =1)
    
    for ii in range (1 ,50) :
        m= 299+ ii
        mod1 =sm. tsa . ARIMA (x[0: m], order =(3 ,0 ,0))
        mod2 =sm. tsa . ARIMA (x[0: m], order =(1 ,0 ,0))
        mod3 =sm. tsa . ARIMA (x[0: m], order =(0 ,0 ,2))
        out1 = mod1 . fit ()
        out2 = mod2 . fit ()
        out3 = mod3 . fit ()
        foreall [ii ,0] = out1 . forecast ( steps =1)
        foreall [ii ,1] = out2 . forecast ( steps =1)
        foreall [ii ,2] = out3 . forecast ( steps =1)
    
    # compute MSE
    xtrue =x [300:350]
    err1 = foreall [: ,0] - xtrue
    err2 = foreall [: ,1] - xtrue
    err3 = foreall [: ,2] - xtrue
    mse1 =np. sum (np. power (err1 ,2) )
    mse2 =np. sum (np. power (err2 ,2) )
    mse3 =np. sum (np. power (err3 ,2) )
    return mse1, mse2, mse3