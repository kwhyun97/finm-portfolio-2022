import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import kurtosis, skew
from scipy.stats import norm

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.neural_network import MLPRegressor

from arch import arch_model
from arch.univariate import GARCH, EWMAVariance

import matplotlib.pyplot as plt

"""
This code was part of FINM Portfolio 2022 HW1 Solutions
"""
def performance_summary(return_data):
    """
        Returns the Performance Stats for given set of returns
        Inputs:
            return_data - DataFrame with Date index and Monthly Returns for different assets/strategies.
        Output:
            summary_stats - DataFrame with annualized mean return, vol, sharpe ratio. Skewness, Excess Kurtosis, Var (0.5) and
                            CVaR (0.5) and Max drawdown based on monthly returns.
    """
    summary_stats = return_data.mean().to_frame('Annualized Return').apply(lambda x: x*12)
    summary_stats['Annualized Volatility'] = return_data.std().apply(lambda x: x*np.sqrt(12))
    summary_stats['Annualized Sharpe Ratio'] = summary_stats['Annualized Return']/summary_stats['Annualized Volatility']

    summary_stats['Skewness'] = return_data.skew()
    summary_stats['Excess Kurtosis'] = return_data.kurtosis()
    summary_stats['VaR (0.05)'] = return_data.quantile(.05, axis = 0)
    summary_stats['CVaR (0.05)'] = return_data[return_data <= return_data.quantile(.05, axis = 0)].mean()
    summary_stats['Min'] = return_data.min()
    summary_stats['Max'] = return_data.max()

    wealth_index = 1000*(1+return_data).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    summary_stats['Max Drawdown'] = drawdowns.min()
    summary_stats['Peak'] = [previous_peaks[col][:drawdowns[col].idxmin()].idxmax() for col in previous_peaks.columns]
    summary_stats['Bottom'] = drawdowns.idxmin()

    recovery_date = []
    for col in wealth_index.columns:
        prev_max = previous_peaks[col][:drawdowns[col].idxmin()].max()
        recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin():]]).T
        recovery_date.append(recovery_wealth[recovery_wealth[col] >= prev_max].index.min())
    summary_stats['Recovery'] = recovery_date

    return summary_stats


"""
This code was part of FINM Portfolio 2022 HW1 Solutions
"""
def tangency_portfolio_rfr(asset_return,cov_matrix, cov_diagnolize = False):
    """
        Returns the tangency portfolio weights in a (1 x n) vector
        Inputs:
            asset_return - return for each asset (n x 1) Vector
            cov_matrix = nxn covariance matrix for the assets
    """
    if cov_diagnolize:
        asset_cov = np.diag(np.diag(cov_matrix))
    else:
        asset_cov = np.array(cov_matrix)
    inverted_cov= np.linalg.inv(asset_cov)
    one_vector = np.ones(len(cov_matrix.index))

    den = (one_vector @ inverted_cov) @ (asset_return)
    num =  inverted_cov @ asset_return
    return (1/den) * num


"""
This code was part of FINM Portfolio 2022 HW1 Solutions
"""
def gmv_portfolio(asset_return,cov_matrix):
    """
        Returns the Global Minimum Variance portfolio weights in a (1 x n) vector
        Inputs:
            asset_return - return for each asset (n x 1) Vector
            cov_matrix = nxn covariance matrix for the assets
    """
    asset_cov = np.array(cov_matrix)
    inverted_cov= np.linalg.inv(asset_cov)
    one_vector = np.ones(len(cov_matrix.index))

    den = (one_vector @ inverted_cov) @ (one_vector)
    num =  inverted_cov @ one_vector
    return (1/den) * num


"""
This code was part of FINM Portfolio 2022 HW1 Solutions
"""
def mv_portfolio_rfr(asset_return,cov_matrix,target_ret,tangency_port):
    """
        Returns the Mean-Variance portfolio weights in a (1 x n) vector when a riskless assset is available
        Inputs:
            asset_return - Excess return over the risk free rate for each asset (n x 1) Vector
            cov_matrix = nxn covariance matrix for the assets
            target_ret = Target Return (Annualized)
            tangency_port = Tangency portfolio
    """
    asset_cov = np.array(cov_matrix)
    inverted_cov= np.linalg.inv(asset_cov)
    one_vector = np.ones(len(cov_matrix.index))

    delta_den = (asset_return.T @ inverted_cov) @ (asset_return)
    delta_num = (one_vector @ inverted_cov) @ (asset_return)
    delta_tilde = (delta_num/delta_den) * target_ret
    return (delta_tilde * tangency_port)


"""
This code was part of FINM Portfolio 2022 HW1 Solutions
"""
def mv_portfolio(asset_return,cov_matrix,target_ret):
    """
        Returns the Mean-Variance portfolio weights in a (1 x n) vector when no riskless assset is available
        Inputs:
            asset_return - total return for each asset (n x 1) Vector
            cov_matrix = nxn covariance matrix for the assets
            target_ret = Target Return (Not-Annualized)
    """
    omega_tan = tangency_portfolio_rfr(asset_return.mean(),cov_matrix)
    omega_gmv = gmv_portfolio(asset_return,cov_matrix)

    mu_tan = asset_return.mean() @ omega_tan
    mu_gmv = asset_return.mean() @ omega_gmv

    delta = (target_ret - mu_gmv)/(mu_tan - mu_gmv)
    mv_weights = delta * omega_tan + (1-delta)*omega_gmv
    return mv_weights


"""
This code was part of FINM Portfolio 2022 HW2 Solutions
"""
def regression_based_performance(factor,fund_ret,rf,constant = True):
    """
        Returns the Regression based performance Stats for given set of returns and factors
        Inputs:
            factor - Dataframe containing monthly returns of the regressors
            fund_ret - Dataframe containing monthly excess returns of the regressand fund
            rf - Monthly risk free rate of return
            constant - whether you allow constant for the regression
        Output:
            summary_stats - (Beta of regression, treynor ratio, information ratio, alpha).
    """
    if constant:
        X = sm.tools.add_constant(factor)
    else:
        X = factor
    y=fund_ret
    model = sm.OLS(y,X,missing='drop').fit()

    if constant:
        beta = model.params[1:]
        alpha = round(float(model.params['const']),6)

    else:
        beta = model.params
    treynor_ratio = ((fund_ret.values-rf.values).mean()*12)/beta[0]
    tracking_error = (model.resid.std()*np.sqrt(12))
    if constant:
        information_ratio = model.params[0]*12/tracking_error
    r_squared = model.rsquared
    if constant:
        return (beta,treynor_ratio,information_ratio,alpha,r_squared,tracking_error)
    else:
        return (beta,treynor_ratio,r_squared,tracking_error)


"""
This code was part of FINM Portfolio 2022 HW2 Solutions
"""
def rolling_regression_param(factor,fund_ret,roll_window = 60):
    """
        Returns the Rolling Regression parameters for given set of returns and factors
        Inputs:
            factor - Dataframe containing monthly returns of the regressors
            fund_ret - Dataframe containing monthly excess returns of the regressand fund
            roll_window = rolling window for regression
        Output:
            params - Dataframe with time-t as the index and constant and Betas as columns
    """
    X = sm.add_constant(factor)
    y= fund_ret
    rols = RollingOLS(y, X, window=roll_window)
    rres = rols.fit()
    params = rres.params.copy()
    params.index = np.arange(1, params.shape[0] + 1)
    return params


"""
This code was part of FINM Portfolio 2022 HW5 Solutions
"""
def factors_regression(seriesY,seriesX):
    """
        Returns a dataframe with factors regression statistics
        Inputs:
            seriesY - Assets excess returns series
            seriesX - Factor excess returns series
        Output:
            Dataframe with 7+ columns
                Mean Return - Mean return of each asset
                Sharpe Ratio - Sharpe Ratio of each asset
                R Squared - R squared value of the multiple factors regression
                Alpha - Alpha of the multiple factors regression
                Information ratio - Information ratio of each asset
                Tryenor - Tryenor ratio of each asset
                Betas - Beta for each factor for an asset
    """
    mean =seriesY.mean()*12
    sharpe = mean/(seriesY.std()*(12**0.5))
    model = sm.OLS(seriesY,sm.add_constant(seriesX)).fit()
    rsq = model.rsquared

    beta = pd.DataFrame(index= [seriesY.name])

    for i,x in enumerate(seriesX):
         beta[x] = model.params[i+1]

    betaCols = [i+'Beta' for i in seriesX]
    beta = beta.rename(columns = dict(zip(beta.columns,betaCols)))

    treynor = mean/beta[beta.columns[0]]
    alpha = model.params[0]*12
    information = alpha/(model.resid.std()*np.sqrt(12))

    RegressionStats = pd.DataFrame({'Mean Return':mean,'Sharpe Ratio':sharpe,'R Squared':rsq,\
                         'Alpha':alpha, 'Information Ratio':information, 'Treynor':treynor},index= [seriesY.name])

    return pd.concat([RegressionStats,beta], axis =1)


"""
This code was part of FINM Portfolio 2022 HW7 Solutions
"""
def OOS_r2(df, factors, start):
    """
    Returns out of sample R-Squared value and the regression parameters
    """
    y = df['SPY']
    X = sm.add_constant(factors)

    forecast_err, null_err = [], []

    for i,j in enumerate(df.index):
        if i >= start:
            currX = X.iloc[:i]
            currY = y.iloc[:i]
            reg = sm.OLS(currY, currX, missing = 'drop').fit()
            null_forecast = currY.mean()
            reg_predict = reg.predict(X.iloc[[i]])
            actual = y.iloc[[i]]
            forecast_err.append(reg_predict - actual)
            null_err.append(null_forecast - actual)

    RSS = (np.array(forecast_err)**2).sum()
    TSS = (np.array(null_err)**2).sum()

    return ((1 - RSS/TSS),reg)


"""
This code was part of FINM Portfolio 2022 HW7 Solutions
"""
def OOS_strat(df, factors, start, weight):
    """

    """
    returns = []
    y = df['SPY']
    X = sm.add_constant(factors)

    for i,j in enumerate(df.index):
        if i >= start:
            currX = X.iloc[:i]
            currY = y.iloc[:i]
            reg = sm.OLS(currY, currX, missing = 'drop').fit()
            pred = reg.predict(X.iloc[[i]])
            w = pred * weight
            returns.append((df.iloc[i]['SPY'] * w)[0])

    df_strat = pd.DataFrame(data = returns, index = df.iloc[-(len(returns)):].index, columns = ['Strat Returns'])
    return df_strat


"""
This code was part of FINM Portfolio 2022 HW7 Solutions
"""
def ml_model_predictions(model_cols, returns, signals, return_col, plots = False):
    """

    """
    forecasts_ML = returns.loc[:,[return_col]].expanding().mean().shift(1).dropna()
    forecasts_ML.columns = ['Expanding Mean']

    score_ML = pd.DataFrame(columns=['Expanding Mean'],index=['score'],data=0)

    methods = ['OLS', 'Tree', 'NN']
    est = dict()

    y = returns.loc[:,[return_col]].iloc[1:].squeeze('columns').ravel()
    X = signals.loc[:,model_cols].shift(1).dropna()

    for method in methods:

        if method == 'OLS':
            est[method] = LinearRegression()
        elif method == 'Tree':
            est[method] = RandomForestRegressor(max_depth=3,random_state=1)
        elif method == 'NN':
            est[method] = MLPRegressor(hidden_layer_sizes=500,random_state=1)

        est[method].fit(X,y)
        forecasts_ML[method] = est[method].predict(X)
        score_ML[method] = est[method].score(X,y)

    forecasts_ML.dropna(inplace=True)
    wts_ML = 100 * forecasts_ML

    spy_ML, _ = returns.loc[:,[return_col]].iloc[1:].align(forecasts_ML, join='right', axis=0)

    fund_returns_ML = wts_ML * spy_ML.values
    fund_returns_ML.insert(0,'Passive', spy_ML)

    if plots:
        fn = X.columns
        fig, axes = plt.subplots(nrows = 1,ncols=1, dpi=500);
        tree.plot_tree(est['Tree'].estimators_[0],feature_names = fn, filled=True)
        if len(model_cols) > 1:
            title_name = '-'.join(str(v) for v in model_cols)
        else:
            title_name = model_cols[0]
        plt.title('Signal - '+title_name, fontsize = 20)

    return fund_returns_ML


"""
This code was part of FINM Portfolio 2022 HW7 Solutions
"""
def oos_ml_model_predictions(model_cols, returns, signals, return_col, window = 60):
    """

    """
    methods = ['OLS', 'Tree', 'NN']
    est = dict()

    forecasts_MLOOS = pd.DataFrame(columns=methods,index=returns.iloc[1:].index,dtype='float64')


    y = returns.loc[:,[return_col]].iloc[1:].squeeze('columns').ravel()
    Xlag = signals.loc[:,model_cols].shift(1).dropna()
    X = signals.loc[:,model_cols]

    for method in methods:

        for t in returns.iloc[1:].index[window-1:]:
            yt = returns.loc[:,[return_col]].iloc[1:].loc[:t].values.ravel()
            Xlag_t = Xlag.loc[:t,:].values
            x_t = X.loc[t,:].values.reshape(1,-1)

            if method == 'OLS':
                est = LinearRegression()
            elif method == 'Tree':
                est = RandomForestRegressor(max_depth=3,random_state=1)
            elif method == 'NN':
                est = MLPRegressor(hidden_layer_sizes=500,random_state=1)

            est.fit(Xlag_t,yt);
            predval = est.predict(x_t)[0]
            forecasts_MLOOS.loc[t,method] = predval

    forecasts_MLOOS.insert(0,'Mean', returns.loc[:,[return_col]].expanding().mean().shift(1).dropna())

    # prefer to date forecast by date of forecasted value, not date it was calculated
    forecasts_MLOOS = forecasts_MLOOS.shift(1).dropna()


    wts_MLOOS = 100 * forecasts_MLOOS

    spy_MLOOS, _ = returns.loc[:,[return_col]].iloc[1:].align(forecasts_MLOOS, join='right', axis=0)

    fund_returns_MLOOS = wts_MLOOS * spy_MLOOS.values
    fund_returns_MLOOS.insert(0,'Passive', spy_MLOOS)

    sigma_t = fund_returns_MLOOS.rolling(24).std()
    relative_vols = pd.DataFrame(sigma_t[['Passive']].to_numpy() / sigma_t.drop(columns=['Passive']).to_numpy(),columns=sigma_t.drop(columns=['Passive']).columns, index=sigma_t.index)
    wts_t = relative_vols * wts_MLOOS
    fund_returns_MLOOS = wts_t * spy_MLOOS.values
    fund_returns_MLOOS.insert(0,'Passive', spy_MLOOS)

    fund_returns_MLOOS.dropna(inplace=True)

    null = returns.loc[:,[return_col]].expanding(window+1).mean().shift(1).dropna()
    actual = returns.loc[:,[return_col]].iloc[window+1:]

    forecast_err = pd.DataFrame()
    null_err = pd.DataFrame()
    for col in forecasts_MLOOS.columns:
        forecast_err[col] = forecasts_MLOOS[col] - actual[return_col]
        null_err[col] = null[return_col] - actual[return_col]

    oos_r2 = 1-(((forecast_err**2).sum())/(null_err**2).sum()).to_frame('OOS R-Squared')


    return (fund_returns_MLOOS,oos_r2)