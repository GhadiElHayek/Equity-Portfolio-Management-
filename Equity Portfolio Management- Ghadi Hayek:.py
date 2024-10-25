Lab1 main script:
# %%
import sys
sys.path.append('/Users/ghadielhayek/Desktop/Lab 1-20240505')  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import epm_kit  # Import your custom toolkit

# File paths for datasets
hedge_funds_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/hedge_funds_dataset.csv'
prices_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/sample_prices_dataset.csv'
portfolios_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/Portfolios_Formed_on_ME_dataset.csv'

# Loading the data
prices = pd.read_csv(prices_path, index_col=0)
portfolios = pd.read_csv(portfolios_path, index_col=0)

# Process prices data
prices_returns = prices.pct_change().dropna()
annualized_return_prices = (1 + prices_returns).prod()**(12 / prices_returns.shape[0]) - 1
annualized_vol_prices = prices_returns.std() * (12 ** 0.5)

print("Annualized Return for Prices:", annualized_return_prices)
print("Annualized Volatility for Prices:", annualized_vol_prices)

# Plot Price Returns and Prices
prices_returns.plot.bar(title="Price Returns")
plt.show()
prices.plot(title="Prices")
plt.show()

# Process portfolio data
returns = portfolios[['Lo 30', 'Hi 30']] / 100
returns.columns = ['SmallCap', 'LargeCap']
returns.plot.line(title="Returns of SmallCap and LargeCap")
plt.show()

# Calculate Annualized Returns and Sharpe Ratio for Portfolios
annualized_return_portfolio = (1 + returns).prod()**(12 / returns.shape[0]) - 1
sharpe_ratio_portfolio = epm_kit.sharpe_ratio(returns, 0.03, 12)
print("Annualized Returns for Portfolio:", annualized_return_portfolio)
print("Sharpe Ratio:", sharpe_ratio_portfolio)

# Maximum Drawdown Calculation using epm_kit
drawdowns = epm_kit.drawdown(returns['LargeCap'])
drawdowns['Drawdown'].plot(title="Drawdown for LargeCap")
plt.show()

# Print Extreme Risk Estimates
print("Skewness of LargeCap:", epm_kit.skewness(returns['LargeCap']))
print("Kurtosis of LargeCap:", epm_kit.kurtosis(returns['LargeCap']))

# Historic VaR and CVaR for LargeCap
print("Historic VaR for LargeCap:", epm_kit.var_historic(returns['LargeCap']))
print("Historic CVaR for LargeCap:", epm_kit.cvar_historic(returns['LargeCap']))

Lab 1 epm_kit:
import pandas as pd
import numpy as np
from scipy.stats import norm

# Function Definitions

def drawdown(return_series: pd.Series):
    """
    Computes the drawdown series for a set of asset returns.
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdowns})

def skewness(r):
    """
    Computes the skewness of the supplied Series or DataFrame.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r**3).mean() / sigma_r**3

def kurtosis(r):
    """
    Computes the kurtosis of the supplied Series or DataFrame.
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    return (demeaned_r**4).mean() / sigma_r**4

def var_historic(r, level=5):
    """
    Returns the historical Value at Risk at a specified level.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR (CVaR) of a Series or DataFrame.
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)

Lab 2 main script:

# Main Lab 2 Script
import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Append your local directory containing `epm_kit`
sys.path.append('/Users/ghadielhayek/Desktop/Lab 1-20240505')

# Reload `epm_kit` to ensure it uses the most updated version
import epm_kit
importlib.reload(epm_kit)

# Load data
industries_data_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/30_Industry_Portfolios.CSV'
ind = epm_kit.get_ind_returns()  # Using the function from `epm_kit`

# Plot drawdown for the 'Food' industry
drawdowns = epm_kit.drawdown(ind["Food"])
drawdowns['Drawdown'].plot.line(figsize=(12, 6))
plt.title('Drawdown for Food Industry')
plt.show()

# Analyze risk with Sharpe ratios
sharpe_ratios = epm_kit.sharpe_ratio(ind, 0.03, 12)
sharpe_ratios.sort_values().plot.bar(title="Industry Sharpe Ratios 1926-2018", color="green")
plt.show()

# Efficient Frontier for selected industries
cols_of_interest = ["Food", "Smoke", "Coal", "Beer", "Fin"]
annualized_rets = epm_kit.annualize_rets(ind[cols_of_interest], 12)
cov_matrix = ind[cols_of_interest].cov()
epm_kit.plot_ef(50, annualized_rets, cov_matrix, show_cal=True, riskfree_rate=0.03)
plt.show()

# Max Sharpe Ratio (MSR) and Capital Allocation Line (CAL)
l = ["Games", "Fin"]
msr_weights = epm_kit.msr(0.1, epm_kit.annualize_rets(ind[l], 12), ind[l].cov())
msr_return = epm_kit.portfolio_return(msr_weights, epm_kit.annualize_rets(ind[l], 12))
msr_volatility = epm_kit.portfolio_vol(msr_weights, ind[l].cov())

plt.scatter([msr_volatility], [msr_return], color='red', marker='*', s=200)  # Mark MSR portfolio
plt.show()

# Plot efficient frontier for 'Games' and 'Fin'
epm_kit.plot_ef2(25, epm_kit.annualize_rets(ind[l], 12), ind[l].cov(), style=".")
plt.show()

Lab 2 epm_kit:
epm_kit.py lab 2
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load and format data
def get_ind_returns():
    ind = pd.read_csv("30_Industry_Portfolios.csv", header=0, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

# Annualize returns
def annualize_rets(r, periods_per_year):
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1

# Annualize volatility
def annualize_vol(r, periods_per_year):
    return r.std() * (periods_per_year ** 0.5)

# Calculate Sharpe ratio
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

# Portfolio return
def portfolio_return(weights, returns):
    return weights.T @ returns

# Portfolio volatility
def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights) ** 0.5

# Two-asset efficient frontier
def plot_ef2(n_points, er, cov, style=".-"):
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=style)

# Minimize volatility for a given target return
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    result = minimize(portfolio_vol, init_guess, args=(cov,), method='SLSQP', options={'disp': False}, constraints=[return_is_target, weights_sum_to_1], bounds=bounds)
    return result.x

# Optimal weights for different target returns
def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    return [minimize_vol(target_return, er, cov) for target_return in target_rs]

# Multi-asset efficient frontier
def plot_ef(n_points, er, cov, show_cal=False, style='.-', riskfree_rate=0):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)
    if show_cal:
        ax.set_xlim(left=0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        cal_x = [0, vol_msr]
        cal_y = [riskfree_rate, r_msr]
        ax.plot(cal_x, cal_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    return ax

# Maximum Sharpe Ratio (MSR) portfolio
def msr(riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol
    result = minimize(neg_sharpe, init_guess, args=(riskfree_rate, er, cov), method='SLSQP', options={'disp': False}, constraints=[weights_sum_to_1], bounds=bounds)
    return result.x

Lab 3 main script: 
import sys
import pandas as pd
import importlib
import epm_kit as epm

sys.path.append('/Users/ghadielhayek/Desktop/Lab 1-20240505')

# Paths to datasets
industries_data_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/30_Industry_Portfolios.csv'
industries_nfirms_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/30_Industry_Portfolios_nfirms.csv'
industries_size_path = '/Users/ghadielhayek/Desktop/Lab 1-20240505/30_Industry_Portfolios_size.csv'

# Load data using `epm_kit` functions
ind_return = epm.get_ind_returns()
ind_nfirms = epm.get_ind_nfirms()
ind_size = epm.get_ind_size()

# Cap-Weighted Market Index Construction
ind_mktcap = ind_nfirms * ind_size
total_mktcap = ind_mktcap.sum(axis="columns")
ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")

# Cap-Weighted Market Index Returns
tmi_return = (ind_capweight * ind_return).sum(axis="columns")
total_market_index = epm.drawdown(tmi_return).Wealth

# Rolling Windows and Correlations
total_market_index["1980":].plot(figsize=(12, 6))
total_market_index["1980":].rolling(window=36).mean().plot()

tmi_tr36rets = tmi_return.rolling(window=36).aggregate(epm.annualize_rets, periods_per_year=12)
tmi_tr36rets.plot(figsize=(12, 5), label="Tr 36 mo Returns", legend=True)
tmi_return.plot(label="Returns", legend=True)

ts_corr = ind_return.rolling(window=36).corr()
ts_corr.index.names = ['date', 'industry']
ind_tr36corr = ts_corr.groupby(level='date').apply(lambda cormat: cormat.values.mean())

tmi_tr36rets.plot(secondary_y=True, legend=True, label="Tr 36 mo return", figsize=(12, 6))
ind_tr36corr.plot(legend=True, label="Tr 36 mo Avg Correlation")

lab 3 epm_kit: 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Function to load and format industry returns
def get_ind_returns():
    ind = pd.read_csv("30_Industry_Portfolios.csv", header=0, index_col=0) / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

# Load the industry firms data
def get_ind_nfirms():
    ind = pd.read_csv("30_Industry_Portfolios_nfirms.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

# Load the industry size data
def get_ind_size():
    ind = pd.read_csv("30_Industry_Portfolios_size.csv", header=0, index_col=0)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

# Function to annualize returns
def annualize_rets(r, periods_per_year):
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1

# Function to annualize volatility
def annualize_vol(r, periods_per_year):
    return r.std() * (periods_per_year**0.5)

# Function to calculate Sharpe ratio
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

# Function to calculate portfolio returns
def portfolio_return(weights, returns):
    return weights.T @ returns

# Function to calculate portfolio volatility
def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5

# Function to minimize volatility
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: target_return - portfolio_return(weights, er)}]
    result = minimize(portfolio_vol, init_guess, args=(cov,), method='SLSQP', options={'disp': False}, bounds=bounds, constraints=constraints)
    return result.x

# Function to implement CPPI strategy
def run_cppi(risky_r, safe_r=None, drawdown=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12

    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)

        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.clip(risky_w, 0, 1)
        safe_w = 1 - risky_w

        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w

        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])

        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }

    return backtest_result

# Summary statistics function
def summary_stats(r, riskfree_rate=0.03):
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": ann_sr
    })

# Simulate Geometric Brownian Motion
def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=(1 + mu)**dt, scale=(sigma * np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    return s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1 - 1

# Function to show GBM
def show_gbm(n_scenarios, mu, sigma):
    s_0 = 100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(12, 5))
    ax.axhline(y=100, ls=":", color="black")
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)

# Function to show CPPI
def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0.0, riskfree_rate=0.03, steps_per_year=12, y_max=100):
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    btr = run_cppi(risky_r=pd.DataFrame(risky_r), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr["Wealth"]

    y_max = wealth.values.max() * y_max / 100
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start * floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures / n_scenarios

    e_shortfall = np.dot(terminal_wealth - start * floor, failure_mask) / n_failures if n_failures > 0 else 0.0

    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)

    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start * floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)

    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85), xycoords='axes fraction', fontsize=24)

    if floor > 0.01:
        hist_ax.axhline(y=start * floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)

