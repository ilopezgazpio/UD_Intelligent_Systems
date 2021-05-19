# Estimating Euler Equation with Generalized Method of Moments with statsmodels
# from https://nbviewer.jupyter.org/gist/josef-pkt/6895915

import numpy as np
import pandas as pd

# conda install -c conda-forge statsmodels
# pip3 install statsmodels

from statsmodels.sandbox.regression import gmm
url = "https://raw.githubusercontent.com/lgazpio/UD_Intelligent_Systems/master/Datasets/consumption.csv"
data = pd.read_csv(url, parse_dates=[0])

data.size
data.shape
data.describe()
print(data.iloc[:5])

# create the lagged and leading variables for the estimation.
# As instruments we use lagged interest rate and current and lagged consumption growth

data['c_growth'] = data['c'] / data['c'].shift(1)
data['c_growth_lag1'] = data['c_growth'].shift(1)
data['r_lag1'] = data['r'].shift(1)
data['r_lag2'] = data['r'].shift(2)
data['r_forw1'] = data['r'].shift(-1)
data['c_lag1'] = data['c'].shift(1)
data['c_forw1'] = data['c'].shift(-1)
data['const'] = 1

data_clean = data.dropna()

endog_df = data_clean[['r_forw1', 'c_forw1', 'c']]
exog_df = endog_df

instrument_df = data_clean[['r_lag1', 'r_lag2', 'c_growth', 'c_growth_lag1','const']]

endog, exog, instrument = map(np.asarray, [endog_df, exog_df, instrument_df])


# Currently statsmodels has two ways of specifying moment conditions.
# The first uses general non-linear functions for the (unconditional) moment condition
# The second version uses an instrumental variables approach with additive error structure

def moment_consumption1(params, exog):
    beta, gamma = params
    r_forw1, c_forw1, c = exog.T  # unwrap iterable (ndarray)

    # moment condition without instrument
    err = 1 - beta * (1 + r_forw1) * np.power(c_forw1 / c, -gamma)
    return -err

endog1 = np.zeros(exog.shape[0])
mod1 = gmm.NonlinearIVGMM(endog1, exog, instrument, moment_consumption1, k_moms=4)
w0inv = np.dot(instrument.T, instrument) / len(endog1)
res1 = mod1.fit([1,-1], maxiter=2, inv_weights=w0inv)

print(res1.summary(yname='Euler Eq', xname=['discount', 'CRRA']))

# We don't need Nelder-Mead in this case, we can use bfgs default directly
# res1_ = mod1.fit([1,-1], maxiter=0, inv_weights=w0inv, opt_method='nm')

res1_hac4_2s = mod1.fit([1, -1], maxiter=2, inv_weights=w0inv, weights_method='hac', wargs={'maxlag':4})
print(res1_hac4_2s.summary(yname='Euler Eq', xname=['discount', 'CRRA']))



def moment_consumption2(params, exog):
    beta, gamma = params
    #endog, exog = args
    r_forw1, c_forw1, c = exog.T  # unwrap iterable (ndarray)

    # 2nd part of moment condition without instrument
    predicted = beta * (1. + r_forw1) * np.power(c_forw1 / c, -gamma)
    return predicted

endog2 = np.ones(exog.shape[0])
mod2 = gmm.NonlinearIVGMM(endog2, exog, instrument, moment_consumption2, k_moms=4)
w0inv = np.dot(instrument.T, instrument) / len(endog2)
res2_hac4_2s = mod2.fit([1,-1], maxiter=2, inv_weights=w0inv, weights_method='hac', wargs={'maxlag':4})

print(res2_hac4_2s.summary(yname='Euler Eq', xname=['discount', 'CRRA']))


res1_hac4_2s.params
res2_hac4_2s.params
res1_hac4_2s.params - res2_hac4_2s.params, np.max(np.abs(res1_hac4_2s.params - res2_hac4_2s.params))
# Stata manual has params [0.9205, -4.2224] and standard errors equal to [0.0135, 1.4739].
# Stata doesn't center the moments by default.
# We can get something closer to the results of Stata by using centered = False as a weights argument:

res_ = mod2.fit([1,-1], maxiter=2, inv_weights=w0inv, weights_method='hac',
                wargs={'maxlag':4, 'centered':False}, optim_args={'disp':0})
print(res_.params)
print(res_.bse)

# more examples on
# https://github.com/josef-pkt/misc/blob/master/notebooks/ex_gmm_gamma.ipynb
# https://stackoverflow.com/questions/49272902/issue-with-using-statsmodels-sandbox-regression-gmm-gmm
