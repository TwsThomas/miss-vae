# -*- coding: utf-8 -*-
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import grf

def tau_mi(xmiss, w, y, m = 10, method = "glm"):
    """ATE estimation via multiple imputation of incomplete covariates
    if method == "glm": computes DR, OLS and OLS with ps estimators on m imputed datasets and 
                        for every method, aggreggates the m estimations into one ATE estimator
    if method == "grf": computes DR with grf package on every imputed dataset and 
                        aggreggates the m estimations into one ATE estimator"""
    res_tau_dr = []
    res_tau_ols = []
    res_tau_ols_ps = []
    for i in range(m):
        imp = IterativeImputer(sample_posterior=True, random_state = i)
        ximp_mice = imp.fit_transform(xmiss)
        if method == "glm":
            print(ximp_mice.shape)
            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(ximp_mice, w, y)
            res_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat, method = method))
            res_tau_ols.append(tau_ols(ximp_mice, w, y))
            res_tau_ols_ps.append(tau_ols_ps(ximp_mice, w, y))
        else:
            res_tau_dr.append(tau_dr(y, w, confounders = ximp_mice, method = method))

    if method == "glm":
        return np.mean(res_tau_dr), np.mean(res_tau_ols), np.mean(res_tau_ols_ps)
    else:
        return np.mean(res_tau_dr), None, None

def tau_mia(xmiss, w, y):
    return None

def tau_grf(x, w, y):
    return None


def get_ps_y01_hat(zhat, w, y):
    # predict with LR
    w = w.reshape((-1,))
    y = y.reshape((-1,))
    n,_ = zhat.shape
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(zhat, w)
    ps_hat = lr.predict_proba(zhat)[:,1]  

    lr = LinearRegression()
    lr.fit(zhat[np.equal(w, np.ones(n)),:], y[np.equal(w, np.ones(n))])
    y1_hat = lr.predict(zhat)
    
    lr = LinearRegression()
    lr.fit(zhat[np.equal(w, np.zeros(n)),:], y[np.equal(w, np.zeros(n))])
    y0_hat = lr.predict(zhat)

    y0_hat = y0_hat.reshape((-1,1))
    y1_hat = y1_hat.reshape((-1,1))
    return ps_hat, y0_hat, y1_hat


def tau_dr(y, w, y0_hat=None, y1_hat=None, ps_hat=None, confounders = None, method = "glm"):
    """Doubly robust ATE estimation
    if method == "glm": provide fitted values y1_hat, y0_hat and ps_hat
                        for the two response surfaces and the propensity scores respectively
    if method == "grf": no need to provide any fitted values but need to provide confounders matrix"""
    y = y.reshape((-1,))
    y0_hat = y0_hat.reshape((-1,))
    y1_hat = y1_hat.reshape((-1,))
    w = w.reshape((-1,))
    assert y0_hat.shape == y.shape
    assert y1_hat.shape == y.shape
    assert w.shape == y.shape

    if method == "glm":
        tau_i = y1_hat - y0_hat + w*(y-y1_hat)/np.maximum(1e-12, ps_hat) -\
            					(1-w)*(y-y0_hat)/np.maximum(1e-12,(1-ps_hat))
        tau = np.mean(tau_i)
    elif method == "grf":
    	raise NotImplementedError("Causal forest estimation not implemented here yet.")
    else:
        raise ValueError("'method' should be choosed between 'glm' and 'grf' in 'tau_dr', got %s", method)
    return tau


def tau_ols(Z_hat, w, y):
    # ATE estimation via OLS regression 

    assert w.shape == y.shape

    y = y.reshape((-1,))
    ZW = np.concatenate((Z_hat, w.reshape((-1,1))), axis=1)
    lr = LinearRegression()
    lr.fit(ZW, y)
    tau = lr.coef_[-1]

    return tau


def tau_ols_ps(zhat, w, y):
    # Difference with linear_tau: add estimated propensity 
    # scores as additional predictor

    assert w.shape == y.shape
    w = w.reshape((-1,))
    y = y.reshape((-1,))
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(zhat, w)
    ps_hat = lr.predict_proba(zhat)

    ZpsW = np.concatenate((zhat,ps_hat, w.reshape((-1,1))), axis=1)
    lr = LinearRegression()
    lr.fit(ZpsW, y)
    tau = lr.coef_[-1]

    return tau