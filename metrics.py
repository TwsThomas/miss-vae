# -*- coding: utf-8 -*-
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def get_ps_y01_hat(zhat, w, y):
    # predict with LR

    n,_ = zhat.shape
    lr = LogisticRegression()
    lr.fit(zhat, w)
    ps_hat = lr.predict_proba(zhat)[:,1]  

    lr = LinearRegression()
    lr.fit(zhat[np.equal(w, np.ones(n)),:], y[np.equal(w, np.ones(n))])
    y1_hat = lr.predict(zhat)
    
    lr = LinearRegression()
    lr.fit(zhat[np.equal(w, np.zeros(n)),:], y[np.equal(w, np.zeros(n))])
    y0_hat = lr.predict(zhat)

    return ps_hat, y0_hat, y1_hat


def tau_dr(y, w, y0_hat, y1_hat, ps_hat, method = "glm"):
    """Doubly robust ATE estimation
    if method == "glm": provide fitted values y1_hat, y0_hat and ps_hat
                        for the two response surfaces and the propensity scores respectively
    if method == "grf": no need to provide any fitted values"""

    assert y0_hat.shape == y.shape
    assert y1_hat.shape == y.shape
    assert w.shape == y.shape

    if method == "glm":
        tau_i = y1_hat - y0_hat + w*(y-y1_hat)/ps_hat -\
            					(1-w)*(y-y0_hat)/(1-ps_hat)
        tau = np.mean(tau_i)
    elif method == "grf":
    	raise NotImplementedError("Causal forest estimation not implemented here yet.")
    else:
        raise ValueError("'method' should be choosed between 'glm' and 'grf' in 'tau_dr', got %s", method)
    return tau


def tau_ols(Z_hat, w, y):
    # ATE estimation via OLS regression 

    ZW = np.concatenate((Z_hat, w.reshape((-1,1))), axis=1)
    lr = LinearRegression()
    lr.fit(ZW, y)
    tau = lr.coef_[-1]

    return tau


def tau_ols_ps(zhat, w, y):
    # Difference with linear_tau: add estimated propensity 
    # scores as additional predictor

    lr = LogisticRegression()
    lr.fit(zhat, w)
    ps_hat = lr.predict_proba(zhat)

    ZpsW = np.concatenate((zhat,ps_hat, w.reshape((-1,1))), axis=1)
    lr = LinearRegression()
    lr.fit(ZpsW, y)
    tau = lr.coef_[-1]

    return tau