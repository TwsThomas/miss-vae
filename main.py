import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

from metrics import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat
from generate_data import gen_lrmf, ampute, gen_dlvm


def exp_cevae(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):

    from cevea_tf import cevae_tf

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
    X_imp = Imputer().fit_transform(X_miss)
    
    y0_hat, y1_hat = cevae_tf(X_imp, w, y, d_cevea=d_cevae,
                             n_epochs=n_epochs)

    # Tau estimated on Zhat=E[Z|X]
    ps_hat = np.ones(len(y0_hat)) / 2
    # res_tau_ols = tau_ols(zhat, w, y)
    # res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
    res_tau_dr_true_ps = tau_dr(y, w, y0_hat, y1_hat, ps, method)

    return res_tau_dr, res_tau_dr_true_ps


def exp_miwae(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1,
		method="glm", **kwargs):

    from miwae import miwae

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
                                 n_epochs=n_epochs)

    # print('shape of outputs miwae:')
    # print('xhat.shape, zhat.shape, zhat_mul.shape:') 
    #    (1000, 200) (1000, 3) (200, 1000, 3)
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    # Tau estimated on Zhat=E[Z|X]
    ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
    res_tau_ols = tau_ols(zhat, w, y)
    res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = []
    res_mul_tau_ols = []
    res_mul_tau_ols_ps = []
    for zhat_b in zhat_mul: 
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
        res_mul_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat, method))
        res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
        res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))

    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps


if __name__ == '__main__':

    print('test exp with default arguments on miwae')
    tau = exp_miwae(n_epochs = 3)
    print('Everything went well.')