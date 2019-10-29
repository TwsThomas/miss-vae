import numpy as np
import pandas as pd


from metrics import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat, tau_mi
from generate_data import gen_lrmf, ampute, gen_dlvm
from dcor import dcor
from softimpute import get_U_softimpute

from joblib import Memory
memory = Memory('cache_dir', verbose=0)


@memory.cache
def exp_baseline(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, citcio = False, seed=0,
        d_cevae=20, n_epochs=402, full_baseline=False,
		method="glm", **kwargs):

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
    
        
    from sklearn.impute import SimpleImputer
    X_imp_mean = SimpleImputer().fit_transform(X_miss)
        
    
    Z_perm = np.random.permutation(Z)
    # Z_rnd = np.random.randn(Z.shape[0], Z.shape[1])

    algo_name = ['Z', 'X', 'X_imp_mean']
    algo_ = [Z, X, X_imp_mean]

    if full_baseline:
        # complete the baseline 
        U_soft = get_U_softimpute(X_miss)
            # need try-except for sklearn version
        try:
            from sklearn.impute import IterativeImputer
            X_imp_mice = IterativeImputer().fit_transform(X_miss)
        except:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            X_imp_mice = IterativeImputer().fit_transform(X_miss)

        algo_name += ['X_imp_mice','U_soft', 'Z_perm']
        algo_ += [X_imp_mice, U_soft, Z_perm]

    tau = dict()
    for name, zhat in zip(algo_name, algo_):
        
        if name == 'X_mi':
            res_tau_dr, res_tau_ols, res_tau_ols_ps = tau_mi(zhat, w, y, method=method)
        
        else:
            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
            res_tau_ols = tau_ols(zhat, w, y)
            res_tau_ols_ps = tau_ols_ps(zhat, w, y)
            res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
        
        
        tau[name] = res_tau_dr, res_tau_ols, res_tau_ols_ps

    return tau

def exp_mi(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, citcio = False, seed=0, m = 10,
        d_cevae=20, n_epochs=402,
        method="glm", **kwargs):

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    tau_dr_mi, tau_ols_mi, tau_ols_ps_mi = tau_mi(X_miss, w, y, m = m, method = method)
    
    return tau_dr_mi, tau_ols_mi, tau_ols_ps_mi

def exp_cevae(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, citcio = False, seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):

    # import here because of differents sklearn version used
    from cevae_tf import cevae_tf
    from sklearn.preprocessing import Imputer

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
    X_imp = Imputer().fit_transform(X_miss)
    
    y0_hat, y1_hat = cevae_tf(X_imp, w, y, d_cevae=d_cevae,
                             n_epochs=n_epochs)

    # Tau estimated on Zhat=E[Z|X]
    ps_hat = np.ones(len(y0_hat)) / 2
    # res_tau_ols = tau_ols(zhat, w, y)
    # res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
    res_tau_dr_true_ps = tau_dr(y, w, y0_hat, y1_hat, ps, method)

    return res_tau_dr, res_tau_dr_true_ps


def exp_miwae(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, citcio = False, seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1, add_wy = False,
		method="glm", **kwargs):

    from miwae import miwae

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, citcio = citcio, prop_miss = prop_miss, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    if add_wy:
        xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
                                     n_epochs=n_epochs, add_wy = add_wy, w=w, y=y)
    else:
        xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
                                     n_epochs=n_epochs, add_wy = add_wy)

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
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat_b, w, y)
        res_mul_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat, method))
        res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
        res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))

    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)

    if Z.shape[1] == zhat.shape[1]:
        dcor_zhat = dcor(Z, zhat)

    dcor_zhat_mul = []
    for zhat_b in zhat_mul: 
        dcor_zhat_mul.append(dcor(Z, zhat_b))
    dcor_zhat_mul = np.mean(dcor_zhat_mul) 

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps, dcor_zhat, dcor_zhat_mul


if __name__ == '__main__':

    from config import args
    args['m'] = 2
    args['n_epochs'] = 3

    print('test exp with default arguments on miwae')
    tau = exp_miwae(**args)
    print('Everything went well.')

    for args['citcio'] in [True, False]:
        print('test exp with default arguments on mi with citcio=', args['citcio'])
        tau = exp_mi(**args)
        print(tau)
        print('Everything went well.')

    print('showing baseline :::')
    from baseline import get_baseline
    args['full_baseline'] = True
    df_base = get_baseline(show=True, **args)