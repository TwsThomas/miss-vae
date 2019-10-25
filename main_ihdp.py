import numpy as np
import pandas as pd


from metrics import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat, tau_mi
from generate_data import ampute
from dcor import dcor

from joblib import Memory
memory = Memory('cache_dir', verbose=0)


@memory.cache
def ihdp_baseline(set_id=1, prop_miss=0.1, seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):
    
    X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])

    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
    
    X_imp_mean = np.zeros(X_miss.shape)
    X_imp_mice = np.zeros(X_miss.shape)
    try:
        from sklearn.impute import SimpleImputer
        X_imp_mean = SimpleImputer().fit_transform(X_miss)
    except:
        pass
    try:
        from sklearn.impute import IterativeImputer
        X_imp_mice = IterativeImputer()().fit_transform(X_miss)
    except:
        pass
    

    tau = dict()
    for name, zhat in zip(['X', 'X_imp_mean'],#, 'X_imp_mice', 'Z_perm'],#, 'X_mi'],
                          [X, X_imp_mean]):#, X_imp_mice, Z_perm]):#, X_miss]):
        
        if name == 'X_mi':
            res_tau_dr, res_tau_ols, res_tau_ols_ps = tau_mi(zhat, w, y, method=method)
        
        else:
            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
            res_tau_ols = tau_ols(zhat, w, y)
            res_tau_ols_ps = tau_ols_ps(zhat, w, y)
            res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
        
        
        tau[name] = res_tau_dr, res_tau_ols, res_tau_ols_ps

    return tau

def ihdp_mi(set_id = 1, prop_miss=0.1, seed=0, m = 10,
        d_cevae=20, n_epochs=402,
        method="glm", **kwargs):

    X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
        

    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)


    tau_dr_mi, tau_ols_mi, tau_ols_ps_mi = tau_mi(X_miss, w, y, m = m, method = method)
    
    return tau_dr_mi, tau_ols_mi, tau_ols_ps_mi

def ihdp_cevae(set_id = 1, prop_miss=0.1, seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):

    # import here because of differents sklearn version used
    from cevae_tf import cevae_tf
    from sklearn.preprocessing import Imputer

    X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
        
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


def ihdp_miwae(set_id = 1, prop_miss=0.1, seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1, add_wy = False,
		method="glm", **kwargs):

    from miwae import miwae

    X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    if add_wy:
        xhat, zhat, zhat_mul = miwae(X_miss, d_miwae=d_miwae, sig_prior = sig_prior,
                                     n_epochs=n_epochs, add_wy = add_wy, w=w, y=y)
    else:
        xhat, zhat, zhat_mul = miwae(X_miss, d_miwae=d_miwae, sig_prior = sig_prior,
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

    dcor_zhat = np.nan

    dcor_zhat_mul = np.nan

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps, dcor_zhat, dcor_zhat_mul


if __name__ == '__main__':

    #print('test exp with default arguments on miwae')
    #tau = exp_miwae(n_epochs = 3)
    #print('Everything went well.')


    #print('test ihdp with default arguments on mi')
    #tau = ihdp_mi(m=2)
    # print('test ihdp with baseline')
    # tau = ihdp_baseline()
    print('test ihdp with miwae')
    tau = ihdp_miwae(n_epochs=2)
    print(tau)
    X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(1) + '.csv')
    print('y1-y0:', np.mean((X.iloc[:,0]==1)*(X.iloc[:,1]  - X.iloc[:,2]) +(X.iloc[:,0]==0)*(X.iloc[:,2]  - X.iloc[:,1])))
    print('mu1-mu0:', np.mean(X.iloc[:,4]  - X.iloc[:,3]))

    print('Everything went well.')

