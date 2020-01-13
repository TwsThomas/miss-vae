import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from metrics import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat, tau_residuals
from generate_data import ampute
from dcor import dcor
from softimpute import get_U_softimpute

#from joblib import Memory
#memory = Memory('cache_dir', verbose=0)


#@memory.cache
def tb_baseline(seed=0,
        full_baseline = False, add_wy = False, sig_prior = 1, d_miwae = 10, n_epochs = 10,
		method="glm", **kwargs):
    
    X = pd.read_csv('./data/tb/tb_tbi_17conf.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,2:])

    
    X_imp_mean = np.zeros(X.shape)
    X_imp_mice = np.zeros(X.shape)
    try:
        from sklearn.impute import SimpleImputer
        X_imp_mean = SimpleImputer().fit_transform(X)
    except:
        pass
    try:
        from sklearn.impute import IterativeImputer
        X_imp_mice = IterativeImputer()().fit_transform(X)
    except:
        pass
    

    algo_name = ['X_imp_mean']
    algo_ = [X_imp_mean]

    if full_baseline:
        # complete the baseline 
        Z_mf = get_U_softimpute(X_miss)
            # need try-except for sklearn version
        try:
            from sklearn.impute import IterativeImputer
            X_imp = IterativeImputer().fit_transform(X_miss)
        except:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            X_imp = IterativeImputer().fit_transform(X_miss)

        algo_name += ['X_imp','Z_mf']#, 'Z_perm']
        algo_ += [X_imp, Z_mf]#, Z_perm]

    tau = dict()
    for name, zhat in zip(algo_name, algo_):
        
        if name == 'X_mi':
            res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid = tau_mi(zhat, w, y, method=method)
        
        else:
            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
            res_tau_ols = tau_ols(zhat, w, y)
            res_tau_ols_ps = tau_ols_ps(zhat, w, y)
            res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
            lr = LinearRegression()
            lr.fit(zhat, y)
            y_hat = lr.predict(zhat)
            res_tau_resid = tau_residuals(y, w, y_hat, ps_hat, method)
        
        tau[name] = res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid

    return tau

def tb_mi(seed=0, m = 10,
        d_cevae=20, n_epochs=402,
        method="glm", **kwargs):
    from metrics import tau_mi

    X = pd.read_csv('./data/tb/tb_tbi_17conf.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,2:])
        

    tau_dr_mi, tau_ols_mi, tau_ols_ps_mi, tau_resid_mi = tau_mi(X, w, y, m = m, method = method)
    
    return tau_dr_mi, tau_ols_mi, tau_ols_ps_mi

def tb_cevae(seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):

    # import here because of differents sklearn version used
    from cevae_tf import cevae_tf
    from sklearn.preprocessing import Imputer

    X = pd.read_csv('./data/tb/tb_tbi_17conf.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,2:])
    
    X_imp = Imputer().fit_transform(X)

    y0_hat, y1_hat = cevae_tf(X_imp, w, y, d_cevae=d_cevae,
                             n_epochs=n_epochs)

    # Tau estimated on Zhat=E[Z|X]
    #ps_hat = np.ones(len(y0_hat)) / 2
    # res_tau_ols = tau_ols(zhat, w, y)
    # res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    #res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
    #res_tau_dr_true_ps = tau_dr(y, w, y0_hat, y1_hat, ps, method)

    res_tau = np.mean(y1_hat - y0_hat)
    return res_tau


def tb_miwae(seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1, add_wy = False,
		method="glm", **kwargs):

    from miwae import miwae

    X = pd.read_csv('./data/tb/tb_tbi_17conf.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,2:])
        
    
    if add_wy:
        xhat, zhat, zhat_mul = miwae(X, d_miwae=d_miwae, sig_prior = sig_prior,
                                     n_epochs=n_epochs, add_wy = add_wy, w=w, y=y)
    else:
        xhat, zhat, zhat_mul = miwae(X, d_miwae=d_miwae, sig_prior = sig_prior,
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

    # lr = LinearRegression()
    # lr.fit(zhat, y)
    # y_hat = lr.predict(zhat)
    # res_tau_resid = tau_residuals(y, w, y_hat, ps_hat, method)

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = []
    res_mul_tau_ols = []
    res_mul_tau_ols_ps = []
    res_mul_tau_resid = []
    for zhat_b in zhat_mul: 
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat_b, w, y)
        res_mul_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat, method))
        res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
        res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))
        # lr = LinearRegression()
        # lr.fit(zhat, y)
        # y_hat = lr.predict(zhat_b)
        # res_mul_tau_resid = tau_residuals(y, w, y_hat, ps_hat, method)

    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
    # res_mul_tau_resid = np.mean(res_mul_tau_resid)

    dcor_zhat = np.nan

    dcor_zhat_mul = np.nan

    # return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps, res_mul_tau_resid, dcor_zhat, dcor_zhat_mul
    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps, dcor_zhat, dcor_zhat_mul


if __name__ == '__main__':

    #print('test exp with default arguments on miwae')
    #tau = exp_miwae(n_epochs = 3)
    #print('Everything went well.')


    # print('test tb with default arguments on mi')
    # tau = tb_mi(m=2)
    # print(tau)
    # print('test tb with baseline')
    # tau = tb_baseline()
    # print(tau)
    # print('test tb with miwae')
    # tau = tb_miwae(n_epochs=2, sig_prior=0.01, d_miwae = 1)
    # print(tau)
    print('test tb with cevae')
    tau = tb_cevae(n_epochs=2, d_cevae=10)
    print(tau)
    
    print('Everything went well.')

