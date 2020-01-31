import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from metrics import tau_dr, tau_ols, tau_ols_ps, get_ps_y01_hat, tau_residuals
from generate_data import ampute
from dcor import dcor
from softimpute import get_U_softimpute

from joblib import Memory
memory = Memory('cache_dir', verbose=0)


@memory.cache
def ihdp_baseline(set_id=1, prop_miss=0.1, seed=0,
        full_baseline = False, add_wy = False, sig_prior = 1, d_miwae = 10, n_epochs = 10,
		method="glm", **kwargs):
    
    X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
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
    

    algo_name = ['X', 'X_imp_mean']
    algo_ = [X, X_imp_mean]

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

def ihdp_mi(set_id = 1, prop_miss=0.1, seed=0, m = 10,
        d_cevae=20, n_epochs=402,
        method="glm", **kwargs):
    from metrics import tau_mi

    X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
        
    print(prop_miss)
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)


    tau_dr_mi, tau_ols_mi, tau_ols_ps_mi, tau_resid_mi = tau_mi(X_miss, w, y, m = m, method = method)
    
    return tau_dr_mi, tau_ols_mi, tau_ols_ps_mi

def ihdp_cevae(set_id = 1, prop_miss=0.1, seed=0,
        d_cevae=20, n_epochs=402,
		method="glm", **kwargs):

    # import here because of differents sklearn version used
    from cevae_tf import cevae_tf
    from sklearn.preprocessing import Imputer

    X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
    
    if prop_miss > 0:
        X_miss = ampute(X, prop_miss = prop_miss, seed = seed)
        X_imp = Imputer().fit_transform(X_miss)
    else:
        X_imp = X

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

def ihdp_miwae_save(out_folder = 'results/', set_id_range = range(1,2), prop_miss=0.1, seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1, add_wy = False,
        method="glm", **kwargs):
    from miwae import miwae
    l_scores = []
    for set_id in set_id_range:
        X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
        w = np.array(X.iloc[:,0]).reshape((-1,1))
        y = np.array(X.iloc[:,1]).reshape((-1,1))
        
        X = np.array(X.iloc[:,5:])
            
        X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

        if set_id == 1:
            if add_wy:
                xhat, zhat, zhat_mul = miwae(X_miss, d_miwae=d_miwae, sig_prior = sig_prior,
                                             n_epochs=n_epochs, add_wy = add_wy, w=w, y=y)
            else:
                xhat, zhat, zhat_mul = miwae(X_miss, d_miwae=d_miwae, sig_prior = sig_prior,
                                             n_epochs=n_epochs, add_wy = add_wy)

            print('shape of outputs miwae:')
            print(xhat.shape, zhat.shape, zhat_mul.shape)

            np.savetxt(out_folder + 'ihdp_prop_miss' + str(prop_miss) +'set_id' +str(set_id) + "_zhat.csv", zhat, delimiter=";")
            zhat_mul
            for i in range(zhat_mul.shape[0]):
                np.savetxt(out_folder+'ihdp_prop_miss' + str(prop_miss) +'set_id' +str(set_id) + "_zhat_m"+str(i)+".csv", zhat_mul[i,:,:], delimiter=";")
    return 0

def ihdp_miwae(set_id_range = range(1,1001), prop_miss=0.1, seed=0,
        d_miwae=3, n_epochs=602, sig_prior=1, add_wy = False,
		method="glm", **kwargs):

    from miwae import miwae
    l_scores = []
    for set_id in set_id_range:
        X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
        w = np.array(X.iloc[:,0]).reshape((-1,1))
        y = np.array(X.iloc[:,1]).reshape((-1,1))
        
        X = np.array(X.iloc[:,5:])
            
        X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

        if set_id == 1:
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
        print(set_id)
        # Tau estimated on Zhat=E[Z|X]
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat, w, y)
        res_tau_ols = tau_ols(zhat, w, y)
        res_tau_ols_ps = tau_ols_ps(zhat, w, y)
        res_tau_dr = tau_dr(y, w, y0_hat, y1_hat, ps_hat, method)
        res_tau_diffmu = np.mean(y1_hat - y0_hat)

        lr = LinearRegression()
        lr.fit(zhat, y)
        y_hat = lr.predict(zhat)
        res_tau_resid = tau_residuals(y, w, y_hat, ps_hat, method)

        # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
        res_mul_tau_dr = []
        res_mul_tau_ols = []
        res_mul_tau_ols_ps = []
        res_mul_tau_resid = []
        res_mul_tau_diffmu = []
        for zhat_b in zhat_mul: 
            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(zhat_b, w, y)
            res_mul_tau_dr.append(tau_dr(y, w, y0_hat, y1_hat, ps_hat, method))
            res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
            res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))
            res_mul_tau_diffmu.append(np.mean(y1_hat - y0_hat))
            lr = LinearRegression()
            lr.fit(zhat, y)
            y_hat = lr.predict(zhat_b)
            res_mul_tau_resid = tau_residuals(y, w, y_hat, ps_hat, method)

        res_mul_tau_dr = np.mean(res_mul_tau_dr)
        res_mul_tau_ols = np.mean(res_mul_tau_ols)
        res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
        res_mul_tau_resid = np.mean(res_mul_tau_resid)
        res_mul_tau_diffmu = np.mean(res_mul_tau_diffmu)

        dcor_zhat = np.nan

        dcor_zhat_mul = np.nan

        score = [prop_miss, d_miwae, sig_prior, n_epochs, add_wy, set_id, res_tau_dr, res_tau_ols, res_tau_ols_ps, res_tau_resid, res_tau_diffmu, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps, res_mul_tau_resid, res_mul_tau_diffmu, dcor_zhat, dcor_zhat_mul]
        l_scores.append(score)
        
    score_data = pd.DataFrame(l_scores, 
                              columns=list(['prop_miss','d_miwae','sig_prior','n_epochs','add_wy','set_id',
                                'res_tau_dr', 'res_tau_ols', 'res_tau_ols_ps', 'res_tau_resid', 'res_tau_diffmu', 
                                'res_mul_tau_dr', 'res_mul_tau_ols', 'res_mul_tau_ols_ps', 'res_mul_tau_resid', 'res_mul_tau_diffmu', 
                                'dcor_zhat', 'dcor_zhat_mul']))
                    
    return score_data

def ihdp_miwae_cv(set_id = 1, prop_miss=0.1, seed=0,
                  d_miwae_list=[10, 100], n_epochs=602, sig_prior_list=[0.1, 1, 10], add_wy = False,
                  method="glm", k_fold = 5, **kwargs):

    from miwae import miwae_cv

    X = pd.read_csv('./data/IHDP/csv/R_ate_ihdp_npci_' + str(set_id) + '.csv')
    w = np.array(X.iloc[:,0]).reshape((-1,1))
    y = np.array(X.iloc[:,1]).reshape((-1,1))
    
    X = np.array(X.iloc[:,5:])
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    if add_wy:
        elbo = miwae_cv(X_miss, d_miwae_list=d_miwae_list, sig_prior_list = sig_prior_list, k_fold = k_fold,
                        n_epochs=n_epochs, add_wy = add_wy, w=w, y=y)
    else:
        elbo = miwae_cv(X_miss, d_miwae_list=d_miwae_list, sig_prior_list = sig_prior_list, k_fold = k_fold,
                        n_epochs=n_epochs, add_wy = add_wy)


    return elbo


if __name__ == '__main__':

    #print('test exp with default arguments on miwae')
    #tau = exp_miwae(n_epochs = 3)
    #print('Everything went well.')


    print('test ihdp with default arguments on mi')
    tau = ihdp_mi(m=2)
    print(tau)
    # print('test ihdp with baseline')
    # tau = ihdp_baseline()
    # print('test ihdp with miwae')
    # tau = ihdp_miwae(n_epochs=2)
    # print(tau)
    # for set_id in range(1,11):
    #     X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
    #     #print('y1-y0:', np.mean((X.iloc[:,0]==1)*(X.iloc[:,1]  - X.iloc[:,2]) +(X.iloc[:,0]==0)*(X.iloc[:,2]  - X.iloc[:,1])))
    #     print('mu1-mu0:', np.mean(X.iloc[:,4]  - X.iloc[:,3]))

    print('Everything went well.')

