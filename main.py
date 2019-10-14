import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from miwae import miwae
from metrics import tau_dr, tau_ols, tau_ols_ps
from generate_data import gen_lrmf, ampute, gen_dlvm


def exp(model="dlvm", n=1000, d=3, p=100, prop_miss=0.1, seed = 0,
        d_miwae=3, n_epochs=602, sig_prior = 1,
		method = "glm"):

    if model == "lrmf":
        Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p, seed = seed)
    elif model == "dlvm":
        Z, X, w, y, ps = gen_dlvm(n=n, d=d, p=p, seed = seed)
    else:
        raise NotImplementedError("Other data generating models not implemented here yet.")
        
    X_miss = ampute(X, prop_miss = prop_miss, seed = seed)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
                                 n_epochs=n_epochs)

    print('shape of outputs miwae:')
    print('xhat.shape, zhat.shape, zhat_mul.shape:') 
    #    (1000, 200) (1000, 3) (200, 1000, 3)
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    # Tau estimated on Zhat=E[Z|X]
    res_tau_ols = tau_ols(zhat, w, y)
    res_tau_ols_ps = tau_ols_ps(zhat, w, y)
    res_tau_dr = tau_dr(zhat, w, y, "glm")
    print('tau_dr =', res_tau_dr)
    print('tau_ols =', res_tau_ols)
    print('tau_ols_ps =', res_tau_ols_ps)

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = []
    res_mul_tau_ols = []
    res_mul_tau_ols_ps = []
    for zhat_b in zhat_mul: 
    	res_mul_tau_dr.append(tau_dr(zhat_b, w, y, "glm"))
    	res_mul_tau_ols.append(tau_ols(zhat_b, w, y))
    	res_mul_tau_ols_ps.append(tau_ols_ps(zhat_b, w, y))

    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)
    print('mul_tau_dr =', res_mul_tau_dr)
    print('mul_tau_ols =', res_mul_tau_ols)
    print('mul_tau_ols_ps =', res_mul_tau_ols_ps)

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps


def plot_n_d():
    range_seed = np.arange(10)
    range_n = [10**4, 10**6]
    range_p = [20, 100, 1000]
    range_prop_miss = [0.1, 0.3, 0]
    range_sig_prior = [0.1, 1, 10]

    l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']
    args_name = ['model', 'seed', 'n', 'p', 'prop_miss', 'sig_prior']

    l_scores = []
    for model in ["dlvm","lrmf"]:
        for seed in range_seed:
            for prop_miss in range_prop_miss:
                for n in range_n:
                    for p in range_p:     
                        for sig_prior in range_sig_prior:
                            # print('start with', n, p, prop_miss)
                            score = exp(model = model, n=n, d=3, p=p, prop_miss=prop_miss,
                                        seed=seed, d_miwae=3,
                                        sig_prior=sig_prior, n_epochs=602)
                            args = (model, seed, n, p, prop_miss, sig_prior)
                            l_scores.append(np.concatenate((args,score)))
                            print('exp with ', args_name)
                            print(args)
                            print('........... DONE !\n\n')

                score_data = pd.DataFrame(l_scores, columns=args_name + l_tau)
                score_data.to_csv('results/plot_nd_temp.csv')
    
    score_data.to_csv('results/plot_nd.csv')


def plot_epoch():

    l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']
    args_name = ['model','n', 'n_epochs']

    l_scores = []
    for model in ["dlvm"]:
        for n in [200, 1000, 10000]:
            for n_epochs in [10, 100, 400, 600, 800]:
                score = exp(model = model, n=n, d=3, p=100, prop_miss=0.1, seed = 0,
                    d_miwae=3, n_epochs=n_epochs, sig_prior = 1,
                    method = "glm")
                args = (model ,n, n_epochs)
                l_scores.append(np.concatenate((args,score)))
                print('exp with ', args_name)
                print(args)
                print('........... DONE !\n\n')

                score_data = pd.DataFrame(l_scores, columns=args_name + l_tau)
                score_data.to_csv('results/plot_epoch_temp.csv')
    
    score_data.to_csv('results/plot_epoch.csv')
    

if __name__ == '__main__':

    # screen -S exp
    # taskset -c 0-23 python3 main.py
    plot_epoch()
    # plot_n_d()
    # exp(n_epochs=602)