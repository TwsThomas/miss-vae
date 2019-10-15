# -*- coding: utf-8 -*-
from metrics import *
from generate_data import *
from main import exp_baseline

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score

def test_get_ps_y01_hat(n=1000, p=2, d=3):

    for gen_name, gen_data in zip(['gen_lrmf','gen_dlvm'], [gen_lrmf, gen_dlvm]):
        print('-----------', gen_name, '----------')
        r2 = []
        r2_y = []
        r2_x = []
        r2_y_x = []
        for i in range(5):

            Z, X, w, y, ps = gen_data(n=n, p=p, d=d, seed=i)

            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)

            r2.append(r2_score(ps, ps_hat))
            y_hat = [y1_hat[i] if w[i] else y0_hat[i] 
                                for i in range(len(y))]
            r2_y.append(r2_score(y, y_hat))

            ps_hat, y0_hat, y1_hat = get_ps_y01_hat(X, w, y)

            r2_x.append(r2_score(ps, ps_hat))
            y_hat = [y1_hat[i] if w[i] else y0_hat[i] 
                                for i in range(len(y))]
            r2_y_x.append(r2_score(y, y_hat))

        print('R2_score of ps_hat (with Z_oracle)=', np.round(np.mean(r2),4),
            '+-', np.round(np.std(r2),4))
        print('R2_score of y_hat (with Z_oracle)=', np.round(np.mean(r2_y),4),
            '+-', np.round(np.std(r2_y),4))
        print('R2_score of ps_hat (with X)=', np.round(np.mean(r2_x),4),
            '+-', np.round(np.std(r2_x),4))
        print('R2_score of y_hat (with X)=', np.round(np.mean(r2_y_x),4),
            '+-', np.round(np.std(r2_y_x),4))

        # assert np.mean(r2) > .9
        # assert np.mean(r2_y) > .9

    return r2, r2_y, r2_x, r2_y_x


# def test_tau_dr():

#     for gen_name, gen_data in zip(['gen_lrmf','gen_dlvm'], [gen_lrmf, gen_dlvm]):
#         print('-----------', gen_name, '----------')
        
#         for i in range(20):

#             Z, X, w, y, ps = gen_data(seed=i)
#             ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)



def get_baseline(n=1000, p=100, d=3, gen_model = 'gen_lrfm'):

    if gen_model == "gen_lrmf":
        gen_data = gen_lrmf
    elif gen_model == "gen_dlvm":
        gen_data = gen_dlvm

    d_tau = defaultdict(list)

    for i in range(5):
        Z, X, w, y, ps = gen_data(n=n, p=p, d=d, seed=i)
        d_tau['tau_dr_yyy_oracle'].append(tau_dr(y, w, y, y, ps))
        # d_tau['tau_ols_oracle'].append(tau_ols(Z, w, y))
        
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_Z_ps_oracle'].append(tau_dr(y, w , y0_hat, y1_hat , ps))
        d_tau['tau_ols_Z_ps_oracle'].append(tau_ols(Z, w, y))

        # use ps_hat from Z.
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_Z'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_Z'].append(tau_ols(Z, w, y))

        # Use X instead of Z.
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(X, w, y)
        d_tau['tau_dr_X'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_X'].append(tau_ols(X, w, y))

        ## permute Z.
        Z_perm = np.random.permutation(Z) 
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z_perm, w, y)
        d_tau['tau_dr_Z_perm'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_Z_perm'].append(tau_ols(Z_perm, w, y))

        ## random Z.
        # Z_rnd = np.random.randn(Z.shape[0], Z.shape[1])
        # ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z_rnd, w, y)
        # d_tau['tau_dr_Zrnd'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        # d_tau['tau_ols_Zrnd'].append(tau_ols(Z_rnd, w, y))

        # ## random y.
        # y_perm = np.random.permutation(y) 
        # ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        # d_tau['tau_dr_y_perm'].append(tau_dr(y_perm, w , y0_hat, y1_hat , ps_hat))
        # d_tau['tau_ols_y_perm'].append(tau_ols(Z_rnd, w, y_perm))

    return d_tau

def boxplot_baseline(nn=10):

    df_base = pd.DataFrame()
    for seed in range(nn):
        d_tau = exp_baseline(seed = seed)
        df = pd.DataFrame(d_tau, index = ['tau_dr','tau_ols','tau_ols2']).T
        df['seed'] = seed
        df_base = pd.concat((df_base, df))
    df_base['method'] = list(df_base.index)
    df_base['1-tau_dr'] = abs(1-df_base['tau_dr'])
    df_base.head()

    sns.swarmplot(x='method', y='1-tau_dr', data=df_base)
    plt.figure()
    sns.boxplot(x='method', y='1-tau_dr', data=df_base)
    return df_base


def plot_baseline(show=False):

    for gen_model in ['gen_lrmf','gen_dlvm']:
        print('---', gen_model, '------')
        d_tau = get_baseline(gen_model = gen_model)
        for name, tau in d_tau.items():
            print(name, np.round(np.mean(tau),4),'+-', np.round(np.std(tau),4))


        plt.hist(d_tau['tau_dr_oracle'], alpha = .4, label='tau_dr_oracle')
        plt.hist(d_tau['tau_dr_perm'], alpha = .4, label='tau_dr_perm')
        plt.legend()
        if show:
            plt.show()

        plt.figure()
        plt.hist(d_tau['tau_ols_oracle'], alpha = .4, label='tau_ols_oracle')
        plt.hist(d_tau['tau_ols_perm'], alpha = .4, label='tau_ols_perm')
        plt.legend()
        if show:
            plt.show()


if __name__ == '__main__':

    print('Running test_metrics.py')
    test_get_ps_y01_hat()
    plot_baseline()

    
"""
Running test_metrics.py
----------- gen_lrmf ----------
R2_score of ps_hat (with Z_oracle)= 0.9376 +- 0.0381
R2_score of y_hat (with Z_oracle)= 0.9829 +- 0.0009
R2_score of ps_hat (with X)= -0.6078 +- 0.1825
R2_score of y_hat (with X)= 0.9818 +- 0.0011
----------- gen_dlvm ----------
R2_score of ps_hat (with Z_oracle)= 0.9376 +- 0.0381
R2_score of y_hat (with Z_oracle)= 0.9829 +- 0.0009
R2_score of ps_hat (with X)= -0.8787 +- 0.1853
R2_score of y_hat (with X)= 0.8566 +- 0.0085
--- gen_lrmf ------
tau_dr_yyy_oracle 0.0 +- 0.0
tau_dr_Z_ps_oracle 1.0014 +- 0.0057
tau_ols_Z_ps_oracle 1.0013 +- 0.0056
tau_dr_Z 1.0015 +- 0.0057
tau_ols_Z 1.0013 +- 0.0056
tau_dr_X 1.0015 +- 0.0078
tau_ols_X 1.0019 +- 0.0078
tau_dr_Z_perm 1.0387 +- 0.0401
tau_ols_Z_perm 1.0387 +- 0.0401
tau_dr_Zrnd 1.0383 +- 0.0401
tau_ols_Zrnd 1.0383 +- 0.0401
tau_dr_y_perm -0.0026 +- 0.0586
tau_ols_y_perm -0.0036 +- 0.0495
--- gen_dlvm ------
tau_dr_yyy_oracle 0.0 +- 0.0
tau_dr_Z_ps_oracle 1.0014 +- 0.0057
tau_ols_Z_ps_oracle 1.0013 +- 0.0056
tau_dr_Z 1.0015 +- 0.0057
tau_ols_Z 1.0013 +- 0.0056
tau_dr_X 1.0257 +- 0.0217
tau_ols_X 1.0246 +- 0.0213
tau_dr_Z_perm 1.0387 +- 0.0401
tau_ols_Z_perm 1.0387 +- 0.0401
tau_dr_Zrnd 1.0383 +- 0.0401
tau_ols_Zrnd 1.0383 +- 0.0401
tau_dr_y_perm -0.0026 +- 0.0586
tau_ols_y_perm -0.0036 +- 0.0495"""