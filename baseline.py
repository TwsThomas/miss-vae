# -*- coding: utf-8 -*-
from metrics import *
from generate_data import *
from main import exp_baseline
from main_ihdp import ihdp_baseline

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score
from joblib import Memory
memory = Memory('cache_dir', verbose=0)


def load_results(expname = 'exp_15.2_10_choux.csv_temp'):
    df = pd.read_csv('results/' + expname)
    df.drop(labels='Unnamed: 0', inplace=True, axis=1)
    df['|1-tau_dr|'] = abs(df['tau_dr'] - 1)
    df['|1-tau_ols|'] = abs(df['tau_ols'] - 1)
    df['|1-tau_ols_ps|'] = abs(df['tau_ols_ps'] - 1)
    print('results.shape', df.shape)
    if 'algo' not in list(df.columns):
        df.loc[:,'algo'] = ['miwae_'] * df.shape[0]
    df.head()
    return df


def get_best_params(df_results, loss = '1-tau_dr'):
    # return best params and df.loc[best_params]
    # according to 'loss'

    args_col = list(set(df_results.columns[:list(df_results.columns).index('tau_dr')]) - set(['seed','set_id','time']))
    best_params = (df_results.groupby(args_col)[loss].mean()).idxmin()
    best_params = {name:value for name,value in zip(args_col, best_params)}
    print('best_params=', best_params)

    df_best = df_results.loc[(df_results[args_col] == best_params).all(axis=1)]

    return best_params, df_best


def boxplot_with_baseline(df_results, df_mice_results=None, loss = 'tau_dr', hue = None,
                          baseline = 'exp', 
                          palette = None, save_plot = None,
                          ground_truth = pd.DataFrame({'tau': [1]})):
    # boxplot all baseline + best of df_results
    best_params = dict()
    if hue is not None:
        df_best = pd.DataFrame()
        for param_hue in df_results[hue].unique():
            best_params, df_best_hue = get_best_params(df_results.loc[df_results[hue] == param_hue], loss = loss)
            df_best = pd.concat((df_best, df_best_hue))

        df_base = pd.DataFrame()
        for param_hue in df_results[hue].unique():
            best_params_hue = best_params.copy()
            best_params_hue[hue] = param_hue
            if baseline == 'exp':
                df_base_hue = get_baseline(**best_params_hue)
            else:
                df_base_hue = get_ihdp_baseline(**best_params_hue)
            df_base_hue[hue] = param_hue
            
            df_base = pd.concat((df_base, df_base_hue))

    else:
        best_params, df_best = get_best_params(df_results, loss = loss)
        if baseline == 'exp':
            df_base = get_baseline(**best_params)
        else:
            df_base = get_ihdp_baseline(**best_params)


    args_col = list(set(df_mice_results.columns[:list(df_mice_results.columns).index('tau_dr')]) - set(['seed','set_id',hue]))
    best_params_list = []
    params_list = []
    for key, value in best_params.items():
        temp = [key,value]
        if key in args_col:
            best_params_list.append(value)
            params_list.append(list(temp))
    df_mice = df_mice_results.loc[(df_mice_results[args_col] == best_params_list).all(axis=1)]

    df_co = pd.concat((df_best, df_base, df_mice), sort=True)
    # sns.boxplot(x='algo', y='1-tau_dr', data=df_co)

    if any(np.isnan(df_co.loc[df_co['algo']=='mice', hue])):
        i = 0
        tmp = pd.DataFrame()
        for param_hue in df_results[hue].unique():
            tt = pd.DataFrame()
            tt = df_co.loc[df_co['algo']=='mice']
            tt[hue] = param_hue
            if i > 0:
                tmp = pd.concat((tmp,tt))
            else:
                tmp = tt.copy()
            i += 1
        df_co = pd.concat((df_co, tmp), sort= True)
            

    # if 'set_id' in best_params.keys():
    #     ground_truth = ground_truth.loc[ground_truth['set_id'] == best_params['set_id']]


    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.swarmplot(x='algo', y=loss, hue = hue, data=df_co, palette = palette)
    if (loss == 'tau_dr') | (loss == 'tau_ols') | (loss == 'tau_ols_ps'):
            plt.axhline(y=np.mean(ground_truth['tau']), color='k')

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #      ncol=3, fancybox=True, title=hue, title_fontsize = 15, fontsize=15)

    lgd1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
         fancybox=True, shadow=True, ncol=4, title=hue, title_fontsize = 15, fontsize=15)
    plt.subplot(1,2,2)
    sns.boxplot(x='algo', y=loss, hue = hue, data=df_co, palette = palette)
    if (loss == 'tau_dr') | (loss == 'tau_ols') | (loss == 'tau_ols_ps'):
            plt.axhline(y=np.mean(ground_truth['tau']), color='k')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #      ncol=3, fancybox=True, title=hue, title_fontsize = 15, fontsize=15)
    lgd2 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
         fancybox=True, shadow=True, ncol=4, title=hue, title_fontsize = 15, fontsize=15)

    if hue is not None:
        for param_hue in df_results[hue].unique():
            print(hue,':', param_hue)
            tmp = df_co.loc[df_co[hue]==param_hue]
            print('MDC.process: mean ', loss,':', 
                  np.mean(tmp.loc[tmp['algo']=='MDC.process', loss]), 
                  'std: ',standard_error(tmp.loc[tmp['algo']=='MDC.process', loss]))
            print('MDC.mi: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='MDC.mi', loss]),
                  'std: ',standard_error(tmp.loc[tmp['algo']=='MDC.mi', loss]))
            print('mice: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='mice', loss]),
                  'std: ',standard_error(tmp.loc[tmp['algo']=='mice', loss]))
            print('mean_imp: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='X_imp_mean', loss]),
                  'std: ',standard_error(tmp.loc[tmp['algo']=='X_imp_mean', loss]))
            print('full: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='X', loss]),
                  'std: ',standard_error(tmp.loc[tmp['algo']=='X', loss]))
    else:
        tmp = df_co
        print('MDC.process: mean ', loss,':', 
              np.mean(tmp.loc[tmp['algo']=='MDC.process', loss]), 
              'std: ',standard_error(tmp.loc[tmp['algo']=='MDC.process', loss]))
        print('MDC.mi: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='MDC.mi', loss]),
              'std: ',standard_error(tmp.loc[tmp['algo']=='MDC.mi', loss]))
        print('mice: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='mice', loss]),
              'std: ',standard_error(tmp.loc[tmp['algo']=='mice', loss]))
        print('mean_imp: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='X_imp_mean', loss]),
              'std: ',standard_error(tmp.loc[tmp['algo']=='X_imp_mean', loss]))
        print('full: mean ', loss,':', np.mean(tmp.loc[tmp['algo']=='X', loss]),
              'std: ',standard_error(tmp.loc[tmp['algo']=='X', loss]))

    
    if save_plot is not None:
        params_list = []
        for key, value in best_params.items():
            temp = [key,value]
            params_list.append(list(temp))

        plt.figure(figsize=(7.5,5))
        sns.boxplot(x='algo', y=loss, hue = hue, data=df_co, palette = palette)
        if (loss == 'tau_dr') | (loss == 'tau_ols') | (loss == 'tau_ols_ps'):
            plt.axhline(y=np.mean(ground_truth['tau']), color='k')
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
        #      ncol=3, fancybox=True, title=hue, title_fontsize = 15, fontsize=15)
        lgd2 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                          fancybox=True, shadow=False, ncol=4, title=hue, title_fontsize = 15, fontsize=15)
        figname = str('./figures/'+'_'.join(''.join(map(str,x)) for x in params_list)+'_'+ 'metric'+loss+'_'+save_plot)
        plt.savefig(figname, bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight',format='pdf')

@memory.cache
def get_baseline(repetitions=10, show=False, loss = '|1-tau_dr|', **kwargs):
    # return baseline with X, X_imp, Z_perm

    df_base = pd.DataFrame()
    for seed in range(repetitions):
        kwargs['seed'] = seed
        d_tau = exp_baseline(**kwargs)
        df = pd.DataFrame(d_tau, index = ['tau_dr','tau_ols','tau_ols_ps','tau_resid']).T

        df['seed'] = seed
        df_base = pd.concat((df_base, df))
    df_base['algo'] = list(df_base.index)
    df_base['|1-tau_dr|'] = abs(1-df_base['tau_dr'])
    df_base['|1-tau_ols|'] = abs(1-df_base['tau_ols'])
    df_base['|1-tau_ols_ps|'] = abs(1-df_base['tau_ols_ps'])
    df_base.head()

    
    if show:
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        sns.swarmplot(x='algo', y=loss, data=df_base)
        plt.subplot(1,2,2)
        sns.boxplot(x='algo', y=loss, data=df_base)
    return df_base

#@memory.cache
def get_ihdp_baseline(set_id = 1, prop_miss=0.1, 
                      method="glm", show=False, loss = '|1-tau_dr|', set_id_range = list(range(1,11)), **kwargs):
    # return baseline with X, X_imp, Z_perm

    df_base = pd.DataFrame()
    for set_id in set_id_range:
        d_tau = ihdp_baseline(set_id = set_id,
                              prop_miss=prop_miss, 
                              method=method)
        X = pd.read_csv('./data/IHDP/csv/ihdp_npci_' + str(set_id) + '.csv')
        tau = np.mean(X.iloc[:,4]  - X.iloc[:,3])
        df = pd.DataFrame(d_tau, index = ['tau_dr','tau_ols','tau_ols_ps']).T
        df['tau'] = tau
        df_base = pd.concat((df_base, df))
    df_base['algo'] = list(df_base.index)
    df_base['|tau-tau_dr|'] = abs(df_base['tau']-df_base['tau_dr'])
    df_base['|tau-tau_ols|'] = abs(df_base['tau']-df_base['tau_ols'])
    df_base['|tau-tau_ols_ps|'] = abs(df_base['tau']-df_base['tau_ols_ps'])
    df_base.head()

    if show:
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        sns.swarmplot(x='algo', y=loss, data=df_base)
        plt.subplot(1,2,2)
        sns.boxplot(x='algo', y=loss, data=df_base)
    return df_base

def test_get_ps_y01_hat(n=1000, p=2, d=3, citcio = False):

    for gen_name, gen_data in zip(['gen_lrmf','gen_dlvm'], [gen_lrmf, gen_dlvm]):
        print('-----------', gen_name, '----------')
        r2 = []
        r2_y = []
        r2_x = []
        r2_y_x = []
        for i in range(5):

            Z, X, w, y, ps = gen_data(n=n, p=p, d=d, citcio = citcio, prop_miss = 0, seed=i)

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

def correlation_tau(df):
    l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    corr = df.corr()
    corr = df[l_tau].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, center=0, #, cmap=cmap
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('tau correlation')
    # plt.savefig('results/tau_correlation_xxx.png')
   
def standard_error(x):
    return np.std(x, ddof=1)/np.sqrt(len(x))

if __name__ == '__main__':
    
    test_get_ps_y01_hat()
    

    
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





# def test_tau_dr():

#     for gen_name, gen_data in zip(['gen_lrmf','gen_dlvm'], [gen_lrmf, gen_dlvm]):
#         print('-----------', gen_name, '----------')
        
#         for i in range(20):

#             Z, X, w, y, ps = gen_data(seed=i)
#             ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)



# def get_baseline(n=1000, p=100, d=3, gen_model = 'gen_lrfm'):

#     if gen_model == "gen_lrmf":
#         gen_data = gen_lrmf
#     elif gen_model == "gen_dlvm":
#         gen_data = gen_dlvm

#     d_tau = defaultdict(list)

#     for i in range(5):
#         Z, X, w, y, ps = gen_data(n=n, p=p, d=d, seed=i)
#         d_tau['tau_dr_yyy_oracle'].append(tau_dr(y, w, y, y, ps))
#         # d_tau['tau_ols_oracle'].append(tau_ols(Z, w, y))
        
#         ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
#         d_tau['tau_dr_Z_ps_oracle'].append(tau_dr(y, w , y0_hat, y1_hat , ps))
#         d_tau['tau_ols_Z_ps_oracle'].append(tau_ols(Z, w, y))

#         # use ps_hat from Z.
#         ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
#         d_tau['tau_dr_Z'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
#         d_tau['tau_ols_Z'].append(tau_ols(Z, w, y))

#         # Use X instead of Z.
#         ps_hat, y0_hat, y1_hat = get_ps_y01_hat(X, w, y)
#         d_tau['tau_dr_X'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
#         d_tau['tau_ols_X'].append(tau_ols(X, w, y))

#         ## permute Z.
#         Z_perm = np.random.permutation(Z) 
#         ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z_perm, w, y)
#         d_tau['tau_dr_Z_perm'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
#         d_tau['tau_ols_Z_perm'].append(tau_ols(Z_perm, w, y))

#         ## random Z.
#         # Z_rnd = np.random.randn(Z.shape[0], Z.shape[1])
#         # ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z_rnd, w, y)
#         # d_tau['tau_dr_Zrnd'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
#         # d_tau['tau_ols_Zrnd'].append(tau_ols(Z_rnd, w, y))

#         # ## random y.
#         # y_perm = np.random.permutation(y) 
#         # ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
#         # d_tau['tau_dr_y_perm'].append(tau_dr(y_perm, w , y0_hat, y1_hat , ps_hat))
#         # d_tau['tau_ols_y_perm'].append(tau_ols(Z_rnd, w, y_perm))

#     return d_tau


# def plot_baseline(show=False):

#     for gen_model in ['gen_lrmf','gen_dlvm']:
#         print('---', gen_model, '------')
#         d_tau = get_baseline(gen_model = gen_model)
#         for name, tau in d_tau.items():
#             print(name, np.round(np.mean(tau),4),'+-', np.round(np.std(tau),4))


#         plt.hist(d_tau['tau_dr_oracle'], alpha = .4, label='tau_dr_oracle')
#         plt.hist(d_tau['tau_dr_perm'], alpha = .4, label='tau_dr_perm')
#         plt.legend()
#         if show:
#             plt.show()

#         plt.figure()
#         plt.hist(d_tau['tau_ols_oracle'], alpha = .4, label='tau_ols_oracle')
#         plt.hist(d_tau['tau_ols_perm'], alpha = .4, label='tau_ols_perm')
#         plt.legend()
#         if show:
#             plt.show()