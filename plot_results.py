import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

l_tau = ['tau_dr', 'tau_ols', 'tau_ols_ps', 'mul_tau_dr', 'mul_tau_ols', 'mul_tau_ols_ps']

def correlation_tau(df):
    # plot correlation amoung the differents metrics tau.

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

def plot(df, tau = 'tau_dr'):
    for n in np.unique(df['n']):
        df_group = df[df['n']==n]
        plt.plot(df_group['n_epochs'], df_group[tau], label='n = ' + str(n))
        plt.xlabel('n_epochs')
        plt.ylabel(tau)
        plt.legend()
    
def plot_all_tau(df):
    plt.figure(figsize=(15,8))
    for i,tau in enumerate(l_tau):
        plt.subplot(3,2,i+1)
        plot(df, tau)