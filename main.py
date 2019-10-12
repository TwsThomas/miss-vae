import numpy as np

from miwae import miwae
from metrics import tau_dr, tau_ols
from generate_data import gen_lrmf, ampute


def exp(n=1000, d=3, p=100, d_miwae=3, n_epochs=1):

    Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p)

    X_miss = ampute(X)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae,
                                 n_epochs=n_epochs)

    print('shape of outputs miwae:')
    print('xhat.shape, zhat.shape, zhat_mul.shape:') 
    #    (1000, 200) (1000, 3) (200, 1000, 3)
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    res_tau_dr = 0 #tau_dr(zhat, w, y, "glm", y1_hat, y0_hat, ps_hat)
    res_tau_ols = tau_ols(zhat, w, y)

    print('tau_dr =', res_tau_dr)
    print('tau_ols =', res_tau_ols)

    return res_tau_dr, res_tau_ols


def plot_n_d():

    range_n = [10**4, 10**6]
    range_p = [20,100, 1000]

    l_scores = []
    for n in range_n:
        for p in range_p:
            print(n, p)
            l_scores.append(exp(n=n, d=3, p=p, d_miwae=3, n_epochs=2))
            print('exp with (n,p)=',n,p,'........... DONE !')

    np.savetxt(l_scores, 'results/plot_n_d.nptxt')

if __name__ == '__main__':
    plot_n_d()
    #exp()