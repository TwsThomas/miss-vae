import numpy as np

from miwae import miwae
from metrics import tau_dr, tau_ols
from generate_data import gen_lrmf, ampute



def exp(n=1000, d=3, p=100, prop_miss=0.1, d_miwae=3, n_epochs=1, sig_prior = 1,
		method = "glm"):


    Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p)

    X_miss = ampute(X)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
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
    range_prop_miss = [0, 0.1, 0.3]
    range_sig_prior = [0.1, 1, 10]


    l_scores = []
    for n in range_n:
        for p in range_p:
        	for prop_miss in range_prop_miss:
        		for sig_prior in range_sig_prior:
	            	print(n, p, prop_miss)
	            	l_scores.append(exp(n=n, d=3, p=p, prop_miss = prop_miss, d_miwae=3, sig_prior = sig_prior, n_epochs=2))
	            	print('exp with (n,p, prop_miss, sig_prior)=',n,p,prop_miss,sig_prior,'........... DONE !')


    np.savetxt(l_scores, 'results/plot_n_d.nptxt')

if __name__ == '__main__':
    plot_n_d()
    #exp()