import numpy as np
from scikit.linear_model import LogisticRegression

from miwae import miwae
from metrics import tau_dr, tau_ols, tau_ols_ps
from generate_data import gen_lrmf, ampute



def exp(n=1000, d=3, p=100, prop_miss=0.1, d_miwae=3, n_epochs=1, sig_prior = 1,
		method = "glm"):

    Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p)

    X_miss = ampute(X, prop_miss = prop_miss)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae, sig_prior = sig_prior,
                                 n_epochs=n_epochs)

    print('shape of outputs miwae:')
    print('xhat.shape, zhat.shape, zhat_mul.shape:') 
    #    (1000, 200) (1000, 3) (200, 1000, 3)
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    # Tau estimated on Zhat=E[Z|X]
	res_tau_ols = tau_ols(zhat, w, y)

	lr = LogisticRegression()
	lr.fit(zhat, w)
	ps_hat = lr.predict_proba(zhat)
	res_tau_ols_ps = tau_ols_ps(zhat, w, y, ps_hat)

	if method == "glm":
	    
	    lr = LinearRegression()
	    lr.fit(zhat[np.equal(w, np.ones(n)),:], y[np.equal(w, np.ones(n))])
	    y1_hat = lr.predict(zhat)

	    lr = LinearRegression()
	    lr.fit(zhat[np.equal(w, np.zeros(n)),:], y[np.equal(w, np.zeros(n))])
	    y0_hat = lr.predict(zhat)

	    res_tau_dr = tau_dr(zhat, w, y, "glm", y1_hat, y0_hat, ps_hat)
	    
	elif method == "grf":
		res_tau_dr = np.nan
    	raise NotImplementedError("Causal forest estimation not implemented here yet.")
	else:
		res_tau_dr = np.nan
        raise ValueError("'method' should be choosed between 'glm' and 'grf' in 'exp', got %s", method)

    print('tau_dr =', res_tau_dr)
    print('tau_ols =', res_tau_ols)
    print('tau_ols_ps =', res_tau_ols_ps)

    # Tau estimated on Zhat^(b), l=1,...,B sampled from posterior
    res_mul_tau_dr = np.empty(zhat_mul.shape[0])
    res_mul_tau_ols = np.empty(zhat_mul.shape[0])
    res_mul_tau_ols_ps = np.empty(zhat_mul.shape[0])
    for b in range(zhat_mul.shape[0]):
    	lr = LogisticRegression()
    	lr.fit(zhat_mul[b,:,:], w)
    	ps_hat = lr.predict_proba(zhat_mul[b,:,:])

    	lr = LinearRegression()
    	lr.fit(zhat_mul[b, np.equal(w, np.ones(n)),:], y[np.equal(w, np.ones(n))])
    	y1_hat = lr.predict(zhat_mul[b,:,:])

    	lr = LinearRegression()
    	lr.fit(zhat_mul[b, np.equal(w, np.zeros(n)),:], y[np.equal(w, np.zeros(n))])
    	y0_hat = lr.predict(zhat_mul[b,:,:])

    	res_mul_tau_dr[b] = tau_dr(zhat_mul[b,:,:], w, y, "glm", y1_hat, y0_hat, ps_hat)
    	res_mul_tau_ols[b] = tau_ols(zhat_mul[b,:,:], w, y)
    	res_mul_tau_ols_ps[b] = tau_ols_ps(zhat_mul[b,:,:], w, y, ps_hat)

    res_mul_tau_dr = np.mean(res_mul_tau_dr)
    res_mul_tau_ols = np.mean(res_mul_tau_ols)
    res_mul_tau_ols_ps = np.mean(res_mul_tau_ols_ps)

    print('mul_tau_dr =', res_mul_tau_dr)
    print('mul_tau_ols =', res_mul_tau_ols)
    print('mul_tau_ols_ps =', res_mul_tau_ols_ps)

    return res_tau_dr, res_tau_ols, res_tau_ols_ps, res_mul_tau_dr, res_mul_tau_ols, res_mul_tau_ols_ps


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