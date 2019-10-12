import numpy as np
from scikit.linear_model import LogisticRegression

from miwae import miwae
from metrics import tau_dr, tau_ols, tau_ols_ps
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