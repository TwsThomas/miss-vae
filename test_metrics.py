from metrics import *
from generate_data import *

from collections import defaultdict
import matplotlib.pyplot as plt
def get_baseline(n=1000, p=100, d=3):

    d_tau = defaultdict(list)

    for i in range(100):
        Z, X, w, y, ps = gen_lrmf(n=n, p=p, d=d, seed=i)
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_oracle'].append(tau_dr(y, w , y0_hat, y1_hat , ps))
        d_tau['tau_ols_oracle'].append(tau_ols(Z, w, y))

        # Use X instead of Z.
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(X, w, y)
        d_tau['tau_dr_X'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_X'].append(tau_ols(X, w, y))

        # use ps_hat from Z.
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_ps_hat'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_ps_hat'].append(tau_ols(Z, w, y))

        ## permute Z.
        Z = np.random.permutation(Z) 
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_perm'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_perm'].append(tau_ols(Z, w, y))

        ## random Z.
        Z = np.random.randn(Z.shape[0], Z.shape[1])
        ps_hat, y0_hat, y1_hat = get_ps_y01_hat(Z, w, y)
        d_tau['tau_dr_white'].append(tau_dr(y, w , y0_hat, y1_hat , ps_hat))
        d_tau['tau_ols_white'].append(tau_ols(Z, w, y))


    return d_tau


if __name__ == '__main__':

    d_tau = get_baseline()
    for name, tau in d_tau.items():
        print(name, np.round(np.mean(tau),4),'+-', np.round(np.std(tau),4))


    plt.hist(d_tau['tau_dr_oracle'], alpha = .4, label='tau_dr_oracle')
    plt.hist(d_tau['tau_dr_perm'], alpha = .4, label='tau_dr_perm')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(d_tau['tau_ols_oracle'], alpha = .4, label='tau_ols_oracle')
    plt.hist(d_tau['tau_ols_perm'], alpha = .4, label='tau_ols_perm')
    plt.legend()
    plt.show()


# tau_dr_oracle 0.9989 +- 0.0067
# tau_ols_oracle 0.9989 +- 0.0067
# tau_dr_X 0.9986 +- 0.0086
# tau_ols_X 0.999 +- 0.0082
# tau_dr_ps_hat 0.9989 +- 0.0067
# tau_ols_ps_hat 0.9989 +- 0.0067
# tau_dr_perm 1.0362 +- 0.034
# tau_ols_perm 1.0362 +- 0.034
# tau_dr_white 1.0365 +- 0.0339
# tau_ols_white 1.0365 +- 0.0339

