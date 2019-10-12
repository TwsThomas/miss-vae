import numpy as np

from miwae import miwae
from metrics import tau_dr, tau_ols
from generate_data import gen_lrmf, ampute


def exp(n=1000, d=3, p=100, d_miwae=3, n_epochs=2):

    Z, X, w, y, ps = gen_lrmf(n=n, d=d, p=p)

    X_miss = ampute(X)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae,
                                 n_epochs=n_epochs)

    print('shape of outputs miwae:')
    print('xhat.shape, zhat.shape, zhat_mul.shape:')
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    tau_dr = tau_dr(Z_hat, w, y, "glm", y1_hat, y0_hat, ps_hat)
    tau_ols = tau_ols(Z_hat, w, y)

    print('tau_dr =', tau_dr)
    print('tau_ols =', tau_ols)

    return tau


if __name__ == '__main__':
    exp()