import numpy as np

from miwae import miwae
from model import linear_tau
from generate_data import gen_linear, ampute


def exp(n=1000, d=3, p=100, d_miwae=3, n_epochs=2):

    Z, X, W, y = gen_linear(n=n, d=d, p=p)

    X_miss = ampute(X)

    xhat, zhat, zhat_mul = miwae(X_miss, d=d_miwae,
                                 n_epochs=n_epochs)

    print('shape of outputs miwae:')
    print(xhat.shape, zhat.shape, zhat_mul.shape)

    tau = linear_tau(zhat, W, y)

    print('tau =', tau)
    return tau


if __name__ == '__main__':
    exp()