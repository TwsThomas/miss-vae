import numpy as np

from sklearn.linear_model import LinearRegression


def linear_tau(Z_hat, W, y):
    ZW = np.concatenate((Z_hat,W), axis=1)
    lr = LinearRegression()
    lr.fit(ZW, y)
    tau = lr.coef_[-1]

    return tau
