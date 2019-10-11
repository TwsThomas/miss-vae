import numpy as np

def gen_linear(n=1000, d=3, p=100):
    Z = np.random.randn(n,d)
    beta = np.random.randn(d,p)
    X = Z.dot(beta)
    assert X.shape == (n,p)

    W = np.random.binomial(1, .5, size=n)
    gamma = np.random.randn(d+1)
    ZW = np.concatenate((Z,W.reshape((-1,1))), axis=1)
    y = ZW.dot(gamma).reshape((-1))
    print(y.shape, Z.shape, W.shape, ZW.shape)
    assert y.shape == (n,)
    assert W.shape == (n,)

    return Z, X, W, y


def ampute(X):
    X_miss = np.copy(X)
    mask = np.random.binomial(1,.2, size=X.shape)
    X_miss[mask] = np.nan
    return X_miss