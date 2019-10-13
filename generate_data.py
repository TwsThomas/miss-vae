import numpy as np

# Low rank matrix factorization
def gen_lrmf(n=1000, d=3, p=100, tau = 1, link = "linear", seed=0,
             noise_sd = 1):

    # V is fixed throughout experiments for given n,p,d
    np.random.seed(0)
    V = np.random.randn(p,d) 

    np.random.seed(seed)
    Z = np.random.randn(n,d)
    
    X = Z.dot(V.transpose())
    assert X.shape == (n,p)

    X = X + noise_sd*np.random.randn(n,p) # add perturbation to observation matrix

    # generate treatment assignment W
    ps, w = gen_treat(Z, link)

    # generate outcome
    y = gen_outcome(Z, w, tau, link)

    print(y.shape, Z.shape, w.shape)
    assert y.shape == (n,)
    assert w.shape == (n,)

    return Z, X, w, y, ps

# Deep Latent Variable Model (here, we use an MLP)
def gen_dlvm(n=1000, d=3, p=100, tau = 1, link = "linear", seed=0,
             h = 5, sd = 0.1):

    # V, W, a, b, alpha, beta are fixed throughout experiments for given n,p,d,h
    np.random.seed(0)
    V = np.random.randn(n,h)
    W = np.random.uniform(h*d).reshape((h,d))
    a = np.random.uniform(h)
    b = np.random.randn(p,1)
    alpha = np.random.randn(h,1)
    beta = np.random.uniform(1)

    np.random.seed(seed)
    Z = np.random.randn(n,d)

    X = np.empty([n,p])
    for i in range(n):
        mu, Sigma = get_dlvm_params(Z[i,:], V, W, a, b, alpha, beta)
        X[i,:] = np.random.multivariate_normal(mu, Sigma, 1)

    assert X.shape == (n,p)

    # generate treatment assignment W
    ps, w = gen_treat(Z, link)

    # generate outcome
    y = gen_outcome(Z, w, tau, link, sd=sd)
    
    print(y.shape, Z.shape, W.shape)
    assert y.shape == (n,)
    assert w.shape == (n,)

    return Z, X, w, y, ps

# Compute expectation and covariance of conditional distribution X given Z
def get_dlvm_params(z, V, W, a, b, alpha, beta):
    
    raise NotImplementedError('TODO: fix here')
    hu = -1  # ??
    h_,p = z.shape
    u = W.dot(z) + a
    mu = V.dot(np.tanh(hu)) + b
    sig = np.exp(alpha.dot(np.tanh(hu)) + beta)
    Sigma = sig*np.identity(p)

    return mu, Sigma

# Generate treatment assignment using confounders Z
def gen_treat(Z, link = "linear"):
    if link == "linear":
        ncolZ = Z.shape[1]
        beta = np.tile([0.3, -0.3], int(np.ceil(ncolZ/2.)))
        beta = beta[:ncolZ]
        f_Z = Z.dot(beta)
        ps = 1/(1+np.exp(-f_Z))
        w = np.random.binomial(1, ps)
        balanced = np.mean(w) > 0.4 and np.mean(w) < 0.6

        # adjust the intercept term if necessary to ensure balanced treatment groups
        offsets = np.linspace(-5, 5, num=50)
        i, best_idx, min_diff = 0, 0, Z.shape[0]
        while i < len(offsets) and not balanced:
            ps = 1/(1+np.exp(-offsets[i] - f_Z))
            w = np.random.binomial(1, ps)
            balanced = np.mean(w) > 0.4 and np.mean(w) < 0.6
            diff = abs(np.mean(w) - np.mean(1-w))
            if diff < min_diff:
                best_idx, min_diff = i, diff
            i += 1
        if (i == len(offsets)):
            ps = 1/(1+np.exp(-offsets[best_idx]-f_Z))
            w = np.random.binomial(1, ps)
    elif link == "nonlinear":
        raise NotImplementedError("Nonlinear w~Z not defined yet.")
    else:
        raise ValueError("'link' should be choosed between linear and nonlinear model for w. got %s", link)
    return ps, w

# Generate outcomes using confounders Z, treatment assignment w and ATE tau
def gen_outcome(Z, w, tau, link = "linear", sd=0.1):
    if link == "linear":
        n = Z.shape[0]
        ncolZ = Z.shape[1]
        epsilon = sd*np.random.randn(n)
        beta = np.tile([-0.2, 0.155, 0.5, -1, 0.2], int(np.ceil(ncolZ/5.)))
        beta = beta[:ncolZ]
        y = 0.5 + Z.dot(beta).reshape((-1)) + tau*w + epsilon
    elif link == "nonlinear":
        raise NotImplementedError("Nonlinear w~Z not defined yet.")
    else:
        raise ValueError("'link' should be choosed between linear and nonlinear model for y. got %s", link)
    return y

# Generate missing values in X such that, on average, X contains 100*prop_miss missing values
def ampute(X, prop_miss = 0.1, seed=0):
    np.random.seed(seed)
    X_miss = np.copy(X)
    mask = np.random.binomial(1,prop_miss, size=X.shape)
    X_miss[mask] = np.nan
    return X_miss