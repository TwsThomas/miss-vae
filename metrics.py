import numpy as np

from sklearn.linear_model import LinearRegression


# Doubly robust ATE estimation
# if method == "glm": provide fitted values y1_hat, y0_hat and ps_hat
#                     for the two response surfaces and the propensity scores respectively
# if method == "grf": no need to provide any fitted values
def tau_dr(Z_hat, w, y, method = "glm", y1_hat, y0_hat, ps_hat):

	if method == "glm":
		tau_i = y1_hat - y0_hat + w*(y-y1_hat)/ps_hat -\
            					(1-w)*(y-y0_hat)/(1-ps_hat)
    	tau = mean(tau_i)
    elif method == "grf":
    	raise NotImplementedError("Causal forest estimation not implemented here yet.")
    else:
        raise ValueError("'method' should be choosed between 'glm' and 'grf' in 'tau_dr', got %s", method)
    return tau

# ATE estimation via OLS regression 
def tau_ols(Z_hat, w, y):
    ZW = np.concatenate((Z_hat, w), axis=1)
    lr = LinearRegression()
    lr.fit(ZW, y)
    tau = lr.coef_[-1]

    return tau

# Difference with linear_tau: add estimated propensity 
# scores as additional predictor
def tau_ols_ps(Z_hat, w, y, ps_hat):
    ZpsW = np.concatenate((Z_hat,ps_hat, w), axis=1)
    lr = LinearRegression()
    lr.fit(ZpsW, y)
    tau = lr.coef_[-1]

    return tau