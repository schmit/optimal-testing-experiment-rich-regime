import numpy as np

def beta_mom(ps):
    """ Method of moments estimator for Beta distribution """
    M = np.mean(ps)
    S = np.var(ps)

    Y = (M * (1-M))/S - 1
    a = M * Y
    b = (1-M) * Y
    return a, b

