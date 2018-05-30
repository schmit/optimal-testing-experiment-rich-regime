import math
import distribution as d

from solvers import binomial_acceptance

def msprt_pval(lr):
    return 1/lr

# Normal MSPRT
def normal_msprt(Xbar, n, mu0, prior, data):
    constant = data.sigma / math.sqrt(data.sigma**2 + n * prior.sigma**2)

    exp_nom = n**2 * prior.sigma**2 * (Xbar - mu0)**2
    exp_denom = 2 * data.sigma**2 * (data.sigma**2 + n * prior.sigma**2)
    exponential = exp_nom / exp_denom
    return constant * math.exp(exponential)

def normal_msprt_test(p, s, prior, alpha=0.05, Nmax=5000, steps=1):
    Sn = 0
    X = d.Binomial(steps, p)
    for n in range(steps, Nmax+1, steps):
        Sn += X.sample()
        data = d.Gaussian(0, (Sn+p)/(n+1) * (1-(Sn+p)/(n+1)))
        pval = normal_msprt_pval(normal_msprt(Sn / n, n, s, prior, data))
        if pval < alpha:
            if Sn / n > s:
                return 1, n, Sn
            return -1, n, Sn
    # no outcome after Nmax steps
    return 0, n, Sn


# Binomial MSPRT
def Beta(a, b):
    return math.exp(logBeta(a, b))

def logBeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def beta_binomial_lr(x, n, p0, a=1, b=1):
    log_nom = logBeta(a + x, b + n - x)
    log_denom = x * math.log(p0) + (n-x) * math.log(1-p0) + logBeta(a, b)
    return math.exp(log_nom - log_denom)

def binomial_msprt_test(p, s, prior, alpha=0.05, N=1000, steps=1):
    Sn = 0
    X = d.Binomial(steps, p)
    for n in range(steps, N+steps, steps):
        Sn += X.sample()
        pval = msprt_pval(beta_binomial_lr(Sn, n, s, prior.a, prior.b))
        if pval < alpha:
            if Sn / n > s:
                return 1, n, Sn
            return -1, n, Sn
    # no outcome after Nmax steps
    return 0, n, Sn

def binomial_bayesian_test(s, prior, alpha=0.05, beta=0.2, N=1000, steps=1):
    def _test(p):
        posterior = prior
        Sn = 0
        X = d.Binomial(steps, p)
        for n in range(steps, N+steps, steps):
            Y = X.sample()
            Sn += Y
            posterior = d.posterior(posterior, X, [Y])

            # obtain at least 5 samples
            if posterior.cdf(s) < alpha and n > 5:
                return 1, n, Sn
            elif 1-posterior.cdf(s) < beta:
                return -1, n, Sn

        # no outcome after Nmax steps
        return 0, n, Sn

    return _test


