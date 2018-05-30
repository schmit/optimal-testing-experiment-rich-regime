import distribution as d
import collections, math, random

import numpy as np

import json

boundary = collections.namedtuple("Boundary", "n accept reject values")

def bisection_search(backtracker, ET, lb=0, ub=None, verbose=False, tolerance=1e-3):
    if ub and lb and (ub - lb)/(lb+1) < tolerance:
        return backtracker(ub)

    if verbose:
        print("Search with E[T] = {:.2f}".format(ET))
    bdys, new_ET = backtracker(ET)
    if new_ET < ET:
        return bisection_search(backtracker, ET=(lb+new_ET)/2, lb=lb, ub=new_ET, verbose=verbose)
    if ub:
        return bisection_search(backtracker, ET=(ET+ub)/2, lb=ET, ub=ub, verbose=verbose)
    return bisection_search(backtracker, 2*ET, lb=ET, ub=ub, verbose=verbose)

def martingale(Yn, n, prior, data, nsamples=1):
    """
    Returns normal martingale distribution Y_{n+m} | Y_n

    where m is the number of samples to draw (default: 1)
    """
    gamma = (prior.tau / data.tau)**2

    return d.Gaussian(Yn, data.sigma / (n + nsamples + gamma) * math.sqrt(nsamples**2/(n+gamma) + nsamples))

def normal_acceptance(n, s, prior, data, alpha=0.05):
    za = d.Gaussian(0, 1).ppf(alpha)
    gamma = data.sigma**2 / prior.sigma**2

    return s - za * data.sigma / math.sqrt(n + gamma)

def normal_value(x, bdy, ET, prior, data, nsamples=1, reset_cost=0):
    Xnew = martingale(x, bdy.n-nsamples, prior, data, nsamples=nsamples)

    # compute values from borders
    prob_accept = 1-Xnew.cdf(bdy.accept)
    prob_reject = Xnew.cdf(bdy.reject)
    value = prob_accept * nsamples + prob_reject * (ET + nsamples + reset_cost)

    # compute intermediate values
    steps = len(bdy.values)
    if steps > 0:
        dt = (bdy.accept - bdy.reject) / steps
        value += sum(dt * Xnew.pdf(y) * (nsamples + ETy) for y, ETy in bdy.values.items())

    return value

def normal_rejection(bdy, ET, prior, data, nsamples, reset_cost=0):
    ub = bdy.accept
    lb = -10

    x = bdy.reject-1

    itr = 0
    while ub - lb > 1e-4 and itr < 100:
        itr += 1
        Vx = normal_value(x, bdy, ET, prior, data, nsamples=nsamples, reset_cost=reset_cost)
        if Vx < ET:
            ub = x
        else:
            lb = x
        x = ub / 2 + lb / 2

    return x

def normal_backtrack(bdy, s, prior, data, ET, alpha=0.05, steps=50, nsamples=1, reset_cost=0):
    n = bdy.n - nsamples

    # find acceptance boundary
    accept = normal_acceptance(n, s, prior, data, alpha)

    # find rejection boundary
    reject = normal_rejection(bdy, ET, prior, data, nsamples, reset_cost=reset_cost)

    # find values
    dt = (accept - reject) / steps
    X = np.linspace(reject + dt/2, accept - dt/2, steps)
    values = {x: normal_value(x, bdy, ET, prior, data,
                              nsamples=nsamples,
                              reset_cost=reset_cost)
              for x in X}

    return boundary(n, accept, reject, values)

def stepsize(n, N):
    c = max(1, N/20)
    gamma = math.log(c) / (N/2 - min(10, N/5))**2
    return np.floor(1+c*np.exp(-gamma * (n - N/2)**2))

def normal_adaptive_backtrack(s, prior, data, N, ET,
                              alpha=0.05, steps=50, stepsize=stepsize, reset_cost=0):
    accept = normal_acceptance(N, s, prior, data, alpha)
    init_bdy = boundary(N, accept, accept, {})

    bdys = [init_bdy]
    while bdys[-1].n > 1:
        nsamples = stepsize(bdys[-1].n, N)
        bdys.append(normal_backtrack(bdys[-1], s, prior, data, ET, alpha, steps=steps, nsamples=nsamples, reset_cost=0))

    bdys.reverse()
    ET = normal_value(0, bdys[0], ET, prior, data, bdys[0].n)

    return bdys, ET

def binomial_acceptance(N, s, prior, X=None, alpha=0.05):
    if X == None:
        X = int(s * N)+1

    if X > N:
        return X

    posterior = d.posterior(prior, d.Binomial(N, 0), [X])
    if posterior.cdf(s) < alpha:
        return X
    return binomial_acceptance(N, s, prior, X=X+1, alpha=alpha)

def binomial_rejection(bdy, ET, prior, nsamples, reset_cost=0):
    ub = bdy.accept
    lb = bdy.n - nsamples

    x = bdy.accept - 1

    while True:
        if binomial_value(x, bdy, ET, prior, nsamples, reset_cost=reset_cost) > ET:
            return x
        x -= 1

def binomial_value(x, bdy, ET, prior, nsamples=1, reset_cost=0):
    n = bdy.n - nsamples
    posterior = d.posterior(prior, d.Binomial(n, 0), [x])

    if x >= bdy.accept:
        return 0

    # distribution of next <nsample> data points
    new_data = d.Binomial(nsamples, posterior.mean)

    # compute values from borders
    prob_accept = 1-new_data.cdf(bdy.accept - x - 1)
    prob_reject = new_data.cdf(bdy.reject - x)
    value = prob_accept * nsamples + prob_reject * (ET + nsamples + reset_cost)

    # compute intermediate values
    value += sum(new_data.pdf(y - x) * (V + nsamples) for y, V in bdy.values.items())

    return value

def binomial_backtrack(bdy, s, prior, ET, alpha=0.05, nsamples=1, reset_cost=0):
    n = bdy.n - nsamples

    # find acceptance boundary
    accept = binomial_acceptance(n, s, prior, alpha=alpha)

    # find rejection boundary
    reject = binomial_rejection(bdy, ET, prior, nsamples)

    # find values
    values = {x: binomial_value(x, bdy, ET, prior,
                                nsamples=nsamples,
                                reset_cost=reset_cost)
              for x in range(reject+1, accept)}

    return boundary(n, accept, reject, values)

def binomial_fast_update(n, nsamples):
    return min(1 + n // 40, nsamples + 1)

def nonadaptive(n, samples):
    return 1

def binomial_adaptive_backtrack(s, prior, N, ET, alpha=0.05, reset_cost=0, sample_update=binomial_fast_update):
    accept = binomial_acceptance(N, s, prior, alpha=alpha)
    init_bdy = boundary(N, accept, accept-1, {})

    bdys = [init_bdy]
    nsamples = 0
    while bdys[-1].n > 1:
        nsamples = sample_update(bdys[-1].n, nsamples)
        bdys.append(binomial_backtrack(bdys[-1], s, prior, ET, alpha, nsamples=nsamples, reset_cost=reset_cost))

    bdys.reverse()
    ET = binomial_value(0, bdys[0], ET, prior, bdys[0].n)

    return bdys, ET


def solve_normal(s, prior, data, N,
                 ET=100, alpha=0.05, cost=0,
                 lb=0, ub=None, verbose=False,
                 steps=50):

    backtracker = lambda ET: normal_adaptive_backtrack(s,
                                                       prior,
                                                       data,
                                                       N,
                                                       ET,
                                                       alpha=alpha,
                                                       reset_cost=cost,
                                                       steps=50)

    boundaries, ET = bisection_search(backtracker, ET, lb, ub, verbose)
    return boundaries, ET


def solve_binomial(s, prior, N,
                   ET=100, alpha=0.05, cost=0,
                   lb=0, ub=None, verbose=False, adaptive=True):

    if adaptive:
        sample_update = binomial_fast_update
    else:
        sample_update = nonadaptive

    backtracker = lambda ET: binomial_adaptive_backtrack(s,
                                                         prior,
                                                         N,
                                                         ET,
                                                         alpha=alpha,
                                                         reset_cost=cost,
                                                         sample_update=sample_update)

    boundaries, ET = bisection_search(backtracker, ET, lb, ub, verbose)
    return boundaries, ET


## Heuristics
def normal_heuristic(s, prior, data, N, T, alpha=0.05, beta=0.3):
    gamma = data.sigma**2 / prior.sigma**2

    boundaries = []
    for n in range(1, N):
        accept = normal_acceptance(n, s, prior, data, alpha=alpha)

        target = normal_acceptance(n+T, s, prior, data, alpha=alpha)
        at_current = d.Gaussian(n/(n+gamma) * target,
                                math.sqrt(n) * data.sigma / (n+gamma))
        reject = at_current.ppf(beta)

        boundaries.append(boundary(n, accept, reject, {}))

    return boundaries

def binomial_heuristic(s, prior, N, T, alpha=0.05, beta=0.3):
    boundaries = []
    for n in range(1, N):
        accept = binomial_acceptance(n, s, prior, alpha=alpha)

        target = binomial_acceptance(n + T, s, prior)/(n + T)
        at_current = d.Binomial(n, target)
        reject = at_current.ppf(beta)

        boundaries.append(boundary(n, accept, reject, {}))

    boundaries.append(boundary(N, accept, accept-1, {}))

    return boundaries

def save_boundary(boundaries, fp):
    json.dump(boundaries, fp)


def load_boundary(fp):
    raw = json.load(fp)
    return [boundary(*x) for x in raw]

