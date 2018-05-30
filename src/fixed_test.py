import distribution as d

from solvers import binomial_acceptance

def fixed_test(N, s, prior, alpha=0.05):
    """ Fixed test: sample N observations and then look at posterior """
    accept = binomial_acceptance(N, s, prior, alpha=alpha)
    def _test(p):
        data = d.Binomial(N, p)
        Sn = data.sample()
        posterior = prior.update(data, Sn)
        if posterior.cdf(s) < alpha:
            return 1, N, Sn
        # if Sn >= accept:
        #     return 1, N, Sn
        return 0, N, Sn

    return _test

def fixed_test_early_stopping(N, s, prior, steps=1, alpha=0.05):
    """ Fixed test but stop early if acceptance threshold is crossed """
    accepts = [binomial_acceptance(n, s, prior, alpha=alpha) for n in range(1, N+1)]

    def _test(p):
        n = steps
        Sn = 0
        data = d.Binomial(steps, p)
        while n < N:
            Sn += data.sample()
            if accepts[n] < Sn:
                return 1, n, Sn
            n += step
        return 0, N, Sn

    return _test
