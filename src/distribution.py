import collections, toolz, math, random
from scipy.stats import norm, beta, binom

class Gaussian(collections.namedtuple('Gaussian', "mu sigma")):
    def __add__(self, other):
        return Gaussian(self.mu + other.mu, math.sqrt(self.sigma**2 + other.sigma**2))

    @property
    def tau(self):
        return 1/self.sigma

    def cdf(self, t):
        """
        Evaluate the CDF of Gaussian distribution at t

        Returns: F(t)
        """
        mu, sigma = self
        return float(norm.cdf(t, mu, sigma))

    def pdf(self, t):
        mu, sigma = self
        return float(norm.pdf(t, mu, sigma))

    def ppf(self, z):
        """
        Evaluate the inverse CDF of Gaussian distribution at alpha
        """
        mu, sigma = self
        return float(norm.ppf(z, mu, sigma))

    def sample(self):
        mu, sigma = self
        return float(norm.rvs(mu, sigma))

    def update(self, data, value):
        assert type(data) == Gaussian, "Data must be Gaussian"
        new_sigma = math.sqrt(1/(self.tau**2 + data.tau**2))
        new_mu = new_sigma**2 * (self.mu / self.sigma**2 + value/data.sigma**2)
        return Gaussian(new_mu, new_sigma)

    @property
    def mean(self):
        return self.mu

    @property
    def mode(self):
        return self.mu

    @property
    def entropy(self):
        return 0.5 * math.log(2 * math.pi * math.e * self.sigma**2)

class Binomial(collections.namedtuple("Binomial", "n p")):
    @property
    def mean(self):
        n, p = self
        return n * p

    def sample(self):
        n, p = self
        return int(binom.rvs(n, p))

    def cdf(self, x):
        n, p = self

        if n*p > 25:
            # use normal approximation
            approximation = Gaussian(n*p, math.sqrt(n * p * (1-p)))
            # continuity correction
            return approximation.cdf(x + 0.5)

        return float(binom.cdf(x, n=n, p=p))

    def pdf(self, x):
        n, p = self

        if n * p > 25:
            # use normal approximation
            return self.cdf(x) - self.cdf(x-1)

        return float(binom.pmf(x, n=n, p=p))

    def ppf(self, q):
        n, p = self
        return float(binom.ppf(q, n=n, p=p))

class Beta(collections.namedtuple("Beta", "a b")):
    def cdf(self, t):
        a, b = self
        return float(beta.cdf(t, a, b))

    def pdf(self, t):
        a, b = self
        return float(beta.pdf(t, a, b))

    def ppf(self, z):
        a, b = self
        return float(beta.ppf(z, a, b))

    def sample(self):
        a, b = self
        return float(beta.rvs(a, b))

    def update(self, data, value):
        assert type(data) == Binomial, "Data must be Binomial"

        a, b = self
        new_a = a + value
        new_b = b + data.n - value
        return Beta(new_a, new_b)

    @property
    def mean(self):
        a, b = self
        return a / (a + b)

    @property
    def mode(self):
        a, b = self
        assert a > 1 and b > 1, "Mode does not exist"
        return (a-1) / (a+b-2)


def posterior(prior, data, samples):
    """
    Returns Gaussian posterior based on prior, data and samples

    Args:
        prior: prior Gaussian distribution (e.g. Gaussian(0, sigma0))
        data: distribution of data (e.g. Gaussian(mu, sigma))
        samples: list of samples from data
    """
    return toolz.reduce(lambda prior, sample: prior.update(data, sample),
                        samples,
                        prior)

def cum_posterior(prior, data, samples):
    """
    Returns list of all posteriors based on prior, data and samples.

    See posterior
    """
    return list(toolz.accumulate(lambda prior, sample: prior.update(data, sample),
                                 samples,
                                 prior))


def conditional_expectation(dist,
                            condition,
                            fn=lambda x: x,
                            is_dist=None,
                            nsamples=5000):
    """ Computes the conditional expectation:

    E(fn(X) | condition(x))

    Args:
        dist: distribution of random variable
        condition: conditional, e.g. lambda x: x > 2
        fn: function to evaluate, defaults lambda x: x
        is_dist: importance sampling distribution
        nsim: number of samples that satisfy the condition, defaults to 5000
        """
    if is_dist == None:
        is_dist = dist
    samples = []
    weights = []

    while len(samples) < nsamples:
        x = is_dist.sample()
        if condition(x):
            weight = dist.pdf(x) / is_dist.pdf(x)
            samples.append(fn(x))
            weights.append(weight)
    return sum(w*x for w, x in zip(weights, samples)) / sum(weights)


def hazard(distribution, x):
    if distribution.cdf(x) == 1:
        return 0
    return distribution.pdf(x) / (1-distribution.cdf(x))
