
import distribution as d

import math, time

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# plt.style.use("minimal")


from solvers import solve_normal, solve_binomial
from solvers import boundary, normal_acceptance, normal_heuristic
from solvers import binomial_acceptance, binomial_heuristic


def plogpq(p, q):
    if p <= 0:
        return 0
    if q <= 0:
        return math.inf
    return p * math.log(p/q)

def binomial_kldiv(p, q):
    assert p.n == q.n
    return plogpq(p.p, q.p) * p.n + plogpq(1-p.p, 1-q.p) * p.n

def normal_kldiv(p, q):
    return math.log(q.sigma / p.sigma) + (p.sigma**2 + (p.mu - q.mu)**2) / (2*q.sigma**2) - 1/2


N = 500
n_s = 1.5
n_prior = d.Gaussian(0, 1)
n_data = d.Gaussian(0, 3)

normal_boundaries, normal_ET = solve_normal(n_s, n_prior, n_data, N, ET=307, lb=300, ub=310, verbose=True)
n_heuristic = normal_heuristic(n_s, n_prior, n_data, N, normal_ET, beta=0.4)

bin_s = 0.4
bin_prior = d.Beta(3, 10)

bin_boundaries, bin_ET = solve_binomial(bin_s, bin_prior, N, ET=200, verbose=True, adaptive=False)
bin_heuristic = binomial_heuristic(bin_s, bin_prior, N, bin_ET, beta=0.3)



f, axes = plt.subplots(1, 2, figsize=(7, 3))

ax = axes[0]

gamma = (n_prior.tau / n_data.tau)**2
ax.plot([bdy.n for bdy in normal_boundaries],
        [bdy.accept for bdy in normal_boundaries], color="green")
ax.plot([bdy.n for bdy in normal_boundaries],
        [bdy.reject for bdy in normal_boundaries], color="red")

ax.plot([bdy.n for bdy in n_heuristic],
        [bdy.reject for bdy in n_heuristic], "--", color="blue")

ax.fill([bdy.n for bdy in normal_boundaries] + [N],
        [bdy.accept for bdy in normal_boundaries] + [4], color="green", alpha=0.3)
ax.fill([bdy.n for bdy in normal_boundaries] + [N],
        [bdy.reject for bdy in normal_boundaries] + [-4], color="red", alpha=0.3)

ax.axhline(n_s, color="black")
ax.set_xlim(0, 150)
ax.set_ylim(normal_boundaries[0].reject, 3)

ax.set_title("Conjugate-Normal boundaries")
ax.set_xlabel("number of samples")
ax.set_ylabel("MAP for $\mu$")

ax = axes[1]

a, b = bin_prior
ax.plot([bdy.n for bdy in bin_boundaries],
        [(bdy.accept+a-1) / (bdy.n+a+b-2) for bdy in bin_boundaries],
        color="green")
ax.plot([bdy.n for bdy in bin_boundaries],
        [(bdy.reject+a-1) / (bdy.n+a+b-2) for bdy in bin_boundaries],
        color="red")
ax.plot([bdy.n for bdy in bin_heuristic],
        [(bdy.reject+a-1) / (bdy.n+a+b-2) for bdy in bin_heuristic],
        "--", color="blue")

ax.fill([bdy.n for bdy in bin_boundaries] + [N, 0],
        [(bdy.accept+a-1) / (bdy.n+a+b-2) for bdy in bin_boundaries] + [1, 1], color="green", alpha=0.3)
ax.fill([bdy.n for bdy in bin_boundaries] + [N, 0],
        [(bdy.reject+a-1) / (bdy.n+a+b-2) for bdy in bin_boundaries] + [0, 0], color="red", alpha=0.3)


ax.axhline(bin_s, color="black")
ax.set_xlim(0, 250)
ax.set_ylim(0.1, 0.65)

ax.set_title("Beta-binomial boundaries")
ax.set_xlabel("number of samples")
ax.set_ylabel("MAP for $p$")

f.tight_layout()
f.savefig("plots/boundaries.pdf")


