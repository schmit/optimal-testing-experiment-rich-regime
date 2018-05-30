import pandas as pd

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# plt.style.use("minimal")

import distribution as d

import random, collections, math
import json, os
from dfply import *
from experiments import summarize, load_baseball_data
import util

def plot_results(plot_df, x, y, ax, err=None, logy=False, legend=False):
    print(y, err)
    # optimal alg
    sdf = plot_df >> mask(X.test == "optimal", X.N == 5000)
    if logy:
        ax.semilogy(sdf[x], sdf[y], "o-", label="optimal")
    else:
        ax.plot(sdf[x], sdf[y], "o-", label="optimal")

    if err is not None:
        ax.fill_between(sdf[x], sdf[y]-sdf[err], sdf[y]+sdf[err], alpha=0.2)


    # heuristic
    sdf = plot_df >> mask(X.test == "heuristic", X.N == 5000)
    ax.plot(sdf[x], sdf[y], "d-", label="heuristic")

    if err is not None:
        ax.fill_between(sdf[x], sdf[y]-sdf[err], sdf[y]+sdf[err], alpha=0.2)

    # fixed alg
    sdf = plot_df >> mask(X.test == "fixed", X.N == 1000)
    ax.plot(sdf[x], sdf[y], "^-", label="fixed")

    if err is not None:
        ax.fill_between(sdf[x], sdf[y]-sdf[err], sdf[y]+sdf[err], alpha=0.2)
    # sdf = plot_df >> mask(X.test == "fixed", X.N == 2000)
    # ax.semilogy(sdf.s, sdf.efficiency, "^-", label="fixed 2k")
    # sdf = plot_df >> mask(X.test == "fixed", X.N == 500)
    # ax.semilogy(sdf.s, sdf.efficiency, "^-", label="fixed 500")

    # fixed-early
    sdf = plot_df >> mask(X.test == "fixed-early", X.N == 1000)
    ax.plot(sdf[x], sdf[y], "v-", label="fixed, early stopping")

    if err is not None:
        ax.fill_between(sdf[x], sdf[y]-sdf[err], sdf[y]+sdf[err], alpha=0.2)

    # msprt
    # sdf = plot_df >> mask(X.test == "msprt", X.N == 4000)
    # ax.plot(sdf[x], sdf[y], "x-", label="msprt")

    # bayesian
    sdf = plot_df >> mask(X.test == "bayesian", X.beta==-1)
    ax.plot(sdf[x], sdf[y], "+-", label="bayesian")

    if err is not None:
        ax.fill_between(sdf[x], sdf[y]-sdf[err], sdf[y]+sdf[err], alpha=0.2)



    if legend:
        ax.legend()




min_AB = 200
alpha = 0.05

batting = load_baseball_data(min_AB)
probs = batting["H/AB"]
ahat, bhat = util.beta_mom(probs)
betahat = d.Beta(ahat, bhat)


df = pd.read_csv("../data/experiment_summary.csv")
df >>= mutate(efficiency = X.n_discoveries / (X.avg_samples * X.n_tests))
df >>= mutate(efficacy = 1/X.efficiency)

df["fdp_err"] = 2*np.sqrt(df.fdp * (1-df.fdp) / df.n_discoveries)
df["power_err"] = 2*np.sqrt(df.power * (1-df.power) / df.true_effects)

plot_df = df >> mask(X.alpha == alpha, X.min_AB == min_AB, X.s < 0.33,
                     X.n_tests == 572100, X.n_discoveries > 0)
plot_df = plot_df.sort_values("s")


f, ax = plt.subplots(2, 2, figsize=(10, 6.25))


hist_ax = ax[0][0]

hist_ax.hist(probs, bins=100, normed=True, label="data", alpha=0.8)

Xvals = np.linspace(min(probs), max(probs), 200)
hist_ax.plot(Xvals, [betahat.pdf(x) for x in Xvals],
             label="Beta({:.1f}, {:.1f})".format(ahat, bhat))
# ax.plot(X, [normalhat.pdf(x) for x in X], label="Normal({:.2f}, {:.2f})".format(muhat, sigmahat))

hist_ax.set_title("Histogram H/AB")
hist_ax.set_xlabel("batting average")
hist_ax.set_ylabel("frequency")
hist_ax.legend()

efficiency_ax = ax[0][1]
fdp_ax = ax[1][0]
power_ax = ax[1][1]

plot_results(plot_df, "s", "efficacy", efficiency_ax, logy=True, legend=False)
plot_results(plot_df, "s", "fdp", fdp_ax, err="fdp_err")
plot_results(plot_df, "s", "power", power_ax, err="power_err", logy=False, legend=True)


efficiency_ax.set_title("Average time to discovery")
efficiency_ax.set_xlabel("threshold")
efficiency_ax.set_ylabel("#samples / #discoveries")
efficiency_ax.set_ylim(3e2, 1e5)


fdp_ax.set_title("False discovery proportion")
fdp_ax.set_xlabel("threshold")
fdp_ax.set_ylabel("#false discoveries / #discoveries")
fdp_ax.axhline(0.05, color="black", alpha=0.2)
fdp_ax.set_ylim(0, 0.08)


power_ax.set_title("Power")
power_ax.set_xlabel("threshold")
power_ax.set_ylabel("#true discoveries / #good batters")
power_ax.set_ylim(0, 0.6)


f.tight_layout()
f.savefig("plots/baseball_results.pdf")

