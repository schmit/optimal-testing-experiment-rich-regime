from fixed_test import fixed_test, fixed_test_early_stopping
from boundary_test import boundary_test
from sprt import binomial_msprt_test, binomial_bayesian_test
from solvers import solve_binomial, binomial_heuristic
import util
import distribution as d

import pandas as pd
import json
import time


def load_baseball_data(min_AB=100):
    df = pd.read_csv("../data/batting.csv")


    df["H/AB"] = df["H"] / df["AB"]
    df["HR/H"] = df["HR"] / df["H"]
    df["HR/AB"] = df["HR"] / df["AB"]

    df = df[df.AB > min_AB]
    return df


def summarize(outcomes, config):
    discoveries = [o for o in outcomes if o["reject"] > 0]
    n_discoveries = len(discoveries)

    nulls = [o for o in outcomes if o["p"] < config["s"]]
    non_nulls = [o for o in outcomes if o["p"] >= config["s"]]

    true_discoveries = sum(o["reject"] > 0 for o in non_nulls)
    false_discoveries = sum(o["reject"] > 0 for o in nulls)
    power = true_discoveries / max(1, len(non_nulls))
    type_one_error =  sum(o["reject"] > 0 for o in nulls) / max(1, len(nulls))
    fdp = sum(o["p"] < config["s"] for o in discoveries) / max(1, n_discoveries)

    avg_samples = sum(o["N"] for o in outcomes) / len(outcomes)
    avg_accept = sum(o["N"] for o in outcomes if o["reject"] > 0) / max(1, n_discoveries)
    avg_stop = sum(o["N"] for o in outcomes if o["reject"] <= 0) / max(1, len(outcomes) - n_discoveries)

    return {"n_tests": len(outcomes),
            "n_discoveries": n_discoveries,
            "true_discoveries": true_discoveries,
            "false_discoveries": false_discoveries,
            "true_effects": len(non_nulls),
            "power": power,
            "type_one_error": type_one_error,
            "fdp": fdp,
            "avg_samples": avg_samples,
            "avg_accept": avg_accept,
            "avg_stop": avg_stop}


def run_experiment(test, probabilities, config):
    def _generate_outcome(p):
        reject, N, Sn = test(p)
        return {"p": p, "reject": reject, "N": N, "Sn": Sn}

    all_probabilities = list(probabilities) * config["nrep"]
    outcomes = [_generate_outcome(p) for p in all_probabilities]

    return {"config": config,
            "outcome": outcomes,
            "summary": summarize(outcomes, config)}

def save_experiment(result, fname):
    try:
        with open(fname, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data ={"data": []}

    data["data"].append(result)

    with open(fname, "w") as f:
        json.dump(data, f)

def run_fixed_test_experiment(probs, config):
    t_start = time.time()

    test = fixed_test(config["N"],
                      config["s"],
                      d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1]),
                      alpha=config["alpha"])

    output = run_experiment(test, probs, config)
    output["time"] = time.time() - t_start

    return output

def run_fixed_test_early_stopping_experiment(probs, config):
    t_start = time.time()

    prior = d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1])
    test = fixed_test_early_stopping(config["N"],
                      config["s"],
                      prior,
                      alpha=config["alpha"])

    output = run_experiment(test, probs, config)
    output["time"] = time.time() - t_start
    return output

def run_msprt_experiment(probs, config):
    t_start = time.time()

    prior = d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1])

    test = lambda p: binomial_msprt_test(p,
                      config["s"],
                      prior,
                      alpha=config["alpha"],
                      N=config["N"],
                      steps=config.get("steps", 10))

    output = run_experiment(test, probs, config)
    output["time"] = time.time() - t_start
    return output

def run_bayesian_test(probs, config):
    t_start = time.time()

    beta = config.get("beta", 0.3)

    prior = d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1])

    if beta < 0:
        beta = (1-prior.cdf(config["s"])) * 0.9
        config["adaptive"] = 1
    else:
        config["adaptive"] = 0

    test = binomial_bayesian_test(
                      config["s"],
                      prior,
                      alpha=config["alpha"],
                      beta=beta,
                      N=config["N"],
                      steps=config.get("steps", 1))

    output = run_experiment(test, probs, config)
    output["beta"] = beta
    output["time"] = time.time() - t_start
    return output

def run_optimal_boundary_experiment(probs, config):
    t_start = time.time()


    prior = d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1])

    print("Finding boundary...")
    boundary, ET = solve_binomial(config["s"],
            prior, config["N"], ET=config.get("ET", 5000), alpha=config["alpha"], verbose=True)


    test = lambda p: boundary_test(p, boundary, slack=0)

    output = run_experiment(test, probs, config)
    output["ET"] = ET
    output["time"] = time.time() - t_start
    return output

def run_heuristic_boundary_experiment(probs, config):
    t_start = time.time()

    prior = d.Beta(config["prior"]["params"][0],
                             config["prior"]["params"][1])

    T = config.get("ET", 2000)
    beta = config.get("beta", 0.3)

    print("Finding boundary...")
    boundary = binomial_heuristic(config["s"],
            prior, config["N"], T=T, alpha=config["alpha"], beta=beta)

    test = lambda p: boundary_test(p, boundary, slack=0)

    output = run_experiment(test, probs, config)
    output["ET"] = T
    output["time"] = time.time() - t_start
    output["beta"] = beta

    return output


