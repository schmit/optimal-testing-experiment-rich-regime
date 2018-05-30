import click
import time

import util
import experiments

supported_tests = {"fixed": experiments.run_fixed_test_experiment,
                   "fixed-early": experiments.run_fixed_test_early_stopping_experiment,
                   "optimal": experiments.run_optimal_boundary_experiment,
                   "msprt": experiments.run_msprt_experiment,
                   "bayesian": experiments.run_bayesian_test,
                   "heuristic": experiments.run_heuristic_boundary_experiment}

targets = ["H/AB", "HR/H", "HR/AB"]

@click.command()
@click.option("--n",
              default=100,
              help="(Maximum) number of samples")
@click.option("--alpha",
              default=0.05,
              help="significance level")
@click.option("--s",
              default=0.3,
              help="threshold")
@click.option("--test",
              type=click.Choice(supported_tests.keys()),
              default="fixed")
@click.option("--target",
              type=click.Choice(targets),
              default="H/AB")
@click.option("--min_ab",
              default=100,
              help="minimum at bats")
@click.option("--ET", default=5000)
@click.option("--nrep", default=1)
@click.option("--beta", default=0.2)
@click.option("--steps", default=1)
def run_experiment(n, alpha, s, test, et, min_ab, target, nrep, beta, steps):
    t_start = time.time()
    print("Running experiment with {} {} {}".format(n, alpha, s))
    print("Test: {}".format(test))

    df = experiments.load_baseball_data(min_ab)

    probs = df[target]
    ahat, bhat = util.beta_mom(probs)

    config = {"test": test,
             "N": n,
             "s": s,
             "prior": {"distribution": "Beta", "params": [ahat, bhat]},
             "alpha": alpha,
             "beta": beta,
             "ET": et,
             "target": target,
             "min_AB": min_ab,
             "nrep": nrep,
             "steps": steps,
             "timestamp": int(time.time())}


    output = supported_tests[test](probs, config)

    experiments.save_experiment(output, "experiments/{}_n{}_s{}_alpha{}_{:.0f}.json".format(test, n, s, alpha, time.time()))
    t_end = time.time()

    print("Experiment took {:.1f} seconds".format(t_end - t_start))

if __name__ == "__main__":
    run_experiment()


