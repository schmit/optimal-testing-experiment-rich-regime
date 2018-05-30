# Optimal testing in the experiment-rich regime
Code to replicate results in "Optimal Testing in the Experiment-rich Regime" paper

## Requirements

- pandas
- dpfly
- matplotlib
- numpy
- scipy
- toolz
- click


## How-to

### Generating data from simulations

- Create experiments folder
- Run all the bash scripts to generate outcomes for tests. This can take many hours
- Run the `summarize_experiments.py` script to parse outcomes

Note, the parsed experiments from our simulations can be found in `data/experiment_summary.csv`

### Creating plots

- Create plots folder
- Run `create_baseball_plot.py` to generate the baseball result plot
- Run `crease_boundary_plot.py` to generate the boundary plot including the heuristic
boundary







