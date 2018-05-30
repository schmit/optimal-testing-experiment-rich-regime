#!/bin/bash

min_ab=200
nrep=100
alpha=0.05
steps=10
n=4000



# adaptive threshold
for s in 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35;
do
    python3 run_experiments.py --test bayesian --s $s --n $n --alpha $alpha --beta -1 --min_ab $min_ab --nrep $nrep --steps $steps &
done

wait


for s in 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35;
do
    for beta in 0.01 0.05 0.1;
    do
        python3 run_experiments.py --test bayesian --s $s --n $n --alpha $alpha --beta $beta --min_ab $min_ab --nrep $nrep --steps $steps &
    done
done


wait
echo All done

