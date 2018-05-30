#!/bin/bash

min_ab=200
nrep=100
alpha=0.05

for s in 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35;
do
    echo Experiments for s=$s
    python3 run_experiments.py --test optimal --s $s --n 5000 --alpha $alpha --min_ab $min_ab --nrep $nrep &
done

wait

echo All done
