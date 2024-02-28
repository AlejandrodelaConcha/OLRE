# OLRE
Online Likelihod-Ratio Estimator. 

## Goal: 
This repository provides the necessary components to reproduce results similar to those described in the paper titled 'Online Non-parametric Likelihood Ratio-Estimation by Pearson Divergence Functional Minimization,' accepted at the 27th International Conference on Artificial Intelligence and Statistics (AISTATS).

## Requirements:
pandas     1.2.5
numpy      1.21.0
scipy      1.7.0
pygsp      0.5.1
matplotlib 3.4.2
mpl_toolkits 
numba     0.55.2

## Code Organization

1) Experiments: contains the elements related to the reported experiment to the paper
2) Model: containts the algorithms described in the papers as well as auxiliar functions related with the simulation of different scenarios.
3) Results: the folder where the results can be stored
4) example.ipnyb: jupyter notebook showing how to use the methods
5) run_experiments.py,generate_valuation_metrics.py,generate_plots.py: these codes generate the required elements to produce similar results to the ones reported to the paper 

## How to generate the results reported on the paper ? 

(This may take days in a normal computer)

0) Fix the results directory where the output will be saved /Users/.../Results
1) Run the following:      

############################ RUN the experiments 

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --method "KLIEP"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --method "KLIEP"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --method "KLIEP"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.1 --method "RULSIF"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.1 --method "RULSIF"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.1 --method "RULSIF"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.5 --method "RULSIF"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.5 --method "RULSIF"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.5 --method "RULSIF"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 1.0 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 1.0 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 1.0 --method "OLRE"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 0.5 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 0.5 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.1 --smoothness 0.5 --method "OLRE"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 1.0 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 1.0 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 1.0 --method "OLRE"

python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 1 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 0.5 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 2 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 0.5 --method "OLRE"
python run_experiments.py  --results_directory "C:/Users/..../Results" --experiment 3 --T 10000 --n_runs 100 --alpha 0.5 --smoothness 0.5 --method "OLRE"

############################ Generate the valuation metrics

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --method "KLIEP"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --method "KLIEP"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --method "KLIEP"

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.1 --method "RULSIF"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.1 --method "RULSIF"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.1 --method "RULSIF

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.5 --method "RULSIF"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.5 --method "RULSIF"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.5 --method "RULSIF"

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.1 --smoothness 1.0 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.1 --smoothness 1.0 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.1 --smoothness 1.0 --method "OLRE"

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.1 --smoothness 0.5 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.1 --smoothness 0.5 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.1 --smoothness 0.5 --method "OLRE"

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.5 --smoothness 1.0 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.5 --smoothness 1.0 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.5 --smoothness 1.0 --method "OLRE"

python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 1 --alpha 0.5 --smoothness 0.5 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 2 --alpha 0.5 --smoothness 0.5 --method "OLRE"
python generate_valuation_metrics.py  --results_directory "C:/Users/..../Results" --experiment 3 --alpha 0.5 --smoothness 0.5 --method "OLRE"

########################### Final plots  

python generate_plots.py --results_directory "C:/Users/..../Results" --experiment 1
python generate_plots.py --results_directory "C:/Users/..../Results" --experiment 2
python generate_plots.py --results_directory "C:/Users/..../Results" --experiment 3
