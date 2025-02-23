# OLRE Online Likelihod-Ratio Estimator. 
Quantifying the difference between two probability density functions, *p* and *q*, using available data, is a fundamental problem in Statistics and Machine Learning. A usual approach for addressing this problem is the density/likelihood-ratio estimation (LRE) between *p* and *q*, which—to our best knowledge—has been investigated mainly for the offline case. In this repository, we introduce a new framework for **online non-parametric LRE (OLRE)** for the setting where pairs of i.i.d. observations *(xₜ ∼ p, xₜ' ∼ q)* are observed over time. The non-parametric nature of our approach has the advantage of being agnostic to the forms of *p* and *q*.
Moreover, we capitalize on the recent advances in **Kernel Methods** and functional minimization to develop an estimator that can be efficiently updated online.

## Goal: 
This repository provides the necessary components to reproduce results similar to those described in the paper 'Online Non-parametric Likelihood Ratio-Estimation by Pearson Divergence Functional Minimization,' accepted at the 27th International Conference on Artificial Intelligence and Statistics (AISTATS).

## Requirements:
- `pandas` 1.2.5
- `numpy` 1.21.0
- `scipy` 1.7.0
- `matplotlib` 3.4.2
- `numba` 0.55.2

## Code Organization

1. **Experiments**: Contains the elements related to the reported experiment in the paper.
2. **Model**: Contains the algorithms described in the papers as well as auxiliary functions related to the simulation of different scenarios.
3. **Results**: The folder where the results can be stored.
4. **example.ipynb**: Jupyter notebook showing how to use the methods.
5. **run_experiments.py, generate_valuation_metrics.py, generate_plots.py**: These scripts generate the required elements to produce similar results to those reported in the paper.

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
