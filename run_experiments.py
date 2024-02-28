import argparse
from Experiments import *
import pickle

def main(results_directory,experiment,T,n_runs,alpha,r,learning_rate,regularization,method):   
    
    if method=="OLRE":  
        learning_rate=lambda t: 4.0/((t)**((2*r)/(2*r+1)))
        regularization=lambda t: 1/(4*(t**(1/(2*r+1))))
        file_name=results_directory+f"Experiment_{experiment}_alpha{alpha}_r{r}_dynamic_learning_rate"
        file_name=file_name.replace(".","")
        run_experiments(experiment,T,alpha,learning_rate,regularization,t_0=100,n_runs=n_runs,file_name=file_name)  
 
    elif method=="KLIEP":  
        file_name=results_directory+f"Experiment_{experiment}_kliep"
        file_name=file_name.replace(".","")
        run_experiments_offline(experiment,T,n_runs=n_runs,file_name=file_name,method=method)
    
    elif method=="RULSIF":
        file_name=results_directory+f"Experiment_{experiment}_alpha{alpha}_rulsif"
        file_name=file_name.replace(".","")
        run_experiments_offline(experiment,T,alpha=alpha,n_runs=n_runs,file_name=file_name,method=method)
        

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment") #### The experiment to be run
    parser.add_argument("--T")  ############## The number of the lenght
    parser.add_argument("--n_runs") #### the number of simulations
    parser.add_argument("--alpha") #### the number of simulations
    parser.add_argument("--smoothness",default=None) #### smoothness parameter
    parser.add_argument("--learning_rate",default=None) #### learning_rate parameter
    parser.add_argument("--regularization",default=None) #### regularization parameter
    parser.add_argument("--method") #### the method to be run 
    
    args=parser.parse_args()
    
    results_directory=str(args.results_directory)
    results_directory= results_directory+'/'
    experiment=int(args.experiment)
    T=int(args.T)
    n_runs=int(args.n_runs)
    
    r=args.smoothness
    if r is not None:
        r=float(r)
        
    learning_rate=args.learning_rate
    if learning_rate is not None:
        learning_rate=float(learning_rate) 
        
    regularization=args.regularization  
    if regularization is not None:
        regularization=float(regularization)
    
    method=str(args.method)
    alpha=float(args.alpha)
  
    main(results_directory,experiment,T,n_runs,alpha,r,learning_rate,regularization,method)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    