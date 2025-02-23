import argparse
from Experiments import *
import pickle

def main(results_directory,experiment,T,n_runs,alpha,smoothness,method):   
    
    if method=="OLRE":  
        learning_rate=lambda t: 4.0/((t)**((2*r)/(2*r+1)))
        regularization=lambda t: 1/(4*(t**(1/(2*r+1))))
        file_name=results_directory+f"Experiment_{experiment}_alpha{alpha}_smoothness{smoothness}"
        file_name=file_name.replace(".","")
        run_experiments(experiment,T,alpha=alpha,smoothness=smoothness,warming_period=100,n_runs=n_runs,file_name=file_name)  
 
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
    parser.add_argument("--T")  ############## The size of the data set
    parser.add_argument("--n_runs") #### the number of simulations
    parser.add_argument("--alpha",default=0.0)  ## alpha: parameter to upper-bound the likelihood-ratio. It should be in the interval (0,1]. 
    parser.add_argument("--smoothness",default=0.5) ## smoothnes: this parameter regulates the smoothness of the likelihood-ratio with respect to the Hilbert Space (beta in the paper). It should be in the interval [0.5,1].
    parser.add_argument("--method") #### the method to be run 
    
    args=parser.parse_args()
    
    results_directory=str(args.results_directory)
    results_directory= results_directory+'/'
    experiment=int(args.experiment)
    T=int(args.T)
    n_runs=int(args.n_runs)
    smoothness=args.smoothness
    method=str(args.method)
    alpha=float(args.alpha)
  
    main(results_directory,experiment,T,n_runs,alpha,smoothness,method)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    