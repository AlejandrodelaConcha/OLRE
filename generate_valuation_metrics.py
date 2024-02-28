# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:00:09 2022

@author: 33768
"""


from Models import *
from Evaluation import *
import matplotlib.pyplot as plt
import pickle 
import argparse
from scipy.special import logsumexp
import scipy

    
def main(results_directory,experiment,alpha,r,learning_rate,regularization,method):    
      
    ############################ Comparison between constant and dynamic
    
    if method=="OLRE":  
        learning_rate=lambda t: 4.0/((t)**((2*r)/(2*r+1)))
        regularization=lambda t: 1/(4*(t**(1/(2*r+1))))
       # file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_r{r}_dynamic_learning_rate"
       # file_name=file_name.replace(".","")
        list_dictionaries,list_theta,list_Kernel=open_results_online(results_directory,experiment,alpha) 
        estimate_errors_online(results_directory,list_dictionaries,list_theta,list_Kernel,experiment,alpha,r)
        
    elif method=="KLIEP":  
      #  file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_offline_"
      #  file_name=file_name.replace(".","")
        list_dictionaries,list_thetas,list_sigmas= open_results_offline(results_directory,experiment,method)
        estimate_errors_offline(results_directory,list_dictionaries,list_thetas,list_sigmas,experiment,method=method)
     
    elif method=="RULSIF":
       # file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_offline_"
        #file_name=file_name.replace(".","")
        list_dictionaries,list_thetas,list_sigmas= open_results_offline(results_directory,experiment,method=method,alpha=alpha)
        estimate_errors_offline(results_directory,list_dictionaries,list_thetas,list_sigmas,experiment,alpha=alpha,method=method)

    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment") #### The experiment to be run
    parser.add_argument("--alpha") #### the number of simulations
    parser.add_argument("--smoothness",default=None) #### smoothness parameter
    parser.add_argument("--method") #### Three types "growing_dictionary","sparse_dictionary","offline" 
    parser.add_argument("--learning_rate",default=None) #### the learning rate if required 
    parser.add_argument("--regularization",default=None) #### the regularizationrate if requires 
    args=parser.parse_args()
    
    results_directory=str(args.results_directory)
    results_directory= results_directory+'/'
    experiment=int(args.experiment)
    
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
    
    main(results_directory,experiment,alpha,r,learning_rate,regularization,method)
    

















