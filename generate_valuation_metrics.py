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

    
def main(results_directory,experiment,alpha,smoothness,method):    
      
    ############################ Comparison between constant and dynamic
    
    if method=="OLRE":  
        list_dictionaries,list_theta,list_Kernel=open_results_online(results_directory,experiment,alpha,smoothness) 
        estimate_errors_online(results_directory,list_dictionaries,list_theta,list_Kernel,experiment,alpha,smoothness)
        
    elif method=="KLIEP":  
        list_dictionaries,list_thetas,list_sigmas= open_results_offline(results_directory,experiment,method)
        estimate_errors_offline(results_directory,list_dictionaries,list_thetas,list_sigmas,experiment,method=method)
     
    elif method=="RULSIF":
        list_dictionaries,list_thetas,list_sigmas= open_results_offline(results_directory,experiment,method=method,alpha=alpha)
        estimate_errors_offline(results_directory,list_dictionaries,list_thetas,list_sigmas,experiment,alpha=alpha,method=method)

    
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment") #### The experiment to be run
    parser.add_argument("--alpha",default=0.0) #### the number of simulations
    parser.add_argument("--smoothness",default=0.5) #### smoothness parameter
    parser.add_argument("--method") #### Three types "growing_dictionary","sparse_dictionary","offline" 
    args=parser.parse_args()
    
    results_directory=str(args.results_directory)
    results_directory= results_directory+'/'
    experiment=int(args.experiment)
    smoothness=float(args.smoothness)
    method=str(args.method)
    alpha=float(args.alpha)
       
    main(results_directory,experiment,alpha,smoothness,method)
    

















