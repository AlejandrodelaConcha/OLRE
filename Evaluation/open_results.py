# -----------------------------------------------------------------------------------------------------------------
# Title:  open_results 
# Author(s): Alejandro de la Concha
# Initial version:  2022-02-12
# Last modified:    2024-02-28             
# This version:     2024-02-28
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): Functions to access the produced results efficiently. 
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy, Evaluation
# -----------------------------------------------------------------------------------------------------------------
# Key words:  likelihood-ratio estimation, Pearson-Divergence estimatation
# -------------------------------------------------------------------------

import pickle
import numpy as np
from Evaluation import *

def open_results_online(results_directory,experiment,alpha,smoothness):
    ########## Function related with the dynamic and the constant learning rates 
    ### Input:
    # resuts_directory= the directory where the results will be stored 
    # experiment= the number of the experiment
    # alpha= the regularization parameter 
    # r= the smothness parameter 
    ### Output
    # list_dictionaries= the list of the elements of each of the dictionaries 
    # list_theta = the parameters that have been fitted according to the model 
    # list_kernel = the kernel that has been used for estimation 
    file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_smoothness{smoothness}"
   
  
    file_name=file_name.replace(".","")
    
    with open(file_name+".pickle", 'rb') as f:
        results=  pickle.load(f)
    
    list_dictionaries=results["dictionaries"]
    list_theta=results["thetas"]
    list_Kernel=results["Kernel"]
  
    return list_dictionaries,list_theta,list_Kernel
   

def open_results_offline(results_directory,experiment,method,alpha=None):
    ########## Function for replicating the offline experiments 
    ### Input:
    # resuts_directory= the directory where the results will be stored 
    # experiment= the number of the experiment
    # method= the offline method to evaluate "RULSIF" vs KLIEP
    # alpha= the regularization parameter 
    ### Output
    # list_dictionaries= the list of the elements of each of the dictionaries 
    # list_thetas= the list of fitted parameters with the model 
    # list_sigmas= the list of width parameters associated with the kernel
     
    if method=="KLIEP":
        file_name=results_directory+f"Experiment_{experiment}_kliep"
        file_name=file_name.replace(".","")
        with open(file_name+".pickle", 'rb') as f:
            results=pickle.load(f)
            
    elif method=="RULSIF":
        file_name=results_directory+f"Experiment_{experiment}_alpha{alpha}_rulsif"
        file_name=file_name.replace(".","")
        with open(file_name+".pickle", 'rb') as f:
            results=pickle.load(f)
        
    
    list_dictionaries=results["dictionaries"]
    list_thetas=results["thetas"]
    list_sigmas=results["sigmas"]
    
    return list_dictionaries,list_thetas,list_sigmas
