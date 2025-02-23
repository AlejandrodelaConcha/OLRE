# -----------------------------------------------------------------------------------------------------------------
# Title:  run_experiments
# Author(s):  
# Initial version:  2022-02-12
# Last modified:    2022-02-12             
# This version:     2022-02-12
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This function generates the experiments appearing in the paper.
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: 
# -----------------------------------------------------------------------------------------------------------------
# Key words:  likelihood-ratio estimation, Pearson-Divergence estimate
# ------------------


from Models import *
from Experiments import *
import pickle
import numpy as np
import copy



def data_experiment_3(N,p=None):  
    ### This function generates the elements comparing the mixture of 5 bivariate Gaussian distribution 
    ## and a Gaussian distribution 
    ## Input
    # N: the number of observations 
    ## Output 
    # data_ref
    # data_test
    gaussian_data=[]
    if p is None: 
        p=(1/5)*np.ones(5)
    u=np.random.choice(np.arange(5),p=p,size=N)
    counts=np.unique(u,return_counts=True)
    means=[np.array([0,0]),np.array([0,5]),np.array([0,-5]),np.array([5,0]),np.array([-5,0])]
    for i in range(len(counts[0])):
        gaussian_data.append(np.random.multivariate_normal(mean=means[i],cov=np.eye(2),size=counts[1][i]))
        
    data_test=np.vstack(gaussian_data)
    index=np.random.choice(len(data_test),size=len(data_test),replace=False)
    data_test=data_test[index]
    data_ref=np.random.multivariate_normal(mean=np.array([0,0]),cov=10*np.eye(2),size=N)
    
    return data_ref,data_test
          

def run_experiments(experiment,T,warming_period,smoothness=0.5,alpha=0.1,n_runs=50,file_name=None):
    ########### This function run the experiments for the online setting described in the paper 
    ### Input
    # experiment= experiment to run 
    # T= size of the samples to preprocess 
    # alpha= alpha related with the regularization of the likelihood ratio 
    # learning_rates: the parameter related with the dual step of the Pseudo-Mirror-Descent 
    # compression_budget: this parameter is used in the sparse approximation carried out via KOMP.
    # sparse_dictionary= whether to use the compression or not
    # n= number of points for initialization 
    # n_runs= number of simulations to run 
    # file_name= name of the file where the results will be saved
    
    list_dictionaries_PEARSON=[]
    list_theta_PEARSON=[]
    list_Kernel_PEARSON=[]
    for i in range(n_runs):
        print(i)
        if experiment==1:
            
            data_ref=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=T)
            data_test=np.random.laplace(loc=0,scale=np.sqrt(0.5),size=T)
            
            dictionary_PEARSON,theta_PEARSON,Kernel_PEARSON=OLRE(data_ref,data_test,warming_period=warming_period,smoothness=smoothness,alpha=alpha)
            theta_PEARSON=[theta_PEARSON[j] for j in range(0,len(theta_PEARSON),100)]
            list_dictionaries_PEARSON.append(copy.deepcopy(dictionary_PEARSON))
            list_theta_PEARSON.append(copy.deepcopy(theta_PEARSON))
            list_Kernel_PEARSON.append(copy.deepcopy(Kernel_PEARSON))
              
  
        if experiment==2:
            data_ref=np.random.multivariate_normal(np.zeros(2),cov=np.eye(2),size=T)
            data_test=np.random.multivariate_normal(np.zeros(2),
                                            cov=np.array([[1,4/5],[4/5,1]]),size=T)
        
            dictionary_PEARSON,theta_PEARSON,Kernel_PEARSON=OLRE(data_ref,data_test,warming_period=warming_period,smoothness=smoothness,alpha=alpha)
            theta_PEARSON=[theta_PEARSON[j] for j in range(0,len(theta_PEARSON),100)]
            list_dictionaries_PEARSON.append(copy.deepcopy(dictionary_PEARSON))
            list_theta_PEARSON.append(copy.deepcopy(theta_PEARSON))
            list_Kernel_PEARSON.append(copy.deepcopy(Kernel_PEARSON))
            
        if experiment==3:
            data_ref,data_test=data_experiment_3(N=T)
        
            dictionary_PEARSON,theta_PEARSON,Kernel_PEARSON=OLRE(data_ref,data_test,warming_period=warming_period,smoothness=smoothness,alpha=alpha)
            theta_PEARSON=[theta_PEARSON[j] for j in range(0,len(theta_PEARSON),100)]
            list_dictionaries_PEARSON.append(copy.deepcopy(dictionary_PEARSON))
            list_theta_PEARSON.append(copy.deepcopy(theta_PEARSON))
            list_Kernel_PEARSON.append(copy.deepcopy(Kernel_PEARSON))

        results_PEARSON={"dictionaries": list_dictionaries_PEARSON,"thetas":list_theta_PEARSON,"Kernel":list_Kernel_PEARSON}
                      
        with open(file_name+".pickle", 'wb') as handle:
            pickle.dump(results_PEARSON, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
def run_experiments_offline(experiment,T,alpha=None,t_0=100,n_runs=50,file_name=None,method=None):
    ########### This function run the experiments for the online setting described in the paper 
    ### Input
    # experiment= experiment to run 
    # T= size of the samples to preprocess 
    # alpha= alpha related with the regularization of the likelihood ratio 
    # n= number of points for initialization 
    # n_runs= number of simulations to run 
    # file_name= name of the file where the results will be saved
    
    list_dictionaries=[]
    list_theta=[]
    list_sigmas=[]

    for i in range(n_runs):
        print(i)
        if experiment==1:
           
            data_ref=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=T+t_0)
            data_test=np.random.laplace(loc=0,scale=np.sqrt(0.5),size=T+t_0)
            
   
        if experiment==2:
            data_ref=np.random.multivariate_normal(np.zeros(2),cov=np.eye(2),size=T+t_0)
            data_test=np.random.multivariate_normal(np.zeros(2),
                                            cov=np.array([[1,4/5],[4/5,1]]),size=T+t_0)
       
            
        if experiment==3:
           data_ref,data_test=data_experiment_3(N=T+t_0)
      
        if method=="KLIEP":
            dictionaries_,thetas_,sigmas_=klieps(data_ref,data_test)
   
        if method=="RULSIF":
            dictionaries_,thetas_,sigmas_,lambs=rulsifs(data_ref,data_test,alpha=alpha)
         
        list_dictionaries.append(copy.deepcopy(dictionaries_))
        list_theta.append(copy.deepcopy(thetas_))
        list_sigmas.append(copy.deepcopy(sigmas_))
         
        
    results={"dictionaries": list_dictionaries,"thetas":list_theta,"sigmas":list_sigmas}            
    print(file_name)     
    with open(file_name+".pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)        











