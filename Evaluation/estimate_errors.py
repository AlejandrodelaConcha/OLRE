# -----------------------------------------------------------------------------------------------------------------
# Title:  estimate_errors
# Author(s):  
# Initial version:  2022-02-12
# Last modified:    2022-02-12             
# This version:     2022-02-12
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This function computes the convergence metrics associated with each of the online likelihood-ratio estimators 
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy,Models.aux_functions
# -----------------------------------------------------------------------------------------------------------------
# Key words:  likelihood-ratio estimation, Pearson-Divergence estimatation, error_values 
# ------------------

import numpy as np 
from Models import *
from Experiments import *
import pickle

def error_graph(data_ref_validation,data_test_validation,list_dictionaries,list_thetas,Kernel,real_likelihood,alpha=0.1):
    
    ######## This functions generate the convergence graphs of the different algorithms. 
    
    ###### Before running this functions, the results of the experiments should be produced. 
    ########### Input 
    ## data_ref_validation: data from the distribution x~P
    ## data_test_validation: data from the distribution x~Q
    ## list_dictionaries: the dicitionaries used to estimate the likelihood ratio at each time stamp 
    ## list_thetas: estimated parameters via the online learning method 
    ## Kernel: kernel function used to approximate the likelihood-ratio
    ## real_likelihood: function to compute the exact relative likelihood-ratio 
    ## alpha: parameter to upper-bound the likelihood ratio.

    ########### Output 
    ## error_cost_function: difference between the cost function evaluated at the estimated likelihood-ratio and the real likelihood ratio
    ## error_divergence: the L2 distance between the estimated likelihood-ratio and the real likelihood-ratio

    error_cost_function=[]
    error_divergence=[]
    
    r_l_ref=real_likelihood(data_ref_validation)
    r_l_test=real_likelihood(data_test_validation)
    
   
    for i in range(0,len(list_thetas)):
        Kernel.dictionary=1.*list_dictionaries[i]
        phi_ref= Kernel.k_V(data_ref_validation).dot(list_thetas[i])
        phi_test=Kernel.k_V(data_test_validation).dot(list_thetas[i])

        if alpha>0:
            L_f_t=-1.0*np.mean(phi_test)+0.5*((1-alpha)*np.mean(phi_ref**2)+alpha*np.mean(phi_test**2))
            L_f=-1.0*np.mean(r_l_test)+0.5*((1-alpha)*np.mean(r_l_ref**2)+alpha*np.mean(r_l_test**2))
        else:
            L_f_t=-1.*np.mean(phi_test)+0.5*(np.mean(phi_ref**2))
            L_f=-1.*np.mean(r_l_test)+0.5*(np.mean(r_l_ref**2))
            
        if alpha>0:
            L_2_distance=(1-alpha)*np.mean((r_l_ref-phi_ref)**2)
            L_2_distance+=alpha*np.mean((r_l_test-phi_test)**2)
        else:
            L_2_distance=np.mean((real_likelihood(data_ref_validation)-phi_ref)**2)
            
        error_cost_function.append(L_f_t-L_f)
        error_divergence.append(1.*L_2_distance)
                  
    return error_cost_function,error_divergence




def error_graph_offline(data_ref_validation,data_test_validation,list_dictionaries,list_sigmas,list_thetas,real_likelihood,method="RULSIF",alpha=0.1,Nystrom=False):
    
    ######## This functions generate the convergence graphs of the different algorithms. 
    
    ###### Before running this functions, the results of the experiments should be produced. 
    ########### Input 
    ## data_ref_validation: data from the distribution x~P
    ## data_test_validation: data from the distribution x~Q
    ## list_dictionaries: the dicitionaries used to estimate the likelihood ratio at each time stamp 
    ## list_thetas: estimated parameters via the online learning method 
    ## Kernel: kernel function used to approximate the likelihood-ratio
    ## real_likelihood: function to compute the exact relative likelihood-ratio 
    ## method: the algorithm that was used to produce the results 
    ## alpha: parameter to upper-bound the likelihood ratio.

    ########### Output 
    ## error_cost_function: difference between the cost function evaluated at the estimated likelihood-ratio and the real likelihood ratio
    ## error_divergence: the L2 distance between the estimated likelihood-ratio and the real likelihood-ratio

    error_cost_function=[]
    error_divergence=[]

    for i in range(0,len(list_thetas)):
        if Nystrom: 
            Kernel=Nystrom_Kernel(dictionary=list_dictionaries[i],sigma=list_sigmas[i])
        else:
            Kernel=Gauss_Kernel(dictionary=list_dictionaries[i],sigma=list_sigmas[i])
            
        phi_ref= Kernel.k_V(data_ref_validation).dot(list_thetas[i])
        phi_test=Kernel.k_V(data_test_validation).dot(list_thetas[i])
        r_l_ref=real_likelihood(data_ref_validation)
        r_l_test=real_likelihood(data_test_validation)    
       
        if method=="KLIEP":
            L_f_t=np.mean(np.log(phi_test+1e-6))
            L_f=np.mean(np.log(real_likelihood(data_test_validation)+1e-6))
            L_2_distance=np.mean((r_l_ref-phi_ref)**2)      
            
            error_cost_function.append(L_f_t-L_f)
            error_divergence.append(1.*L_2_distance)
         
        if method=="RULSIF": 
             
            if alpha>0:
                 r_test=1.*phi_test
              #   r_test[r_test<0]=0
                 r_ref=1.*phi_ref
              #   r_ref[r_ref<0]=0
                 L_f_t=-1.0*np.mean(r_test)+0.5*((1-alpha)*np.mean(r_ref**2)+alpha*np.mean(r_test**2))
                 L_f=-1.0*np.mean(r_l_test)+0.5*((1-alpha)*np.mean(r_l_ref**2)+alpha*np.mean(r_l_test**2))
            else:
                 L_f_t=-1.*np.mean(r_test)+0.5*(np.mean(r_ref**2))
                 L_f=-1.*np.mean(r_l_test)+0.5*(np.mean(r_l_ref**2))
             
            if alpha>0:
                 L_2_distance=(1-alpha)*np.mean((real_likelihood(data_ref_validation)-r_ref)**2)
                 L_2_distance+=alpha*np.mean((real_likelihood(data_test_validation)-r_test)**2)
            else:
                 L_2_distance=np.mean((real_likelihood(data_ref_validation)-r_ref)**2)
            
            error_cost_function.append(L_f_t-L_f)
            error_divergence.append(1.*L_2_distance)
        
    return error_cost_function,error_divergence


def estimate_errors_online(results_directory,list_dictionaries,list_theta,list_kernel,experiment,alpha,r,learning_rate=None):
    
    dictionary="dynamic"
    file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_r{r}_"+dictionary+f"_learning_rate_Error"
       
    list_errors_PEARSON=[]
    list_L2_distance_PEARSON=[]
    
    if experiment==1:
        data_ref_validation=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=10000)
        data_test_validation=np.random.laplace(loc=0,scale=np.sqrt(0.5),size=10000)
        
        for i in range(len(list_dictionaries)):
            print(i)         
            errors_PEARSON,L2_distance=error_graph(data_ref_validation,data_test_validation,list_dictionaries[i],list_theta[i],list_kernel[i],
                                                 real_likelihood=lambda x: r_uniform_laplace(x,alpha=alpha),alpha=alpha)
        
            list_errors_PEARSON.append(np.array(errors_PEARSON))
            list_L2_distance_PEARSON.append(np.array(L2_distance))
        
    elif experiment==2:
        
        data_ref_validation=np.random.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),size=10000)
        data_test_validation=np.random.multivariate_normal(mean=np.zeros(2),cov=np.array([[1,4/5],[4/5,1]]),size=10000)
        
        for i in range(len(list_dictionaries)):
            print(i)
        
            errors_PEARSON,L2_distance=error_graph(data_ref_validation,data_test_validation,list_dictionaries[i],list_theta[i],list_kernel[i],
                                                 real_likelihood=lambda x: r_bivariate_normal(x,alpha=alpha),alpha=alpha)
        
            list_errors_PEARSON.append(np.array(errors_PEARSON))
            list_L2_distance_PEARSON.append(np.array(L2_distance))
            
    elif experiment==3:
        
        data_ref_validation,data_test_validation=data_experiment_3(N=10000)
        
        for i in range(len(list_dictionaries)):
            print(i)
        
            errors_PEARSON,L2_distance=error_graph(data_ref_validation,data_test_validation,list_dictionaries[i],list_theta[i],list_kernel[i],
                                                 real_likelihood=lambda x: r_normal_mixture(x,alpha=alpha),alpha=alpha)
        
            list_errors_PEARSON.append(np.array(errors_PEARSON))
            list_L2_distance_PEARSON.append(np.array(L2_distance))
            
    error_L2_distance={"error": list_errors_PEARSON,"L2":list_L2_distance_PEARSON}  
    
    file_name=file_name.replace(".","")    
    with open(file_name+".pickle", 'wb') as handle:
        pickle.dump( error_L2_distance,handle, protocol=pickle.HIGHEST_PROTOCOL)
    


def estimate_errors_offline(results_directory,list_dictionaries,list_thetas,list_sigmas,experiment,alpha=None,method="KLIEP"):
    
    list_errors=[]
    list_L2_distance=[]
    
    if experiment==1:
        data_ref_validation=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=10000)
        data_test_validation=np.random.laplace(loc=0,scale=np.sqrt(0.5),size=10000)
        
        if method=="KLIEP":
            alpha=0
        
        for i in range(len(list_thetas)):
            print(i)
            
            errors_,L2_distance_=error_graph_offline(data_ref_validation,data_test_validation,list_dictionaries[i],list_sigmas[i],list_thetas[i],
                                                 real_likelihood=lambda x: r_uniform_laplace(x,alpha=alpha),method=method,alpha=alpha)
            list_errors.append(np.array(errors_))
            list_L2_distance.append(np.array(L2_distance_))
        
    elif experiment==2:
        
        data_ref_validation=np.random.multivariate_normal(mean=np.zeros(2),cov=np.eye(2),size=10000)
        data_test_validation=np.random.multivariate_normal(mean=np.zeros(2),cov=np.array([[1,4/5],[4/5,1]]),size=10000)
        
        if method=="KLIEP":
            alpha=0
        
        for i in range(len(list_thetas)):
            print(i)
            
 
            errors_,L2_distance_=error_graph_offline(data_ref_validation,data_test_validation,list_dictionaries[i],list_sigmas[i],list_thetas[i],
                                                 real_likelihood=lambda x: r_bivariate_normal(x,alpha=alpha),method=method,alpha=alpha)
            list_errors.append(np.array(errors_))
            list_L2_distance.append(np.array(L2_distance_))
        

    elif experiment==3:
        
        data_ref_validation,data_test_validation=data_experiment_3(N=10000)
        
        if method=="KLIEP":
            alpha=0
        
        for i in range(len(list_thetas)):
            print(i)
 
            errors_,L2_distance_=error_graph_offline(data_ref_validation,data_test_validation,list_dictionaries[i],list_sigmas[i],list_thetas[i],
                                                 real_likelihood=lambda x:  r_normal_mixture(x,alpha=alpha),method=method,alpha=alpha)
            list_errors.append(np.array(errors_))
            list_L2_distance.append(np.array(L2_distance_))
        
            
        
    error_={"error": list_errors,"L2":list_L2_distance}  
    
    if method=="KLIEP":
        file_name=results_directory+f"Experiment_{experiment}_errors_offline_kliep"
    elif method=="RULSIF": 
        file_name=results_directory+f"Experiment_{experiment}_alpha{alpha}_errors_offline_rulsif"
        
        
    file_name=file_name.replace(".","")    
    with open(file_name+".pickle", 'wb') as handle:
        pickle.dump(error_,handle, protocol=pickle.HIGHEST_PROTOCOL)
        

