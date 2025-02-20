# -----------------------------------------------------------------------------------------------------------------
# Title:  Offline_LRE
# Author(s): Alejandro de la Concha
# Initial version:  2021-05-17
# Last modified:    2025-02-17             
# This version:     2025-02-17  
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This code provides the implementation of OLRE
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: Models.Offline_LRE,numpy scipy
# -----------------------------------------------------------------------------------------------------------------
# Key words: OLRE, KERNEL Methods
# ---------------------------------------------------------------------------------------------------------


from Models.Offline_LRE import *
import copy
from numpy.linalg import inv,pinv
import scipy

def r_estimate(x,Kernel,dictionary,theta):
    #### This function compute the output of the estimated likelihood-ratio 
    ### Input:
    # x: point to evaluate 
    # Kernel: kernel function used to approximate the likelihood-ratio
    # dictionary: dictionary used for the approximation 
    # theta: estimated theta parameter 
    ### Output:
    #  estimated likelihood-ratio evaluated at x     
    Kernel.dictionary=transform_data(np.vstack(dictionary))
    theta=1.*theta
    phi=Kernel.k_V(x).dot(theta)
    return phi

def OLRE(data_ref,data_test,warming_p,smoothness=None,alpha=0.1):
    
    assert isinstance(warming_p,)
    
    
    ########## This function is the implementation of the online likelihood-ratio estimation based on the Pearson divergence 
    ## data_ref: data from the distribution x~P
    ## data_test: data from the distribution x~Q
    ## t_0: number of observations used in the initialization 
    ## learning_rates: the parameter related with stochastic approximation 
    ## regularization: the parameter associated with the Tikinov regularization
    ## alpha: parameter to upper-bound the likelihood ratio.
    
    ### Output 
    ## list_dictionaries:list of the used dictionaries at every time t
    ## list_thetas:list of parameters estimated at everytime time t
    ## kernel: kernel used during the approximations 
    
    learning_rate=lambda t: 4.0/((t+warming_p)**((2*smoothness)/(2*smoothness+1)))
    regularization=lambda t: 1/(4*(t+warming_p)**(1/(2*smoothness+1)))

        
    rulsif_= RULSIF(data_ref[:t_0],data_test[:t_0],alpha=alpha)
    lamb=rulsif_.gamma
    kernel=rulsif_.kernel
    dictionary=[]
    kernel.n=0
    kernel.dictionary=[]
    list_dictionaries=[]
    list_thetas=[]
     
    t=warming_p
    new_point_ref=1.*data_ref[t]
    new_point_test=1.*data_test[t]

    dictionary.append(new_point_ref)
    dictionary.append(new_point_test)
    

    theta=np.array([0,learning_rate(t)])
    kernel.dictionary=transform_data(np.vstack(dictionary))
    

    theta=1.*theta
    contribution_ref=kernel.k(new_point_ref).dot(theta)
    contribution_test=kernel.k(new_point_test).dot(theta)

    dictionary=np.vstack(dictionary)
    
    for t in range(t_0+1,len(data_ref)):

        new_point_ref=1.*data_ref[t]
        new_point_test=1.*data_test[t]
        
        dictionary=np.vstack((dictionary,new_point_ref,new_point_test))
        new_phi_ref=kernel.new_phi(dictionary,new_point_ref)[0]
        new_phi_test=kernel.new_phi(dictionary,new_point_test)[0]
         
        gradient_lost_ref=new_phi_ref[:len(new_phi_ref)-2].dot(theta)
        gradient_lost_test=new_phi_test[:len(new_phi_test)-2].dot(theta)
        
 
        theta_candidate=np.hstack(((1.-1.*learning_rate(t)*regularization(t))*theta,-1.*learning_rate(t)*gradient_lost_ref*(1-alpha),
                                   -1.*learning_rate(t)*(alpha*gradient_lost_test-1)))
        
        theta=1.*theta_candidate
        dictionary=1.*dictionary
        
        kernel.dictionary=transform_data(dictionary)
        list_dictionaries.append(1.*kernel.dictionary)
        list_thetas.append(1.*theta)    
        
    return list_dictionaries,list_thetas,kernel

























