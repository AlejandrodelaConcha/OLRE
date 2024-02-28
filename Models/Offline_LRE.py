# -----------------------------------------------------------------------------------------------------------------
# Title:  Offline_LRE
# Author(s): Alejandro de la Concha
# Initial version:  2021-05-17
# Last modified:    2024-02-28              
# This version:     2024-05-28
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this function is to provide implementations of KLIEP and RULSIF
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy,numba 
# -----------------------------------------------------------------------------------------------------------------
# Key words: KLIEP,RULSIF
# ---------------------------------------------------------------------------------------------------------

import numpy as np
import copy
from numpy import linalg as LA
from Models.aux_functions import *
from scipy.sparse.linalg import eigsh

def r_estimate_offline(x,Kernel,theta):
    #### This function compute the output of the estimated likelihood-ratio 
    ### Input:
    # x: point to evaluate 
    # Kernel: kernel function used to approximate the likelihood-ratio
    # theta: estimated theta parameter    
    ### Output:
    #  estimated likelihood-ratio evaluated at x 
    theta=1.*theta
    phi=Kernel.k_V(x).dot(theta)
    return phi


    
class KLIEP():
########## Class implementing KLIEP 
# Based on the paper "Direct importance estimation with model selection and its application to covariate shift adaptation."  Sugiyama, M 2007
    def __init__(self,data_ref,data_test,k_cross_validation=5,tol=1e-2,lr=1e-5,verbose=False):     
    
    ## Input
    # data_ref: data points representing the distribution p(.)
    # data_test: data points representing the distribution q(.) 
    # tol: level of accepted tolerence in the estimation
    # k_cross_validation: number of splits to do for cross validation 
    # tol= tolerated error 
    # lr: learning rate associated with the optimization problem
    # verbose: function to print the fitting results 
        
        self.data_ref=transform_data(data_ref)
        self.data_test=transform_data(data_test)
        self.tol=tol
        self.k_cross_validation=k_cross_validation
        self.lr=lr
        kernel_1=self.initializalize_kernel(all_test) 
        self.kernel=self.model_selection(kernel_1,verbose=verbose)

     
    def initializalize_kernel(self):

    ## Output
    # kernel_1: A initialized kernel with a given dictionary 
        if all_test:
            sigma=get_sigma(self.data_test)
            dictionary=1*self.data_test
     
        else:
            if self.data_test.shape[0]>100:
                index=np.random.choice(len(self.data_test),replace=False,size=100)
                sigma=get_sigma(self.data_test[index])
                dictionary=1*self.data_test[index]
            else:
                sigma=get_sigma(self.data_test)
                dictionary=1*self.data_test
       
        if sigma==0:
            sigma=1e-6
                
        kernel_1=Gauss_Kernel(dictionary,sigma=sigma)
                   
        return kernel_1
    
    def fit(self,data_ref=None,data_test=None,kernel=None,verbose=False): 
    #### Function estimating the theta parameter for a given set of observations comming from p(.) and q(.)    
    ## Input
    # data_ref: data points representing the distribution p(.)
    # data_test: data points representing the distribution q(.) 
    # kernel: kernel to be used in the method
    # verbose: whether or not to print the results of the optimization process    
    ## Output
    # theta: list with n_nodes elements, each element is the estimated parameter for the node v 
    
        if  data_ref is not None:
            data_ref=transform_data(data_ref)
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=transform_data(data_test)
        else:
            data_test=self.data_test
    
        if kernel is None:
            kernel=self.kernel

        EPS=1e-6
    
        phi_test = kernel.k_V(data_test)
        phi_ref= kernel.k_V(data_ref)
        
        b= np.mean(phi_ref, axis=0)
        b = b.reshape(-1, 1)

        theta= np.ones((kernel.n, 1)) / kernel.n
        previous_objective = -np.inf
        objective = np.mean(np.log(np.dot(phi_test, theta) + EPS))
        if verbose:
                print("Alpha's optimization : iter %i -- Obj %.4f"%(0, objective))
        k = 0
        n_iter=0
        while np.abs(objective-previous_objective) > self.tol and n_iter<10000:
            n_iter+=1
            if verbose:
                if k%100 == 0:
                    print("Alpha's optimization : iter %i -- Obj %.4f"%(k, objective))
                    print( np.abs(objective-previous_objective) )

           
            previous_objective = 1.*objective
            theta = theta+ self.lr * np.dot(
                np.transpose(phi_test), 1./(np.dot(phi_test, theta) + EPS)
            )
            theta = theta+ b * ((((1-np.dot(np.transpose(b), theta)) /
                            (np.dot(np.transpose(b), b) + EPS))))
            theta[theta<0] = 0
            theta= theta/(np.dot(np.transpose(b), theta) + EPS)
            objective = np.mean(np.log(np.dot(phi_test, theta) + EPS))
            k += 1
            
          
        return theta
    
    def model_selection(self,kernel,verbose=False):
        #### This function identifies the optimal hyperparameters related to the data         
        ###### Input
        # kernel: kernel to be used in the method
        # verbose: whether or not to print intermediate steps      
        #### Output
        # kernel_: kernel initialized with the selected dictionary and the optimal value of alpha
 
        sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])*kernel.sigma
        max_ = -np.inf
        j_scores_ = {}
        best_params_={}
        N_test=self.data_test.shape[0]
        N_ref=self.data_ref.shape[0]
   
        ref_index_validation=[np.arange(N_ref)[int(i*(N_ref/self.k_cross_validation)):int((i+1)*((N_ref/self.k_cross_validation)))] for i in range(self.k_cross_validation)]
        test_index_validation=[np.arange(N_test)[int(i*(N_test/self.k_cross_validation)):int((i+1)*((N_test/self.k_cross_validation)))] for i in range(self.k_cross_validation)]
        ref_index_train=[]
        test_index_train=[]
        
        for i in range(self.k_cross_validation):
            if i==0:
                ref_index_train.append(np.hstack(ref_index_validation[i+1:]))
                test_index_train.append(np.hstack(test_index_validation[i+1:]))
            elif i==(self.k_cross_validation-1):
                ref_index_train.append(np.hstack(ref_index_validation[:i]))
                test_index_train.append(np.hstack(test_index_validation[:i]))
            else:
                ref_index_train.append(np.hstack((np.hstack(ref_index_validation[:i]),np.hstack(ref_index_validation[i+1:]))))
                test_index_train.append(np.hstack((np.hstack(test_index_validation[:i]),np.hstack(test_index_validation[i+1:]))))


        cost_vector=np.zeros(len(sigma_list))
        for s in range(len(sigma_list)):
            kernel_=Gauss_Kernel(kernel.dictionary,sigma=sigma_list[s])
            aux_cost=np.zeros(self.k_cross_validation)
           
            for i in range(self.k_cross_validation):
                theta=self.fit(self.data_ref[ref_index_train[i]],self.data_test[test_index_train[i]],kernel=kernel_,verbose=False)
                phi=kernel.k_V(self.data_test[test_index_validation[i]])
                aux_cost[i]=np.mean(np.log(phi.dot(theta)+1e-6))
            
            aux_params={"sigma":sigma_list[s]}
            j_scores_[str(aux_params)]=np.mean(aux_cost)
            if verbose:
                print("Parameters %s -- J-score = %.3f"% (str(aux_params), j_scores_[str(aux_params)]))
            if j_scores_[str(aux_params)] > max_:
                best_params_ = aux_params
                max_ = j_scores_[str(aux_params)]  
                   
        kernel_=Gauss_Kernel(dictionary=kernel.dictionary,sigma=best_params_["sigma"])
                
        return  kernel_
           
    def KL_divergence(self,data_ref=None,data_test=None,theta=None):
    #### Function estimating the Kulback Liebler Divergence 
    ## Input
    # data_ref: data points representing the distribution p(.)
    # data_test: data points representing the distribution q(.)
    ## Output 
    # score: Pearson Divergence at the node level 
    
        data_ref=transform_data(data_ref)
        data_test=transform_data(data_test)  
        if theta is None:
            theta=self.fit(data_ref,data_test)
        
        score=np.mean(np.log(self.r_(theta,data_test)+1e-6))
        
        return score
    
    def r_(self,theta,data):
        ######## Likelihood ratio estimation  
        #### Input
        # theta: estimated parameter
        # data: datapoints to evaluate in the likelihood ratios
        #### Output
        # ratio: likelihood-ratio  evaluated at the points n data
        
        phi=self.kernel.k_V(transform_data(data))
        ratio=phi.dot(theta)
 
        return ratio





class RULSIF():
########## Class implementing RULSIF
# Based on the paper "Relative density-ratio estimation for robust distribution comparison."  Yamada et al., 2011

    def __init__(self,data_ref,data_test,alpha=0.1,verbose=False):     
    
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution p_v'(.) 
    # alpha: regularization parameter associated with the upperbound of the likelihood ratio
    # verbose: whether or not print intermediate results  
        
        self.data_ref=transform_data(data_ref)
        self.data_test=transform_data(data_test)
        self.alpha=alpha
        kernel_1=self.initializalize_kernel() 
        self.kernel,self.gamma=self.model_selection(kernel_1,verbose)

     
    def initializalize_kernel(self):

    ## Output
    # kernel_1: A initialized kernel with a given dictionary 

     
        if self.data_test.shape[0]>100:
            index=np.random.choice(len(self.data_test),replace=False,size=100)
            index.sort()
            sigma=get_sigma(self.data_test[index])
            dictionary=1*self.data_test[index]
            self.index=index
        else:
            sigma=get_sigma(self.data_test)
            dictionary=1*self.data_test
            self.index=np.arange(len(self.data_test))
            
        if sigma==0:
            sigma=1e-6
                
        kernel_1=Gauss_Kernel(dictionary,sigma=sigma)
                   
        return kernel_1
    
    def fit(self,data_ref=None,data_test=None,kernel=None,gamma=None): 
    #### Function estimating the theta parameter for a given set of observations comming from p(.) and q(.)  
    ## Input
    # data_ref: data points representing the distribution p_v(.)
    # data_test: data points representing the distribution p_v'(.) 
    # kernel: kernel to be used in the method
    # gamma: penalization constant related with the sparsness of the parameters   
    ## Output
    # theta: list with n_nodes elements, each element is the estimated parameter for the node v 
    
        if  data_ref is not None:
            data_ref=transform_data(data_ref)
        else:
            data_ref=self.data_ref
            
        if  data_test is not None:
            data_test=transform_data(data_test)
        else:
            data_test=self.data_test
    
        if kernel is None:
            kernel=self.kernel
        
        if gamma is None:
            gamma=self.gamma
    
        n_centers=kernel.n
        phi_test=kernel.k_V(data_test)
        phi_ref=kernel.k_V(data_ref)
   
        N_test=len(phi_test)
        N_ref=len(phi_ref)
        
        H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+self.gamma*np.eye(n_centers), h)

        return theta
    
    def model_selection(self,kernel,verbose=False):
        #### This function identifies the optimal hyperparameters related to the data     
        ###### Input
        # kernel: kernel to be used in the method
        # verbose: whether or not to print intermediate steps  
        ###### Output
        # kernel_: kernel initialized with the selected dictionary and the optimal value of alpha    
        
        sigma_list=np.array([0.6,0.8,1.0,1.2,1.4])*kernel.sigma
        max_ = -np.inf
        j_scores_ = {}
        best_params_={}
        N_test=self.data_test.shape[0]
        N_ref=self.data_ref.shape[0]
        N_min=np.min((N_ref,N_test))
        gamma_list = np.logspace(-6,1,6)*((1/np.min((N_ref,N_test)))**(3/4))
        
        if N_ref<N_test:
            index_data = np.random.choice(
                            N_test,
                            N_ref,
                            replace=False)
        elif N_test<N_ref:
            index_data = np.random.choice(
                            N_ref,
                            N_test,
                            replace=False)
            
        
        for s in range(len(sigma_list)):
            kernel_=Gauss_Kernel(dictionary=kernel.dictionary,sigma=sigma_list[s])

            
            if N_ref<N_test:
                phi_test = kernel_.k_V(self.data_test[index_data])
                phi_ref  = kernel_.k_V(self.data_ref)
                
            elif N_test<N_ref:
                phi_test = kernel_.k_V(self.data_test)
                phi_ref =  kernel_.k_V(self.data_ref[index_data])
            else:
                phi_test =  kernel_.k_V(self.data_test)
                phi_ref =   kernel_.k_V(self.data_ref)
            
        
            H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref 
         
            h = np.mean(phi_test, axis=0)
          
            h = h.reshape(-1, 1)
            for g in gamma_list:
                B = H + np.identity(kernel.n) * (g * (N_test - 1) / N_test)
                BinvX = np.linalg.solve(B, phi_test.T)
                XBinvX = phi_test.T * BinvX
                D0 = np.ones(N_min) * N_test- np.dot(np.ones(kernel.n), XBinvX)
                diag_D0 = np.diag((np.dot(h.T, BinvX) / D0).ravel())
                B0 = np.linalg.solve(B, h * np.ones(N_min)) + np.dot(BinvX, diag_D0)
                diag_D1 = np.diag(np.dot(np.ones(kernel.n), phi_ref.T * BinvX).ravel())
                B1 = np.linalg.solve(B,  phi_ref.T) + np.dot(BinvX, diag_D1)
                B2 = (N_test- 1) * (N_ref* B0 - B1) / (N_test* (N_ref - 1))
                B2[B2<0]=0
                r_s = (phi_ref.T * B2).sum(axis=0).T
                r_t= (phi_test.T * B2).sum(axis=0).T   
                score = ((1-self.alpha)*(np.dot(r_s.T, r_s).ravel() / 2. + self.alpha*np.dot(r_t.T, r_t).ravel() / 2.  - r_t.sum(axis=0)) /N_min).item()  # LOOCV
                aux_params={"sigma":sigma_list[s],"gamma":g}
                j_scores_[str(aux_params)]=-1*score
               
                if verbose:
                    print("Parameters %s -- J-score = %.3f"% (str(aux_params),score))
                if j_scores_[str(aux_params)] > max_:
                   best_params_ = aux_params
                   max_ = j_scores_[str(aux_params)]  
                   
        kernel_=Gauss_Kernel(dictionary=kernel.dictionary,sigma=best_params_["sigma"])
                
        return  kernel_,best_params_["gamma"]
        
           
    def PE_divergence(self,data_ref=None,data_test=None):
    ####### Function estimating the Pearson Divergence
    ## Input
    # data_ref: data points representing the distribution p_v'(.)
    # data_test: data points representing the distribution p_v(.)
    ## Output 
    # score: Pearson Divergence 
        phi_test=self.kernel.k_V(data_test)
        phi_ref=self.kernel.k_V(data_ref)
  
        N_test=len(phi_test)
        N_ref=len(phi_ref)
       
        H=self.alpha*np.einsum('ji,j...',  phi_test,phi_test) / N_test + (1-self.alpha)*np.einsum('ji,j...',  phi_ref,phi_ref)/N_ref
        h = np.mean(phi_test, axis=0)
        h = h.reshape(-1, 1)
        theta = np.linalg.solve(H+self.gamma*np.eye(self.kernel.n), h)

        score=(theta.transpose()).dot(H).dot(theta)
        score*=-0.5
        score+=theta.transpose().dot(h)
        score-=0.5 
        return score
    
    def r_(self,theta,data):
        ######## Likelihood ratio estimation  
        #### Input
        # theta: estimated parameter
        # data: datapoints to evaluate in the likelihood ratios
        #### Output
        # ratio: likelihood-ratio  evaluated at the points n data
        
        phi=self.kernel.k_V(transform_data(data))
        ratio=phi.dot(theta)
 
        return ratio
    
  
def klieps(data_ref,data_test):    
    ######### This function estimates the KLIEP likelihood-ratio estimator as the number of observations increases
    ### Input
    ## data_ref: data from the distribution x~P
    ## data_test: data from the distribution x~Q

    ### Output 
    ## list_dictionaries:list of the used dictionaries at every time t
    ## list_thetas:list of parameters estimated at everytime time t
    ## list_sigmas:list of identified optima sigma parameters  
    
    list_thetas=[]
    list_dictionaries=[]
    list_sigmas=[]
  
    for t in range(0,len(data_ref)-100,100):
        kliep_= KLIEP(data_ref[:t+100],data_test[:t+100])
        theta=kliep_.fit(data_ref=data_ref[:t+100],data_test=data_test[:t+100])
        list_thetas.append(1.*theta)     
        list_dictionaries.append(1.*kliep_.kernel.dictionary)
        list_sigmas.append(copy.deepcopy(kliep_.kernel.sigma))

    return list_dictionaries,list_thetas,list_sigmas


def rulsifs(data_ref,data_test,alpha=0.1):    
    ######### This function estimates the RULSIF likelihood-ratio estimator as the number of observations increases 
    
    ### Input 
    ## data_ref: data from the distribution x~P
    ## data_test: data from the distribution x~Q

    ### Output 
    ## list_dictionaries:list of the used dictionaries at every time t
    ## list_thetas:list of parameters estimated at everytime time t
    ## list_sigmas:list of identified optima sigma parameters   
    
    
    list_thetas=[]
    list_dictionaries=[]
    list_sigmas=[]
    list_lamb=[]
    
    for t in range(0,len(data_ref)-100,100):
        rulsif_= RULSIF(data_ref[:t+100],data_test=data_test[:t+100],alpha=alpha)
        theta=rulsif_.fit(data_ref=data_ref[:t+100],data_test=data_test[:t+100])
        list_thetas.append(1.*theta)     
        list_dictionaries.append(1.*rulsif_.kernel.dictionary)
        list_sigmas.append(copy.deepcopy(rulsif_.kernel.sigma))
        list_lamb.append(copy.deepcopy(rulsif_.gamma))
               
    return  list_dictionaries,list_thetas,list_sigmas,list_lamb


    
    
    
    
    
    
    