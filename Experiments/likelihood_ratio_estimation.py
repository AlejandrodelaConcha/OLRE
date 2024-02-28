# -----------------------------------------------------------------------------------------------------------------
# Title:  Likelihood-ratio-estimation 
# Author(s):  
# Initial version:  2022-02-12
# Last modified:    2022-02-12             
# This version:     2022-02-12
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This function generates the real likelihood-value estimates for the evaluation of the experiments
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: sklearn,pandas,Models.aux_functions
# -----------------------------------------------------------------------------------------------------------------
# Key words:  likelihood-ratio estimation, Pearson-Divergence estimate
# ------------------

from scipy import stats
import numpy as np 
from scipy.special import logsumexp

def r_bivariate_normal(x,alpha):
    #### This function compute the real-likelihood ratio between two bivariate Gaussian distributions
    ### Input:
    # x: point to evaluate 
    # alpha: regularization parameter

    ### Output:
    #  real-likelihood ratio evaluated at the point x
    
    log_r_alpha=(1-alpha)*stats.multivariate_normal.pdf(x,np.zeros(2))+alpha*stats.multivariate_normal.pdf(x,np.zeros(2),np.array([[1,4/5],[4/5,1]]))
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=stats.multivariate_normal.logpdf(x,np.zeros(2),np.array([[1,4/5],[4/5,1]]))
    return np.exp(log_r_alpha)

def r_uniform_laplace(x,alpha):
    ### This function compares the likelihood-ratio between a laplace distribution and a uniform distribution 
    # x: point to evaluate 
    # alpha: regularization parameter 
    
    ### Output:
    #  real-likelihood ratio evaluated at the point x
    
    log_r_alpha=alpha*stats.laplace.pdf(x,loc=0,scale=np.sqrt(0.5))+(1-alpha)*stats.uniform.pdf(x,loc=-np.sqrt(3),scale=2*np.sqrt(3))
    log_r_alpha[log_r_alpha<=0]=1e-16
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=stats.laplace.logpdf(x,loc=0,scale=np.sqrt(0.5))
    return np.exp(log_r_alpha)

def r_normal_mixture(x,alpha):
    ### This function compares the likelihood-ratio between a mixture of gaussian distribution and a bivariate gaussian distribution  
    # x: point to evaluate 
    # alpha: regularization parameter 
    
    ### Output:
    #  real-likelihood ratio evaluated at the point x
    
    means=[np.array([0,0]),np.array([0,5]),np.array([0,-5]),np.array([5,0]),np.array([-5,0])] 
    log_r_alpha=[stats.multivariate_normal.logpdf(x,mean=m) for m in means]
    log_r_alpha=np.vstack(log_r_alpha)
    log_r_alpha=logsumexp(log_r_alpha,axis=0)
    aux_log_r_alpha=(1-alpha)*stats.multivariate_normal.pdf(x,mean=np.array([0,0]),cov=10*np.eye(2))
    aux_log_r_alpha+=alpha*np.exp(log_r_alpha)*(1/5)
    aux_log_r_alpha=np.log(aux_log_r_alpha)
    log_r_alpha=log_r_alpha-aux_log_r_alpha
    r_alpha=np.exp(log_r_alpha)*(1/5)
    return r_alpha

def r_mul_normal(x,alpha,dim=5):
    ### This function compares the likelihood-ratio between two multivarige gaussian distributions of dimension dim with differen covariance matrix
    # x: point to evaluate 
    # alpha: regularization parameter 
    # dim: dimension of the vectores   
    ### Output:
    #  real-likelihood ratio evaluated at the point x
    matrix_ref=np.zeros((dim,dim))
    matrix_test=4/5*np.ones((dim,dim))
    mean=np.zeros(dim)
    np.fill_diagonal(matrix_ref,1)
    np.fill_diagonal(matrix_test,1)
    log_r_alpha=(1-alpha)*stats.multivariate_normal.pdf(x,mean,matrix_ref)+alpha*stats.multivariate_normal.pdf(x,mean,matrix_test)
    log_r_alpha=-1*np.log(log_r_alpha)
    log_r_alpha+=stats.multivariate_normal.logpdf(x,mean,matrix_test)
    return np.exp(log_r_alpha)

