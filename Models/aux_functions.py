# -----------------------------------------------------------------------------------------------------------------
# Title:  aux_functions
# Author(s): Alejandro de la Concha
# Initial version:  2020-05-17
# Last modified:    2024-02-28              
# This version:     2024-05-28
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to define the functions associated with the Gaussian Kernel
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy,numba 
# -----------------------------------------------------------------------------------------------------------------
# Key words: Gaussian Kernel, coherence, Kernel Methods 
# ---------------------------------------------------------------------------------------------------------

import numpy as np
from numba import njit,jit

@jit(nopython=True)
def calc_dist(A,B,sqrt=False):
##### This function is a fast version of the eclidiant distance between elements of two matrices 
## Input: A,B: matrices of size M,n and K,n
## sqrt: Whether to obtain the square root of the distance
## Output: matrix of size MxK with the distance between the rows of A and B.
         
  dist=np.dot(A,B.T)

  TMP_A=np.empty(A.shape[0],dtype=A.dtype)
  for i in range(A.shape[0]):
    sum=0.
    for j in range(A.shape[1]):
      sum+=A[i,j]**2
    TMP_A[i]=sum

  TMP_B=np.empty(B.shape[0],dtype=A.dtype)
  for i in range(B.shape[0]):
    sum=0.
    for j in range(B.shape[1]):
      sum+=B[i,j]**2
    TMP_B[i]=sum

  if sqrt==True:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=np.sqrt(-2.*dist[i,j]+TMP_A[i]+TMP_B[j])
  else:
    for i in range(A.shape[0]):
      for j in range(B.shape[0]):
        dist[i,j]=-2.*dist[i,j]+TMP_A[i]+TMP_B[j]
  return dist

@jit(nopython=True)
def calc_dist_L1(A,B,sqrt=False):
    ######### This function is a fast version of the distance between elements of two matrices
    ## Input: A,B: matrices of size M,n and K,n
    ## sqrt: Whether to obtain the square root of the distance
    ## Output: matrix of size MxK with the distance between the rows of A and B.
         
    N=A.shape[0]
    M=B.shape[0]
    dist=np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            dist[i,j]=np.sum(np.abs(A[i]-B[j]))  
    return dist
    
    

class Gauss_Kernel(object):
#### Class implementing the Gaussian kernel
    def __init__(self,dictionary,sigma):
    ## Input
    # dictionary: An initial dictionary
    # sigma: Width parameter 
        self.dictionary=dictionary
        self.sigma=sigma
        self.n=self.dictionary.shape[0]

    def k(self,x):
    ### Function estimating the feature transformation of point x with respect to a given dictionary
    ## Input
    # x: point to evaluate 
    ## Output 
    # feature map related with the kernel 
        if self.n==1:
            distances=np.linalg.norm(x-self.dictionary)**2
            distances*=-1/(2.*self.sigma**2)
            distances=np.atleast_2d(np.array(np.exp(distances),dtype=np.float64))
            return distances
        else:       
          #  distances=distance.cdist(np.atleast_2d(x),self.dictionary,'sqeuclidean')
            distances=calc_dist(np.atleast_2d(x),self.dictionary)
            distances*=-1/(2.*self.sigma**2)
            return np.exp(distances)


    def k_V(self,x):
    ### Function estimating the feature transformation of a set of points with respect to a given dictionary
    ## Input
    # x: points to be valuated
    ## Ouput
    # A matrix whose rows are the feature maps of the given point.
        x=transform_data(x)        
        if len(x)==1:
            self.k(x)           
        else:         
            distances= calc_dist(x,self.dictionary)
            distances*=-1/(2.*self.sigma**2)
            return np.exp(distances)
  
    def coherence(self,x): 
    ### Function implementing the coherence of a point with respect to a given dictionary 
    ## Input
    # x: point to evaluate 
    ## Output
    # k_: feature map related with the point x
    # mu_0: coherence with respect to the availabledictionary
    
        k_=self.k(x)       
        mu_0=np.max(np.abs(k_))       
        return k_,mu_0
 
    def add_dictionary(self,x):   
    ##### Function to add a points to the dictionary
    ## Input
    # x: point to add 
        self.dictionary=np.vstack((self.dictionary,x))
        self.n=len( self.dictionary)
     
    def delete_from_dictionary(self,index):  
    ##### Function to eliminate a given index from the dictionary 
    ## Input
    # index: position of the point to eliminate from the dictionary 
        
        if index==0:
            self.dictionary=1.*self.dictionary[1:]
        elif index>=self.n-1:
            self.dictionary=1.*self.dictionary[:self.n-1]
        else:
            self.dictionary=np.vstack((1.*self.dictionary[:index],1.*self.dictionary[index+1:]))     
        self.n=len(self.dictionary)  
        
    def new_phi(self,data,new_value):
    ##### This function k(x,new_value) where x is a set of observations
    ### Input
    ## data: the data to be evaluated against new value
    ## new value: the point used as reference.
    ### Output: 
    ## the matrix k(data,new_value)
        data=transform_data(data)
        new_value=np.atleast_2d(new_value)
        distances= calc_dist(new_value,data)
        distances*=-1/(2.*self.sigma**2)
        return np.exp(distances)
    
    def get_internal_coherences(self):
    ### This function computes the coherence of all elements of the dictionary with the respect to the
    ## the dictionary without that element. 
    
    ## Output: 
    ## coherences: a vector where each entry in the coherence of that specific element in the dictionary 
        coherences=[]
        aux_coherence=self.k(self.dictionary[0])[0]
        coherences.append(np.max(np.abs(aux_coherence[1:])))
        for i in range(len(self.dictionary)-1):
            aux_coherence=self.k(self.dictionary[i])[0]
            coherences.append(np.max(np.abs(np.hstack((aux_coherence[:i],aux_coherence[i+1:])))))
        i=len(self.dictionary)-1
        aux_coherence=self.k(self.dictionary[i])[0]
        coherences.append(np.max(np.abs(np.array((aux_coherence[:i])))))
        coherences=np.array(coherences)
        return coherences
   
      
def get_sigma(data):   
    #### Fuction implementing the median heuristic
    ## Input
    # data: data points to use in the estimation of sigma 
    ## Output
    # sigma: media of the distance function. 
    data=transform_data(data)  
    n=data.shape[0]      
    distances=calc_dist(data,data,sqrt=True)
    distances[np.isnan(distances)]=0.0
    sigma=np.sqrt((np.median(distances[np.triu_indices(n,1)])**2)/2)  
    return sigma

def transform_data(x):
    ####### Function taking care of addresing possible problems with the numpy array 
    if len(x.shape)==1:
        return np.atleast_2d(x.astype(np.float64)).T
    else:
        return np.atleast_2d(x.astype(np.float64))


















   