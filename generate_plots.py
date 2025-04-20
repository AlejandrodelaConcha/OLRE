# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 22:48:45 2022

@author: 33768
"""


import matplotlib.pyplot as plt
import pickle 
import argparse
import numpy as np
from mpl_toolkits import mplot3d


def main(results_directory,experiment):

    ########################### KL 
  
    T=10000
    alpha=0.1
    file_name=results_directory+"/"+f"Experiment_{experiment}_errors_offline_kliep"
    file_name=file_name.replace(".","") 

    with open(file_name+".pickle", 'rb') as f:
        results_KLIEP=  pickle.load(f)
        
    alpha=0.1
    file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_errors_offline_rulsif"
    file_name=file_name.replace(".","") 

    with open(file_name+".pickle", 'rb') as f:
        results_RULSIF=  pickle.load(f)
        
    alpha=0.5
    file_name=results_directory+"/"+f"Experiment_{experiment}_alpha{alpha}_errors_offline_rulsif"
    file_name=file_name.replace(".","") 

    with open(file_name+".pickle", 'rb') as f:
        results_RULSIF_05=  pickle.load(f)
        
        
    alpha=0.1
    smoothness=1.0
    
    file_name=results_directory+"/"+f"Experiment_{experiment}_errors_alpha{alpha}_smoothness{smoothness}_olre"
    file_name=file_name.replace(".","")
    
    with open(file_name+".pickle", 'rb') as f:
        results_dynamic_r1_alpha_01=  pickle.load(f)
        
    
    alpha=0.1
    smoothness=0.5
    
    file_name=results_directory+"/"+f"Experiment_{experiment}_errors_alpha{alpha}_smoothness{smoothness}_olre"
    file_name=file_name.replace(".","")
    
    with open(file_name+".pickle", 'rb') as f:
        results_dynamic_r05_alpha_01=  pickle.load(f)
        

    ##########################################################
        
     
    alpha=0.5
    smoothness=1.0
     
    file_name=results_directory+"/"+f"Experiment_{experiment}_errors_alpha{alpha}_smoothness{smoothness}_olre"
    file_name=file_name.replace(".","")
     
    with open(file_name+".pickle", 'rb') as f:
         results_dynamic_r1_alpha_05=  pickle.load(f)
           
    alpha=0.5
    smoothness=0.5
     
    file_name=results_directory+"/"+f"Experiment_{experiment}_errors_alpha{alpha}_smoothness{smoothness}_olre"
    file_name=file_name.replace(".","")
     
    with open(file_name+".pickle", 'rb') as f:
         results_dynamic_r05_alpha_05=  pickle.load(f)
         
        
    list_L2_distance_kliep=np.vstack(results_KLIEP["L2"])
    list_L2_distance_rulsif=np.vstack(results_RULSIF["L2"])
    list_L2_distance_rulsif_05=np.vstack(results_RULSIF_05["L2"])
    
    list_L2_dynamic_r1_alpha_01=np.vstack(results_dynamic_r1_alpha_01["L2"])
    list_L2_dynamic_r05_alpha_01=np.vstack(results_dynamic_r05_alpha_01["L2"])
    
    list_L2_dynamic_r1_alpha_05=np.vstack(results_dynamic_r1_alpha_05["L2"])
    list_L2_dynamic_r05_alpha_05=np.vstack(results_dynamic_r05_alpha_05["L2"])
   
    mean_log_L2_distance_kliep=np.mean(np.log(list_L2_distance_kliep),0)
    std_log_L2_distance_kliep=np.std(np.log(list_L2_distance_kliep),0)

    mean_log_L2_distance_rulsif=np.mean(np.log(list_L2_distance_rulsif),0)
    std_log_L2_distance_rulsif=np.std(np.log(list_L2_distance_rulsif),0)
    
    mean_log_L2_distance_rulsif_05=np.mean(np.log(list_L2_distance_rulsif_05),0)
    std_log_L2_distance_rulsif_05=np.std(np.log(list_L2_distance_rulsif_05),0)
        
    mean_log_L2_distance_dynamic_r1_alpha_01=np.mean(np.log(list_L2_dynamic_r1_alpha_01),0)
    std_log_L2_distance_dynamic_r1_alpha_01=np.std(np.log(list_L2_dynamic_r1_alpha_01),0)
 
    mean_log_L2_distance_dynamic_r05_alpha_01=np.mean(np.log(list_L2_dynamic_r05_alpha_01),0)
    std_log_L2_distance_dynamic_r05_alpha_01=np.std(np.log(list_L2_dynamic_r05_alpha_01),0)
    
    mean_log_L2_distance_dynamic_r05_alpha_05=np.mean(np.log(list_L2_dynamic_r05_alpha_05),0)
    std_log_L2_distance_dynamic_r05_alpha_05=np.std(np.log(list_L2_dynamic_r05_alpha_05),0)
 
    mean_log_L2_distance_dynamic_r1_alpha_05=np.mean(np.log(list_L2_dynamic_r1_alpha_05),0)
    std_log_L2_distance_dynamic_r1_alpha_05=np.std(np.log(list_L2_dynamic_r1_alpha_05),0)

    fig,ax=plt.subplots(1,1,figsize=(20,15))
    
    ax.plot(np.arange(0,10000,100),mean_log_L2_distance_kliep,color="m",
      label=r'$KLIEP \ \ (\alpha=0.0)$',linestyle=(0,(3,1,1,1)),linewidth = '10')
    ax.fill_between(np.arange(0,10000,100),mean_log_L2_distance_kliep-std_log_L2_distance_kliep,
            mean_log_L2_distance_kliep+std_log_L2_distance_kliep, 
                color="m", alpha=0.1)
    
    ax.plot(np.arange(0,10000,100),mean_log_L2_distance_rulsif,color="lime",
       label=r'$RULSIF \  \alpha=0.1$',linestyle="dotted",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100),mean_log_L2_distance_rulsif-std_log_L2_distance_rulsif,
               mean_log_L2_distance_rulsif+std_log_L2_distance_rulsif, 
                color='lime', alpha=0.1)
    
    ax.plot(np.arange(0,10000,100),mean_log_L2_distance_rulsif_05,color="darkorange",
       label=r'$RULSIF \  \alpha=0.5$',linestyle="dotted",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100),mean_log_L2_distance_rulsif_05-std_log_L2_distance_rulsif_05,
               mean_log_L2_distance_rulsif_05+std_log_L2_distance_rulsif_05, 
                color='darkorange', alpha=0.1)
     
    ax.plot(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r1_alpha_01,color="green",
       label=r'OLRE $\quad \! \alpha=0.1,r=1.0$',linestyle="dashed",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r1_alpha_01-std_log_L2_distance_dynamic_r1_alpha_01,
              mean_log_L2_distance_dynamic_r1_alpha_01+std_log_L2_distance_dynamic_r1_alpha_01, 
                color='green', alpha=0.1)
    
    ax.plot(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r05_alpha_01,color="olive",
       label=r'$\quad \qquad \quad \! \alpha=0.1,r=0.5$',linestyle="solid",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r05_alpha_01-std_log_L2_distance_dynamic_r05_alpha_01,
              mean_log_L2_distance_dynamic_r05_alpha_01+std_log_L2_distance_dynamic_r05_alpha_01, 
                color='royalblue', alpha=0.1)
    
    ax.plot(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r1_alpha_05,color="red",
       label=r'OLRE $\quad \! \alpha=0.5,r=1.0$',linestyle="dashed",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r1_alpha_05-std_log_L2_distance_dynamic_r1_alpha_05,
              mean_log_L2_distance_dynamic_r1_alpha_05+std_log_L2_distance_dynamic_r1_alpha_05, 
                color='red', alpha=0.1)
    
    ax.plot(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r05_alpha_05,color="maroon",
       label=r'$\quad \qquad \quad \! \alpha=0.5,r=0.5$',linestyle="solid",linewidth = '10')
    ax.fill_between(np.arange(0,10000,100), mean_log_L2_distance_dynamic_r05_alpha_05-std_log_L2_distance_dynamic_r05_alpha_05,
              mean_log_L2_distance_dynamic_r05_alpha_05+std_log_L2_distance_dynamic_r05_alpha_05, 
                color='maroon', alpha=0.1)
       
    ax.set_ylabel(r'$\log(\mathbb{E}_{p^{\alpha}(x)}[(f_t-r^{\alpha})^2(x)])$',fontsize=50)
    ax.set_xlabel(r'$\#$ of training samples processed',fontsize=50)
    
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    ax.legend(loc='upper right',ncol=2,fontsize=35)

    file_name=results_directory+"/"+f"Experiment_{experiment}_results"
    file_name=file_name.replace(".","")
    plt.savefig(file_name+".pdf")




if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--results_directory") #### Dictionary where the results will be saved
    parser.add_argument("--experiment",type=int) 
    args=parser.parse_args()
    results_directory=str(args.results_directory)
    results_directory= results_directory+'/'
    experiment=args.experiment
    main(results_directory,experiment)






results_directory= "C:/Users/alexd/Documents/OLRE/Results"
experiment=4









