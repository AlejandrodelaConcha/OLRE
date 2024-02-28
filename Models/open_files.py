# -----------------------------------------------------------------------------------------------------------------
# Title:  open_files
# Author(s):  
# Initial version:  2020-07-01
# Last modified:    2020-07-01             
# This version:     2020-07-01
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this code is to clean and access the DataSets 
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies: numpy,numba 
# -----------------------------------------------------------------------------------------------------------------
# Key words: OCKGD. f-divergence, Likelihood-ratio estimation 
# ---------

import os
import pandas as pd
import numpy as np
import ruptures as rpt
from itertools import product
import soundfile as sf
import copy
import librosa
import matplotlib.pyplot as plt
import pickle
from Models.CP_LRE import *
from Models.online_methods import *

def list_subdirectories(folder_path):
    ############ This code list all the subfolders of a given directory. 
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # subdirectories= list of subdirectories 
    
    subdirectories = []
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            subdirectories.append(os.path.join(root, directory))
    return subdirectories


def get_csv_files(folder_path):
    ########## This code list all the csv files from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # csv_files = list of names of the csv files in a directory  
    
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(file)
    return csv_files

def get_label_files(folder_path):
    ########## This code list all the csv files from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # label_files = list of names of files with the labels of the activities.  
    
    label_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.label'):
                label_files.append(file)
    return label_files

def get_sound_files(folder_path):
    ########## This code list all the sound files in raw format from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # sound_files = list of names of files with the sounds for the speech data  in raw.  
    
    sound_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.raw'):
                sound_files.append(file)
    return sound_files


def get_label_files_speech(folder_path):
    ########## This code list all segmentation files from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # label_files = list of names of files with the labels of the activities.  

    label_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.seg'):
                label_files.append(file)
    return label_files



def get_time_series(folder_path):
    ########## This code list all the csv files from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # csv_files = list of names of files with the cleaned time series .  
    
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if 'time_series' in file:
                csv_files.append(file)
    return csv_files


def get_change_points(folder_path):
    ########## This code list all the csv files from a given directory
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # csv_files = list of names of files with the cleaned change_ponts series .  
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if 'change_points' in file:
                csv_files.append(file)
    return csv_files

############################################# Human activity 

def create_time_series_change_points_HASC2011_corpus(folder_path,output_directory): 
    ########## This function cleans the original time-series and produces the files with time series and change-points
    ## Input: 
    # folder_path= path of interest 
    ## Output: 
    # csv_files = list of names of files with the cleaned change_ponts series .

    subdirectories = list_subdirectories(folder_path)

    for sub in subdirectories:
        print(sub)
        csv_file=get_csv_files(sub)
        for f in csv_file:
            file_name=sub+'\\'+f
            df = pd.read_csv(file_name,header=None)
            df.columns=["t","x","y","z"]
            df["signal"]=np.sqrt(df["x"]**2+df["y"]**2+df["y"]**2+df["z"]**2)
            file_to_save=output_directory+'\\time_series'+'\\'+f
            df[["t","signal"]].to_csv(file_to_save)
        
 ######################################################################

    for sub in subdirectories:
        #  print(sub)
        label_file=get_label_files(sub)
        for f in label_file:
            file_name=sub+'\\'+f
            df = pd.read_csv(file_name,skiprows=1,header=None)
            df.columns=["begining","end","activity"]
            f=f.replace('.','_')
            file_to_save=output_directory+'\\change_points'+'\\'+f+".csv"
            df.to_csv(file_to_save)
            
            
def segment_HASC2011_corpus(time_series_directory,change_point_directory,cleaned_data_directory):
    time_series_files_list=get_csv_files(time_series_directory)
    change_points_files_list=get_csv_files(change_point_directory)
    index_activities=dict()  

    for k in range(len(time_series_files_list)):
        print(time_series_files_list[k])
        time_series = pd.read_csv(time_series_directory+"\\"+time_series_files_list[k])
        change_points=pd.read_csv(change_point_directory+"\\"+change_points_files_list[k])
        for i in range(len(change_points["begining"])-1):
            begining=change_points["begining"][i]
            end=change_points["end"][i+1]
            #cp=change_points["end"][i]
            cp=change_points["begining"][i+1]
            pair_activities=change_points["activity"][i]
            sub_time_series=time_series[(time_series["t"]<=end)*(time_series["t"]>=begining)]
            cp=[0,np.abs(cp-sub_time_series["t"]).argmin(),len(sub_time_series)]
            if cp[1]>1000:
                sub_time_series= sub_time_series["signal"]
                cp=pd.DataFrame(cp)
      
                if pair_activities in index_activities.keys():
                    index_activities[pair_activities]+=1
                else: 
                    index_activities[pair_activities]=1
                    
                output_directory=cleaned_data_directory+"\\"+pair_activities
                sub_time_series.to_csv(cleaned_data_directory+"\\"+pair_activities+"\\"+pair_activities+"_"+"time_series"+"_"+str(index_activities[pair_activities])+".csv")
                cp.to_csv(cleaned_data_directory+"\\"+pair_activities+"\\"+pair_activities+"_"+"change_points"+"_"+str(index_activities[pair_activities])+".csv")
    print( index_activities)
  
def create_time_series_change_points_CENSREC_1_C(folder_path,output_directory): 
    ########## This function cleans the original time-series and produces the files with time series and change-points
    ## Input: 
    # folder_path= path of interest 
    # outout_directory= directory to save data points 

    list_subdirectories=["\\close"+"\\RESTAURANT_SNR_HIGH","\\close"+"\\RESTAURANT_SNR_LOW","\\close"+"\\STREET_SNR_HIGH","\\close"+"\\STREET_SNR_LOW",
                         "\\remote"+"\\RESTAURANT_SNR_HIGH","\\remote"+"\\RESTAURANT_SNR_LOW","\\remote"+"\\STREET_SNR_HIGH","\\remote"+"\\STREET_SNR_LOW"]

 #   subdirectories = list_subdirectories(folder_path)
    
    sample_rate = 48000 
    
    for sub in list_subdirectories:
        label_files=get_label_files_speech(folder_path+sub)
        for f in  label_files:
       #     print(file_name)
            file_name=folder_path+sub+'\\'+f
            df = pd.read_csv(file_name,header=None,sep=' ')
            df.columns=["begining","end"]
            f=f.replace(".seg",".csv")
            sub_aux=sub.replace("\\","_")
            file_to_save=output_directory+'\\change_points'+'\\'+sub_aux+"_changepoint"+"_"+f
            df.to_csv(file_to_save)
               
        
    for sub in list_subdirectories:
        raw_files=get_sound_files(folder_path+sub)
        for f in  raw_files:
          #   print(file_name)
            file_name=folder_path+sub+'\\'+f
            data, _ = sf.read(file_name, channels=1, samplerate=sample_rate, format='RAW', subtype='PCM_16')
            data=data.T
            frame_duration = 0.0010  # Frame duration in seconds
            frame_length = int(sample_rate * frame_duration)
            hop_length = int(sample_rate * 0.0005) 
             # Applying a Hamming window
            # Step 6: Extract MFCC features
            n_mfcc = 13 # Number of MFCC coefficients to extract
            mfcc = librosa.feature.mfcc(y=np.squeeze(data), sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
            mfcc=mfcc.transpose()
            
            df=pd.DataFrame(audio_data)
            df.columns=["signal"]
            f=f.replace(".raw",".csv")
            sub_aux=sub.replace("\\","_")
            file_to_save=output_directory+'\\time_series'+'\\'+sub_aux+"_"+f
            df.to_csv(file_to_save)
            
            
##################### New


def create_time_series_change_points_CENSREC_1_C(folder_path,output_directory): 
    ########## This function cleans the original time-series and produces the files with time series and change-points
    ## Input: 
    # folder_path= path of interest 
    # outout_directory= directory to save data points 

    list_subdirectories=["\\close"+"\\RESTAURANT_SNR_HIGH","\\close"+"\\RESTAURANT_SNR_LOW","\\close"+"\\STREET_SNR_HIGH","\\close"+"\\STREET_SNR_LOW",
                         "\\remote"+"\\RESTAURANT_SNR_HIGH","\\remote"+"\\RESTAURANT_SNR_LOW","\\remote"+"\\STREET_SNR_HIGH","\\remote"+"\\STREET_SNR_LOW"]

 #   subdirectories = list_subdirectories(folder_path)
    
    sample_rate = 48000 
    
    for sub in list_subdirectories:
        label_files=get_label_files_speech(folder_path+sub)
        for f in  label_files:
       #    Change_points
            print(f)
            file_name=folder_path+sub+'\\'+f
            df = pd.read_csv(file_name,header=None,sep=' ')
            df.columns=["begining","end"]
            df=df/(sample_rate * 0.0005)
            df=df.round(0)
            
            ########## Time series 
            
            file_name=folder_path+sub+'\\'+f.replace("seg",'raw')
            data, _ = sf.read(file_name, channels=1, samplerate=sample_rate, format='RAW', subtype='PCM_16')
            data=data.T
            frame_duration = 0.0010  # Frame duration in seconds
            frame_length = int(sample_rate * frame_duration)
            hop_length = int(sample_rate * 0.0005) 
            n_mfcc = 13 # Number of MFCC coefficients to extract
            mfcc = librosa.feature.mfcc(y=np.squeeze(data), sr=sample_rate, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
            mfcc=mfcc.transpose()
            
            mean_mfcc=np.mean(mfcc[:int(df['begining'][0])],axis=0)
            std_mfcc=np.std(mfcc[:int(df['begining'][0])],axis=0)
            standardized=(mfcc-mean_mfcc)/std_mfcc
            standardized=pd.DataFrame(standardized)
            
            f=f.replace(".seg",".csv")
            sub_aux=sub.replace("\\","_")
            file_to_save=output_directory+'\\change_points'+'\\'+sub_aux+"_changepoint"+"_"+f
            df.to_csv(file_to_save)
            
            sub_aux=sub.replace("\\","_")
            file_to_save=output_directory+'\\time_series'+'\\'+sub_aux+"_"+f
            standardized.to_csv(file_to_save)            
            
  

def segment_SPEECH_corpus(time_series_directory,change_point_directory,cleaned_data_directory):
    
    ##### This functions generates the segmented and clean datasets related with the noise and SPEECh 
    ## Input 
    # 
    
    time_series_files_list=get_csv_files(time_series_directory)
    change_points_files_list=get_csv_files(change_point_directory)

    names=["RESTAURANT_SNR_HIGH","RESTAURANT_SNR_LOW","STREET_SNR_HIGH","STREET_SNR_LOW"]

    aux_change_points_files=np.array([cp.replace("_changepoint","") for cp in  change_points_files_list])
    index_activities=dict()  
    for k in range(len(time_series_files_list)):
 
        for n in names:
            if n in time_series_files_list[k]:
                if  n in index_activities.keys():
                    index_activities[n]+=1
                else:
                    index_activities[n]=1
                scenario=copy.deepcopy(n) 
                print(scenario)      
        
        index_cp=np.where(aux_change_points_files==time_series_files_list[k])[0][0]
        time_series = pd.read_csv(time_series_directory+"\\"+time_series_files_list[k],index_col=0)
        change_points=pd.read_csv(change_point_directory+"\\"+change_points_files_list[index_cp])
    
    ############ Firt time series
        begining=0
        cp=int(change_points["begining"][0])
        end=int(change_points["end"][0])
        sub_time_series=time_series[begining:end]
        cp=[0,int(np.abs(cp-sub_time_series.index).argmin()),len(sub_time_series)]
        if cp[1]>500:
     #   rpt.display(sub_time_series, cp,cp)
            cp=pd.DataFrame(cp)
            sub_time_series.to_csv(cleaned_data_directory+"\\"+scenario+"\\"+"time_series"+"_"+str(index_activities[scenario])+".csv")
            cp.to_csv(cleaned_data_directory+"\\"+scenario+"\\"+"change_points"+"_"+str(index_activities[scenario])+".csv")

        for i in range(1,len(change_points["begining"])):
            index_activities[ scenario]+=1
            begining=int(change_points["end"][i-1])
            cp=int(change_points["begining"][i])
            end=int(change_points["end"][i])
            sub_time_series=time_series[begining:end]
            cp=[0,int(np.abs(cp-sub_time_series.index).argmin()),len(sub_time_series)]
            if cp[1]>500:
           # rpt.display(sub_time_series, cp,cp)
                cp=pd.DataFrame(cp)
                sub_time_series.to_csv(cleaned_data_directory+"\\"+scenario+"\\"+"time_series"+"_"+str(index_activities[scenario])+".csv")
                cp.to_csv(cleaned_data_directory+"\\"+scenario+"\\"+"change_points"+"_"+str(index_activities[scenario])+".csv")
                      

#########################

folder_path = "C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\0_sequence"
output_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data"
create_time_series_change_points_HASC2011_corpus(folder_path,output_directory)

activity=["walk","jog","stay","stDown","stUP","skip"]    

for d in activity: 
    os.makedirs(output_directory+"\\"+d)   
    
time_series_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data\\time_series"
change_point_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data\\change_points"
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data"

segment_HASC2011_corpus(time_series_directory,change_point_directory,cleaned_data_directory)

pair_activity='jog'
directory_activities=cleaned_data_directory+"\\"+pair_activity
time_series_files_list=get_time_series(directory_activities)
change_points_files_list=get_change_points(directory_activities)

sub_time_series = pd.read_csv(directory_activities+"\\"+time_series_files_list[5])["signal"]
change_points= pd.read_csv(directory_activities+"\\"+change_points_files_list[5])

data_ref=sub_time_series[:300]
time_series=np.array(sub_time_series[300:])
plt.plot(time_series)

scores_kcusum=KCUSUM(data_ref,time_series,delta=0.5)        
plt.plot(np.vstack(scores_kcusum))
        

B_0=50
N=30
scores_scan_B=scan_B_statistic(data_ref,time_series,B_0,N)
        
plt.plot(time_series[50:])
plt.plot(scores_scan_B)   


##################

n=100
alpha=0.1
scores_rulsif=rulsif(data_ref,time_series,n,alpha)
plt.plot(scores_rulsif)

n=50
threshold_coherence=0.5
scores_NOUGAT=NOUGAT(data_ref,time_series,n,threshold_coherence)

plt.plot(scores_NOUGAT)

n=50
eta=0.8
gamma=0.8
scores_kliep=kliep(data_ref,time_series,n,eta,gamma)

plt.plot(scores_kliep)


alpha=0.1
aux_learning_rate=0.01
t_0=200
w=50
learning_rate=lambda t: 4.0/((t+t_0)**0.5)
regularization=lambda t: 1.0/(4*(t+t_0)**0.5)

scores_PEARSON=pearsuit_cp(data_ref,time_series,t_0,w,learning_rate,regularization,alpha)


plt.axvline(x=change_points["0"][1]-(t_0+w+300),color="red") 
plt.plot(time_series[t_0+w:])   
plt.plot(scores_PEARSON)
plt.plot(scores_scan_B[t_0+w-B_0:])  

folder_path = "C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\speechdata"
output_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data"
create_time_series_change_points_CENSREC_1_C(folder_path,output_directory)


################## Open audio
data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\speechdata"
output_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\time_series"



        
############## Open change_point

data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\speechdata"
output_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\change_point"


list_subdirectories=["\\close"+"\\RESTAURANT_SNR_HIGH","\\close"+"\\RESTAURANT_SNR_LOW","\\close"+"\\STREET_SNR_HIGH","\\close"+"\\STREET_SNR_LOW",
                     "\\remote"+"\\RESTAURANT_SNR_HIGH","\\remote"+"\\RESTAURANT_SNR_LOW","\\remote"+"\\STREET_SNR_HIGH","\\remote"+"\\STREET_SNR_LOW"]

sample_rate = 48000 
    
for sub in list_subdirectories:
    label_files=get_label_files_speech(data_directory+sub)
    for f in  label_files:
        print(file_name)
        file_name=data_directory+sub+'\\'+f
        df = pd.read_csv(file_name,header=None,sep=' ')
        df.columns=["begining","end"]
        f=f.replace(".seg",".csv")
        sub_aux=sub.replace("\\","_")
        file_to_save=output_directory+'\\'+sub_aux+"_changepoint"+"_"+f
        df.to_csv(file_to_save)
    
    

########################### Get change_points 

    
time_series_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\time_series"
change_point_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\change_points"
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"
segment_SPEECH_corpus(time_series_directory,change_point_directory,cleaned_data_directory)


scenario='STREET_SNR_HIGH'
folder_path=cleaned_data_directory+"\\"+scenario+"\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)

first_time_series=pd.read_csv(folder_path+time_series_list[0],index_col=0).to_numpy()
first_change_point=pd.read_csv(folder_path+change_point_list[0],index_col=0).to_numpy()

second_time_series=pd.read_csv(folder_path+time_series_list[300],index_col=0).to_numpy()
second_change_point=pd.read_csv(folder_path+change_point_list[300],index_col=0).to_numpy()


data_ref_1=np.array(first_time_series[:300])
time_series_1=np.array(first_time_series[300:])

data_ref_2=np.array(second_time_series[:250])
time_series_2=np.array(second_time_series[250:])


B_0=50
N=30
scores_scan_B=scan_B_statistic(data_ref_1,time_series_1,B_0,N)

scores_kcusum=KCUSUM(data_ref_1,time_series_1,delta=0.5)

alpha=0.1
t_0=300
learning_rate=lambda t: 4.0/((t+t_0)**(2/3))
regularization=lambda t: 1.0/(4*(t+t_0)**(1/3))
w=50

scores_PEARSON=pearsuit_cp(data_ref_1,time_series_1,t_0,w,learning_rate,regularization,alpha)

n=50
threshold_coherence=0.3
scores_NOUGAT=NOUGAT(data_ref_1,time_series_1,n,threshold_coherence)
plt.plot(scores_NOUGAT)

n=50
alpha=0.1
scores_rulsif=rulsif(data_ref_1,time_series_1,n,alpha)

n=50
eta=0.01
gamma=0.01
scores_kliep=kliep(data_ref_1,time_series_1,n,eta,gamma)
plt.plot(scores_kliep)



plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_PEARSON)
plt.axvline(x=first_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_kcusum[t_0+w-1:])
plt.axvline(x=first_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_scan_B[t_0+w-1-B_0:])
plt.axvline(x=first_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_NOUGAT[t_0+w-1-2*n:])
plt.axvline(x=first_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_rulsif[t_0+w-2*n:])
plt.axvline(x=first_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON)),scores_kliep[t_0+w-2*n:])
plt.axvline(x=first_change_point[1][0]-300,color="red")




##################################################### Second_experiment 

second_time_series=pd.read_csv(folder_path+time_series_list[30],index_col=0).to_numpy()
second_change_point=pd.read_csv(folder_path+change_point_list[30],index_col=0).to_numpy()

data_ref_2=np.array(second_time_series[:300])
time_series_2=np.array(second_time_series[300:])
second_change_point=pd.read_csv(folder_path+change_point_list[30],index_col=0).to_numpy()

B_0=50
N=30
scores_scan_B_2=scan_B_statistic(data_ref_2,time_series_2,B_0,N)

scores_kcusum_2=KCUSUM(data_ref_2,time_series_2,delta=0.5)

alpha=0.1
t_0=300
#learning_rate=lambda t: 4.0/((t+t_0)**(2/3))
#regularization=lambda t: 1.0/(4*(t+t_0)**(1/3))
learning_rate=lambda t: 4.0/((t+t_0)**(0.5))
regularization=lambda t: 1.0/(4*(t+t_0)**(0.5))
w=50

scores_PEARSON_2=pearsuit_cp(data_ref_2,time_series_2,t_0,w,learning_rate,regularization,alpha)

alpha=0.1
t_0=300
learning_rate=lambda t: 4.0/((t+t_0)**(2/3))
regularization=lambda t: 1.0/(4*(t+t_0)**(1/3))
#learning_rate=lambda t: 4.0/((t+t_0)**(0.5))
#regularization=lambda t: 1.0/(4*(t+t_0)**(0.5))
w=50

scores_PEARSON_3=pearsuit_cp(data_ref_2,time_series_2,t_0,w,learning_rate,regularization,alpha)

n=50
threshold_coherence=0.3
scores_NOUGAT_2=NOUGAT(data_ref_2,time_series_2,n,threshold_coherence)

n=50
alpha=0.1
scores_rulsif_2=rulsif(data_ref_2,time_series_2,n,alpha)

n=50
eta=0.01
gamma=0.01
scores_kliep_2=kliep(data_ref_2,time_series_2,n,eta,gamma)


plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),time_series_2[t_0+w-1:])
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_PEARSON_2)
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_PEARSON_3)
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_kcusum_2[t_0+w-1:])
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_scan_B_2[t_0+w-1-B_0:])
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_NOUGAT_2[t_0+w-1-2*n:])
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_rulsif_2[t_0+w-2*n:])
plt.axvline(x=second_change_point[1][0]-300,color="red")
plt.plot(np.arange(t_0+w,t_0+w+len(scores_PEARSON_2)),scores_kliep_2[t_0+w-2*n:])
plt.axvline(x=second_change_point[1][0]-300,color="red")

############# Parameters to check. 

##### Scan B-statistic 
### sigma=0.1,0.5,1,2.0,5.0
### N=10,20,30 


######## 10% of sequence to tune the hyperparameters 




B_0=50
N=30
scores_scan_B_2=scan_B_statistic(data_ref_2,time_series_2,B_0,N)


B_0=50
N=30
scores_scan_B_3=scan_B_statistic(data_ref_2,time_series_2,B_0,N,scale=5.0)


cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"

scenario='STREET_SNR_HIGH'
folder_path=cleaned_data_directory+"\\"+scenario+"\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
test_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"test_set\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)
len(change_point_list)
len(change_point_list)

np.random.seed(0)
index_train_set=np.random.randint(len(time_series_list),size=70)
index_test_set=set(np.arange(0,len(time_series_list)))-set(index_train_set)

for i in index_train_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(train_data_path+time_series_list[i])
    change_points_.to_csv(train_data_path+change_point_list[i])


for i in index_test_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(test_data_path+time_series_list[i])
    change_points_.to_csv(test_data_path+change_point_list[i])
    
scenario='STREET_SNR_LOW'
folder_path=cleaned_data_directory+"\\"+scenario+"\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
test_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"test_set\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)
len(change_point_list)
len(change_point_list)

np.random.seed(1)
index_train_set=np.random.randint(len(time_series_list),size=70)
index_test_set=set(np.arange(0,len(time_series_list)))-set(index_train_set)

for i in index_train_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(train_data_path+time_series_list[i])
    change_points_.to_csv(train_data_path+change_point_list[i])


for i in index_test_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(test_data_path+time_series_list[i])
    change_points_.to_csv(test_data_path+change_point_list[i])    
    
#####################################


scenario='RESTAURANT_SNR_HIGH'
folder_path=cleaned_data_directory+"\\"+scenario+"\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
test_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"test_set\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)
len(change_point_list)
len(change_point_list)

np.random.seed(2)
index_train_set=np.random.randint(len(time_series_list),size=70)
index_test_set=set(np.arange(0,len(time_series_list)))-set(index_train_set)

for i in index_train_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(train_data_path+time_series_list[i])
    change_points_.to_csv(train_data_path+change_point_list[i])


for i in index_test_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(test_data_path+time_series_list[i])
    change_points_.to_csv(test_data_path+change_point_list[i])  


############################################


scenario='RESTAURANT_SNR_LOW'
folder_path=cleaned_data_directory+"\\"+scenario+"\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
test_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"test_set\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)
len(change_point_list)
len(change_point_list)

np.random.seed(3)
index_train_set=np.random.randint(len(time_series_list),size=70)
index_test_set=set(np.arange(0,len(time_series_list)))-set(index_train_set)

for i in index_train_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(train_data_path+time_series_list[i])
    change_points_.to_csv(train_data_path+change_point_list[i])


for i in index_test_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(test_data_path+time_series_list[i])
    change_points_.to_csv(test_data_path+change_point_list[i])  


###########################################

from itertools import product

scenario='STREET_SNR_LOW'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"
train_data_path=cleaned_data_directory+scenario+"\\"+"train_set\\"
results_path=cleaned_data_directory+scenario+"\\"+"results\\"
B_0=50

def find_parameter_scanB_statistic():
    
    sigmas=[0.1,0.5,1,2.0,5.0]
    Ns=[10,20,30]
    
    parameters_combination=list(product(sigmas,Ns))
    
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"N:"+str(pc[1])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]
    
    for pc in parameters_combination:
        print(pc)
        for i in range(len(time_series_list)):
            data_ref=time_series_[i][:300]
            time_series=time_series_[i][300:]
            scores=np.hstack(scan_B_statistic(data_ref,time_series,B_0,N=pc[1],scale=pc[0]))
            dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"N:"+str(pc[1])].append(copy.deepcopy(scores))
            
    with open(results_path+'scan_B_statistic.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
            
#########################     

scenario='STREET_SNR_HIGH'

results_path=cleaned_data_directory+scenario+"\\"+"results\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
def find_parameter_KERNELCUSUM(train_data_path,results_path): 
    
    sigmas=[0.1,0.5,1,2.0,5.0]
    deltas=[0.0,0.1,0.5,1.0]
    
    parameters_combination=list(product(sigmas,deltas))
    
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"deltas:"+str(pc[1])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]
    
    for pc in parameters_combination:
        print(pc)
        for i in range(len(time_series_list)):
            data_ref=time_series_[i][:300]
            time_series=time_series_[i][300:]
            scores=np.hstack(KCUSUM(data_ref,time_series,delta=pc[1],scale=pc[0]))
            dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"deltas:"+str(pc[1])].append(copy.deepcopy(scores))
   
    with open(results_path+'kernel_cusum.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
    
#######################################

#scenario='STREET_SNR_HIGH'
#scenario='STREET_SNR_LOW'
#scenario='RESTAURANT_SNR_LOW'
#scenario='RESTAURANT_SNR_HIGH'
results_path=cleaned_data_directory+scenario+"\\"+"results\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
n=50

def find_parameter_NOUGAT(train_data_path,results_path,n): 
    
    lamb=np.logspace(-6,1,6)
    sigmas=[0.1,0.5,1.0,2.0,5.0]
    coherence=[0.1,0.3,0.5]
    
    parameters_combination=list(product(lamb,sigmas,coherence))
    
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["lamb:"+str(pc[1])+"sigma_scale:"+str(pc[1])+" "+"coherence:"+str(pc[2])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]
 
    for pc in parameters_combination:
        print(pc)
        for i in range(len(time_series_list)):
            data_ref=time_series_[i][:300]
            time_series=time_series_[i][300:]
            scores=NOUGAT(data_ref,time_series,n,threshold_coherence=pc[2],rulsif_initialization=False,scale=pc[1],lamb=pc[0])
            dictionary_parameters["lamb:"+str(pc[1])+"sigma_scale:"+str(pc[1])+" "+"coherence:"+str(pc[2])].append(copy.deepcopy(scores))
   
    with open(results_path+'nougat.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
########################### RULSIF

#scenario='STREET_SNR_HIGH'
#scenario='STREET_SNR_LOW'
scenario='RESTAURANT_SNR_LOW'
#scenario='RESTAURANT_SNR_HIGH'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"
results_path=cleaned_data_directory+scenario+"\\"+"results\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
n=50
alpha=[0.1,0.5,0.9]

def find_parameter_RULSIF(train_data_path,results_path,n): 
    
    alphas=[0.1,0.5,0.9]

    dictionary_parameters=dict()
    for alpha in alphas:
        dictionary_parameters["alpha:"+str(alpha)]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]

    for alpha in alphas:
        print(alpha)
        for i in range(len(time_series_list)):
            data_ref=time_series_[i][:300]
            time_series=time_series_[i][300:]
            scores=rulsif(data_ref,time_series,n,alpha)
            dictionary_parameters["alpha:"+str(alpha)].append(copy.deepcopy(scores))
   
    with open(results_path+'rulsif.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  

################################# 


#scenario='STREET_SNR_HIGH'
#scenario='STREET_SNR_LOW'
#scenario='RESTAURANT_SNR_LOW'
scenario='RESTAURANT_SNR_HIGH'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"
results_path=cleaned_data_directory+scenario+"\\"+"results\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
n=50

def find_parameter_KLIEP(train_data_path,results_path,n): 
    
    learning_rate=[0.1,0.5,0.9]
    regularization=[0.1,0.5,0.9]
    parameters_combination=list(product(learning_rate,regularization))
 
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["eta:"+str(pc[0])+"gamma:"+str(pc[1])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]

    for pc in parameters_combination:
     print(pc)
     for i in range(len(time_series_list)):
         data_ref=time_series_[i][:300]
         time_series=time_series_[i][300:]
         scores=kliep(data_ref,time_series,n,eta=pc[0],gamma=pc[1])
         dictionary_parameters["eta:"+str(pc[0])+"gamma:"+str(pc[1])].append(copy.deepcopy(scores))
 
    with open(results_path+'kliep.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
################### PEARSUIT


#scenario='STREET_SNR_HIGH'
#scenario='STREET_SNR_LOW'
#scenario='RESTAURANT_SNR_LOW'
scenario='RESTAURANT_SNR_HIGH'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\CENSREC-1-C\\cleaned_data\\"
results_path=cleaned_data_directory+scenario+"\\"+"results\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
n=50

def find_parameter_PEARSUIT(train_data_path,results_path,n): 
    
    r=[0.5,1.0]
    t_0=[50,100,300]
    alphas=[0.1,0.5,0.9]
    
    parameters_combination=list(product(r,t_0,alphas))
 
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["r:"+str(pc[0])+"t_0:"+str(pc[1])+"alpha:"+str(pc[2])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]

    for pc in parameters_combination:
     print(pc)
     for i in range(len(time_series_list)):
         data_ref=time_series_[i][:300]
         time_series=time_series_[i][300:]
         r=pc[0]
         t_0=pc[1]
         alpha=pc[2]
         learning_rate=lambda t: 4.0/((t+t_0)**((2*r)/(2*r+1)))
         regularization=lambda t: 1.0/(4*(t+t_0)**(1/(2*r+1)))
         scores=pearsuit_cp(data_ref,time_series,t_0,n,learning_rate,regularization,alpha)
         dictionary_parameters["r:"+str(pc[0])+"t_0:"+str(pc[1])+"alpha:"+str(pc[2])].append(copy.deepcopy(scores))
 
    with open(results_path+'pearsuit.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

################################################ NOUGAT 
activity=["walk","jog","stay","stDown","stUP","skip"]    

scenario='skip'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data"
folder_path=cleaned_data_directory+"\\"+scenario+"\\"
train_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"train_set\\"
test_data_path=cleaned_data_directory+"\\"+scenario+"\\"+"test_set\\"

time_series_list=get_time_series(folder_path)
change_point_list=get_change_points(folder_path)

np.random.seed(0)
index_train_set=np.random.randint(len(time_series_list),size=int(len(time_series_list)/3))
index_test_set=set(np.arange(0,len(time_series_list)))-set(index_train_set)

for i in index_train_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(train_data_path+time_series_list[i])
    change_points_.to_csv(train_data_path+change_point_list[i])


for i in index_test_set: 
    time_series_=pd.read_csv(folder_path+time_series_list[i],index_col=0)
    change_points_=pd.read_csv(folder_path+change_point_list[i],index_col=0)
    time_series_.to_csv(test_data_path+time_series_list[i])
    change_points_.to_csv(test_data_path+change_point_list[i])
 
#############################

from itertools import product

activity=["walk","jog","stay","stDown","stUP","skip"]    

scenario='\\walk'
cleaned_data_directory="C:\\Users\\33768\\Desktop\\datasets\\HASC2011corpus\\cleaned_data"
train_data_path=cleaned_data_directory+scenario+"\\"+"train_set\\"
results_path=cleaned_data_directory+scenario+"\\"+"results\\"

def find_parameter_scanB_statistic():
    
    sigmas=[0.1,0.5,1,2.0,5.0]
    Ns=[10,20,30]
    
    parameters_combination=list(product(sigmas,Ns))
    
    dictionary_parameters=dict()
    for pc in parameters_combination:
        dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"N:"+str(pc[1])]=[]
        
    time_series_list=get_time_series(train_data_path)
    change_point_list=get_change_points(train_data_path)
    
    time_series_=[pd.read_csv(train_data_path+t,index_col=0).to_numpy() for t in time_series_list]
    change_points_=[pd.read_csv(train_data_path+cp,index_col=0).to_numpy() for cp in  change_point_list]
    
    for pc in parameters_combination:
        print(pc)
        for i in range(len(time_series_list)):
            data_ref=time_series_[i][:300]
            time_series=time_series_[i][300:]
            scores=np.hstack(scan_B_statistic(data_ref,time_series,B_0,N=pc[1],scale=pc[0]))
            dictionary_parameters["sigma_scale:"+str(pc[0])+" "+"N:"+str(pc[1])].append(copy.deepcopy(scores))
            
    with open(results_path+'scan_B_statistic.pickle', 'wb') as handle:
        pickle.dump(dictionary_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)







        


