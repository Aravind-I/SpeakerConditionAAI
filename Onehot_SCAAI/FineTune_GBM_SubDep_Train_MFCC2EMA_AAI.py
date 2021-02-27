#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint ,EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.utils.generic_utils import Progbar
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from keras.layers import LSTM, Dropout, GRU, Convolution1D,  MaxPooling1D, Flatten,Reshape
from keras.layers import Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
import sys
import numpy
import librosa
import tensorflow as tf
import HTK
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import pearsonr
from keras.models import load_model

# In[10]:


NoUnits=256 #LSTM units
BatchSize=50 #50
NoEpoch=50
htkfile = HTK.HTKFile()
std_frac=0.25
n_mfcc=13
inputDim=n_mfcc


# In[11]:


def Get_Wav_EMA_PerFile(EMA_file,Wav_file,F):
    EmaMat=scipy.io.loadmat(EmaDir+EMA_file);
    EMA_temp=EmaMat['EmaData'];
    EMA_temp=np.transpose(EMA_temp)# time X 18
    Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12
    MeanOfData=np.mean(Ema_temp2,axis=0) 
    Ema_temp2-=MeanOfData
    C=0.5*np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
    Ema=np.divide(Ema_temp2,C) # Mean remov & var normailized
    [aE,bE]=Ema.shape
    
    #print F.type
    EBegin=np.int(BeginEnd[0,F]*100)
    EEnd=np.int(BeginEnd[1,F]*100)
    
    ### MFCC ###
    htkfile.load(MFCCpath+Wav_file[:-4]+'.mfc')
    feats = np.asarray(htkfile.data)
    mean_G = np.mean(feats, axis=0)
    std_G = np.std(feats, axis=0)
    feats = std_frac*(feats-mean_G)/std_G
    MFCC_G = feats
    TimeStepsTrack=EEnd-EBegin
    return Ema[EBegin:EEnd,:], MFCC_G[EBegin:EEnd,:n_mfcc],TimeStepsTrack # with out silence

def EvalMetric(X,Y):
    CC=[pearsonr(X[:,i],Y[:,i]) for i in range(0,12)]
    rMSE=np.sqrt(np.mean(np.square(X-Y),axis=0))
    return np.array(CC)[:,0],rMSE

# In[12]:


DataDir= '/home2/data/ARAVIND/End2End/SPIRE_EMA/DataBase/'
Subs=os.listdir(DataDir)
TrainMsub=[ 'AshwinHebbar', 'VigneshM', 'Advith', 'NikhilB', 'Pavan_P', 'AdvaithP', 'Shoubik', 'PhaniKumar', 'Prakhar_G','Parth_S']
TrainFsub=[ 'Babitha', 'DivyaGR','Vidhi', 'Anwesa', 'Nisha', 'Chandana','Varshini','Rupasi','Shiny','AtreyeeS']
ValMsub=['Ashwin_N', 'AvinashKumar']
ValFsub=['Tanaya']

RemainingSubs=['RaviKiran_R', 'Samik', 'Pavan', 'ManuG', 'Vignesh', 'GokulS']


# In[13]:


ExcludeSet1=['Ankur', 'Anand_S' , 'Prasad','Ankur_C','Monika', 'SriRamya','Harshini','Jisha','Anisha']


# In[14]:


Valsubs=ValMsub+ValFsub
Trainsubs=TrainMsub+TrainFsub


print('..compiling model')
mdninput_Lstm = keras.Input(shape=(None,inputDim))
mdninput_LstmD=TimeDistributed(Dense(200))(mdninput_Lstm)
lstm_1=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(mdninput_LstmD)
lstm_2a=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_1)
lstm_2=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_2a)
output=TimeDistributed(Dense(12, activation='linear'))(lstm_2)
model = keras.models.Model(mdninput_Lstm,output)
model.summary()
print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
model.compile(optimizer='adam', loss='mse')

# In[18]:
TSCC=np.zeros((1,12))
TSrmse=np.zeros((1,12))


OutDir='FineTune_SD_AAI_Model/'
RootDir='/home2/data/ARAVIND/End2End/SPIRE_EMA/'
for ss in np.arange(0,len(Trainsubs)):

    X_valseq=[];Youtval=[];
    X_trainseq=[];Youttrain=[];
    X_testseq=[];Youttest=[];
    TT_Test=[];TT_Train=[];TT_Valid=[]
    print('Loading Training data')
#for ss in np.arange(0,len(Trainsubs)):
    Sub=Trainsubs[ss]#'Anand_S'
    print(Sub)
    WavDir=RootDir+'DataBase/'+Sub+'/Neutral/WavClean/';
    EmaDir=RootDir+'DataBase/'+Sub+'/Neutral/EmaClean/';
    BeginEndDir=RootDir+'/StartStopMat/'+Sub+'/';
    MFCCpath=RootDir+'/DataBase/'+Sub+'/Neutral/MfccHTK/'

    EMAfiles=sorted(os.listdir(EmaDir))
    Wavfiles=sorted(os.listdir(WavDir))
    StartStopFile=os.listdir(BeginEndDir)
    StartStopMAt=scipy.io.loadmat(BeginEndDir+StartStopFile[0])
    BeginEnd=StartStopMAt['BGEN']
    #window_size=500


    F=5 # Fold No
    '''
    for i in np.arange(0,460):
        E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
        #W_t=W_t#[np.newaxis,:,:,:]
        E_t=E_t#[np.newaxis,:,:]
        M_t=M_t#[np.newaxis,:,:]
        Youttrain.append(E_t)
        X_trainseq.append(M_t)
        TT_Train.append(TT)
    '''
    for i in np.arange(0,460):
        if  (((i+F)%10)==0):# Test
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            #W_t=W_t[np.newaxis,:,:,np.newaxis]
            E_t=E_t[np.newaxis,:,:]
            M_t=M_t[np.newaxis,:,:]
            Youttest.append(E_t)
            X_testseq.append(M_t)
            TT_Test.append(TT)
            #print('Test '+str(i))
        elif (((i+F+1)%10)==0):# Validation
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            #W_t=W_t#[np.newaxis,:,:,:]
            E_t=E_t#[np.newaxis,:,:]
            M_t=M_t #forward_process(M_t[:,:n_mcep])
            Youtval.append(E_t)
            X_valseq.append(M_t)
            TT_Valid.append(TT)
        else: # Train
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            #W_t=W_t#[np.newaxis,:,:,:]
            E_t=E_t#[np.newaxis,:,:]
            M_t=M_t #forward_process(M_t[:,:n_mcep])#[np.newaxis,:,:]
            Youttrain.append(E_t)
            X_trainseq.append(M_t)
            TT_Train.append(TT)
    # In[23]:


    print("No:of Train Samples:"+str(len(X_trainseq)))
    print("No:of Val Samples:"+str(len(X_valseq)))


    # In[24]:


    #https://github.com/keras-team/keras/issues/9788
    TT_Total=(np.concatenate([np.array(TT_Test),np.array(TT_Valid),np.array(TT_Train)]))
    TT_max=400#np.max(TT_Total)
    #padded = pad_sequences(sequences, padding='post',maxlen=TT_max)
    X_valseq=pad_sequences(X_valseq, padding='post',maxlen=TT_max,dtype='float')
    Youtval=pad_sequences(Youtval, padding='post',maxlen=TT_max,dtype='float')
    X_trainseq=pad_sequences(X_trainseq, padding='post',maxlen=TT_max,dtype='float')
    Youttrain=pad_sequences(Youttrain, padding='post',maxlen=TT_max,dtype='float')
    #X_testseq=pad_sequences(X_testseq, padding='post',maxlen=TT_max)
    #Youttest=pad_sequences(Youttest, padding='post',maxlen=TT_max)


    # In[ ]:

    '''
    print('..compiling model')
    mdninput_Lstm = keras.Input(shape=(None,inputDim))
    mdninput_LstmD=TimeDistributed(Dense(200))(mdninput_Lstm)
    lstm_1=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(mdninput_LstmD)
    lstm_2a=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_1)
    lstm_2=Bidirectional(LSTM(NoUnits, return_sequences=True,activation='tanh'))(lstm_2a)
    output=TimeDistributed(Dense(12, activation='linear'))(lstm_2)
    model = keras.models.Model(mdninput_Lstm,output)
    model.summary()
    
    print('\n\nModel with input size {}, output size {}'.format(model.input_shape, model.output_shape))
    model.compile(optimizer='adam', loss='mse')
    '''
    print("Loading Pooled Model")
    PoolModelName='Pooled_AAI_Model/Pool_AAI_Batch50_AtreyeeS_LSTMunits_256__.h5'
    model=load_model(PoolModelName)
    #OutFileName='final_mfcc_'+Sub+'_F1_'+str(CNN1_filters)+'_len'+str(CNN1_flength)+'_F2_'+str(CNN2_filters)+'_len'+str(CNN2_flength)+'LSTMunits_'+str(NoUnits)+'_'
    OutFileName='FineTune_SD_AAI_'+'Batch'+ str(BatchSize)+'_'+Sub+'_LSTMunits_'+str(NoUnits)+'_'
    fName=OutFileName
    print('..fitting model')

    checkpointer = ModelCheckpoint(filepath=OutDir+fName + '_.h5', verbose=0, save_best_only=True)
    checkpointer1 = ModelCheckpoint(filepath=OutDir+fName + '_weights.h5', verbose=0, save_best_only=True, save_weights_only=True)
    earlystopper =EarlyStopping(monitor='val_loss', patience=5)
    history=model.fit(X_trainseq,Youttrain,validation_data=(X_valseq,Youtval),nb_epoch=NoEpoch, batch_size=BatchSize,verbose=1,shuffle=True,callbacks=[checkpointer,checkpointer1,earlystopper])


    print('====Testing=====')
    print("No:of Train Samples:"+str(len(X_testseq)))
    predSeq=np.empty((1,len(Youttest)), dtype=object);
    YtestOrg=np.empty((1,len(Youttest)), dtype=object);
    tCC=[]
    trMSE=[]
    for i in np.arange(0,len(Youttest)):
        s_in=X_testseq[i]
        #s_in=s_in[np.newaxis,:,0:inputDim]
        val=model.predict(s_in);
        predSeq[0,i]=val
        #InSeq[0,i]=s_in
        YtestOrg[0,i]=Youttest[i]
        iCC,irMSE=EvalMetric(np.squeeze(val),np.squeeze(Youttest[i]))
        tCC.append(iCC)
        trMSE.append(irMSE)


    print(np.mean(np.array(tCC),axis=0))
    print(np.mean(np.array(trMSE),axis=0))



    TSrmse=np.concatenate((TSrmse,np.mean(np.array(trMSE),axis=0)[np.newaxis,:]),axis=0)
    TSCC=np.concatenate((TSCC,np.mean(np.array(tCC),axis=0)[np.newaxis,:]),axis=0)


    # In[68]:


scipy.io.savemat(OutDir+OutFileName+'_Results.mat',{'RMSE':TSrmse,'Corr':TSCC})
