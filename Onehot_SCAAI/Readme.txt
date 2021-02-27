JASA-EL Codes: 
======
Cross_Embed_Eval_MFCC2EMA_AAI.py:
load Trained SC-AAI model and do cross speaker evaluation; Fig 4
This just save the RMSE and CC

Cross_Embed_Save_estimate_AAI.py:
load Trained SC-AAI model and do cross speaker evaluation; Fig 4
This code saves the direct estimation values

Embed_Eval_MFCC2EMA_AAI.py:
load Trained SC-AAI model and do Matched speaker evaluation; Fig 4
This just save the RMSE and CC

Embed_Train_MFCC2EMA_AAI.py:
Training SC-AAI model with one-hot speaker emeddings

FineTune_GBM_SubDep_Train_MFCC2EMA_AAI.py:
load GBM model and fine tune for each speaker (Pooled_AAI_Model/Pool_AAI_Batch50_AtreyeeS_LSTMunits_256__.h5)
FineTune_SD_AAI-- GBM+SD finetuning
GM-FSD

FineTune_Models_Save_estimate_AAI.py:
Eval  GBM+SD finetuning model
This code saves the direct estimation values
GM-FSD

Utils:
HTK.py

Pool_Model_Save_estimate_AAI.py:
load GBM-AAI model and save the  direct estimation values

Pool_Train_MFCC2EMA_AAI.py:
Train GBM-AAI model

SubDep_Model_estimate_AAI.py:
load SD-AAI model and save the  direct estimation values

SubDep_Train_MFCC2EMA_AAI.py:
Train SD-AAI model

Finetune_SC_AAI.py
Finetune SC-AAI model

SPK_ID_MFCC/Pool_Train_MFCC2spkId.py
Train closed set speaker ID network
=======


