import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import yaml
from data_utils import dataset
from sys import argv,stdout

def _aug(batch):
    new = batch[:,-1:,:]
    return np.concatenate((batch,new),axis=1)

def _lossfuc_mae(x,y):
    return np.mean(np.abs(x-y))

def _lossfuc_rmse(x,y):
    return np.sqrt(np.mean(np.square(x-y)))

def _lossfuc_smape(x,y):
    return 2*np.mean(np.abs(x-y)/(x+y))

with open('config.yaml') as f:
    config = yaml.load(f) 
data = dataset(config)
for i in range(config['city_number']):
    tr = data.tr[i]
    te = data.te[i]
    WindowSize = 81
    N,T,C = tr.shape
    trainingFeatureSet = []
    trainingLabelSet = []
    for n in range(N):
        t = 0
        while t+WindowSize<T:
            trainingFeatureSet.append(tr[n,t:t+WindowSize,:].flatten())
            trainingLabelSet.append(tr[n,t+WindowSize,0])
            t = t+1
    trainingFeatureSet = np.array(trainingFeatureSet)
    trainingLabelSet = np.array(trainingLabelSet)
    
    N,T,C = te.shape
    testingFeatureSet = []
    testingLabelSet = []
    for n in range(N):
        t = 0
        while t+WindowSize<T:
            testingFeatureSet.append(te[n,t:t+WindowSize,:].flatten())
            testingLabelSet.append(te[n,t+WindowSize,0])
            t = t+1
    testingFeatureSet = np.array(testingFeatureSet)
    testingLabelSet = np.array(testingLabelSet)
    
    # One-Step
    train_data = lgb.Dataset(trainingFeatureSet, label=trainingLabelSet)
    test_data = lgb.Dataset(testingFeatureSet, label=testingLabelSet)
    lgb_model = lgb.LGBMRegressor(boosting_type="gbdt", num_leaves=81, reg_alpha=0, reg_lambda=0.01,
        max_depth=-1, n_estimators=2000, objective='mae',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=0,min_child_samples = 50,  learning_rate=0.07, random_state=2019, metric="mae",n_jobs=-1)
    eval_set = [(testingFeatureSet, testingLabelSet)]
    lgb_model.fit(trainingFeatureSet, trainingLabelSet, eval_set=eval_set, eval_metric=['mae','rmse'], verbose=10, early_stopping_rounds=30)
    print(config['city_name_'+str(i)],'=======',lgb_model.best_score_)


    # Inference
    input_length = 81
    InferenceLength = 48
    time_length = te.shape[1]
    num_node = te.shape[0]
    output = np.array(te)
    pred_point = []
    counter = 0
    while counter+input_length+InferenceLength<time_length:
        _process = round(counter/time_length*100.0,2)
        stdout.write(str(_process)+'% Inference Finished          \r')
        stdout.flush()
        _dataSlice = te[:,counter:counter+input_length]
        for i in range(InferenceLength):
            pd = lgb_model.predict(np.reshape(_dataSlice,[num_node,-1]))
            output[:,counter+input_length+i,0] = pd
            _dataSlice = _aug(_dataSlice)[:,1:]
            _dataSlice[:,-1,:] = output[:,counter+input_length+i,:]
        pred_point.append(counter+input_length)
        counter = counter+InferenceLength
    
    pred_point = np.array(pred_point)
    loss_mae = []
    loss_rmse = []
    for i in range(InferenceLength):
        loss_mae.append(_lossfuc_mae(output[:,pred_point+i,0],te[:,pred_point+i,0]))
        loss_rmse.append(_lossfuc_rmse(output[:,pred_point+i,0],te[:,pred_point+i,0]))
    print('Inference Done.')
    #print('MAE Loss: ',np.array(loss_mae)*500)
    #print('RMSE Loss: ',np.array(loss_rmse)*500)
    for i in [3,6,12,24,48]:
        if i>InferenceLength: break
        print('MAE Average: ',np.mean(loss_mae[:i])*500)
        print('RMSE Average: ',_lossfuc_rmse(np.array(loss_rmse[:i]),0)*500)


