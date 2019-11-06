import numpy as np
import pandas as pd
from sklearn import svm
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

k = ['rbf','linear'][0]

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
    
    clf = svm.SVR(kernel=k)
    perm = np.linspace(0,trainingFeatureSet.shape[0]-1,trainingFeatureSet.shape[0])
    np.random.shuffle(perm)
    perm = perm[:20000].astype(int)
    clf.fit(trainingFeatureSet[perm], trainingLabelSet[perm])
    result = clf.predict(testingFeatureSet)
    mae = np.mean(np.abs(result-testingLabelSet))*500
    rmse = np.sqrt(np.mean(np.square(result-testingLabelSet)))*500
    print(config['city_name_'+str(i)],'=======',mae,rmse)


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
            pd = clf.predict(np.reshape(_dataSlice,[num_node,-1]))
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
