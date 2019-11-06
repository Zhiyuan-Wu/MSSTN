import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import os
import yaml
from data_utils import dataset

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
    clf = SGDRegressor(loss='squared_loss',random_state=2019,learning_rate='optimal',max_iter=1000)

    clf.fit(trainingFeatureSet, trainingLabelSet)
    result = clf.predict(testingFeatureSet)
    mae = np.mean(np.abs(result-testingLabelSet))*500
    rmse = np.sqrt(np.mean(np.square(result-testingLabelSet)))*500
    print(config['city_name_'+str(i)],'=======',mae,rmse)
