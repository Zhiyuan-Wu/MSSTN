import numpy as np
import tensorflow as tf
from model import ReiWa
from data_utils import *
import yaml
import os
from sys import argv,stdout
import time

def _aug(batch):
    new = batch[:,-1:,:]
    return np.concatenate((batch,new),axis=1)

def _lossfuc_mae(x,y):
    return np.mean(np.abs(x-y))

def _lossfuc_rmse(x,y):
    return np.sqrt(np.mean(np.square(x-y)))

def _lossfuc_smape(x,y):
    return 2*np.mean(np.abs(x-y)/(x+y))

def draw_pred(gt,pd,n=0,A=500):
    N = pd.shape[1]
    idx = list(np.arange(N)[pd[0,:,1]==-233])
    gt = gt*A
    pd = pd*A
    plt.plot(np.arange(N),gt[n,:,0],'r')
    plt.plot(np.arange(N),pd[n,:,0],'b--')    
    for i in idx:
        plt.vlines(i,np.min((gt[n,i,0],pd[n,i,0]))-5,np.max((gt[n,i,0],pd[n,i,0]))+5,linestyles='dashed')
    plt.ylabel('PM2.5 Concentration',fontsize='large')
    plt.title('Prediction Result',fontsize='large')
    plt.legend(['Ground Truth','Prediction'],fontsize='large')
    plt.show()    

def infer(model,sess,te,te2,InferenceLength,name):
    num_nodes = te.shape[0]
    time_length = te.shape[1]
    input_channels = te.shape[2]

    input_length = 81
    output = np.zeros((time_length,num_nodes,InferenceLength))
    output2 = np.zeros((time_length,te2.shape[0],InferenceLength))
    counter = 0
    while counter+input_length+InferenceLength<time_length:
        _process = round(counter/time_length*100.0,2)
        stdout.write(str(_process)+'% Inference Finished          \r')
        stdout.flush()
        _dataSlice = te[:,counter:counter+input_length]
        _dataSlice2 = te2[:,counter:counter+input_length]
        for i in range(InferenceLength):
            pd,pd2 = sess.run([model.pred_set[0],model.pred_mid],{model.input_set[0]:_aug(_dataSlice),model.input_mid:_aug(_dataSlice2)})
            output[counter+input_length,:,i] = pd[:,-1,0]
            output2[counter+input_length,:,i] = pd2[:,-1,0]
            _dataSlice = _aug(_dataSlice)[:,1:]
            _dataSlice2 = _aug(_dataSlice2)[:,1:]
            _dataSlice[:,-1,:] = te[:,counter+input_length+i,:]
            _dataSlice[:,-1,0] = pd[:,-1,0]
            _dataSlice2[:,-1,:] = te2[:,counter+input_length+i,:]
            _dataSlice2[:,-1,0] = pd2[:,-1,0]
        counter = counter+1

    np.save("result/"+name+"_pred.npy",output)
    np.save("result/"+name+"_raw.npy",te)
    
if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.load(f)
        config['seq_len'] = 82
        model_version = config['Inference_Model']
    

    model = ReiWa(config)
    data = dataset(config)
    version = time.strftime('%Y%m%d_%H%M%S')
    tfcfg = tf.ConfigProto()
    saver = tf.train.Saver()
    tfcfg.gpu_options.allow_growth = True
    with tf.Session(config=tfcfg) as sess:
        saver.restore(sess, save_path='model/'+model_version+'/model')
        infer(model,sess,data.te[0],data.te_mid,48,version)
    _debug = np.array([2,3,3])
    #