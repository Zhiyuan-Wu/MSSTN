import numpy as np
import tensorflow as tf
from model import DCRNN,LSTM
from data_utils import *
import yaml
import os
from sys import argv
import time

if __name__=='__main__':
    with open('config.yaml') as f:
        config = yaml.load(f)    
    model = DCRNN(config)
    print('Model Setup Done.')
    data = dataset(config)

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_device'])
    version = time.strftime('%Y%m%d_%H%M%S')
    np.random.seed(2333333)
    tf.set_random_seed(2333333)
    tfcfg = tf.ConfigProto()
    saver = tf.train.Saver()
    #tfcfg.gpu_options.allow_growth = True
    with tf.Session(config=tfcfg) as sess:
        if len(argv)==1 or 'train' in argv[1]:
            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, save_path='model/'+'ReiWa20190404_124825'+'/model')
            os.mkdir('model/'+config['ModelName']+version)
            lr = float(config['learning_rate'])
            best_model_recorder = config['BaseLineMAE']
            moving_average_recorder = 0
            for epoch in range(config['epoch_num']):
                train_ls = []
                for batch_num in range(data.tr_batch_num):
                    batch_mid,batch_set = data.tr_get_batch()
                    feed_dict = {model.learning_rate:lr}
                    for i,batch in enumerate(batch_set):
                        feed_dict[model.input_set[i]] = batch
                    _,ls = sess.run([model.train_op,model.loss],feed_dict)
                    #ls2 = sess.run([model.loss_mid]+[model.loss_set[i] for i in config['city_number']],feed_dict)
                    train_ls.append(ls)

                if (epoch+1)%config['print_every_n_epochs'] == 0:
                    train_ls = np.mean(train_ls)*500
                    print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],' Done, Train loss ',round(train_ls,4))
                
                if (epoch+1)%config['learning_rate_decay_every_n_epochs'] == 0:
                    lr = lr/config['learning_rate_decay']
                    print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],', Learning rate decay to ',lr)

                if (epoch+1)%config['test_every_n_epochs'] == 0:
                    test_ls = []
                    test_ls2 = []
                    test_ls3 = []
                    for batch_num in range(data.te_batch_num):
                        batch_mid,batch_set = data.te_get_batch()
                        feed_dict = {model.learning_rate:lr}
                        for i,batch in enumerate(batch_set):
                            feed_dict[model.input_set[i]] = batch
                        ls1 = sess.run(model.loss,feed_dict)
                        ls2 = sess.run([model.loss_mid]+[model.loss_set[i] for i in range(config['city_number'])],feed_dict)
                        ls3 = sess.run(model.loss_rmse_set,feed_dict)
                        test_ls.append(ls1)
                        test_ls2.append(ls2)
                        test_ls3.append(ls3)
                    test_ls = np.mean(test_ls)*500
                    test_ls2 = np.mean(test_ls2,0)*500
                    test_ls3 = np.sqrt(np.mean(test_ls3,0))*500
                    moving_average_recorder = config['moving_average_factor']*moving_average_recorder+(1-config['moving_average_factor'])*test_ls
                    print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],' Done, Test loss ',round(test_ls,4),'/',np.round(test_ls2,4),'/',np.round(test_ls3,4))
                    if test_ls2[1]<best_model_recorder:
                        saver.save(sess,'model/'+config['ModelName']+version+'/model')
                        print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],' Model Save Success. New record ',test_ls2[1])
                        lr = lr/(config['learning_rate_decay_at_best']*best_model_recorder/test_ls2[1]-config['learning_rate_decay_at_best']+1)
                        print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],', Learning rate decay to ',lr)
                        best_model_recorder = test_ls2[1]

            print('==============')
            print('['+config['ModelName']+version+']Training DONE. ')
            print('best_model_recorder ',best_model_recorder)
            print('moving_average_recorder ',moving_average_recorder)

            _debug = np.array([2,3,3])
        if len(argv)>1 and 'test' in argv[1]:
            test_ls = []
            test_ls2 = []
            test_ls3 = []
            for batch_num in range(data.te_batch_num):
                batch_mid,batch_set = data.te_get_batch()
                feed_dict = {model.learning_rate:lr}
                for i,batch in enumerate(batch_set):
                    feed_dict[model.input_set[i]] = batch
                ls1 = sess.run(model.loss,feed_dict)
                ls2 = sess.run([model.loss_mid]+[model.loss_set[i] for i in range(config['city_number'])],feed_dict)
                ls3 = sess.run(model.loss_rmse_set,feed_dict)
                test_ls.append(ls1)
                test_ls2.append(ls2)
                test_ls3.append(ls3)
            test_ls = np.mean(test_ls)*500
            test_ls2 = np.mean(test_ls2,0)*500
            test_ls3 = np.sqrt(np.mean(test_ls3,0))*500
            moving_average_recorder = config['moving_average_factor']*moving_average_recorder+(1-config['moving_average_factor'])*test_ls
            print('['+config['ModelName']+version+']epoch ',epoch,'/',config['epoch_num'],' Done, Test loss ',round(test_ls,4),'/',np.round(test_ls2,4),'/',np.round(test_ls3,4))