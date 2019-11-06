import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg

class DCRNN(object):
    def __init__(self, args):
        self.default_model(args)
        
    
    def default_model(self,args):

        def calculate_normalized_laplacian(adj):
            # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
            adj = sp.coo_matrix(adj)
            d = np.array(adj.sum(1))
            d_inv_sqrt = np.power(d, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
            return normalized_laplacian
        
        def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
            if undirected:
                adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
            L = calculate_normalized_laplacian(adj_mx)
            if lambda_max is None:
                lambda_max, _ = linalg.eigsh(L, 1, which='LM')
                lambda_max = lambda_max[0]
            L = sp.csr_matrix(L)
            M, _ = L.shape
            I = sp.identity(M, format='csr', dtype=L.dtype)
            L = (2 / lambda_max * L) - I
            return L.astype(np.float32)
        
        def _build_sparse_matrix(L):
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col))
            L = tf.SparseTensor(indices, L.data, L.shape)
            return tf.sparse_reorder(L)
        
        def gconv(x,supports,output_size,bias_start=0.0,name='gconv'):
            batch_size = x.get_shape()[0].value
            num_nodes = x.get_shape()[1].value
            input_size = x.get_shape()[2].value
            dtype = x.dtype

            x0 = tf.transpose(x, perm=[1, 2, 0])
            x0 = tf.reshape(x0, shape=[num_nodes, input_size * batch_size])
            x = tf.expand_dims(x0, axis=0)

            with tf.variable_scope(name) as scope:
                if args['max_diffusion_step'] == 0:
                    pass
                else:
                    for support in supports:
                        x1 = tf.sparse_tensor_dense_matmul(support, x0)
                        x = tf.concat([x, tf.expand_dims(x1, axis=0)], axis=0)

                        for k in range(2, args['max_diffusion_step'] + 1):
                            x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                            x = tf.concat([x, tf.expand_dims(x2, axis=0)], axis=0)
                            x1, x0 = x2, x1

                num_matrices = len(supports) * args['max_diffusion_step'] + 1  # Adds for x itself.
                x = tf.reshape(x, shape=[num_matrices, num_nodes, input_size, batch_size])
                x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
                x = tf.reshape(x, shape=[batch_size * num_nodes, input_size * num_matrices])

                weights = tf.get_variable(
                    'weights', [input_size * num_matrices, output_size], dtype=dtype,
                    initializer=tf.contrib.layers.xavier_initializer())
                x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

                biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                        initializer=tf.constant_initializer(bias_start, dtype=dtype))
                x = tf.nn.bias_add(x, biases)
                return tf.reshape(x, [batch_size, num_nodes, output_size])
        
        def dcrnn_cell(input,h,supports):
            r = tf.sigmoid( gconv(tf.concat([input,h],axis=-1),
                supports,32,bias_start=0,name='gate_r') )
            u = tf.sigmoid( gconv(tf.concat([input,h],axis=-1),
                supports,32,bias_start=0,name='gate_u') )
            c = tf.tanh( gconv(tf.concat([input,h*r],axis=-1),
                supports,32,bias_start=0,name='gate_c') )
            h_new = u*h + (1-u)*c
            return c,h_new
        
        def u_law_decoder(x,bits=5):
            x = tf.cast(x,tf.float32)
            x = (tf.exp((x+0.5)/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)
            return tf.expand_dims(x,-1)
        
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.input_set = []
        self.pred_set = []
        self.loss_set = []
        self.loss_rmse_set = []

        for i in range(args['city_number']):
            city_name = args['city_name_'+str(i)]
            self.input_set.append(tf.placeholder(tf.float32, shape=(args['num_nodes_'+str(i)], args['seq_len'], args['input_dim_'+str(i)]), name='input'+str(i)))
            input_TNC = tf.transpose(self.input_set[-1], perm=[1, 0, 2])

            adj_matrix = np.load(args['adj_matrix_'+str(i)])
            with tf.variable_scope('DCRNN'+str(i)) as scope: 
                L = calculate_scaled_laplacian(adj_matrix,lambda_max=None)
                supports = [_build_sparse_matrix(L)]
                h = tf.zeros([1,args['num_nodes_'+str(i)],32])
                w = tf.get_variable(
                    '1_by_1_skip', [1, 32, 32],
                    initializer=tf.contrib.layers.xavier_initializer())
                output = []
                for time_step in range(args['seq_len']-1):
                    c,h = dcrnn_cell(input_TNC[time_step:time_step+1],h,supports)
                    output.append(c)
                    if time_step == 0:
                        scope.reuse_variables()
                pred_TNC = tf.concat(output,0)
                pred_NTC = tf.transpose(pred_TNC, perm=[1, 0, 2])
                pred = tf.nn.conv1d(pred_NTC, w, 1, 'SAME')
                pred_prob = tf.nn.softmax(pred)
                pred = tf.nn.conv1d(pred_prob, u_law_decoder(tf.constant(np.arange(2**5).reshape(1,2**5))), 1, 'SAME')
                y = self.input_set[-1][:,1:,0:1]
                loss = tf.reduce_mean(tf.abs(pred-y))
                loss_rmse = tf.reduce_mean(tf.square(pred-y))
                self.pred_set.append(pred)
                self.loss_set.append(loss)
                self.loss_rmse_set.append(loss_rmse)
        
        # Compute loss
        self.loss_mid = tf.constant(0)
        self.loss = tf.add_n(self.loss_set)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gvs)
        pass


class LSTM(object):
    def __init__(self, args):
        self.default_model(args)
        
    
    def default_model(self,args):
        def Dense(x,output_size,bias_start,name):
            with tf.variable_scope(name) as scope: 
                batch_size = x.get_shape()[0].value
                num_nodes = x.get_shape()[1].value
                input_size = x.get_shape()[2].value
                dtype = x.dtype
                x = tf.reshape(x, shape=[batch_size*num_nodes,input_size])
                weights = tf.get_variable(
                        'weights', [input_size, output_size], dtype=dtype,
                        initializer=tf.contrib.layers.xavier_initializer())
                x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

                biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                            initializer=tf.constant_initializer(bias_start, dtype=dtype))
                x = tf.nn.bias_add(x, biases)
                return tf.reshape(x, [batch_size, num_nodes, output_size])
        
        def rnn_cell(input,h):
            r = tf.sigmoid( Dense(tf.concat([input,h],axis=-1),
                32,bias_start=0,name='gate_r') )
            u = tf.sigmoid( Dense(tf.concat([input,h],axis=-1),
                32,bias_start=0,name='gate_u') )
            c = tf.tanh( Dense(tf.concat([input,h*r],axis=-1),
                32,bias_start=0,name='gate_c') )
            h_new = u*h + (1-u)*c
            return c,h_new
        
        def u_law_decoder(x,bits=5):
            x = tf.cast(x,tf.float32)
            x = (tf.exp((x+0.5)/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)
            return tf.expand_dims(x,-1)
        
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.input_set = []
        self.pred_set = []
        self.loss_set = []
        self.loss_rmse_set = []

        for i in range(args['city_number']):
            city_name = args['city_name_'+str(i)]
            self.input_set.append(tf.placeholder(tf.float32, shape=(args['num_nodes_'+str(i)], args['seq_len'], args['input_dim_'+str(i)]), name='input'+str(i)))
            input_TNC = tf.transpose(self.input_set[-1], perm=[1, 0, 2])

            with tf.variable_scope('RNN'+str(i)) as scope: 
                h = tf.zeros([1,args['num_nodes_'+str(i)],32])
                w = tf.get_variable(
                    '1_by_1_skip', [1, 32, 32],
                    initializer=tf.contrib.layers.xavier_initializer())
                output = []
                for time_step in range(args['seq_len']-1):
                    c,h = rnn_cell(input_TNC[time_step:time_step+1],h)
                    output.append(c)
                    if time_step == 0:
                        scope.reuse_variables()
                pred_TNC = tf.concat(output,0)
                pred_NTC = tf.transpose(pred_TNC, perm=[1, 0, 2])
                pred = tf.nn.conv1d(pred_NTC, w, 1, 'SAME')
                pred_prob = tf.nn.softmax(pred)
                pred = tf.nn.conv1d(pred_prob, u_law_decoder(tf.constant(np.arange(2**5).reshape(1,2**5))), 1, 'SAME')
                y = self.input_set[-1][:,1:,0:1]
                loss = tf.reduce_mean(tf.abs(pred-y))
                loss_rmse = tf.reduce_mean(tf.square(pred-y))
                self.pred_set.append(pred)
                self.loss_set.append(loss)
                self.loss_rmse_set.append(loss_rmse)
        
        # Compute loss
        self.loss_mid = tf.constant(0)
        self.loss = tf.add_n(self.loss_set)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gvs)
        pass

if __name__=="__main__":
    pass