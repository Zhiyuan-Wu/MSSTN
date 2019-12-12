import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse import linalg
class MSSTN(object):
    def __init__(self, args):
        # model
        self.default_model(args)

    def default_model(self,args):
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.input_set = []
        self.pred_set = []
        self.loss_set = []
        self.loss_mse_set = []
        

        # Set up region-scale gragh
        self.input_mid = tf.placeholder(tf.float32, shape=(args['num_nodes_mid'], args['seq_len'], args['input_dim_mid']), name='input_mid')
        input_TNC_mid = tf.transpose(self.input_mid, perm=[1, 0, 2])
        if not args['Independent_opt']:
            spatial_feature_TNC_mid = self.ChebNet(input_TNC_mid,np.load(args['adj_matrix_mid']),'gconv_mid',args)
            spatial_feature_mid = tf.transpose(spatial_feature_TNC_mid, perm=[1, 0, 2])
            #self.pred_mid, self.loss_mid = self.DCN(tf.concat([self.input_mid,spatial_feature_mid],axis=2), 'DCN_mid', args)
        else:
            spatial_feature_TNC_mid_set = [self.ChebNet(input_TNC_mid,np.load(args['adj_matrix_mid']),'gconv_mid_'+str(i),args) for i in range(args['city_number'])] 
            spatial_feature_mid_set = [tf.transpose(spatial_feature_TNC_mid_set[i], perm=[1, 0, 2]) for i in range(args['city_number'])]
            #self.pred_mid, self.loss_mid = self.DCN(tf.concat([self.input_mid,spatial_feature_mid],axis=2), 'DCN_mid', args)
        self.pred_mid, self.loss_mid, self.loss_mse_mid = self.DCN(self.input_mid, 'DCN_mid', args)

        # Set up other city-scale gragh
        for i in range(args['city_number']):
            city_name = args['city_name_'+str(i)]
            self.input_set.append(tf.placeholder(tf.float32, shape=(args['num_nodes_'+str(i)], args['seq_len'], args['input_dim_'+str(i)]), name='input'+str(i)))
            input_TNC = tf.transpose(self.input_set[-1], perm=[1, 0, 2])
            spatial_feature_TNC = self.ChebNet(input_TNC,np.load(args['adj_matrix_'+str(i)]),'gconv'+str(i),args)
            spatial_feature = tf.transpose(spatial_feature_TNC, perm=[1, 0, 2])
            if not args['Independent_opt']:
                spatial_feature_mid_for_here = tf.stack([spatial_feature_mid[args['city_index_'+str(i)]] for _ in range(args['num_nodes_'+str(i)])],0)
            else:
                spatial_feature_mid_for_here = tf.stack([spatial_feature_mid_set[i][args['city_index_'+str(i)]] for _ in range(args['num_nodes_'+str(i)])],0)
            pred, loss, loss_mse = self.DCN(tf.concat([self.input_set[-1],spatial_feature,spatial_feature_mid_for_here],axis=2), 'DCN'+str(i), args)
            self.pred_set.append(pred)
            self.loss_set.append(loss)
            self.loss_mse_set.append(loss_mse)

        # Compute loss
        self.loss = self.loss_mid + tf.add_n(self.loss_set)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gvs)
        pass
    
    def model1(self,args):
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.input = tf.placeholder(tf.float32, shape=(args['num_nodes'], args['seq_len'], args['input_dim']), name='input')
        self.input2 = tf.placeholder(tf.float32, shape=(args['num_nodes2'], args['seq_len'], args['input_dim2']), name='input2')
        input_TNC = tf.transpose(self.input, perm=[1, 0, 2])
        spatial_feature_TNC = self.ChebNet(input_TNC,np.load(args['adj_matrix_path']),'gconv',args)
        spatial_feature = tf.transpose(spatial_feature_TNC, perm=[1, 0, 2])
        self.pred, self.loss1 = self.DCN(tf.concat([self.input,spatial_feature],axis=2), 'DCN', args)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss1)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
        self.train_op = optimizer.apply_gradients(clipped_gvs)
        pass
    
    def DCN(self, input, name, args):
        def CasualResUnit(x,res_channel=args['res_channel'],skip_channel=args['skip_channel'],name='CasualResUnit'):
            n = x.shape[-1].value
            w = tf.get_variable('kernel',[3, n, 2*res_channel],initializer=tf.contrib.layers.xavier_initializer())
            _x = tf.pad(x,[[0,0],[2,0],[0,0]])
            y = tf.nn.conv1d(_x, w, 1, 'VALID')
            h,g = tf.split(y, 2, axis=-1)
            h = tf.tanh(h)
            g = tf.sigmoid(g)
            h = h*g
            w = tf.get_variable('1_by_1_res',[1, res_channel, n],initializer=tf.contrib.layers.xavier_initializer())
            o = tf.nn.conv1d(h, w, 1, 'SAME')
            #o = tf.nn.relu(o)
            o = o+x
            w = tf.get_variable('1_by_1_skip',[1, res_channel, skip_channel],initializer=tf.contrib.layers.xavier_initializer())
            skip = tf.nn.conv1d(h, w, 1, 'SAME')
            return o,skip
        
        def DilatedConv(x,dilated=args['dilated'],res_channel=args['res_channel'],skip_channel=args['skip_channel'],name='DilatedConv'):
            n = x.shape[-1].value
            l = x.shape[1].value
            if l%dilated != 0:
                print('Dilated Error, when dilated at '+str(dilated))
                exit()
            num = l//dilated
            with tf.variable_scope(name) as scope:
                x = tf.reshape(x,[-1,num,dilated,n])
                _out = []
                _skip = []
                for i in range(dilated):
                    out, skip = CasualResUnit(x[:,:,i],res_channel=res_channel,skip_channel=skip_channel,name='CasualResUnit#'+str(i))
                    _out.append(out)
                    _skip.append(skip)
                    if i==0:
                        scope.reuse_variables()
                o = tf.stack(_out,axis=2)
                o = tf.reshape(o,[-1,l,n])
                skip = tf.stack(_skip,axis=2)
                skip = tf.reshape(skip,[-1,l,skip_channel])
                return o,skip
        
        def u_law_encoder(x,bits=5):
            bits = 2**bits
            x = tf.clip_by_value(x,0.0,0.99999)
            x = tf.floor(tf.log(1+bits*x)*bits/np.log(1+bits))
            x = tf.one_hot(tf.cast(tf.squeeze(x,-1),tf.int32),depth=bits)
            return x
        
        def u_law_decoder(x,bits=5):
            x = tf.cast(x,tf.float32)
            #x = ((tf.exp(x/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)+(tf.exp((x+1.0)/(2**bits)*np.log(1+(2**bits)))-1.0)/(2**bits))/2.0
            x = (tf.exp((x+0.5)/(2**bits)*np.log(1.0+(2**bits)))-1.0)/(2**bits)
            return tf.expand_dims(x,-1)
        
        def SingleChannelNetwork(x,channel,bits,encoder,decoder,name='Channel'):
            with tf.variable_scope(name) as scope:
                y = x[:,1:,channel:channel+1]
                #y = encoder(y,bits)
                x = x[:,:-1,:]
                _skip = [x]
                #w = tf.Variable(tf.random_normal([1, args['input_dim']+args['gconv_channels'], args['res_channel']], stddev=0.1),name='1_by_1_x')
                w = tf.get_variable(
                    '1_by_1_x', [1, x.shape[-1].value, args['res_channel']],
                    initializer=tf.contrib.layers.xavier_initializer())
                x = tf.nn.conv1d(x, w, 1, 'SAME')
                for i in range(args['DilatedConvLayers']):
                    if i==0:
                        o,skip = DilatedConv(x,dilated=3**i,name='DilatedConv'+str(i))
                        _skip.append(skip)
                    else:
                        o,skip = DilatedConv(o,dilated=3**i,name='DilatedConv'+str(i))
                        _skip.append(skip)
                skip = tf.concat(_skip,axis=-1)
                skip = tf.nn.relu(skip)
                #skip = tf.nn.dropout(skip,keep_prob_output)
                #w = tf.Variable(tf.random_normal([1, skip.shape[-1].value, args['n_hidden']], stddev=0.1),name='1_by_1_skip1')
                w = tf.get_variable(
                    '1_by_1_skip1', [1, skip.shape[-1].value, args['n_hidden']],
                    initializer=tf.contrib.layers.xavier_initializer())
                skip = tf.nn.conv1d(skip, w, 1, 'SAME')
                skip = tf.nn.relu(skip)
                #skip = tf.nn.dropout(skip,keep_prob_output)
                #w = tf.Variable(tf.random_normal([1, args['n_hidden'], 2**bits], stddev=0.1),name='1_by_1_skip2')
                w = tf.get_variable(
                    '1_by_1_skip2', [1, args['n_hidden'], 2**bits],
                    initializer=tf.contrib.layers.xavier_initializer())
                pred = tf.nn.conv1d(skip, w, 1, 'SAME')
                pred_prob = tf.nn.softmax(pred)
                #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
                pred = tf.nn.conv1d(pred_prob, decoder(tf.constant(np.arange(2**bits).reshape(1,2**bits))), 1, 'SAME')
                loss = tf.reduce_mean(tf.abs(pred-y))
                loss_mse = tf.reduce_mean(tf.square(pred-y))
                return pred,loss,loss_mse
        
        pred,loss,loss_mse = SingleChannelNetwork(input,0,5,u_law_encoder,u_law_decoder,name=name)
        return pred,loss,loss_mse

    def ChebNet(self,input,adj_matrix,name,args):
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
        
        with tf.variable_scope(name) as scope: 
            L = calculate_scaled_laplacian(adj_matrix,lambda_max=None)
            supports = [_build_sparse_matrix(L)]
            for i in range(args['gconv_layers']):
                input = gconv(input,supports,args['gconv_channels'],bias_start=0,name='gconv'+str(i))
                input = tf.nn.sigmoid(input)
            return input

    def output_projection(self,x,name='output_layer'):
        pass

    def loss_func(self,x):
        pass
        




if __name__=="__main__":
    import yaml
    with open('config.yaml') as f:
        config = yaml.load(f)
    model = ReiWa(config)
    #param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    _debug = np.array([2,3,3])
