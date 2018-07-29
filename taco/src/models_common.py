import tensorflow as tf
# from tensorflow.contrib.keras.python.keras.layers import Conv2D
# from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
import numpy as np
dense = tf.layers.dense
reg = tf.nn.l2_loss
import taco.src.utils as U
from taco.src.image_models import image_models


EPS = 4e-1
SIGMA_INIT =0.8


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float64)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def sigma_initialiser(val):
    def _initialzer(shape,dtype=None,partition_info=None):
        out = val*np.ones(*shape).astype(np.float64)
        return tf.constant(out)
    return _initialzer


GRU = tf.contrib.rnn.GRUCell
initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)


class HMLP(object):
    def __init__(self,x, nb_outputs, units=[100, 20], scope="", residual=False,keep_prob = None, discrete=False):
        # this is equivalent to the MLP regressor.
        # so this is basically only a function.
        x_alt =x_st= x
        keep_prob = tf.placeholder_with_default(tf.to_float(1.), shape=[],
                                                name='keep_prob') if keep_prob is None else keep_prob

        self.l2_ac = [];self.l2_stp = []
        with tf.variable_scope(scope):

            for n in range(len(units)):
                x = dense(x, units[n], tf.nn.relu, name='fcp' + str(n),
                          kernel_regularizer=reg)

                self.l2_ac.append(tf.get_collection('regularization_losses',scope+'/fcp' + str(n)))

                x_st = dense(x_st, units[n], tf.nn.relu, name='fcp_s' + str(n),
                          kernel_regularizer=reg)

                self.l2_stp.append(tf.get_collection('regularization_losses', scope +'/fcp_s' + str(n)))

                x_st = tf.nn.dropout(x_st, keep_prob)

                x = tf.nn.dropout(x, 1.)

            if residual:
                x = tf.concat((x, x_alt), axis=1)


            self.l2_ac = tf.reduce_sum(self.l2_ac)
            self.l2_stp = tf.reduce_sum(self.l2_stp)

            self.logits=logits = dense(x_st, 2, name='logits')#kernel_initializer=normalized_columns_initializer(1.0)

            probs = tf.nn.softmax(logits, name='terminate_prob')
            ep = 1e-20
            stop_entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs + ep), reduction_indices=1))
            stop_sample = tf.multinomial(logits, 1)

            outputs = dense(x, nb_outputs, name='outputs')
            outputs = tf.cast(outputs, tf.float64)
            if not discrete:

                action_dist =tf.contrib.distributions.MultivariateNormalDiag(outputs,[tf.constant(0.6,dtype=tf.float64)]*nb_outputs)
            else:

                action_dist =  tf.contrib.distributions.OneHotCategorical(logits=outputs)
        self.stop_prob = probs
        self.stop_sample = stop_sample
        self.stop_entropy = stop_entropy
        self.action_out = outputs
        self.action_dist = action_dist


def recurrent_stack(x,seq_len,init=None, units=[100, 20], bidirectional = False):
    if not bidirectional:
        cells = [GRU(hidden) for hidden in units]
        stack_fw = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
        rnn_out, state = tf.nn.dynamic_rnn(stack_fw, x, seq_len, dtype=tf.float32,initial_state=init,
                                                         time_major=False)
        x = tf.reshape(rnn_out, [-1, units[-1]])


    else:
        stack_bw = tf.contrib.rnn.MultiRNNCell([GRU(hidden) for hidden in units],
                                               state_is_tuple=True)
        stack_fw = tf.contrib.rnn.MultiRNNCell([GRU(hidden) for hidden in units],
                                               state_is_tuple=True)

        rnn_out, state = tf.nn.bidirectional_dynamic_rnn(stack_fw, stack_bw, x, seq_len,
                                                                       dtype=tf.float32, time_major=False)
        outputs = [tf.reshape(out, [-1, units[-1]]) for out in rnn_out]

        x = tf.concat(outputs, 1)
    return x, state

class ModularPolicy(object):
    def __init__(self,nb_subtasks,in_dim, nb_outputs, x = None,
                 units=[100,20],architecture = 'mlp', load=False,discrete=False, mlp_complex = True, image=False,s_norm = (0,1),a_norm = (0,1),):

        self.s_norm = s_norm
        self.a_norm = a_norm
        self.units = units
        self.nb_subtasks = nb_subtasks
        if load:
            self.load()
        else:
            with tf.device('/device:GPU:0'):
                # TODO: Put a random recurrent policy, on nav world.
                # FIRST MAKE SURE THAT THIS CAN BE TRAINED FROM THE ctc
                #  POINT OF VIEW AND ALSO BE LOADED
                self.x = x = tf.placeholder(tf.float32, [None, None, in_dim], name='input') if x is None else x
                shape = tf.shape(x)
                bs = shape[0]
                self.seq_len = tf.placeholder(tf.int32, [None], name='sequence_length')
                self.keep_prob = tf.placeholder_with_default(tf.to_float(1.), shape=[], name='keep_prob')
                self.recurrent = True if architecture == 'recurrent' or architecture == 'bi-recurrent' else False
                self.image = image
                self.mlp_units = units if mlp_complex is True or self.recurrent is False else []

                if self.recurrent and not self.image:
                    zero_states = [tf.zeros((bs,u),dtype = tf.float32) for u in units]
                    self.state = self.last_state = tuple([tf.placeholder_with_default(zero_states[i],shape = [None,units[i]]) for i in range(len(units))])
                    self.gru_state = tuple([np.zeros((1, u)) for u in self.units])

                    x,self.last_state = recurrent_stack(x,self.seq_len,init = self.state,units=units)

                elif self.image:
                    bs, max_seq_len = tf.shape(x)[0], tf.shape(x)[1]
                    self.im_dim = (112, 112, 3)
                    with tf.variable_scope('conv'):
                        x = image_models[self.image](x, bs, max_seq_len, self.im_dim)
                        x = tf.nn.dropout(x,1.)
                else:
                    bs, max_seq_len = tf.shape(x)[0], tf.shape(x)[1]
                    x = tf.reshape(self.x, (bs * max_seq_len, in_dim))
                self.mlps = [HMLP(x, nb_outputs, units=self.mlp_units, scope=str(i),
                                  keep_prob=self.keep_prob, discrete=discrete) for i in range(nb_subtasks)]

                self.actions = [mlp.action_out for mlp in self.mlps]
                self.stop_samples = [mlp.stop_sample for mlp in self.mlps]
                self.stop_probs = [mlp.stop_prob for mlp in self.mlps]
                self.action_dists = [mlp.action_dist for mlp in self.mlps]
                self.entropy = [mlp.stop_entropy for mlp in self.mlps]
                self.save()

    def forward(self, input,module, dropout=1.):

        sess = self.sess
        if self.recurrent:
            action,state = sess.run([self.actions[module],self.last_state],
                              {self.x: input, self.keep_prob: dropout,self.seq_len :[1],self.state :self.gru_state})
            self.gru_state = state
        else:
            action = sess.run(self.actions[module],
                              {self.x: input, self.keep_prob: dropout})
        return action

    def zero_state(self):
        if self.recurrent:
            self.gru_state = tuple([np.zeros((1,u)) for u in self.units])
        else:
            pass

    def sample_stop(self, input,module,dropout = 1):
        sess = self.sess

        if self.recurrent:
            #zero_state = tuple([np.zeros((len
            stop,state = sess.run([self.stop_samples[module],self.last_state],
                            {self.x: input, self.keep_prob: dropout, self.seq_len: [1], self.state: self.gru_state})
            self.gru_state = state
        return np.squeeze(stop)



    def forward_full(self,input,module,dropout = 1):
        input = (np.array(input) - self.s_norm[0]) / self.s_norm[1]
        input = input

        sess = self.sess
        if self.recurrent:
            action,stop, state = sess.run([self.actions[module],self.stop_samples[module], self.last_state],
                                     {self.x: input, self.keep_prob: dropout, self.seq_len: [1],
                                      self.state: self.gru_state})
            self.gru_state = state
        else:
            action,stop= sess.run([self.actions[module],self.stop_samples[module]],
                                     {self.x: input, self.keep_prob: dropout})

        action  = action[0]*self.a_norm[1] + self.a_norm[0]
        return action,np.squeeze(stop)

    def load(self):
        self.actions = [tf.get_collection(str(i) + '/actions')[0] for i in range(self.nb_subtasks)]
        self.stop_probs = [tf.get_collection(str(i) + '/stop_probs')[0] for i in range(self.nb_subtasks)]
        self.stop_samples = [tf.get_collection(str(i) + '/stop_samples')[0] for i in range(self.nb_subtasks)]
        #self.action_dists = [tf.get_collection(str(i) + '/action_dists')[0] for i in range(self.nb_subtasks)]
        #self.outputs = tf.get_collection(self.scope + '/outputs')[0]
        self.x = tf.get_collection('input')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]

    def save(self):

        tf.add_to_collection('input', self.x)
        tf.add_to_collection('keep_prob', self.keep_prob)
        #tf.add_to_collection(self.scope + '/train_op_actions', self.train_op_actions)
        #tf.add_to_collection(self.scope + '/train_op_stop', self.train_op_stop)
        [tf.add_to_collection(str(i) + '/actions', self.actions[i]) for i in range(self.nb_subtasks)]
        [tf.add_to_collection(str(i) + '/stop_samples', self.stop_samples[i]) for i in range(self.nb_subtasks)]
        [tf.add_to_collection(str(i) + '/stop_probs', self.stop_probs[i]) for i in range(self.nb_subtasks)]
        #[tf.add_to_collection(str(i) + '/action_dists', self.action_dists[i]) for i in range(self.nb_subtasks)]


def load_policy(model_dir):

    # for now I need. the model directory to load.
    # load the parameters that you used to train also.
    #model_dir = '../../../c2tc/results/tac/jaco2/model/'
    gpu_opt = U.gpu_config(False)
    import yaml
    import os
    params = yaml.load(open(os.path.join(model_dir,'params_inference.yaml'),'r+'))
    policy = ModularPolicy(params['nb_subtasks'],params['in_dim'],params['out_dim']
                           ,units = params['units'],architecture = params['architecture'],load=False,mlp_complex = True,
                           discrete = params['discrete'],s_norm = params['s_mu'],a_norm = params['a_mu'],image=params['image'])
    saver = tf.train.Saver()
    sess = tf.Session(config=gpu_opt)
    policy.sess = sess
    saver.restore(sess, os.path.join(model_dir,'model.chpt')) # restore the variables.
    return policy
