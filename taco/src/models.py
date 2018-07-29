import tensorflow as tf
import numpy as np
import taco.src.tac_loss as tc

import taco.src.ngctc_loss as ngctc
import taco.src.utils as U
from taco.src.models_common import ModularPolicy,recurrent_stack,GRU,initializer
from taco.src.image_models import image_models
dense = tf.layers.dense
reg = tf.nn.l2_loss


EPS = 4e-1
SIGMA_INIT =0.8

def switch(val,thresh):
    return tf.maximum(val,thresh)


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

class TACO(object):
    def __init__(self,in_dim, out_dim,policy = None,entropy_reg = 0):

        self.policy = policy


        self.lr = lr = tf.placeholder_with_default(0.0001,None,name = 'learning_rate')

        self.x =x = policy.x
        self.a = a = tf.placeholder(tf.float32, [None, None, out_dim], name='actions')  # data input actions
        self.entropy_reg = float(entropy_reg)

        bs, max_seq_len = tf.shape(x)[0], tf.shape(x)[1]
        a = tf.reshape(self.a, (bs * max_seq_len, out_dim))


        self.seq_len_ph = self.policy.seq_len
        mlps = self.policy.mlps
        # finish the graph.
        actions = tf.convert_to_tensor([tf.reshape(action, (bs, max_seq_len, out_dim)) for action in self.policy.actions])
        self.actions = tf.transpose(actions, (1, 2, 0, 3))

        action_probs = [dist.prob(tf.cast(a,tf.float64)) for dist in self.policy.action_dists]

        action_probs = [tf.where(tf.is_nan(ac_prob),tf.zeros_like(ac_prob),ac_prob) for ac_prob in action_probs]

        action_probs = tf.convert_to_tensor([tf.reshape(action_prob, (bs, max_seq_len)) for action_prob in action_probs])
        self.action_probs = action_probs = tf.transpose(action_probs, (1, 2, 0))
        stop_probs = tf.convert_to_tensor([tf.reshape(tf.cast(stop_prob,tf.float64), (bs, max_seq_len, 2)) for stop_prob in self.policy.stop_probs])
        self.stop_probs = stop_probs = tf.transpose(stop_probs, (1, 2, 0, 3))  # batch, time, label_size,stop,non_stop

        self.target_seq = tf.placeholder(tf.int32, [None, None])
        self.targ_len_ph = tf.placeholder(tf.int32, [None])

        # now the loss TAC loss function.
        with tf.device('/cpu:0'):
            self.loss = loss = tc.tac_loss(action_probs, stop_probs, self.target_seq, self.seq_len_ph, self.targ_len_ph)

            self.decoded = tc.tac_decode(action_probs, stop_probs, self.target_seq, self.seq_len_ph, self.targ_len_ph)
        opt = tf.train.AdamOptimizer(lr)
        vars = tf.trainable_variables()
        grads = tf.gradients(loss, vars)
        self.train_op = opt.apply_gradients(zip(grads, vars))
        summ = []
        summ.append(tf.summary.scalar('Loss', self.loss))
        #summ.append(tf.summary.histogram('Logits', logits_sum))
        #summ.append(tf.summary.histogram('Gradients', grads))
        #summ.append(tf.summary.histogram('Action probs', self.action_probs))
        #[summ.append(tf.summary.histogram(v.name, v)) for v in tf.trainable_variables()]
        #summ.append(tf.summary.histogram('GT actions', self.a))
        #summ.append(tf.summary.histogram('Actions', self.actions))
        #summ.append(tf.summary.histogram('Stop probs', self.stop_probs[:,:,:,0]))
        # if architecture=='recurrent' or architecture=='bi-recurrent':
        #     summ.append(tf.summary.histogram('RNN out', x))
        self.summary_op = tf.summary.merge(summ)


    def initialise(self,sess=None):
        self.sess = sess if sess is not None else tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def backward(self, X, A, Y, seq_len, targ_len,lr,dropout=1.):
        fd = {self.x: X,
              self.a: A,
              self.target_seq: Y,
              self.seq_len_ph: seq_len,
              self.targ_len_ph: targ_len,
              self.lr :lr,
              self.policy.keep_prob : dropout}
        _, loss,summary = self.sess.run([self.train_op, self.loss,self.summary_op], fd)
        return loss,summary

    def get_action_probs(self, X, A, seq_len):
        fd = {self.x: X,
              self.a: A,
              self.seq_len_ph: seq_len}
        action_probs = self.sess.run(self.action_probs, fd)
        return action_probs

    def debug_pass(self, X, A, Y, seq_len, targ_len):
        fd = {self.x: X,
              self.a: A,
              self.target_seq: Y,
              self.seq_len_ph: seq_len,
              self.targ_len_ph: targ_len}
        msq = self.sess.run([self.actions,self.stop_probs], fd)
        return msq

    def forward(self, X,seq_len = None):
        if self.recurrent:
            return self.sess.run([self.actions, self.stop_probs], {self.x: X,self.seq_len_ph:seq_len})
        else:
            return self.sess.run([self.actions, self.stop_probs], {self.x : X})

    def decode(self, X, A, Y, seq_len, targ_len):
        fd = {self.x: X,
              self.a: A,
              self.target_seq: Y,
              self.seq_len_ph: seq_len,
              self.targ_len_ph: targ_len}
        decoded = self.sess.run(self.decoded, fd)
        for n,dec in enumerate(decoded[1:]):
            decoded[n+1,:] = np.array([Y[n][d] for d in dec])
        decoded = np.delete(decoded,0,axis=0)
        return np.array(decoded)


class BCModel(object):
    def __init__(self, nb_subtasks, in_dim, out_dim, units=[100, 20], architecture='mlp', l2_reg=0.0001,
                 mlp_complex=True, discrete=False, image=False):
        self.lr = lr = tf.placeholder_with_default(0.0001, None, name='learning_rate')

        self.x = x = tf.placeholder(tf.float32, [None, None, in_dim], name='input')  # data input is in sequence
        self.a = a = tf.placeholder(tf.float32, [None, None, out_dim], name='actions')  # data input actions
        self.stop_targets = tf.placeholder(tf.int32, [None,None])
        self.architecture = architecture
        self.recurrent = True if architecture == 'recurrent' or architecture == 'bi-recurrent' else False
        self.mlp_units = units if mlp_complex is True or self.recurrent is False else []

        bs, max_seq_len = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(self.x, (bs * max_seq_len, in_dim))
        a = tf.reshape(self.a, (bs * max_seq_len, out_dim))
        stop = tf.reshape(self.stop_targets, (bs * max_seq_len,1))

        self.policy = ModularPolicy(nb_subtasks, in_dim, out_dim, x=self.x, units=units, architecture=architecture,
                                    discrete=discrete, mlp_complex=mlp_complex, image=image)

        self.seq_len_ph = self.policy.seq_len
        seq_mask = tf.sequence_mask(self.seq_len_ph,dtype=tf.float32)
        seq_mask = tf.reshape(seq_mask,(bs*max_seq_len,1))


        if discrete:
            seq_mask_action = tf.sequence_mask(self.seq_len_ph - 1, maxlen=max_seq_len, dtype=tf.float32)
            seq_mask_action = tf.reshape(seq_mask_action, (bs * max_seq_len, 1))

            self.losses_action = [tf.losses.softmax_cross_entropy(a, action, weights=tf.squeeze(seq_mask_action)) for action in
                                  self.policy.actions]
        else:
            seq_mask_action = tf.sequence_mask(self.seq_len_ph, maxlen=max_seq_len, dtype=tf.float32)
            seq_mask_action = tf.reshape(seq_mask_action, (bs * max_seq_len, 1))

            self.losses_action = [tf.losses.mean_squared_error(a,action,weights=seq_mask_action)
                                  for n,action in enumerate(self.policy.actions)]


        self.losses_stop =  [tf.losses.softmax_cross_entropy(tf.squeeze(tf.one_hot(stop, 2),1),mlp.logits,
                                weights=tf.squeeze(seq_mask)) for mlp in self.policy.mlps]

        self.losses_total = [tf.to_float(loss_ac) + tf.to_float(loss_stop) + l2_reg*(mlp.l2_stp + 0.01*mlp.l2_ac)
                             for loss_ac,loss_stop,mlp in zip(self.losses_action,self.losses_stop,self.policy.mlps)]

        vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=str(i)) + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv') for i in range(nb_subtasks)]

        opt = tf.train.AdamOptimizer(lr)
	    #opt = tf.train.RMSPropOptimizer(lr)

        self.grads = [tf.gradients(loss,var) for loss,var in zip(self.losses_total,vars)]
        self.train_ops = [opt.apply_gradients(zip(grad, var)) for grad,var in zip(self.grads,vars)]

        # so then we simply call the train op that the sequence we are given is related to.
        # The dataset is simply Sequences for each policy instead of separate datapoints.
        # Assume that the temporal correlation wont be very important if batches are large
        summ = []
        summ.append([tf.summary.scalar('Loss action_'+str(i),self.losses_action[i]) for i in range(nb_subtasks)])
        summ.append([tf.summary.scalar('Reg_action'+str(i),self.policy.mlps[i].l2_ac) for i in range(nb_subtasks)])
        summ.append([tf.summary.scalar('Reg_stop' + str(i), self.policy.mlps[i].l2_stp) for i in range(nb_subtasks)])

        #summ.append(tf.summary.histogram('input',x))
        [summ.append(tf.summary.histogram(v.name, v)) for v in tf.trainable_variables()]
 #      [summ.append(tf.summary.histogram(var.op.name, values=grad)) for grad, var in grads]
        [summ.append(tf.summary.histogram(g.name, g)) for g in self.grads[0]]
        #summ.append(tf.summary.histogram('GT actions', self.a))
        #summ.append([tf.summary.histogram('Actions', self.policy.actions[i]) for i in range(nb_subtasks)])
        self.summary_op = tf.summary.merge(summ)

    def initialize(self,sess=None):
        self.sess = sess if sess is not None else tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def backward(self, X, A,A_stop, seq_len, module,lr,compute_summary = True,dropout = 1.):
        fd = {self.x: X,
              self.a: A,
              self.stop_targets:A_stop,
              self.seq_len_ph: seq_len,
              self.lr :lr,
              self.policy.keep_prob : dropout}
        if compute_summary:
            _, loss_ac,loss_stop,summary = self.sess.run([self.train_ops[module], self.losses_action[module],self.losses_stop[module],self.summary_op], fd)
            return loss_ac,loss_stop,summary
        else:
            _, loss_ac, loss_stop = self.sess.run(
                [self.train_ops[module], self.losses_action[module], self.losses_stop[module]], fd)
            return loss_ac, loss_stop

class CTC(object):
    def __init__(self, nb_subtasks, state_dim, action_dim, units=[100, 20], architecture='bi-recurrent',gaps= True,clip_val = 0.5, image=False):
        self.lr = lr = tf.placeholder_with_default(0.0001,None,name = 'learning_rate')
        self.x = x = tf.placeholder(tf.float32, [None, None, state_dim], name='input')  # data input
        self.a = a = tf.placeholder(tf.float32, [None, None, action_dim], name='actions')  # data input actions
        self.targ_len_ph = tf.placeholder(tf.int32, [None])
        self.seq_len = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.concat_in = tf.concat((x,a),axis=2)
        self.units = units
        self.architecture = architecture
        self.recurrent = True if architecture=='recurrent' or architecture=='bi-recurrent' else False
        self.image = image
        self.gaps = gaps

        shape = tf.shape(self.concat_in)
        batch_s, max_timesteps = shape[0], shape[1]

        # 1d array of size [batch_size]
        bs, max_seq_len = tf.shape(self.concat_in)[0], tf.shape(self.concat_in)[1]
        if self.image:

            bs, max_seq_len = tf.shape(x)[0], tf.shape(x)[1]
            self.im_dim = (112, 112, 3)
            x = image_models[self.image](x, bs, max_seq_len, self.im_dim)
            # output_size = 100
            # x = dense(x, output_size, tf.nn.relu, name='cnn_fcp', kernel_regularizer=reg)
            x = tf.reshape(x, (bs, max_seq_len, x.get_shape()[1]))
            self.concat_in = tf.concat((x,a), axis=2)

        if self.recurrent:

            bi_d = True if self.architecture =='bi-recurrent' else False

            if bi_d:
                x, state = recurrent_stack(self.concat_in, self.seq_len, init=[None, None], units=units, bidirectional=bi_d)
            else:
                x,state = recurrent_stack(self.concat_in,self.seq_len,units= units,bidirectional=bi_d)
        else:
            dim = self.concat_in.get_shape()[2] if image else state_dim+action_dim

            x = tf.reshape(self.concat_in,(bs*max_seq_len,dim))


            for n in range(len(units)):
                x = dense(x, units[n], tf.nn.relu, name='fcp' + str(n),
                          kernel_regularizer=reg)

                x = tf.nn.dropout(x, self.keep_prob)
        do = tf.nn.dropout(x,keep_prob=self.keep_prob)
        # Reshaping to apply the same weights over the timesteps
        nb_subtasks = nb_subtasks+1 if gaps else nb_subtasks
        self.logits = dense(do, nb_subtasks, name='fcp', kernel_regularizer=reg)
        # Reshaping back to the original shape
        self.logits = tf.reshape(self.logits, [batch_s, -1, nb_subtasks])
        # Time major
        self.logits = tf.transpose(self.logits, (1, 0, 2))
        self.out_probs = tf.cast(tf.transpose(tf.nn.softmax(self.logits),(1,0,2)),tf.float64)
        if not gaps:
            self.targets = tf.placeholder(tf.int32, [None, None])

            self.loss = loss = ngctc.ngctc_loss(self.out_probs, self.targets, self.seq_len,
                                            self.targ_len_ph)
            self.decoded = ngctc.ngctc_decode(self.out_probs,self.targets,self.seq_len,self.targ_len_ph)
        else:
            self.targets = tf.sparse_placeholder(tf.int32)
            self.loss = loss = tf.nn.ctc_loss(self.targets, self.logits, self.seq_len,ctc_merge_repeated=True,preprocess_collapse_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len)
        cost = tf.reduce_mean(loss)

        vars = tf.trainable_variables()

        #### this sections applies clipping but doesnt work with conv layers as mismatch in float type

        # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, vars), clip_val)
        # optimizer = tf.train.AdamOptimizer(lr)
        # optimizer = tf.train.RMSPropOptimizer(lr)
        # self.train_op = optimizer.apply_gradients(zip(grads, vars))

        ####

        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(cost)

    def initialise(self,sess=None):
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())
    def forward(self,X,A,seq_len,dropout = 1.):
        out = self.sess.run([self.out_probs],feed_dict = {self.x : X,self.a:A,self.seq_len : seq_len,self.keep_prob:dropout})

        return out

    def decode(self,X,A,Y,seq_len,targ_len):
        fd = {self.x: X,
              self.a: A,
              self.targets: Y,
              self.seq_len: seq_len,
              self.targ_len_ph: targ_len,self.keep_prob:1}
        decoded = self.sess.run(self.decoded, fd)
        for n, dec in enumerate(decoded[1:]):
            decoded[n + 1, :] = np.array([Y[n][d] for d in dec])
        decoded = np.delete(decoded, 0, axis=0)
        return np.array(decoded)

    def backward(self, X, A, Y, seq_len, targ_len, lr,dropout = 1):
        if self.gaps:
            Y = U.sparse_tuple_from(Y)

        fd =  {self.x : X,
               self.a : A,
               self.targets : Y,
               self.seq_len : seq_len,
               self.lr:lr,
               self.keep_prob:dropout,
               self.targ_len_ph:targ_len}

        loss,_ = self.sess.run([self.loss,self.train_op],feed_dict =fd)
        return loss,None

    def get_rnn_out(self,inputs,seq_len):
        out = self.sess.run([self.rnn_out],feed_dict = {self.x : inputs,self.seq_len : seq_len})

        return out

    def train(self,inputs,targets):
        pass



if __name__=='__main__':
    pass
