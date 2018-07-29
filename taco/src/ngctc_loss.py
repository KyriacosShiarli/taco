import tensorflow as tf
import numpy as np

EPS = 1e-20
def forward_ngctc(pol_probs,T,inference = False):

    # pol probs are policy stop probs. which is similar to the stop actions but throgh a global controller.
    # in this case the stop action can be seen as the probability for the next policy in the sequence
    T  = tf.to_int32(T)
    # act_probs = Tx|l|, stop_probs = Tx |l|*2
    lab_len = tf.shape(pol_probs)[1]
    alpha = tf.zeros([lab_len],dtype=tf.float64,)
    alpha = tf.concat([[pol_probs[0,0]],alpha[1:]],0)
    eye = tf.eye(lab_len,dtype=tf.float64)
    eye2 = tf.eye(lab_len-1, dtype=tf.float64)
    C = tf.reduce_sum(alpha)
    alpha = alpha/C


    def scan_op(inp,i):
        curr_alpha, curr_C = inp
        diag = curr_alpha*eye
        diag2 = curr_alpha[:-1] * eye2
        non_stops = tf.matmul(diag,tf.expand_dims(pol_probs[i,:],1))[:,0]

        stops = tf.matmul(diag2,tf.expand_dims(pol_probs[i,1:],1))[:,0]
        cut_stop = non_stops[1:] + stops


        #print tf.shape(cut_stop)
        all_stops = tf.concat([[non_stops[0]],cut_stop],0)

        def time_mask_full(): return tf.ones([lab_len],dtype=tf.float64)

        def time_mask_partial(): return tf.concat((tf.zeros([lab_len-T+i],dtype=tf.float64),tf.ones([T-i],dtype=tf.float64)),0)


        time_mask = tf.cond(i<T-lab_len+1,time_mask_full,time_mask_partial)
        new_alpha = all_stops*time_mask
        return new_alpha/tf.reduce_sum(new_alpha),tf.reduce_sum(new_alpha)+EPS

    irange = tf.range(1,T,dtype=tf.int32) # this only goes until T. So it will ignore longer sequences.
    alphas,Cs = tf.scan(scan_op,irange,initializer=(alpha,C))
    if not inference:
        return tf.concat(([C],Cs),0) # this returs a vector
    else:
        return tf.concat(([alpha],alphas),0)


def ngctc_loss(term_probs, targets,seq_len,tar_len):
    bs = tf.to_int32(tf.shape(term_probs)[0])
    #loss = 0.
    cond = lambda j,loss: tf.less(j, bs)
    j = tf.constant(0,dtype=tf.int32)
    loss = tf.constant(0,dtype=tf.float64)
    def body(j,loss):
        idx = tf.expand_dims(targets[j,:tar_len[j]],1)
        st = tf.transpose(term_probs[j], (1, 0))
        st = tf.transpose(tf.gather_nd(st, idx), (1, 0))
        length = seq_len[j]
        loss += -tf.reduce_sum(tf.log(forward_ngctc(st, length))/tf.to_double(bs)) # negative log likelihood for whole batch
        return tf.add(j,1),loss # average loss over batches

    out = tf.while_loop(cond,body,loop_vars= [j,loss])

    return out[1]


def ngctc_decode(term_probs, targets,seq_len,tar_len):
    max_seq_len  = tf.to_int32(tf.reduce_max(seq_len))
    bs = tf.to_int32(tf.shape(term_probs)[0])
    #loss = 0.
    cond = lambda j,loss: tf.less(j, bs)
    j = tf.constant(0,dtype=tf.int32)
    decoded = tf.zeros([1,max_seq_len],dtype=tf.int32)
    def body(j,decoded):
        idx = tf.expand_dims(targets[j,:tar_len[j]],1)
        st = tf.transpose(term_probs[j], (1, 0))
        st = tf.transpose(tf.gather_nd(st, idx), (1, 0))
        length = tf.to_int32(seq_len[j])
        alphas = forward_ngctc(st, length,inference=True) # get essentially the probability of being at each node
        dec = tf.to_int32(tf.argmax(alphas,axis=1)) # decode that by taking the argmax for each column of alphas
        dec = tf.concat([dec,tf.zeros([max_seq_len-length],dtype=tf.int32)],axis=0)

        decoded = tf.concat([decoded,[dec]],axis=0)

        return tf.add(j,1),decoded

    out = tf.while_loop(cond,body,loop_vars= [j,decoded],shape_invariants=[tf.TensorShape(None),tf.TensorShape([None, None])])
    return out[1]



if __name__=="__main__":
    def softmax(x,axis = 0):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.expand_dims(np.max(x,axis=axis),axis=axis))
        return e_x / np.expand_dims(e_x.sum(axis=axis),axis=axis)

    def test_loss():
        # create some dummy inputs
        D = 20 # dictionary length
        bs = 100
        T = 500 # all lengths 20
        variable_lengths = np.random.randint(100,390,size = bs)
        targ_lengths = np.random.randint(5, 15, size=bs,dtype=np.int32)
        T_arr = tf.convert_to_tensor(variable_lengths)
        targ_len = tf.convert_to_tensor(targ_lengths)


        np.random.uniform(0, 1, size=[bs, 20, D])

        p_stop = np.array(np.random.uniform(0, 1, size = [bs,T, D]),dtype=np.float32)
        p_stop = softmax(p_stop,axis=2)
        prob = tf.convert_to_tensor(p_stop)
        action_probs = tf.convert_to_tensor(np.array(np.random.uniform(0, 0.00001, size= [bs,T, D]),dtype=np.float32))


        targets = tf.convert_to_tensor(np.random.randint(0,D-1,size = [bs,np.amax(targ_lengths)]))



        with tf.Session() as sess:
            out = sess.run(ngctc_loss(prob,targets,T_arr,targ_len))
        ce()
    test_loss()







