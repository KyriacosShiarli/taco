import tensorflow as tf
import numpy as np

EPS = 0.000001

def forward_tac_log(act_probs,stop_probs,T,inference= False):
    T  = tf.to_int32(T)
    # act_probs = Tx|l|, stop_probs = Tx |l|*2
    lab_len = tf.shape(act_probs)[1]
    alpha = tf.zeros_like(act_probs[0,:],dtype=tf.float64)
    alpha = tf.concat([[tf.log(act_probs[0,0])],alpha[1:]],0)
    #eye = tf.eye(lab_len,dtype=tf.float64)
    stop_probs = tf.log(stop_probs)
    # so here no normalisation is needed in theory.

    def scan_op(curr_alpha,i):
        non_stops = tf.where(tf.equal(tf.zeros_like(curr_alpha),curr_alpha),-np.inf*tf.ones_like(curr_alpha),curr_alpha + stop_probs[i,:,0])
        stops = tf.where(tf.equal(tf.zeros_like(curr_alpha), curr_alpha), -np.inf*tf.ones_like(curr_alpha),
                             curr_alpha + stop_probs[i, :, 1])

        cut_stop = tf.reduce_logsumexp([non_stops[1:], stops[:-1]],reduction_indices=0)


        #print tf.shape(cut_stop)
        all_stops = tf.concat([[non_stops[0]],cut_stop],0)

        def time_mask_full(): return tf.ones([lab_len],dtype=tf.float64)

        def time_mask_partial(): return tf.concat((tf.zeros([lab_len-T+i],dtype=tf.float64),tf.ones([T-i],dtype=tf.float64)),0)


        time_mask = tf.cond(i<T-lab_len+1,time_mask_full,time_mask_partial)

        new_alpha = tf.where(tf.is_inf(all_stops), tf.zeros_like(all_stops),(all_stops + tf.log(act_probs[i, :])) * time_mask)

        return new_alpha

    irange = tf.range(1,T,dtype=tf.int32) # this only goes until T. So it will ignore longer sequences.
    alphas = tf.scan(scan_op,irange,initializer=(alpha))
    if not inference:
        return alphas[-1,-1] # this returns a vector
    else:
        return tf.concat(([alpha],alphas),0)



def forward_tac_tf(act_probs,stop_probs,T,inference= False):
    T  = tf.to_int32(T)
    # act_probs = Tx|l|, stop_probs = Tx |l|*2
    lab_len = tf.shape(act_probs)[1]
    alpha = tf.zeros_like(act_probs[0,:],dtype=tf.float64)
    alpha = tf.concat([[act_probs[0,0]],alpha[1:]],0)
    eye = tf.eye(lab_len,dtype=tf.float64)
    C = tf.reduce_sum(alpha)
    alpha = alpha/C


    def scan_op(inp,i):
        curr_alpha, curr_C = inp
        diag = curr_alpha*eye
        non_stops = tf.matmul(diag,tf.expand_dims(stop_probs[i,:,0],1))[:,0]

        stops = tf.matmul(diag,tf.expand_dims(stop_probs[i,:,1],1))[:,0]
        cut_stop = non_stops[1:] + stops[:-1]


        #print tf.shape(cut_stop)
        all_stops = tf.concat([[non_stops[0]],cut_stop],0)

        def time_mask_full(): return tf.ones([lab_len],dtype=tf.float64)

        def time_mask_partial(): return tf.concat((tf.zeros([lab_len-T+i],dtype=tf.float64),tf.ones([T-i],dtype=tf.float64)),0)

        time_mask = tf.cond(i<T-lab_len+1,time_mask_full,time_mask_partial)

        new_alpha = all_stops * act_probs[i, :]*time_mask #if not inference else all_stops*time_mask
        return new_alpha/tf.reduce_sum(new_alpha),tf.reduce_sum(new_alpha)

    irange = tf.range(1,T,dtype=tf.int32) # this only goes until T. So it will ignore longer sequences.
    alphas,Cs = tf.scan(scan_op,irange,initializer=(alpha,C))
    if not inference:
        return tf.concat(([C],Cs),0) # this returs a vector
    else:
        return tf.concat(([alpha],alphas),0)


def tac_decode(action_probs, term_probs, targets,seq_len,tar_len):
    # For now a non batch version.
    # T length of trajectory. D size of dictionary. l length of label. B batch_size
    # actions_prob_tensors.shape [B,max(seq_len),D]
    # stop_tensors.shape [B,max(seq_len),D,2] #
    # targets.shape [B,max(tar_len)] # zero padded label sequences.
    # seq_len the actual length of each sequence.
    # tar_len the actual length of each target sequence
    # because the loss was only implemented per example, the batch version is simply in a loop rather than a matrix.
    max_seq_len  = tf.to_int32(tf.reduce_max(seq_len))
    bs = tf.to_int32(tf.shape(action_probs)[0])
    #loss = 0.
    cond = lambda j,loss: tf.less(j, bs)
    j = tf.constant(0,dtype=tf.int32)
    decoded = tf.zeros([1,max_seq_len],dtype=tf.int32)
    def body(j,decoded):
        idx = tf.expand_dims(targets[j,:tar_len[j]],1)
        ac = tf.transpose(tf.gather_nd(tf.transpose(action_probs[j]), idx))
        st = tf.transpose(term_probs[j], (1, 0, 2))
        st = tf.transpose(tf.gather_nd(st, idx), (1, 0, 2))
        length = tf.to_int32(seq_len[j])
        alphas = forward_tac_tf(ac, st, length,inference=True) # get essentially the probability of being at each node
        dec = tf.to_int32(tf.argmax(alphas,axis=1)) # decode that by taking the argmax for each column of alphas
        dec = tf.concat([dec,tf.zeros([max_seq_len-length],dtype=tf.int32)],axis=0)

        decoded = tf.concat([decoded,[dec]],axis=0)

        return tf.add(j,1),decoded

    out = tf.while_loop(cond,body,loop_vars= [j,decoded],shape_invariants=[tf.TensorShape(None),tf.TensorShape([None, None])])

    return out[1]


def tac_loss(action_probs, term_probs, targets,seq_len,tar_len,safe = False):
    # For now a non batch version.
    # T length of trajectory. D size of dictionary. l length of label. B batch_size
    # actions_prob_tensors.shape [B,max(seq_len),D]
    # stop_tensors.shape [B,max(seq_len),D,2] #
    # targets.shape [B,max(tar_len)] # zero padded label sequences.
    # seq_len the actual length of each sequence.
    # tar_len the actual length of each target sequence
    # because the loss was only implemented per example, the batch version is simply in a loop rather than a matrix.
    bs = tf.to_int32(tf.shape(action_probs)[0])
    #loss = 0.
    cond = lambda j,loss: tf.less(j, bs)
    j = tf.constant(0,dtype=tf.int32)
    loss = tf.constant(0,dtype=tf.float64)
    def body(j,loss):
        idx = tf.expand_dims(targets[j,:tar_len[j]],1)
        ac = tf.transpose(tf.gather_nd(tf.transpose(action_probs[j]), idx))
        st = tf.transpose(term_probs[j], (1, 0, 2))
        st = tf.transpose(tf.gather_nd(st, idx), (1, 0, 2))
        length = seq_len[j]
        if safe:
            loss += -forward_tac_log(ac, st, length) / tf.to_double(bs)  # negative log likelihood
        else:
            loss += -tf.reduce_sum(tf.log(forward_tac_tf(ac, st, length))/tf.to_double(bs)) # negative log likelihood for whole batch
        return tf.add(j,1),loss # average loss over batches

    out = tf.while_loop(cond,body,loop_vars= [j,loss])

    return out[1]

if __name__=="__main__":

    def test_sum_log():
        x = tf.convert_to_tensor(-np.inf*(np.ones(2)))
        x_mask = tf.where(tf.is_inf(x), -np.inf*tf.ones_like(x), x)
        sum_log =tf.reduce_logsumexp(x)
        with tf.Session() as sess:
            mas = sess.run(x_mask)
            out = sess.run(sum_log)
        # print mas
        # print out


    def test_loss():
        # create some dummy inputs
        D = 20 # dictionary length
        bs = 100
        T = 1000  # all lengths 20
        variable_lengths = np.random.randint(T-10,T-1,size = bs,dtype = np.int32)
        targ_lengths = np.random.randint(3, 6, size=bs,dtype=np.int32)
        T_arr = tf.convert_to_tensor(variable_lengths)
        targ_len = tf.convert_to_tensor(targ_lengths)

        np.random.uniform(0, 1, size=[bs, 20, D])

        p_stop = np.array(np.random.uniform(0, 1, size = [bs,T, D]),dtype=np.float64)
        p_n_stop = 1 - p_stop
        prob = tf.convert_to_tensor(np.array(np.stack((p_stop, p_n_stop), axis=3),dtype=np.float64))
        action_probs = tf.convert_to_tensor(np.array(np.random.uniform(0, 1e-70, size= [bs,T, D]),dtype=np.float64))
        targets = tf.convert_to_tensor(np.random.randint(0,D-1,size = [bs,np.amax(targ_lengths)]))
        import time
        with tf.Session() as sess:
            tic = time.time()
            out1 = sess.run(tac_loss(action_probs,prob,targets,T_arr,targ_len,safe=True))
            #print "SAFE",time.time()-tic
            tic = time.time()
            out2 = sess.run(tac_loss(action_probs, prob, targets, T_arr, targ_len, safe=False))
            #print "Classic", time.time() - tic


    def test_decode():
        # create some dummy inputs
        D = 20 # dictionary length
        bs = 100
        T = 40 # all lengths 20
        variable_lengths = np.random.randint(10,39,size = bs)
        targ_lengths = np.random.randint(10, 20, size=bs,dtype=np.int32)
        T_arr = tf.convert_to_tensor(variable_lengths)
        targ_len = tf.convert_to_tensor(targ_lengths)

        np.random.uniform(0, 1, size=[bs, 20, D])

        p_stop = np.array(np.random.uniform(0, 1, size = [bs,T, D]),dtype=np.float64)
        p_n_stop = 1 - p_stop
        prob = tf.convert_to_tensor(np.array(np.stack((p_stop, p_n_stop), axis=3),dtype=np.float64))
        action_probs = tf.convert_to_tensor(np.array(np.random.uniform(0, 1, size= [bs,T, D]),dtype=np.float64))


        targets = tf.convert_to_tensor(np.random.randint(0,D-1,size = [bs,np.amax(targ_lengths)]))



        with tf.Session() as sess:
            out = sess.run(tac_decode(action_probs,prob,targets,T_arr,targ_len))
    test_loss()




