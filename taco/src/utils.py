import numpy as np
import tensorflow as tf
import os

def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def get_normalisers(X,A,discrete,normalise_actions,image):
    if normalise_actions ==False:
        a_mu, a_std = (0, 1)
    else:
        a_mu, a_std = asym_stats(A)
        a_std = asym_max(A)
        zer = np.where(a_std < 0.000001)[0]
        a_std[zer] = 1

    if discrete:
        s_mu,s_std = (0,1)
        a_mu, a_std = (0,1)
    elif image:
        s_mu, s_std = (0, 255.)
    else:
        s_mu, s_std = asym_stats(X)
        s_std =asym_max(X)
    return s_mu,s_std,a_mu,a_std

def remove_sequential_duplicates(list):
    out =[]
    for n,e in enumerate(list):
        if n ==0 or e!=list[n-1]:
            out.append(e)
    return out

def preprocess(data,idx,image):

    print("Sorting out data")
    X = [data['states'][i] for i in idx]
    A = [data['actions'][i] for i in idx]

    if image!=False:
        images = [data['images'][i] for i in idx]
        _, wid, hei, chan = np.shape(images[0])
        X = [0] * len(images)
        for i in range(len(images)):
            X[i] = [images[i][j].reshape((wid*hei*chan)) for j in range(len(images[i]))] # flatten images and scale [0,1]

        del images
    # mild supervision
    all_labels = []
    tasks = [data['tasks'][i] for i in idx]
    ons = [data['gt_onsets'][i] for i in idx]
    for i in idx:
        all_labels.extend(data['tasks'][i])

    unique = np.sort(np.unique(all_labels))
    # re cast substask names. Y is the mild supervisio per trajectory.
    Y = []
    onsets = []
    for en, task in enumerate(tasks):
        su_tasks = []
        os = []
        for en2, subtask in enumerate(task):
            su_tasks.append(np.where(subtask == unique)[0][0])
        for en2, subtask in enumerate(ons[en]):
            os.append(np.where(subtask == unique)[0][0])
        Y.append(su_tasks)
        onsets.append(os)

    return X,A,Y,onsets,unique


def list_slice(lst,num_slices):
    # slices the list into num_slices lists. returns a list of lists.
    # good for variable length sequences.
    out= []
    lengths = len(lst)/int(num_slices)
    mod = len(lst)%int(num_slices)
    for i in range(num_slices):
        tmp = lst[i*lengths:i*lengths+lengths] if i<num_slices-1 else lst[-(lengths+mod):]
        out.append(tmp)
    return out


def gpu_config(use):
    if use!=False:
        gpu_opt = tf.ConfigProto(allow_soft_placement=True)
        gpu_opt.gpu_options.allow_growth = True
        #gpu_opt.gpu_options.per_process_gpu_memory_fraction = 0.8
        os.environ['CUDA_VISIBLE_DEVICES'] = str(use)
    else:
        gpu_opt = tf.ConfigProto(allow_soft_placement=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return gpu_opt



def experiment_init(name,base_dir):
    #experiment_name = name + '/' if name[-1]!='/' else name
    res_dir = base_dir + '/' if name[-1]!='/' else base_dir
    exp_dir = os.path.join(base_dir,name)
    mod_dir = os.path.join(exp_dir , 'model/')
    mkdir(res_dir), mkdir(exp_dir), mkdir(mod_dir)
    return exp_dir,mod_dir


def asym_stats(list_o_lists):
    all = []
    for n, traj in enumerate(list_o_lists):
        all.extend(traj)

    return np.mean(all,axis=0),np.std(all,axis=0)

def asym_max(list_o_lists):
    all = []
    for n, traj in enumerate(list_o_lists):
        all.extend(traj)

    return np.amax(np.abs(all),axis=0)

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

def get_accuracy(ground_truth,model,X,A,Y,norm_x,norm_a,full = False):
    if full:
        perm = range(len(X))
    else:
        nb = int(len(X)*0.5) if int(len(X)*0.1)>1 else 1
        perm = np.random.permutation(len(X))[:nb]
    batch_size = 2
    nb_iter = int(len(perm)/batch_size if len(perm)%batch_size==0 else len(perm)/batch_size+1)
    accuracy = []
    decoded = []
    for k in range(nb_iter):
        fr = k*batch_size
        to = k*batch_size + batch_size if k < nb_iter-1 else len(perm)
        idx = perm[fr:to]
        gt = [ground_truth[i] for i in idx]
        s_mu,s_std = norm_x
        a_mu, a_std = norm_a
        X_padded, sl = pad_sequences([X[i] for i in idx])
        X_padded = (X_padded - s_mu) / s_std

        A_padded, sl = pad_sequences([A[i] for i in idx])

        A_padded = (A_padded - a_mu) / a_std
        A_padded = A_padded[:,:,:]
        Y_padded, tl = pad_sequences([Y[i] for i in idx])
        local_dec = model.decode(X_padded, A_padded, Y_padded, sl, tl)
        #local_dec = batch_size * seq_length
        for n, (dec, l) in enumerate(zip(local_dec, sl)):
            accuracy.append(np.sum(dec[:l] == gt[n], dtype=np.float32) / l)
            decoded.append(list(dec[:l]))

        #decoded = local_dec if k==0 else np.vstack((decoded,local_dec))


    return accuracy,np.mean(accuracy),np.array(decoded)


