import numpy as np
import tensorflow as tf
from taco.src.models import BCModel
import taco.src.utils as U
import pickle
import yaml

import os

def seg_of_seg(segmentation):
    # segmentation of segmentation.
    #
    # output is a list of datapoint index. sub_task_index, and sequence indeces
    out = []
    for n,i in enumerate(segmentation):
        # this is each datapoint
        data_idx = n
        s_t = []
        idxs = []
        for n2, j in enumerate(i):
            if n2 == 0:
                start_idx = 0
                s_t.append(j)
            elif j!=i[n2-1]:
                s_t.append(j)
                end_idx = n2
                idxs.append([start_idx,end_idx])
                start_idx = n2
            if n2 ==len(i)-1:
                idxs.append([start_idx,len(i)])

        out.append([s_t,idxs])

    ce()
    return out



def behavioral_cloning(data,nb_models,segmentation,**kwargs):

    if 'gpu' in kwargs and kwargs['gpu'] != False:
        print(kwargs['gpu'])
        gpu_opt = U.gpu_config(kwargs['gpu'])
    else:
        gpu_opt = U.gpu_config(False)


    X,A = data['states'],data['actions']
    s_mu,s_std,a_mu,a_std  = U.get_normalisers(X,A,kwargs['discrete'], True, kwargs['image'])

    state_dim = X[0][0].shape[0]
    action_dim = A[0][0].shape[0]
    A_ac = [[] for _ in range(nb_models)]
    X_in = [[] for _ in range(nb_models)]
    A_stop = [[] for _ in range(nb_models)]
    # datasets are as follows create a dataset for each model.
    # each dataset has shape (nb_data,sequence,features)

    segmented = seg_of_seg(segmentation)

    # now we assign the datapoints to each dataset.
    for d,seq in enumerate(segmented):
        point = np.array(X[d])
        ac_point = np.array(A[d])
        for n,sub_pol in enumerate(seq[0]):
            start = seq[1][n][0]
            end = seq[1][n][1] if n == len(seq[0])-1 else seq[1][n][1]+1

            X_in[sub_pol].append(point[start:end])
            A_ac[sub_pol].append(ac_point[start:end])
            zer = np.zeros(len(A_ac[sub_pol][-1]))
            zer[-1]=1
            A_stop[sub_pol].append(zer)

    batch_prop = kwargs['batch_proportion'] if 'batch_proportion' in kwargs else 0.1
    dp_per_epoch = [int(len(X_in[i])*batch_prop) +1 for i in range(nb_models)]

    l2_reg = kwargs['l2_reg'] if 'l2_reg' in kwargs else 0.00001
    model = BCModel(nb_models,state_dim,action_dim,units = kwargs['units'],architecture=kwargs['architecture'],
                    discrete=kwargs['discrete'], image=kwargs['image'],l2_reg=l2_reg)
    sess = tf.Session(config=gpu_opt)
    model.sess = sess

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(kwargs['model_dir'], model.sess.graph)

    for i in range(kwargs['epochs']):
        print("EPOCH",i)

        summarize= True if i %100==0 else False
        for mod in range(nb_models):
            loss_mod = 0
            loss_stop_mod = 0
            for batch in range(int(1/batch_prop)):
                idx = np.random.randint(len(X_in[mod]),size = dp_per_epoch[mod])
                X_b = [X_in[mod][k] for k in idx]
                X_b,sl = U.pad_sequences(X_b)
                X_b = (X_b-s_mu)/s_std


                A_b = [A_ac[mod][k] for k in idx]
                A_b, sl = U.pad_sequences(A_b)
                A_b = (A_b-a_mu)/a_std
                

                As_b = [A_stop[mod][k] for k in idx]
                As_b, sl = U.pad_sequences(As_b)
                if summarize:
                    loss_ac,loss_stop,summary = model.backward(X_b,A_b,As_b,sl,mod,kwargs['lr'],compute_summary=summarize)
                else:
                    loss_ac, loss_stop= model.backward(X_b, A_b, As_b, sl, mod, kwargs['lr'],
                                                                                 compute_summary=summarize)
                loss_mod+=loss_ac
                loss_stop_mod += loss_stop
            if summary is not None and summarize:
                writer.add_summary(summary,i)
                writer.flush()
            if i%1==0:
                print('Model',mod)
                print('Loss',loss_mod*batch_prop)
                print('Loss stop', loss_stop_mod * batch_prop)

    if kwargs['model_dir'] is not None:
        saver = tf.train.Saver()
        saver.save(model.sess, kwargs['model_dir'] + 'model.chpt')
        params_inf = {}
        params_inf['s_mu'] = (s_mu, s_std)
        params_inf['a_mu'] = (a_mu, a_std)
        params_inf['units'] = kwargs['units']
        params_inf['nb_subtasks'] = nb_models
        params_inf['in_dim'] = state_dim
        params_inf['out_dim'] = action_dim
        params_inf['discrete'] = kwargs['discrete']
        params_inf['architecture'] = kwargs['architecture']
        params_inf['image'] = kwargs['image']
        yaml.dump(params_inf, open(kwargs['model_dir'] + 'params_inference.yaml', 'w'))
        #yaml.dump(unique, open(model_dir + 'actor_map.yaml', 'wb'))
    tf.reset_default_graph()
    model.sess.close()

    del model.sess


def BC(data,**kwargs):
    print("Parameters:")
    for key in kwargs.keys():
        print(key, ':', kwargs[key])
    # these are variable length
    print("Preprocessing")
    X = data['states'][:kwargs['nb_datapoints']]
    A = data['actions'][:kwargs['nb_datapoints']]
    image = kwargs['image']

    if image:
        print('doing BC images')
        images = data['images'][:kwargs['nb_datapoints']]
        _, wid, hei, chan = np.shape(images[0])
        X = [0] * len(images)
        for i in range(len(images)):
            X[i] = [images[i][j].reshape((wid*hei*chan)) for j in range(len(images[i]))] # flatten images
    # mild supervision
    all_labels = []
    for task in data['tasks'][:kwargs['nb_datapoints']]:
        all_labels.extend(task)
    unique = np.sort(np.unique(all_labels))
    nb_subtasks = unique.shape[0]
    # re cast substask names. Y is the mild supervision per trajectory.
    Y = []
    onsets = []
    for en, task in enumerate(data['tasks'][:kwargs['nb_datapoints']]):
        su_tasks = []
        os = []
        for en2, subtask in enumerate(task):
            su_tasks.append(np.where(subtask == unique)[0][0])
        for en2, subtask in enumerate(data['gt_onsets'][en]):
            os.append(np.where(subtask == unique)[0][0])
        Y.append(su_tasks)
        onsets.append(os)
    sl = [len(x) for x in X]
    if kwargs['model_dir'] is not None:
        yaml.dump(kwargs, open(kwargs['model_dir'] + 'params.yaml', 'w'))
        yaml.dump(unique, open(kwargs['model_dir'] + 'actor_map.yaml', 'w'))

    segmentation = onsets
    data['states'] = X
    data['actions'] = A
    behavioral_cloning(data,nb_subtasks,segmentation,**kwargs)

if __name__=="__main__":
    exp_dir,mod_dir = U.experiment_init('bc_dart_full','../results/nav')
    params = {'units':[300],
              'nb_datapoints': 50,
              'architecture':'mlp',
              'batch_proportion': '0.01',
              'image': False,
               'discrete':False,
              'model_dir':mod_dir,
              'epochs':100,
              'lr':0.005,
              'l2_reg':0.0001}
    data = pickle.load(open('../data/nav/dataset_04.p', 'rb'),encoding='bytes')
    BC(data,**params)
