import numpy as np
import pickle
from taco.src import utils as U
from taco.src import models as M
from taco.src import models_common as Mc
from taco.src import behavioral_cloning as bc
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

PLOT_NUM = 0

# main training script for taco.


def AC(data,**params):
    nb_datapoints = params['nb_datapoints'] if params['nb_datapoints']!=-1 else len(data['states'])
    test_size = params['test_size'] if 'test_size' in params else 0
    print("Parameters:")
    for key in params.keys():
        print(key, ':', params[key])

    # GPU configuration
    if 'gpu' in params and params['gpu'] !=False:
        gpu_opt = U.gpu_config(params['gpu'])
    else:
        gpu_opt = U.gpu_config(False)


    # Preprocessing
    X,A,Y,onsets,unique = U.preprocess(data,range(nb_datapoints),params['image'])
    if test_size >0:
        test_idx = range(nb_datapoints,nb_datapoints+test_size)
        X_test, A_test, Y_test, onsets_test, unique_test = U.preprocess(data, test_idx, params['image'])
    nb_subtasks = unique.shape[0]

    # normalisation values
    s_mu, s_std, a_mu, a_std = U.get_normalisers(X, A, params['discrete'], params['normalise_actions'], params['image'])

    #GRAPH DEFINITION
    in_dim = X[0][0].shape[0]
    out_dim = A[0][0].shape[0]
    print("Building the Graph")
    if params['algorithm'] == ('tac' or 'taco'):
        policy = Mc.ModularPolicy(nb_subtasks,in_dim,out_dim,units = params['units'],architecture=params['architecture'],
                                    discrete=params['discrete'],mlp_complex = params['mlp_complex'], image=params['image'])

        algorithm = M.TACO(in_dim, out_dim,policy,entropy_reg=params['entropy_reg'])

    elif params['algorithm'] == 'ctc':
        algorithm = M.CTC(nb_subtasks, in_dim, out_dim, units=params['units'], architecture=params['architecture'],
                      gaps=params['gaps'], image=params['image'])

    sess = tf.Session(config=gpu_opt)
    algorithm.initialise(sess=sess)
    writer = tf.summary.FileWriter(params['results_dir'], algorithm.sess.graph)

    batch_per_epoch = int(len(X) /params['batch_size'] if len(X)%float(params['batch_size']) ==0
                          else (len(X)/params['batch_size']) + 1)



    do = params['dropout']
    # TRAINING
    print("Training...")
    all_loss = []
    all_accuracy = []
    for j in range(params['epochs']):
        batch_loss =[]
        idx = np.random.permutation(range(len(X)))
        for i in range(batch_per_epoch):
            fr = i*params['batch_size'] ; to = (i+1)*params['batch_size']
            # Batch padding and normalisation
            X_batch = [X[k] for k in idx[fr:to]]
            X_batch,seq_len = U.pad_sequences(X_batch)
            X_batch = (X_batch-s_mu)/s_std
            A_batch = [A[k] for k in idx[fr:to]]
            A_batch, seq_len = U.pad_sequences(A_batch)
            A_batch = (A_batch - a_mu) / a_std if params['normalise_actions'] else A_batch
            A_batch =  A_batch +  np.random.normal(0, params['action_noise'],
                                            np.shape(A_batch)) if not params['discrete'] else A_batch
            A_batch = A_batch[:,:,:]
            Y_batch = [Y[k] for k in idx[fr:to]]
            Y_batch,targ_len = U.pad_sequences(Y_batch)
            #backward_pass
            loss,summary = algorithm.backward(X_batch,A_batch,Y_batch,seq_len,targ_len,params['lr'],dropout=do)
            batch_loss.extend([loss])
        do*=params['dropout_decay']


        if j%5 == 0:
            acc,mu_acc,dec = U.get_accuracy(onsets,algorithm,X,A,Y,[s_mu,s_std],[a_mu,a_std],full=False)
            all_accuracy.append(mu_acc)
            print("Sampled Accuracy:",mu_acc)

        if summary is not None:
            writer.add_summary(summary, j)
            writer.flush()

        all_loss.append(np.mean(batch_loss))
        print("EPOCH",j)
        print("LOSS", all_loss[-1])

    accuracy,mu_acc,decoded = U.get_accuracy(onsets,algorithm,X,A,Y,[s_mu,s_std],[a_mu,a_std],full=True)
    all_accuracy.append(mu_acc)

    # Bookeeping and evaluations
    if test_size>0:
        test_accuracy, test_mu_acc, test_decoded = U.get_accuracy(onsets_test, algorithm,
                                            X_test, A_test, Y_test, [s_mu, s_std], [a_mu, a_std], full=True)
    else:
        test_mu_acc = 0

    sl = [len(x) for x in X]

    if params['model_dir'] is not None:
        saver = tf.train.Saver()
        saver.save(algorithm.sess, params['model_dir']+'model.chpt')
        other_data = {'losses': [all_loss],'accuracy':accuracy,'mean_accuracy':list(all_accuracy),'test_accuracy':float(test_mu_acc)}

        yaml.dump(other_data, open(params['model_dir'] + 'stats.yaml', 'w'))
        yaml.dump(params, open(params['model_dir'] + 'params.yaml', 'w'))
        yaml.dump(unique, open(params['model_dir'] + 'actor_map.yaml', 'w'))
        if params['algorithm'] != 'ctc':
            params_inf={}
            params_inf['nb_subtasks'] = nb_subtasks
            params_inf['in_dim'] = in_dim
            params_inf['out_dim'] = out_dim
            params_inf['s_mu'] = (s_mu,s_std)
            params_inf['a_mu'] = (a_mu, a_std)
            params_inf['units'] = params['units']
            params_inf['architecture'] = params['architecture']
            params_inf['discrete'] = params['discrete']
            params_inf['image'] = params['image']
            yaml.dump(params_inf,open(params['model_dir']+'params_inference.yaml','w'))

    if params['plot']:
        for j in range(0,PLOT_NUM):
            f,ax = plt.subplots()
            idx_vis = j
            colours = ['r','g','m','k']
            lgd = []
            gt = map(float, onsets[idx_vis])
            lgd.extend(ax.plot(decoded[j,:sl[idx_vis]],c='r',alpha=0.6, label='Model'))
            lgd.extend(ax.plot(np.array(gt),c = 'b',alpha = 0.6,label = 'Ground Truth'))
            plt.legend(handles = lgd)
            f.savefig(params['results_dir']+str(j)+'.png')
    tf.reset_default_graph()
    algorithm.sess.close()

    del algorithm.sess
    # If using CTC behavioral cloning happens here
    if params['algorithm'] == 'ctc':
        segmentation = decoded
        params['ctc-il']['discrete'] = params['discrete']
        params['ctc-il']['model_dir'] = params['model_dir']
        params['ctc-il']['gpu'] = params['gpu'] if "gpu" in params else False
        params['ctc-il']['image'] = params['image']

        data['states'] = X
        data['actions'] = A
        bc.behavioral_cloning(data,nb_subtasks,segmentation,**params['ctc-il'])

    return np.mean(accuracy),decoded
if __name__=="__main__":
    import os
    def mkdir(dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)

    experiment_name = 'jgsaw/'
    res_dir  = '../results/tac/'
    exp_dir =  res_dir+experiment_name
    mod_dir = exp_dir+'model/'


    params = {
        'epochs':500,
        'lr': 0.005,
        'batch_size': 10,
        'nb_datapoints' : 100,
        'test_size':5,
        'model_dir' : mod_dir,
        'results_dir':res_dir,
        'plot':False,
        'units': [100],
        'architecture': 'mlp',
        'algorithm': 'tac',
        'mlp_complex': True,
        'gpu': False,
        'image': False,
        'entropy_reg': 0.0,
        'discrete': False,
        'action_noise':0.0, # 0.1 for Jigsaws
        'normalise_actions':True,
        'dropout':0.7,
        'dropout_decay':1.001,
        'gaps': False,
        'ctc-il':{ # parameters for the imitation network when things are not end-to-end. Only used for CTC-IL methods
            'batch_size':10,
            'dropout':1.,
            'epochs':2,
            'model_dir':mod_dir,
            'units':[100],
            'lr':0.01,
            'architecture': 'recurrent'
        }
    }


    mkdir(res_dir),mkdir(exp_dir),mkdir(mod_dir)

    data = pickle.load(open('../data/nav/dataset_04.p', 'rb'),encoding='bytes')
    AC(data,**params)
