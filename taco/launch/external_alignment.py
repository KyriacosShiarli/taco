import pickle
from taco.src import utils as U
from taco.src import models as M
from taco.src import models_common as mc
import yaml

import os
PLOT_NUM = 0
def external_accuracy(data,**kwargs):


    nb_datapoints = kwargs['nb_datapoints'] if kwargs['nb_datapoints'] != -1 else len(data['states'])
    test_size = kwargs['test_size'] if 'test_size' in kwargs else 100
    model_dir = kwargs['model_dir']

    path  =os.path.join(model_dir,'model/')
    print("Loading policy")
    policy = mc.load_policy(path)
    loaded_params = yaml.load(open(path+'/params.yaml'))
    image = loaded_params['image']
    discrete = loaded_params['discrete']


    print("Parameters:")
    for key in kwargs.keys():
        print(key, ':', kwargs[key])
    # these are variable length


    X,A,Y,onsets,unique = U.preprocess(data,range(nb_datapoints),image)
    test_idx = range(nb_datapoints,nb_datapoints+test_size)
    X_test, A_test, Y_test, onsets_test, unique_test = U.preprocess(data, test_idx, image)

    nb_subtasks = unique.shape[0]
    in_dim = X[0][0].shape[0]

    out_dim = A[0][0].shape[0]


    print("Building Algorithm")

    model = M.TACO(in_dim, out_dim, policy=policy)
    model.sess = policy.sess
    s_mu, s_std, a_mu, a_std = U.get_normalisers(X, A, discrete,True, image)

    test_accuracy, test_mu_acc, test_decoded = U.get_accuracy(onsets_test, model,
                                X_test, A_test, Y_test, [s_mu, s_std], [a_mu, a_std], full=True)
    print(test_mu_acc)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('experiment_dir', type=str,
                        help="Name for the experiment")

    parser.add_argument('domain', type=str,
                        help="Name for the domain")


    parser.add_argument('dataset', type=str,
                        help="Name for the dataset within the domain")

    parser.add_argument('--num_train', default=200, type=int,
                        help="Number of demonstrations for train accuracy")


    parser.add_argument('--num_test', default=100, type=int,
                        help="Number demonstrations for test accuracy")

    args = parser.parse_args()


    experiment_name = 'jgsaw/'
    res_dir  = '../results/tac/'
    exp_dir =  res_dir+experiment_name
    mod_dir = exp_dir+'model/'


    params = {
        'nb_datapoints':args.num_train,
        'test_size':args.num_test,
        'model_dir' : '../results/'+args.experiment_dir +'/',
        'visual': True if args.domain == 'dial_visual' else False
    }

    if args.domain == 'dial':
        base_data_dir = '../data/dial/'
    elif args.domain == 'dial_visual':
        base_data_dir = '../data/dial_visual/'
    elif args.domain == 'nav':
        base_data_dir = '../data/nav/'
    elif args.domain == 'craft':
        base_data_dir = '../data/craft/'
    data = pickle.load(open(base_data_dir+args.dataset, 'rb'),encoding='bytes')
    external_accuracy(data,**params)



