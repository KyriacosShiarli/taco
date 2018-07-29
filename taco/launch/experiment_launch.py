import argparse
from taco.src.behavioral_cloning import BC
from taco.src.alignment_control import AC
import taco.src.utils as U
import os
import pickle
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('name', type=str,
                    help="Name for the experiment")

domains = ["nav","dial","dial_visual","craft"]

parser.add_argument('domain', type=str,
                    help="Domain the dataset is associated with",choices = domains)

methods = ["ctc","taco","bc"]
parser.add_argument('algorithm', type=str,
                    help="CTC, TACO, BC",choices =methods)


parser.add_argument('dataset', type=str,
                    help="Dataset to be used for training. Provide the name of the dataset withing the "
                         "folder denoted by the selected domain")

parser.add_argument('-c','--cfg', type=str,default = 'default',
                    help='config name')

parser.add_argument('-n', '--nb_datapoints', default=-1, type=int,
                    help="number of datapoints to use. -1 is all. ")

parser.add_argument('-a', '--architecture', default='mlp', type=str,
                    help="Architecture to be used")

args = parser.parse_args()
if not os.path.exists('../results/'):
    os.makedirs('../results/')
data_dir = os.path.join(args.dataset)
data = pickle.load(open('../data/'+args.domain+"/"+data_dir,'rb'),encoding='bytes')
res_dir = os.path.join('../results/',args.name)
exp_name = args.domain+"_"+args.algorithm+"_"+args.cfg+"_"+str(args.nb_datapoints)+"_"+args.architecture
exp_dir,mod_dir = U.experiment_init(exp_name,res_dir)

if args.cfg == 'default':
    cfg = args.algorithm+"_"+args.domain+"_base.yaml"
else:
    cfg = args.cfg if args.cfg[-5:] == '.yaml' else args.cfg + '.yaml'

params = yaml.load(open('../cfg/'+args.domain+"/"+cfg,'r+'))
params['nb_datapoints'] = args.nb_datapoints if args.nb_datapoints !=-1 else len(data['states'])

if args.algorithm == 'ctc':
    params['model_dir'] = mod_dir
    params['results_dir'] = exp_dir
    params['architecture'] = args.architecture
    if args.architecture =='mlp':
        params['units'] = [params['ctc-il']['units'][i] + 100 for i in range(len(params['ctc-il']['units']))]
    runner = AC

if args.algorithm ==('taco' or 'tac'):
    params['architecture'] = args.architecture
    params['model_dir'] = mod_dir
    params['results_dir'] = exp_dir
    runner = AC

if args.algorithm == 'bc':
    params['model_dir'] = mod_dir
    runner = BC

runner(data,**params)


