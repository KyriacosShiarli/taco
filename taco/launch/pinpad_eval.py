import argparse
import os
import pickle
import yaml
from taco.src.jacopinpad_eval import run_eval
import pandas as pd



def get_details(name):
    method = size = None
    if name.find('ctc') != -1 and name.find('bi') != -1:
        method = "CTC-BC (GRU)"
    elif name.find('ctc') != -1 and name.find('ml') != -1:
        method = "CTC-BC (MLP)"
    elif name.find('bc') != -1:
        method = "GT-BC"
    elif name.find('tac') != -1:
        method = "TACO"
    if name.find('100') != -1:
        size = 100
    elif name.find('300') != -1:
        size = 300
    elif name.find('500') != -1 or name.find('-1') != -1:
        size = 500
    return method,size


# This script will launch mujoco experiments for models within a directory. For each model in the directory it will run
# a jacopinpad_eval session that will then store an evaluation yaml with the conditions of the experiments.
# the percentage task accuracy and the the percentage of completed subtasks.

# Arguements: Number of evaluation sketches. Length of sketch. Directory where all the models are stored.


parser = argparse.ArgumentParser()

parser.add_argument('experiment_dir', type=str,
                    help="Name for the experiment")

parser.add_argument('--render', action='store_true',default = False,
                    help='render the evaluation')
parser.add_argument('-n', '--num_trials', default=100, type=int,
                    help="Number of trials to evaluate on")

parser.add_argument('-l', '--sketch_len',nargs='+', default=5, type=int,
                    help="Number of trials to evaluate on")
args = parser.parse_args()
total_evaluation = {}
repetitions = 5
#df = pd.DataFrame(columns = ['Method', 'Datasize', 'Accuracy', 'Accuracy(Subtask)', 'Length'])
df = pd.read_pickle(os.path.join(args.experiment_dir, 'pinpad_evaluation.p'))
print(df)
names = [name for name in os.listdir(args.experiment_dir) if os.path.isdir(args.experiment_dir+name) ]
models_dir = [args.experiment_dir + name + '/model/' for name in names]
for length in args.sketch_len:
    print('LENGTH',length)
    for i in range(len(names)):
        print("EXPERIMENT",names[i])
        method,size = get_details(names[i])
        eval_dir = args.experiment_dir+names[i]+'/evaluation.yaml'
        if os.path.exists(eval_dir):
            evaluation = yaml.load(open(eval_dir,'rb'))
        else:
            evaluation = {}
        for _ in range(repetitions):

            result = run_eval(args.num_trials,models_dir[i],'targ_def_easier',length)
            df = df.append({"Method":method,"Datasize":size,"Accuracy":result['task_accuracy'],
                       "Accuracy(Subtask)":result['subtask_accuracy'],'Length':length},ignore_index=True)

            print(df)

        result.pop('detail')
        result.pop('summary')
        result.pop('total')
        evaluation[str(args.num_trials)+'_'+str(length)] = result
        yaml.dump(evaluation,open(eval_dir,'w+'))
        total_evaluation[names[i]] = evaluation
        df.to_pickle(os.path.join(args.experiment_dir, 'pinpad_evaluation.p'), protocol=2)
    yaml.dump(total_evaluation, open(os.path.join(args.experiment_dir,'evaluation.yaml'),'w+'))

#this result is a detail of % tasks completed. %subtasks completed. And detal