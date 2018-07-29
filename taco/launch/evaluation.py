import numpy as np
import pickle
import yaml
import numpy
from taco.src.nav_world import model_eval
import os
import pandas as pd
import argparse

def evaluate(exp_dir,domain, num_trials = 20,sketch_length = [5],visual = False,render = False):
    if domain=='dial' or domain=='dial_visual':
        from taco.src.jacopinpad_tools import env_init
        from taco.src.jacopinpad_eval import run_eval
    if domain =='dial':
        model, sim, control_scheme, viewer = env_init(render=render)
        env = {'model':model,'sim':sim,'control_scheme':control_scheme,'viewer':viewer}

    dirs = [os.path.join(exp_dir,name) for name in os.listdir(exp_dir)]
    names = [name for name in os.listdir(exp_dir)]
    print(names)
    df = pd.DataFrame(columns = ['Method','Datasize','Accuracy'])
    for n, d in enumerate(dirs):
        if os.path.isdir(d):
            if names[n].find('ctc') !=-1 and names[n].find('bi') !=-1:
                method = "CTC-BC (GRU)"
            elif names[n].find('ctc') != -1 and names[n].find('ml') != -1:
                method = "CTC-BC (MLP)"
            elif names[n].find('bc') != -1:
                method = "GT-BC"
            elif names[n].find('tac') != -1:
                method = "TACO"
            size = yaml.load(open(os.path.join(d,'model','params.yaml')))['nb_datapoints']
            print("Evaluating:",names[n])
            if domain =='nav':
                params = {'zero_shot':True}
                for trial in range(num_trials):
                    render = render if trial == 0 else False
                    score,accuracy = model_eval(os.path.join(d,'model'),10,animate=render,**params)
                df = df.append(pd.DataFrame([[method, size, accuracy]], columns=['Method', 'Datasize', 'Accuracy']),
                               ignore_index=True)
            elif domain == 'dial':
                for length in sketch_length:
                    print('LENGTH', length)
                    for _ in range(num_trials):
                        result = run_eval(args.num_trials, os.path.join(d,'model'), 'targ_def', length,visual,render=render,**env)
                        df = df.append({"Method": method, "Datasize": size, "Accuracy": result['task_accuracy'],
                                        "Accuracy(Subtask)": result['subtask_accuracy'], 'Length': length},
                                       ignore_index=True)

            # else:
            #     raise ValueError("Domain not defined")
    print(df)
    from taco.src.plotting import factorplot
    factorplot(["Datasize","Accuracy","Method"],df,os.path.join(exp_dir,domain+'evaluation_plot'))

    pd.to_pickle(df,exp_dir+'normal_shot.p')
    # Put a couple of plots here so that you can see whats going on.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment_dir', type=str,
                        help="Name for the experiment")

    parser.add_argument('domain', type=str,
                        help="Name for the experiment")


    parser.add_argument('--render', action='store_true', default=False,
                        help='render the evaluation')

    parser.add_argument('--visual', action='store_true', default=False,
                        help='render the evaluation')
    parser.add_argument('-n', '--num_trials', default=20, type=int,
                        help="Number of trials to evaluate on")

    parser.add_argument('-l', '--sketch_len', nargs='+', default=[5], type=int,
                        help="Number of trials to evaluate on")

    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    args = parser.parse_args()
    evaluate(args.experiment_dir,args.domain, args.num_trials, args.sketch_len,visual=args.visual,render = args.render)
