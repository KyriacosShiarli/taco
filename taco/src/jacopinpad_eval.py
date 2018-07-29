#!/usr/bin/env python3
"""
Example script for manual control of Jaco arm sim
"""
import os
import numpy as np
from taco.src.jacopinpad_tools import env_init,move_to_position,default_controller_init,get_joint_adresses,reset_state
import pickle
from taco.src.models_common import load_policy
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf

RENDER = False
SUBSAMP = 3
VISUAL = False

def run_eval(num_trials,model_dir,targ_def,sketch_length  = 5,visual = False, render = False,**kwargs):
    VISUAL = visual;
    RENDER = render
    if 'model' in kwargs:
        model, sim, control_scheme, viewer = kwargs['model'],kwargs['sim'],kwargs['control_scheme'],kwargs['viewer']
    else:
        model, sim, control_scheme, viewer = env_init(render=True)
    data_dir = '../data/dial/'+targ_def+'.p'
    if RENDER:
        viewer.render()
    # load the target defs as before.
    #
    img_dims = np.array([112, 112], dtype=np.int)
    capture_dims = np.array([202, 212], dtype=np.int)
    if VISUAL:
        frame = sim.render(capture_dims[0], capture_dims[1], camera_name="camera_main")
    joint_addr = get_joint_adresses(sim)
    controller = default_controller_init()
    policy = load_policy(model_dir)
    targ_def = pickle.load(open(data_dir, 'rb'))
    nb_points = len(targ_def['joint_angles'])-1
    init_pos = targ_def['joint_angles'][-1]


    global_score = {'detail':[],'summary':[],'total':[]}
    plt.ion()
    for i in range(num_trials):
        print('Reset to initial position.')
        reset_state(sim,init_pos)
        move_to_position(sim, controller, init_pos,render=False,viewer=viewer)
        sketch_idx = 0
        ctr = 0
        local_score = []
        targ_num =  2 * np.random.randint(0, int(nb_points / 2), 1)[0]
        sub_task = int(targ_num / 2)
        print ('SUB - TASK',sub_task)
        while True:
            target = targ_def['joint_angles'][targ_num+1]
            target_xyz = targ_def['xyz'][targ_num+1]
            control_scheme.execute_step()
            sim.step()
            ja = sim.get_state().qpos[joint_addr]
            dist = np.ravel(np.copy(sim.data.body_xpos[-1]) - np.array(targ_def['xyz'])[range(1, nb_points + 1, 2)])
            state = np.concatenate((ja,dist))

            if ctr%SUBSAMP==0:
                if VISUAL:
                    frame = sim.render(capture_dims[1], capture_dims[0], camera_name="camera_main")
                    state = frame

                    rem = np.array((capture_dims - img_dims) / 2, dtype=np.int)
                    state = state[rem[0]:-rem[0], 20:-80]
                    # if ctr%500==0:
                    #     frame = np.swapaxes(state, 0, 1)
                    #     plt.imshow(frame)
                    #     plt.pause(0.000001)
                    state = np.ravel(state) #/ 255.
                    factor = 1.
                else:
                    factor = 1.0

                    # viewer.render()
                targ_fin = np.zeros(9)
                targ_fin[-3:] = np.array([0.68,0.68,0.68])
                a_fingers = controller.act(ja, targ_fin)

                a,terminate = policy.forward_full([[state]], sub_task)
                if VISUAL:
                    a[-3:] = a_fingers[-3:]
                a*=factor
                a = np.min([a, controller.c_high], axis=0)
                a = np.max([a, np.array(controller.c_low)], axis=0)

                noise_var = np.abs(controller.c_low)*0.0


                #noise_var[0]+=np.random.uniform(0,0.15)
                #noise_var[1]+=np.random.uniform(0,0.15)
                dart_noise = np.random.uniform(-noise_var,noise_var, size=9)
                a+=dart_noise


            if RENDER and not VISUAL:
                viewer.render()

            sim.data.ctrl[:] = a
            ctr+=1
            d_xyz = target_xyz - np.copy(sim.data.body_xpos[-1])
            if np.sum(np.abs(d_xyz)) < 0.06 and terminate:
                sketch_idx +=1
                local_score.append(1)
                print(np.sum(np.abs(d_xyz)))
                print("-----------------")
                if sketch_idx == sketch_length:
                    break
                else:
                    while True:
                        targ_num_sug = 2 * np.random.randint(0, int(nb_points / 2), 1)
                        targ_num_sug = targ_num_sug[0]
                        if targ_num_sug != targ_num:
                            break
                    targ_num = targ_num_sug
                sub_task = int(targ_num / 2)
                print(sub_task)

                ctr=0

            elif ctr>15000 or terminate:
                if terminate:
                    print ("Terminated")
                else:
                    print ("TIMEOUT")

                sketch_idx +=1
                local_score.append(0)
                print(np.sum(np.abs(d_xyz)))
                print("-----------------")
                if sketch_idx == sketch_length:
                    break
                else:
                    while True:
                        targ_num_sug = 2 * np.random.randint(0, int(nb_points / 2), 1)
                        targ_num_sug = targ_num_sug[0]
                        if targ_num_sug != targ_num:
                            break
                    targ_num = targ_num_sug
                sub_task = int(targ_num / 2)
                print(sub_task)
                ctr = 0

        print ('LOCAL SCORE',local_score)
        global_score['detail'].append(local_score)
        global_score['summary'].append(1 if np.sum(local_score) ==len(local_score) else 0)
    global_score['total'] = float(np.sum(global_score['summary'])/float(len(global_score['summary'])))
    global_score['task_accuracy'] = float(np.sum(global_score['summary']) / float(len(global_score['summary'])))
    global_score['subtask_accuracy'] = float(np.sum(global_score['detail'], dtype=np.float32) /np.prod(
                                                      np.array(global_score['detail']).shape))

    policy.sess.close()
    tf.reset_default_graph()
    print ('GLOBAL_SCORE',global_score['total'])
    del policy.sess
    del policy
    viewer = None
    del viewer,sim,model
    return global_score



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir',type=str,
                        help="Directory for the model")

    parser.add_argument('dataset_name',type=str,
                        help="Name of dataset the model was trained on")
    parser.add_argument('--render','-r',default=False,action = 'store_true',
                        help="To render or not")
    parser.add_argument('-n','--num_trials', default=20,type=int,
                        help="Number of trials to evaluate on")
    parser.add_argument('-l','--sketch_len', default=5,type=int,
                        help="Number of trials to evaluate on")


    parser.add_argument('--visual','-v',default=False,action = 'store_true',
                        help="Use visual input for control")



    args = parser.parse_args()
    RENDER = args.render
    VISUAL = args.visual
    score = run_eval(args.num_trials,args.model_dir,args.dataset_name,sketch_length=args.sketch_len,
                     visual = args.visual,render = args.render)
    print(score)

    yaml.dump(score,open(os.path.join(args.model_dir,'evaluation.yaml'),'w'))
