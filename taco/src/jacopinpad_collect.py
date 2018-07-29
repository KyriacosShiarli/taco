#!/usr/bin/env python3
"""
Example script for manual control of Jaco arm sim
"""

import numpy as np
from jacopinpad_tools import env_init,PIcontroller,move_to_position,default_controller_init,get_joint_adresses,reset_state
import pickle
import utils as U
import matplotlib.pyplot as plt

U.mkdir('../data/jaco/')

RENDER = True
SUBSAMP = 20

t = 0



def define_targets(targets):
    model, sim, control_scheme, viewer = env_init()
    num_targets = np.size(targets, axis=0)
    joint_addr = get_joint_adresses(sim)
    controller = default_controller_init()
    targ_idx = 0
    xyz = [];
    rpy = [];
    joint_angles = []  # basically a datapoint starts here.
    ctr = 0
    while True:
        control_scheme.execute_step()
        sim.step()

        ja = sim.get_state().qpos[joint_addr]
        if RENDER:
            viewer.render()
        # ice()

        a =controller.act(ja, targets[targ_idx])
        sim.data.ctrl[:] = a
        # e = ja- targets[targ_idx]
        e = controller.accum_error[-1]
        ctr+=1

        if ctr > 15000:
            ctr=0
            targ_idx+=1
            if targ_idx >= num_targets:  # and os.getenv('TESTING') is not None:
                break


        if np.sum(np.abs(e))<0.02:
            print ('done')
            ctr=0
            xyz.append(np.copy(sim.data.body_xpos[-1])) # xyz of fingers
            rpy.append(sim.data.body_xpos[-1]) # roll_pitch_yaw 0f fingers (in quaternion)
            joint_angles.append(ja)
            targ_idx+=1
            if targ_idx >= num_targets:  # and os.getenv('TESTING') is not None:
                break
            # first collect like 25 of these The target point collection. Then assign 5 of these to each colour.
    targ_def = {'joint_angles':joint_angles,'xyz':xyz,'rpy':rpy}
    pickle.dump(targ_def,open('../data/jaco/targ_def_easier.p','wb'))


def get_permuted_indexes(nb_points):
    idx = np.random.permutation(nb_points)
    while True:
        # idx = np.random.randint(len(PERMUTATIONS))
        # return PERMUTATIONS[idx]
        if np.sum(idx == np.arange(nb_points)) == nb_points:
            idx = np.random.permutation(nb_points)
        else:
            return idx

def collect_data(num_sketches,len_sketch,dataset_name,img_collect=False,permute = False):
    model, sim, control_scheme, viewer = env_init(render = RENDER)
    #screenshot = ScreenshotTaker(sim = sim, camera_name = 'camera_one', screenshot_dir = '../data/jaco/images2')

    joint_addr = get_joint_adresses(sim)
    controller = default_controller_init()
    data_dir = '../data/jaco/'+dataset_name +'.p'
    dataset = {'states': [], 'actions': [], 'gt_onsets': [], 'tasks': [], 'images': []}
    targ_def = pickle.load(open('../data/dial/targ_def.p', 'rb'))

    init_pos = targ_def['joint_angles'][-1]
    transition_length = 30
    nb_points =10# (len(targ_def['joint_angles'])-1)/2

    for i in range(num_sketches):

        # create a digit permutation:
        if permute:
            perm_idx= get_permuted_indexes(nb_points) # this
            print("Permutation")
            print(perm_idx)
            print(np.arange(nb_points))
        else:
            perm_idx = np.arange(nb_points)

        targets_init = [targ_def['joint_angles'][::2][i] for i in perm_idx]
        targets_fin = [targ_def['joint_angles'][1::2][i] for i in perm_idx]
        targets_xyz_init = [targ_def['xyz'][::2][i] for i in perm_idx]
        targets_xyz_fin = [targ_def['xyz'][1::2][i] for i in perm_idx]
        #targets_fin = targets_init
        #targets_xyz_fin = targets_xyz_init

        # init_pos = init_poses[np.random.randint(len(init_poses))]
        print('Reset to initial position.')
        init_pos_noise= init_pos + np.random.uniform(-0.1,0.1,9)
        init_pos_noise[0] = init_pos_noise[0] + np.random.uniform(0.0, 0.2)
        init_pos_noise[1] = init_pos_noise[1] + np.random.uniform(-0.2, 0.0)
        reset_state(sim,init_pos_noise)
        move_to_position(sim, controller, init_pos_noise,render=RENDER,viewer=viewer)
        xyz = [];rpy=[];joint_angles = [] # basically a datapoint starts here.
        states =[]
        actions =[]
        onsets =[]
        sketch =[]
        images=[]
        t = 0
        im = 0
        dart_amount = np.random.choice([0,0.2,0.3])# if i < int(num_sketches/1.5) else 0.001
        dart_noise_curr = np.random.normal(0, 0.01, size=9)
        noise_presist = np.random.randint(5, 30)
        print("Noise Presist", noise_presist)
        dart_ctr = noise_presist + 1

        img_dims = np.array([112,112],dtype=np.int)
        capture_dims = np.array([202,212],dtype=np.int)

        targ_num = np.random.randint(0, int(nb_points))
        plt.ion()
        inter = True
        while True:
            target_int = targets_init[targ_num]
            target_fin = targets_fin[targ_num]
            target_xyz_int = targets_xyz_init[targ_num]
            target_xyz_fin = targets_xyz_fin[targ_num]
            control_scheme.execute_step()
            sim.step()
            ja = sim.get_state().qpos[joint_addr]
            xyz.append(np.copy(sim.data.body_xpos[-1]))  # xyz of fingers
            rpy.append(np.copy(sim.data.body_xpos[-1]))  # roll_pitch_yaw 0f fingers (in quaternion)
            joint_angles.append(ja)
            if RENDER:
                viewer.render()
            # ice()
            a_int = controller.act(ja, target_int)

            a_fin = controller.act(ja, target_fin)
            #a_fin = a_int
            if t<transition_length:
                a_int[:-3] = a_int[:-3]*(t/float(transition_length))*0.8
                a_fin[:-3] =a_fin[:-3] *(t/float(transition_length))*0.8
                a_int[1] = -1.3
                a_fin[1] = -1.3

            decay_factor = 1 / 1500
            weight = np.exp(-decay_factor * t)
            if np.sum(np.abs(xyz[-1] - target_xyz_int)) > 0.05 and inter:
                a = a_int  #* weight + a_fin * (1 - weight) # control somewhere between top and down position

            else:
                inter = False
                a = a_fin
            #a = a_fin

            if dart_ctr>noise_presist:
                #dart_amount = np.random.choice([0,0.5])
                dart_noise_prev = dart_noise_curr
                noise_var = np.abs(controller.c_low)*dart_amount

                noise_var[0]+=np.random.uniform(0,0.1)
                #noise_var[1]+=np.random.uniform(0,0.15)
                dart_noise_curr = np.random.normal(0,noise_var, size=9)
                dart_ctr=0
            d_fac = float((dart_ctr/noise_presist))
            dart_noise = d_fac*dart_noise_curr + (1-d_fac)*dart_noise_prev
            sim.data.ctrl[:] = a + dart_noise_curr
            dart_ctr+=1

            if im % (SUBSAMP) == 0 and img_collect:
                frame = sim.render(capture_dims[1], capture_dims[0] , camera_name="camera_main")
                rem = np.array((capture_dims - img_dims)/2,dtype=np.int)
                frame = frame[rem[0]:-rem[0],20:-80]
                # frame = np.swapaxes(frame, 0, 1)
                # plt.imshow(frame)
                # plt.pause(0.000001)
                images.append(frame)
            im += 1

            e = ja - target_fin
            e_xyz = xyz[-1] - target_xyz_fin

            # time to construct the state space.
            # you need the joint angles and the eucledian distance from all possible targets
            dist = np.ravel(xyz[-1] - targets_xyz_fin)
            states.append(np.concatenate((ja,dist)))
            actions.append(np.array(a))
            onsets.append(targ_num)

            t += 1
            term_cond = np.sum(np.abs(e_xyz))
            #print(term_cond)
            #term_cond = np.abs(e_xyz[-1])

            if term_cond < 0.01:
                t=0
                inter = True
                sketch.append(targ_num)
                print(np.sum(np.abs(e)))
                print(np.sum(np.abs(e_xyz)))
                print("-----------------")
                print("sketch" + str(np.array(sketch)))
                print("time " + str(t))
                # plt.plot(np.array(actions)[:,2])
                # plt.show()
                #
                # ce()
                if len(sketch) == len_sketch:
                    break
                else:
                    while True:
                        targ_num_sug =np.random.randint(0, nb_points)
                        if targ_num_sug != targ_num:
                            break
                    targ_num = targ_num_sug
                    t = 0

                # first collect like 25 of these The target point collection. Then assign 5 of these to each colour.
        data_to_save = list(range(1, nb_points + 1, 2))
        dataset['init_pos'] = init_pos
        dataset['targets_ja'] = [targ_def['joint_angles'][i] for i in data_to_save]
        dataset['targets_xyz'] = [targ_def['xyz'][i] for i in data_to_save]
        dataset['targets_rpy'] = [targ_def['rpy'][i] for i in data_to_save]
        dataset['states'].append(states[::SUBSAMP])
        dataset['actions'].append(actions[::SUBSAMP])
        dataset['gt_onsets'].append(onsets[::SUBSAMP])
        dataset['tasks'].append(sketch)
        dataset['images'].append(images)

        #import matplotlib.pyplot as plt

        print(len(dataset['states'][-1]),len(images))
        print('dataset ' + str(i))
        # print('max_angle (rad) ' + str(np.max([np.amax(dataset['states'][i][:]) for i in range(len(dataset['states']))])))

    # screenshot.save_screenshots()
    pickle.dump(dataset, open(data_dir, 'wb'),protocol=2)

def analyse_data(dataset_name='test_dataset'):
    targ_def = pickle.load(open('../data/jaco/' + dataset_name + '.p', 'rb'))
    print(str(targ_def['states'][0]))
    TIME_RES = 100
    dataset = {'mean_state': [], 'var_state': [], 'max_state': [], 'min_state': [], 'mean_action': [], 'var_action': [], 'max_action': [], 'min_action': [], 'num_eps': []}
    ep_lengths = [np.size(targ_def['states'][i], axis=0) for i in range(len(targ_def['states']))]
    max_length = np.max(ep_lengths)
    min_length = np.min(ep_lengths)
    mean_length = np.mean(ep_lengths)
    std_length = np.std(ep_lengths)
    num_episodes = len(targ_def['states'])
    num_eps = []

    # timeline plots
    for i in range(max_length):
        M = 0
        S = 0
        dataset['mean_state'].append(M)
        dataset['var_state'].append(S)
        dataset['max_state'].append(-float('inf')*np.ones(np.size(targ_def['states'][0],axis=1)))
        dataset['min_state'].append(float('inf')*np.ones(np.size(targ_def['states'][0],axis=1)))
        num_eps.append(1)
        for j in range(num_episodes):
            try:
                n = num_eps[-1]
                S += ((n * targ_def['states'][j][i] - M) ** 2) / (n * (n + 1))
                M += targ_def['states'][j][i]
                dataset['mean_state'][i] = M / n
                dataset['var_state'][i] = S / (n + 1)
                for k in range(np.size(targ_def['states'][0],axis=1)):
                    dataset['max_state'][i][k] = targ_def['states'][j][i][k] if targ_def['states'][j][i][k] > dataset['max_state'][i][k] else dataset['max_state'][i][k]
                    dataset['min_state'][i][k] = targ_def['states'][j][i][k] if targ_def['states'][j][i][k] < dataset['min_state'][i][k] else dataset['min_state'][i][k]
                num_eps[-1] += 1
            except IndexError:
                pass
            continue
    return dataset

def visualise_data(dataset):
    # plt.figure(1)
    _, ax_full = plt.subplots(9, sharex=True)
    for j in range(9):
        # ax_full = ['ax' + str(k) for k in range(9)]
        # _, ax_full = plt.subplots(9, sharex=True)
        ax_full[j].plot(range(len(dataset['mean_state'])),[dataset['mean_state'][i][j] for i in range(len(dataset['mean_state']))])
        # ax_full[j].fill_between(range(len(dataset['mean_state'])),[dataset['mean_state'][i][j] - (dataset['var_state'][i][j]) ** 0.5 for i in range(len(dataset['mean_state']))],[dataset['mean_state'][i][j] + (dataset['var_state'][i][j]) ** 0.5 for i in range(len(dataset['mean_state']))], alpha=0.4)
        ax_full[j].plot(range(len(dataset['max_state'])),[dataset['max_state'][i][j] for i in range(len(dataset['max_state']))])
        ax_full[j].plot(range(len(dataset['min_state'])),[dataset['min_state'][i][j] for i in range(len(dataset['min_state']))])
    plt.show()



if __name__ == '__main__':
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode',type=str,
                        help="Mode: collect or define")

    parser.add_argument('--render','-r',default=False,action = 'store_true',
                        help="To render or not")
    parser.add_argument('-n','--nb_datapoints', default=2,type=int,
                        help="number of points to define or collect")

    parser.add_argument('--name', default='test_dataset',type=str,
                        help="Name of collected dataset")

    parser.add_argument('--joint_targets', default='joint_targets_easy',type=str,
                        help="Definitions for joint angles for each button")

    parser.add_argument('--vis', default=False,action='store_true',
                        help="Definitions for joint angles for each button")

    parser.add_argument('--img_collect', default=False,action='store_true',
                        help="collect images in the data")

    parser.add_argument('--permute', default=False,action='store_true',
                        help="permute  the target ")

    args = parser.parse_args()
    RENDER = args.render

    if args.mode == 'define':
        targets_loaded = yaml.load(open("../cfg/" + args.joint_targets + ".yaml"))

        fingers = [0.68, 0.68, 0.68]
        a = 0.14
        b = -0.07
        targets = np.array([targets_loaded[i] + fingers for i in range(0, 105, 5)])

        define_targets(targets)
    elif args.mode == 'collect':
        collect_data(args.nb_datapoints,4,args.name,img_collect=args.img_collect,permute = args.permute)
        if args.vis:
            analysis = analyse_data(dataset_name=args.name)
            visualise_data(analysis)
    else:
        print("Unknown mode. Use define or collect")


