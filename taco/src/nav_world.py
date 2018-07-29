import numpy as np
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from taco.src.models_common import load_policy
import tensorflow as tf


def sample_pos(world_size,rang = True):
    if rang:
        x = np.random.uniform(world_size[0],world_size[1],1)
        y = np.random.uniform(world_size[0], world_size[1],1)
    else:
        x = np.random.uniform(0,world_size[0],1)
        y = np.random.uniform(0, world_size[1],1)

    return np.array([x,y])

def get_state(agent_pos,pos_list,type = 'abs'):
    if type == 'abs':
        state = np.vstack(([agent_pos],pos_list))
    elif type == 'rel':
        state = agent_pos - np.array(pos_list)
    return state

def get_action(state,target,vel=0.1):
    # get the direction. resolve the components in that velocity.
    direc = state[target+1,:]-state[0,:]
    direction = np.arctan2(direc[0],direc[1])

    if np.linalg.norm(direc)<0.1:
        vel = np.linalg.norm(direc)

    dx = vel*np.sin(direction)
    dy = vel*np.cos(direction)

    return np.array((dx,dy))



class NavEnv(object):
    def __init__(self,location_noise,transition_noise,relative = True,done_thresh = 0.001):
        self.loc_noise = location_noise
        self.trans_noise = transition_noise
        self.relative = relative
        self.done_thresh = done_thresh

    def reset(self):
        self.abs_state = np.zeros((5,2))
        self.abs_state[0,:] =  np.squeeze(sample_pos([0.2,0.5],rang=True))
        self.abs_state[1, :] = np.array([0.8,1]) + np.random.normal(loc=0,scale = self.loc_noise,size=2)
        self.abs_state[2, :] =np.array([0.7,0.2]) + np.random.normal(loc=0,scale = self.loc_noise,size=2)
        self.abs_state[3, :] = np.array([0.1,0.3]) + np.random.normal(loc=0,scale = self.loc_noise,size=2)
        self.abs_state[4, :] = np.array([0.2,1.2]) + np.random.normal(loc=0,scale = self.loc_noise,size=2)
        return self.abs_state,self._state_from_abs()


    def step(self,action,target):
        self.abs_state[0,:]+=action + np.random.normal(0,self.trans_noise,size=2)*np.linalg.norm(action)
        return self.abs_state,self._state_from_abs(),self.check_done(target)

    def check_done(self,target):
        if np.linalg.norm(self.abs_state[0] - self.abs_state[target+1])<self.done_thresh:
            return True
        else:
            return False

    def _state_from_abs(self):
        out =self.abs_state[0] - self.abs_state[1:] if self.relative else  self.abs_state
        return np.ravel(out)

def collect_data(num_traj,save_dir,animate=False,**kwargs):
    transition_noise = kwargs['transition_noise'] if 'transition_noise' in kwargs.keys() else 0.05
    location_noise = kwargs['location_noise'] if 'location_noise' in kwargs.keys() else 0.2
    env = NavEnv(location_noise,transition_noise)
    dataset = {'states':[],'actions':[],'gt_onsets':[],'tasks':[],'params':kwargs}
    train_sketches = [[0,1,3],[1,2,3],[3,2,1],[2,3,0],[1,2,0],[1,0,2],[1,3,2],[0,3,2],[1,2,3],[3,2,0],[2,1,3],[1,3,0]]

    if animate:
        fig = plt.figure()
        ims = []

    for i in range(num_traj):
        g_state,r_state = env.reset()
        colours = ['b', 'r', 'g', 'y', 'k']
        # randomly sample integets that constitute the sketch
        sketch = train_sketches[np.random.randint(0,len(train_sketches),1)[0]]
        curr_idx = 0
        curr_subtask = sketch[curr_idx]
        sketch_traj = []
        # begin trajectory
        traj_states = []
        traj_actions = []
        while True:
            sketch_traj.append(curr_subtask) # This gives us ground truth about the task executed at each timestep
            #all_pos = np.array([agent_pos,red_pos,green_pos,yel_pos,black_pos])
            #state = np.ravel(get_state(all_pos[0],all_pos[1:],type = 'rel'))
            traj_states.append(r_state)
            action = get_action(g_state,curr_subtask)
            g_state,r_state,done = env.step(action,curr_subtask)
            traj_actions.append(action)
            if animate:
                ims.append((plt.scatter(g_state[:, 0], g_state[:, 1], c=colours),))
            if done:
                if curr_idx<len(sketch)-1:
                    curr_idx+=1

                    curr_subtask = sketch[curr_idx]
                else:
                    dataset['states'].append(traj_states)
                    dataset['actions'].append(traj_actions)
                    dataset['gt_onsets'].append(sketch_traj)
                    dataset['tasks'].append(sketch)
                    break
    save_dir = save_dir if save_dir[-2:] == '.p' else save_dir+'.p'
    pickle.dump(dataset,open(save_dir,'wb'))

    if animate:
        print('WRITING')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                           blit=True)
        im_ani.save(save_dir+'im_mod.mp4', writer=writer)



def evaluate_model(nb_traj,policy,save_dir= None,animate=False,**kwargs):
    transition_noise = kwargs['transition_noise'] if 'transition_noise' in kwargs.keys() else 0.02
    location_noise = kwargs['location_noise'] if 'location_noise' in kwargs.keys() else 0.1
    env = NavEnv(location_noise, transition_noise,done_thresh=0.03)
    if kwargs['zero_shot']:
        test_sketches = [[0, 1, 3,0], [3, 2, 3,2], [0, 2, 1,0], [1, 3, 0,2], [1, 3, 0,3], [1, 3, 2,1], [1, 3, 2,3], [2, 3, 2,1], [1, 2, 3,1],
                      [3, 1, 0,1], [3, 1, 2,0], [2, 3, 0,1],[1,3,1,3],[2,3,2,3],[1,2,1,2],[0,1,0,1],[0,2,0,2],[0,3,0,3]]
    else:
        test_sketches = [[0, 1, 3], [1, 2, 3], [3, 2, 1], [2, 3, 0], [1, 2, 0], [1, 0, 2], [1, 3, 2], [0, 3, 2],
                          [1, 2, 3], [3, 2, 0], [2, 1, 3], [1, 3, 0]]

    score = []
    if animate:
        fig = plt.figure()
        ims = []
    for i in range(nb_traj):
        task_score = []
        g_state,r_state = env.reset()
        colours = ['b', 'r', 'g', 'y', 'k']
        # randomly sample integets that constitute the sketch
        sketch = test_sketches[np.random.randint(0,len(test_sketches),1)[0]]
        curr_idx = 0
        curr_subtask = sketch[curr_idx]
        counter=0
        while True:
            action,stop = policy.forward_full([[r_state]], curr_subtask, dropout=1.)
            g_state,r_state,done = env.step(action,curr_subtask)
            if stop == 1 or counter >100:
                if done:
                    task_score.append(1)
                else:
                    task_score.append(0)
                if curr_idx < len(sketch) - 1:
                    curr_idx += 1
                    curr_subtask = sketch[curr_idx]
                    counter = 0
                else:
                    score.append(task_score)
                    break

            if animate:
                ims.append((plt.scatter(g_state[:, 0], g_state[:, 1], c=colours),))
            counter+=1
    if animate:
        print('writing video at:',save_dir+'im_mod.mp4')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                           blit=True)
        im_ani.save(save_dir+'im_mod.mp4', writer=writer)
    acc = 0
    for s in score:
        if np.sum(sum(s)) == len(s):
            acc+=1
    acc /=float(len(score))
    # also returns accuracy so you dont calculate it outside
    return score,acc

def model_eval(model_dir,num_points,animate = True,**kwargs):
    policy = load_policy(model_dir)
    score,accuracy =  evaluate_model(num_points,policy,save_dir=model_dir,animate=animate,**kwargs)
    policy.sess.close()
    tf.reset_default_graph()
    del policy.sess
    return score,accuracy




if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('name',type=str,
                        help="Directory for data to be collected")

    parser.add_argument('-n','--num_trials', default=1000,type=int,
                        help="Number of datapoints to collect")
    parser.add_argument('-r','--render', action='store_true',
                        help="Render the dataset")

    args = parser.parse_args()
    if not os.path.exists('../data/nav/'):
        os.makedirs('../data/nav/')
    params = {'transition_noise':0.01,'location_noise':0.3}
    collect_data(args.num_trials,os.path.join('../data/nav/',args.name),animate = args.render,**params)
