import envs.mp_env
import torch.nn as nn
import torch
from models.rl_model import QAgent, QAgentAsymmetric, QAgentForgetting
from models.rnn_model import RLRNN
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import dill as pickle
from analysis_scripts.logistic_regression import logistic_regression_masked

def make_env(env_params, env_type = None):

    if env_type is None:
        env_type = env_params.get('env_type','mp')

    if 'mp' in env_type:
        env = envs.mp_env.MPEnv(fixed_length=env_params['fixed_length'],reset_time=env_params['reset_time'],
                                opponent=env_params['opponent'],
                                opp_params=env_params['opponents_params'])               
        
    env.reset()
    return env

def load_model(path, ep=None):
    '''
    Loads a model from a path assuming that there's a parameter file in the directory
    '''
    try:
        net_params = pickle.load(open(os.path.join(path,'model_input_dict.pkl'),'rb'))
    except:
        net_params = pickle.load(open(os.path.join(path,'params.pkl'),'rb'))
    model = RLRNN(**net_params)
    if ep is None:
        model.load_model(path)
    else:
        model.load_model_ep(ep)

class RLTester(nn.Module):
    '''
    Wrapper for RL to enable it to be used in lieu of an RNN module
    '''
    def __init__(self, alpha=0.2, gamma=0, init_action=None, bias=0, epsilon=0.05, path=None, load=True, max_steps = 1000, env = None, deterministic = True, asymmetric = False, forgetting=False,beta =1):
        super(RLTester, self).__init__()
        if not asymmetric and not forgetting:
            self.policy = QAgent(alpha, gamma, init_action, bias, epsilon, path, load, temperature=beta, env = env, deterministic=deterministic)
        elif forgetting:
            self.policy = QAgentForgetting(alpha, gamma, init_action, bias, epsilon, load, env = env, deterministic=deterministic, temperature = 1/beta)
        else:
            self.policy = QAgentAsymmetric(alpha, gamma, init_action, bias, epsilon, load, env = env, deterministic=deterministic, beta = beta)
        self.QRL = None
        self.RL = None
        self.num_lstms = 1
        self.hidden_dim = 1
        self.action_dim = env.action_space.n
        self.max_steps = max_steps
        self.policy.QRL = None
        self.DETERMINISTIC=True
        self.num_rnns = 1
        self.env = env
        self.policy.RL = None
        self.reset_time = self.env.reset_time
        
    def generate_data(self, nits = 500, p = .25):
        ''' generates data for use in analysis functions. Returns a df (like monkey data) '''
        episode_actions = []
        episode_rewards = []
        mask_win = []
        mask_loss = []
        mask_frac = []
        episode_states = []
        for i in range(nits):
            self.policy.reset()
            state = self.env.reset()
            prev_action = self.env.action_space.sample()
            prev_action = torch.nn.functional.one_hot(torch.tensor(prev_action), num_classes=self.env.action_space.n).float()
            reward = torch.randint(2,(1,))
            actions = []
            rewards = []
            states = []
            ep_frac = []
            for j in range(self.reset_time):
                
                reward = reward * prev_action
                action, _ , _ , _ = self.forward(state, prev_action, reward)
                state, reward, done, info = self.env.step(action)
                prev_action = action  # Update for next iteration
                frac_flag = info['fractional']
                if isinstance(frac_flag,int):
                    ep_frac.append(frac_flag)
                else:
                    ep_frac.append(1)
                # action = action[1] #because it's passing a one-hot and we want it in binary 
                actions.append(action[1])
                rewards.append(reward)
                states.append(state[1])
                # action, _ , _ , _ = self.forward(state, last_action, reward)
                # last_action = action
                # action = action[1] #because it's passing a one-hot and we want it in binary 
                # actions.append(action)
                # state, reward, done, _ = self.env.step(action)
                # rewards.append(reward)
                # reward = torch.nn.functional.one_hot(torch.tensor(reward,dtype=torch.int64),num_classes=2).float()
                # self.Qs = self.forward(self.params,self.Qs,action, reward, None, None)
            episode_actions.append(actions)
            episode_rewards.append(rewards)
            episode_states.append(states)   
            mask_frac.append(ep_frac)
        mask_win = [np.roll(episode_rewards[i], 1) for i in range(len(episode_rewards))]
        mask_loss = [1-mask for mask in mask_win]
        masks = [None, mask_win, mask_loss, mask_frac]  
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        episode_states = np.array(episode_states)
        
        # create df from these data
        bundle_dict = {'id' : [], 'trial' : [], 'monkey_choice' : [], 'reward' : [], 'computer_choice' : []}
        for epi in range(len(episode_actions)):
            for tr in range(len(episode_actions[epi])):
                bundle_dict['id'].append(epi)
                bundle_dict['trial'].append(tr)
                bundle_dict['monkey_choice'].append(episode_actions[epi][tr])
                bundle_dict['reward'].append(episode_rewards[epi][tr])
                bundle_dict['computer_choice'].append(episode_states[epi][tr])
        bundle = pd.DataFrame.from_dict(bundle_dict)
        bundle = bundle.sort_values(by=['id','trial'])
        return bundle, masks
        # print('Win Rate: {}'.format(np.mean(episode_rewards)))
        # for mask in masks:
        #     logistic_regression_masked(False, bundle, mask = mask, err = True, fitted_RL=True)
        
        
    def fit_logistic(self):
        from torch.nn.functional import one_hot
        episode_actions = []
        episode_rewards = []
        mask_win = []
        mask_loss = []
        for i in range(100):
            self.policy.reset()
            state = self.env.reset()
            action = self.env.action_space.sample()
            reward = np.random.randint(0,2)
            actions = []
            rewards = []
            self.Qs = self.policy.forward(one_hot(torch.tensor(state,dtype=torch.int64),num_classes=2).view(1,1,-1),
                                          one_hot(torch.tensor(action,dtype=torch.int64),num_classes=2).view(1,1,-1),
                                          one_hot(torch.tensor(reward,dtype=torch.int64),num_classes=2).view(1,1,-1))
            for j in range(self.reset_time):
                action = np.argmax(self.policy.Qs)
                actions.append(action)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                self.Qs = self.policy.forward(one_hot(torch.tensor(state,dtype=torch.int64),num_classes=2).view(1,1,-1),
                                          one_hot(torch.tensor(action,dtype=torch.int64),num_classes=2).view(1,1,-1),
                                          one_hot(torch.tensor(reward,dtype=torch.int64),num_classes=2).view(1,1,-1))
            episode_actions.append(actions)
            episode_rewards.append(rewards)
        mask_win = [np.roll(episode_rewards[i], 1) for i in range(len(episode_rewards))]
        mask_loss = [1-mask for mask in mask_win]
        masks = [None, mask_win, mask_loss]  
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        
        # create df from these data
        bundle_dict = {'id' : [], 'trial' : [], 'monkey_choice' : [], 'reward' : [], 'computer_choice' : []}
        for epi in range(len(episode_actions)):
            for tr in range(len(episode_actions[epi])):
                bundle_dict['id'].append(epi)
                bundle_dict['trial'].append(tr)
                bundle_dict['monkey_choice'].append(episode_actions[epi][tr])
                bundle_dict['reward'].append(episode_rewards[epi][tr])
                bundle_dict['computer_choice'].append(-1)
        bundle = pd.DataFrame.from_dict(bundle_dict)
        # bundle = (None, episode_actions, episode_rewards, None, None, None)
        print('Win Rate: {}'.format(np.mean(episode_rewards)))
        for mask in masks:
            logistic_regression_masked(False, bundle, mask = mask, err = True, fitted_RL=True)
        
    def test(self, env = None, nits = 100):
        '''
        Test the policy on the environment
        '''
        if env is None:
            env = self.env
        # self.max_steps = env.reset_time
        # self.max_steps = max_steps
        avg_rewards = []
        optimal_rewards = []
        
        for i in range(nits):
            # env.reset()
            self.policy.reset()
            state = env.reset()
            last_action = torch.nn.functional.one_hot(torch.tensor(env.action_space.sample()), num_classes=env.action_space.n).float()
            
    
            reward = env.step(last_action)[1] # shouldnt be step because of how certain envs are set up
            
            state = torch.tensor(state).view(1,1,1)
            last_action = torch.Tensor(last_action).view(1,1,len(last_action))
            done = False
            total_reward = 0
            steps = 0
            
            try:
                optimal_rewards.append(max(env.p))
                print(env.p)
            except:
                optimal_rewards.append(max(env.p_dist))
                print(env.p_dist)
            
            while not done and steps < self.max_steps:  
                last_reward = torch.Tensor(reward * last_action).view(1,1,-1)
                state = torch.tensor(state)
                action, pr, _ = self.policy.get_action(state, last_action,None,last_reward)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                last_action = action
                if steps == self.max_steps:
                    done = True
                # print(self.policy.Qs)
            avg_rewards.append(total_reward/steps)
        return np.mean(avg_rewards), np.std(avg_rewards), np.mean(optimal_rewards)
    
    def forward(self, state, last_action, last_reward, hidden_in=None):
        '''
        Forward pass for the model
        '''
        return *self.policy.get_action(state, last_action, last_reward, hidden_in), hidden_in
    
    def visualize_behavior(self,env = None):
        '''
        Plot the Q values for 1 set of trials
        '''
        if env is None:
            env = self.env
        # self.max_steps = env.reset_time
        self.max_steps =1000
        avg_rewards = []
        all_Qs = []
        
        
        self.policy.reset()
        last_action, state = env.reset()
        optimal_index = np.argmax(env.p_dist)
        state = torch.tensor(state).view(1,1,1)
        last_action = torch.Tensor(last_action).view(1,1,len(last_action))

        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < self.max_steps:
            state = torch.tensor(state)
            action, pr, _ = self.policy.get_action(state, last_action,None)
            action = action.detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            last_action = action
            all_Qs.append(copy.deepcopy(self.policy.Qs.detach().numpy()))
            
        all_Qs = np.vstack(all_Qs).squeeze().T

        for i in range(env.action_space.n):
            if i == optimal_index:
                c = 'r'
                a = 1
            else:
                c = 'y'
                a = .5
            plt.plot(all_Qs[i],c = c,alpha = a)
            
class RLTester_Softmax(RLTester):
    def __init__(self, alpha=0.2, gamma=0, init_action=None, bias=0, epsilon=0.05, path=None, load=True, max_steps = 1000, env = None, deterministic = False, temperature = 1,asymmetric = False):

        super(RLTester_Softmax, self).__init__()
        # initialize the same way as RLTester
        self.policy = QAgentAsymmetric(alpha, gamma, init_action, bias, epsilon, path, load, env = env, deterministic=deterministic,temperature = temperature)
        self.max_steps = max_steps
        self.env = env
        
    def forward(self, state, last_action, last_reward, hidden_in=None):
        '''
        Forward pass for the model. similar to RLTester, but uses softmax instead of argmax
        '''
        return *self.policy.get_action(state, last_action, last_reward, hidden_in), hidden_in
    
