import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

Episode = namedtuple('Episode',
                        ('state', 'action', 'last_action', 'next_state', 'reward', 'done',
                         'hidden_in', 'hidden_out'))  

#build baseline but using policy network structure


class ReplayMemory:
    def __init__(self, capacity=1e6, RNN=True):
        self.Episode = namedtuple('Episode',
                        ('state', 'action', 'last_action', 'next_state', 'reward', 'done',
                         'hidden_in', 'hidden_out'))   
        self.capacity = int(capacity)
        self.memory = deque([], maxlen=self.capacity)
        self.RNN = RNN

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Episode(*args))

    def sample(self, batch_size):
        '''Samples batch_size episodes'''
        s, a, la, ns,r,d, hi, ho = [],[],[],[],[],[],[],[]
        batches = random.sample(self.memory, batch_size)
        
        for b in batches:
            states,actions,last_actions, next_states, rewards, done, hidden_in, hidden_out = b
            s.append(states)
            a.append(actions)
            la.append(last_actions)
            ns.append(next_states)
            r.append(rewards)
            d.append(done)
            hi.append(hidden_in) # h_in: (1, batch_size=1, hidden_size)
            ho.append(hidden_out)
           
        hi = torch.cat(hi,dim=1) #do i need to detach?
        ho = torch.cat(ho,dim=1)
         
        return s, a, la, ns, r, d, hi, ho 

    def __len__(self):
        return len(self.memory)

    def get_length(self):
        return self.__len__()
    
    def reset(self):
        self.memory = deque([], maxlen=int(self.capacity))


class BaselineNetwork(nn.Module):
    def __init__(self, hidden_dim = 128, num_rnns = 1, environment = None, gamma = .99, batch_size = 16,
                eps = .05, tau = .005, lr = 1e-4, eps_decay = 500) -> None:
        super(BaselineNetwork,self).__init__()
        self.env = environment
        self.max_episodes = 100000
        self.max_steps = self.env.reset_time
        
        self.batch_size = batch_size
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n 
        self.action_dim = num_actions
        self.hidden_dim = hidden_dim
        self.network = DQN(num_inputs, num_actions, dim = hidden_dim, num_rnns = num_rnns, env = self.env, gamma = gamma, batch_size = 16,
                eps = eps, tau = tau, lr = lr, eps_decay = eps_decay)
        self.target = DQN(num_inputs, num_actions, dim = hidden_dim, num_rnns = num_rnns, env = self.env, gamma = gamma, batch_size = 16,
                eps = eps, tau = tau, lr = lr, eps_decay = eps_decay)
        self.target.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory()
        self.steps_done = 0
        
        self.num_rnns = num_rnns
        
    def evaluate_model(self,env):
        '''Runs one episode of the environment'''
        # env = copy.deepcopy(self.environment)
        # env.clear_data()

        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        episode_value = []
        episode_policy = []
        state =  env.reset()
        if isinstance(state,int):
            state = torch.Tensor([state])
        if isinstance(state,np.ndarray):
            state = torch.Tensor(state).view(1,1,-1)
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=self.action_dim).view(1,1,-1)
        
        hidden_out = torch.zeros([self.num_rnns,1, self.hidden_dim], dtype=torch.float)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        # hidden_out = torch.randn([self.num_rnns,1, self.hidden_dim], dtype=torch.float,device=device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             

        hidden_out = hidden_out
        

        for step in range(self.max_steps):
            # last_action = action
            hidden_in = hidden_out
            
            action, hidden_out = self.network.select_action(state, last_action, hidden_in)
            next_state, reward, done, _ = env.step(action)
            action = action.to(torch.float32)
            action = action.view(1,1,-1)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1,1,-1)

            # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
            # env.render()       
                
            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out
            if step == self.max_steps - 1:
                done = True
            episode_state.append(state)
            episode_action.append(action)
            episode_last_action.append(last_action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)
            episode_done.append(done) 


            
                
            state = next_state
            last_action = action
            if done:
                break
        episode_state = torch.cat(episode_state,dim=1)
        episode_action = torch.cat(episode_action, dim=1)
        episode_last_action = torch.cat(episode_last_action, dim=1)
        episode_reward = torch.tensor(episode_reward).requires_grad_(False).view(1,-1,1)
        episode_next_state = torch.cat(episode_next_state, dim=1)
        episode_done = torch.tensor(np.float32(episode_done)).view(1,-1,1)
        self.memory.push(episode_state, episode_action, episode_last_action, episode_next_state, episode_reward, 
                episode_done, ini_hidden_in, ini_hidden_out)
        
        return torch.mean(episode_reward.to(torch.float32))
    
    
    
    # def evaluate_model(self, env):
    #     '''Runs one episode of the environment'''
    #     episode_state = []
    #     episode_action = []
    #     episode_last_action = []
    #     episode_reward = []
    #     episode_next_state = []
    #     episode_done = []
    #     state =  env.reset()
    #     # if isinstance(state,int):
    #     state = torch.Tensor(state)
        
    #     last_action = env.action_space.sample()
    #     last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=self.action_dim).view(1,1,env.action_space.n).numpy()
        
    #     hidden_out = torch.zeros([self.num_rnns,1, self.hidden_dim], dtype=torch.float)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
    #     # hidden_out = torch.randn([self.num_rnns,1, self.hidden_dim], dtype=torch.float) / np.sqrt(2) # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        
    #     for step in range(self.max_steps):
    #         # last_action = action
    #         hidden_in = hidden_out
    #         last_action = torch.tensor(last_action)
    #         action, hidden_out = self.network.select_action(state, last_action, hidden_in)
    #         next_state, reward, done, _ = env.step(action)
    #         action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=self.action_dim).numpy()
    #         # action = action.to(torch.float32)
    #         next_state = torch.tensor(next_state, dtype=torch.float32).view(1,1,-1)
    #         # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
    #         # env.render()       
                
    #         if step == 0:
    #             ini_hidden_in = hidden_in
    #             ini_hidden_out = hidden_out
    #         if step == self.max_steps - 1:
    #             done = True
    #         episode_state.append(state)
    #         episode_action.append(action)
    #         episode_last_action.append(last_action)
    #         episode_reward.append(reward)
    #         episode_next_state.append(next_state)
    #         episode_done.append(done) 


    #         state = next_state
    #         last_action = action
    #         if done:
    #             break
    #     # episode_state = torch.tensor(episode_state)
    #     # episode_action = torch.tensor(episode_action)
    #     # episode_last_action = torch.cat(episode_last_action,dim)
    #     # episode_reward = torch.tensor(episode_reward)
    #     # episode_next_state = torch.tensor(episode_next_state)
    #     # episode_done = torch.tensor(episode_done)
    #     episode_state = torch.cat(episode_state,dim=1)
    #     episode_action = torch.cat(episode_action, dim=1)
    #     episode_last_action = torch.cat(episode_last_action, dim=1)
    #     episode_reward = torch.tensor(episode_reward).requires_grad_(False).view(1,-1,1)
    #     episode_next_state = torch.cat(episode_next_state, dim=1)
    #     episode_done = torch.tensor(np.float32(episode_done)).view(1,-1,1)
    #     self.memory.push(episode_state, episode_action, episode_last_action, episode_next_state, episode_reward, 
    #             episode_done, ini_hidden_in, ini_hidden_out)
        
    #     return torch.sum(episode_reward)/torch.numel(episode_reward)
    # RIGHT NOW DOESNT PUSH TO STACK, DOESNT DO A LOT OF THINGS. PROBABLY REWRITE AS NORMAL
    # AND USE  
    
    def update(self):
        # transitions = self.memory.sample(self.network.BATCH_SIZE)
        # batch = Episode(*transitions)        
        
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                   batch.next_state)), dtype=torch.bool)
        # non_final_next_states = torch.cat([torch.tensor(s) for s in batch.next_state
        #                                             if s is not None])
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # reward_batch = torch.cat(batch.reward)
        # last_action_batch  = torch.cat(batch.last_action)
        # next_state_batch = torch.cat(batch.next_state)
        # done_batch = torch.cat(batch.done)
        # hidden_in_batch = torch.cat(batch.hidden_in)
        
        # state_action_values = self.network(state_batch).gather(1, action_batch)
        # next_state_values = torch.zeros(self.BATCH_SIZE)
        
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target(non_final_next_states).max(1).values
        # expected_state_action_values = (next_state_values * self.network.GAMMA) + reward_batch
        
        
        
        state, action, last_action, next_state, reward, done, hidden_in, hidden_out, = self.memory.sample(self.batch_size)
        
        # state      = torch.stack(state).unsqueeze(-1)
        # next_state = torch.stack(next_state).unsqueeze(-1)
        # action     = torch.stack(action).squeeze().type(state.dtype)
        # last_action     = torch.stack(last_action).squeeze().type(state.dtype)
        # reward     = torch.stack(reward).unsqueeze(-1)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        # done       = torch.stack(done).unsqueeze(-1)
        
        state      = torch.cat(state).to(torch.float32)
        next_state = torch.cat(next_state).to(torch.float32)
        action     = torch.cat(action).to(torch.float32)
        last_action= torch.cat(last_action).to(torch.float32)
        reward     = torch.stack(reward).to(torch.float32)
        done       = torch.cat(done).to(torch.float32)
        
        
        state_action_values, _ = self.network(state, last_action, hidden_in)
        with torch.no_grad():
            new_action, ho = self.target(state, last_action, hidden_in)
            # new_action, ho = self.target.select_action(next_state, action, hidden_in)
            # new_action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=self.action_dim)
            new_action = torch.nn.functional.one_hot(new_action.argmax(dim=-1),num_classes=self.action_dim).type(state.dtype)

            next_state_values,_ = self.target(next_state, new_action, hidden_in)
            # next_state_values,_ = self.target(next_state, last_action, hidden_in)
            next_state_values = next_state_values.max(-1).values.unsqueeze(-1)# is this right tho
        expected_state_action_values = (next_state_values * self.network.GAMMA) + reward
        
        
        
        # Compute Huber loss
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        l = loss.to(torch.float32)
        l.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), .5)
        self.optimizer.step()
        
        target_net_state_dict = self.target.state_dict()
        policy_net_state_dict = self.network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.network.TAU + target_net_state_dict[key]*(1-self.network.TAU)
        self.target.load_state_dict(target_net_state_dict)
        self.network.steps_done += 1
        self.target.steps_done +=1
        
        return loss
        
    # def save_model_ep(self, ep):
    #     torch.save(self.network.state_dict(), os.path.join(self.model_path,self.model_name +'_'+ str(ep)))
        
    def optimize_model(self, nits = False, save=False, max_patience = 1000, prune = True):
        ''' optimizes model, then loads best model and runs test suite, saving the figures'''
        training_iter = 0
        self.network.train()
        self.target.train()
        # self.sq2.train()
        if nits == False:
            nits = self.batch_size*8
        losses = []
        
        env = self.env
        rewards = []
        prev_max = -1
        min_ep = 0
        m1 = 0
        m2 = 0
        patience = 0
        for eps in range(self.max_episodes):
            batch_rewards = []
            self.memory.reset()
            
            for b in range(nits):
                batch_rewards.append(self.evaluate_model(env))
               
            rewards.append(np.mean(batch_rewards))
            # self.writer.add_scalar('Rewards/train',np.array(rewards[-1]),training_iter)
            # self.sq2.train()
            if (rewards[-1]) >= prev_max:
                prev_max = rewards[-1]
                if min_ep == 0:
                    min_ep = eps
                # self.save_model_ep(training_iter)
                best_model = training_iter     
                self.training_iters = training_iter
            training_iter+=1

            if len(rewards) > self.batch_size:
                if m1 == 0:
                    m1 = max(rewards[-int(self.batch_size):])
                m2 = rewards[-1]

                if m1 > m2:
                    patience += 1
                    if patience > max_patience:
                        # self.writer.close()
                        # self.load_model_ep(best_model)
                        # self.save_model(self.model_path)
                        print(best_model)
                        self.best_model = best_model
                        if prune == True:
                            self.prune_saves(rewards,0)
                        if not save:
                            plt.plot(rewards)
                            plt.title('Rewards')
                            plt.xlabel('Episode')
                            plt.ylabel('Reward')
                            plt.axvline(x=best_model, color='r', linestyle='--')
                            plt.show()
                            plt.title('Policy Network Loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Episode')
                            plt.axvline(x=best_model, color='r', linestyle='--')
                            plt.plot(losses)
                            plt.show()
                        else:
                            plt.plot(rewards)
                            plt.title('Rewards')
                            plt.xlabel('Episode')
                            plt.ylabel('Reward')
                            plt.axvline(x=best_model, color='r', linestyle='--')
                            plt.savefig(os.path.join(self.model_path,self.model_name,'rewards.png'))
                            plt.clf()
                            plt.title('Policy Network Loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Episode')
                            plt.plot(losses)
                            plt.axvline(x=best_model, color='r', linestyle='--')
                            plt.savefig(os.path.join(self.model_path,self.model_name,'policy.png'))
                            plt.clf()
                        break
                else: 
                    patience = 0
                    m1 = m2


                
            losses.append(torch.sum(self.update()).item()) 
            print('Episode: ', eps, '| Episode Reward: ', rewards[-1], '| loss: ', losses[-1])



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, dim = 128, num_rnns = 1, env = None, gamma = .99, batch_size = 16,
                eps = .05, tau = .005, lr = 1e-4, eps_decay = 500):
        super(DQN,self).__init__()
        
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = 0.9
        self.EPS_END = eps
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = lr
        self.steps_done = 0
        # Get number of actions from gym action space
        # Get the number of state observations
        _ = env.reset()


        
        self.layer1 = nn.Linear(num_inputs+num_actions, dim)
        self.layer2 = nn.RNN(dim,dim, nonlinearity = 'relu', num_layers = num_rnns)
        self.layer3 = nn.Linear(dim,dim)
        self.layer4 = nn.Linear(dim,num_actions)
        
        self.env = env
        
        self.num_inputs = num_inputs 
        self.num_actions = num_actions
        self.hidden_dim = dim
        
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer3.weight,nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer4.weight,nonlinearity='relu')
        nn.init.uniform_(self.layer1.bias)
        nn.init.uniform_(self.layer3.bias)
        nn.init.uniform_(self.layer4.bias)
                
        # nn.init.normal_(self.layer2.weight_hh_l0, std = .5)
        nn.init.orthogonal_(self.layer2.weight_hh_l0)
        nn.init.kaiming_normal_(self.layer2.weight_ih_l0, nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer4.weight,nonlinearity='relu')
        
    
    def forward(self, state,last_action, hidden):
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        x = torch.cat([state,last_action],-1)
        x = F.relu(self.layer1(x))
        x, hidden = self.layer2(x, hidden)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = x.permute(1,0,2)
        return x, hidden
    
    def select_action(self,state, last_action, hidden):
        num_dims = len(state.shape)
        if num_dims < 3: #then we know we're doing one step at a time instead of based on batches, and can unsqueeze so that we can write general code
            state = torch.Tensor(state).squeeze().view(1,1,self.num_inputs)
            last_action = torch.Tensor(last_action).squeeze().view(1,1,self.num_actions)        
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            x, hidden = self.forward(state, last_action, hidden)
            # x = x.max(-1).indices.view(1, 1)
            x = x.max(-1).indices

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                sample = random.random()
                eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                    math.exp(-1. * self.steps_done / self.EPS_DECAY)
                # if sample > eps_threshold:
                if sample < eps_threshold:
                    # IS THIS RIGHT HIDDEN STATE TO RETURN
                    # return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), torch.zeros_like(hidden)
                    
                    # return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), hidden
                    x[i,j] = self.env.action_space.sample()
                
        # self.steps_done += 1
        self.steps_done += torch.numel(state)
        if len(x.shape) != 3:
            x = torch.nn.functional.one_hot(x,num_classes=self.num_actions).view(1,1,-1)
        return x, hidden

        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         # t.max(1) will return the largest column value of each row.
        #         # second column on max result is index of where max element was
        #         # found, so we pick action with the larger expected reward.
        #         x, hidden = self.forward(state, last_action, hidden)
        #         return x.max(-1).indices.view(1, 1), hidden
        # else:
        #     # IS THIS RIGHT HIDDEN STATE TO RETURN
        #     # return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), torch.zeros_like(hidden)
            
        #     # return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), hidden
        #     return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long), hidden
        
        