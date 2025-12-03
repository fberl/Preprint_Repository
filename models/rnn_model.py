import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple, deque
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
import models.rl_model as rl_model
import sys
import dill as pickle
import glob
import copy
import torch.nn.utils.parametrize as parametrize
import analysis_scripts.test_suite as test_suite
import envs.mp_env
from collections.abc import Iterable
import json
# import torch.multiprocessing as multiprocessing
# from joblib import Parallel, delayed
# num_cores = min(2,multiprocessing.cpu_count())
# GPU = True
torch.manual_seed(628980)
# from joblib import Parallel, delayed




# mac = False
# device_idx = 0
# if mac: 
#     device = torch.device("mps")
# else :
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")


class ReplayMemory:
    def __init__(self, capacity=1e6, RNN=True):
        self.Episode = namedtuple('Episode',
                        ('policy','value','state', 'action', 'last_action', 'next_state', 'reward','last_reward', 'done',
                         'hidden_in', 'hidden_out'))   
        self.capacity = int(capacity)
        self.memory = deque([], maxlen=self.capacity)
        self.RNN = RNN

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Episode(*args))

    def sample(self, batch_size):
        '''Samples batch_size episodes'''
        p,v,s, a, la, ns,r, lr,d, hi, ho = [],[],[],[],[],[],[],[],[], [],[]
        batches = random.sample(self.memory, batch_size)
        
        for b in batches:
            policy,values,states,actions,last_actions, next_states, rewards, last_rewards, done, hidden_in, hidden_out = b
            p.append(policy)
            v.append(values)
            s.append(states)
            a.append(actions)
            la.append(last_actions)
            ns.append(next_states)
            r.append(rewards)
            lr.append(last_rewards)
            d.append(done)
            hi.append(hidden_in) # h_in: (1, batch_size=1, hidden_size)
            ho.append(hidden_out)
           
        hi = torch.cat(hi,dim=1) #do i need to detach?
        ho = torch.cat(ho,dim=1)
         
        return p,v,s, a, la, ns, r, lr, d, hi, ho 

    def __len__(self):
        return len(self.memory)

    def get_length(self):
        return self.__len__()
    
    def reset(self):
        self.memory = deque([], maxlen=int(self.capacity))


class ZeroDiagonal(nn.Module):
    def forward(self, X):
        return X - torch.diag(torch.diag(X))

class IdentityDiagonal(nn.Module):
    def __init__(self,tau=1 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
    
    def forward(self, X):
        return ZeroDiagonal()(X) - torch.eye(X.shape[0],device=device) / self.tau    

class IdentityDiagonalAdaptive(nn.Module):
    def __init__(self,tau=1 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
    
    def forward(self, X):
        return ZeroDiagonal()(X) - torch.eye(X.shape[0],device=device) / self.tau.forward()

class NegSemiDefTau(nn.Module):
    def __init__(self, tau) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau,dtype=torch.float),requires_grad=True)
    
    def forward(self):
        return torch.clamp(self.tau, max=0)

class SumOne(nn.Module):
    def __init__(self, norm_p=2):
        super().__init__()
        self.norm_p = norm_p
        
    def forward(self,X):
        X = torch.clamp(X, min = -1, max= 1)
        return X/torch.norm(X, p=self.norm_p)

# class SumOneL2(nn.Module):
#     def forward(self,X):
#         X = torch.clamp(X, min = -1, max= 1)
#         return X/torch.norm(X)
    
class MultLayer(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([alpha,1-alpha],dtype=torch.float32,requires_grad=False))
            
    def forward(self,X,Y, extra):
        return X * self.weight[0] + Y * self.weight[1]   
    
class MultLayerAdaptiveSimple(nn.Module):
    def __init__(self, alpha,):
        super().__init__()
        self.weight1 = MultLayer(alpha[0])
        self.weight2 = MultLayer(alpha[1])
        self.weight = nn.ModuleList([self.weight1, self.weight2])
        
    def forward(self,X,Y,reward):
        res = torch.zeros_like(X)
        for i in range(reward.shape[0]):
            for j in range(reward.shape[1]):
                res[i,j,:] = X[i,j,:] * self.weight[int(reward[i,j,0])].weight[0] + Y[i,j,:] * self.weight[int(reward[i,j,0])].weight[1]   
        return res
    
def normalized_positive_init(shape, gain=1.0):
    R = torch.normal(0,1,size = shape)
    A = torch.matmul(R,R.T) / shape[0]
    #compute max eigenvalue 
    e = torch.max(torch.linalg.eigvals(A))
    return gain * A / e
    
# generates a positive semidefinite matrix with zero eigenvalues, then adds in our diagonal constraint
def normalized_init(shape, gain=1.0):
    R = torch.normal(0,1,size = shape)
    A = torch.matmul(R,R.T) / shape[0]
    #compute max normed eigenvalue 
    e = torch.max(torch.abs(torch.linalg.eigvals(A)))
    return gain * A / e

def recurrent_input_init(hidden_size, gain=1.0):
    whx = torch.normal(0,1/hidden_size,size=(hidden_size,hidden_size))
    alpha = np.sqrt(2) * (1.2) / (max(hidden_size,6)-2.4)
    return torch.nn.Parameter(whx * alpha)
    
def random_init(hidden_size, tau=1 , gain=1.0, eps = None):
    enough_pos = False
    not_too_large = False
    while(not enough_pos and not not_too_large):
        randm = torch.normal(0, gain, (hidden_size, hidden_size))
        # randm = torch.rand(size=(hidden_size,hidden_size)) * gain
        # epsilon = np.sqrt(2)/np.sqrt(hidden_size)
        if eps is not None:
            epsilon = eps
        else:
            l = (torch.max(torch.abs(torch.linalg.eigvals(randm))))
            epsilon = np.sqrt(l)/np.sqrt(hidden_size)
            epsilon = epsilon/2
        randm = randm / np.sqrt(hidden_size) * (1+1/tau - epsilon ) #epsilon to deal with numberical instability causing blowups. should depend on matrix size
        randm = randm - torch.diag(torch.diag(randm)) - torch.eye(hidden_size,device=device) / tau
        
        spectrum = torch.linalg.eigvals(randm)
        reals = torch.real(spectrum)
        enough_pos = torch.sum(reals >= 0)
        not_too_large = torch.max(reals) < 1
        
    return torch.nn.Parameter(randm)

def random_init_uniform(hidden_size, tau=1 , gain=1.0):
    enough_pos = False
    not_too_large = False
    while(not enough_pos and not not_too_large):
        randm = torch.rand(size=(hidden_size,hidden_size)) * gain
        l = (torch.max(torch.abs(torch.linalg.eigvals(randm))))
        epsilon = 1/4 * np.sqrt(l)/np.sqrt(hidden_size)
        randm = randm / np.sqrt(hidden_size) * (1+1/tau - epsilon ) #epsilon to deal with numberical instability causing blowups. should depend on matrix size
        randm = randm - torch.diag(torch.diag(randm)) - torch.eye(hidden_size,device=device) / tau
        
        spectrum = torch.linalg.eigvals(randm)
        reals = torch.real(spectrum)
        enough_pos = torch.sum(reals >= 0)
        not_too_large = torch.max(reals) < 1
        
    return torch.nn.Parameter(randm)
    

#TODO: ADD SUPPORT FOR MORE THAN 1 RECURRENT LAYER
class LeakyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=False, tau = 1, activation = F.tanh):
        super(LeakyRNN, self).__init__()
        self.tau = tau
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden = None
        self.activation = activation
        self.readout = False
        
        self.input2h = nn.Linear(input_size, hidden_size, bias=bias) 
        self.h2h = nn.Linear(hidden_size, hidden_size,bias=bias) 
        # constrain diagonal of weight matrix (autapses) to be zero
        with torch.no_grad():
            self.h2h.weight = random_init(hidden_size,tau=tau, eps=None, gain=1/np.sqrt(2))
            self.input2h.weight = nn.init.kaiming_normal_(self.input2h.weight,nonlinearity='relu')
        parametrize.register_parametrization(self.h2h,"weight",IdentityDiagonal(self.tau))
        
        if bias != False:
            self.input2h.bias.data.uniform_(-bias, bias)
            self.h2h.bias.data.uniform_(-bias, bias)
            
    def init_hidden(self,input):
        batch_size = input[1]
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
     
    # here is our LEAKY modification
    # to keep the same structure but not leaky, add h_new = h_new + (1-1/self.tau)*hidden to the end
    def recurrence(self, input, hidden):
        h_new = (1-1/self.tau)*hidden + self.input2h(input) + self.h2h(self.activation(hidden))
        h_new = torch.clamp(h_new, min=-1e3,max=1e3)
        return h_new
   
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        # Stack together output from all time steps
        output = torch.cat(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden

class QRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, shared_layers = [], activation = F.relu, bias=3e-3, num_rnns=1, tau = 1, leaky=False, adaptive_tau=False, use_linear2=True):
        super(QRNN,self).__init__()
        self.tau = 1
        
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.shared_layers = shared_layers
        self.num_shared = len(shared_layers)
        self.activation = activation
        self.use_linear2 = use_linear2
        self.linear1 = nn.Linear(num_inputs+num_actions, hidden_dim)
        if leaky:
            self.rnn1 = LeakyRNN(hidden_dim, hidden_dim, num_layers = num_rnns, activation=activation, tau = tau)
        else:
            self.rnn1 = nn.RNN(hidden_dim, hidden_dim, num_layers = num_rnns, nonlinearity='relu',dropout=.1)
            nn.init.orthogonal_(self.rnn1.weight_ih_l0)
            nn.init.orthogonal_(self.rnn1.weight_hh_l0)
            self.rnn1.bias_ih_l0.data.uniform_(-bias, bias)
            self.rnn1.bias_hh_l0.data.uniform_(-bias, bias)
            
        if self.use_linear2:
            self.linear2 = nn.Linear(hidden_dim,hidden_dim)

        self.linear3 = nn.Linear(hidden_dim,1)

        nn.init.constant_(self.linear1.bias, 0)
        if self.use_linear2:
            nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear3.bias, 0)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        if self.use_linear2:
            nn.init.kaiming_normal_(self.linear2.weight,nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear3.weight,nonlinearity='relu')

        
    def forward(self, state, last_action, hidden_in):
        state = state.permute(1,0,2).to(device)
        last_action = last_action.permute(1,0,2).to(device)
        # action = action.permute(1,0,2).to(device)
        x = torch.cat([state, last_action], -1) 
        x = self.activation(self.linear1(x))
        x, rnn_hidden = self.rnn1(x, hidden_in)  # no activation after rnn
        if self.use_linear2:
            x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, rnn_hidden    # lstm_hidden is actually tuple: (hidden, cell) 
    
class PolicyRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, activation = F.relu, bias = 3e-3, num_rnns = 1, RL = False, alpha = .05, weighting = [1,0],
                 epsilon = .05, q_bias = 0, init_action = None, path = None, load = True, gamma = 0, scaletype = 'adaptive', leaky=False, tau = 1,env = None,
                 learn_RL = False, q_init = 0.0, fix_weights = True, use_linear2=True, norm_p=1, forgetting=False):
        super(PolicyRNN,self).__init__()
        
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_linear2 = use_linear2
        self.norm_p = norm_p
        
        self.linear1 = nn.Linear(num_inputs+2*num_actions, hidden_dim)
        if leaky:
            self.rnn1 = LeakyRNN(hidden_dim, hidden_dim, num_layers = num_rnns, activation=activation, tau = tau)
        else:
            self.rnn1 = nn.RNN(hidden_dim, hidden_dim, num_layers = num_rnns)
            nn.init.orthogonal_(self.rnn1.weight_ih_l0)
            nn.init.orthogonal_(self.rnn1.weight_hh_l0)
            self.rnn1.bias_ih_l0.data.uniform_(-bias, bias)
            self.rnn1.bias_hh_l0.data.uniform_(-bias, bias)
            
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        if self.use_linear2:
            self.linear2 = nn.Linear(hidden_dim,hidden_dim)

        nn.init.constant_(self.linear1.bias, 0)
        if self.use_linear2:
            nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear3.bias, 0)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        if self.use_linear2:
            nn.init.kaiming_normal_(self.linear2.weight,nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear3.weight,nonlinearity='relu')


        self.num_shared = 0
        
        self.scaletype = scaletype
        if scaletype == 'adaptive':
            self.alphas = weighting
            self.weights = MultLayerAdaptiveSimple(self.alphas)
            # parametrize.register_parametrization(self.weights,"weight",SumOne())
            for w in self.weights.weight:
                parametrize.register_parametrization(w,'weight',SumOne(norm_p=self.norm_p))
                if fix_weights:
                    w.requires_grad_(False)
        
        else:
            self.alpha = weighting
            # self.alpha=1/2
            self.weights = MultLayer(self.alpha)
            self.weights.weight = nn.Parameter(torch.tensor([self.alpha,1-self.alpha],dtype=torch.float32))
            parametrize.register_parametrization(self.weights,"weight",SumOne(norm_p=self.norm_p))
            if fix_weights:
                # Use a safer way to disable gradients for weights
                for param in self.weights.parameters():
                    param.requires_grad = False

        
        if RL:
            if forgetting:
                self.RL = rl_model.QAgentForgetting(alpha=alpha, delta = gamma, 
                                                epsilon=epsilon, bias=q_bias, init_action=init_action, path=path, load=load, env = env)
            else:  
                if isinstance(gamma,float) or not isinstance(gamma,Iterable):
                    self.RL = rl_model.QAgent(alpha=alpha, gamma = gamma, q_init=q_init,
                                                    epsilon=epsilon, bias=q_bias, init_action=init_action, path=path, load=load, env = env)
                else:
                        # alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32),requires_grad=True)
                        # gamma = nn.Parameter(torch.tensor(gamma,dtype=torch.float32),requires_grad=True) 
                    
                    self.RL = rl_model.QAgentAsymmetric(alpha=alpha, delta = gamma,  
                                                    epsilon=epsilon, bias=q_bias, init_action=init_action, path=path, load=load, env = env, 
                                                    learn_RL=learn_RL)
        else:
            self.RL = None
        
        
        # self.weights.requires_grad_(False)

    def forward(self, state, last_action, reward, hidden_in, softmax_dim=-1):
        #inputs -> linear 1 -> relu -> rnn -> linear2 -> relu -> linear 3 -> softmax
        
        if self.RL is not None:
            self.RL.forward(state,last_action,reward) 
            
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        reward = reward.permute(1,0,2)
        
        x = torch.cat([reward,state, last_action],-1)
        x = self.activation(self.linear1(x))
        x, hidden_out = self.rnn1(x, hidden_in)
        if self.use_linear2:
            x = self.activation(self.linear2(x))
        x = self.linear3(x)
        

        x = x.permute(1,0,2)

        probs = x.clone()


        return probs, hidden_out
    
    
    def fixed_point_recurrence(self, state, last_action,hidden_in):
        '''Helper function for processing inputs when looking for fixed points'''
        #inputs -> linear 1 -> relu -> rnn -> linear2 -> relu -> linear 3 -> softmax

        Qs = torch.zeros_like(state)

        if self.RL is not None:
            self.RL.forward(state,last_action) # no need to permute?
            Qs = self.RL.Qs
        
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        
        x = torch.cat([state, last_action],-1)
        x = self.activation(self.linear1(x))
        return self.rnn1.recurrence(x, hidden_in)
    
    def evaluate(self, state, last_action, reward, hidden_in, eps = 1e-4, done=False, deterministic=True):
        '''Generates an action given the inputs'''
        probs, hidden_out = self.forward(state,last_action,reward, hidden_in,softmax_dim=-1)
        z = (probs == 0.0).float() * eps
        log_probs = torch.log(F.softmax(probs,dim=-1) + z)
        scale = 1
        if deterministic:
            if self.RL is not None:
                Qs = self.RL.Qs
                log_probs = self.weights(log_probs,torch.log(F.softmax(Qs,dim=-1) + z),reward)
                probs = log_probs.exp() 
            else:
                probs = F.softmax(probs, dim = -1)      
                
        if not deterministic:  
            probs = F.softmax(probs, dim = -1)
            if self.RL is not None:
                Q_pfc = probs.detach()   
                Q_bg = F.softmax(self.RL.Qs.detach(), dim = -1)
                
                p_bg = 1-self.alpha
                bg_chooses = torch.rand((Q_pfc.shape[0],Q_pfc.shape[1])) <= p_bg
                bg_chooses = bg_chooses.unsqueeze(-1).float()

                probs = probs * (1-bg_chooses) + Q_bg * bg_chooses
    
            log_probs = torch.log(probs + z)

        new_action = torch.nn.functional.one_hot(torch.argmax(probs, axis=2),num_classes=probs.shape[-1])
        
            
        return new_action, log_probs, hidden_out
    
    def get_action(self, state, last_action, rewards, hidden_in, deterministic = True):
        num_dims = min(len(state.shape),len(last_action.shape))
        if num_dims < 3: #then we know we're doing one step at a time instead of based on batches, and can unsqueeze so that we can write general code
            state = torch.Tensor(state).squeeze().view(1,1,len(state)).to(device, dtype=torch.float32)
            last_action = torch.Tensor(last_action).squeeze().view(1,1,len(last_action)).to(device, dtype=torch.float32)
        else:
            state = torch.Tensor(state).to(device, dtype=torch.float32)
            last_action = torch.Tensor(last_action).to(device, dtype=torch.float32)
        actions, log_probs, hidden_out = self.evaluate(state, last_action, rewards, hidden_in, deterministic=deterministic)
        
        return actions.view(2).detach().cpu(), log_probs, hidden_out
        
#copies one network to another
def copy_network(original,copy):
    copy.load_state_dict(original.state_dict())

class RLRNN(nn.Module):
    def __init__(self, hidden_dim=32, max_episodes = 4096, max_steps = 150, batch_size = 32, 
                 update_itr = 1, DETERMINISTIC = True, environment = None, model_path = None, weighting = .5,
                 num_rnns = 1, activation = F.relu, RL = True, l1=.005, l2=.01, Qalpha = .05,Qgamma=0, scaletype='standard',gamma = .95, 
                 lambd = 0, fix_weights = True, LBFGS = False, lr = None, leaky_q = True, leaky_policy = True, leaky_tau = 1, 
                 policy_scale = 5, learn_RL = False, q_init = 0.0, use_linear2 = True, norm_p = 1, forgetting=False):
        passed_args = locals()
        del passed_args['self']

        super(RLRNN,self).__init__()
        self.RL = RL
        self.hidden_dim = hidden_dim
        self.max_episodes = max_episodes
        self.l1 = l1
        self.l2 = l2
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.update_itr = update_itr
        self.DETERMINISTIC = DETERMINISTIC
        self.gamma = gamma
        self.lambd = lambd
        self.LBFGS = LBFGS
        self.leaky_q = leaky_q
        self.leaky_policy = leaky_policy
        self.leaky_tau = leaky_tau 
        self.m1 = 0
        self.m2 = 0
        self.patience = 0
        self.training_iter = 0
        self.best_model = 0
        self.saved_iters = 0
        self.prev_max = -1
        self.min_ep = 0

        if environment is None:
            self.environment = envs.mp_env.MPEnv(fixed_length=True, reset_time=max_steps)
        else:
            self.environment = environment
        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'model','PFCBG_{}'.format(self.environment.opponent_name))
        else:
            self.model_path = model_path
        self.model_path = os.path.join(os.getcwd(), model_path)
        self.num_rnns = num_rnns
        self.activation = activation        
        self.activity = {}
        
        self.log_dir = None if model_path is None else os.path.dirname(model_path)
        self.model_name = None if model_path is None else os.path.basename(model_path)
        # self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.model_name)

        self.replay_buffer = ReplayMemory()
        self.state_dim = self.environment.observation_space.shape[0]
        self.action_dim = self.environment.action_space.n 
        
        
        self.Qalpha = Qalpha
        self.Qgamma = Qgamma
        self.policy = PolicyRNN(self.state_dim, self.action_dim, self.hidden_dim, leaky=leaky_policy, tau=leaky_tau, activation = self.activation,
                                weighting=weighting, scaletype=scaletype,num_rnns=self.num_rnns, RL = self.RL,q_init=q_init, alpha=Qalpha, gamma = Qgamma,
                                env=self.environment, fix_weights=fix_weights, use_linear2=use_linear2, norm_p=norm_p, forgetting=forgetting).to(device)
        self.sq1 = QRNN(self.state_dim, self.action_dim, self.hidden_dim, leaky=leaky_q, tau=leaky_tau, num_rnns=self.num_rnns, 
                        activation=self.activation, use_linear2=use_linear2).to(device)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        
        self.policy_scale = policy_scale
        self.num_updates = 0
        self.lr = lr
        
        if lr is not None:
            q_lr, policy_lr = lr
        else:
            q_lr, policy_lr = (5e-4,7e-5)        
        if self.LBFGS:
            # param_dict = [{'params': self.sq1.parameters(),'lr': q_lr}, {'params': self.policy.parameters(), 'lr': policy_lr}]
            # self.optimizer = optim.LBFGS(param_dict, lr=1e-3, max_iter=20, max_eval=20, history_size=100, line_search_fn='strong_wolfe')
            self.optimizer = optim.LBFGS([self.sq1.parameters(),self.policy.parameters()], lr=np.sqrt(q_lr*policy_lr), max_iter=20, max_eval=20, history_size=100, line_search_fn='strong_wolfe')
        else:
            self.sq1_optim = optim.AdamW(self.sq1.parameters(), lr=q_lr, amsgrad=True, eps=1e-4,weight_decay=1e-3)
            
            # Get all policy parameters except weights parameters
            weights_params = set(self.policy.weights.parameters())
            policy_params_without_weights = [p for p in self.policy.parameters() if p not in weights_params]
            
            self.policy_optim = optim.AdamW(policy_params_without_weights, lr=policy_lr, amsgrad=True, eps=1e-4, weight_decay=1e-4)
            if not fix_weights:
                self.policy_optim.add_param_group({'params': self.policy.weights.parameters(), 'lr': 1e-1*policy_lr})
            
            if learn_RL: #add BG parameters to optimizer
                self.policy_optim.add_param_group({'params': self.policy.RL.parameters(), 'lr':1e-1 *  policy_lr})
                # self.policy_optim.add_param_group({'params': self.policy.weights.parameters(), 'lr': 1e-1*policy_lr})
        
        #create folder
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        #save input params as dict so i don't need to know what was used a priori
        with open(os.path.join(self.model_path,'model_input_dict.pkl'), 'wb') as handle:
            pickle.dump(passed_args,handle)
            
            
            # self.sq1_optim = optim.SGD(self.sq1.parameters(), lr=q_lr)
            # self.policy_optim = optim.SGD(self.policy.parameters(),lr=policy_lr)
            
            # policy_params = list(self.policy.parameters())
            # mixing_weights = ['self.policy.weights.weight']
            # params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in mixing_weights, self.policy.named_parameters()))))
            # base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in mixing_weights, self.policy.named_parameters()))))
            # policy_lrs = [{'params': base_params, 'lr': 20*policy_lr}, {'params': params, 'lr': policy_lr}]
            # # policy_lrs = [{'params': policy_params[i], 'lr': policy_lr} for i in range(len(policy_params))]
            # # policy_lrs[]
            # self.policy_optim = optim.AdamW(policy_lrs, amsgrad=True, eps=1e-4, weight_decay=1e-4)
       
    
    def forward(self, state, last_action, reward, hidden_in):
        new_action, log_prob, hidden_out = self.policy.evaluate(state, last_action, reward, hidden_in, deterministic=self.DETERMINISTIC)
        values,_ =self.sq1(state, last_action, hidden_in)
        # if self.policy.RL is not None:
        #     self.policy.RL.reset()
        return new_action, log_prob, values, hidden_out
        
    def compute_entropy(self, log_probs):
        return - (log_probs.exp() * log_probs).mean()
    
    def compute_avg_entropy(self, log_probs):
        return - (log_probs.exp() * log_probs).sum() / (log_probs.shape[0] * log_probs.shape[1])
    
    
    # def calculate_returns(self,rewards, dones, gamma):
    #     result = np.empty_like(rewards)
    #     result[-1] = rewards[-1]
    #     for t in range(len(rewards)-2, -1, -1):
    #         result[t] = rewards[t] + gamma*(1-dones[t])*result[t+1]
    #     return result

    # def calculate_advantages(self,TD_errors, lam, gamma):
    #     result = np.empty_like(TD_errors)
    #     result[-1] = TD_errors[-1]
    #     for t in range(len(TD_errors)-2, -1, -1):
    #         result[t] = TD_errors[t] + gamma*lam*result[t+1]
    #     return result

    # MY IMPLEMENTATION
    def compute_returns_gae(self,rewards, dones, values, next_values, gamma, lambd, normalize = False):
        values = values.detach()
        next_values = next_values.detach()
        returns = values[:,-1,:]
        advantages = 0
        
        all_returns = torch.zeros((rewards.shape[0],rewards.shape[1], rewards.shape[2]))
        all_advantages = torch.zeros((values.shape[0],values.shape[1], values.shape[2]))
        
        for t in reversed(range(all_returns.shape[1])):
            mask = 1-dones[:,t,:]
            returns = rewards[:,t,:] + returns * gamma * mask
            deltas = rewards[:,t,:] + next_values[:,t,:] * gamma * mask - values[:,t,:]
            advantages = advantages * gamma * lambd * mask + deltas
            all_returns[:,t,:] = returns
            all_advantages[:,t,:] = advantages
            
        if normalize:
            all_advantages = (all_advantages - all_advantages.mean()) / all_advantages.std()
            all_returns = (all_returns - all_returns.mean()) / all_returns.std()
        target = all_advantages + values
        return target, all_advantages


    
    def update(self):
        # torch.autograd.set_detect_anomaly(True)
        activity = []
        policy_Qs = []
        BG_Qs = []
        activity_Q1 = []
        activity_Q2 = []
        
        
        def getActivityQ1(model, input, output):
            activity_Q1.append(output[0].detach())
        
        def getActivity(model, input, output):
            activity.append(output[0].detach())
        
        def getQPolicy(model, input,output):
            policy_Qs.append(output.detach())
        def getQBG(model,input,output):
            BG_Qs.append(F.softmax(self.policy.RL.Qs,dim=-1))
            
            
        l1_loss = 0
        l2_loss = 0
        
        policy, value,state, action, last_action, next_state, reward, last_reward, done, hidden_in, hidden_out = self.replay_buffer.sample(self.batch_size)
        
        # value      = torch.FloatTensor(np.array(value)).to(device)
        # state      = torch.FloatTensor(np.array(state)).to(device)
        # next_state = torch.FloatTensor(np.array(next_state)).to(device)
        # action     = torch.FloatTensor(np.array(action)).to(device)
        # last_action     = torch.FloatTensor(np.array(last_action)).to(device)
        # reward     = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        # done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
 
        policy     = torch.cat(policy).to(device)
        value      = torch.cat(value).to(device)
        state      = torch.cat(state).to(device)
        next_state = torch.cat(next_state).to(device)
        action     = torch.cat(action).to(device)
        last_action= torch.cat(last_action).to(device)
        reward     = torch.cat(reward).to(device)
        last_reward = torch.cat(last_reward).to(device)
        done       = torch.cat(done).to(device)
        
        #add hooks
        hsq1 = self.sq1.rnn1.register_forward_hook(getActivityQ1)
        hpolicy1 = self.policy.rnn1.register_forward_hook(getActivity)
        
        
        
        #reset Qs before doing anything      
        if self.policy.RL is not None:
            self.policy.RL.reset(keep_first=False)
            Qp = self.policy.linear3.register_forward_hook(getQPolicy)
            Qbg =self.policy.linear3.register_forward_hook(getQBG)
        
        new_action, log_prob, values, _ = self.forward(state,last_action, last_reward, hidden_in.detach())
        next_new_action, next_log_prob, next_values, _ = self.forward(next_state, new_action, (reward*last_action).detach().type(torch.float32), hidden_out.detach()) #right hidden state?
        
        # new_action, log_prob, values, hidden_out = self.forward(next_state,action, hidden_in)
        #remove hooks
        hsq1.remove()
        hpolicy1.remove()
        if self.policy.RL is not None:
            Qp.remove()
            Qbg.remove()
            
            policy_Qs = torch.vstack(policy_Qs).permute(1,0,2)
            BG_Qs = torch.vstack(BG_Qs)

        if self.leaky_q == True: 
            l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.h2h.weight)) + self.l1*torch.mean(torch.abs(self.policy.rnn1.h2h.weight))
            # l2_q1 = self.l2 * torch.mean(self.sq1.rnn1.h2h.weight**2)  
            l2_q1 = self.l2*torch.mean(torch.vstack(activity)**2) + self.l2*torch.mean(torch.vstack(activity_Q1)**2)
        else:    
            l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.weight_hh_l0)) + self.l1*torch.mean(torch.abs(self.policy.rnn1.weight_hh_l0))
            # l2_q1 = self.l2 * torch.mean(self.sq1.rnn1.weight_hh_l0**2) + self.l2*torch.mean(torch.vstack(activity_Q1)**2)  * 5
            l2_q1 = self.l2*torch.mean(torch.vstack(activity)**2) + self.l2*torch.mean(torch.vstack(activity_Q1)**2)

        
        # if self.leaky_q == True:
        #     l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.h2h.weight))
        #     l2_q1 = self.l2*torch.mean(torch.vstack(activity_Q1)**2)  
        # else:    
        #     l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.weight_hh_l0))
        #     l2_q1 = self.l2*torch.mean(torch.vstack(activity_Q1)**2)  
        
        
        # l1_q1 = 0
        # l2_q1 = 0
        
        target, advantage = self.compute_returns_gae(reward, done, values.detach(), next_values.detach(), self.gamma, self.lambd)
        # advantage = self.compute_returns_gae(reward, done, values.detach(), self.gamma, self.lambd)        
        target = target.detach()        
        
        #TODO: TEST CODE SNIPPET BELOW
        # returns = (returns - returns.mean()) / (returns.std() + 1e-4)
        
        # q1_loss = 0.5* self.soft_q_criterion1(values, returns) + l1_q1 + l2_q1 
        # q1_loss = 0.5*(target-value).pow(2).mean() + l1_q1 + l2_q1 
        
        q1_loss = 0.5*(target-values).pow(2).mean() + l1_q1 + l2_q1 

        #train the Actor (Policy/π) network        
        
        # a = torch.argmax(action, dim=-1).unsqueeze(-1)
        # pi_a= log_prob.gather(-1,a)
        # pi_a = log_prob[:,:-1,:]
        # policy_loss = -(pi_a*advantage.detach()).mean()
                # advantage = advantage.detach()
                
        advantage = advantage.detach()
        logits = (policy * action.detach()).sum(-1).unsqueeze(-1)
        # logits = logits[:,:-1,:]
        policy_loss = -(logits * advantage).mean()
        
        
        # policy_loss = -(log_prob*advantage.detach()).mean()

        # add l1 and l2 component on recurrent layer to loss
        if self.leaky_policy:
            l1_loss = self.l1 * torch.mean(torch.abs(self.policy.rnn1.h2h.weight)) # sparse weights
        else:
            l1_loss = 0
        l2_loss += torch.mean(torch.vstack(activity)**2) * self.l2 # distributed activity 
        # BG_reg = 0
        # # tries to make sure magnitudes of Qs are similar for PFC and BG. Regularizing term
        # if self.policy.RL is not None:
        #     BG_reg = torch.mean((policy_Qs)**2 - torch.abs(BG_Qs)**2)
        # BG_reg = BG_reg.detach() * self.l2  * 1/20
        # policy_loss = policy_loss + l1_loss + l2_loss + BG_reg
        policy_loss = policy_loss + l1_loss + l2_loss 

        
        #compute entropy of policy. eNTROPY Should be negative to boost exploration, so subtract from total loss
        # entropy = (log_prob.exp()*(-1*log_prob)).mean() 
        entropy = self.compute_entropy(log_prob)
        
        #total loss
        entropy_scale = .0001 + (.1- .0001) * \
                    np.exp(-1. * self.num_updates / 100)
        ac_loss = self.policy_scale*policy_loss + q1_loss - entropy * entropy_scale
        
        torch.nn.utils.clip_grad_norm_(self.sq1.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)

        
        self.sq1_optim.zero_grad()
        self.policy_optim.zero_grad()
        ac_loss.backward()
        self.policy_optim.step()
        self.sq1_optim.step()      
            
        self.num_updates += 1
        print((q1_loss.item(), self.policy_scale * policy_loss.item()))
        # print(self.policy.weights.weight)
        return q1_loss.item(), self.policy_scale * policy_loss.item()
    
    def update_LBFGS(self):
        activity = []
        activity_Q1 = []
        
        def getActivityQ1(model, input, output):
            activity_Q1.append(output.detach())
        
        def getActivity(model, input, output):
            activity.append(output[0].detach())
        
        state, action, last_action, next_state, reward, done, hidden_in, hidden_out, = self.replay_buffer.sample(self.batch_size)
        
        state      = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action     = torch.FloatTensor(np.array(action)).to(device)
        last_action     = torch.FloatTensor(np.array(last_action)).to(device)
        reward     = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
 
        #add hooks
        hsq1 = self.sq1.linear3.register_forward_hook(getActivityQ1)
        hpolicy1 = self.policy.rnn1.register_forward_hook(getActivity)
        
        #reset Qs before doing anything      
        if self.policy.RL is not None:
            self.policy.RL.reset(keep_first=False)
        
        new_action, log_prob, values, hidden_out = self.forward(state,last_action, hidden_in)
        
        #remove hooks
        hsq1.remove()
        hpolicy1.remove()

        if self.leaky_q == True:
            l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.h2h.weight))
            l2_q1 = self.l2 * torch.mean(self.sq1.rnn1.h2h.weight**2) + self.l2*torch.mean(torch.vstack(activity_Q1)**2)  * 5
        else:    
            l1_q1 = self.l1 * torch.mean(torch.abs(self.sq1.rnn1.weight_hh_l0))
            l2_q1 = self.l2 * torch.mean(self.sq1.rnn1.weight_hh_l0**2) + self.l2*torch.mean(torch.vstack(activity_Q1)**2)  * 5
        
        

        def closure():
            self.sq1_optim.zero_grad()
            values, _ = self.sq1(state, last_action, hidden_in)
            q1_loss = 0.5* self.soft_q_criterion1(values, self.compute_returns_gae(reward,done,values.detach(),self.gamma,0)) + l1_q1 + l2_q1 
            q1_loss.backward()
            new_action, log_prob, values, hidden_out = self.forward(state,last_action, hidden_in)
            advantage = self.compute_returns_gae(reward, done, values.detach(), self.gamma, self.lambd)
            policy_loss = -(log_prob*advantage).mean()
            l1_loss = self.l1 * torch.mean(torch.abs(self.policy.rnn1.h2h.weight)) # sparse weights
            l2_loss += torch.mean(torch.vstack(activity)**2) * self.l2 # distributed activity 
            policy_loss = policy_loss + l1_loss + l2_loss
            entropy = (log_prob.exp()*(-1*log_prob)).mean()
            policy_loss = policy_loss + entropy * .0001
            ac_loss = policy_loss + q1_loss 
            ac_loss.backward()
            return ac_loss
        
        
        torch.nn.utils.clip_grad_norm_(self.sq1.parameters(), .5)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), .5)

        op = self.optimizer.step(closure).item()
        
        print(op)
        return op
        
    def evaluate_model(self,env):
        '''Runs one episode of the environment'''
        # env = copy.deepcopy(self.environment)
        # env.clear_data()
        
        
        if self.policy.RL is not None:
            self.policy.RL.reset()
            
        episode_state = []
        episode_action = []
        episode_last_action = []
        episode_reward = []
        episode_next_state = []
        episode_done = []
        episode_value = []
        episode_policy = []
        episode_last_reward = []
        state =  env.reset()
        if isinstance(state,int):
            state = torch.Tensor([state])
        if isinstance(state,np.ndarray):
            state = torch.Tensor(state).view(1,1,-1)
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=self.action_dim).view(1,1,-1)
        
        reward = 0
        if isinstance(self.environment,envs.mp_env.MPEnv):
            reward = (np.random.rand() > .5) 
        hidden_out = torch.zeros([self.num_rnns,1, self.hidden_dim], dtype=torch.float,device=device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        # hidden_out = torch.randn([self.num_rnns,1, self.hidden_dim], dtype=torch.float,device=device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             

        hidden_out = hidden_out.to(device)
        
        if self.policy.RL is not None: 
            self.policy.RL.reset()
        for step in range(self.max_steps):
            # last_action = action
            hidden_in = hidden_out
            last_reward = reward * last_action
            action, policy, value, hidden_out = self.forward(state, last_action, last_reward, hidden_in) # MAKE IT TAKE PREVIOUS REWARD FOR BG 
            next_state, reward, done, _ = env.step(action)
            action = action.to(torch.float32)
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
            episode_last_reward.append(last_reward)
            episode_next_state.append(next_state)
            episode_done.append(done) 
            episode_value.append(value)
            episode_policy.append(policy)

            
                
            state = next_state
            last_action = action
            if done:
                break
        episode_state = torch.cat(episode_state,dim=1)
        episode_action = torch.cat(episode_action, dim=1)
        episode_last_action = torch.cat(episode_last_action, dim=1)
        episode_last_reward = torch.cat(episode_last_reward, dim=1)
        episode_reward = torch.tensor(episode_reward).requires_grad_(False).view(1,-1,1)
        episode_next_state = torch.cat(episode_next_state, dim=1)
        episode_done = torch.tensor(np.float32(episode_done)).view(1,-1,1)
        episode_value = torch.cat(episode_value, dim=1)
        episode_policy = torch.cat(episode_policy, dim=1)
        self.replay_buffer.push(episode_policy,episode_value,episode_state, episode_action, episode_last_action, episode_next_state, episode_reward, episode_last_reward,
                episode_done, ini_hidden_in, ini_hidden_out)

        
        return torch.mean(episode_reward.to(torch.float32))
    
    def test_model(self, nits = 1000):
        '''Runs test suite on environment'''
        rewards = []
        for i in range(nits):
            rewards.append(self.evaluate_model(self.environment))
        return np.mean(rewards)
    
    def optimize_model(self, nits = False, save=False, max_patience = 1000, prune = True, save_every = 100, cleanup_every = 200):
        ''' optimizes model, then loads best model and runs test suite, saving the figures'''
        self.policy.train()
        self.sq1.train()
        # self.sq2.train()
        if nits == False:
            nits = self.batch_size*8
        losses = []
        
        env = self.environment
        rewards = []

        
        # Calculate how many episodes we still need to train
        episodes_remaining = self.max_episodes - self.training_iter
        
        for eps in range(episodes_remaining):
            batch_rewards = []
            self.replay_buffer.reset()
            
            for b in range(nits):
                batch_rewards.append(self.evaluate_model(env))
            
            #trying parallel
            # batch_rewards = Parallel(n_jobs = 8,prefer="threads")(delayed(self.evaluate_model)(copy.deepcopy(env))
            #                                           for _ in range(nits))
            
            rewards.append(np.mean(batch_rewards))
            # self.writer.add_scalar('Rewards/train',np.array(rewards[-1]),training_iter)
            self.policy.train()
            self.sq1.train()
            # self.sq2.train()
            if (rewards[-1]) >= self.prev_max:
                self.prev_max = rewards[-1]
                if self.min_ep == 0: 
                    self.min_ep = self.training_iter
                else:
                    self.min_ep = self.training_iter
                self.save_model_ep(self.training_iter)
                self.best_model = self.training_iter     
            if self.training_iter % save_every == 0:
                self.save_model_ep(self.training_iter)
                
            # Periodically cleanup old models
            if self.training_iter % cleanup_every == 0 and self.training_iter > cleanup_every:
                self.selective_cleanup()
                
            self.training_iter+=1

            if self.training_iter > self.batch_size:
                if self.m1 == 0:
                    self.m1 = max(rewards[-int(self.batch_size):])
                self.m2 = rewards[-1]

                if self.m1 > self.m2:
                    self.patience += 1
                    if self.patience > max_patience:
                        # self.writer.close()
                        self.load_model_ep(self.best_model)
                        if prune == True:
                            self.prune_saves(rewards,0)
                        self.save_model(self.model_path)
                        if not save:
                            plt.plot(rewards)
                            plt.title('Rewards')
                            plt.xlabel('Episode')
                            plt.ylabel('Reward')
                            plt.axvline(x=self.best_model, color='r', linestyle='--')
                            plt.show()
                            plt.title('Policy Network Loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Episode')
                            plt.axvline(x=self.best_model, color='r', linestyle='--')
                            plt.plot(losses)
                            plt.show()
                        else:
                            plt.plot(rewards)
                            plt.title('Rewards')
                            plt.xlabel('Episode')
                            plt.ylabel('Reward')
                            plt.axvline(x=self.best_model, color='r', linestyle='--')
                            plt.savefig(os.path.join(self.model_path,'rewards.png'))
                            plt.clf()
                            plt.title('Policy Network Loss')
                            plt.ylabel('Loss')
                            plt.xlabel('Episode')
                            plt.plot(losses)
                            plt.axvline(x=self.best_model, color='r', linestyle='--')
                            plt.savefig(os.path.join(self.model_path,'policy.png'))
                            plt.clf()
                        break
                else: 
                    self.patience = 0
                    self.m1 = self.m2
            
            if self.LBFGS:
                losses.append(self.update_LBFGS()[1])
            else:
                
                losses.append(sum(self.update())) 
            print('Episode: ', self.training_iter, '| Episode Reward: ', rewards[-1], '| Episode Length: ', self.max_steps)

        # return self.test_eval
    
    def selective_cleanup(self, path='', keep_interval=5):
        """
        Selectively delete model checkpoints, keeping only:
        a) The best model (self.best_model)
        b) The current highest saved model (self.training_iter) 
        c) Recent models (within keep_interval of current)
        d) Models at regular intervals for debugging
        """
        if path == '':
            path = self.model_path
            
        # Handle case where model_name might be None or problematic
        if self.model_name is None:
            # Use basename of model_path as fallback
            model_name = os.path.basename(path) if path else 'RLRNN'
        else:
            model_name = self.model_name
            
        print(f"Cleanup: Looking for files in path: {path}")
        print(f"Cleanup: Using model_name: {model_name}")
        
        # Get all saved model files with more robust pattern matching
        actor_pattern = os.path.join(path, f'{model_name}_actor_*')
        critic_pattern = os.path.join(path, f'{model_name}_critic1_*')
        
        print(f"Cleanup: Actor pattern: {actor_pattern}")
        print(f"Cleanup: Critic pattern: {critic_pattern}")
        
        actor_files = glob.glob(actor_pattern)
        critic_files = glob.glob(critic_pattern)
        
        print(f"Cleanup: Found {len(actor_files)} actor files, {len(critic_files)} critic files")
        
        # Extract episode numbers from filenames
        saved_episodes = set()
        for f in actor_files:
            try:
                # Extract the episode number from the filename
                filename = os.path.basename(f)
                ep_num = int(filename.split('_')[-1])
                saved_episodes.add(ep_num)
            except (ValueError, IndexError) as e:
                print(f"Cleanup: Warning - couldn't parse episode number from {f}: {e}")
                continue

        print(f"Cleanup: Found saved episodes: {sorted(saved_episodes)}")
                
        # Determine which episodes to keep
        episodes_to_keep = set()
        
        # Always keep the best model
        if self.best_model > 0:
            episodes_to_keep.add(self.best_model)
            print(f"Cleanup: Keeping best model: {self.best_model}")
            
        # Keep current/recent models (within keep_interval)
        current_ep = self.training_iter
        for ep in saved_episodes:
            if ep >= current_ep - keep_interval:
                episodes_to_keep.add(ep)
                
        print(f"Cleanup: Episodes to keep: {sorted(episodes_to_keep)}")
 
        # Delete files for episodes not in keep list
        episodes_to_delete = saved_episodes - episodes_to_keep
        
        print(f"Cleanup: Episodes to delete: {sorted(episodes_to_delete)}")
        
        deleted_count = 0
        for ep in episodes_to_delete:
            # Delete actor file
            actor_file = os.path.join(path, f'{model_name}_actor_{ep}')
            if os.path.exists(actor_file):
                try:
                    os.remove(actor_file)
                    deleted_count += 1
                    print(f"Cleanup: Deleted {actor_file}")
                except OSError as e:
                    print(f"Cleanup: Error deleting {actor_file}: {e}")
                
            # Delete critic file  
            critic_file = os.path.join(path, f'{model_name}_critic1_{ep}')
            if os.path.exists(critic_file):
                try:
                    os.remove(critic_file)
                    deleted_count += 1
                    print(f"Cleanup: Deleted {critic_file}")
                except OSError as e:
                    print(f"Cleanup: Error deleting {critic_file}: {e}")
                
        print(f"Cleanup: Successfully deleted {deleted_count} files ({len(episodes_to_delete)} episodes), kept {len(episodes_to_keep)} episodes")

    def cluster_test(self, nits = False,save='', save_img_separate = '', max_patience = 1000, suite_nits = 1000, prune = True, cleanup_every = 200):
        if save == '':
            save = self.model_path
        if not os.path.exists(save):
            os.makedirs(save)
                
        self.optimize_model(nits=nits,save=save, max_patience=max_patience, prune = prune, cleanup_every=cleanup_every)

        if len(save_img_separate) > 0:
            if not os.path.exists(save_img_separate):
                os.makedirs(save_img_separate)
            save_img_separate = os.path.join(save_img_separate,self.model_name)
            test_suite.test_suite(self,self.environment,order = 12,save=save_img_separate, nits = suite_nits)
        else:
            test_suite.test_suite(self,self.environment,order = 12,save=save, nits = suite_nits)   
    
    
          
    def save_model(self, path= '', exact = False):
        if path is None or path == '':
            path = self.model_path
        # path = os.path.dirname(os.path.join(os.getcwd(),self.model_path))
        if not os.path.exists(path):
            os.makedirs(path)
        if exact == False:
            torch.save(self.sq1.state_dict(), os.path.join(path,self.model_name+'_critic1'))
            # torch.save(self.sq2.state_dict(), os.path.join(path,self.model_name,self.model_name+'_critic2'))
            torch.save(self.policy.state_dict(), os.path.join(path,self.model_name+ '_actor'))
        else:
            torch.save(self.sq1.state_dict(), path+'_critic1')
            torch.save(self.sq2.state_dict(), path+'_critic2')
            torch.save(self.policy.state_dict(), path+ '_actor')

    def load_model(self, path,exact=False):
        # path = os.path.dirname(os.path.join(os.getcwd(),self.model_path))
        if path is None or path == '':
            path = self.model_path
        if exact == False:
            self.sq1.load_state_dict(torch.load(os.path.join(path,self.model_name+'_critic1')),strict=False)
            # self.sq2.load_state_dict(torch.load(os.path.join(path,self.model_name,self.model_name+'_critic2')))
            self.policy.load_state_dict(torch.load(os.path.join(path,self.model_name+'_actor')),strict=False)
        else:
            self.sq1.load_state_dict(torch.load(path+'_critic1'))
            # self.sq2.load_state_dict(torch.load(path+'_critic2'))
            self.policy.load_state_dict(torch.load(path+'_actor'))
        self.sq1.eval()
        # self.sq2.eval()
        self.policy.eval()
    
    def save_model_ep(self, ep, path = ''):
        if path is None or path == '':
            path = self.model_path
        # path = os.path.dirname(os.path.join(os.getcwd(),self.model_path))
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.sq1.state_dict(), os.path.join(path,self.model_name + '_critic1_{}'.format(int(ep))))
        # torch.save(self.sq2.state_dict(), os.path.join(path,self.model_name,self.model_name + '_critic2_{}'.format(int(ep))))
        torch.save(self.policy.state_dict(), os.path.join(path,self.model_name+'_actor_{}'.format(int(ep))))
        # save the variables used in optimization in one json file
        with open(os.path.join(path,self.model_name+'_training_state.json'),'w') as f:
            json.dump({
                'm1': float(self.m1),
                'm2': float(self.m2),
                'patience': int(self.patience),
                'prev_max': float(self.prev_max),
                'min_ep': int(self.min_ep),
                'training_iter': int(self.training_iter),
                'best_model': int(self.best_model),
                'saved_iters': int(self.saved_iters)
            },f)
    
    
    # def prune_saves(self,test_rewards, min_episode, path = ''):
    #     # self.writer.flush()
    #     # self.writer.close()
    #     tr = sum([np.mean(i) for i in test_rewards])
    #     max_rew = np.argmax(tr)
        
    #     if path is None or path == '':
    #         path = self.model_path
            
    #     self.saved_iters = max_rew + min_episode
        
    #     self.load_model_ep(self.saved_iters,path=path)
    #     # files = glob.glob(os.path.join(path,'*'))
    #     files = glob.glob(os.path.join(path,self.model_name,self.model_name+'_*'))
    #     for f in files:
    #         if os.path.isfile(f):
    #             os.remove(f)
    #     self.save_model(path)
    #     # with open(os.path.join(path,'model.pkl'),'wb') as file:
    #     # with open(os.path.join(path,self.model_name,self.model_name + '_model.pkl'),'wb') as file:
    #     #     pickle.dump(self,file)
    
    # optimize already loads best so try not loading, just deleting
    def prune_saves(self,test_rewards, min_episode, path = ''):
        files = glob.glob(os.path.join(path,self.model_name,self.model_name+'_*'))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
        files = glob.glob(os.path.join(path,self.model_name+'_*'))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
        # self.save_model(path)
        # with open(os.path.join(path,'model.pkl'),'wb') as file:
        # with open(os.path.join(path,self.model_name,self.model_name + '_model.pkl'),'wb') as file:
        #     pickle.dump(self,file
    def load_model_ep(self, ep, path = ''):
        if path is None or path == '':
            path = self.model_path
        # path = os.path.dirname(os.path.join(os.getcwd(), 'MP-RNN',self.model_path))
        self.sq1.load_state_dict(torch.load(os.path.join(path,self.model_name + '_critic1_{}'.format(int(ep))),map_location=device),strict=False)
        # self.sq2.load_state_dict(torch.load(os.path.join(path,self.model_name,self.model_name + '_critic2_{}'.format(int(ep))),map_location=device))
        self.policy.load_state_dict(torch.load(os.path.join(path,self.model_name+'_actor_{}'.format(int(ep))),map_location=device),strict=False)

        self.sq1.eval()
        # self.sq2.eval()
        self.policy.eval()
        if os.path.exists(os.path.join(path,self.model_name+'_training_state.json')):
            # load the variables as how  we saved it in the json file
            with open(os.path.join(path,self.model_name+'_training_state.json'),'r') as f:
                training_state = json.load(f)
            self.m1 = training_state['m1']
            self.m2 = training_state['m2']
            self.patience = training_state['patience']
            self.prev_max = training_state['prev_max']
            self.min_ep = training_state['min_ep']
            # Set training_iter to the episode we're loading from (not from JSON)
            # This ensures training continues from the correct episode
            self.training_iter = int(ep)
            self.best_model = training_state['best_model'] 
            self.saved_iters = training_state['saved_iters']
            
    
    def reset_optim(self):
        self.sq1_optim = optim.AdamW(self.sq1.parameters(), lr=self.lr[0], amsgrad=True, eps=1e-4,weight_decay=1e-3)
        self.policy_optim = optim.AdamW(self.policy.parameters(),lr=self.lr[1], amsgrad=True, eps=1e-4, weight_decay=1e-4)
        
    # def test_model(self, niters=None):
    #     rolling_avg = 0
    #     # env = envs.mp_env.MPEnv(fixed_length=True)
    #     env = self.environment
    #     # env = envs.mp_env.MPEnv(show_opp=False,train=False, fixed_length=True)
        
    #     accs = []
        
        
    #     if niters is None:
    #         niters = self.batch_size
        
    #     # self.load_model(self.model_path)
    #     for eps in range(niters):
            
    #         state =  env.reset()
    #         last_action = env.action_space.sample()
    #         last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2)
    #         episode_state = []
    #         episode_action = []
    #         episode_last_action = []
    #         episode_reward = []
    #         er = 0
    #         episode_next_state = []
    #         episode_done = []
    #         hidden_out = torch.zeros([self.num_rnns,1, self.hidden_dim], dtype=torch.float).to(device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
            
    #         for step in range(self.max_steps):
    #             hidden_in = hidden_out

    #             action, _, hidden_out = self.policy.get_action(state, last_action, hidden_in, deterministic = self.DETERMINISTIC)
    #             next_state, reward, done, _ = env.step(action)
    #             # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2)

    #             if step == 0:
    #                 ini_hidden_in = hidden_in
    #                 ini_hidden_out = hidden_out
    #             episode_state.append(state)
    #             episode_action.append(action.numpy())
    #             episode_last_action.append(last_action)
    #             episode_reward.append(reward)
    #             episode_next_state.append(next_state)
    #             episode_done.append(done) 

    #             env.render()   


    #             er += reward
    #             state=next_state
    #             last_action = action
    #         accs.append(np.sum(er)/(step+1))
    #         print('Episode: ', eps, '| Episode Reward: ', np.sum(er)/(step+1))
    #         rolling_avg = (rolling_avg*(eps) + np.sum(er)/step)/(eps+1)
    #     print(rolling_avg)
    #     accs = np.array(accs)
    #     print('Win Rate: {}, std: {}'.format(accs.mean(),accs.std()))
    #     #plot the accs as a distribution? and break up into strategy?

    # #generate trials, then use it to test
    # def test_eval(self):
    #     sys.stdout = open(os.devnull, 'w')        
    #     env = copy.deepcopy(self.environment)
    #     env.reset()
    #     rewards = [] 
        
    #     for eps in range(3*self.batch_size):
    #         state =  env.reset()
    #         episode_reward = 0
            
    #         last_action = env.action_space.sample()
    #         last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2).numpy()
    #         hidden_out = torch.zeros([self.num_rnns,1, self.hidden_dim], dtype=torch.float,device=device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
    #         for step in range(self.max_steps):
    #             hidden_in = hidden_out                
    #             action, _, hidden_out = self.policy.get_action(state, last_action, hidden_in, deterministic = self.DETERMINISTIC)
    #             next_state, reward, done, _ = env.step(action)
    #             # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
    #             state = next_state
    #             last_action = action
    #             episode_reward += reward
    #             if done:
    #                 break
    #         rewards.append((episode_reward)/(step+1))
    #     sys.stdout = sys.__stdout__
    #     return torch.Tensor(rewards), torch.std(torch.Tensor(rewards))
                
