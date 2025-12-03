'''
Q-learning agent implementing the RL (Reinforcement Learning) component in our project.
'''
import numpy as np
import torch 
import pickle
import envs.mp_env
import torch.nn as nn

rng = np.random.default_rng(28980)

mac = False
device_idx = 0
if mac: 
    device = torch.device("mps")
else :
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

#FOR TESTING PURPOSES
device = torch.device("cpu")


class QAgent(nn.Module):
    '''
    Basic Q-learning agent sans exploration
    '''
    def __init__(self, alpha = 0.05, gamma = 0, init_action = None, bias = 0, epsilon=.05, path = None, load = True, temperature = 1,deterministic=True, env = None,q_init = 0.0):
        super(QAgent,self).__init__()
        self.temp = torch.ones(1,dtype=torch.float32, requires_grad=True, device=device) * temperature
        self.path = path
        self.load = load
        self.alpha = alpha
        self.gamma = gamma
        self.deterministic = deterministic
        # self.state = state
        self.Qs = torch.full((1,1,2), q_init, requires_grad=False,device = device,dtype=torch.float)
        # self.Qs = torch.zeros_like(state,requires_grad=False,dtype=torch.int)
        # self.last_action = last_action
        # self.rewards = np.zeros(self.states.shape[0],self.states.shape[1],1)
        # self.rewards = torch.zeros_like(state,requires_grad=False,dtype=torch.int)

        self.pchooseright = 0.5 #default starting prob
        self.bias = bias
        self.epsilon = epsilon
        self.pchooseright = 0.5
        # self.act_hist = [self.prev_action]
        self.biasinfo = None
        self.max_choices = []
        self.enviroment = env

        if isinstance(env, envs.mp_env.MPEnv):
            self.RPEupdate = self.RPEupdateMP
        else:
            # raise ValueError('Environment not recognized')
            self.RPEupdate = self.RPEupdateBandit

        # if (self.path is not None) and (self.load == True):
        #     self.load_model()
             
    def __str__(self):
        return 'Batching-enabled Q Learner'

    #only want this to run once
    def RPEupdateMP(self, state,last_action, rewards = None):
        """Reward Prediction Error Calculation and Update of Q values"""
        rewards = torch.zeros_like(torch.Tensor(state),requires_grad=False,dtype=torch.long)
        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                a =  last_action.argmax(axis=-1)
                rewards[seq,i,a] = 2*(state[seq,i][1] == last_action[seq,i][1])-1                
                self.Qs[seq,i,a] = self.Qs[seq,i-1,a] * (1-self.alpha) + self.alpha*(rewards[seq,i,a] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
        #updates all action state pairs instead of just the 
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         rewards[seq,i,int(last_action[seq,i][1])] = 2*(state[seq,i][1] == last_action[seq,i][1])-1
        #         self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    def RPEupdateBandit(self, state,last_action, rewards):
        """Reward Prediction Error Calculation and Update of Q values"""
        # rewards = torch.zeros_like(torch.Tensor(last_action),requires_grad=False,dtype=torch.int)
        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                a =  last_action.argmax(axis=-1)
                rewards[seq,i,a] = 2*(rewards[seq,i,a])-1                
                self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    def RPEupdateRPS(self, state,last_action, reward):
        """Reward Prediction Error Calculation and Update of Q values"""
        rewards = torch.zeros_like(torch.Tensor(last_action),requires_grad=False,dtype=torch.float)
        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                a =  last_action.argmax(axis=-1)
                rewards[seq,i,a] = 2*(reward[seq,i])-1  
                self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    
    
    def get_last_saved(self):
        return self.action, self.pchooseright,self.biasinfo
    
    def save_model(self, path=None):
        if path is None:
            with open(self.path, "w+") as f:
                pickle.dump(self.__dict__,f)
        else:
            with open(path, "w+") as f:
                pickle.dump(self.__dict__,f)
                
    def load_model(self, path=None):
        if path is None:
            with open(self.path, "r+") as f:
                self.__dict__ = pickle.load(f)
        else:
            with open(path, "r+") as f:
                self.__dict__ = pickle.load(f)
        
        
    def forward(self,state,last_action, rewards = None):
        if last_action.shape != self.Qs.shape:
            self.Qs = torch.full_like(last_action, 0.5, requires_grad=False,dtype=torch.float)
        self.RPEupdate(state,last_action, rewards)
        # print(self.Qs)
        return self.Qs


    # #fully deterministic
    # def action(self):
    #     action_data = {'Qs':self.Qs,'bias':self.bias}
        
    #     self.ql, self.qr = self.Qs.detach().squeeze().cpu()

    #     self.max_choices.append(np.argmax(self.Qs.cpu()))
    #     self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp.detach().cpu()))
        
    #     if self.deterministic == True:
    #         action = 1 if self.pr >= .5 else 0
    #     else:   
    #         action = rng.choice([0,1], p=[1-self.pr, self.pr])

    #     return action, self.pr, action_data



    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
        
        # Handle both 2D and 3D Q-value tensors
        if self.Qs.dim() == 3:
            # For 3D tensor, use the last timestep (most recent)
            q_vals = self.Qs[0, -1, :]  # [batch=0, last_timestep, actions]
        elif self.Qs.dim() == 2:
            # For 2D tensor, use the last row (most recent)
            q_vals = self.Qs[-1, :]  # [last_timestep, actions]
        else:
            # For 1D tensor, use directly
            q_vals = self.Qs
        
        # Use softmax with temperature (like asymmetric and forgetting models)
        probs = torch.nn.functional.softmax(q_vals/self.temp,dim=-1).squeeze().cpu().detach().numpy()
        
        # Choose action based on whether we're deterministic or not
        if self.deterministic == False:
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(np.arange(0,len(q_vals),1))
            else:
                # choose index based on probabilities from probs
                action = np.random.choice(np.arange(0,len(q_vals),1),p=probs)
        else:
            # For deterministic case, still use temperature-based softmax but pick argmax
            action = np.argmax(probs)
        
        self.max_choices.append(action)
        self.pr = probs[action]
        
        return action, self.pr, action_data


    #MOSTLY DETERMINISTIC, CHOOSES SIDE WITH HIGHEST PROBABILITY
    # def action(self):
    #     action_data = {'Qs':self.Qs,'bias':self.bias}
        
        
    #     # self.ql, self.qr = self.Qs
    #     # self.max_choices.append(np.argmax(self.Qs))
    #     # self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp))
        
    #     # self.ql, self.qr = self.Qs.detach().cpu()
    #     self.ql, self.qr = self.Qs.detach().squeeze().cpu()

    #     self.max_choices.append(np.argmax(self.Qs.cpu()))
    #     self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp.detach().cpu()))
        
    #     #MAKE BIAS RANDOMLY DISTRIBUTED VARIABLE WITH MEAN BIAS and s.d. ?
    #     # work backwards: epsilon at 50/50 corresponds to what?
    #     # if at pr + bias, choose ~N(b,.02) for 5% deviation maximum?
    #     # note that for small %, It's roughly normal so it's a good approx
    #     # self.pr += self.bias 
    #     if self.deterministic == True:
    #         self.pr += np.random.normal(self.bias,self.epsilon/2)

    #         action = 1 if self.pr >= .5 else 0
    #     else:   
    #         action = rng.choice([0,1], p=[1-self.pr, self.pr])

    #     return action, self.pr, action_data
    
    def reset(self, keep_first=False):
        """
        Properly reset QAgent state without reinitializing the object.
        This preserves the tensor structure and properly resets Q-values.
        """
        if keep_first == False:
            # Reset Q-values to initial state based on current tensor structure
            if self.Qs.dim() == 1:
                # For 1D tensors (direct usage), initialize to [0.5, 0.5]
                self.Qs = torch.full((2,), 0.5, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 2:
                # For 2D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 3:
                # For 3D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            
            # Reset other state variables to initial values
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
        else:
            # Keep first implementation (existing logic) - also initialize to 0.5
            newQs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            # NEED TO ACCOUNT FOR BATCHES
            if self.Qs.dim() >= 2:
                newQs[:,0,:] = self.Qs[:,0,:]
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
            self.Qs = newQs.to(device)


    #action wrapper for testing WSLS behavior
    def get_action(self,state, last_action, rewards, hidden_in=None,deterministic=True):
        self.forward(state.reshape((1,1,-1)),last_action.reshape((1,1,-1)),rewards.reshape((1,1,-1))) # have to update for all code in PFC module as well
        action, pr, _ = self.action()
        action = torch.tensor(action)
        return torch.nn.functional.one_hot(action,last_action.shape[-1]), pr, hidden_in

class QAgentForgetting(QAgent):
    # In the forgetting model, timescale is not 1/alpha, it's 1/(1-alpha). to fix this, i changed the update a bit
    
    def __init__(self, alpha = .2, delta = [.15, -.08], init_action = None, bias = 0, epsilon=0, path = None,
                 load = True, temperature = 1,deterministic=True, env = None, learn_BG = False):
        super(QAgentForgetting,self).__init__(alpha , 0, init_action , bias, epsilon, path, load, temperature ,deterministic, env)
        self.delta = delta
        self.learn_BG = learn_BG
        self.temperature = temperature
        if self.learn_BG:
            self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float32),requires_grad=True)
            self.delta = nn.Parameter(torch.tensor(delta,dtype=torch.float32),requires_grad=True)

    
    def reset(self, keep_first=False):
        """
        Properly reset QAgent state without reinitializing the object.
        This preserves the tensor structure and properly resets Q-values.
        """
        if keep_first == False:
            # Reset Q-values to initial state based on current tensor structure
            if self.Qs.dim() == 1:
                # For 1D tensors (direct usage), initialize to [0.5, 0.5]
                self.Qs = torch.full((2,), 0, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 2:
                # For 2D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 3:
                # For 3D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0, requires_grad=False, device=device, dtype=torch.float)
            
            # Reset other state variables to initial values
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
        else:
            # Keep first implementation (existing logic) - also initialize to 0.5
            newQs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            # NEED TO ACCOUNT FOR BATCHES
            if self.Qs.dim() >= 2:
                newQs[:,0,:] = self.Qs[:,0,:]
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
            self.Qs = newQs.to(device)

    def RPEupdateMP(self, state,last_action, rewards = None):
        """Reward Prediction Error Calculation and Update of Q values"""
        # Ensure Qs has correct shape - handle both 2D (neural network) and 3D (standalone) cases
        if self.Qs.dim() == 2:
            # If 2D (for neural network compatibility), expand to 3D for processing
            if self.Qs.shape != (state.shape[0] * state.shape[1], 2):
                self.Qs = torch.full((state.shape[0] * state.shape[1], 2), 0.5, 
                                    requires_grad=False, device=device, dtype=torch.float)
            # Reshape to 3D for consistent processing
            Qs_3d = self.Qs.view(state.shape[0], state.shape[1], -1)
        else:
            # If not 3D, initialize properly with 0.5 (like forgetting model)
            if self.Qs.shape != (state.shape[0], state.shape[1], 2):
                self.Qs = torch.full((state.shape[0], state.shape[1], 2), 0.5, 
                                    requires_grad=False, device=device, dtype=torch.float)
            Qs_3d = self.Qs
        
        # Process updates on 3D tensor
        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                a = last_action[seq,i][-1].to(torch.int)
                rew = rewards[seq,i][a]

                # Fix delta indexing: use 0 for win (rew=1), 1 for loss (rew=0)
                delta_idx = 0 if rew == 1 else 1
                # Handle both tensor and list cases for self.delta
                if isinstance(self.delta, (list, tuple)):
                    delta = self.delta[delta_idx]
                else:
                    # If it's a tensor, access properly
                    delta = self.delta[delta_idx].item() if hasattr(self.delta, 'item') else self.delta[delta_idx]
                
                # Apply update to 3D tensor
                Qs_3d = Qs_3d * (1-self.alpha)
                Qs_3d[seq,i,a] += delta
        
        # Update self.Qs with the correct shape
        if self.Qs.dim() == 2:
            # Flatten back to 2D for neural network compatibility
            self.Qs = Qs_3d.view(-1, Qs_3d.shape[-1])
        else:
            self.Qs = Qs_3d
                
    def forward(self,state,last_action, rewards = None):
        self.RPEupdate(state,last_action, rewards)
        return self.Qs
    
    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
        
        # Handle both 2D and 3D Q-value tensors
        if self.Qs.dim() == 3:
            # For 3D tensor, use the last timestep (most recent)
            q_vals = self.Qs[0, -1, :]  # [batch=0, last_timestep, actions]
        elif self.Qs.dim() == 2:
            # For 2D tensor, use the last row (most recent)
            q_vals = self.Qs[-1, :]  # [last_timestep, actions]
        else:
            # For 1D tensor, use directly
            q_vals = self.Qs
        
        action = np.argmax(q_vals.cpu())
        self.max_choices.append(action)
        probs = torch.nn.functional.softmax(q_vals/self.temperature,dim=-1).cpu().detach().numpy()
        
        # if self.deterministic == False:, first do epsilon-greedy. then if model choice is rolled, choose based on probabilities from softmax
        # otherwise, just choose the model action based on the max
        if self.deterministic == False:
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(np.arange(0,len(q_vals),1))
            else:
                # choose index based on probabilities from probs
                action = np.random.choice(np.arange(0,len(q_vals),1),p=probs)

        self.pr = probs[action]
        return action, self.pr, action_data
        
        # """Reward Prediction Error Calculation and Update of Q values"""
        # rewards = torch.zeros_like(torch.Tensor(state),requires_grad=False,dtype=torch.long)
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         a =  last_action.argmax(axis=-1)
        #         alpha = self.alpha[not (state[seq,i][1] == last_action[seq,i][1])]
        #         rewards[seq,i,a] = 2*(state[seq,i][1] == last_action[seq,i][1])-1                
        #         self.Qs[seq,i,a] = self.Qs[seq,i-1,a] * (1-alpha) + alpha*(rewards[seq,i,a] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
        #updates all action state pairs instead of just the 
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         rewards[seq,i,int(last_action[seq,i][1])] = 2*(state[seq,i][1] == last_action[seq,i][1])-1
        #         self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    # def RPEupdateBandit(self, state,last_action, rewards):
    #     """Reward Prediction Error Calculation and Update of Q values"""
    #     # rewards = torch.zeros_like(torch.Tensor(last_action),requires_grad=False,dtype=torch.int)
    #     for seq in range(state.shape[0]): #batch
    #         for i in range(0,state.shape[1]): #element in sequence
    #             a =  last_action.argmax(axis=-1)
    #             rewards[seq,i,a] = 2*(rewards[seq,i,a])-1                
    #             self.Qs[seq,i] = self.Qs[seq,i-1] * (1-alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    # def RPEupdateRPS(self, state,last_action, reward):
    #     """Reward Prediction Error Calculation and Update of Q values"""
    #     rewards = torch.zeros_like(torch.Tensor(last_action),requires_grad=False,dtype=torch.float)
    #     for seq in range(state.shape[0]): #batch
    #         for i in range(0,state.shape[1]): #element in sequence
    #             a =  last_action.argmax(axis=-1)
    #             rewards[seq,i,a] = 2*(reward[seq,i])-1  
    #             self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0

class QAgentAsymmetric(QAgent): # instead of a forgetting model, we have two alpha parameters
    # In the forgetting model, timescale is not 1/alpha, it's 1/(1-alpha). to fix this, i changed the update a bit
    
    def __init__(self, alpha = [.2,.2], init_action = None, bias = 0, epsilon=0, path = None,
                 load = True, beta = 1,deterministic=True, env = None, learn_BG = False):
        super(QAgentAsymmetric,self).__init__(alpha , 0, init_action , bias, epsilon, path, load, 1/beta ,deterministic, env)
        self.temperature = 1/beta
        self.learn_BG = learn_BG
        if self.learn_BG:
            self.alpha_win = nn.Parameter(torch.tensor(alpha[0],dtype=torch.float32),requires_grad=True)
            self.alpha_loss = nn.Parameter(torch.tensor(alpha[1],dtype=torch.float32),requires_grad=True)

    
    def reset(self, keep_first=False):
        """
        Properly reset QAgent state without reinitializing the object.
        This preserves the tensor structure and properly resets Q-values.
        """
        if keep_first == False:
            # Reset Q-values to initial state based on current tensor structure
            if self.Qs.dim() == 1:
                # For 1D tensors (direct usage), initialize to [0.5, 0.5]
                self.Qs = torch.full((2,), 0.5, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 2:
                # For 2D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            elif self.Qs.dim() == 3:
                # For 3D tensors, preserve structure but reset values to 0.5
                self.Qs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            
            # Reset other state variables to initial values
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
        else:
            # Keep first implementation (existing logic) - also initialize to 0.5
            newQs = torch.full_like(self.Qs, 0.5, requires_grad=False, device=device, dtype=torch.float)
            # NEED TO ACCOUNT FOR BATCHES
            if self.Qs.dim() >= 2:
                newQs[:,0,:] = self.Qs[:,0,:]
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
            self.Qs = newQs.to(device)

    def RPEupdateMP(self, state,last_action, rewards = None):
        """Reward Prediction Error Calculation and Update of Q values"""
        # Ensure Qs has correct shape - handle both 2D (neural network) and 3D (standalone) cases
        if self.Qs.dim() == 2:
            # If 2D (for neural network compatibility), expand to 3D for processing
            if self.Qs.shape != (state.shape[0] * state.shape[1], 2):
                self.Qs = torch.full((state.shape[0] * state.shape[1], 2), 0.5, 
                                    requires_grad=False, device=device, dtype=torch.float)
            # Reshape to 3D for consistent processing
            Qs_3d = self.Qs.view(state.shape[0], state.shape[1], -1)
        else:
            # If not 3D, initialize properly with 0.5 (like forgetting model)
            if self.Qs.shape != (state.shape[0], state.shape[1], 2):
                self.Qs = torch.full((state.shape[0], state.shape[1], 2), 0.5, 
                                    requires_grad=False, device=device, dtype=torch.float)
            Qs_3d = self.Qs
        
        # Process updates on 3D tensor
        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                a = last_action[seq,i][-1].to(torch.int)
                rew = rewards[seq,i][a]

                # For asymmetric model: use different alpha based on reward
                if rew > 0:
                    alpha = self.alpha[0]  # alpha_win
                else:
                    alpha = self.alpha[1]  # alpha_loss
                
                # RPE update: For chosen action: Qnew = Q + alpha[reward]*(reward-Q)
                # For unchosen action: Qnew = Q (no change)
                prediction_error = rew - Qs_3d[seq,i,a]
                Qs_3d[seq,i,a] = Qs_3d[seq,i,a] + alpha * prediction_error
                # Unchosen action remains unchanged: Qs_3d[seq,i,1-a] = Qs_3d[seq,i,1-a]
        
        # Update self.Qs with the correct shape
        if self.Qs.dim() == 2:
            # Flatten back to 2D for neural network compatibility
            self.Qs = Qs_3d.view(-1, Qs_3d.shape[-1])
        else:
            self.Qs = Qs_3d
                
    def forward(self,state,last_action, rewards = None):
        self.RPEupdate(state,last_action, rewards)
        return self.Qs
    
    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
        
        # Handle both 2D and 3D Q-value tensors
        if self.Qs.dim() == 3:
            # For 3D tensor, use the last timestep (most recent)
            q_vals = self.Qs[0, -1, :]  # [batch=0, last_timestep, actions]
        elif self.Qs.dim() == 2:
            # For 2D tensor, use the last row (most recent)
            q_vals = self.Qs[-1, :]  # [last_timestep, actions]
        else:
            # For 1D tensor, use directly
            q_vals = self.Qs
        
        action = np.argmax(q_vals.cpu())
        self.max_choices.append(action)
        probs = torch.nn.functional.softmax(q_vals/self.temperature,dim=-1).cpu().detach().numpy()
        
        # if self.deterministic == False:, first do epsilon-greedy. then if model choice is rolled, choose based on probabilities from softmax
        # otherwise, just choose the model action based on the max
        if self.deterministic == False:
            if np.random.uniform(0,1) < self.epsilon:
                action = np.random.choice(np.arange(0,len(q_vals),1))
            else:
                # choose index based on probabilities from probs
                action = np.random.choice(np.arange(0,len(q_vals),1),p=probs)

        self.pr = probs[action]
        return action, self.pr, action_data

class QAgentEpsilon(nn.Module):
    '''
    Basic Q-learning agent sans exploration
    '''
    def __init__(self, alpha = 0.05, discount_factor = 0, init_action = None, bias = 0, epsilon=.05, path = None, load = True, temperature = 1,deterministic=True):
        super(QAgentEpsilon,self).__init__()
        self.temp = torch.ones(1,dtype=torch.float32, requires_grad=True, device=device) * temperature
        self.path = path
        self.load = load
        self.alpha = alpha
        self.gamma = discount_factor
        self.deterministic = deterministic
        # self.state = state
        self.Qs = torch.zeros(2,requires_grad=False,device = device,dtype=torch.float)
        # self.Qs = torch.zeros_like(state,requires_grad=False,dtype=torch.int)
        # self.last_action = last_action
        # self.rewards = np.zeros(self.states.shape[0],self.states.shape[1],1)
        # self.rewards = torch.zeros_like(state,requires_grad=False,dtype=torch.int)

        self.pchooseright = 0.5 #default starting prob
        self.bias = bias
        self.epsilon = epsilon
        self.pchooseright = 0.5
        # self.act_hist = [self.prev_action]
        self.biasinfo = None
        self.max_choices = []

            
        if (self.path is not None) and (self.load == True):
            self.load_model()
             
    def __str__(self):
        return 'Batching-enabled Q Learner'

    # def RPEupdateMP(self, state,last_action, rewards = None):
    #     """Reward Prediction Error Calculation and Update of Q values"""
    #     rewards = torch.zeros_like(torch.Tensor(state),requires_grad=False,dtype=torch.long)
    #     for seq in range(state.shape[0]): #batch
    #         for i in range(0,state.shape[1]): #element in sequence
    #             a =  last_action.argmax(axis=-1)
    #             rewards[seq,i,a] = 2*(state[seq,i][1] == last_action[seq,i][1])-1                
    #             self.Qs[seq,i,a] = self.Qs[seq,i-1,a] * (1-self.alpha) + self.alpha*(rewards[seq,i,a] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    

    #only want this to run once
    def RPEupdate(self, state,last_action):
        """Reward Prediction Error Calculation and Update of Q values"""
        # Qsa = self.Qs[int(self.act_hist[-1])]
        # maxQ_Sprime = max(self.Qs)
        # self.Qs[self.act_hist[-1]]  = Qsa + self.alpha * (reward + self.gamma * maxQ_Sprime - Qsa)
        
        # all_actions =last_action[:,:,1] # 0 is left, 1 is right
        
        rewards = torch.zeros_like(torch.Tensor(state),requires_grad=False,dtype=torch.int)


        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                #have to select action somehow, or does it matter?
                a =  int(last_action[seq,i][1])
                # rewards[seq,i,a] = 2*(state[seq,i][1] == last_action[seq,i][1])-1           
                rewards[seq,i,a] = (state[seq,i][1] == last_action[seq,i][1])          
     
                self.Qs[seq,i,a] = self.Qs[seq,i-1,a] * (1-self.alpha) + self.alpha*(rewards[seq,i,a] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0

        # Qsa = self.Qs[self.act_hist[-1]]
        # maxQ_Sprime = max(self.Qs)
        # self.Qs[self.act_hist[-1]]  = Qsa + self.alpha * (reward + self.gamma * maxQ_Sprime - Qsa)
       
    
        #updates all action state pairs instead of just the 
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         #have to select action somehow, or does it matter?

        #         rewards[seq,i,int(last_action[seq,i][1])] = 2*(state[seq,i][1] == last_action[seq,i][1])-1

        #         self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    
    
    
    def get_last_saved(self):
        return self.action, self.pchooseright,self.biasinfo
    
    def save_model(self, path=None):
        if path is None:
            with open(self.path, "w+") as f:
                pickle.dump(self.__dict__,f)
        else:
            with open(path, "w+") as f:
                pickle.dump(self.__dict__,f)
                
    def load_model(self, path=None):
        if path is None:
            with open(self.path, "r+") as f:
                self.__dict__ = pickle.load(f)
        else:
            with open(path, "r+") as f:
                self.__dict__ = pickle.load(f)
        
        
    def forward(self,state,last_action):
        # rew = 1 - rew #if opponent wins, agent loses--rew is agent reward, so flip bit
        if state.shape != self.Qs.shape:
            self.Qs = torch.zeros_like(state,requires_grad=False,dtype=torch.float)
        self.RPEupdate(state,last_action)
        return self.Qs
        # action, pchooseright, biasinfo = self.action()
        # self.action = action
        # self.pchooseright = pchooseright
        # self.biasinfo = biasinfo
        # self.act_hist.append(action)
        # return action,pchooseright,biasinfo

    #fully deterministic
    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
        
        
        # self.ql, self.qr = self.Qs
        # self.max_choices.append(np.argmax(self.Qs))
        # self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp))
        
        # self.ql, self.qr = self.Qs.detach().cpu()
        self.ql, self.qr = self.Qs.detach().squeeze().cpu()

        self.max_choices.append(np.argmax(self.Qs.cpu()))
        self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp.detach().cpu()))
        
        #MAKE BIAS RANDOMLY DISTRIBUTED VARIABLE WITH MEAN BIAS and s.d. ?
        # work backwards: epsilon at 50/50 corresponds to what?
        # if at pr + bias, choose ~N(b,.02) for 5% deviation maximum?
        # note that for small %, It's roughly normal so it's a good approx
        # self.pr += self.bias 
    
        
        if rng.uniform(0,1) > self.epsilon:
            action = 1 if self.pr >= .5 else 0
        else:   
            action = rng.choice([0,1])

        return action, self.pr, action_data



    #MOSTLY DETERMINISTIC, CHOOSES SIDE WITH HIGHEST PROBABILITY
    # def action(self):
    #     action_data = {'Qs':self.Qs,'bias':self.bias}
        
        
    #     # self.ql, self.qr = self.Qs
    #     # self.max_choices.append(np.argmax(self.Qs))
    #     # self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp))
        
    #     # self.ql, self.qr = self.Qs.detach().cpu()
    #     self.ql, self.qr = self.Qs.detach().squeeze().cpu()

    #     self.max_choices.append(np.argmax(self.Qs.cpu()))
    #     self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)/self.temp.detach().cpu()))
        
    #     #MAKE BIAS RANDOMLY DISTRIBUTED VARIABLE WITH MEAN BIAS and s.d. ?
    #     # work backwards: epsilon at 50/50 corresponds to what?
    #     # if at pr + bias, choose ~N(b,.02) for 5% deviation maximum?
    #     # note that for small %, It's roughly normal so it's a good approx
    #     # self.pr += self.bias 
    #     if self.deterministic == True:
    #         self.pr += np.random.normal(self.bias,self.epsilon/2)

    #         action = 1 if self.pr >= .5 else 0
    #     else:   
    #         action = rng.choice([0,1], p=[1-self.pr, self.pr])

    #     return action, self.pr, action_data
    
    def reset(self, keep_first=False):
        if keep_first == False:
            self.__init__(self.alpha, self.gamma , self.action, self.bias , self.epsilon, self.path, self.load)
        else:
            newQs = torch.zeros_like(self.Qs, requires_grad=False, device=device, dtype=torch.float)
            # NEED TO ACCOUNT FOR BATCHES
            newQs[:,0,:] = self.Qs[:,0,:]             
            self.pchooseright = 0.5
            self.biasinfo = None
            self.max_choices = []
            self.Qs = newQs.to(device)


    #action wrapper for testing WSLS behavior
    def get_action(self,state, last_action, hidden_in, deterministic=True):
        self.forward(state.reshape((1,1,-1)),last_action.reshape((1,1,-1)))
        return self.action()


def test(iters=150):
    max_steps = 500
    rolling_avg = 0
    for eps in range(iters):
        env = envs.mp_env.MPEnv(show_opp=True,train=True,fixed_length=False,reset_time = max_steps, opponents=['all', 'epsilonqlearn'])
        state = env.reset()[0]
        agent =  QAgent(alpha = .05, discount_factor=.01)
        episode_reward = 0
        action = agent.action()[0]
        if state == action:
            reward = 1
        else:
            reward = 0
        for step in range(max_steps):
            action = agent.step(reward)[0]
            next_state, reward, done, _ = env.step(action)
            
            env.render()   


            episode_reward += reward
            state=next_state[0]

            if done:
                break
        print()
        print('Episode: ', eps, '| Episode Reward: ', episode_reward/max_steps)
        rolling_avg = (rolling_avg*(eps) + np.sum(episode_reward)/step)/(eps+1)
    print(rolling_avg)

# test()


'''
Q-learning for WSLS comparison
'''

class QWSLS(QAgent):
    def RPEupdate(self,state,last_action):
        """Reward Prediction Error Calculation and Update of Q values"""
        state = torch.tensor(state)
        last_action = torch.tensor(last_action)
        rewards = torch.zeros_like(state,requires_grad=False,dtype=torch.int)


        #have to select action somehow, or does it matter?
        a =  int(last_action[1])
        rewards[a] = 2*(state[1] == last_action[1])-1                
        self.Qs[a] = self.Qs[a] * (1-self.alpha) + self.alpha*(rewards[a] +self.gamma * torch.max(self.Qs)) #is this correct if gamma != 0

        # Qsa = self.Qs[self.act_hist[-1]]
        # maxQ_Sprime = max(self.Qs)
        # self.Qs[self.act_hist[-1]]  = Qsa + self.alpha * (reward + self.gamma * maxQ_Sprime - Qsa)
       
    
        #updates all action state pairs instead of just the 
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         #have to select action somehow, or does it matter?

        #         rewards[seq,i,int(last_action[seq,i][1])] = 2*(state[seq,i][1] == last_action[seq,i][1])-1

        #         self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    
    

'''
Q-learning agent implementing the RL (Reinforcement Learning) component in our project.
'''
import numpy as np
import torch 
import pickle
import os
from scipy import stats
import envs.mp_env
import torch.nn as nn

# rng = np.random.default_rng(28980)

# mac = False
# device_idx = 0
# if mac: 
#     device = torch.device("mps")
# else :
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")


# TODO: Two agents, one looking at current state and one looking at current state + previous action
# i.e. 4 states, 2 actions. That might just correspond to a higher discount f actor though
class QTester(nn.Module):
    '''
    Basic Q-learning agent sans exploration
    '''
    def __init__(self, alpha = 0.05, discount_factor = 0, init_action = None, bias = 0, epsilon=.05, path = None, load = True):
        super(QTester,self).__init__()
        self.path = path
        self.load = load
        self.alpha = alpha
        self.gamma = discount_factor
        # self.state = state
        self.Qs = torch.zeros(2,requires_grad=False,device = device,dtype=torch.float)
        # self.Qs = torch.zeros_like(state,requires_grad=False,dtype=torch.int)
        # self.last_action = last_action
        # self.rewards = np.zeros(self.states.shape[0],self.states.shape[1],1)
        # self.rewards = torch.zeros_like(state,requires_grad=False,dtype=torch.int)

        self.pchooseright = 0.5 #default starting prob
        self.bias = bias
        self.epsilon = epsilon
        self.pchooseright = 0.5
        # self.act_hist = [self.prev_action]
        self.biasinfo = None
        self.max_choices = []

            
        if (self.path is not None) and (self.load == True):
            self.load_model()
             
    def __str__(self):
        return 'Deterministic Q'

    #only want this to run once
    def RPEupdate(self, state,last_action):
        """Reward Prediction Error Calculation and Update of Q values"""
        # Qsa = self.Qs[int(self.act_hist[-1])]
        # maxQ_Sprime = max(self.Qs)
        # self.Qs[self.act_hist[-1]]  = Qsa + self.alpha * (reward + self.gamma * maxQ_Sprime - Qsa)
        
        # all_actions =last_action[:,:,1] # 0 is left, 1 is right
        
        rewards = torch.zeros_like(state,requires_grad=False,dtype=torch.int)


        for seq in range(state.shape[0]): #batch
            for i in range(0,state.shape[1]): #element in sequence
                #have to select action somehow, or does it matter?
                a =  int(last_action[seq,i][1])
                rewards[seq,i,a] = 2*(state[seq,i][1] == last_action[seq,i][1])-1                
                self.Qs[seq,i,a] = self.Qs[seq,i-1,a] * (1-self.alpha) + self.alpha*(rewards[seq,i,a] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0

        # Qsa = self.Qs[self.act_hist[-1]]
        # maxQ_Sprime = max(self.Qs)
        # self.Qs[self.act_hist[-1]]  = Qsa + self.alpha * (reward + self.gamma * maxQ_Sprime - Qsa)
       
    
        #updates all action state pairs instead of just the 
        # for seq in range(state.shape[0]): #batch
        #     for i in range(0,state.shape[1]): #element in sequence
        #         #have to select action somehow, or does it matter?

        #         rewards[seq,i,int(last_action[seq,i][1])] = 2*(state[seq,i][1] == last_action[seq,i][1])-1

        #         self.Qs[seq,i] = self.Qs[seq,i-1] * (1-self.alpha) + self.alpha*(rewards[seq,i] +self.gamma * torch.max(self.Qs[seq,i-1])) #is this correct if gamma != 0
    
    
    
    
    def get_last_saved(self):
        return self.action, self.pchooseright,self.biasinfo
    
    def save_model(self, path=None):
        if path is None:
            with open(self.path, "w+") as f:
                pickle.dump(self.__dict__,f)
        else:
            with open(path, "w+") as f:
                pickle.dump(self.__dict__,f)
                
    def load_model(self, path=None):
        if path is None:
            with open(self.path, "r+") as f:
                self.__dict__ = pickle.load(f)
        else:
            with open(path, "r+") as f:
                self.__dict__ = pickle.load(f)
        
        
    def forward(self,state,last_action):
        # rew = 1 - rew #if opponent wins, agent loses--rew is agent reward, so flip bit
        if state.shape != self.Qs.shape:
            self.Qs = torch.zeros_like(state,requires_grad=False,dtype=torch.float)
        self.RPEupdate(state,last_action)
        return self.Qs
        # action, pchooseright, biasinfo = self.action()
        # self.action = action
        # self.pchooseright = pchooseright
        # self.biasinfo = biasinfo
        # self.act_hist.append(action)
        # return action,pchooseright,biasinfo

    #wrapper for getting action
    def get_action(self,state,last_action,hidden_in,deterministic=True):
        self.forward(torch.Tensor(state),torch.Tensor(last_action))
        return self.action()
    
    
    #MOSTLY DETERMINISTIC, CHOOSES SIDE WITH HIGHEST PROBABILITY
    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
       
        self.ql, self.qr = self.Qs
        self.max_choices.append(np.argmax(self.Qs))
        self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)))
        
        #MAKE BIAS RANDOMLY DISTRIBUTED VARIABLE WITH MEAN BIAS and s.d. ?
        # work backwards: epsilon at 50/50 corresponds to what?
        # if at pr + bias, choose ~N(b,.02) for 5% deviation maximum?
        # note that for small %, It's roughly normal so it's a good approx
        # self.pr += self.bias 
        self.pr += np.random.normal(self.bias,self.epsilon/2)

        action = 1 if self.pr >= .5 else 0

        return action, self.pr, action_data
    
    def reset(self):
        self.__init__(self.alpha, self.gamma , self.action, self.bias , self.epsilon, self.path, self.load)



def test(iters=150):
    max_steps = 500
    rolling_avg = 0
    for eps in range(iters):
        env = envs.mp_env.MPEnv(show_opp=True,train=True,fixed_length=False,reset_time = max_steps, opponents=['all', 'epsilonqlearn'])
        state = env.reset()[0]
        agent =  QAgent(alpha = .05, discount_factor=.01)
        episode_reward = 0
        action = agent.action()[0]
        if state == action:
            reward = 1
        else:
            reward = 0
        for step in range(max_steps):
            action = agent.step(reward)[0]
            next_state, reward, done, _ = env.step(action)
            
            env.render()   


            episode_reward += reward
            state=next_state[0]

            if done:
                break
        print()
        print('Episode: ', eps, '| Episode Reward: ', episode_reward/max_steps)
        rolling_avg = (rolling_avg*(eps) + np.sum(episode_reward)/step)/(eps+1)
    print(rolling_avg)

# test()