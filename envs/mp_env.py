import numpy as np
import neurogym as ngym
import torch
from neurogym import spaces
# from .matching_pennies_fast import matching_pennies
from .matching_pennies import matching_pennies,matching_pennies_frac
try:
    from .matching_pennies_numba import matching_pennies_numba, matching_pennies_frac_numba
    NUMBA_AVAILABLE = True
    
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available, falling back to standard implementation")



rng = np.random.default_rng(428980)

class MPEnv(ngym.TrialEnv):
    def __init__(self, rewards = None, opponent = "all", reset_time = 150, fixed_length = True, opp_params = None, fixed_opp = True, use_numba = True):
        super().__init__()
        if rewards is not None:
            self.rewards = rewards
        else:
            self.rewards =  {'abort' : -1, 'correct': +1., 'select':0.0, 'fail':0}
        
        self.updated = False 
        self.model_hist = [] #agent choice history
        self.env_hist = []
        self.reward_hist = [] #agent reward history
        self.f_flag = [] #fractional flag
        self.opponent_action = None #saves value of opponent action to compare against
        self.pright_mp = [] #for mp opponent
        self.biases = [] #for mp opponent
        self.opp_params = opp_params
        self.opp_params_old = opp_params
        self.opp_kwargs = None
        self.all_opponents = ['lrplayer','1','0','2','patternbandit','all','reversalbandit','epsilonqlearn','softmaxqlearn','mimicry','wslsplayer','deterministic', 'wsls','fractional'] #list opponent names
        self.opponent_name = opponent
        self.opponent_ind = self.all_opponents.index(self.opponent_name)
        self.opponent = opponent
        
        # Choose implementation based on availability and preference
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
       
        if self.use_numba:
            self.matching_pennies = matching_pennies_numba if opponent != 'fractional' else matching_pennies_frac_numba
            print(f"Using numba-optimized matching pennies for opponent: {opponent}")
        else:
            self.matching_pennies = matching_pennies if opponent != 'fractional' else matching_pennies_frac
            
        self.fixed_length = fixed_length
        self.fixed_len=self.fixed_length
        self.fixed_opp = fixed_opp #whether it plays the same type of opponent or opponent randomly chosen each time
        self.timing = {'outcome':100}
        
        if opponent is None:
            self.opponents = self.all_opponents
        else:
            self.opponents = [opponent]
        self.probs = [i/len(self.opponents) for i in range(len(self.opponents))]  #uniform cdf of opponents for random draw, for now
        self.base_reset_time = self.reset_time = reset_time #reset opponent every reset_time
        self.set_opponent(opponent, self.sample_params(opp_params),replace=False)
        self.draw_opponent()
        # self.set_opponent(opponent) #initial opponent. If len(opponent) > 1, then opponent is chosen randomly

        self.act_dict = {'left' : 0, 'right' : 1} #maybe should make -1,1?
        self.action_space = spaces.Discrete(2,name=self.act_dict)
        self.last_opp_choice = None
        self.ob_dict = {'stimulus':[0,1]}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                    dtype=np.float32,name = self.ob_dict)
        # self.reset()
    
    def sample_params(self, params):
        ps = {}
        if params is not None:
            for op in params.keys():
                ps[op] = {}
                for p in params[op].keys():
                    if isinstance(params[op][p],list):
                        if len(params[op][p]) > 1:
                            ps[op][p] = rng.choice(params[op][p])
                        else:
                            ps[op][p] = params[op][p][0]
                    else:
                        ps[op][p] = params[op][p]
        return ps
    
    def sample_subset(self, params, output):
        for key in params.keys():
            if isinstance(params[key],list) and len(params[key]) > 1:
                output[key] = rng.choice(params[key])
            else:
                if isinstance(params[key],list):
                    output[key] = params[key][0]
                else:
                    output[key] = params[key]
        return output

    def update_params(self,opponent = None,params=None):
            if params is not None and opponent is not None:
                self.opp_params[opponent] = params[opponent]
            else:
                self.opp_params.update(params)
    
    def set_opponent(self, opponent, opp_kwargs = None, replace  = True):
        if opponent=='opponent':
            opponent = 'lrplayer'
            
        if isinstance(opponent,list) and len(self.opponents) > 1:
            self.opponent = rng.choice(opponent)
        else:
            self.opponent = opponent
 
        if ((self.opp_params is not None) and (opponent not in self.opp_params)) or (replace == True and not (opp_kwargs is None)):
            self.update_params(opponent,opp_kwargs)
        
        if opp_kwargs is None and len(list(self.opp_params.keys())) > 1:
            if opponent in self.opp_params.keys():
                self.opp_kwargs = self.opp_params[self.opponent_name]
            else:
                self.opp_kwargs = opp_kwargs
        else: 
            # opp_dict = {opponent:opp_kwargs}
            # self.update_params(opponent,opp_kwargs)
            # self.opp_params.update(opp_kwargs)
            self.opp_kwargs = opp_kwargs


        opp_kwargs = opp_kwargs[opponent]
        if opponent=='opponent':
            opponent = 'lrplayer'
        '''
        Given an opponent and arguments, set the environment opponent and related parameters to that opponent
        '''
        self.opponent_name = opponent
        # print(self.opponent_name)
        
        if opponent == "softmaxqlearn":
            opp = SoftmaxQlearn(**opp_kwargs)
        elif opponent == "epsilonqlearn":
            opp = EpsilonQlearn(**opp_kwargs)
        elif opponent == "patternbandit":
            opp = PatternBandit(**opp_kwargs)
        elif opponent == "reversalbandit":
            # opp_kwargs['rtrain'] = self.train #need to add in this variable to make sure updates work as expected here
            opp = ReversalBandit(**opp_kwargs)
        elif opponent in ['all',0,1,2,'0','1','2','fractional']:
            opp = opponent
        elif opponent == "lrplayer":
            opp = LRPlayer(**opp_kwargs)
        elif opponent == "wslsplayer":
            opp = LRPlayer(**opp_kwargs)
        elif opponent == "mimicry":
            opp = MimicryPlayer(**opp_kwargs)
        elif opponent == 'wsls':
            opp = WSLSPlayer(**opp_kwargs)
        elif opponent == 'deterministic':
            opp = DeterministicPlayer(*opp_kwargs)
        else:
            raise ValueError("Agent type not found")
        self.opponent = opp
        # self.opp_kwargs = opp_kwargs
        self.opponent_ind = self.all_opponents.index(self.opponent_name)

        return self.opponent_name,self.opponent,self.opp_kwargs,self.opponent_ind
        
    
    def __str__(self):
        return f"matching pennies with opponent {self.opponent_name}"
        
    def draw_opponent(self):
        '''
        Draw an opponent randomly from the list of opponents, draw parameters from uniform set of parameters
        '''
        # self.opponent = rng.choice(self.opponents, self.probs)
        if isinstance(self.opponents,list) and len(self.opponents) > 1:
            self.opponent = rng.choice(self.opponents)
            # print
        # self.opp_params = self.opp_params_old
        
        opp_kwargs = {}
        gen = self.opp_params is not None  #if we are using the generated versions rather than the default
        if gen:
                params = self.opp_params[self.opponent_name] #params is a dict of lists of the set of parameters to draw from
                for key in params.keys():
                    if isinstance(params[key],list) and len(params[key]) > 1:
                        opp_kwargs[key] = rng.choice(params[key])
                    else:
                        if isinstance(params[key],list):
                            opp_kwargs[key] = params[key][0]
                        else:
                            opp_kwargs[key] = params[key]
        if self.opponent_name in ["all",0,1,2,'0','1','2','fractional']:
            if gen:
                bias = params['bias'] #getting the dict of the range
                depth = params['depth']
                opp_kwargs['bias'] = 0 if bias == [] else rng.choice(bias) #random uniform draw according to params
                if self.opponent_name == 'fractional':
                    p_mp2 = params['p_mp2'] if 'p_mp2' in params.keys() else 0.25
                    opp_kwargs['p_mp2'] = p_mp2
                if depth == []:
                    opp_kwargs['depth'] = 4
                else:
                    if isinstance(depth,list):
                        opp_kwargs['depth'] = rng.choice(depth)
                    else:
                        opp_kwargs['depth'] = depth
                
                # opp_kwargs['depth']= 4 if depth == [] else rng.choice(depth)
            else: #default range
                opp_kwargs['bias'] = rng.choice([0,-0.02,-0.05,0.02,0.05],p = [0.7,0.1,0.05,0.1,0.05])
                opp_kwargs['depth'] = rng.choice([2,4,8],p=[0.3,0.5,0.2])
            opp_kwargs['p'] = 0.05
    
        if self.opponent_name == 'patternbandit':
            #get rid of the binary numbers with only 1s ,*range(16,31)
            if 'pattern' in params.keys():
                    opp_kwargs['pattern'] = params['pattern']
            else:
                if gen:
                    if 'length' in params.keys():
                        lengths = params['length']
                    length = rng.choice(lengths) #random draw of the length (this makes it evenly distributed)
                    l = list(range(2**(length-1),2**length-1)) #ex: length 5 = range(16,31) --> 2**4: 2**5 - 1
                else:
                    l = [*range(4,7),*range(8,15)]
                opp_kwargs['pattern'] = bin(rng.choice(l))[2:]
            # print(opp_kwargs['pattern'])

        if self.opponent_name == "reversalbandit":
            if gen:
                pr = params['pr']
                update = params['update']
                opp_kwargs['pr'] = rng.choice(pr)
                opp_kwargs['update'] = rng.choice(update)
            else:
                opp_kwargs['pr'] = rng.choice(np.arange(0.05,1,0.05))
                opp_kwargs['update'] = rng.choice([50,75,100,125])
            # opp_kwargs['rtrain'] = self.train
        if type(self.opponent_name) == str and self.opponent_name[-6:] == "qlearn":
            if gen:
                lr = params['lr']
                gamma = params['gamma']
                bias = params['bias']
                for key in params.keys():
                    if isinstance(params[key],list) and len(params[key]) > 1:
                        opp_kwargs[key] = rng.choice(params[key])
                    else:
                        if isinstance(params[key],list):
                            opp_kwargs[key] = params[key][0]
                        else:
                            opp_kwargs[key] = params[key]
                # opp_kwargs['lr'] = rng.choice(lr)
                # opp_kwargs['gamma'] = rng.choice(gamma)
                # opp_kwargs['bias'] = rng.choice(bias)
            else:
                opp_kwargs['lr'] = rng.choice([0.25,0.5,1])
                opp_kwargs['gamma'] = rng.choice([0.5,0.6,0.75,0.9,0.99])
                opp_kwargs['bias'] = rng.choice([0,-0.02,-0.05,0.02,0.05],p = [0.5,0.1,0.15,0.1,0.15])
        if self.opponent_name == "epsilonqlearn":
            if not gen:
                opp_kwargs['epsilon'] = rng.choice([0.0,0.1,0.2,0.5],p =[0.33,0.23,0.22,0.22])
        if self.opponent_name == "softmaxqlearn":
            if not gen:
                opp_kwargs['temp'] = rng.choice([0.1,0.5,1,2,3])
        if self.opponent_name == "lrplayer":
            if gen:
                b = params['b']
                # len_choice = params['len_choice']
                # len_outcome = params['len_outcome']
                choice_betas = params['choice_betas']
                outcome_betas = params['outcome_betas']
                len_choice = len(choice_betas) #doesn't matter for normal usage, just testing WSLS
                len_outcome=len_choice
                if isinstance(b,list):
                    opp_kwargs['b'] = rng.choice(b)
                else:
                    opp_kwargs['b'] = b
                # l_choice = rng.choice(len_choice)
                # l_outcome = rng.choice(len_outcome)
                l_choice = len_choice
                l_outcome = len_outcome
                draw_random_params = lambda x,data: [rng.choice(data) for _ in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(l_choice,choice_betas)
                opp_kwargs['outcome_betas'] = draw_random_params(l_outcome,outcome_betas)
            else:
                opp_kwargs['b'] = rng.choice([0,0,0,0,0,0,0,0,0,-0.1,0.1,-0.25,0.25,0.5,-0.5])
                draw_random_params = lambda x: [round(rng.choice([0,0,0,0,0,0,-1,1,-0.7,0.7,-0.5,0.5,
                                                                        -0.4,0.4,-1.5,1.5,-2,2]),1) for i in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(rng.choice(np.arange(3)))
                opp_kwargs['outcome_betas'] = draw_random_params(rng.choice(np.arange(3)))
        if self.opponent_name == "wslsplayer":
            if gen:
                b = params['b']
                # len_choice = params['len_choice']
                # len_outcome = params['len_outcome']
                choice_betas = params['choice_betas']
                outcome_betas = params['outcome_betas']
                len_choice = len(choice_betas)
                len_outcome=len_choice
                
                opp_kwargs['b'] = rng.choice(b)
                l_choice = rng.choice(len_choice)
                l_outcome = rng.choice(len_outcome)
                draw_random_params = lambda x,data: [rng.choice(data) for _ in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(l_choice,choice_betas)
                opp_kwargs['outcome_betas'] = [draw_random_params(outcome_betas),0]
            else:
                opp_kwargs['b'] = rng.choice([0,0,0,0,0,0,0,0,0,-0.1,0.1,-0.25,0.25,0.5,-0.5])
                draw_random_params = lambda x: [round(rng.choice([0,0,0,0,0,0,-1,1,-0.7,0.7,-0.5,0.5,
                                                                        -0.4,0.4,-1.5,1.5,-2,2]),1) for i in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(rng.choice(np.arange(3)))
                opp_kwargs['outcome_betas'] = draw_random_params(rng.choice(np.arange(3)))
        if self.opponent_name == "mimicry":
            if gen:
                n = params['n']
                opp_kwargs['n'] = rng.choice(n)
            else:
                opp_kwargs['n'] = rng.choice(np.arange(0,5))
        if self.opponent_name == "wsls":
            opp_kwargs = {}
            # self.draw_opponent()

        opp_kwargs = {self.opponent_name : opp_kwargs}
        self.opp_kwargs = opp_kwargs
        return self.set_opponent(self.opponent_name,opp_kwargs, replace=False)        
    
    def clear_data(self): 
        '''effectively reset the environment
        '''
        self.model_hist = []
        self.reward_hist = []
        self.env_hist = []

        self.pright_mp = []
        self.biases = []
        self.f_flag = []
        self.updated=False
    
    def reset(self):
        """
        Override the reset method to ensure proper episode reset including reset_time
        """
        # Reset the trial counter to the base value
        if self.fixed_len:
            self.reset_time = self.base_reset_time
        else:
            self.reset_time = self.base_reset_time + rng.randint(0, self.base_reset_time // 2)
        
        # Clear all history data
        self.clear_data()
        
        # Draw new opponent if not fixed
        if not self.fixed_opp:
            self.draw_opponent()
        
        # Reset other trial-specific variables
        self.done = False
        self.action = None
        self.opponent_action = None
        self.last_opp_choice = None
        
        # Call parent reset method
        return super().reset()
    
    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        Here you have to set:
        The trial periods: fixation, stimulus...
        Optionally, you can set:
        The ground truth: the correct answer for the created trial.
        """
        self.done = False
        self.action = None
        if len(self.model_hist) > self.reset_time:
            #if we are ready to reset
            avgchoice = sum(self.model_hist)/len(self.model_hist)
            avgrew = sum(self.reward_hist)/len(self.reward_hist)
            #print an update
            # print(f"opponent {self.opponent}, avg choice {avgchoice} avg reward {avgrew} on {len(self.reward_hist)} trials")
            self.clear_data()   
            #draw new opponent\
            self.draw_opponent()   # it's drawing opponent instead of 
            if self.fixed_len:
                self.reset_time = self.base_reset_time
            else:
                self.reset_time = self.base_reset_time + rng.random() * self.base_reset_time //2 #change the reset time according to a random walk with mean reset_time
            self.done = True

        if self.opponent_name in ['all',0,1,2,'0','1','2','fractional']: #matching pennies is just a function, so we can call it here
            bias = self.opp_kwargs[self.opponent_name].get('bias',0)
            depth = self.opp_kwargs[self.opponent_name].get('depth',4)
            pval = self.opp_kwargs[self.opponent_name].get('p',0.05)
            if self.opponent_name == 'fractional':
                pf = self.opp_kwargs[self.opponent_name].get('p_mp2',0.25)
                opponent_action,pchooseright,biasinfo,frac_flag = self.matching_pennies(self.model_hist,self.reward_hist,depth,pval,self.opponent_name,bias, pf) 
                self.f_flag.append(frac_flag)
            else:
                opponent_action,pchooseright,biasinfo = self.matching_pennies(self.model_hist,self.reward_hist,depth,pval,self.opponent_name,bias) 
                self.f_flag.append(False)
            
        # elif self.opponent_name.conta
        else: #other opponents are class based, so we use the step function to update the model with the most recent choices and get the output
            # opponent_action,pchooseright,biasinfo = self.opponent.get_last_saved()
            if len(self.model_hist) > 0:

                opponent_action,pchooseright,biasinfo = self.opponent.step(self.model_hist[-1],self.reward_hist[-1]) #should consider just calculating over history
            else:
                # print(-1)
                opponent_action,pchooseright,biasinfo = self.opponent.get_last_saved()

            if self.opp_kwargs != self.opponent.opp_kwargs:
                self.opp_kwargs = self.opponent.opp_kwargs #fix the saved opponent kwargs if the agent has changed parameters

        self.pright_mp.append(pchooseright)
        self.pr = pchooseright
        self.biases.append(biasinfo)   
        self.seen = False
        trial = {'opponent_action': opponent_action}
        #convert {0,1} action to one hot vector to add into obsself.pchooseright ervation 
        stim_action = [0,0]
        if self.last_opp_choice is not None:
            stim_action[self.last_opp_choice] += 1
        self.last_opp_choice = opponent_action
        
        self.opponent_action = opponent_action
        
            #only use the outcome period as the single step in the episode
        self.add_period(['outcome'])
        self.set_groundtruth(opponent_action,'outcome')

        self.add_ob(stim_action,'outcome', where = 'stimulus') #is this even correct? outcome should be 1 or 0
    
        return trial
    
    
    
    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        obs = self.ob_now
        reward = 0
        done = False
       
        # print(self.opponent_name)
       
        self.action = action 
        self.seen = True

        # if self.in_period('outcome'):
        #we will always be here for episodic
        pr = self.pr
        #reversal_bandit is a 2-bandit task rather than a matching-pennies task
        #this adds the probabilistic condition that the agent can win without the "correct choice" 
        # if isinstance(self.action, (np.ndarray, np.generic,torch.Tensor)):
        if isinstance(self.action, (np.ndarray,torch.Tensor)):
            self.action = self.action.squeeze().argmax()
        # if isinstance(self.opponent_action, (np.ndarray, np.generic,torch.Tensor)):
        #     self.opponent_action = self.action.squeeze()[1]            
        rev_bandit_condition = (self.opponent_name == 'reversalbandit' and rng.random() < (self.action*(pr) + (1-self.action)*(1-pr)))
        if float(self.action) == float(self.opponent_action) or rev_bandit_condition:
        # if (float(self.action) - float(self.opponent_action)).any() or rev_bandit_condition:
            reward = self.rewards['correct']
            self.performance = 1
            flag = True
        else:
            reward = self.rewards['fail']
            flag = True
        self.reward_hist.append(reward)
        self.model_hist.append(self.action) #error here
        self.env_hist.append(self.opponent_action)
        done = self.done
        new_trial = True
        info = {'new_trial': new_trial}# 'gt': self.opponent_action}
        #changed 3rd parameter to new_trial in order to see if mean episode length occurs
        #torch.nn.functional.one_hot(torch.LongTensor(self.opponent_action),num_classes=2)
        
        if self.opponent == 'fractional':
            # use info as proxy for whether MP2 engaged
            info['fractional'] = self.f_flag[-1]
        else:
            info['fractional'] = False  
        return obs, reward, done, info