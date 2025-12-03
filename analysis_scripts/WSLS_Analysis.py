import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import envs.mp_env
from analysis_scripts.yule_walker import yule_walker, pacf
import os
import sqlite3
import pandas as pd

# model_num = sys.argv[1]

# set_plotting_params()
mac = False
device_idx = 0
if mac: 
    device = torch.device("mps")
else :
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

print(device)

rng = np.random.default_rng()


def convert_params_env(parameters):
    '''
    Using parameters, make an environment 
    Input: 
        params (dict): the paramters to make the environment from. This should be of structure train_params['env'] 
        useparams (boolean): whether or not to use the default opponent
    Output:
        env (neurogym environment)
    '''
    opponent_dist_kwargs = parameters['opponents_params']
    opp = rng.choice(list(parameters['opponents_params'])) #randomly draw at the start
    if opp == "lrplayer":       
        #because this doesn't work with the same random draw, we need to specifically draw the parameters
        opp_kwargs = {}
        opp_kwargs['choice_betas'] = opponent_dist_kwargs[opp]['choice_betas']
        opp_kwargs['outcome_betas'] = opponent_dist_kwargs[opp]['outcome_betas']
    else:
        # opp_kwargs = {k:rng.choice(v) for k,v in opponent_dist_kwargs[opp].items()}
        opp_kwargs = opponent_dist_kwargs[opp]
    return opp_kwargs


def test_model(env,model, hidden_in = None):
    state =  torch.tensor(env.reset())
    last_action = env.action_space.sample()
    last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2)
    episode_state = []
    episode_action = []
    episode_last_action = []
    episode_reward = []
    er = 0
    episode_next_state = []
    episode_done = []
    episode_hidden_states = []
    try:
        model.policy.reset()
    except:
        pass
        
    hidden_out = torch.zeros([model.num_rnns,1, model.hidden_dim], dtype=torch.float,device=device)
    hidden_out = hidden_in
    
    for step in range(min(model.max_steps,env.reset_time)):
        hidden_in = hidden_out
        action,_, hidden_out = model.policy.get_action(state, last_action, hidden_in, deterministic = True)
        next_state, reward, done, _ = env.step(action)
        episode_state.append(state)
        episode_action.append(action)
        episode_last_action.append(last_action)
        episode_reward.append(reward)
        episode_next_state.append(next_state)
        episode_done.append(done) 
        episode_hidden_states.append(hidden_out)
        
        state=next_state
        last_action = action
    
    return (episode_action, episode_state, episode_reward, episode_hidden_states)

def environment_generator(env_params):

    env = envs.mp_env.MPEnv(fixed_length=env_params['fixed_length'],reset_time=env_params['reset_time'],
                            opponent=env_params['opponent'],
                            opp_params=env_params['opponents_params'])                                                      
    
    
    return env


def WSLS_Analysis(model1, params, ntests = 500,ntrials=1):

    '''
    Given input arguments to the script, test the agent on win-stay/lose-switch variations.
    saves data, then analyze and plot the WSLS prob as a function of consecutive trials, as well as reward.
    If the opponent can be initialized with many different parameters, run with a higher number of ntests
    Arguments:
        args (ArgParse): arguments from script:
            model: the model to analyze
            params: parameters used for model and environment
            ntests: number of tests to run in order to generate error bars 
            ntests: how many consecutive trials to run the analysis for
    Returns:
        None
    '''
    
    opp_name = params['env_params']['opponent']
    env = environment_generator(params['env_params'])

    trial_len = env.reset_time


    stayprob = torch.zeros((ntests,trial_len-1))
    switchprob = torch.zeros((ntests,trial_len-1))
    rewprob = torch.zeros((ntests,trial_len-1))
    _, axes = plt.subplots(1,1,figsize=(7,5),sharey=True)

    models = [model1]
    for modelindex, model in enumerate(models):
        episodes = []
        for i in range(ntests):
            env.draw_opponent()
            for j in range(ntrials):
                if j == 0:
                    hidden_in = torch.zeros([model.num_rnns,1, model.hidden_dim], dtype=torch.float,device=device)
                else:
                    hidden_in = episode_hidden_states[-1]
                            
                trials_dat = test_model(env,model,hidden_in=hidden_in)
                episodes.append(trials_dat)
                
                _, _, _, episode_hidden_states = trials_dat
                env.reset()
                
        for k,data in enumerate(episodes):
            actions, states, rewards, hidden_outs = data
            stay = [actions[i+1][1] == actions[i][1] if rewards[i] == 1 else 0 for i in range(len(actions)-1)]
            switch = [actions[i+1][1] != actions[i][1]  if rewards[i] == 0 else 0 for i in range(len(actions)-1)]
            pstay = np.cumsum(stay) / np.arange(1,len(actions))
            pswitch = np.cumsum(switch) / np.arange(1,len(actions))
            rew = np.cumsum(rewards) / (np.arange(len(actions))+1)            
            stayprob[k] = torch.FloatTensor(pstay)
            rewprob[k] = torch.FloatTensor(rew[:-1])
            switchprob[k] = torch.FloatTensor(pswitch)

        plt.rcParams['font.size'] = 10

        for j,data in enumerate([stayprob/rewprob,switchprob/(1-rewprob),(stayprob/rewprob+switchprob/(1-rewprob))/2,rewprob]):
            if j == 0:
                l = "Win Stay"
            if j == 1:
                l = "Lose Switch"
            if j == 2:
                l = 'Win Stay + Lose Switch'
            elif j == 3:
                l = "Reward "
            stderr = torch.std(data,axis=0).numpy() / np.sqrt(ntests) #stderr
            m = torch.nanmean(data,axis=0).numpy()
            upper = m + 0.5*stderr
            lower = m - 0.5*stderr
            
            axes.plot(m,label=l)
            axes.plot(upper, color='tab:blue', alpha=0.1)
            axes.plot(lower, color='tab:blue', alpha=0.1)
            axes.fill_between(np.arange(ntrials),lower, upper, alpha=0.2)
    plt.ylim(0,1.1)
    axes.set_xlabel("Trial")
    axes.set_ylabel("Probability")
    BG = 'PFC'
    if model.policy.QRL is not None:
        BG = 'PFC + BG'
    axes.set_title('WSLS {}{}'.format(opp_name, BG) )
    lgd = axes.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    plt.tight_layout()
    plt.show()

def autoAnalysis(model, params, numtests = 10):
    '''
    Given a model, looks at the autocorrelation between the decisions values as well as the resulting behavior, i.e. WSLS
    '''
    
    '''
    Given input arguments to the scrip, test the agent on win-stay/lose-switch variations 
    saves data, then analyze and plot
    Arguments:
        args (ArgParse): arguments from script:
            runindex: model run index of the trained agent
            modeltype: whether the model is SSP or default A2C
            trainiters: how long the model has trained for
            ntests: how long the model should be tested for
    Returns:
        None
    '''
    
    opp_name = params['env_params']['opponent']
    env = environment_generator(params['env_params'])

    actions = []
    rewards = []
    for i in range(numtests):
        env.reset()
        data = test_model(env,model)
        actions.append(data[0])
        rewards.append(data[2])
    ars = []
    pacfs = []
    converted_actions = []
    converted_rewards = []
    for i in range(numtests):
        c = []
        r = []
        for j in range(len(actions[i])):
            c.append(actions[i][j].numpy()[1])
            r.append(rewards[i][j])
        converted_actions.append(c)
        converted_rewards.append(r)
    for action in converted_actions:
        ar = yule_walker(action,4)
        plt.plot(ar)
        ars.append(ar)
    plt.title('Autoregression coefficients for current choice based on previous')
    plt.show()

    for action in converted_actions:
        pcfs = pacf(action,20)
        plt.plot(pcfs)
        pacfs.append(ar)
    plt.title('PACF for choice')
    plt.show()
    
    #REGRESS PROBABILITY OF STAYING BASED ON PREVIOUS REWARDS:
    stays = np.zeros_like(converted_actions)[:,:-1]
    for i in range(numtests):
        stays[i] = np.equal(converted_actions[i][:-1], converted_actions[i][1:])
    
    for action in stays:
        ar = yule_walker(action,4)
        plt.plot(ar)
        ars.append(ar)   
    
    plt.title('Autoregression for Stay Probability based on previous rewards') 
    plt.show()
    
    for action in stays:
        pcfs = pacf(action,20)
        plt.plot(pcfs)
        pacfs.append(ar)
    plt.title('PACF for rewards')
    plt.show()
    
def WSLS_Analysis_pregenerated(data, save = '', axes = None):

    '''
    Given input arguments to the script, test the agent on win-stay/lose-switch variations.
    saves data, then analyze and plot the WSLS prob as a function of consecutive trials, as well as reward.
    If the opponent can be initialized with many different parameters, run with a higher number of ntests
    Arguments:
        args (ArgParse): arguments from script:
            model: the model to analyze
            data : data to be analyzed
            ntests: number of tests to run in order to generate error bars 
            ntests: how many consecutive trials to run the analysis for
    Returns:
        None
    '''
    

    

    episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, BGChoices = data
    ntrials = len(episode_actions[0])
    ntests = len(episode_actions)
    stayprob = np.zeros((ntests,ntrials-1))
    switchprob = np.zeros((ntests,ntrials-1))
    rewprob = np.zeros((ntests,ntrials-1))
    if axes is None:
        _, axes = plt.subplots(1,1,figsize=(7,5),sharey=True)
        
    for k in range(len(episode_actions)):
        actions = episode_actions[k]
        rewards = episode_rewards[k]
        stay = [bool(actions[i+1] == actions[i]) if rewards[i] == 1 else 0 for i in range(len(actions)-1)]
        switch = [bool(actions[i+1]!= actions[i]) if rewards[i] == 0 else 0 for i in range(len(actions)-1)]
        assert len(actions) == len(rewards)
        # assert len(actions) == ntrials
        pstay = np.cumsum(stay) / np.arange(1,len(actions))
        pswitch = np.cumsum(switch) / np.arange(1,len(actions))
        rew = np.cumsum(rewards) / (np.arange(len(actions))+1)            
        stayprob[k] = pstay
        rewprob[k] =rew[:-1]
        switchprob[k] = pswitch
        plt.rcParams['font.size'] = 10

    for j,dat in enumerate([stayprob/rewprob,switchprob/(1-rewprob),(stayprob/rewprob+switchprob/(1-rewprob))/2,rewprob]):
        if j == 0:
            l = "Win Stay"
        if j == 1:
            l = "Lose Switch"
        if j == 2:
            l = 'Win Stay + Lose Switch'
        elif j == 3:
            l = "Reward "
        # m = gaussian_filter(data.mean(axis=0),sigma=1)
        stderr = np.std(dat,axis=0)/ np.sqrt(ntests) #stderr
        m = np.nanmean(dat,axis=0)
        upper = m + 0.5*stderr
        lower = m - 0.5*stderr
        
        axes.plot(m,label=l)
        axes.plot(upper, color='tab:blue', alpha=0.1)
        axes.plot(lower, color='tab:blue', alpha=0.1)
        axes.fill_between(np.arange(ntrials-1),lower, upper, alpha=0.2)
    plt.ylim(0,1)
    axes.set_xlabel("Trial")
    axes.set_ylabel("Probability")
    axes.axhline(0.5, color='black', linestyle='--',label='optimal (random actor)')

    axes.set_title('WSLS Characterization')
    lgd = axes.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    plt.tight_layout()
    if save is not '':
        plt.savefig(save+"_WSLSbehavior.png",dpi=200)
    plt.show()
    

def query_behavioral(task="mp_trials"):
    query = """
    SELECT sessions.id, t.trial, t.monkey_choice, t.reward, t.computer_choice, sessions.animal, sessions.name, sessions.ord
        FROM {task} as t, {task} as tp, sessions WHERE
            tp.session_id == t.session_id AND -- Alignment
            t.session_id == sessions.id AND
            tp.trial == t.trial - 1 AND -- Select previous trial
            tp.session_id == t.session_id AND
            t.tr_good == 1 
    ;
    """.format(task=task)
    return query

def query_monkey_behavior(path = None):
    '''Gets monkey behavior for MP and Search for each opponent type face and each monkey'''    
    if path is None:
        con = sqlite3.connect("matching-pennies-lite.sqlite")
    else:
        con = sqlite3.connect(path)

    pdMP = pd.read_sql_query(query_behavioral(),con)
    pdMP['task'] = 'mp'
    
    pdS = pd.read_sql_query(query_behavioral(task='search_trials'),con)
    pdS['task'] = 'search'
    
    
    df = pd.concat([pdMP,pdS]).sort_values(by=['id','trial'])

    return df

def parse_monkey_behavior(df, order):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_monkey(sessdat['monkey_choice'].to_numpy(),sessdat['computer_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order)
        regressors.append(data)
    reg = np.vstack(regressors)
    return reg

# def staggered_mean(datum):
#     '''computes weighted mean from staggered matrix'''

def data_parse_monkey(data, lens = 200):
    '''Parses choice, reward data into WSLS format'''
    data[data['']]
    
    # loot over sessions and figure out which have more than len entries
    sessions = []
    data = data.sort_values(by=['id','trial'])
    var = {}
    err = {}
    
    choices = []
    rewards = []
    
    stayprob = []
    switchprob = []
    rewprob = []
    triallens = []
    wsls = []
    
    # stayerrs = []
    # switcherrs = []
    # rewerrs = []
    # wslserrs = []
    
    
    for sess in data['id'].unique():
        sessdata = data[data['id'] == sess]
        
        
        choice = sessdata['monkey_choice'].numpy()
        reward = sessdata['reward'].numpy()
        
        stay = [choice[i+1] == choice[i] if reward[i] == 1 else 0 for i in range(len(choice)-1)]
        switch = [choice[i+1]!= choice[i]  if reward[i] == 0 else 0 for i in range(len(choice)-1)]
        assert len(choice) == len(rewards)
        pstay = np.sum(stay) / np.arange(1,len(choice))
        pswitch = np.sum(switch) / np.arange(1,len(choice))
        rew = np.sum(rewards) / (np.arange(len(choice))+1)   
        n = len(choice)
        triallens.append(n)
        stayprob.append(pstay/rew)
        switchprob.append(pswitch/(1-rew))
        wsls.append((pstay/rew+pswitch/(1-rew))/2)
        
        

        
        if len(sessdata) > lens:
            sessions.append(sess)
            choices.append(choice[:lens])
            rewards.append(reward[:lens])
    
    
    choices = np.vstack(choices)
    rewards = np.vstack(rewards)
    
    stayprob = np.array(stayprob)
    switchprob = np.array(switchprob)
    rewprob = np.array(rewprob)
    triallens = np.array(triallens)
    wsls = np.array(wsls)
    
    wsq = np.sqrt(np.sum(triallens**2))
    err['WinStay'] = np.std(stayprob) * wsq
    err['LoseSwitch'] = np.std(switchprob) * wsq
    err['Reward'] = np.std(switchprob) * wsq
    err['WSLS'] = np.std(wsls) * wsq
    
    stayprob = np.sum(triallens*stayprob)/np.sum(triallens)
    switchprob = np.sum(triallens*switchprob)/np.sum(triallens)
    rewprob = np.sum(triallens*rewprob) / np.sum(triallens)
    wsls = np.sum(triallens * wsls) / np.sum(triallens)
    
    var['WinStay'] = stayprob
    var['LoseSwitch'] = switchprob
    var['Reward'] = rewprob
    var['WSLS'] = wsls
    

    
    return choices, rewards, var, err
    
def WSLS_Single(actions, rewards, window_width = 10, axes = None):
    
    '''
    Given a single episode, computes the WSLS probabilities isomg a sliding window
    '''
    

    

    # episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, BGChoices = data
    
    ntrials = len(rewards)
    stayprob = np.zeros((ntrials-1))

    if axes is None:
        _, axes = plt.subplots(1,1,figsize=(7,5),sharey=True)
        

    stay = [bool(actions[i+1] == actions[i]) if rewards[i] == 1 else 0 for i in range(len(actions)-1)]
    switch = [bool(actions[i+1]!= actions[i]) if rewards[i] == 0 else 0 for i in range(len(actions)-1)]
    assert len(actions) == len(rewards)
    # assert len(actions) == ntrials
    # pstay = np.cumsum(stay) / np.arange(1,len(actions))
    # pswitch = np.cumsum(switch) / np.arange(1,len(actions))
    rew = np.cumsum(rewards) / (np.arange(len(actions))+1)            
    # stayprob = pstay
    # rewprob =rew[:-1]
    # switchprob = pswitch
    plt.rcParams['font.size'] = 10

    meaner = np.ones(window_width)/window_width
    stayprob = np.convolve(stay,meaner,mode='valid')
    rewprob = np.convolve(rewards,meaner,mode='valid')[:-1]
    switchprob = np.convolve(switch,meaner,mode='valid')
    
    

    for j,dat in enumerate([stayprob/rewprob,switchprob/rewprob,rewprob]):
        if j == 0:
            l = "Win Stay"
        if j == 1:
            l = "Lose Switch"
        elif j == 2:
            l = "Reward "

        # m = gaussian_filter(data.mean(axis=0),sigma=1)
        # stderr = np.std(dat,axis=0)/ np.sqrt(window_width) #stderr
        # m = np.nanmean(dat,axis=0)
        # upper = m + 0.5*stderr
        # lower = m - 0.5*stderr
        
        axes.plot(dat,label=l)
        # axes.plot(upper, color='tab:blue', alpha=0.1)
        # axes.plot(lower, color='tab:blue', alpha=0.1)
        # axes.fill_between(np.arange(ntrials-1),lower, upper, alpha=0.2)
    plt.ylim(0,1)
    axes.set_xlabel("Trial")
    axes.set_ylabel("Probability")
    axes.axhline(0.5, color='black', linestyle='--',label='optimal (random actor)')

    axes.set_title('WSLS Characterization')
    lgd = axes.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    plt.tight_layout()
    plt.show()
    
def WSLS_Single_Variable(actions, rewards, module_selection, window_width=10, axes = None):
    
    '''
    Given a single episode, computes the WSLS probabilities isomg a sliding window
    '''
    

    

    # episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, BGChoices = data
    
    ntrials = len(rewards)
    stayprob = np.zeros((ntrials-1))

    if axes is None:
        _, axes = plt.subplots(1,1,figsize=(7,5),sharey=True)
        

    stay = [bool(actions[i+1] == actions[i]) if rewards[i] == 1 else 0 for i in range(len(actions)-1)]
    switch = [bool(actions[i+1]!= actions[i]) if rewards[i] == 0 else 0 for i in range(len(actions)-1)]
    assert len(actions) == len(rewards)
    # assert len(actions) == ntrials
    # pstay = np.cumsum(stay) / np.arange(1,len(actions))
    # pswitch = np.cumsum(switch) / np.arange(1,len(actions))
    rew = np.cumsum(rewards) / (np.arange(len(actions))+1)            
    # stayprob = pstay
    # rewprob =rew[:-1]
    # switchprob = pswitch
    plt.rcParams['font.size'] = 10

    meaner = np.ones(window_width)/window_width
    stayprob = np.convolve(stay,meaner,mode='valid')
    rewprob = np.convolve(rewards,meaner,mode='valid')[:-1]
    switchprob = np.convolve(switch,meaner,mode='valid')
    domprob =  np.convolve(1/2*(module_selection+1),meaner,mode='valid')


    for j,dat in enumerate([stayprob/rewprob,switchprob/rewprob,rewprob, domprob]):
        if j == 0:
            l = "Win Stay"
        if j == 1:
            l = "Lose Switch"
        elif j == 2:
            l = "Reward "
            continue
        elif j == 3:
            l = "Dominance"
        # m = gaussian_filter(data.mean(axis=0),sigma=1)
        # stderr = np.std(dat,axis=0)/ np.sqrt(window_width) #stderr
        # m = np.nanmean(dat,axis=0)
        # upper = m + 0.5*stderr
        # lower = m - 0.5*stderr
        
        axes.plot(dat,label=l)
        # axes.plot(upper, color='tab:blue', alpha=0.1)
        # axes.plot(lower, color='tab:blue', alpha=0.1)
        # axes.fill_between(np.arange(ntrials-1),lower, upper, alpha=0.2)
    plt.ylim(0,1)
    axes.set_xlabel("Trial")
    axes.set_ylabel("Probability")
    axes.axhline(0.5, color='black', linestyle='--',label='optimal (random actor)')

    axes.set_title('WSLS Characterization')
    lgd = axes.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    plt.tight_layout()
    plt.show()
    
    
def WSLS_Monkey(df = None, db_path=None, monkey=None, task = None, lens=200):
    
    '''
    pulls monkey data, then formats and passes to WSLS_pregenerated
    if trial len < len, exclude from plot. if >, truncate. show average WSLS somewhere, i.e. side box
    '''
    
    if df is None:
        monkey_dat = query_monkey_behavior(db_path)
    else:
        monkey_dat = df
    if not isinstance(task,list):
        task = [task]
    if not isinstance(monkey,list):
        monkey = [monkey]
        
    
    choices, rewards, var, err = data_parse_monkey(monkey_dat, lens = lens)
    
    dat = [[],choices,rewards,[],[],[]]
    

    WSLS_Analysis_pregenerated(dat)
    
    for key in var.keys():
        print('{} prob: {}, err: {}'.format(key, var[key], err[key]))
        