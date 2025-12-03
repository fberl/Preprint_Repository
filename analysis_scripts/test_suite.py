# import analysis_scripts.QV_analysis as QV_analysis  # Module not available
import analysis_scripts.logistic_regression as logistic_regression
import numpy as np
import torch
import analysis_scripts.WSLS_Analysis as WSLS_Analysis
import analysis_scripts.population_coding as population_coding

def generate_data(model,env,nits=1, qvals = False):
    """
    Given a model and an environment, this function will generate firing rates for episodes. 
    """
    RNNChoices = []
    RLChoices = []
    modelChoices= []
    RNNChoice = []

    def getRNN(model, input, output):
        # RNNChoice.append(output.detach().permute(1,0,2).numpy())
        if qvals:
            RNNChoice.append(torch.softmax(output.detach().permute(1,0,2),dim=-1).squeeze().numpy()[1])
        else:
            RNNChoice.append(np.round(torch.softmax(output.detach().permute(1,0,2),dim=-1).squeeze().numpy()[1]))
        
    if model.policy.RL is not None:
        h1 = model.policy.linear3.register_forward_hook(getRNN)
    
    episode_actions = []
    episode_rewards = []
    episode_states = []
    episode_hiddens = []
    episode_last_rewards = []


    while len(episode_actions) < nits:
        state = env.reset()
        if isinstance(state,int):
            state = torch.Tensor([state])
        if isinstance(state,np.ndarray):
            state = torch.Tensor(state).view(1,1,-1)
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=model.action_dim).view(1,1,-1)
        
        reward = (np.random.rand() > .5) 
        hidden_out = torch.zeros([model.num_rnns,1, model.hidden_dim], dtype=torch.float)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        # hidden_out = torch.randn([self.num_rnns,1, self.hidden_dim], dtype=torch.float,device=device)  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        
        if model.policy.RL is not None: 
            RNNChoice = []
            RLChoice = []
            model.policy.RL.reset()
        rewards = []
        done = False
        rewards = [] 
        actions = []
        states = []
        hiddens = []
        for step in range(model.max_steps):
            # last_action = action
            states.append(state.argmax().numpy())
            hidden_in = hidden_out
            last_reward = reward * last_action
            action, policy, value, hidden_out = model.forward(state, last_action, last_reward, hidden_in) # MAKE IT TAKE PREVIOUS REWARD FOR RL 
            next_state, reward, done, _ = env.step(action)
            action = action.to(torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32).view(1,1,-1)
            hiddens.append(hidden_out.squeeze().view(-1,1))
            if step == model.max_steps - 1:
                done = True
            # episode_state.append(state)
            actions.append(action.detach().squeeze().argmax().numpy())
            rewards.append(reward)
            if model.policy.RL is not None:
                if qvals:
                    RLChoice.append(torch.nn.functional.softmax(model.policy.RL.Qs.detach(),dim=-1).squeeze().numpy()[1])#IS THSI RIGHT
                else:
                    RLChoice.append(np.round(torch.nn.functional.softmax(model.policy.RL.Qs.detach(),dim=-1).squeeze().numpy()[1]))#IS THSI RIGHT
            

            state = next_state
            last_action = action
            if done:
                break
        # episode_state = torch.cat(episode_state,dim=1)
        # episode_action = torch.cat(episode_action, dim=1)
        # episode_reward = torch.tensor(episode_reward).requires_grad_(False).view(1,-1,1)
        
        
        episode_actions.append(np.vstack(actions))
        episode_rewards.append(np.vstack(rewards))
        episode_states.append(np.vstack(states))
        episode_hiddens.append(torch.hstack(hiddens))
        if model.policy.RL is not None:
            RNNChoices.append(np.vstack(RNNChoice))
            RLChoices.append(np.vstack(RLChoice))
        # last_action = env.action_space.sample()
        # last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2).numpy()


    
    # episode_states, episode_actions, episode_rewards, episode_hiddens = format_data(episode_states, episode_actions, episode_rewards, episode_hiddens)
    if model.policy.RL is not None: 
        h1.remove()
        
    return episode_states, episode_actions, episode_rewards, episode_hiddens, RNNChoices, RLChoices

def test_suite(model, env, nits=100, order = 12, save = ''):
    data = generate_data(model,env,nits)
    if len(save) == True:
        np.save(save+"_data.npy",data)
    
    # visualize Q values
    if model.policy.RL is None:
        # QV_analysis.measure_Qs_test_suite(model, env, save = save)  # Module not available
        # look at WSLS behavior
        WSLS_Analysis.WSLS_Analysis_pregenerated(data, save = save)
    else:
        # QV_analysis.Qs_WSLS_RL(model, env, data, save = save)  # Module not available
        pass
    # fit logistic regression to data
    logistic_regression.model_logistic_regression_pregenerated(model, data, order=order, save = save,  err = True, sliding=30)
    # look at population coding
    # population_coding.combined_coding(model, env, order, nits, data, save = save)
    # if model.policy.RL is not None:
    #     population_coding.RL_coding(model, data,order = order)
    
def combinatorial_test(model, env, nits = 100, order = 12):
    data = generate_data(model,env,nits)
    
    logistic_regression.model_logistic_regression_combinatorial(model, data, order=order)
    
def logistic_test(model, env, nits = 100, order = 12, sliding = 30):
    data = generate_data(model,env,nits)
    
    logistic_regression.model_logistic_regression_pregenerated(model, data, order=order,  err = True, sliding=sliding)