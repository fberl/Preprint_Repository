"""
Archived/unused functions from logistic_regression.py
These functions are not currently used in the codebase but are preserved for reference.
"""

from scipy.optimize import minimize, least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats
import seaborn as sns

# Import functions from main logistic_regression that archived functions may need
from analysis_scripts.logistic_regression import (
    sigmoid, logit, logistic_regression_colinear, data_parse,
    logistic_regression_no_bias, logistic_regression,
    query_monkey_behavior, parse_monkey_behavior, parse_monkey_behavior_reduced,
    parse_monkey_behavior_strategic, parse_monkey_behavior_combinatorial,
    data_parse_monkey, monkey_logistic_regression, create_order_data,
    general_logistic_regressors, fit_glr, data_parse_sliding_strategic,
    data_parse_sliding_WSLS
)


# Originally at line 45
def logit_monkey(x,w,b):
    x = sigmoid(np.dot(x,w) + b)
    y = sigmoid(b)
    x[(x == y) & (y!=.5)] = .5
    return x
#0 left, 1 right; x can take values from {-1,1} or {0,1} depending on predictor
# def objective_function(theta, X, y):
#     m = len(y)
#     h = logit(X,theta[:-1],theta[-1])
#     cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
#     Xaugmented = np.concatenate((X, np.ones((m, 1))), axis=1)
#     grad = (1 / m) * Xaugmented.T.dot(h - y)
#     return cost, grad


# Originally at line 216
def objective_function_reduced(theta, X, y, mask):
    m = len(y)
    l2 = .01
    l1 = 0

    if mask is not None:
        y = y[mask]
        X = X[mask]
    h = logit(X,theta,0)
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta**2) + l1*np.abs(theta).sum()
    try:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta) # extra terms are for computing the gradient
    except:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta)
    
    
    grad = (1 / m) * Xaugmented.T.dot(h - y)
    return cost, grad


# Originally at line 236
def objective_function_l2(theta, X, y):
    m = len(y)
    h = logit(X,theta[:-1],theta[-1])
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + 0.5 * np.sum(theta**2)
    # Xaugmented = np.concatenate((X, np.ones((m, 1))), axis=1)
    # grad = (1 / m) * Xaugmented.T.dot(h - y)
    return cost


# Originally at line 246
def objective_function_nojac(theta, X, y):
    m = len(y)
    h = logit(X,theta[:-1],theta[-1])
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    return cost


# Originally at line 553
def computeSigma(fit, method, y):
    errfunc = lambda p, method, y: method(*p) - y
    hessian = np.diag(np.linalg.inv(fit['jac'].T @ fit['jac']))
    MSE = (errfunc(fit['x'], method, y)**2).sum()/(len(y)-len(fit['x']))

    return np.sqrt(hessian * MSE)

# format data based on policy network behaviors
#coded inefficiently for readability, since efficiency doesn't matter for this function

# def data_parse(actions, state, rewards, order=1):
    
#     # variables of interest: 
#     # {-1,1} for agent choice in previous trial (order 1-n) 
#     # {-1,1} for computer choice in previous trial (order 1-n)
#     # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
#     # {-1, 0, 1} if left and not rewarded on prior trial, rewarded, or right rewarded (basically, swap on win)
#     numvars = 4
    
#     data = np.zeros((len(actions) - order,numvars,order))
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             for k in range(0,data.shape[2]):
#                 if j==0: #agent choice
#                     if actions[i-k-1] == 0:
#                         data[i,j,k] = -1
#                     else:
#                         data[i,j,k] = 1
#                 if j==1: #opponent choice
#                     if state[i-k] == 0:
#                         data[i,j,k] = -1
#                     else:
#                         data[i,j,k] = 1
#                 if j==2: #rewarded and stay
#                     if rewards[i-k-1] == 1:
#                         if actions[i-k-1] == 0:
#                             data[i,j,k] = -1
#                         else:
#                             data[i,j,k] = 1
#                 if j==3: #rewarded and swap
#                     if rewards[i-k-1] == 0:
#                         if actions[i-k-1] == 0:
#                             data[i,j,k] = -1
#                         else:
#                             data[i,j,k] = 1
#     # flatten last two dimensions to combine them. I think this needs to be done carefully
#     # or the regressors will get mixed up
#     data = data.reshape(data.shape[0],-1, order='F')
#     return data


# Originally at line 680
def model_logistic_regression(model,env, nits=10, order=5, predict_on_choice=False):

    PFCChoices = []
    RLChoices = []
    modelChoices= []
    
    def getPFC(model, input, output):
        # PFCChoice.append(output.detach().permute(1,0,2).numpy())
        PFCChoice.append(np.round(torch.softmax(output.detach().permute(1,0,2),dim=-1).squeeze().numpy()[1]))

        
    def getBG(model, input, output):
        RLChoice.append(output.detach())
    
    def getModel(model, input, output):
        modelChoice.append(output.detach())
    
    if model.policy.RL is not None:
        h1 = model.policy.linear3.register_forward_hook(getPFC)

    env.reset()
    rewards = [] 
    actions = []
    states = []
    
    regressors = []

    env.clear_data()

    if isinstance(order,int):
        order = [order]
        
        
    episode_rewards = []
    episode_actions = []
    episode_states = []
    for eps in range(nits):
        state =  env.reset()
        if model.policy.RL is not None: 
            PFCChoice = []
            RLChoice = []
            model.policy.RL.reset()
        rewards = []
        actions = []
        states = []
        modelChoice = []
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2).numpy()
        hidden_out = torch.zeros([model.num_rnns,1, model.hidden_dim], dtype=torch.float)
        
        for step in range(model.max_steps):
            states.append(state[1])
        
            hidden_in = hidden_out                
            action, log_probs, hidden_out = model.policy.get_action(state, last_action, hidden_in, deterministic = model.DETERMINISTIC)
            
            actions.append(action.detach().squeeze().numpy()[1])   
            if model.policy.RL is not None:
            #     PFCChoice.append(torch.softmax(model.policy.RL.PFC.Q.detach().permute(1,0,2),dim=-1).numpy()[1])
                RLChoice.append(np.round(F.softmax(model.policy.RL.Qs.detach(),dim=-1).squeeze().numpy()[1]))#IS THSI RIGHT
                
            choice = log_probs.detach().exp().squeeze().numpy()
            modelChoice.append(choice[1])
            # modelChoice.append(np.log(choice[1]/choice[0]))
                
            next_state, reward, done, _ = env.step(action[1])
            rewards.append(reward)
            # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
            state = next_state
            last_action = action
            if done:
                break
        episode_rewards.append(rewards)
        episode_actions.append(actions)
        episode_states.append(states)    
        modelChoices.append(modelChoice)
        if model.policy.RL is not None:
            PFCChoices.append(PFCChoice)
            RLChoices.append(RLChoice)
        
    if model.policy.RL is not None: 
        h1.remove()


    plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards)
    all_sols = {}
    
    for ord in order:
        sols = {}
        ord = min(ord, len(episode_actions[0]) - 1)
        if model.policy.RL is not None:
            pfC = np.vstack(PFCChoices)[:,ord:].ravel()
            bgC = np.vstack(RLChoices)[:,ord:].ravel()
        modelC = np.vstack(modelChoices)[:,ord:].ravel()
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse(episode_actions[i], episode_states[i], episode_rewards[i], ord))

        regressors = np.vstack(regressors)
    
        
        
        all_actions = np.vstack(episode_actions)[:,ord:].ravel()
        # return logistic_regression(regressors,all_actions)
                    
        prob_regression = logistic_regression(regressors,modelC)
        action_regression = logistic_regression(regressors, all_actions)

        # prob_regression = logistic_regression_l2(regressors,modelChoices)
        # action_regression = logistic_regression_l2(regressors, all_actions)


        sols['prob'] = prob_regression.x
        sols['action'] = action_regression.x
        
        # sigma = prob_regression.hess_inv
        # sols['prob_err'] = prob_regression.
        # sols['action_err']
        
        if model.policy.RL is not None:
            PFCChoices = np.vstack(PFCChoices)
            RLChoices = np.vstack(RLChoices)
            
            pfc_regression = logistic_regression(regressors,pfC)
            bg_regression = logistic_regression(regressors,bgC)
            sols['pfc'] = pfc_regression.x
            sols['bg'] = bg_regression.x
            
        

 
        # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
        # ax1.set_xlabel(r'\theta^T x + b')
        # action_fit = regressors @ sols['prob'][:-1] + sols['prob'][-1]
        # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        # ax1.set_title(r'Network P(Right) regression. $accuracy= {}$'.format(acc))
        # smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
        # ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        # ax1.set_ylabel('action')
        # bins = np.linspace(min(action_fit),max(action_fit),20)
        # hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        # bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        # hist1 = hist1[0]
        # hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        # bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        # hist2 = hist2[0]
        # kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
        # kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
        # ax1.set_xlim(min(bins),max(bins))
        # ax1.plot(smigmoid,kde1, color = 'b')
        # ax3 = ax1.twinx()
        # ax3.invert_yaxis()
        # ax3.plot(smigmoid,kde2, color = 'b')        
        # ax1.axvline(0,0)        
        # ax1.set_ylim(0,1)
        # ax3.set_ylim(1,0)
        
        
        
        # ax2.set_xlabel(r'trial number')
        # action_fit = regressors @ sols['prob'][:-1] + sols['prob'][-1]
        # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        # xord = np.arange(1,1+ord)
        # ax2.plot(xord,sols['prob'][:ord])
        # ax2.plot(xord,sols['prob'][ord:2*ord])
        # ax2.plot(xord,sols['prob'][2*ord:3*ord])
        # ax2.plot(xord,sols['prob'][3*ord:4*ord])
        # ax2.set_title(r'probability regression coefficients'.format(acc))
        # ax2.set_ylabel('coefficient')
        # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
        
        
        
        # fig.show()

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
        ax1.set_xlabel(r'$\theta^T x + b$')
        action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
        smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
        ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        ax1.set_ylabel('action')
        bins = np.linspace(min(action_fit),max(action_fit),20)
        hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        hist1 = hist1[0]
        hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        hist2 = hist2[0]
        kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
        kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
        ax1.set_xlim(min(bins),max(bins))
        ax1.plot(smigmoid,kde1, color = 'b')
        ax3 = ax1.twinx()
        ax3.invert_yaxis()
        ax3.plot(smigmoid,kde2, color = 'b')        
        ax1.axvline(0,0)        
        ax1.set_ylim(0,1)
        ax3.set_ylim(1,0)
        ax3.set_yticks([0,1])
        ax3.set_yticklabels(['Right','Left'])
        
        xord = np.arange(1,1+ord)

        ax2.set_xlabel(r'trial number')
        action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax2.plot(xord,sols['action'][:ord])
        ax2.plot(xord,sols['action'][ord:2*ord])
        ax2.plot(xord,sols['action'][2*ord:3*ord])
        ax2.plot(xord,sols['action'][3*ord:4*ord])
        ax2.set_title(r'action regression coefficients'.format(acc))
        ax2.set_ylabel('coefficient')
        ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
        ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
        
        # fig, ax1 = plt.subplots(1,1,figsize=(10,4))   
        # ax1.plot(smigmoid,kde1, color = 'b')
        # ax1.plot(smigmoid,kde2, color = 'r')

        
        # fig.show()
        
        if model.policy.RL is not None:


            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
            ax1.set_xlabel(r'$\theta^T x + b$')
            action_fit = regressors @ sols['pfc'][:-1] + sols['pfc'][-1]
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax1.set_title(r'Network PFC Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(action_fit),max(action_fit),20)
            hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])



            ax2.set_xlabel(r'trial number')
            action_fit = regressors @ sols['pfc'][:-1] + sols['pfc'][-1]
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax2.plot(xord,sols['pfc'][:ord])
            ax2.plot(xord,sols['pfc'][ord:2*ord])
            ax2.plot(xord,sols['pfc'][2*ord:3*ord])
            ax2.plot(xord,sols['pfc'][3*ord:4*ord])
            ax2.set_title(r'PFC regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

            # fig.show()

            
            
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
            ax1.set_xlabel(r'$\theta^T x + b$')
            action_fit = regressors @ sols['bg'][:-1] + sols['bg'][-1]
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax1.set_title(r'Network RL Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(action_fit),max(action_fit),20)
            hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])   
            
            ax2.set_xlabel(r'trial number')
            action_fit = regressors @ sols['bg'][:-1] + sols['bg'][-1]
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax2.plot(xord,sols['bg'][:ord])
            ax2.plot(xord,sols['bg'][ord:2*ord])
            ax2.plot(xord,sols['bg'][2*ord:3*ord])
            ax2.plot(xord,sols['bg'][3*ord:4*ord])
            ax2.set_title(r'RL regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

        if len(order) == 1:
            all_sols = sols
        else:
            all_sols[ord] = sols
    if model.policy.RL is not None:
        
        xord = np.arange(1,1+12)
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.set_title('PFC AR coefficients')
        ax2.set_title('PFC PACF')   
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax1.set_ylabel('coefficient')
        for i in range(PFCChoices.shape[1]):
            try:
                ax1.plot(xord,yw.yule_walker(PFCChoices[i],12),alpha=.02, c='b')
            except:
                continue
            try:
                ax2.plot(xord,yw.pacf(PFCChoices[i],12),alpha=.02, c='b')
            except:
                continue

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.set_title('RL AR coefficients')
        ax2.set_title('RL PACF')   
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax1.set_ylabel('coefficient')
        for i in range(RLChoices.shape[1]):
            try:
                acf = yw.yule_walker(RLChoices[i],12)
                if np.abs(sum(acf)) < 1e4:
                    ax1.plot(xord,acf,alpha=.02, c='b')
            except:
                continue
            try:
                pacf = yw.pacf(RLChoices[i],12)
                if np.abs(sum(pacf)) < 1e4:
                    ax2.plot(xord,pacf,alpha=.02, c='b')
            except:
                continue
        
        
    plt.show()

    return all_sols


#changes the weights and looks for a more monotonic decay.
# def reweighted_logistic_test()


# take a model for RL only and generates a logistic regression


# Originally at line 1048
def model_logistic_BG(model,env, nits=10, order=12, predict_on_choice=False, bias = False, interval = 200, combinatorial = False):
    PFCChoices = []
    RLChoices = []
    modelChoices= []
    
    def getPFC(model, input, output):
        # PFCChoice.append(output.detach().permute(1,0,2).numpy())
        PFCChoice.append(np.round(torch.softmax(output.detach().permute(1,0,2),dim=-1).squeeze().numpy()[1]))

        
    def getBG(model, input, output):
        RLChoice.append(output.detach())
    
    def getModel(model, input, output):
        modelChoice.append(output.detach())
    
    if model.policy.RL is not None:
        h1 = model.policy.linear3.register_forward_hook(getPFC)

    # env.reset()
    rewards = [] 
    actions = []
    states = []
    
    regressors = []

    env.clear_data()

    if isinstance(order,int):
        order = [order]
        
        
    episode_rewards = []
    episode_actions = []
    episode_states = []
    for eps in range(nits+1):
        model.policy.reset()
        state =  env.reset()
        if model.policy.RL is not None: 
            PFCChoice = []
            RLChoice = []
            model.policy.RL.reset()
        rewards = []
        actions = []
        states = []
        modelChoice = []
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2).numpy()
        hidden_out = (torch.zeros([model.num_lstms,1, model.hidden_dim], dtype=torch.float), \
            torch.zeros([model.num_lstms,1,  model.hidden_dim], dtype=torch.float))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        if eps == 0:
            continue
        for step in range(model.max_steps):

            states.append(state[1])
        
            hidden_in = hidden_out                
            # action, log_probs, hidden_out = model.policy.get_action(torch.Tensor(state.reshape((1,1,-1))), torch.Tensor(last_action.reshape((1,1,-1))), reward, hidden_in, deterministic = model.DETERMINISTIC)
            action, log_probs, hidden_out = model.policy.get_action(torch.Tensor(state.reshape((1,1,-1))), torch.Tensor(last_action.reshape((1,1,-1))), None, torch.ones_like(torch.tensor(state)), deterministic = model.DETERMINISTIC)

            actions.append(action.detach().squeeze().numpy()[1])   
            if model.policy.RL is not None:
            #     PFCChoice.append(torch.softmax(model.policy.RL.PFC.Q.detach().permute(1,0,2),dim=-1).numpy()[1])
                RLChoice.append(np.round(F.softmax(model.policy.RL.Qs.detach(),dim=-1).squeeze().numpy()[1]))#IS THSI RIGHT
                
            # choice = np.round(log_probs.squeeze())
            # modelChoice.append(choice)
            # modelChoice.append(np.log(choice[1]/choice[0]))
                
            next_state, reward, done, _ = env.step(action[1])
            rewards.append(reward)
            # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
            state = next_state
            last_action = action
            if done:
                break
        episode_rewards.append(rewards)
        episode_actions.append(actions)
        episode_states.append(states)    
        modelChoices.append(modelChoice)
        if model.policy.RL is not None:
            PFCChoices.append(PFCChoice)
            RLChoices.append(RLChoice)
        
    if model.policy.RL is not None: 
        h1.remove()

    all_sols = {}
    modelChoices = np.vstack(modelChoices)

    
    for ord in order:
        sols = {}
        ord = min(ord, len(episode_actions[0]) - 1)


        # modelC = modelChoices[:,ord:].ravel()
        regressors = []
        for i in range(len(episode_actions)):
            # regressors.append(data_parse(episode_actions[i], episode_states[i], episode_rewards[i], ord))
            if combinatorial == True:
                regressors.append(data_parse_combinatorial(episode_actions[i], episode_rewards[i], ord))

            else:
                regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))

        regressors = np.vstack(regressors)
        
        # all_actions = np.vstack(episode_actions)[:,ord+combinatorial*100:].ravel()
        all_actions = np.vstack(episode_actions)[:,ord].ravel()
        # return logistic_regression(regressors,all_actions)
                    
        action_regression = logistic_regression(regressors, all_actions, bias = bias)


        # plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards)

        sols['action'] = action_regression.x

        # fig.show()
        xord = np.arange(1,1+ord)

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
        ax1.set_xlabel(r'$\theta^T x + b$')
        if bias == True:
            action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        else:
            action_fit = regressors @ sols['action']       
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
        smigmoid = np.linspace(min(action_fit),max(action_fit),100)
        ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        ax1.set_ylabel('action')
        bins = np.linspace(min(action_fit),max(action_fit),20)
        hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        hist1 = hist1[0]
        hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        hist2 = hist2[0]
        ax1.set_xlim(min(bins),max(bins))
        ax3 = ax1.twinx()
        if len(all_actions[all_actions == 0]) > 0:
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0])(smigmoid) * .1
            ax1.plot(smigmoid,kde1, color = 'b')
        if len(all_actions[all_actions == 1]) > 0:
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1])(smigmoid) *.1  
            ax3.plot(smigmoid,kde2, color = 'b')        

        ax3.invert_yaxis()
        ax1.axvline(0,0)        
        ax1.set_ylim(0,1)
        ax3.set_ylim(1,0)
        
        ax2.set_xlabel(r'trial number')

        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax2.plot(xord,sols['action'][:ord])
        ax2.plot(xord,sols['action'][ord:2*ord])
        ax2.plot(xord,sols['action'][2*ord:3*ord])
        # ax2.plot(xord,sols['action'][3*ord:4*ord])
        ax2.set_title(r'action regression coefficients')
        ax2.set_ylabel('coefficient')
        # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
        ax2.legend(['agent choice', 'win stay', 'lose switch'])
        ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
            
        if len(order) == 1:
            all_sols = sols
        else:
            all_sols[ord] = sols

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.set_title('RL AR coefficients')
        ax2.set_title('RL PACF')   
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax1.set_ylabel('coefficient')
        for i in range(modelChoices.shape[1]):
            try:
                acf = yw.yule_walker(modelChoices[i],12)
                if np.abs(sum(acf)) < 1e4:
                    ax1.plot(xord,acf,alpha=.02, c='b')
            except:
                continue
            try:
                pacf = yw.pacf(modelChoices[i],12)
                if np.abs(sum(pacf)) < 1e4:
                    ax2.plot(xord,pacf,alpha=.02, c='b')
            except:
                continue
        
        
    plt.show()

    return all_sols


# Originally at line 1245
def model_logistic_basic(model,env, nits=10, order=12, predict_on_choice=False):
    PFCChoices = []
    RLChoices = []
    modelChoices= []
    
    def getPFC(model, input, output):
        # PFCChoice.append(output.detach().permute(1,0,2).numpy())
        PFCChoice.append(np.round(torch.softmax(output.detach().permute(1,0,2),dim=-1).squeeze().numpy()[1]))

        
    
    if model.policy.RL is not None:
        h1 = model.policy.linear3.register_forward_hook(getPFC)

    env.reset()
    rewards = [] 
    actions = []
    states = []
    
    regressors = []

    env.clear_data()

    if isinstance(order,int):
        order = [order]
        
        
    episode_rewards = []
    episode_actions = []
    episode_states = []
    for eps in range(nits):
        state =  env.reset()
        if model.policy.RL is not None: 
            PFCChoice = []
            RLChoice = []
            model.policy.RL.reset()
        rewards = []
        actions = []
        states = []
        modelChoice = []
        
        last_action = env.action_space.sample()
        last_action = torch.nn.functional.one_hot(torch.tensor(last_action),num_classes=2).numpy()
        hidden_out = (torch.zeros([model.num_lstms,1, model.hidden_dim], dtype=torch.float), \
            torch.zeros([model.num_lstms,1,  model.hidden_dim], dtype=torch.float))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
        for step in range(model.max_steps):
            states.append(state[1])
        
            hidden_in = hidden_out                
            action, log_probs, hidden_out = model.policy.get_action(torch.Tensor(state.reshape((1,1,-1))), torch.Tensor(last_action.reshape((1,1,-1))), hidden_in, deterministic = model.DETERMINISTIC)
            
            actions.append(action.detach().squeeze().numpy()[1])   
            if model.policy.RL is not None:
            #     PFCChoice.append(torch.softmax(model.policy.RL.PFC.Q.detach().permute(1,0,2),dim=-1).numpy()[1])
                RLChoice.append(np.round(F.softmax(model.policy.RL.Qs.detach(),dim=-1).squeeze().numpy()[1]))#IS THSI RIGHT
                
            choice = np.round(log_probs.squeeze())
            modelChoice.append(choice)
            # modelChoice.append(np.log(choice[1]/choice[0]))
                
            next_state, reward, done, _ = env.step(action[1])
            rewards.append(reward)
            # action = torch.nn.functional.one_hot(torch.tensor(action),num_classes=2).numpy()
            state = next_state
            last_action = action
            if done:
                break
        episode_rewards.append(rewards)
        episode_actions.append(actions)
        episode_states.append(states)    
        modelChoices.append(modelChoice)
        if model.policy.RL is not None:
            PFCChoices.append(PFCChoice)
            RLChoices.append(RLChoice)
        
    if model.policy.RL is not None: 
        h1.remove()

    all_sols = {}
    modelChoices = np.vstack(modelChoices)

    
    for ord in order:
        sols = {}
        ord = min(ord, len(episode_actions[0]) - 1)


        modelC = modelChoices[:,ord:].ravel()
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_two_regressors(episode_actions[i], episode_states[i], episode_rewards[i], ord))

        regressors = np.vstack(regressors)
        
        
        all_actions = np.vstack(episode_actions)[:,ord:].ravel()
        # return logistic_regression(regressors,all_actions)
                    
        action_regression = logistic_regression(regressors, all_actions)


        # plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards)

        sols['action'] = action_regression.x

        # fig.show()
        xord = np.arange(1,1+ord)

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
        ax1.set_xlabel(r'$\theta^T x + b$')
        action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
        smigmoid = np.linspace(min(action_fit),max(action_fit),100)
        ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        ax1.set_ylabel('action')
        bins = np.linspace(min(action_fit),max(action_fit),20)
        hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        hist1 = hist1[0]
        hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        hist2 = hist2[0]
        ax1.set_xlim(min(bins),max(bins))
        ax3 = ax1.twinx()
        if len(all_actions[all_actions == 0]) > 0:
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0])(smigmoid) * .1
            ax1.plot(smigmoid,kde1, color = 'b')
        if len(all_actions[all_actions == 1]) > 0:
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1])(smigmoid) *.1  
            ax3.plot(smigmoid,kde2, color = 'b')        

        ax3.invert_yaxis()
        ax1.axvline(0,0)        
        ax1.set_ylim(0,1)
        ax3.set_ylim(1,0)
        
        ax2.set_xlabel(r'trial number')
        action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax2.plot(xord,sols['action'][:ord])
        ax2.plot(xord,sols['action'][ord:2*ord])
        ax2.set_title(r'action regression coefficients')
        ax2.set_ylabel('coefficient')
        ax2.legend(['agent choice', 'computer choice'])
        ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
            
        if len(order) == 1:
            all_sols = sols
        else:
            all_sols[ord] = sols

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
        ax1.set_title('RL AR coefficients')
        ax2.set_title('RL PACF')   
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax1.set_ylabel('coefficient')
        for i in range(modelChoices.shape[1]):
            try:
                acf = yw.yule_walker(modelChoices[i],12)
                if np.abs(sum(acf)) < 1e4:
                    ax1.plot(xord,acf,alpha=.02, c='b')
            except:
                continue
            try:
                pacf = yw.pacf(modelChoices[i],12)
                if np.abs(sum(pacf)) < 1e4:
                    ax2.plot(xord,pacf,alpha=.02, c='b')
            except:
                continue
        
        
    plt.show()

    return all_sols



# #hosmer lemeshow test based on logistic regression
# def hosmer_lemeshow_test(fit, X, y, bins=10):
#     #get predicted probabilities
#     p = logit(X,fit['x'][:-1],fit['x'][-1])
#     #get deciles of predicted probabilities
#     deciles = pd.qcut(p, bins, labels=False)
#     #get observed and expected values for each decile
#     observed = np.zeros(bins)
#     expected = np.zeros(bins)
#     for i in range(bins):
#         observed[i] = np.mean(y[deciles==i])
#         expected[i] = np.mean(p[deciles==i])
#     #compute chi-squared statistic
#     chi2 = np.sum((observed - expected)**2 / expected)
#     return chi2


# Originally at line 1887
def monkey_logistic_regression_reduced(df = None, db_path=None, order=5, 
                                       monkey=None, task = None, combinatorial=False, vif = False, err = False, sliding = None): 
    if df is None:
        monkey_dat = query_monkey_behavior(db_path)
    else:
        monkey_dat = df
    if not isinstance(task,list):
        task = [task]
    if not isinstance(monkey,list):
        monkey = [monkey]
    
    if isinstance(order,int):
        order = [order]
    
    
    task_dict = {}
    for t in task:
        task_dat = monkey_dat[monkey_dat['task'] == t]
        monkey_dict = {}
        for mk in monkey:
            iter_dat = task_dat[(task_dat['animal'] == mk)]
            all_sols = {}
            max_sess_len  = iter_dat['id'].value_counts().max()
            # plot_correlation_analysis_monkey(iter_dat, max_ord =4)
            # monkeyChoices_stacked = create_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            # monkeyChoices_stacked = create_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            monkeyChoices_all = create_choice_list(iter_dat)
            for ord in order:
                if combinatorial == True:
                    regressors = parse_monkey_behavior_combinatorial(iter_dat,ord, err)
                    all_actions = create_order_data_combinatorial(iter_dat, ord, err)

                else:
                    regressors = parse_monkey_behavior_reduced(iter_dat,ord, vif, err)
                    all_actions = create_order_data(iter_dat,ord, err)
                        
                    
                sols = {}
                if not err: # fit entire dataset
                    action_regression = logistic_regression(regressors, all_actions, bias=False)
                    sols['action'] = action_regression.x                    

                else: # fits each session individually and then 
                    if sliding is not None: # fit using sliding windows, then compute accuracy and error bars using that sliding window
                        # does this need to be weighed s.t. start and end trials aren't discounted disproportionately?
                        fits = [] 
                        subregs = []
                        subacts = []
                        for index in range(len(regressors)):
                            regs = regressors[index]
                            acts = all_actions[index]

                            for i in range(len(acts) - sliding):
                                fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=False).x)
                                subregs.append(regs[i:i+sliding])
                                subacts.append(acts[i:i+sliding])
                        fits = np.array(fits)
                        sols['action'] = np.mean(fits,axis=0).squeeze()
                        sols['err'] = (np.std(fits,axis=0)/np.sqrt(len(fits))).squeeze()
                        regressors = np.array(subregs)
                        all_actions = np.array(subacts) # do i need to weight this somehow
                            
                    else:
                        fits = []
                        lens = []

                        for index in range(len(regressors)):
                            fits.append(logistic_regression(regressors[index],all_actions[index],bias=False).x)
                            lens.append(len(all_actions[index]))
                        fits = np.array(fits)
                        sols['action'] = np.average(fits,axis=0,weights=lens).squeeze()
                        sols['err'] = (np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens))/np.sqrt(sum(lens))).squeeze()
                        
                    action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
                    all_actions = np.hstack(all_actions)
                # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                # ax1.set_xlabel(r'\theta^T x + b')
                # action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
                # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                # ax1.set_title(r'Monkey Actions regression. $accuracy = {}$'.format(acc))
                # ax1.scatter(action_fit, all_actions)
                # smigmoid = np.linspace(min(action_fit),max(action_fit),100)
                # ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                # ax1.set_ylabel('action')
                
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                ax1.set_xlabel(r'$\theta^T x + b$')
                if not err:
                    action_fit = regressors @ sols['action']
                    # action_fit = np.ravel([regressors[index] @ action_fit])
                # action_fit = action_fit[~np.isnan(action_fit)]
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                ax1.set_title(r'Monkey {} Actions regression: accuracy = {:.3f}'.format(mk,acc))
                smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
                ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                ax1.set_ylabel('action')
                bins = np.linspace(min(action_fit),max(action_fit),20)
                hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
                bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
                hist1 = hist1[0]
                hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
                bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
                hist2 = hist2[0]
                kde1 = stats.gaussian_kde(action_fit[all_actions == 0])(smigmoid) * .1
                kde2 = stats.gaussian_kde(action_fit[all_actions == 1])(smigmoid) *.1  
                ax1.set_xlim(min(bins),max(bins))
                ax1.plot(smigmoid,kde1, color = 'b')
                ax3 = ax1.twinx()
                ax3.invert_yaxis()
                ax3.plot(smigmoid,kde2, color = 'b')        
                ax1.axvline(0,0)        
                ax1.set_ylim(0,1)
                ax3.set_ylim(1,0)
                
                ax2.set_xlabel(r'trial number')
                # action_fit = regressors @ sols['action']
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                xord = np.arange(1,1+ord)
                reggy = ['agent choice', 'win stay', 'lose switch']
                if not err:
                    for i in range(len(sols['action'])//ord):
                        ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
                else:
                    for i in range(len(sols['action'])//ord):
                        prop_cycle = plt.rcParams['axes.prop_cycle']
                        colors = prop_cycle.by_key()['color']
                        ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
                        ax2.fill_between(xord,sols['action'][i*ord:(i+1)*ord] - sols['err'][i*ord:(i+1)*ord], sols['action'][i*ord:(i+1)*ord]+sols['err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
                    
                # ax2.plot(xord,sols['action'][:ord])
                # ax2.plot(xord,sols['action'][ord:2*ord])
                # ax2.plot(xord,sols['action'][2*ord:3*ord])
                # ax2.plot(xord,sols['action'][3*ord:4*ord])
                ax2.set_title(r'Monkey {} Action Regression Coefficients'.format(mk))
                ax2.set_ylabel('coefficient')
                # ax2.legend(['agent choice', 'win stay', 'lose switch'])
                ax2.legend()
                ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

                
                for i in range(len(monkeyChoices_all)):
                    try:
                        acf = yw.yule_walker(monkeyChoices_all[i],12)
                        if np.abs(sum(acf)) < 1e4:
                            ax1.plot(xord,acf,alpha=.02, c='b')
                    except:
                        continue
                    try:
                        pacf = yw.pacf(monkeyChoices_all[i],12)
                        if np.abs(sum(pacf)) < 1e4:
                            ax2.plot(xord,pacf,alpha=.02, c='b')
                    except:
                        continue
        
                if len(order) == 1:
                    all_sols = sols
                else:
                    all_sols[ord] = sols
            monkey_dict[mk] = all_sols
        task_dict[t] = monkey_dict
        
    plt.show()

    return task_dict


# Originally at line 2084
def monkey_session_regression(df = None, db_path=None, order=5, monkey=None, task = None, combinatorial=False): 
    if df is None:
        monkey_dat = query_monkey_behavior(db_path)
    else:
        monkey_dat = df
    if not isinstance(task,list):
        task = [task]
    if not isinstance(monkey,list):
        monkey = [monkey]
    
    if isinstance(order,int):
        order = [order]
    
    
    task_dict = {}
    for t in task:
        task_dat = monkey_dat[monkey_dat['task'] == t]
        monkey_dict = {}
        for mk in monkey:
            iter_dat = task_dat[(task_dat['animal'] == mk)]
            all_sols = {}
            max_sess_len  = iter_dat['id'].value_counts().max()
            # plot_correlation_analysis_monkey(iter_dat, max_ord =4)
            # monkeyChoices_stacked = create_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            # monkeyChoices_stacked = create_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            monkeyChoices_all = create_choice_list(iter_dat)
            for ord in order:
                if combinatorial == True:
                    regressors = parse_monkey_behavior_combinatorial(iter_dat,ord)
                    all_actions = create_order_data_combinatorial(iter_dat,ord)

                else:
                    regressors = parse_monkey_behavior_reduced(iter_dat,ord)
                    all_actions = create_order_data(iter_dat,ord)

                sols = {}
                action_regression = logistic_regression(regressors, all_actions, bias=False)
                sols['action'] = action_regression.x
                
                # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                # ax1.set_xlabel(r'\theta^T x + b')
                # action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
                # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                # ax1.set_title(r'Monkey Actions regression. $accuracy = {}$'.format(acc))
                # ax1.scatter(action_fit, all_actions)
                # smigmoid = np.linspace(min(action_fit),max(action_fit),100)
                # ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                # ax1.set_ylabel('action')
                
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                ax1.set_xlabel(r'$\theta^T x + b$')
                action_fit = regressors @ sols['action']
                # action_fit = action_fit[~np.isnan(action_fit)]
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                ax1.set_title(r'Monkey {} Actions regression: accuracy = {:.3f}'.format(mk,acc))
                smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
                ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                ax1.set_ylabel('action')
                bins = np.linspace(min(action_fit),max(action_fit),20)
                hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
                bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
                hist1 = hist1[0]
                hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
                bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
                hist2 = hist2[0]
                kde1 = stats.gaussian_kde(action_fit[all_actions == 0])(smigmoid) * .1
                kde2 = stats.gaussian_kde(action_fit[all_actions == 1])(smigmoid) *.1  
                ax1.set_xlim(min(bins),max(bins))
                ax1.plot(smigmoid,kde1, color = 'b')
                ax3 = ax1.twinx()
                ax3.invert_yaxis()
                ax3.plot(smigmoid,kde2, color = 'b')        
                ax1.axvline(0,0)        
                ax1.set_ylim(0,1)
                ax3.set_ylim(1,0)
                
                ax2.set_xlabel(r'trial number')
                action_fit = regressors @ sols['action']
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                xord = np.arange(1,1+ord)
                for i in range(len(sols['action'])//ord):
                    ax2.plot(xord,sols['action'][i*ord:(i+1)*ord])
                # ax2.plot(xord,sols['action'][:ord])
                # ax2.plot(xord,sols['action'][ord:2*ord])
                # ax2.plot(xord,sols['action'][2*ord:3*ord])
                # ax2.plot(xord,sols['action'][3*ord:4*ord])
                ax2.set_title(r'Monkey {} Action Regression Coefficients'.format(mk))
                ax2.set_ylabel('coefficient')
                ax2.legend(['agent choice', 'win stay', 'lose switch'])
                ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

                
                for i in range(len(monkeyChoices_all)):
                    try:
                        acf = yw.yule_walker(monkeyChoices_all[i],12)
                        if np.abs(sum(acf)) < 1e4:
                            ax1.plot(xord,acf,alpha=.02, c='b')
                    except:
                        continue
                    try:
                        pacf = yw.pacf(monkeyChoices_all[i],12)
                        if np.abs(sum(pacf)) < 1e4:
                            ax2.plot(xord,pacf,alpha=.02, c='b')
                    except:
                        continue
        
                if len(order) == 1:
                    all_sols = sols
                else:
                    all_sols[ord] = sols
            monkey_dict[mk] = all_sols
        task_dict[t] = monkey_dict
        
    plt.show()

    return task_dict

#plots a heatmap correlation of each regressor in the logistic regression for the model
# def plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards,  max_ord = 12, save = ''):
#     regressors = []
#     for i in range(len(episode_actions)):
#         # regressors.append(data_parse(episode_actions[i], episode_states[i], episode_rewards[i], max_ord))
#         regressors.append(data_parse(episode_actions[i],  episode_rewards[i], max_ord))

#     regressors = np.vstack(regressors)

#     fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
#     fig.suptitle('Correlation Analysis of Model Regressors')
#     ax1.set_title('Agent Choice')
#     ax2.set_title('Computer Choice')
#     ax3.set_title('Win Stay')
#     ax4.set_title('Lose Switch')
    
#     ax1.set_xlabel('order')
#     ax1.set_ylabel('order')
#     ax2.set_xlabel('order')
#     ax2.set_ylabel('order')
#     ax3.set_xlabel('order')
#     ax3.set_ylabel('order')
#     ax4.set_xlabel('order')
#     ax4.set_ylabel('order')
    
#     all_regressors = []
    
#     labels = [str(i) for i in range(1,max_ord+1)]
#     axs = [ax1, ax2, ax3, ax4]
#     for i in range(len(axs)):
#         corrmat = np.zeros((max_ord,max_ord))
#         all_regressors.append(regressors[:,i*max_ord:(i+1)*max_ord])
#         for j in range(max_ord):
#             for k in range(max_ord):
#                 corrmat[j,k] = np.corrcoef(regressors[:,i*max_ord+j].ravel(),regressors[:,i*max_ord+k].ravel())[0,1]
#         im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
#         axs[i].set_xticklabels(['']+ labels)
#         axs[i].set_yticklabels(['']+ labels)    
#         axs[i].xaxis.set_major_locator(MultipleLocator(1))
#         axs[i].yaxis.set_major_locator(MultipleLocator(1))
#     cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
#     fig.colorbar(im,cax= cbar_ax)
    
#     if save != '':
#         plt.savefig(save + '_RegressorAutoCorrelation.png')
    
#     fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
    
#     axs = [ax1, ax2, ax3, ax4, ax5, ax6]
#     labels1 = ['Agent Choice', 'Agent Choice', 'Agent Choice', 'Computer Choice','Computer Choice', 'Win Stay']
#     labels2 = ['Computer Choice', 'Win Stay', 'Lose Switch', 'Win Stay', 'Lose Switch', 'Lose Switch']
#     correlator1 = [all_regressors[0], all_regressors[0], all_regressors[0], all_regressors[1], all_regressors[1], all_regressors[2]]
#     correlator2 = [all_regressors[1], all_regressors[2], all_regressors[3], all_regressors[2], all_regressors[3], all_regressors[3]]
#     for i in range(len(axs)):
#         axs[i].set_xlabel('order')
#         axs[i].set_ylabel('order')
#         axs[i].set_title('{} vs {}'.format(labels1[i],labels2[i]))
#         corrmat = np.zeros((max_ord,max_ord))
#         for j in range(max_ord):
#             for k in range(max_ord):
#                 corrmat[j,k] = np.corrcoef(correlator1[i][:,j].ravel(),correlator2[i][:,k].ravel())[0,1]
#         im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
#         axs[i].set_xticklabels(['']+ labels)
#         axs[i].set_yticklabels(['']+ labels)
#         axs[i].xaxis.set_major_locator(MultipleLocator(1))
#         axs[i].yaxis.set_major_locator(MultipleLocator(1))
    
#     if save != '':
#         plt.savefig(save+'_RegressorCrossCorrelation.png')


# Originally at line 2860
def data_parse_sliding(actions, rewards,order=1):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {1, 0, -1} if left and not rewarded on prior trial, rewarded, or right not rewarded (basically, swap on win)
    numvars = 4
    state = np.zeros(len(actions))
    for i in range(len(actions)):
        state[i] = actions[i] if rewards[i] == 1 else not actions[i]

    data = np.zeros((len(actions) - order,numvars,order))

    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1+1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #opponent choice
                    if state[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==2: #win stay
                    if rewards[i-k] == 1:
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==3: #lose switch
                    if rewards[i-k] == 0:
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    data = data.reshape(data.shape[0],-1, order='C')
    return data


# Originally at line 3246
def glr_comparison(ax,model, data, order=5, average=False):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    sols = {}
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
    else:
        # format into episode actions and rewards
        episode_actions = []
        episode_rewards = []
        sessions = data['id'].unique()
        for s in sessions:        
            sessdat = data[data['id'] == s].sort_values(by=['id','trial'])
            episode_actions.append(sessdat['monkey_choice'].to_numpy())
            episode_rewards.append(sessdat['reward'].to_numpy())
        

    
    # parameters to plot:    
    # params = [(1,0), (1,1), (2,1), (2,2)]
    # params = [(2,1)]
    params = [(1,0), (1,1), (2,1), (2,2)]

    # Define consistent colors for coefficient types (matching other figures)
    color_map = {
        'win stay': 'blue',
        'win switch': 'orange', 
        'lose stay': 'green',
        'lose switch': 'red',
        'action': 'purple',
        'win': 'blue',
        'lose': 'red',
        'win repeat': 'blue',
        'win change': 'orange',
        'lose repeat': 'green',
        'lose change': 'red',
        'repeat win': 'blue',
        'change win': 'orange',
        'repeat lose': 'green',
        'change lose': 'red',
    }
    
    marker_map = {
        'win stay': 'o',
        'win switch': '^', 
        'lose stay': 'v',
        'lose switch': 's',
        'action': 'o',
        'win': 'o',
        'lose': 's',
        'win repeat': 'o',
        'win change': '^',
        'lose repeat': 'v',
        'lose change': 's',
        'repeat win': 'o',
        'change win': '^',
        'repeat lose': 'v',
        'change lose': 's',
    }

    # Create figure with subplots for coefficients and performance comparison
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[3, 3, 1.5], hspace=0.35, wspace=0.3)
    
    # Coefficient plots in top 2 rows
    ax_coeffs = []
    for row in range(2):
        for col in range(2):
            ax_coeffs.append(fig.add_subplot(gs[row, col]))
    
    # Performance comparison plots in bottom row
    ax_insample = fig.add_subplot(gs[2, :2])
    ax_outsample = fig.add_subplot(gs[2, 2:])
    
    # Store accuracies for performance plots
    in_sample_accs = {}
    out_sample_accs = {}
    
    # need to iterate over every episode or session
    labels = []
    for i in range(len(params)):
        sols[params[i]] = {}
        regressors = []
        ys = []
        fits = []
        lens = []
        xord = np.arange(1,order+1)

        for j in range(len(episode_actions)):
            # regressors_j,labels_j = general_logistic_regressors(episode_actions[j], episode_rewards[j], regression_order=order, a_order=params[i][0], r_order=params[i][1])
            regressors_j, y ,labels_j = general_logistic_regressors(episode_actions[j], episode_rewards[j], regression_order=order, a_order=params[i][0], r_order=params[i][1])
            regressors.append(regressors_j)
            ys.append(y)
        labels.append(labels_j)

        if average: # fit the same way we fit other functions
            fits = [logistic_regression_colinear(regressors[j],ys[j],bias=True)['x'] for j in range(len(regressors))]
            # fits = [logistic_regression(regressors[j],episode_actions[j][order:],bias=True,colinear=True)['x'] for j in range(len(regressors))]
            lens = [len(y_episode) for y_episode in ys]
            sols[params[i]]['action'] = np.average(fits, axis=0, weights=lens).squeeze()
            accs = [np.mean(np.round(sigmoid(regressors[j] @ fits[j][:-1] + fits[j][-1])) == ys[j]) for j in range(len(regressors))]

        else:
        # for j in range(len(regressors)):
        #     fits.append(logistic_regression_colinear(regressors[j],episode_actions[j],bias=True)['x'])            
        #     lens.append(len(episode_actions[j]))
            fits = logistic_regression_colinear_multiepisode(regressors,ys,bias=True)['x']
            sols[params[i]]['action'] = fits
            accs = [np.mean(np.round(sigmoid(regressors[j] @ fits[:-1] + fits[-1])) == ys[j]) for j in range(len(regressors))]
        
        # Compute out-of-sample accuracy using leave-one-session-out cross-validation
        oos_accs = []
        for holdout_idx in range(len(regressors)):
            # Train on all sessions except holdout
            train_regressors = [regressors[j] for j in range(len(regressors)) if j != holdout_idx]
            train_ys = [ys[j] for j in range(len(ys)) if j != holdout_idx]
            
            if len(train_regressors) > 0:
                # Fit on training data using the same method as in-sample
                if average:
                    # Fit each session separately and average
                    train_fits = [logistic_regression_colinear(train_regressors[j], train_ys[j], bias=True)['x'] 
                                  for j in range(len(train_regressors))]
                    train_lens = [len(train_ys[j]) for j in range(len(train_ys))]
                    holdout_fit = np.average(train_fits, axis=0, weights=train_lens).squeeze()
                else:
                    # Fit all sessions together
                    holdout_fit = logistic_regression_colinear_multiepisode(train_regressors, train_ys, bias=True)['x']
                
                # Test on holdout session
                holdout_pred = np.round(sigmoid(regressors[holdout_idx] @ holdout_fit[:-1] + holdout_fit[-1]))
                holdout_acc = np.mean(holdout_pred == ys[holdout_idx])
                oos_accs.append(holdout_acc)
        
        # Store accuracies
        in_sample_accs[params[i]] = np.mean(accs)
        out_sample_accs[params[i]] = np.mean(oos_accs) if oos_accs else np.nan
        
        # can add in err by boostrapping the fits later
        # sols[params[i]]['err'] = np.sqrt(np.average((sols[params[i]]['action'] - fits)**2,axis=0,weights=lens)).squeeze()
        # fit the regressors
        ax_coeffs[i].axhline(linestyle='--', color='k', alpha=.5) 
        ax_coeffs[i].set_title(f'{params[i]} - In-Sample: {np.mean(accs):.3f}, Out-of-Sample: {np.mean(oos_accs) if oos_accs else 0:.3f}', 
                               fontsize=11, fontweight='bold')
        
        print(f'{params[i]} model - In-sample accuracy: {np.mean(accs):.3f}, Out-of-sample accuracy: {np.mean(oos_accs) if oos_accs else 0:.3f}')
        # n_action_features = 2**(params[i][0]-1) if params[i][0] > 0 else 1
        # n_reward_features = 2**params[i][1] if params[i][1] > 0 else 1
        # n_features_per_lag = n_action_features * n_reward_features
        
        # # Reshape coefficients: (n_features_per_lag, regression_order)
        # coeffs_reshaped = sols[params[i]]['action'][:-1].reshape(n_features_per_lag, order)
        
        # # Generate feature labels and optimal ordering
        # feature_labels, reorder_indices = generate_feature_ordering(params[i][0], params[i][1])
        
        # # Apply reordering
        # coeffs_reordered = coeffs_reshaped[reorder_indices, :]
        
        # # Flatten back
        # reordered_coeffs = coeffs_reordered.flatten()
        
        for j in range(len(sols[params[i]]['action'])//order):
            # ax_coeffs[i].plot(xord,reordered_coeffs[j*order:(j+1)*order],label=labels[i][j])
            # ax_coeffs[i].plot(xord,sols[params[i]]['action'][j:-1:(2**(params[i][0]-1)*2**(params[i][1]))],label=labels[i][j])
            label = labels[i][j]
            color = color_map.get(label, f'C{j}')  # Use default color if not in map
            marker = marker_map.get(label, 'o')
            ax_coeffs[i].plot(xord, sols[params[i]]['action'][j*order:(j+1)*order], 
                            marker=marker, linestyle='-', linewidth=2, markersize=6,
                            label=label, color=color)

        ax_coeffs[i].legend(loc='best', fontsize=9)
        ax_coeffs[i].set_xlabel('Trials Back', fontsize=10)
        ax_coeffs[i].set_ylabel('Coefficient', fontsize=10)
    
    # Plot performance comparison
    param_labels = [f'{p}' for p in params]
    x_pos = np.arange(len(params))
    
    # In-sample performance
    in_sample_vals = [in_sample_accs[p] for p in params]
    bars_in = ax_insample.bar(x_pos, in_sample_vals, color=['cyan', 'lightblue', 'cornflowerblue', 'royalblue'], alpha=0.8)
    ax_insample.set_xticks(x_pos)
    ax_insample.set_xticklabels(param_labels)
    ax_insample.set_ylabel('Accuracy', fontsize=11)
    ax_insample.set_title('In-Sample Performance - All Monkeys', fontsize=12, fontweight='bold')
    ax_insample.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax_insample.set_ylim([0.5, max(in_sample_vals) * 1.05])
    
    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars_in, in_sample_vals)):
        ax_insample.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Out-of-sample performance
    out_sample_vals = [out_sample_accs[p] for p in params]
    bars_out = ax_outsample.bar(x_pos, out_sample_vals, color=['pink', 'lightcoral', 'indianred', 'firebrick'], alpha=0.8)
    ax_outsample.set_xticks(x_pos)
    ax_outsample.set_xticklabels(param_labels)
    ax_outsample.set_ylabel('Accuracy', fontsize=11)
    ax_outsample.set_title('Out-Of-Sample Performance - All Monkeys', fontsize=12, fontweight='bold')
    ax_outsample.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax_outsample.set_ylim([0.5, max(out_sample_vals) * 1.05])
    
    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars_out, out_sample_vals)):
        ax_outsample.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Highlight that (2,1) is better than (1,1) for out-of-sample
    if out_sample_accs[(2,1)] > out_sample_accs[(1,1)]:
        # Draw arrow or annotation
        ax_outsample.annotate('', xy=(2, out_sample_accs[(2,1)]), xytext=(1, out_sample_accs[(1,1)]),
                             arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax_outsample.text(1.5, (out_sample_accs[(2,1)] + out_sample_accs[(1,1)])/2 + 0.005, 
                         f'+{(out_sample_accs[(2,1)] - out_sample_accs[(1,1)])*100:.2f}%',
                         ha='center', fontsize=10, fontweight='bold', color='green',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))
    
    fig.suptitle('GLR Analysis: Monkey E Coefficients & All Monkeys Performance', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    return fig, ax_coeffs, sols, in_sample_accs, out_sample_accs
    
# def generate_feature_ordering(a_order, r_order):
#     """
#     Generate optimal feature ordering and labels for any parameter combination.
    
#     Strategy:
#     1. Group by reward context first (win/lose conditions)
#     2. Then by action patterns within each reward context
#     3. This creates intuitive "win-stay/lose-shift" type groupings
#     """
    
#     # Generate all possible feature combinations
#     n_action_features = 2**(a_order-1) if a_order > 0 else 1  
#     n_reward_features = 2**r_order if r_order > 0 else 1
    
#     feature_combinations = []
#     feature_labels = []
    
#     # Generate action pattern labels
#     if a_order == 1:
#         action_patterns = ['right', 'left'] 
#     elif a_order == 2:
#         action_patterns = ['stay', 'switch']
#     else:
#         # For higher orders, create hierarchical labels
#         action_patterns = generate_hierarchical_labels(a_order-1, 'stay', 'switch')
    
#     # Generate reward pattern labels  
#     if r_order == 0:
#         reward_patterns = ['']
#     elif r_order == 1:
#         reward_patterns = ['win', 'lose']
#     else:
#         # For higher orders, create hierarchical labels
#         reward_patterns = generate_hierarchical_labels(r_order, 'win', 'lose')
    
#     # Create all combinations (following kronecker product order)
#     original_order = []
#     for a_idx in range(n_action_features):
#         for r_idx in range(n_reward_features):
#             if r_order == 0:
#                 label = action_patterns[a_idx] if a_order > 1 else f'action_{a_idx}'
#             else:
#                 if a_order == 1:
#                     label = f"{reward_patterns[r_idx]} {action_patterns[a_idx]}"
#                 else:
#                     label = f"{reward_patterns[r_idx]} {action_patterns[a_idx]}"
            
#             original_order.append((a_idx, r_idx, label))
    
#     # Create optimal reordering - group by reward context first, then action patterns
#     reordered = []
#     reorder_indices = []
    
#     if r_order > 0:
#         # Group by reward patterns first
#         for r_idx in range(n_reward_features):
#             for a_idx in range(n_action_features):
#                 original_idx = a_idx * n_reward_features + r_idx
#                 reordered.append(original_order[original_idx])
#                 reorder_indices.append(original_idx)
#     else:
#         # No reward context, just use action order
#         reordered = original_order
#         reorder_indices = list(range(len(original_order)))
    
#     # Extract just the labels
#     labels = [item[2] for item in reordered]
    
#     return labels, reorder_indices
    
    

# def generate_hierarchical_labels(depth, prefix1, prefix2):
#     """Generate hierarchical labels for higher-order features."""
#     if depth == 1:
#         return [prefix1, prefix2]
#     else:
#         sub_labels = generate_hierarchical_labels(depth-1, prefix1, prefix2)
#         result = []
#         for label in sub_labels:
#             result.append(f"{prefix1} {label}")
#         for label in sub_labels:
#             result.append(f"{prefix2} {label}")
#         return result


# Originally at line 3712
def model_logistic_regression_sliding(actions, rewards,dominance, window_size= 10, order=2, save = '', BG =True, mp_mem = 4):
    '''Selects regions of dominance longer than window size and fits a logistic regression to that region'''
    # assert(order < len(actions), 'order must be less than the length of the data')
    # assert(window_size < len(actions), 'window size must be less than the length of the data')
    # assert(len(actions) == len(rewards), 'actions and rewards must be the same length')
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    pfc = dominance >= 0
    bg = dominance <= 0
    
    BGs = []
    PFCs = []
    snippet = []
    flag = 0
    # get indices of regions of dominance
    for i in range(len(dominance)):
        if flag == 0 and dominance[i] != 0:
            flag = dominance[i]
        elif flag != 1 and dominance[i] > 0:
            flag = 1
            BGs.append(snippet)
            snippet = []
        elif flag != -1 and dominance[i] < 0:
            flag = -1
            PFCs.append(snippet)
            snippet = []
        snippet.append(i)
        if i == len(dominance)-1:
            if flag == 1:
                PFCs.append(snippet)
            else:
                BGs.append(snippet)
    # now cut off any snippets less than window size
    # print('Average PFC snippet length: {}, num windows:{}'.format(np.mean([len(i) for i in PFCs]), len(PFCs)))
    # print('Average BG snippet length: {}, num windows: {}'.format(np.mean([len(i) for i in BGs]), len(BGs)))
    # print("PFC:{} %% of the time, BG:{} %% of the time".format(len(np.hstack(PFCs))/len(dominance), len(np.hstack(BGs))/len(dominance)))
    PFC_trimmed = [np.array(i[mp_mem:]) for i in PFCs if len(i) >= window_size]
    BG_trimmed = [np.array(i[mp_mem:]) for i in BGs if len(i) >= window_size]
    
    PFC_data = []
    BG_data = []
    for i in range(len(PFC_trimmed)):
        PFC_data.append(data_parse_sliding_WSLS(actions[PFC_trimmed[i]], rewards[PFC_trimmed[i]], order=order))
    PFC_a = np.hstack([actions[i[order+mp_mem:]] for i in PFCs if len(i) >= window_size])
    PFC_data = np.vstack(PFC_data)
    PFC_reg = logistic_regression(PFC_data, PFC_a, bias=False).x

    if BG==True:
        for i in range(len(BG_trimmed)):
            BG_data.append(data_parse_sliding_WSLS(actions[BG_trimmed[i]], rewards[BG_trimmed[i]], order=order))
        BG_data = np.vstack(BG_data)
        BG_a = np.hstack([actions[i[order+mp_mem:]] for i in BGs if len(i) >= window_size])
        BG_reg = logistic_regression(BG_data, BG_a, bias=False).x


    
    # for window in range(len(actions)-window_size-order):
    #     reg = logistic_regression(data[window:window+window_size], actions[window+order:order+window+window_size], bias=False)
    #     regressors.append(reg.x)
    ord = order
    xord = np.arange(1,1+ord)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
    ax1.set_xlabel(r'$\theta^T x + b$')
    action_fit = PFC_data @ PFC_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == PFC_a)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[PFC_a == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[PFC_a == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    kde1 = stats.gaussian_kde(action_fit[PFC_a == 0],bw_method=.1)(smigmoid) * .1
    kde2 = stats.gaussian_kde(action_fit[PFC_a == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    ax2.set_xlabel(r'trial number')
    action_fit = PFC_data @ PFC_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == PFC_a)
    for i in range(len(PFC_reg)//ord):
        # print(PFC_reg[i])
        ax2.plot(xord,PFC_reg[ord*i:ord*(i+1)])
    # ax2.plot(xord,PFC_reg[:ord])
    # ax2.plot(xord,PFC_reg[ord:2*ord])
    # ax2.plot(xord,PFC_reg[2*ord:3*ord])
    # ax2.plot(xord,PFC_reg[3*ord:4*ord])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    ax2.legend(['agent choice','win stay', 'lose switch'])

    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    
    fig.show()

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
    ax1.set_xlabel(r'$\theta^T x + b$')
    action_fit = BG_data @ BG_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == BG_a)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[BG_a == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[BG_a == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    kde1 = stats.gaussian_kde(action_fit[BG_a == 0],bw_method=.1)(smigmoid) * .1
    kde2 = stats.gaussian_kde(action_fit[BG_a == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    ax2.set_xlabel(r'trial number')
    # action_fit = BG_data @ BG_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == BG_a)
    for i in range(len(BG_reg)//ord):
        # print(BG_reg[i])
        ax2.plot(xord,BG_reg[ord*i:ord*(i+1)])  
    # ax2.plot(xord,BG_reg[:ord])
    # ax2.plot(xord,BG_reg[ord:2*ord])
    # ax2.plot(xord,BG_reg[2*ord:3*ord])
    # ax2.plot(xord,BG_reg[3*ord:4*ord])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    ax2.legend(['agent choice', 'win stay', 'lose switch'])

    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    
    fig.show()
    
    print('Average PFC snippet length: {}, num windows:{}'.format(np.mean([len(i) for i in PFCs]), len(PFCs)))
    print('Average BG snippet length: {}, num windows: {}'.format(np.mean([len(i) for i in BGs]), len(BGs)))
    print("PFC:{} %% of the time, BG:{} %% of the time".format(len(np.hstack(PFCs))/len(dominance), len(np.hstack(BGs))/len(dominance)))
    # return np.vstack(regressors)


# Originally at line 3876
def model_logistic_regression_reduced(actions,rewards, order=5, save = ''):



    regressors  =data_parse_sliding_WSLS(actions,rewards, ord)


    
    
    # all_actions = np.vstack(episode_actions)[:,ord:].ravel()
    # return logistic_regression(regressors,all_actions)
                
    # prob_regression = logistic_regression(regressors,modelC)
    action_regression = logistic_regression(regressors, actions, bias=0)

    # prob_regression = logistic_regression_l2(regressors,modelChoices)
    # action_regression = logistic_regression_l2(regressors, all_actions)


    # sols['prob'] = prob_regression.x
    sols = {}
    sols['action'] = action_regression.x
    
    
    # sigma = prob_regression.hess_inv
    # sols['prob_err'] = prob_regression.
    # sols['action_err']
    

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
    ax1.set_xlabel(r'$\theta^T x + b$')
    action_fit = regressors @ sols['action']
    acc = np.mean(np.round(sigmoid(action_fit)) == actions)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[actions == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[actions == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    kde1 = stats.gaussian_kde(action_fit[actions == 0],bw_method=.1)(smigmoid) * .1
    kde2 = stats.gaussian_kde(action_fit[actions == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    xord = np.arange(1,1+ord)

    ax2.set_xlabel(r'trial number')
    action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
    acc = np.mean(np.round(sigmoid(action_fit)) == actions)
    ax2.plot(xord,sols['action'][:ord])
    ax2.plot(xord,sols['action'][ord:2*ord])
    ax2.plot(xord,sols['action'][2*ord:3*ord])
    ax2.plot(xord,sols['action'][3*ord:4*ord])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    if save != '':
        plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
    
    plt.show()

    return sols


# Originally at line 3954
def model_logistic_regression_sliding_monkey(actions, rewards,dominance, window_size= 10, order=2, save = '', BG =True, mp_mem = 4, combinatorial=False):
    # assert(order < len(actions), 'order must be less than the length of the data')
    # assert(window_size < len(actions), 'window size must be less than the length of the data')
    # assert(len(actions) == len(rewards), 'actions and rewards must be the same length')
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    pfc = dominance >= 0
    bg = dominance <= 0
    
    BGs = []
    PFCs = []
    snippet = []
    flag = 0
    # get indices of regions of dominance
    for i in range(len(dominance)):
        if flag == 0 and dominance[i] != 0:
            flag = dominance[i]
        elif flag != 1 and dominance[i] > 0:
            flag = 1
            BGs.append(snippet)
            snippet = []
        elif flag != -1 and dominance[i] < 0:
            flag = -1
            PFCs.append(snippet)
            snippet = []
        snippet.append(i)
        if i == len(dominance)-1:
            if flag == 1:
                PFCs.append(snippet)
            else:
                BGs.append(snippet)
    # now cut off any snippets less than window size
    # print('Average PFC snippet length: {}, num windows:{}'.format(np.mean([len(i) for i in PFCs]), len(PFCs)))
    # print('Average BG snippet length: {}, num windows: {}'.format(np.mean([len(i) for i in BGs]), len(BGs)))
    # print("PFC:{} %% of the time, BG:{} %% of the time".format(len(np.hstack(PFCs))/len(dominance), len(np.hstack(BGs))/len(dominance)))
    PFC_trimmed = [np.array(i[mp_mem:]) for i in PFCs if len(i) >= window_size]
    BG_trimmed = [np.array(i[mp_mem:]) for i in BGs if len(i) >= window_size]
    
    PFC_data = []
    BG_data = []
    for i in range(len(PFC_trimmed)):
        PFC_data.append(data_parse_sliding_WSLS(actions[PFC_trimmed[i]], rewards[PFC_trimmed[i]], order=order))
    PFC_a = np.hstack([actions[i[order+mp_mem:]] for i in PFCs if len(i) >= window_size])
    PFC_data = np.vstack(PFC_data)
    PFC_reg = logistic_regression(PFC_data, PFC_a, bias=False).x

    if BG==True:
        for i in range(len(BG_trimmed)):
            BG_data.append(data_parse_sliding_WSLS(actions[BG_trimmed[i]], rewards[BG_trimmed[i]], order=order))
        BG_data = np.vstack(BG_data)
        BG_a = np.hstack([actions[i[order+mp_mem:]] for i in BGs if len(i) >= window_size])
        BG_reg = logistic_regression(BG_data, BG_a, bias=False).x


    
    # for window in range(len(actions)-window_size-order):
    #     reg = logistic_regression(data[window:window+window_size], actions[window+order:order+window+window_size], bias=False)
    #     regressors.append(reg.x)
    ord = order
    xord = np.arange(1,1+ord)

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
    ax1.set_xlabel(r'$\theta^T x + b$')
    action_fit = PFC_data @ PFC_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == PFC_a)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[PFC_a == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[PFC_a == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    # kde1 = stats.gaussian_kde(action_fit[PFC_a == 0],bw_method=.1)(smigmoid) * .1
    # kde2 = stats.gaussian_kde(action_fit[PFC_a == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    # ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    # ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    ax2.set_xlabel(r'trial number')
    action_fit = PFC_data @ PFC_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == PFC_a)
    for i in range(len(PFC_reg)//ord):
        # print(PFC_reg[i])
        ax2.plot(xord,PFC_reg[ord*i:ord*(i+1)])
    # ax2.plot(xord,PFC_reg[:ord])
    # ax2.plot(xord,PFC_reg[ord:2*ord])
    # ax2.plot(xord,PFC_reg[2*ord:3*ord])
    # ax2.plot(xord,PFC_reg[3*ord:4*ord])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    ax2.legend(['agent choice','win stay', 'lose switch'])

    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    
    fig.show()

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
    ax1.set_xlabel(r'$\theta^T x + b$')
    action_fit = BG_data @ BG_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == BG_a)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[BG_a == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[BG_a == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    # kde1 = stats.gaussian_kde(action_fit[BG_a == 0],bw_method=.1)(smigmoid) * .1
    # kde2 = stats.gaussian_kde(action_fit[BG_a == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    # ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    # ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    ax2.set_xlabel(r'trial number')
    # action_fit = BG_data @ BG_reg
    acc = np.mean(np.round(sigmoid(action_fit)) == BG_a)
    for i in range(len(BG_reg)//ord):
        # print(BG_reg[i])
        ax2.plot(xord,BG_reg[ord*i:ord*(i+1)])  
    # ax2.plot(xord,BG_reg[:ord])
    # ax2.plot(xord,BG_reg[ord:2*ord])
    # ax2.plot(xord,BG_reg[2*ord:3*ord])
    # ax2.plot(xord,BG_reg[3*ord:4*ord])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    ax2.legend(['agent choice', 'win stay', 'lose switch'])

    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    
    fig.show()
    
    print('Average PFC snippet length: {}, num windows:{}'.format(np.mean([len(i) for i in PFCs]), len(PFCs)))
    print('Average BG snippet length: {}, num windows: {}'.format(np.mean([len(i) for i in BGs]), len(BGs)))
    print("PFC:{} %% of the time, BG:{} %% of the time".format(len(np.hstack(PFCs))/len(dominance), len(np.hstack(BGs))/len(dominance)))
    # return np.vstack(regressors)


# Originally at line 4115
def fit_stationarity_monkey(df = None, db_path = None, order=5, monkey=None,sliding=None, diff = True):
    '''takes a monkey, stacks all the sessions together, and then plots the. 
    Stack the algo 1 and algo 2 and see if we can see nonstationarity in the variables.
    Try ratio of first two lose switch, or difference, since that seems to be the 
    variable that changes most between algorithms.
    Plots the time course of the variables, as well as autocorrelation across sessions'''
     
    if df is None:
        monkey_dat = query_monkey_behavior(db_path)
    else:
        monkey_dat = df
        if monkey is not None:
            monkey_dat = monkey_dat[monkey_dat['animal'] == monkey]


    # generate data
    # stack algo 1 -> algo 2
    # fit each session and plot the 
    sorted_dat = monkey_dat.sort_values(by = ['task','id'])
    regressors = parse_monkey_behavior_reduced(sorted_dat[sorted_dat['task'] == 1],order,err=True) + \
        parse_monkey_behavior_reduced(sorted_dat[sorted_dat['task'] == 2],order, err=True)
    all_actions = create_order_data(sorted_dat[sorted_dat['task']==1],order,err=True) + \
        create_order_data(sorted_dat[sorted_dat['task']==2],order,err=True)

    new_task_tr = len(sorted_dat[sorted_dat['task']==1]['id'].unique())
    
    ls1 = 2*order       # index of lose switch first component

    eps = 1e-4
    ars = []
    lratios = []
    sliding_ratios = []
    ls1diff = []
    ls2diff = []
    l1 = 1
    l2 = 1
    for index in range(len(regressors)):
        fit = logistic_regression(regressors[index],all_actions[index],bias=False).x
        if not diff:
            lratios.append(fit[ls1+1]/(fit[ls1]+eps))
            ls1diff.append(fit[ls1]/l1)
            l1 = fit[ls1]
            ls2diff.append(fit[ls1+1]/l2)
            l2 = fit[ls1+1]
        else:
            lratios.append(fit[ls1+1] - fit[ls1])
            ls1diff.append(fit[ls1]-l1)
            l1 = fit[ls1]
            ls2diff.append(fit[ls1+1]-l2)
            l2 = fit[ls1+1]
        
    #sliding window fits
    if sliding is None:
        sliding = 50

    for index in range(len(regressors)):
        regs = regressors[index]
        acts = all_actions[index]
        sess_ratio = []
        for i in range(len(acts) - sliding):
            fits = logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=False).x
            if not diff:
                filt = median_filter(fits[ls1+1]/(fits[ls1]+eps), 3*order)
                sliding_ratios.append(filt)
            else:
                sliding_ratios.append(fits[ls1+1]-(fits[ls1]))
            sess_ratio.append(sliding_ratios[-1])
        ars.append(yw.yule_walker(sess_ratio, order=1))
    
    fig, ax = plt.subplots(4,1, figsize = (12,12),dpi=300)
    
    fig.suptitle('Monkey {} Stationarity'.format(monkey))
    ax[0].set_xlabel('Session')
    ax[1].set_xlabel('Session')
    ax[2].set_xlabel('Sliding Window')
    ax[3].set_xlabel('Session')
    if not diff:
        ax[0].set_title('Session-Averaged LS(2)/LS(1) AR') # from sliding window
        ax[1].set_title('Session-Averaged LS(2)/LS(1)')
        ax[2].set_title('{} Trial Sliding Window LS(2)/LS(1)'.format(sliding))
    
        ax[0].set_ylabel('AR(LS(2)/LS(1))')
        ax[1].set_ylabel('LS(2)/LS(1)')
        ax[2].set_ylabel('LS(2)/LS(1)')
        ax[3].set_ylabel(r'$LS_t/LS_{t-1}$')
    else:
        ax[0].set_title('Session-Averaged LS(2)-LS(1) AR') # from sliding window
        ax[1].set_title('Session-Averaged LS(2)-LS(1)')
        ax[2].set_title('{} Trial Sliding Window LS(2)-LS(1)'.format(sliding))
    
        ax[0].set_ylabel('AR(LS(2)-LS(1))')
        ax[1].set_ylabel('LS(2)-LS(1)')
        ax[2].set_ylabel('LS(2)-LS(1)')
        ax[3].set_ylabel(r'$LS_t-LS_{t-1}$')
    
    ax[0].plot(ars)
    ax[1].plot(lratios)
    ax[2].plot(sliding_ratios)
    if not diff:
        ax[3].semilogy(ls1diff)
        ax[3].semilogy(ls2diff)
    else:
        ax[3].plot(ls1diff)
        ax[3].plot(ls2diff)
    ax[3].legend(['LS(1)','LS(2)'])
    
    ax[0].axvline(new_task_tr,linestyle='--', color = 'k', alpha =.5)
    ax[1].axvline(new_task_tr,linestyle='--', color = 'k', alpha =.5)
    ax[3].axvline(new_task_tr-1,linestyle='--', color = 'k', alpha =.5)
    
    plt.tight_layout()
    plt.show()


# Originally at line 4228
def AR_comparison(df = None, db_path = None, order = 5, monkeys = None, sliding = None, diff =True, task = None, overlap = False):
    '''For each session, computes all sliding windows of a given length and 
    uses them to calculate the autocorrelation and variance between LS(1) and LS(2)'''
    
    if df is None:
        dat = query_monkey_behavior(db_path)
    else:
        dat = df
    if monkeys is not None:
        if isinstance(monkey,int):
            monkeys = [monkeys]
        else:
            monkeys = list(monkeys)
    else:
        monkeys = list(dat['animal'].unique())
    if task is not None:
        dat = dat[dat['task'] == task]
        
    # def acf(x, length=20):
        # return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        # for i in range(1, length)])    
    def acf(x, length=20):
        return [[1]+[np.cov(x[:-i],x[i:])[0,1]/np.sqrt(np.var(x[:-i]) * np.var(x[i:])) \
                for i in range(1,length)]][0]
        # return [np.mean([np.cov(x[j],x[j-i])/np.sqrt(np.var(x[j]) * np.var(x[j-i])) \
        #         for j in range(i,len(x[0]))]) for i in range(0,length)]
    
    col_counter = 0
    fig, ax = plt.subplots(3,1, figsize = (8,12),dpi=300)
    fig.suptitle('Monkey Autocorrelation')
    for monkey in monkeys:
        monkey_dat = dat[dat['animal'] == monkey]

        sorted_dat = monkey_dat.sort_values(by = ['task','id'])
        regressors = parse_monkey_behavior_reduced(sorted_dat[sorted_dat['task'] == 1],order,err=True) + \
            parse_monkey_behavior_reduced(sorted_dat[sorted_dat['task'] == 2],order, err=True)
        all_actions = create_order_data(sorted_dat[sorted_dat['task']==1],order,err=True) + \
            create_order_data(sorted_dat[sorted_dat['task']==2],order,err=True)

        new_task_tr = len(sorted_dat[sorted_dat['task']==1]['id'].unique())
        
        ls1 = 2*order       # index of lose switch first component

        eps = 1e-4
        ars = []
        acfs = [] 
        lratios = []
        sliding_ratios = []
            
        #sliding window fits
        if sliding is None:
            sliding = 50

        for index in range(len(regressoxrs)):
            regs = regressors[index]
            acts = all_actions[index]
            sess_ratio = []
            if overlap:
                for i in range(len(acts) - sliding):
                    fits = logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=False).x
                    if not diff:
                        # filt = median_filter(fits[ls1+1]/(fits[ls1]+eps), 3*order)
                        filt = fits[ls1+1]/(fits[ls1]+eps)
                        sliding_ratios.append(filt)
                    else:
                        sliding_ratios.append(fits[ls1+1]-(fits[ls1]))
                    sess_ratio.append(sliding_ratios[-1])
            else:
                pass
            acfs.append(acf(np.array(sess_ratio),length=100))
            ars.append(np.corrcoef(sess_ratio[1:],sess_ratio[:-1])[0,1])
            lratios.append(np.var(sess_ratio)) 
        acfs = np.vstack(acfs)
        acf_err = np.std(acfs,axis=0)
        acfs = np.mean(acfs,axis=0)
        ax[0].set_xlabel('Session')
        ax[1].set_xlabel('Session')
        ax[2].set_xlabel('t')
        if not diff:
            ax[0].set_title('Session-Averaged LS(2)/LS(1) Autocorrelation') # from sliding window
            ax[1].set_title('Session-Averaged LS(2)/LS(1) Variance')
            
            ax[0].set_ylabel('LS(2)/LS(1) Autocorrelation')
            ax[1].set_ylabel('LS(2)/LS(1) Variance')
        else:
            ax[0].set_title('Session-Averaged LS(2)-LS(1) Autocorrelation') # from sliding window
            ax[1].set_title('Session-Averaged LS(2)-LS(1) Variance')
            
            ax[0].set_ylabel('LS(2)-LS(1) Autocorrelation')
            ax[1].set_ylabel('LS(2)-LS(1) Variance')
        ax[2].set_ylabel('correlation')
        ax[2].set_title('ACF')
        ax[0].plot(ars)
        ax[1].plot(lratios)
        ax[2].plot(acfs)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        ax[2].fill_between(np.arange(len(acfs)),acfs - acf_err, acfs+acf_err, alpha = .25, facecolor = colors[col_counter])
        col_counter += 1
    
    
    ax[1].legend(monkeys)
    plt.tight_layout()
    plt.show()


# Originally at line 4689
def WSLS_reward_comparison(ax,model, data, order=5, save = '', bias = False, err = False, sliding = None, mask = None, fitted_RL = False,legend = False):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    sols = {}
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
        ord = min(order, len(episode_actions[0]) - 1)    
        if fitted_RL: 
            all_actions = np.vstack(episode_actions).T[:,ord:]
        else:
            all_actions = np.hstack(episode_actions).T[:,ord:]
        if mask is not None:
            mask = np.hstack(mask).T[:,ord:]
        else:
            mask = [None] * all_actions.shape[0]
        regressors = []
        regressors_reward = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))
            regressors_reward.append(data_parse_two_regressors(episode_actions[i],episode_states[i], episode_rewards[i], ord))
    else:
        ord = order
        regressors = parse_monkey_behavior_reduced(data,ord,False,err)
        regressors_reward = parse_monkey_behavior_minimal(data,ord,err)
        all_actions = create_order_data(data,ord, err)
        if mask is not None:
            mask = [mask[i][ord:] for i in range(len(mask))]
        else:
            if err:
                mask = [None] * len(all_actions)
     
    if not err:
        all_actions = all_actions.ravel()
        regressors = np.vstack(regressors)
        action_regression = logistic_regression(regressors, all_actions,bias=bias,mask=mask)
        reward_regression = logistic_regression_reduced(regressors_reward, all_actions, action_regression.x[:ord])
        sols['action'] = action_regression.x
        sols['reward'] = reward_regression.x[ord:]
    else:
        if sliding is not None:
            fits = [] 
            subregs = []
            subacts = []

            for index in range(len(regressors)):
                regs = regressors[index]
                acts = all_actions[index]

                for i in range(len(acts) - sliding):
                    fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=bias).x)
                    subregs.append(regs[i:i+sliding])
                    subacts.append(acts[i:i+sliding])
            fits = np.array(fits)
            sols['action'] = np.mean(fits,axis=0).squeeze()
            sols['err'] = np.std(fits,axis=0).squeeze()
            regressors = np.array(subregs)
            all_actions = np.array(subacts) # do i need to weight this somehow
                
        else:
            fits = []
            lens = []

            for index in range(len(regressors)):
                fits.append(logistic_regression(regressors[index],all_actions[index],bias=bias, mask=mask[index]).x)
                lens.append(len(all_actions[index]))
            fits = np.array(fits)
            sols['action'] = np.average(fits,axis=0,weights=lens).squeeze()
            sols['err'] = np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens)).squeeze()
        
        if bias:
            action_fit = np.hstack([regressors[index] @ fits[index][:-1] + fits[index][-1] for index in range(len(fits))])
        else:
            action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
        all_actions = np.hstack(all_actions)
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
    
    
    xord = np.arange(1,1+ord)

    # ax.set_xlabel(r'trial number')
    ax.set_xticks(range(1,ord+1))
    # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    # sols['action'] = [sols['action'][i] if sols['action'][i] != 0 else np.nan for i in range(len(sols['action']))]
    # sols['reward'] = [sols['reward'][i] if sols['reward'][i] != 0 else np.nan for i in range(len(sols['reward']))]
    reggy = ['agent choice', 'win stay', 'lose switch','reward']
    
    if not err:
        for i in range(len(sols['action'])//ord):
            ax.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
        ax.plot(xord,sols['reward'], label = reggy[-1])
    else:
        for i in range(len(sols['action'])//ord):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            ax.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
            ax.fill_between(xord,sols['action'][i*ord:(i+1)*ord] - sols['err'][i*ord:(i+1)*ord], sols['action'][i*ord:(i+1)*ord]+sols['err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
    # ax.set_title(r'action regression coefficients'.format(acc))
    # ax.set_ylabel('coefficient')
    if legend:
        ax.legend()
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    # ax2.legend(['agent choice', 'win stay', 'lose switch'])
    ax.axhline(linestyle = '--', color = 'k', alpha = .5)
    if save != '':
        plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
    
    return ax

# like logistic_regression_masked but returns only fit accuracy

