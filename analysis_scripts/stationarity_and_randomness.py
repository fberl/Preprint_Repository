import numpy as np
import matplotlib.pyplot as plt
import scipy
from analysis_scripts.logistic_regression import paper_logistic_accuracy,paper_logistic_accuracy_combinatorial,\
        create_order_data, parse_monkey_behavior_reduced, parse_monkey_behavior_combinatorial, logistic_regression,\
            histogram_logistic_accuracy, histogram_logistic_accuracy_combinatorial, sigmoid, histogram_logistic_accuracy_strategic, paper_logistic_accuracy_strategic
from envs.mp_env import MPEnv
import pandas as pd
from models.misc_utils import make_env
from analysis_scripts.LLH_behavior_RL import single_session_fit, multi_session_fit
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from random import random
from scipy.stats import binomtest
from seaborn import kdeplot, violinplot
from cycler import cycler
from analysis_scripts.yule_walker import yule_walker, pacf
from matplotlib.lines import Line2D


def compare_algos(df, fit_type, n = 10):
    # takes in the list of data for mp1 and mp2
    # use bins larger than one session?
    # uses that to compute confidence intervals for the parameter estimates for each monkey
    # then shows that CIs are non-overlapping or otherwise have some probability of being different
    # returns a plot with the CIs for each monkey
    # data input is in form of a dataframe
    # fit_type is either 'RL' or 'LR'
    mp1_data = df[(df['task'] == 1)][fit_type].iloc[-n:]
    mp2_data = df[(df['task'] == 2)][fit_type].iloc[-n:]
    
    # isolate the last n trials
    

# how often does matching pennies guess randomly for the monkey instead of by finding a pattern?
def matching_pennies_check(choice,rew,maxdepth,alpha,algo = "all",const_bias = 0):
    testAlpha = alpha/2
    pComputerRight,bias,biasDepth,maxDev,whichAlg = 0.5 + const_bias ,0,-1,0,0

    if choice is None:
        return (1 if random() < pComputerRight else 0), pComputerRight, [0,-1,0,0,const_bias]
    data = np.array(choice)
    choice, rew = np.array(choice) , np.array(rew) #recode as 1/2
    choice = np.append(choice,None)
    rew = np.append(rew,None)
   
    #algorithm 1
    if algo == 1 or algo == "all" or algo == '1':
        for depth in range(maxdepth):
            if len(data) < depth + 1:
                continue
            if depth == 0: #count all right choices to check overall side bias
                countN = len(choice)
                countRight = np.sum(data)
            else: #look for bias on trials where recent choice history was the same
                histseq = np.zeros(len(choice))
                for trial in range(depth+1,len(choice)):
                    seq = 0
                    for currdepth in range(1,depth):
                        seq += choice[trial-currdepth]* (10**(currdepth-1))
                    histseq[trial] = seq
                idx = np.where(histseq == histseq[-1])[0][:-1]
                # if not idx:
                if len(idx) == 0:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)

            # pRightBias = 1 - binom_test(countRight-1,countN,0.5) #p(X>=x)
            # pLeftBias = binom_test(countRight,countN,0.5) #p(X<=x)
            pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
            pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
            np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 1

    #algorithm 2
    if algo == 2 or algo == "all" or algo == '2':
        #checks choice and reward history for bias, no depth 0 and reward history included
        for depth in range(maxdepth):
            if len(data) < depth+1:
                continue
            chistseq = np.empty(len(choice))
            rhistseq = np.empty(len(rew))

            for trial in range(depth+1,len(choice)):
                cseq, rseq = 0,0
                for currdepth in range(1,depth):
                    cseq += choice[trial-currdepth] * 10 ** (currdepth-1)
                    rseq += rew[trial-currdepth] * 10 ** (currdepth-1)
                chistseq[trial] = cseq
                rhistseq[trial] = rseq
            idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
            idx = idx[0][:-1]
            # if not idx:
            if len(idx) == 0:
                continue
            countRight = np.sum((data[idx]))
            countN = len(idx)
            # pRightBias = 1 - binom_test(countRight-1,countN,0.5) #p(X>=x)
            # pLeftBias = binom_test(countRight,countN,0.5) #p(X<=x)
            
            pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
            pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
                np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 2

    biasInfo = [bias,biasDepth,maxDev,whichAlg,const_bias]
    #might need to flip 0 and 1!
    computerChoice = 1 if random() < pComputerRight + const_bias else 0
    # return computerChoice,pComputerRight,biasInfo
    # this second term is the probability of choosing right for the computer
    return computerChoice, pComputerRight + const_bias



def process_data(data,mp_only = False):
    episode_rewards = []
    episode_actions = []
    cflag = mp_only
    tasklist = list(sorted(data['task'].unique()))
    center = 0
    for task in tasklist:
        if task == 0: continue
        task_data = data[data['task'] == task]
        if cflag == 0 and len(tasklist) > 0:
            cflag = 1
            center = len(task_data['id'].unique())
        for sess in list(sorted(task_data['id'].unique())):
            ep = task_data[task_data['id'] == sess]
            episode_actions.append(ep['monkey_choice'].to_numpy())
            episode_rewards.append(ep['reward'].to_numpy())
    return episode_actions, episode_rewards, center
    
def matching_pennies_random_prob(data, maxdepth = 4, alpha = .05, algo = 'all', const_bias = 0, ax = None):
    plot_flag = 0
    episode_actions, episode_rewards = process_data(data)
    episode_chance = []
    episode_err = []
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        chance = []
        for trial in range(len(actions)):
            choice = actions[:trial]
            rew = rewards[:trial] 
            choice, pRight = matching_pennies_check(choice,rew,maxdepth,alpha,algo,const_bias)
            if pRight >= .5:
                guess_prob = 1 - pRight
            else:
                guess_prob = pRight
            chance.append(guess_prob)
        episode_chance.append(np.mean(chance))
        episode_err.append(np.std(chance[-1]))
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    ax.set_title('Random Guessing Probability for Matching Pennies Algorithim {}'.format(algo))
    ax.set_xlabel('Session')
    ax.set_ylabel('Probability of Random Guess')
    session_nums = np.arange(1,1+len(episode_chance))
    episode_chance = np.array(episode_chance)
    episode_err = np.array(episode_err)
    ax.plot(session_nums,episode_chance)
    ax.fill_between(session_nums, episode_chance - episode_err, episode_chance + episode_err, alpha = .25)
    if plot_flag:
        plt.show()
    return ax

 
def matching_pennies_random_prob_model(data, maxdepth = 4, alpha = .05, algo = 'all', const_bias = 0, ax = None):
    plot_flag = 0
    episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, BGChoices = data
    episode_chance = []
    episode_err = []
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        chance = []
        for trial in range(len(actions)):
            choice = actions[:trial]
            rew = rewards[:trial] 
            choice, pRight = matching_pennies_check(choice,rew,maxdepth,alpha,algo,const_bias)
            if pRight >= .5:
                guess_prob = 1 - pRight
            else:
                guess_prob = pRight
            chance.append(guess_prob)
        episode_chance.append(np.mean(chance))
        episode_err.append(np.std(chance[-1]))

    session_nums = np.arange(1,1+len(episode_chance))
    episode_chance = np.array(episode_chance)
    episode_err = np.array(episode_err)
    return episode_chance, episode_err

       
def RL_prediction_accuracy(data,model='asymmetric',const_beta = False, const_gamma = True, punitive = False, decay = False, ftol=1e-8,
                            ax = None, alpha=None, mp_only = False, labels = True,monkey=None,**plot_kwargs):
    plot_flag = 0
    episode_actions, episode_rewards, center  = process_data(data,mp_only=mp_only)
    fit_accuracy = []
    coeff_dict = {'1':[],'2':[]}
    fit_dict = {'1':[],'2':[]}  
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        fit, perf = single_session_fit(actions,rewards,model = model,const_beta = const_beta,const_gamma = const_gamma, 
                                       punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
        fit_accuracy.append(perf)
        if session < center:
            fit_dict['1'].append(perf)
            coeff_dict['1'].append(fit)
        else:
            fit_dict['2'].append(perf)
            coeff_dict['2'].append(fit)

    # if ax is None:
    #     plot_flag = 1
    #     fig, ax = plt.subplots()
    if ax != None:
        if labels:
            ax.set_title('RL Prediction Accuracy for {}'.format(model))
        ax.set_xlabel('Session', fontsize = 16)
        ax.set_ylabel('Prediction Accuracy', fontsize = 16)
        session_nums = np.arange(1,1+len(episode_actions)) - center + .5
        line_handle = ax.plot(session_nums,fit_accuracy,label = monkey)
        if plot_flag:
            plt.show()
    else:
        line_handle = None
    return fit_dict, line_handle, coeff_dict

def histogram_RL_accuracy(data,model='asymmetric',const_beta = False, const_gamma = True, punitive = False, decay = False, ftol=1e-8,
                            ax = None, alpha=None, mp_only = False, labels = True, **plot_kwargs):
    episode_actions, episode_rewards, center  = process_data(data,mp_only=mp_only)
    fit, perf = multi_session_fit(episode_actions,episode_rewards,model = model,const_beta = const_beta,const_gamma = const_gamma, 
                                       punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
    return fit, perf

                          
                         
def violin_RL_accuracy(data,model='asymmetric',const_beta = False, const_gamma = True, punitive = False, decay = False, ftol=1e-8,
                            ax = None, alpha=None, mp_only = False, labels = True, **plot_kwargs):
    episode_actions, episode_rewards, center  = process_data(data,mp_only=mp_only)
    # fit, perf = multi_session_fit(episode_actions,episode_rewards,model = model,const_beta = const_beta,const_gamma = const_gamma, 
    #                                    punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
    fits = []
    perfs = []
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        fit, perf = single_session_fit(actions,rewards,model = model,const_beta = const_beta,const_gamma = const_gamma, 
                                       punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
        fits.append(fit)
        perfs.append(perf)
    return fits, perfs

                
def violin_LR_accuracy(data,order =5,bias = False, mask = None,ax = None, labels = True,mp_only = False, **plot_kwargs):
    episode_actions, episode_rewards, center  = process_data(data,mp_only)
    fits = []
    perfs = []
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        fit, perf = paper_logistic_accuracy_strategic(actions,rewards,order,bias,mask)
        fits.append(fit)
        perfs.append(perf)
    return fits, perfs
def logistic_prediction_accuracy(data,order =5,bias = False, mask = None,ax = None, labels = True,mp_only = False, **plot_kwargs):
    plot_flag = 0
    episode_actions, episode_rewards, center  = process_data(data,mp_only)
    fit_accuracy = []
    fit_dict = {'1':[],'2':[]}
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        perf = paper_logistic_accuracy(actions,rewards,order,bias,mask)
        fit_accuracy.append(perf)
        
        if session < center:
            fit_dict['1'].append(perf)
        else:
            fit_dict['2'].append(perf)

            
    # if ax is None:
    #     plot_flag = 1
    #     fig, ax = plt.subplots()
    if ax != None:
        if labels:
            ax.set_title('Logistic Regression Prediction Accuracy')
        ax.set_xlabel('Session')
        ax.set_ylabel('Prediction Accuracy')
        session_nums = np.arange(1,1+len(episode_actions)) - center + .5
        line_handle = ax.plot(session_nums,fit_accuracy, **plot_kwargs)
        if plot_flag:
            plt.show()
    else:
        line_handle = None
    return fit_dict, line_handle

# takes data for N monkeys for MP 1 and MP 2
# Aligns the data such that the last MP 1 session and the first MP 2 session are in the 
# same place on the plot, and plots a vertical line to denote the transition.
# RL Predictions are solid, while logistic predictions are dashed.
# The x-axis is the session number, and the y-axis is the prediction accuracy.
# Each monkey is Labeled, and assigned a different color
def plot_predictiability_monkeys(data, ax = None,hist_ax = None, **plot_kwargs):
    plot_flag = 0
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    # if not isinstance(data,list):
    #     data = [data]
    colors = []
    monkeys = data['animal'].unique()
    pred1_RL = []
    pred2_RL = []
    pred1_log = []
    pred2_log = []
    for monkey in monkeys:
        monkey_data = data[data['animal'] == monkey]
        data_rl, handle = RL_prediction_accuracy(monkey_data,ax = ax, labels = False,monkey = monkey,**plot_kwargs)
        colors.append(handle)
        data_log, _ = logistic_prediction_accuracy(monkey_data,ax = ax,labels = False, color = ax.lines[-1].get_color(),linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        
        pred1_RL.extend(data_rl['1'])
        pred2_RL.extend(data_rl['2'])
        pred1_log.extend(data_log['1'])
        pred2_log.extend(data_log['2'])
        
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    ax.set_xticks([center1, center2])
    ax.set_xticklabels(['Algorithm 1', 'Algorithm 2'])
    
    black_line = mlines.Line2D([], [], color='black', marker='s',linestyle="-",
                          markersize=0, label='RL Model')
    black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle="--",
                          markersize=0, label = 'Logistic Regression')
    
    ax.set_title('RL and LR Prediction Accuracy for Monkeys Playing Matching Pennies')
    ax.set_ylabel('Accuracy')
    
    prop_cycle = prop_cycle = plt.rcParams['axes.prop_cycle']
    cols = prop_cycle.by_key()['color']

    if hist_ax != None:
        kdeplot(pred1_RL, ax = hist_ax, label='MP 1 RL', clip=(.5,1), color = cols[0]) 
        kdeplot(pred1_log, ax = hist_ax, label='MP 1 LR', clip=(.5,1), color=cols[0],linestyle = '--')
        kdeplot(pred2_RL, ax = hist_ax, label='MP 2 RL', clip=(.5,1), color = cols[1])
        kdeplot(pred2_log, ax = hist_ax, label='MP 2 LR', clip=(.5,1), linestyle = '--', color = cols[1])
    
    # ax.legend(handles = [black_line,black_line2], ['RL Model', 'Logistic Regression'])
    # ax.legend(colors, monkeys)
    


    # handles = [mpatches.Patch(color=line.get_color()) for line in scatter1.legend_elements()[0]]
    # leg1 = ax.legend(handles, handle.legend_elements()[1], bbox_to_anchor=(1.04, 1), loc="upper left", title="Legend")
    # ax.add_artist(leg1)

    # handles = [mlines.Line2D([], [], marker=marker, mec='k', mfc='w', ls='') for marker in ['o', '^']]
    # ax.legend(handles, ['Radial', 'Transit'], loc=(1.01, 0),title="Detection")
    # ax.legend('Monkey')
    # ax.legend()
    ax2 = ax.twinx()
    styles = ['-', '--']
    labels = ['RL', 'LR']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc = 'upper right')
    
    if plot_flag:
        plt.show()
    return ax

def plot_predictiability_monkeys_violin(data, mp2_data = None, ax = None,hist_ax = None, RL = True, Log = True, combinatorial = False, order=5,alpha=None,**plot_kwargs,):
    plot_flag = 0
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    # if not isinstance(data,list):
    #     data = [data]
    colors = []
    monkeys = data['animal'].unique()
    pred1_RL = []
    pred2_RL = []
    pred1_log = []
    pred2_log = []
    monkey_list_dict = []
    monkeys_reorder = [13, 112, 18]
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}
    
    # Collect coefficient dictionaries for each monkey
    all_coeff_dicts = {}

    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    for monkey in monkeys:
        monkey_data = data[data['animal'] == monkey]
        if RL == True:
            data_rl, handle, coeff_dict = RL_prediction_accuracy(monkey_data,ax = ax, labels = False,monkey = monkeys_reorder_dict[monkey],alpha=alpha,**plot_kwargs)
            # Store coefficient dictionary for this monkey
            all_coeff_dicts[monkeys_reorder_dict[monkey]] = coeff_dict
        else:
            data_rl, _, coeff_dict = RL_prediction_accuracy(monkey_data,ax = None,alpha=alpha, labels = False,**plot_kwargs)
            # Store coefficient dictionary for this monkey
            all_coeff_dicts[monkeys_reorder_dict[monkey]] = coeff_dict
        if Log == True:
            if RL == True:
                data_log, _ = logistic_prediction_accuracy(monkey_data,ax = ax,labels = False, color = ax.lines[-1].get_color(),linestyle = 'dashed', **plot_kwargs)
            else:
                # else:
                data_log, _ = logistic_prediction_accuracy(monkey_data,ax = ax,labels = False, **plot_kwargs)
                if combinatorial:
                    data_rl, _ = logistic_prediction_accuracy_combinatorial(monkey_data,order = 2*order, ax = ax, color= ax.lines[-1].get_color(),
                                                                            labels = False,linestyle = 'dashed', **plot_kwargs)
        else:
            data_log, _ = logistic_prediction_accuracy(monkey_data,ax = None,labels = False, color = ax.lines[-1].get_color(),linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        
        # pred1_RL.extend(data_rl['1'])
        # pred2_RL.extend(data_rl['2'])
        # pred1_log.extend(data_log['1'])
        # pred2_log.extend(data_log['2'])
        
        dic = [{'monkey':monkey, 'RL':data_rl['1'][i], 'Log':data_log['1'][i], 'task':1} for i in range(len(data_rl['1']))]
        dic.extend([{'monkey':monkey, 'RL':data_rl['2'][i], 'Log':data_log['2'][i], 'task':2} for i in range(len(data_rl['2']))])
        monkey_list_dict.extend(dic)
        
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    ax.set_xticks([center1, center2])
    ax.set_xticklabels(['Algorithm 1', 'Algorithm 2'], fontsize = 16)
    
    if RL and Log:
        black_line = mlines.Line2D([], [], color='black', marker='s',linestyle="-",
                            markersize=0, label='RL Model')
        black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle="--",
                            markersize=0, label = 'Logistic Regression')
    elif RL:
        black_line = mlines.Line2D([], [], color='black', marker='s',linestyle="-",
                            markersize=0, label='RL Model')
    elif Log and combinatorial:
        black_line = mlines.Line2D([], [], color='black', marker='s',linestyle="-",
                            markersize=0, label='Logistic Regression')
        black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle="--",
                            markersize=0, label = 'Expanded Regression')
    elif Log:
        black_line2 = mlines.Line2D([], [], color='black', marker='s',linestyle="--",
                            markersize=0, label = 'Logistic Regression')
        
    
    if Log and RL:
        title = 'RL and LR Prediction Accuracy for Monkeys \n Playing Matching Pennies'
    elif Log:
        title = 'LR Prediction Accuracy for Monkeys \n Playing Matching Pennies'
    elif RL:
        title = 'RL Prediction Accuracy for Monkeys \n Playing Matching Pennies'
        
    ax.set_title(title, fontsize = 20)
    ax.set_ylabel('Accuracy', fontsize = 16)
    # ax.set_yticks([0.5,0.6,0.7,0.8,0.9,1],)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)    
    prop_cycle = prop_cycle = plt.rcParams['axes.prop_cycle']
    cols = prop_cycle.by_key()['color']

        
    if hist_ax != None:
        monkey_df = pd.DataFrame(monkey_list_dict)
        colors_ws = []
        colors_ls = []
        positions_ws = [0, 2, 4]
        positions_ls = [1, 3, 5]
        positions_labels = [.5, 2.5,4.5]
        labels = [monkeys_reorder_dict[monkey] for monkey in monkeys_reorder]
        #win stay for task 1 for each monkey
        if RL == True or combinatorial == True:
            if Log == False:
                side = 'both'
            elif combinatorial:
                side = 'high'
            else:
                side = 'low'
            RL_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['RL'] for monkey in monkeys_reorder]
            RL_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['RL'] for monkey in monkeys_reorder]
            vp1 = hist_ax.violinplot(RL_1, positions_ws, widths=0.5,side=side, showmeans=True, showextrema=False)
            vp3 = hist_ax.violinplot(RL_2, positions_ls, widths=0.5,side=side, showmeans=True, showextrema=False)

        if Log == True:
            if combinatorial == False:
                side = 'both'
            else:
                side = 'low'
            logistic_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['Log'] for monkey in monkeys_reorder]
            logistic_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['Log'] for monkey in monkeys_reorder]
            vp2 = hist_ax.violinplot(logistic_1, positions_ws, widths=0.5,side=side, showmeans=True, showextrema=False)
            vp4 = hist_ax.violinplot(logistic_2, positions_ls, widths=0.5,side=side, showmeans=True, showextrema=False)
        hist_ax.set_xticks(positions_labels)
        hist_ax.set_xticklabels(labels)
        # ax5 = hist_ax.twinx()

        hist_ax.set_xticks(positions_labels)
        hist_ax.set_xticklabels(labels)
        
        vp_labels = []
        vp_handles = []
        try:
            for pc in vp1['bodies']:
                pc.set_facecolor('mediumspringgreen')
                pc.set_edgecolor('black')
            vp_labels.append('RL 1' if not combinatorial else 'E LR 1')   
            vp_handles.append(mpatches.Patch(color='mediumspringgreen'))
        except:
            pass
        
        try:
            for pc in vp2['bodies']:
                pc.set_facecolor('mediumaquamarine')
                pc.set_edgecolor('black')
            if combinatorial:
                vp_labels.insert(0,'LR 1')
                vp_handles.insert(0,mpatches.Patch(color='mediumaquamarine'))

            else:
                vp_labels.append('LR 1')
                vp_handles.append(mpatches.Patch(color='mediumaquamarine'))
        except:
            pass
        
        try:
            for pc in vp3['bodies']:
                pc.set_facecolor('coral')
                pc.set_edgecolor('black')
            vp_labels.append('RL 2' if not combinatorial else 'E LR 2')
            vp_handles.append(mpatches.Patch(color='coral'))
        except:
            pass
        
        try:
            for pc in vp4['bodies']:
                pc.set_facecolor('orangered')
                pc.set_edgecolor('black')
            if combinatorial:
                vp_labels.insert(-1,'LR 2')
                vp_handles.insert(-1,mpatches.Patch(color='orangered'))

            else:
                vp_labels.append('LR 2')
                vp_handles.append(mpatches.Patch(color='orangered'))
        except:
            pass
    if hist_ax != None:
        # hist_ax.legend(loc='upper left',labels=vp_labels,
        #        handles=vp_handles,frameon=False,ncol=len(vp_labels))
        
            box = hist_ax.get_position()
            hist_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            leg = hist_ax.legend(loc='center left', bbox_to_anchor=(.8, 0.5),labels=vp_labels,
                           handles=vp_handles,frameon=False)
            # leg = hist_ax.legend(loc='outside center left',labels=vp_labels,
                        #    handles=vp_handles,frameon=False)
            leg.set_in_layout(False)
            # fig.set_layout_engine('none')
            
    # ax.legend(handles = [black_line,black_line2], labels=['RL Model', 'Logistic Regression'])
    # ax.legend(colors, monkeys)
    


    # handles = [mpatches.Patch(color=line.get_color()) for line in scatter1.legend_elements()[0]]
    # leg1 = ax.legend(handles, handle.legend_elements()[1], bbox_to_anchor=(1.04, 1), loc="upper left", title="Legend")
    # ax.add_artist(leg1)

    # handles = [mlines.Line2D([], [], marker=marker, mec='k', mfc='w', ls='') for marker in ['o', '^']]
    # ax.legend(handles, ['Radial', 'Transit'], loc=(1.01, 0),title="Detection")
    ax.legend(loc = 'upper right',frameon=False)
    # ax.legend()
    
    ax2 = ax.twinx()
    styles = ['-', '--']
    if RL and Log:
        labels = ['RL', 'LR']
    else:
        pass
        # if RL:
        #     labels = ['RL']
        # elif Log:
        #     labels = ['LR'] if not combinatorial else ['LR','Expanded LR']
        # for ss, sty in enumerate(labels):
        #     ax2.plot(np.NaN, np.NaN, ls=styles[ss],
        #             label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    if Log or RL:
        ax2.legend(loc = 'upper right',frameon=False)
    
    if plot_flag:
        plt.show()
    return ax, all_coeff_dicts 

# fits a model to all the data, then computes accuracy. Also checks for overfitting for Logistic Regressiom
def compute_predictability(data, order=5,overfit = False, violin = False):
    monkeys = list(data['animal'].unique())
    overfits = {}
    perf_fits = {}
    violin_monkeys = {}
    
    for monkey in monkeys:
        monkey_data = data[data['animal'] == monkey]
        
        # RL_trunc, perf_RL_trunc = histogram_RL_accuracy(monkey_data,ax = None, labels = False,alpha=1/order, mp_only=True)
        
        # trunc_order = max(int(1/(1-RL_trunc[0])),2) #let's round down to be generous
        # LR_trunc = histogram_logistic_accuracy(monkey_data, order = trunc_order)  
        # LR_trunc = histogram_logistic_accuracy_strategic(monkey_data, order = trunc_order)
        
        
        RL_full, perf_RL_full = histogram_RL_accuracy(monkey_data,ax = None, labels = False,mp_only=True,alpha=None)
        
        if violin:
            violin_RL, violin_perf = violin_RL_accuracy(monkey_data,ax = None, labels = False,mp_only=True,alpha=None)
            # violin_monkeys[monkey] = {'RL':violin_RL, 'perf':violin_perf}
            violin_LR, violin_perf_LR = violin_LR_accuracy(monkey_data,ax = None, labels = False,mp_only=True,alpha=None)
            violin_monkeys[monkey] = {'RL':violin_RL, 'perf':violin_perf, 'LR':violin_LR, 'perf_LR':violin_perf_LR}
            
        # full_order = max(int(1/(1-RL_full[0])),5) #let's round down to be generous
        full_order = 5
        
        # LR_full = histogram_logistic_accuracy_combinatorial(monkey_data,order = full_order)
        LR_full = histogram_logistic_accuracy_strategic(monkey_data,order = full_order)

        
        # perf_fits[monkey] = {'RL':perf_RL_trunc, 'LR':LR_trunc, 'RL_full':perf_RL_full, 'LR_full':LR_full}
        perf_fits[monkey] = {'RL':perf_RL_full, 'LR':LR_full}        
        #also check if full LR is overfitting by comparing out of sample performance for regular and extended LR
        if overfit: 
            # split the monkey data up into training and testing
            # then fit lrfull and lr trunc with same memory as lrfull and compare
            test_sessions = monkey_data['id'].unique()[-5:]
            train_sessions = monkey_data['id'].unique()[:-5]
            
            test_data = monkey_data[monkey_data['id'].isin(test_sessions)]
            train_data = monkey_data[monkey_data['id'].isin(train_sessions)]
            # test_data_LR = np.vstack(parse_monkey_behavior_reduced(test_data,full_order,False,False))
            # train_data_LR = np.vstack(parse_monkey_behavior_reduced(train_data,full_order,False,False))
            # test_data_Combo = np.vstack(parse_monkey_behavior_combinatorial(test_data,full_order,False))
            # train_data_Combo = np.vstack(parse_monkey_behavior_combinatorial(train_data,full_order,False))
            test_data_LR = np.vstack(parse_monkey_behavior_reduced(test_data,order,False,False))
            train_data_LR = np.vstack(parse_monkey_behavior_reduced(train_data,order,False,False))            
            test_data_Combo = np.vstack(parse_monkey_behavior_reduced(test_data,full_order,False,False))
            train_data_Combo = np.vstack(parse_monkey_behavior_reduced(train_data,full_order,False,False))      
            test_actions = create_order_data(test_data,full_order, False).ravel()
            train_actions = create_order_data(train_data,full_order, False).ravel()
            
            # now fit to the training data 
            LR = logistic_regression(train_data_LR, train_actions,bias=False,mask=None).x
            Combo = logistic_regression(train_data_Combo, train_actions,bias=False,mask=None, ).x

            # Combo = logistic_regression(train_data_Combo, train_actions,bias=False,mask=None).x
  
            # now compute accuracy on test data
            fit_LR = test_data_LR @ LR
            acc_LR = np.mean(np.round(sigmoid(fit_LR)) == test_actions)
            
            fit_Combo = test_data_Combo @ Combo
            acc_Combo = np.mean(np.round(sigmoid(fit_Combo)) == test_actions)
            
            
            overfits[monkey] = {'LR':acc_LR, 'Combo':acc_Combo}
            
    if violin:
        return violin_monkeys    
    return perf_fits, overfits
        

def plot_RL_timescales(data, rl_model = 'asymmetric', window = 0, ax = None, hist_ax = None, **plot_kwargs):
    plot_flag = 0
    
    if rl_model == 'simple':
        model_kwargs = {'const_beta': True, 'const_gamma': True, 'punitive': True, 'decay': True, 'ftol': 1e-8, 'model' : 'simple'}
    elif rl_model == 'forgetting':
        model_kwargs = {'const_beta': True, 'const_gamma': True, 'punitive': False, 'decay': False, 'ftol': 1e-8, 'model' : 'forgetting'}
    elif rl_model == 'asymmetric':
        model_kwargs = {'const_beta': False, 'const_gamma': True, 'punitive': False, 'decay': False, 'ftol': 1e-8, 'model' : 'asymmetric'}
    
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    colors = []
    # monkey_label_dict = 
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}
    monkeys = data['animal'].unique()
    handles = []
    mp1_alphas = []
    mp2_alphas = []
    for monkey in monkeys:    
        alphas = []
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        for i in range(window,len(episode_actions) - window):
            actions = episode_actions[i-window:i+window+1]
            rewards = episode_rewards[i-window:i+window+1]
            if window == 0:
                actions = np.array(actions)
                rewards = np.array(rewards)
                fit, perf = single_session_fit(actions[0],rewards[0],**model_kwargs)
            else:
                # actions = [np.array(a) for a in actions]
                # rewards = [np.array(r) for r in rewards]
                fit, perf = multi_session_fit(actions,rewards,**model_kwargs)
            alphas.append(fit[0])
            if i < center:
                mp1_alphas.append(fit[0])
            else:
                mp2_alphas.append(fit[0])
            if perf < .5:
                Exception('RL performs worse than chance')
        
        session_nums = np.arange(1,1+len(alphas)) - center
        alphas = np.array(alphas)
        # ax.plot(session_nums,1/alphas)
        handle = ax.plot(session_nums,alphas,label = monkeys_reorder_dict[monkey])
        ax.set_yscale('log')
        

        # colors.append(ax.lines[-1].get_color())
        
    if hist_ax !=None:
        kdeplot(mp1_alphas, ax = hist_ax,label='MP 1 Timescales', clip=(0,1)) 
        kdeplot(mp2_alphas, ax = hist_ax,label='MP 2 Timescales', clip=(0,1)) 
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    # ax.set_xticks([center1, center2])
    # ax.set_xticklabels(['Algorithm 1', 'Algorithm 2 '])
    ax.set_xticks([])
    ax.set_title(r"RL fit $\alpha$'s for Monkeys Playing Matching Pennies")
    ax.set_ylabel(r'RL fit $\alpha$')
    ax.legend(handles, ['Monkey C', 'Monkey E', 'Monkey F'], loc = 'upper right')
        
    if plot_flag:
        plt.show()
    
    return ax



def plot_RL_timescales_violin(data, rl_model = 'asymmetric', window = 0, ax = None, hist_ax = None, **plot_kwargs):
    plot_flag = 0
    
    if rl_model == 'simple':
        model_kwargs = {'const_beta': True, 'const_gamma': True, 'punitive': True, 'decay': True, 'ftol': 1e-8, 'model' : 'simple'}
    elif rl_model == 'forgetting':
        model_kwargs = {'const_beta': True, 'const_gamma': True, 'punitive': False, 'decay': False, 'ftol': 1e-8, 'model' : 'forgetting'}
    elif rl_model == 'asymmetric':
        model_kwargs = {'const_beta': False, 'const_gamma': True, 'punitive': False, 'decay': False, 'ftol': 1e-8, 'model' : 'asymmetric'}
    
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    colors = []
    # monkey_label_dict = 
    monkeys = data['animal'].unique()
    handles = []
    mp1_alphas = []
    mp2_alphas = []
    monkey_list_dict = []

    for monkey in monkeys:    
        alphas = []
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        for i in range(window,len(episode_actions) - window):
            actions = episode_actions[i-window:i+window+1]
            rewards = episode_rewards[i-window:i+window+1]
            if window == 0:
                actions = np.array(actions)
                rewards = np.array(rewards)
                fit, perf = single_session_fit(actions[0],rewards[0],**model_kwargs)
            else:
                # actions = [np.array(a) for a in actions]
                # rewards = [np.array(r) for r in rewards]
                fit, perf = multi_session_fit(actions,rewards,**model_kwargs)
            alphas.append(fit[0])
            if i < center:
                mp1_alphas.append(fit[0])
            else:
                mp2_alphas.append(fit[0])
            if perf < .5:
                Exception('RL performs worse than chance')
                
            dic  = {'monkey':monkey, 'alpha':fit[0], 'task':1 if i < center else 2}
            monkey_list_dict.append(dic)
        session_nums = np.arange(1,1+len(alphas)) - center
        alphas = np.array(alphas)
        # ax.plot(session_nums,1/alphas)
        handle = ax.plot(session_nums,alphas)
        ax.set_yscale('log')
        
        

        # colors.append(ax.lines[-1].get_color())
    
    monkeys_reorder = [13, 112, 18]
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}   
    if hist_ax != None:
        monkey_df = pd.DataFrame(monkey_list_dict)
        colors_ws = []
        colors_ls = []
        #win stay for task 1 for each monkey
        a1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['alpha'] for monkey in monkeys_reorder]
        a2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['alpha'] for monkey in monkeys_reorder]
        positions_labels = [.5, 2.5,4.5]
        labels = [monkeys_reorder_dict[monkey] for monkey in monkeys_reorder]
        hist_ax.set_xticks(positions_labels)
        hist_ax.set_xticklabels(labels)
        vp1 = hist_ax.violinplot(a1, positions_labels, widths=0.5,side='low', showmeans=True, showextrema=False)
        # ax5 = hist_ax.twinx()
        vp2 = hist_ax.violinplot(a2, positions_labels, widths=0.5,side='high', showmeans=True, showextrema=False)
        # ax5.set_xticks([])
        # ax5.set_xticks(positions_labels)
        # ax5.set_xticklabels(labels)
        for pc in vp1['bodies']:
            pc.set_facecolor('deepskyblue')
            pc.set_edgecolor('black')
            
        for pc in vp2['bodies']:
            pc.set_facecolor('orchid')
            pc.set_edgecolor('black')
        
    
    
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    # ax.set_xticks([center1, center2])
    # ax.set_xticklabels(['Algorithm 1', 'Algorithm 2 '])
    ax.set_xticks([])
    ax.set_title(r"RL fit $\alpha$'s for Monkeys Playing Matching Pennies")
    ax.set_ylabel(r'RL fit $\alpha$')
    # ax.legend(handles, ['Monkey C', 'Monkey E', 'Monkey F'], loc = 'upper right')
    
    hist_ax.legend(ncol = 2,loc = 'upper center', handles=[mpatches.Patch(color='deepskyblue'),mpatches.Patch(color='orchid')], frameon=False,  labels=['MP 1', 'MP 2'])
    if plot_flag:
        plt.show()
    
    return ax


def plot_logistic_coefficients(data, order = 2, ax = None, hist_ax = None, **plot_kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    monkeys = data['animal'].unique()
    colors = []
    wratios_1 = []
    lratios_1 = []
    wratios_2 = []
    lratios_2 = []
    for monkey in monkeys:    
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        monkey_data.sort_values(by=['task','id','trial'])
        regressors = parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 1],order,err=True) + \
            parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 2],order, err=True)
        all_actions = create_order_data(monkey_data[monkey_data['task']==1],order,err=True) + \
            create_order_data(monkey_data[monkey_data['task']==2],order,err=True)
        
        ls1 = 2*order       # index of lose switch first component
        ws1 = order         # index of win switch first component
        lratios = []
        wratios = []

        l_handles = []
        w_handles = []
        
        for index in range(len(regressors)):
            fit = logistic_regression(regressors[index],all_actions[index],bias=False).x
        
            lratios.append(fit[ls1+1] - fit[ls1])
            wratios.append(fit[ws1+1] - fit[ws1])
            
            if index < center:
                lratios_1.append(fit[ls1+1] - fit[ls1])
                wratios_1.append(fit[ws1+1] - fit[ws1])
            else:
                lratios_2.append(fit[ls1+1] - fit[ls1])
                wratios_2.append(fit[ws1+1] - fit[ws1])
            
            # lratios.append(fit[ls1]/max(fit[ls1:]))
            # wratios.append(fit[ws1]/max(fit[ws1:ls1]))
        
        session_nums = np.arange(1,1+len(wratios)) - center
        wh = ax.plot(session_nums,wratios, linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        lh = ax.plot(session_nums,lratios,c = colors[-1], **plot_kwargs)
        
        w_handles.append(wh)
        l_handles.append(lh)
        
    ax3 = ax.twinx()
    
    if hist_ax != None:
        kdeplot(wratios_1, ax = hist_ax, label='Win Stay 1', color = colors[0])
        kdeplot(lratios_1, ax = hist_ax, label='Lose Switch 1',  color = colors[0], linestyle = '--')
        kdeplot(wratios_2, ax = hist_ax, label='Win Stay 2', color = colors[1])
        kdeplot(lratios_2, ax = hist_ax, label='Lose Switch 2', color = colors[1], linestyle = '--')
    labs = ['Monkey C', 'Monkey E', 'Monkey F']
    for c, color in enumerate(colors):
        ax3.plot(np.NaN, np.NaN, c = colors[c], label = labs[c])
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc = 'lower left')
    

    ax.set_title('Monkey Behavioral Regression Stationarity')
    # ax.set_xlabel('Session')
    # ax.set_ylabel(r'$LS_t-LS_{t-1}$')
    ax.set_ylabel(r'$R(-1) - R(-2)$')    
    
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    # ax.set_xticks([center1, center2])
    # ax.set_xticklabels(['Algorithm 1', 'Algorithm 2 '])
    ax.set_xticks([])
    
    ax2 = ax.twinx()
    styles = ['--', '-']
    labels = ['Win Stay', 'Lose Switch']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc = 'lower right')
    color_legends = ['C','E','F']
    style_legends = ['Win Stay', 'Lose Switch']


def plot_logistic_coefficients_violin(data, order = 2, ax = None, hist_ax = None, **plot_kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    monkeys = data['animal'].unique()
    colors = []
    wratios_1 = []
    lratios_1 = []
    wratios_2 = []
    lratios_2 = []
    
    wratios_1_hist = []
    lratios_1_hist = []
    wratios_2_hist = []
    lratios_2_hist = []
    
    monkey_list_dict = []
    monkey_ratio_dict = []
    for monkey in monkeys:    
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        monkey_data.sort_values(by=['task','id','trial'])
        regressors = parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 1],order,err=True) + \
            parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 2],order, err=True)
        all_actions = create_order_data(monkey_data[monkey_data['task']==1],order,err=True) + \
            create_order_data(monkey_data[monkey_data['task']==2],order,err=True)
        
        ls1 = 2*order       # index of lose switch first component
        ws1 = order         # index of win switch first component
        lratios = []
        wratios = []
        
        lr2 = []
        wr2 = []

        l_handles = []
        w_handles = []
        

        
        for index in range(len(regressors)):
            fit = logistic_regression(regressors[index],all_actions[index],bias=False).x
        
            lratios.append(fit[ls1+1] - fit[ls1])
            wratios.append(fit[ws1+1] - fit[ws1])
            
            lr2.append(fit[ls1+1] /fit[ls1])
            wr2.append(fit[ws1+1] /fit[ws1])
            
            

            dic = {'monkey':monkey, 'session':index, 'win stay':fit[ws1+1] - fit[ws1], 
                   'lose switch':fit[ls1+1] - fit[ls1], 'task': 1 if index < center else 2}
            monkey_list_dict.append(dic)
            monkey_ratio_dict.append({'monkey':monkey, 'win stay':fit[ws1] / max(fit[ws1:ls1]), 
                   'lose switch':fit[ls1] /max(fit[ls1:]), 'task': 1 if index < center else 2})
            # lratios.append(fit[ls1]/max(fit[ls1:]))
            # wratios.append(fit[ws1]/max(fit[ws1:ls1]))
        
        session_nums = np.arange(1,1+len(wratios)) - center
        wh = ax.plot(session_nums,wratios, linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        lh = ax.plot(session_nums,lratios,c = colors[-1], **plot_kwargs)
        
        w_handles.append(wh)
        l_handles.append(lh)
        
    # ax3 = ax.twinx()
    
    monkeys_reorder = [13, 112, 18]
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}
    if hist_ax != None:
        # monkey_df = pd.DataFrame(monkey_list_dict)
        monkey_df = pd.DataFrame(monkey_ratio_dict)
        colors_ws = []
        colors_ls = []
        #win stay for task 1 for each monkey
        win_stay_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['win stay'] for monkey in monkeys_reorder]
        win_stay_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['win stay'] for monkey in monkeys_reorder]
        lose_switch_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['lose switch'] for monkey in monkeys_reorder]
        lose_switch_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['lose switch'] for monkey in monkeys_reorder]
        positions_ws = [0, 2, 4]
        positions_ls = [1, 3, 5]
        positions_labels = [.5, 2.5,4.5]
        labels = [monkeys_reorder_dict[monkey] for monkey in monkeys_reorder]
        hist_ax.set_xticks([])
        # hist_ax.set_xticks(positions_labels)
        # hist_ax.set_xticklabels(labels)
        vp1 = hist_ax.violinplot(win_stay_1, positions_ws, widths=0.5,side='low', showmeans=True, showextrema=False)
        # ax5 = hist_ax.twinx()
        vp2 = hist_ax.violinplot(win_stay_2, positions_ws, widths=0.5,side='high', showmeans=True, showextrema=False)
        vp3 = hist_ax.violinplot(lose_switch_1, positions_ls, widths=0.5,side='low', showmeans=True, showextrema=False)
        vp4 = hist_ax.violinplot(lose_switch_2, positions_ls, widths=0.5,side='high', showmeans=True, showextrema=False)
        hist_ax.set_xticks([])
        hist_ax.set_xticks(positions_labels)
        hist_ax.set_xticklabels(labels)
        hist_ax.set_yscale('symlog')
        for pc in vp1['bodies']:
            pc.set_facecolor('thistle')
            pc.set_edgecolor('black')
            
        for pc in vp2['bodies']:
            pc.set_facecolor('plum')
            pc.set_edgecolor('black')
        
        for pc in vp3['bodies']:
            pc.set_facecolor('goldenrod')
            pc.set_edgecolor('black')
        
        for pc in vp4['bodies']:
            pc.set_facecolor('darkgoldenrod')
            pc.set_edgecolor('black')


        # violinplot(data = pd.DataFrame(monkey_list_dict), 
        #                    x = 'task', y = ['win_stay','lose_switch'], ax = hist_ax, label='Win Stay', color = colors[0])
    # labs = ['Monkey C', 'Monkey E', 'Monkey F']
    # for c, color in enumerate(colors):
    #     ax3.plot(np.NaN, np.NaN, c = colors[c], label = labs[c])
    # ax3.get_yaxis().set_visible(False)
    # ax3.legend(loc = 'lower left')
    
    # hist_ax.legend(loc = 'upper left', handles=[vp1, vp2,vp3,vp4], labels = ['WS 1', 'WS 2', 'LS 1', 'LS 2'])
    
            
    ax.set_title('Monkey Behavioral Regression Stationarity')
    # ax.set_xlabel('Session')
    # ax.set_ylabel(r'$LS_t-LS_{t-1}$')
    ax.set_ylabel(r'$R(-1) - R(-2)$')    
    
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    # ax.set_xticks([center1, center2])
    # ax.set_xticklabels(['Algorithm 1', 'Algorithm 2 '])
    ax.set_xticks([])
    if hist_ax != None:
        hist_ax.legend(loc='lower center',labels = ['WS 1', 'WS 2', 'LS 1', 'LS 2'],handles=[mpatches.Patch(color='thistle'),mpatches.Patch(color='plum'),
                        mpatches.Patch(color='goldenrod'),mpatches.Patch(color='darkgoldenrod')], ncol=4, frameon=False)
    
    labs = ['Monkey C', 'Monkey E', 'Monkey F']
    ax3 = ax.twinx()

    for c, color in enumerate(colors):
        ax3.plot(np.NaN, np.NaN, c = colors[c], label = labs[c])
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc = 'lower left', frameon=False)
    
    
    ax2 = ax.twinx()
    styles = ['--', '-']
    labels = ['Win Stay', 'Lose Switch']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    line1 = Line2D([0], [0], label='Win Stay', color='k', linestyle='--')
    line2 = Line2D([0], [0], label='Lose Switch', color='k', linestyle='-')
    
    ax2.legend(handles= [line1,line2],loc = 'lower right', labels = ['Win Stay', 'Lose Switch'], frameon = False)
    color_legends = ['C','E','F']
    style_legends = ['Win Stay', 'Lose Switch']
    

def wsls_autocorrelation(monkey_df, data_weighted, data_swapping,axis,strategic = True, order = 5):
    '''
    Calculate the autocorrelation of the win stay and lose switch variables for the strategic monkeys
    and the RNN and RLRNN data
    '''
    if strategic == True:
        # strategics = ['E']
        strategics = ['E','D','I','K']

    else:
        strategics = monkey_df['animal'].unique()
    
    # create and stack the data
    # need to use separate arrays for each because stacking them might create 
    # spurious correlations, so we need to also save lengths so we can use them 
    # compute weighted mean and error
    
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        They weights are in effect first normalized so that they 
        sum to 1 (and so they must not all be 0).

        values, weights -- NumPy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights,axis=0)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights, axis=0)
        return (average, np.sqrt(variance))

    
    ws_monkey = []
    ls_monkey = []
    ws_s = [] # ws and ls for weighted and swapping models
    ls_s = []
    ws_w = []
    ls_w = []
    
    _, w_action, w_reward, _,_, _= data_weighted
    _, s_action, s_reward, _, _,_ = data_swapping
    
    for monkey in strategics:
        monkey_data = monkey_df[monkey_df['animal'] == monkey]        
        monkey_data.sort_values(by=['task','id','trial'])
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        # regressors = parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 1],2,err=True) + \
        #     parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 2],2, err=True)
        # all_actions = create_order_data(monkey_data[monkey_data['task']==1],2,err=True) + \
        #     create_order_data(monkey_data[monkey_data['task']==2],2,err=True)
        for episode in range(len(episode_actions)):
            actions = episode_actions[episode]
            rewards = episode_rewards[episode]
            ws_monkey.append([((actions[i] == actions[i-1]) and rewards[i-1]) for i in range(1,len(actions))])
            ls_monkey.append([((actions[i] != actions[i-1]) and 1-rewards[i-1]) for i in range(1,len(actions))])

    monkey_err = []
    monkey_ws_ac = []
    monkey_ls_ac = []
    weights = []
    for i in range(len(ws_monkey)):
        ws = ws_monkey[i]
        ls = ls_monkey[i]
        ws_ac = pacf(ws,order)
        ls_ac = pacf(ls,order)
        monkey_ws_ac.append(ws_ac)
        monkey_ls_ac.append(ls_ac)
        weights.append(len(ws))
    
    monkey_ws_avg, monkey_ws_err = weighted_avg_and_std(monkey_ws_ac,weights,)
    monkey_ls_avg, monkey_ls_err = weighted_avg_and_std(monkey_ls_ac,weights)
    
    
    axis.plot(np.arange(1,order+1),monkey_ws_avg,label = 'Win Stay', color='xkcd:british racing green')
    axis.plot(np.arange(1,order+1),monkey_ls_avg,label = 'Lose Switch', color='xkcd:deep red')
    # axis.fill_between(np.arange(1,order+1),monkey_ws_avg - monkey_ws_err,monkey_ws_avg + monkey_ws_err,alpha=.3,color='xkcd:british racing green')
    # axis.fill_between(np.arange(1,order+1),monkey_ls_avg - monkey_ls_err,monkey_ls_avg + monkey_ls_err,alpha=.3,color='xkcd:deep red')
    
    for i in range(len(w_action)):
        ws_s.append([((s_action[i][j] == s_action[i][j-1]) and s_reward[i][j-1]) for j in range(1,len(s_action[i]))])
        ls_s.append([((s_action[i][j] != s_action[i][j-1]) and 1-s_reward[i][j-1]) for j in range(1,len(s_action[i]))])
        ws_w.append([((w_action[i][j] == w_action[i][j-1]) and w_reward[i][j-1]) for j in range(1,len(w_action[i]))])
        ls_w.append([((w_action[i][j] != w_action[i][j-1]) and 1-w_reward[i][j-1]) for j in range(1,len(w_action[i]))])
    
    switching_ws = []
    switching_ls = []
    weighted_ws = []
    weighted_ls = []
    for i in range(len(ws_s)):
        ws = ws_s[i]
        ls = ls_s[i]
        ws_ac = pacf(ws,order)
        ls_ac = pacf(ls,order)
        
        switching_ws.append(ws_ac)
        switching_ls.append(ls_ac)
        
        ws = ws_w[i]
        ls = ls_w[i]
        ws_ac = pacf(ws,order)
        ls_ac = pacf(ls,order)
        
        weighted_ws.append(ws_ac)
        weighted_ls.append(ls_ac)

    axis.plot(np.arange(1,order+1),np.mean(switching_ws,axis=0),label = 'Win Stay', color='xkcd:darkish green')
    axis.plot(np.arange(1,order+1),np.mean(switching_ls,axis=0),label = 'Lose Switch', color='xkcd:cherry red')

    axis.plot(np.arange(1,order+1),np.mean(weighted_ws,axis=0),label = 'Win Stay', color='xkcd:darkish green', linestyle = '--')
    axis.plot(np.arange(1,order+1),np.mean(weighted_ls,axis=0),label = 'Lose Switch', color='xkcd:cherry red', linestyle = '--')
    axis.fill_between(np.arange(1,order+1),np.mean(switching_ws,axis=0) - np.std(switching_ws,axis=0),np.mean(switching_ws,axis=0) + np.std(switching_ws,axis=0),alpha=.3,color='xkcd:darkish green')
    axis.fill_between(np.arange(1,order+1),np.mean(switching_ls,axis=0) - np.std(switching_ls,axis=0),np.mean(switching_ls,axis=0) + np.std(switching_ls,axis=0),alpha=.3,color='xkcd:cherry red')

    
    axis.fill_between(np.arange(1,order+1),np.mean(weighted_ws,axis=0) - np.std(weighted_ws,axis=0),np.mean(weighted_ws,axis=0) + np.std(weighted_ws,axis=0),alpha=.3,color='xkcd:darkish green')
    axis.fill_between(np.arange(1,order+1),np.mean(weighted_ls,axis=0) - np.std(weighted_ls,axis=0),np.mean(weighted_ls,axis=0) + np.std(weighted_ls,axis=0),alpha=.3,color='xkcd:cherry red')
    
    axis.set_title('Win Stay and Lose Switch Autocorrelation')
    axis.set_xlabel('Lag')
    axis.set_ylabel('PACF')
    
    axis.legend(['Monkey WS','Monkey LS','Strategic WS','Strategic LS','Weighted WS','Weighted LS'])
    
def wsls_crosscorrelation(monkey_df, data_weighted, data_swapping,axis,strategic = True, order = 5):
    '''
    Calculate the autocorrelation of the win stay and lose switch variables for the strategic monkeys
    and the RNN and RLRNN data
    '''
    if strategic == True:
        # strategics = ['E']
        strategics = ['E','D','I','K']

    else:
        strategics = monkey_df['animal'].unique()
    
    # create and stack the data
    # need to use separate arrays for each because stacking them might create 
    # spurious correlations, so we need to also save lengths so we can use them 
    # compute weighted mean and error
    
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        They weights are in effect first normalized so that they 
        sum to 1 (and so they must not all be 0).

        values, weights -- NumPy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights,axis=0)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights, axis=0)
        return (average, np.sqrt(variance))

    
    ws_monkey = []
    ls_monkey = []
    ws_s = [] # ws and ls for weighted and swapping models
    ls_s = []
    ws_w = []
    ls_w = []
    
    _, w_action, w_reward, _,_, _= data_weighted
    _, s_action, s_reward, _, _,_ = data_swapping
    
    for monkey in strategics:
        monkey_data = monkey_df[monkey_df['animal'] == monkey]        
        monkey_data.sort_values(by=['task','id','trial'])
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        # regressors = parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 1],2,err=True) + \
        #     parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 2],2, err=True)
        # all_actions = create_order_data(monkey_data[monkey_data['task']==1],2,err=True) + \
        #     create_order_data(monkey_data[monkey_data['task']==2],2,err=True)
        for episode in range(len(episode_actions)):
            actions = episode_actions[episode]
            rewards = episode_rewards[episode]
            ws_monkey.append([((actions[i] == actions[i-1]) and rewards[i-1])*2-1 for i in range(1,len(actions))])
            ls_monkey.append([((actions[i] != actions[i-1]) and 1-rewards[i-1])*2-1 for i in range(1,len(actions))])

    monkey_err = []
    monkey_corrs = []
    weights = []
    for i in range(len(ws_monkey)):
        ws = ws_monkey[i]
        ls = ls_monkey[i]
        monkey_corr = scipy.signal.correlate(ws,ls,mode='same')
        monkey_corr = monkey_corr/len(ws)
        mc = np.argmax(monkey_corr)
        monkey_corr = monkey_corr[mc:mc+order]
        monkey_corrs.append(monkey_corr)
        weights.append(len(ws))
    
    monkey_corr_avg, monkey_corr_err = weighted_avg_and_std(monkey_corrs,weights)
    
    
    axis.plot(np.arange(1,order+1),monkey_corr_avg,label = 'Monkeys', color='black')
    # axis.fill_between(np.arange(1,order+1),monkey_ws_avg - monkey_ws_err,monkey_ws_avg + monkey_ws_err,alpha=.3,color='xkcd:british racing green')
    # axis.fill_between(np.arange(1,order+1),monkey_ls_avg - monkey_ls_err,monkey_ls_avg + monkey_ls_err,alpha=.3,color='xkcd:deep red')
    
    for i in range(len(w_action)):
        ws_s.append([((s_action[i][j] == s_action[i][j-1]) and s_reward[i][j-1])*2-1 for j in range(1,len(s_action[i]))])
        ls_s.append([((s_action[i][j] != s_action[i][j-1]) and 1-s_reward[i][j-1])*2-1 for j in range(1,len(s_action[i]))])
        ws_w.append([((w_action[i][j] == w_action[i][j-1]) and w_reward[i][j-1])*2-1 for j in range(1,len(w_action[i]))])
        ls_w.append([((w_action[i][j] != w_action[i][j-1]) and 1-w_reward[i][j-1])*2-1 for j in range(1,len(w_action[i]))])
    
    switching_corrs = []
    weighted_corrs = []
    for i in range(len(ws_s)):
        ws = ws_s[i]
        ls = ls_s[i]

        switching_corr = scipy.signal.correlate(ws,ls,mode='same')
        switching_corr = switching_corr/len(ws)
        sc = np.argmax(switching_corr)
        switching_corr = switching_corr[sc:sc+order]
        
        switching_corrs.append(switching_corr)
        
        ws = ws_w[i]
        ls = ls_w[i]
        weighted_corr = scipy.signal.correlate(ws,ls,mode='same')
        weighted_corr = weighted_corr/len(ws)   
        wc = np.argmax(weighted_corr)
        weighted_corr = weighted_corr[wc:wc+order]
        weighted_corrs.append(weighted_corr)

    axis.plot(np.arange(1,order+1),np.mean(switching_corrs,axis=0),label = 'Switching', color='xkcd:darkish green')
    axis.plot(np.arange(1,order+1),np.mean(weighted_corrs,axis=0),label = 'Weighted', color='xkcd:cherry red')

    # axis.plot(np.arange(1,order+1),np.mean(weighted_ws,axis=0),label = 'Win Stay', color='xkcd:darkish green', linestyle = '--')
    # axis.plot(np.arange(1,order+1),np.mean(weighted_ls,axis=0),label = 'Lose Switch', color='xkcd:cherry red', linestyle = '--')
    # axis.fill_between(np.arange(1,order+1),np.mean(switching_ws,axis=0) - np.std(switching_ws,axis=0),np.mean(switching_ws,axis=0) + np.std(switching_ws,axis=0),alpha=.3,color='xkcd:darkish green')
    # axis.fill_between(np.arange(1,order+1),np.mean(switching_ls,axis=0) - np.std(switching_ls,axis=0),np.mean(switching_ls,axis=0) + np.std(switching_ls,axis=0),alpha=.3,color='xkcd:cherry red')

    
    # axis.fill_between(np.arange(1,order+1),np.mean(weighted_ws,axis=0) - np.std(weighted_ws,axis=0),np.mean(weighted_ws,axis=0) + np.std(weighted_ws,axis=0),alpha=.3,color='xkcd:darkish green')
    # axis.fill_between(np.arange(1,order+1),np.mean(weighted_ls,axis=0) - np.std(weighted_ls,axis=0),np.mean(weighted_ls,axis=0) + np.std(weighted_ls,axis=0),alpha=.3,color='xkcd:cherry red')
    
    axis.set_title('Win Stay and Lose Switch Crosscorrelation')
    axis.set_xlabel('Lag')
    axis.set_ylabel('Corr')
    
    axis.legend(['Monkeys','Switching','Weighted'])
    
    
    
    
def plot_logistic_combinatorial_violin(data, order = 2, ax = None, hist_ax = None, **plot_kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    monkeys = data['animal'].unique()
    colors = []

    
    monkey_list_dict = []
    monkey_ratio_dict = []
    for monkey in monkeys:    
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center  = process_data(monkey_data)
        monkey_data.sort_values(by=['task','id','trial'])
        # regressors = parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 1],order,err=True) + \
        #     parse_monkey_behavior_reduced(monkey_data[monkey_data['task'] == 2],order, err=True)
        
        
        regressors = parse_monkey_behavior_combinatorial(monkey_data[monkey_data['task'] == 2],order, err=True)
        all_actions = create_order_data(monkey_data[monkey_data['task']==2],order,err=True)
        
        
        # regressors = parse_monkey_behavior_combinatorial(monkey_data[monkey_data['task'] == 1],order,err=True) + \
        #     parse_monkey_behavior_combinatorial(monkey_data[monkey_data['task'] == 2],order, err=True)
        # all_actions = create_order_data(monkey_data[monkey_data['task']==1],order,err=True) + \
        #     create_order_data(monkey_data[monkey_data['task']==2],order,err=True)
        
        ls1 = 2*order       # index of lose switch first component
        ws1 = order         # index of win switch first component
        lratios = []
        wratios = []
        
        lr2 = []
        wr2 = []

        l_handles = []
        w_handles = []
        

        
        for index in range(len(regressors)):
            fit = logistic_regression(regressors[index],all_actions[index],bias=False).x
        
            lratios.append(fit[ls1+1] - fit[ls1])
            wratios.append(fit[ws1+1] - fit[ws1])
            
            lr2.append(fit[ls1+1] /fit[ls1])
            wr2.append(fit[ws1+1] /fit[ws1])
            
            

            dic = {'monkey':monkey, 'session':index, 'win stay':fit[ws1+1] - fit[ws1], 
                   'lose switch':fit[ls1+1] - fit[ls1], 'task': 1 if index < center else 2}
            monkey_list_dict.append(dic)
            monkey_ratio_dict.append({'monkey':monkey, 'win stay':fit[ws1+1] /fit[ws1], 
                   'lose switch':fit[ls1+1] /fit[ls1], 'task': 1 if index < center else 2})
            # lratios.append(fit[ls1]/max(fit[ls1:]))
            # wratios.append(fit[ws1]/max(fit[ws1:ls1]))
        
        session_nums = np.arange(1,1+len(wratios)) - center
        wh = ax.plot(session_nums,wratios, linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        lh = ax.plot(session_nums,lratios,c = colors[-1], **plot_kwargs)
        
        w_handles.append(wh)
        l_handles.append(lh)
        
    # ax3 = ax.twinx()
    
    monkeys_reorder = [13, 112, 18]
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}
    if hist_ax != None:
        # monkey_df = pd.DataFrame(monkey_list_dict)
        monkey_df = pd.DataFrame(monkey_ratio_dict)
        colors_ws = []
        colors_ls = []
        #win stay for task 1 for each monkey
        win_stay_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['win stay'] for monkey in monkeys_reorder]
        win_stay_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['win stay'] for monkey in monkeys_reorder]
        lose_switch_1 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 1)]['lose switch'] for monkey in monkeys_reorder]
        lose_switch_2 = [monkey_df[(monkey_df['monkey'] == monkey) & (monkey_df['task'] == 2)]['lose switch'] for monkey in monkeys_reorder]
        positions_ws = [0, 2, 4]
        positions_ls = [1, 3, 5]
        positions_labels = [.5, 2.5,4.5]
        labels = [monkeys_reorder_dict[monkey] for monkey in monkeys_reorder]
        hist_ax.set_xticks([])
        # hist_ax.set_xticks(positions_labels)
        # hist_ax.set_xticklabels(labels)
        vp1 = hist_ax.violinplot(win_stay_1, positions_ws, widths=0.5,side='low', showmeans=True, showextrema=False)
        # ax5 = hist_ax.twinx()
        vp2 = hist_ax.violinplot(win_stay_2, positions_ws, widths=0.5,side='high', showmeans=True, showextrema=False)
        vp3 = hist_ax.violinplot(lose_switch_1, positions_ls, widths=0.5,side='low', showmeans=True, showextrema=False)
        vp4 = hist_ax.violinplot(lose_switch_2, positions_ls, widths=0.5,side='high', showmeans=True, showextrema=False)
        hist_ax.set_xticks([])
        hist_ax.set_xticks(positions_labels)
        hist_ax.set_xticklabels(labels)
        hist_ax.set_yscale('symlog')
        for pc in vp1['bodies']:
            pc.set_facecolor('thistle')
            pc.set_edgecolor('black')
            
        for pc in vp2['bodies']:
            pc.set_facecolor('plum')
            pc.set_edgecolor('black')
        
        for pc in vp3['bodies']:
            pc.set_facecolor('goldenrod')
            pc.set_edgecolor('black')
        
        for pc in vp4['bodies']:
            pc.set_facecolor('darkgoldenrod')
            pc.set_edgecolor('black')


        # violinplot(data = pd.DataFrame(monkey_list_dict), 
        #                    x = 'task', y = ['win_stay','lose_switch'], ax = hist_ax, label='Win Stay', color = colors[0])
    # labs = ['Monkey C', 'Monkey E', 'Monkey F']
    # for c, color in enumerate(colors):
    #     ax3.plot(np.NaN, np.NaN, c = colors[c], label = labs[c])
    # ax3.get_yaxis().set_visible(False)
    # ax3.legend(loc = 'lower left')
    
    # hist_ax.legend(loc = 'upper left', handles=[vp1, vp2,vp3,vp4], labels = ['WS 1', 'WS 2', 'LS 1', 'LS 2'])
    
            
    ax.set_title('Monkey Behavioral Regression Stationarity')
    # ax.set_xlabel('Session')
    # ax.set_ylabel(r'$LS_t-LS_{t-1}$')
    ax.set_ylabel(r'$R(-1) - R(-2)$')    
    
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    
    center1 =-1 *  len(data[data['task'] == 1]['id'].unique()) /2 
    center2 = len(data[data['task'] == 2]['id'].unique()) / 2
    # ax.set_xticks([center1, center2])
    # ax.set_xticklabels(['Algorithm 1', 'Algorithm 2 '])
    ax.set_xticks([])
    if hist_ax != None:
        hist_ax.legend(loc='lower center',labels = ['WS 1', 'WS 2', 'LS 1', 'LS 2'],handles=[mpatches.Patch(color='thistle'),mpatches.Patch(color='plum'),
                        mpatches.Patch(color='goldenrod'),mpatches.Patch(color='darkgoldenrod')], ncol=4, frameon=False)
    
    labs = ['Monkey C', 'Monkey E', 'Monkey F']
    ax3 = ax.twinx()

    for c, color in enumerate(colors):
        ax3.plot(np.NaN, np.NaN, c = colors[c], label = labs[c])
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc = 'lower left', frameon=False)
    
    
    ax2 = ax.twinx()
    styles = ['--', '-']
    labels = ['Win Stay', 'Lose Switch']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    line1 = Line2D([0], [0], label='Win Stay', color='k', linestyle='--')
    line2 = Line2D([0], [0], label='Lose Switch', color='k', linestyle='-')
    
    ax2.legend(handles= [line1,line2],loc = 'lower right', labels = ['Win Stay', 'Lose Switch'], frameon = False)
    color_legends = ['C','E','F']
    style_legends = ['Win Stay', 'Lose Switch']
    
    
    
def logistic_prediction_accuracy_combinatorial(data,order =5,fit_all =False, mp_only = False,bias = False, mask = None,ax = None, labels = True, **plot_kwargs):
    plot_flag = 0
    episode_actions, episode_rewards, center  = process_data(data)
    fit_accuracy = []
    fit_dict = {'1':[],'2':[]}
    # if fit_all: # fit all using same regression and evaluate performance
    #     fit = paper_logistic_regression(None,False,data=data,order=decay_order,err=False)
    
    for session in range(len(episode_actions)):
        actions = episode_actions[session]
        rewards = episode_rewards[session]
        perf = paper_logistic_accuracy_combinatorial(actions,rewards,order,bias,mask)
        fit_accuracy.append(perf)
        
        if session < center:
            fit_dict['1'].append(perf)
        else:
            fit_dict['2'].append(perf)

            
    # if ax is None:
    #     plot_flag = 1
    #     fig, ax = plt.subplots()
    if ax != None:
        if labels:
            ax.set_title('Logistic Regression Prediction Accuracy')
        ax.set_xlabel('Session')
        ax.set_ylabel('Prediction Accuracy')
        session_nums = np.arange(1,1+len(episode_actions)) - center + .5
        line_handle = ax.plot(session_nums,fit_accuracy, **plot_kwargs)
        if plot_flag:
            plt.show()
    else:
        line_handle = None
    return fit_dict, line_handle



def plot_predictiability_comparison(data, ax = None,hist_ax = None, RL = True, Log = True, combinatorial = False, order=5,**plot_kwargs):
    plot_flag = 0
    if ax is None:
        plot_flag = 1
        fig, ax = plt.subplots()
    # if not isinstance(data,list):
    #     data = [data]
    colors = []
    monkeys = data['animal'].unique()
    datas= []

    ax.set_prop_cycle(cycler('color', plt.cm.Dark2.colors))

    for monkey in monkeys:
        monkey_data = data[data['animal'] == monkey]
        
        data_log, _ = logistic_prediction_accuracy_combinatorial(monkey_data,order=order,ax = ax,labels = False,linestyle = 'dashed', **plot_kwargs)
        data_log2, _ = logistic_prediction_accuracy(monkey_data,ax = ax,order=order,labels = False,linestyle = 'dashed', **plot_kwargs)
        colors.append(ax.lines[-1].get_color())
        
        # pred1_RL.extend(data_rl['1'])
        # pred2_RL.extend(data_rl['2'])
        # pred1_log.extend(data_log['1'])
        # pred2_log.extend(data_log['2'])
        datas.append(np.array(data_log['1'])- np.array(data_log2['1']))
        # dic = [{'monkey':monkey, 'Log':data_log['2'][i] -  data_log2['2'][i], 'task':2} for i in range(len(data_log['2']))]
        # monkey_list_dict.extend(dic)
        ax.plot(datas[-1])
    ax.axvline(x = 0, color = 'black', linestyle = 'dotted',alpha=.4)
    center = 0

    title = r'$\Delta$ between Full Logistic Regression and WSLS Logistic Regression'
    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    
    prop_cycle = prop_cycle = plt.rcParams['axes.prop_cycle']
    cols = prop_cycle.by_key()['color']

        
    monkeys_reorder = [13, 112, 18]
    monkeys_reorder_dict = {13:'C',112:'F',18:'E'}

    ax2 = ax.twinx()
    styles = ['-', '--']
    labels = ['RL', 'LR']
    for ss, sty in enumerate(styles):
        ax2.plot(np.NaN, np.NaN, ls=styles[ss],
                label=labels[ss], c='black')
    ax2.get_yaxis().set_visible(False)
    if Log or RL:
        ax2.legend(loc = 'upper right',frameon=False)
    
    if plot_flag:
        plt.show()
    return ax


def compute_rl_to_lr_ratio_violin_data(data, strategic_monkeys=['E', 'D', 'I'], nonstrategic_monkeys=['C', 'H', 'F', 'K'], 
                                       model='asymmetric', order=5, const_beta=False, const_gamma=True, 
                                       punitive=False, decay=False, ftol=1e-8, alpha=None, bias=False, mask=None):
    """
    Compute the ratio of RL performance to logistic regression accuracy for each session,
    separated by strategic/nonstrategic monkeys and MP1/MP2 tasks.
    
    Returns data structured for violin plotting.
    """
    
    # Monkey classification
    strategic_data = {'MP1': [], 'MP2': []}
    nonstrategic_data = {'MP1': [], 'MP2': []}
    
    monkeys = data['animal'].unique()
    
    for monkey in monkeys:
        monkey_data = data[data['animal'] == monkey]
        episode_actions, episode_rewards, center = process_data(monkey_data)
        
        for session in range(len(episode_actions)):
            actions = episode_actions[session]
            rewards = episode_rewards[session]
            
            # Compute RL performance using test_performance_asymmetric
            try:
                rl_fit, rl_perf = single_session_fit(actions, rewards, model=model, 
                                                   const_beta=const_beta, const_gamma=const_gamma, 
                                                   punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
                
                # Compute logistic regression accuracy using paper_logistic_accuracy_strategic
                _, lr_perf = paper_logistic_accuracy_strategic(actions, rewards, order=order, bias=bias, mask=mask)
                
                # Compute ratio
                if lr_perf > 0:  # Avoid division by zero
                    ratio = rl_perf / lr_perf
                else:
                    continue  # Skip this session if LR performance is 0
                
                # Determine task (MP1 or MP2)
                task = 'MP1' if session < center else 'MP2'
                
                # Classify monkey as strategic or nonstrategic
                if monkey in strategic_monkeys:
                    strategic_data[task].append(ratio)
                elif monkey in nonstrategic_monkeys:
                    nonstrategic_data[task].append(ratio)
                    
            except Exception as e:
                print(f"Error processing monkey {monkey}, session {session}: {e}")
                continue
    
    return strategic_data, nonstrategic_data


def plot_rl_to_lr_ratio_violin(data, ax=None, strategic_monkeys=['E', 'D', 'I'], nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                               model='asymmetric', order=5, const_beta=False, const_gamma=True, 
                               punitive=False, decay=False, ftol=1e-8, alpha=None, bias=False, mask=None):
    """
    Create a violin plot comparing RL/LR performance ratios between strategic and nonstrategic monkeys
    for both MP1 and MP2 tasks.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the data
    strategic_data, nonstrategic_data = compute_rl_to_lr_ratio_violin_data(
        data, strategic_monkeys=strategic_monkeys, nonstrategic_monkeys=nonstrategic_monkeys,
        model=model, order=order, const_beta=const_beta, const_gamma=const_gamma,
        punitive=punitive, decay=decay, ftol=ftol, alpha=alpha, bias=bias, mask=mask
    )
    
    # Prepare data for violin plot
    plot_data = []
    labels = []
    colors = ['lightblue', 'lightcoral', 'skyblue', 'salmon']
    
    # Strategic MP1
    if strategic_data['MP1']:
        plot_data.append(strategic_data['MP1'])
        labels.append('Strategic\nMP1')
    
    # Strategic MP2  
    if strategic_data['MP2']:
        plot_data.append(strategic_data['MP2'])
        labels.append('Strategic\nMP2')
    
    # Nonstrategic MP1
    if nonstrategic_data['MP1']:
        plot_data.append(nonstrategic_data['MP1'])
        labels.append('Nonstrategic\nMP1')
    
    # Nonstrategic MP2
    if nonstrategic_data['MP2']:
        plot_data.append(nonstrategic_data['MP2'])
        labels.append('Nonstrategic\nMP2')
    
    if not plot_data:
        print("No data available for violin plot")
        return ax
    
    # Create violin plot
    positions = range(1, len(plot_data) + 1)
    violins = ax.violinplot(plot_data, positions, widths=0.7, showmeans=True, showextrema=False)
    
    # Color the violins
    for i, violin in enumerate(violins['bodies']):
        violin.set_facecolor(colors[i % len(colors)])
        violin.set_edgecolor('black')
        violin.set_alpha(0.7)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('RL Performance / LR Accuracy', fontsize=12)
    ax.set_title('RL to LR Performance Ratio:\nStrategic vs Nonstrategic Monkeys', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=1 (equal performance)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    
    # Print summary statistics
    print("\nSummary Statistics (RL/LR Ratio):")
    print("="*50)
    
    for i, (label, values) in enumerate(zip(labels, plot_data)):
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            n_sessions = len(values)
            print(f"{label}: Mean={mean_val:.3f} ± {std_val:.3f}, N={n_sessions}")
        else:
            print(f"{label}: No data available")
    
    return ax
