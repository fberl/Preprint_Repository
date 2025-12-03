import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression
from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec

# general idea for plot: logistic regression for RL only. logistic regression for RNN only. 
# Logistic regression for RLRNN with RNN and RL modules inset smaller. to the right. 
# length of image roughly 3.5x size of a single plot.




def generate_comparison_figure(RLRNN_path, RNN_path, env_params,RL_params, 
                               algorithm = 2, monkey = 'E', late_only = True, nits = 2):

    env = make_env(env_params)
    
    
    # RL_model = RL(alpha = .2, gamma = [.3,-.15], env = env, asymmetric= True, load=False)
    RL_model = RL(**RL_params)    
    RLRNN_model =  load_model(RLRNN_path)
    RNN_model = load_model(RNN_path)
    
    RL_data, masks = RL_model.generate_data(nits)
    RLRNN_data = generate_data(RLRNN_model,env, nits = nits)
    RNN_data = generate_data(RNN_model, env, nits = nits)
    # format the output of RLRNN_data s.t. we have inputs for RNN_only and RL_only
    rl_module_data = (RLRNN_data[0], RLRNN_data[-1],RLRNN_data[2],None, None, None)
    rnn_module_data = (RLRNN_data[0], RLRNN_data[-2],RLRNN_data[2],None, None, None)
    
    RL_WR = np.mean(RL_data[2])
    RNN_WR = np.mean(RNN_data[2])
    RLRNN_WR = np.mean(RLRNN_data[2])
    
    #create figure
    fig = plt.figure(layout='constrained')
    gs = GridSpec(2,7, figure=fig) 
    RL_ax = fig.add_subplot(gs[:,0:2])
    RNN_ax = fig.add_subplot(gs[:,2:4])
    RLRNN_ax = fig.add_subplot(gs[:,4:6])
    rnn_module_ax = fig.add_subplot(gs[1,6])
    rl_module_ax = fig.add_subplot(gs[0,6])
    
    axs = [RL_ax, RNN_ax, RLRNN_ax, rnn_module_ax, rl_module_ax]

    
    paper_logistic_regression(axs[0],False,data=RL_data, legend = True)
    paper_logistic_regression(axs[1],True, data=RNN_data)
    paper_logistic_regression(axs[2],True, data=RLRNN_data)
    paper_logistic_regression(axs[3],True, data=rl_module_data)
    paper_logistic_regression(axs[4],True, data=rnn_module_data)

    # labels and titles
    fig.suptitle('Behavioral Analysis for Models vs. Matching Pennies')
    titles = ['RL Model\n WR : {:.2f}'.format(RL_WR), 'RNN Model\n WR : {:.2f}'.format(RNN_WR), 'RNN + RL Model\n WR : {:.2f}'.format(RLRNN_WR),
              'RL module', 'RNN module']

    for i in len(axs):
        axs[i].set_title(titles[i])
    
    axs[0].set_ylabel('Logistic Regression Coefficient')
    fig.text(0.5, 0.04, 'Trials Back', ha='center')
    
    plt.show()
    