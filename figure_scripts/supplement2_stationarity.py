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
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec,  GridSpecFromSubplotSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior
from analysis_scripts.stationarity_and_randomness import compute_predictability,plot_predictiability_monkeys_violin, plot_RL_timescales_violin, plot_logistic_coefficients_violin
from analysis_scripts.LLH_behavior_RL import single_session_fit
import pickle

def plot_RL_beta_violin(data, rl_model='asymmetric', ax=None, hist_ax=None, **plot_kwargs):
    """Plot beta parameters from RL model fits for each monkey"""
    if ax is None:
        fig, ax = plt.subplots()
    
    monkeys = data['animal'].unique()
    monkey_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    all_betas_1 = []
    all_betas_2 = []
    
    for i, monkey in enumerate(monkeys):
        monkey_data = data[data['animal'] == monkey]
        
        # Separate algorithm 1 and 2 data
        algo1_data = monkey_data[monkey_data['task'] == 1]
        algo2_data = monkey_data[monkey_data['task'] == 2]
        
        monkey_name = monkey
        if monkey == 13:
            monkey_name = 'C'
        elif monkey == 112:
            monkey_name = 'F'
        elif monkey == 18:
            monkey_name = 'E'
        
        color = monkey_colors[i % len(monkey_colors)]
        
        # Fit RL model to each session and extract beta
        algo1_betas = []
        algo2_betas = []
        
        # Algorithm 1 sessions
        for session_id in algo1_data['id'].unique():
            session_data = algo1_data[algo1_data['id'] == session_id]
            if len(session_data) > 20:  # Minimum trials for fitting
                try:
                    actions = session_data['monkey_choice'].values
                    rewards = session_data['reward'].values
                    fit_params, _ = single_session_fit(
                        actions, rewards, 
                        model=rl_model, 
                        const_beta=False, 
                        const_gamma=True
                    )
                    if len(fit_params) >= 3:
                        beta = fit_params[2]  # Beta is third parameter
                        algo1_betas.append(beta)
                except:
                    continue
        
        # Algorithm 2 sessions
        for session_id in algo2_data['id'].unique():
            session_data = algo2_data[algo2_data['id'] == session_id]
            if len(session_data) > 20:  # Minimum trials for fitting
                try:
                    actions = session_data['monkey_choice'].values
                    rewards = session_data['reward'].values
                    fit_params, _ = single_session_fit(
                        actions, rewards, 
                        model=rl_model, 
                        const_beta=False, 
                        const_gamma=True
                    )
                    if len(fit_params) >= 3:
                        beta = fit_params[2]  # Beta is third parameter
                        algo2_betas.append(beta)
                except:
                    continue
        
        # Plot violins for this monkey
        if algo1_betas:
            positions_1 = np.full(len(algo1_betas), -1 + i*0.15)
            ax.violinplot([algo1_betas], positions=[positions_1[0]], widths=0.1, 
                         showmeans=True, showmedians=False)
            all_betas_1.extend(algo1_betas)
        
        if algo2_betas:
            positions_2 = np.full(len(algo2_betas), 1 + i*0.15)
            ax.violinplot([algo2_betas], positions=[positions_2[0]], widths=0.1, 
                         showmeans=True, showmedians=False)
            all_betas_2.extend(algo2_betas)
    
    # Format the plot
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Algorithm 1', 'Algorithm 2'])
    ax.set_ylabel('Beta (Temperature)', fontsize=12)
    ax.set_title(f'RL Model Beta Parameters ({rl_model})', fontsize=14)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # Add histogram if requested
    if hist_ax is not None:
        hist_ax.hist(all_betas_1, bins=20, alpha=0.7, label='Algorithm 1', density=True)
        hist_ax.hist(all_betas_2, bins=20, alpha=0.7, label='Algorithm 2', density=True)
        hist_ax.set_xlabel('Beta')
        hist_ax.set_ylabel('Density')
        hist_ax.legend()
        hist_ax.set_title('Beta Distribution')
     
def stationarity_supplement(mpbeh_path, num_cutoff=5,num_sessions=5, rl_model = 'simple', window=1):
    fig = plt.figure(layout='constrained', figsize=(14, 12), dpi = 300)  # Much larger size
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)  # Added spacing parameters

    all_data = load_behavior(mpbeh_path,algorithm=None, monkey = None)
    
    # Create subplot layout with better spacing:
    # Row 0: Predictability plot (spans 2 cols) + histogram (1 col)
    # Row 1: Timescale plot (spans 2 cols) + histogram (1 col)  
    # Row 2: Beta plot (spans 2 cols) + histogram (1 col)
    
    predictability_ax = fig.add_subplot(gs[0,:-1])
    predictability_hist_ax = fig.add_subplot(gs[0,-1])
    
    timescale_ax = fig.add_subplot(gs[1,:-1])
    timescale_hist_ax = fig.add_subplot(gs[1,-1])
    
    beta_ax = fig.add_subplot(gs[2,:-1])
    beta_hist_ax = fig.add_subplot(gs[2,-1])

    # Generate existing plots
    plot_RL_timescales_violin(all_data, ax = timescale_ax, rl_model = rl_model, window=window, hist_ax = timescale_hist_ax)
    
    timescale_hist_ax.set_title('RL Timescales')
    timescale_hist_ax.set_xlabel(r'Timescale^{-1}')
    
    plot_predictiability_monkeys_violin(all_data, ax = predictability_ax, hist_ax=predictability_hist_ax, RL=False, combinatorial=True)
    predictability_hist_ax.set_title('Logistic Regression Predictability')
    predictability_hist_ax.set_xlabel('Accuracy')
    
    # Add new beta plot
    plot_RL_beta_violin(all_data, rl_model='asymmetric', ax=beta_ax, hist_ax=beta_hist_ax)
    
    fig.suptitle('Supplement 2: Monkey Behavioral Variation', fontsize=20, y=0.95)  # Lower title position
    
    return fig