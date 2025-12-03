"""
Model Prediction Sequence Matching Analysis

This module provides functions to compare sequence matching between neural network models 
and monkey behavioral data. 

MULTITHREADING SUPPORT:
- All functions now support both parallel and sequential processing
- By default, functions use multiprocessing for speed (recommended)
- Sequential processing is available for debugging

USAGE EXAMPLES:

# Use parallel processing (default, recommended for speed):
results = compare_sequence_matching_parallel(model_params, env_params)
results = compare_all_models_parallel(model_params1, model_params2, env_params)

# Use sequential processing (for debugging or if only one core available):
results = compare_sequence_matching_sequential(model_params, env_params)
results = compare_all_models_sequential(model_params1, model_params2, env_params)

# Use the main functions with explicit control:
results = compare_sequence_matching(model_params, env_params, use_multiprocessing=True, num_workers=4)
results = compare_all_models(model_params1, model_params2, env_params, use_multiprocessing=False)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
import torch
import concurrent.futures
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from analysis_scripts.LLH_behavior_RL import multi_session_fit
from matplotlib.gridspec import GridSpec
from figure_scripts.monkey_E_learning import load_behavior
import pickle
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

nonstrategic_monkeys = ['C', 'H', 'F', 'K']  # Non-strategic monkeys
strategic_monkeys = ['E', 'D', 'I']  # Strategic monkeys

stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'

stitched_p_cluster = '/home/fmb35/project/BG-PFC-RNN/cluster_scripts/regraining/stitched_monkey_data.pkl'

monkeys = nonstrategic_monkeys + strategic_monkeys

# loads a model, loads the data, and then loops through model and data and predicts 
# does a session by session comparison of the model prediction and the data
# Takes the Q values and plots that versus the probabilities of 

def process_monkey(monkey_data, model = None,model_params=None, env_params=None):
    """Process data for a single monkey in parallel"""
    # Check if monkey_data is empty or doesn't have animal column
    if len(monkey_data) == 0 or 'animal' not in monkey_data.columns or len(monkey_data['animal']) == 0:
        print(f"Warning: Empty or invalid monkey data provided to process_monkey")
        return None, [], [], [], [], [], [], []
    
    monkey = monkey_data['animal'].iloc[0]  # Get the monkey identifier
    monkey_sequences = []
    model_sequences = []
    computer_sequences = []
    delta_Q_sequences = []
    session_accuracies = []
    trial_accuracies = []
    session_trial_counts = []  # Track number of trials per session
    
    # Load model within the process to avoid pickling issues
    if model == None:
        print(f"Loading model for monkey {monkey}...")
        model = load_model(env_params, model_params)
    else:
        print(f"Using provided model for monkey {monkey}...")
    # Put model in evaluation mode and disable gradients
    model.eval()
    
    with torch.no_grad():
        for session in sorted(monkey_data['id'].unique()): 
            session_data = monkey_data[monkey_data['id'] == session]
            monkey_sequences.append(session_data['monkey_choice'].values)
            computer_sequences.append(session_data['computer_choice'].values)
            
            # Track the number of trials in this session (excluding the first trial used for initialization)
            n_trials = len(session_data) - 1  # -1 because we start from trial 1
            session_trial_counts.append(n_trials)
            
            # initialize hidden state
            hidden_in = torch.zeros([model.num_rnns, 1, model.hidden_dim], dtype=torch.float) 
            
            model_session_sequence = []
            delta_q_values = []
            
            for trial in range(1, len(session_data)):
                last_monkey_trial = monkey_sequences[-1][trial-1]
                monkey_trial = monkey_sequences[-1][trial]
                computer_trial = computer_sequences[-1][trial]
                last_computer_trial = computer_sequences[-1][trial-1]
                
                last_computer_action = torch.nn.functional.one_hot(torch.tensor(last_computer_trial, dtype=torch.long), num_classes=model.action_dim).view(1,1,-1)
                last_monkey_action = torch.nn.functional.one_hot(torch.tensor(last_monkey_trial, dtype=torch.long), num_classes=model.action_dim).view(1,1,-1)
                
                # convert to float
                last_monkey_action = last_monkey_action.float()
                
                reward = (monkey_trial == last_computer_trial) # computer trial 
                last_reward = reward * last_monkey_action
                # manually step through the model
                new_action, log_prob, values, hidden_in = model.forward(last_computer_action, last_monkey_action, last_reward, hidden_in)
                
                # convert from log_prob to prob to delta Q
                prob = torch.exp(log_prob.detach())
                delta_Q = torch.log(-1*prob/(1-prob))
                model_session_sequence.append(new_action.numpy()[0][0][-1])
                delta_q_values.append(delta_Q.numpy()[0][0][-1])
            
            model_sequences.append(model_session_sequence)
            delta_Q_sequences.append(delta_q_values)
            
            # Calculate accuracy for this session
            # Compare sequences starting from index 1 since first action is initialization
            trial_accuracies.append(monkey_sequences[-1][1:] == model_sequences[-1])
            accuracy = np.mean(trial_accuracies[-1])
            session_accuracies.append(accuracy)
            
    
    # Clean up model to free memory
    if model_params != None:
        del model
    
    print(f'Processed monkey {monkey}. Performance: {np.mean(session_accuracies)}')
    
    return monkey, session_accuracies, session_trial_counts, monkey_sequences, model_sequences, computer_sequences, delta_Q_sequences, trial_accuracies


def compute_single_model_perf(model, stitched_p = None):
    '''This function is for computing the performance of a single currently loaded model across all monkeys.
    It is similar to compare_sequence_matching_sequential except instead of passing model params, we pass the model'''
    
    if isinstance(stitched_p,type(None)):      
        try:
            with open(stitched_p, 'rb') as f:
                monkey_dat = pickle.load(f)
            monkey_dat = monkey_dat[monkey_dat['task'] == 'mp']
            if len(monkey_dat) == 0:
                # throw exception
                raise Exception('we are on cluster stitched_p')
        except:
            with open(stitched_p_cluster, 'rb') as f:
                monkey_dat = pickle.load(f)
            monkey_dat = monkey_dat[monkey_dat['task'] == 'mp']

    else:
        monkey_dat = stitched_p
    monkey_result_dict = {}
    monkey_result_dict['strategic'] = {}
    monkey_result_dict['nonstrategic'] = {}
    
    name_str = os.path.basename(model.model_path)
    # if there is a (), we know it is hybrid model, else it is a normal RNN
    if '(' in name_str:
        param_str = name_str.split('(')[1].split(')')[0]
        id_str = name_str.split(')')[-1].split('_')[1:]
    else:
        param_str = 'RNN'
    id_str = name_str.split('_')[-1]
    
    s_ta = []
    ns_ta = []

    # Get monkeys directly from the provided data instead of hardcoded lists
    available_monkeys = monkey_dat['animal'].unique()
    print(f"Found monkeys in provided data: {available_monkeys}")

    for monkey in available_monkeys:
        monkey_result_dict[monkey] = {}
        mdat = monkey_dat[monkey_dat['animal'] == monkey]
        
        _, session_accuracies, session_trial_counts, monkey_sequences, model_sequences, \
            computer_sequences, delta_Q_sequences, trial_accuracies = process_monkey(mdat,model=model)
        
        monkey_result_dict[monkey][param_str] = {}
        monkey_result_dict[monkey][param_str][id_str] = trial_accuracies

        # Categorize results into strategic and nonstrategic groups
        if monkey in strategic_monkeys:
            s_ta.extend(trial_accuracies)
        elif monkey in nonstrategic_monkeys:
            ns_ta.extend(trial_accuracies)

    # now aggregate all the results for strategic and nonstrategic monkeys
    monkey_result_dict['strategic'][param_str] = {}
    monkey_result_dict['nonstrategic'][param_str] = {}
    monkey_result_dict['strategic'][param_str][id_str] = s_ta
    monkey_result_dict['nonstrategic'][param_str][id_str] = ns_ta
    
    return monkey_result_dict
    
def monkey_data_combiner(df, monkeys):
    # this will combine all the monkeys passed into a format that can be used for fitting multiesession
    actions = []
    rewards = []
    
    for monkey in monkeys:
        mdf = df[df['animal'] == monkey]
        for session in mdf['id'].unique():
            sdf = mdf[mdf['id'] == session]
            actions.append(sdf['monkey_choice'].to_numpy())
            rewards.append(sdf['reward'].to_numpy())
    return actions, rewards

def compute_RL_and_RNN_sequences(rlrnn_path, rnn_path,rl_model_type, use_multiprocessing = True):
    perf_dict = {}
    
    # loads the data for the monkeys
    monkey_dat = pickle.load(stitched_p)
    monkey_dat = monkey_dat[monkey_dat['task'] == 'mp']
    
    # fits the RNNs in parallel and RL in sequence

    if rlrnn_path == None:
        pass
    else:
        pass
    
    if rnn_path == None:
        pass
    else:
        pass
    
    # fit RL model to each monkey as well as each monkey group:
    RL_dict = {}
    RL_dict['strategic'] = {}
    RL_dict['nonstrategic'] = {}
        
    episode_actions_s, episode_rewards_s = monkey_data_combiner(monkey_dat, strategic_monkeys)
    episode_actions_ns, episode_rewards_ns = monkey_data_combiner(monkey_dat, nonstrategic_monkeys)
    
    fit_params_strategic, performance_strategic = multi_session_fit(actions=episode_actions_s,
            rewards=episode_rewards_s,
            model=model_type,
            punitive=False,      # Rewards are already 0/1
            decay=False,         # No decay
            ftol=1e-8,          # Tolerance for optimization
            const_beta=False,    # Allow beta to vary
            const_gamma=True,    # Fix gamma to 0
            disable_abs=False    # Apply abs to ensure positive parameters where appropriate
        )
    RL_dict['strategic']['params'] = fit_params_strategic
    RL_dict['strategic']['model_type'] = rl_model_type
    RL_dict['strategic']['perf'] = performance_strategic
    fit_params_nonstrategic, performance_nonstrategic = multi_session_fit(actions=episode_actions_ns,
            rewards=episode_rewards_ns,
            model=rl_model_type,
            punitive=False,      # Rewards are already 0/1
            decay=False,         # No decay
            ftol=1e-8,          # Tolerance for optimization
            const_beta=False,    # Allow beta to vary
            const_gamma=True,    # Fix gamma to 0
            disable_abs=False    # Apply abs to ensure positive parameters where appropriate
        )
    RL_dict['nonstrategic']['params'] = fit_params_nonstrategic
    RL_dict['nonstrategic']['model_type'] = rl_model_type
    RL_dict['nonstrategic']['perf'] = performance_nonstrategic
    for monkey in nonstrategic_monkeys + strategic_monkeys:
        # fit an RL model to each monkey
        RL_dict[monkey] = {}
        episode_actions, episode_rewards = monkey_data_combiner(monkey_dat, monkey) 
        fit_params, fit_perf = multi_session_fit(actions = episode_actions,
            rewards = episode_rewards,
            model=model_type,
            punitive=False,      # Rewards are already 0/1
            decay=False,         # No decay
            ftol=1e-8,          # Tolerance for optimization
            const_beta=False,    # Allow beta to vary
            const_gamma=True,    # Fix gamma to 0
            disable_abs=False    # Apply abs to ensure positive parameters where appropriate
        )
        
        RL_dict[monkey]['params'] = fit_params
        RL_dict[monkey]['model_type'] = rl_model_type
        RL_dict[monkey]['perf'] = fit_perf
        
    
    perf_dict['RL'] = RL_dict
    
    return perf_dict
    

def compare_results(RLRNN_path, RNN_path, RL_path):
    pass

def compare_sequence_matching(model_params, env_params, data_path=None, num_workers=None, plot=True, use_multiprocessing=True):
    # load data
    if data_path == None:
        data = load_data(stitched_p)
    else:
        data = load_data(data_path)
    
    # only MP2 data
    data = data[data['task'] == 'mp']
    
    # Set default number of workers if not specified
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Split data by monkey
    monkey_data_dict = {monkey: data[data['animal'] == monkey] for monkey in monkeys}
    
    all_monkey_accuracies = {}
    monkey_sequences = {}
    model_sequences = {}
    computer_sequences = {}
    delta_Q_sequences = {}
    
    if use_multiprocessing and num_workers > 1:
        print(f"Processing monkeys using {num_workers} parallel workers...")
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks for each monkey's data
            future_to_monkey = {
                executor.submit(process_monkey, monkey_data_dict[monkey], model_params, env_params): monkey 
                for monkey in monkeys
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_monkey):
                monkey = future_to_monkey[future]
                try:
                    monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences, trial_accuracies = future.result()
                    all_monkey_accuracies[monkey_id] = accuracies
                    monkey_sequences[monkey_id] = m_sequences
                    model_sequences[monkey_id] = mod_sequences
                    computer_sequences[monkey_id] = comp_sequences
                    delta_Q_sequences[monkey_id] = q_sequences
                    print(f"Processed monkey {monkey_id}")
                except Exception as exc:
                    print(f'Monkey {monkey} generated an exception: {exc}')
                    import traceback
                    traceback.print_exc()
    else:
        print("Processing monkeys sequentially...")
        # Process each monkey sequentially (fallback for debugging or single-core)
        for monkey in monkeys:
            print(f"Processing monkey {monkey}...")
            monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences, trial_accuracies = process_monkey(
                monkey_data_dict[monkey], model_params, env_params
            )
            all_monkey_accuracies[monkey_id] = accuracies
            monkey_sequences[monkey_id] = m_sequences
            model_sequences[monkey_id] = mod_sequences
            computer_sequences[monkey_id] = comp_sequences
            delta_Q_sequences[monkey_id] = q_sequences

    # Create figure with subplots only if plotting is enabled
    if plot:
        fig = plt.figure(figsize=(20, 15))
        # Update the GridSpec to have 7 columns (3 for strategic + 4 for non-strategic monkeys)
        gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1], wspace=0.3, hspace=0.4)
        
        # Plot strategic monkeys (first 3)
        for i, monkey in enumerate(strategic_monkeys):
            ax = plt.subplot(gs[0, i])
            if monkey in all_monkey_accuracies:
                ax.plot(all_monkey_accuracies[monkey], marker='o')
                mean_accuracy = np.mean(all_monkey_accuracies[monkey])
                ax.text(0.05, 0.95, f'Mean: {mean_accuracy:.3f}', transform=ax.transAxes, 
                        verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}, fontsize=16)
            else:
                ax.text(0.5, 0.5, f'No data\nfor {monkey}', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'{monkey}', fontsize=18)
            ax.set_xlabel('Session', fontsize=16)
            ax.set_ylabel('Accuracy', fontsize=16)
            ax.set_ylim(0, 1)
            ax.grid(True)
        
        # Plot non-strategic monkeys (last 4)
        for i, monkey in enumerate(nonstrategic_monkeys):
            ax = plt.subplot(gs[1, i])
            if monkey in all_monkey_accuracies:
                ax.plot(all_monkey_accuracies[monkey], marker='o')
                mean_accuracy = np.mean(all_monkey_accuracies[monkey])
                ax.text(0.05, 0.95, f'Mean: {mean_accuracy:.3f}', transform=ax.transAxes, 
                        verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}, fontsize=16)
            else:
                ax.text(0.5, 0.5, f'No data\nfor {monkey}', transform=ax.transAxes, 
                        ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'{monkey}', fontsize=18)
            ax.set_xlabel('Session', fontsize=16)
            ax.set_ylabel('Accuracy', fontsize=16)
            ax.set_ylim(0, 1)
            ax.grid(True)
        
        # Calculate group averages and errors
        strategic_means = [np.mean(all_monkey_accuracies[monkey]) for monkey in strategic_monkeys if monkey in all_monkey_accuracies]
        nonstrategic_means = [np.mean(all_monkey_accuracies[monkey]) for monkey in nonstrategic_monkeys if monkey in all_monkey_accuracies]
        
        strategic_avg = np.mean(strategic_means) if strategic_means else 0
        nonstrategic_avg = np.mean(nonstrategic_means) if nonstrategic_means else 0
        
        strategic_err = np.std(strategic_means) / np.sqrt(len(strategic_means)) if strategic_means else 0  # SEM
        nonstrategic_err = np.std(nonstrategic_means) / np.sqrt(len(nonstrategic_means)) if nonstrategic_means else 0  # SEM
        
        # Use the empty subplot for group comparison
        ax = plt.subplot(gs[0, 3])
        x = np.arange(2)
        width = 0.5
        
        # Create bar plot with error bars
        bars = ax.bar(x, [strategic_avg, nonstrategic_avg], width, 
                      yerr=[strategic_err, nonstrategic_err], 
                      capsize=10, color=['#1f77b4', '#ff7f0e'], 
                      error_kw={'elinewidth': 2, 'capthick': 2})
        
        # Add labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Strategic', 'Non-strategic'], fontsize=14)
        ax.set_ylabel('Mean Accuracy', fontsize=16)
        ax.set_title('Group Comparison', fontsize=18)
        
        # Add text with exact values
        ax.text(0, strategic_avg + strategic_err + 0.02, f'{strategic_avg:.3f}±{strategic_err:.3f}', 
                ha='center', va='bottom', fontsize=14)
        ax.text(1, nonstrategic_avg + nonstrategic_err + 0.02, f'{nonstrategic_avg:.3f}±{nonstrategic_err:.3f}', 
                ha='center', va='bottom', fontsize=14)
        
        ax.set_ylim(0.45, 0.65)  # Adjust as needed for your data
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    return monkey_sequences, model_sequences, computer_sequences, delta_Q_sequences

def compare_all_models(model_params1, model_params2, env_params, data_path=None, num_workers=None, use_multiprocessing=True):
    """
    Compare sequence matching and performance of two neural networks and a fitted asymmetric RL model.
    The RL model is fit to each group of monkeys' data individually using multi_session_fit.
    
    Args:
        model_params1: Parameters for the first neural network model
        model_params2: Parameters for the second neural network model
        env_params: Environment parameters
        data_path: Path to monkey data
        num_workers: Number of parallel workers
        use_multiprocessing: Whether to use multiprocessing for neural network models
    """
    # Load data
    if data_path == None:
        data = load_data(stitched_p)
    else:
        data = load_data(data_path)
    
    # Only MP2 data
    data = data[data['task'] == 'mp']
    
    # Set default number of workers if not specified
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Split data by monkey
    monkey_data_dict = {monkey: data[data['animal'] == monkey] for monkey in monkeys}
    
    # Dictionary to store results for each model (using the same keys as compare_sequence_matching)
    all_monkey_accuracies_1 = {}
    monkey_sequences_1 = {}
    model_sequences_1 = {}
    computer_sequences_1 = {}
    delta_Q_sequences_1 = {}
    
    all_monkey_accuracies_2 = {}
    monkey_sequences_2 = {}
    model_sequences_2 = {}
    computer_sequences_2 = {}
    delta_Q_sequences_2 = {}
    
    all_monkey_accuracies_rl = {}
    monkey_sequences_rl = {}
    model_sequences_rl = {}
    computer_sequences_rl = {}
    delta_Q_sequences_rl = {}
    
    # Model names for display
    model_names = {
        'Model 1': 'Hybrid Network',
        'Model 2': 'RNN',
        'RL Model': 'Asymmetric RL'
    }
    
    # Fit asymmetric RL models to each monkey's data and process results first
    # RL fitting is kept sequential as it's already optimized
    print(f"\nFitting and processing {model_names['RL Model']} models for each monkey...")
    
    for monkey in monkeys:
        print(f"Processing monkey {monkey}...")
        
        # Get monkey data
        monkey_data = monkey_data_dict[monkey]
        
        # Prepare data for multi_session_fit
        episode_actions = []
        episode_rewards = []
        
        # Group data by session ID
        for _, session in monkey_data.groupby('id'):
            # Extract monkey actions and rewards
            actions = session['monkey_choice'].values
            rewards = session['reward'].values
            
            # Add to episodes
            episode_actions.append(actions)
            episode_rewards.append(rewards)
        
        # Use multi_session_fit to fit an asymmetric RL model to all sessions
        fit_params, performance = multi_session_fit(
            actions=episode_actions,
            rewards=episode_rewards,
            punitive=False,      # Rewards are already 0/1
            decay=False,         # No decay
            const_beta=False,    # Allow beta to vary
            const_gamma=True     # Fix gamma to 0
        )
        
        # Extract parameters
        alpha_pos, alpha_neg, beta, gamma = fit_params
        print(f"Monkey {monkey} RL parameters: alpha_pos={alpha_pos:.3f}, alpha_neg={alpha_neg:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, performance={performance:.3f}")
        
        # Now run predictions session by session using the fitted model parameters
        accuracies = []
        rl_sequences = []
        monkey_seqs = []
        computer_seqs = []
        q_values_seqs = []
        
        # Evaluate model on each session
        for session_idx, session in enumerate(monkey_data.groupby('id')):
            _, session_data = session
            session_length = len(session_data)
            monkey_sequence = session_data['monkey_choice'].values
            computer_sequence = session_data['computer_choice'].values
            reward_sequence = session_data['reward'].values
            
            # Initialize Q-values for this session
            q_values = np.zeros((session_length, 2))  # [left, right] for each trial
            
            # Initialize predictions and performance tracking
            rl_predictions = np.zeros(session_length, dtype=int)
            
            # First trial is random
            rl_predictions[0] = np.random.choice([0, 1])
            
            # Simulate RL model for each trial using fitted parameters
            for t in range(1, session_length):
                # Make choice based on Q-values and beta
                p_right = 1 / (1 + np.exp(-beta * (q_values[t-1, 1] - q_values[t-1, 0])))
                choice = 1 if np.random.random() < p_right else 0
                rl_predictions[t] = choice
                
                # Update Q-values based on outcome from previous trial
                prev_reward = reward_sequence[t-1]
                prev_choice = monkey_sequence[t-1]
                
                # Apply gamma discount to all Q-values (though gamma is likely 0)
                q_values[t, 0] = q_values[t-1, 0] * (1 - gamma)
                q_values[t, 1] = q_values[t-1, 1] * (1 - gamma)
                
                # Update Q value for chosen action with asymmetric learning rates
                if prev_reward > 0:  # positive outcome
                    q_values[t, prev_choice] += alpha_pos * (prev_reward - q_values[t-1, prev_choice])
                else:  # negative outcome
                    q_values[t, prev_choice] += alpha_neg * (prev_reward - q_values[t-1, prev_choice])
            
            # Calculate accuracy for this session
            correct_trials = sum(rl_predictions == monkey_sequence)
            session_accuracy = correct_trials / session_length
            accuracies.append(session_accuracy)
            
            # Store sequences
            rl_sequences.append(rl_predictions)
            monkey_seqs.append(monkey_sequence)
            computer_seqs.append(computer_sequence)
            q_values_seqs.append(q_values[:, 1] - q_values[:, 0])  # Store delta-Q values
        
        # Store results for this monkey
        all_monkey_accuracies_rl[monkey] = accuracies
        monkey_sequences_rl[monkey] = monkey_seqs
        model_sequences_rl[monkey] = rl_sequences
        computer_sequences_rl[monkey] = computer_seqs
        delta_Q_sequences_rl[monkey] = q_values_seqs
    
    # Process neural network models with multiprocessing
    if use_multiprocessing and num_workers > 1:
        print(f"\nProcessing neural network models using {num_workers} parallel workers...")
        
        # Process first neural network model
        print(f"Processing {model_names['Model 1']}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_monkey = {
                executor.submit(process_monkey, monkey_data_dict[monkey], model_params1, env_params): monkey 
                for monkey in monkeys
            }
            
            for future in concurrent.futures.as_completed(future_to_monkey):
                monkey = future_to_monkey[future]
                try:
                    monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences = future.result()
                    all_monkey_accuracies_1[monkey_id] = accuracies
                    monkey_sequences_1[monkey_id] = m_sequences
                    model_sequences_1[monkey_id] = mod_sequences
                    computer_sequences_1[monkey_id] = comp_sequences
                    delta_Q_sequences_1[monkey_id] = q_sequences
                    print(f"Processed monkey {monkey_id} with Model 1")
                except Exception as exc:
                    print(f'Monkey {monkey} (Model 1) generated an exception: {exc}')
                    import traceback
                    traceback.print_exc()
        
        # Process second neural network model
        print(f"Processing {model_names['Model 2']}...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_monkey = {
                executor.submit(process_monkey, monkey_data_dict[monkey], model_params2, env_params): monkey 
                for monkey in monkeys
            }
            
            for future in concurrent.futures.as_completed(future_to_monkey):
                monkey = future_to_monkey[future]
                try:
                    monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences = future.result()
                    all_monkey_accuracies_2[monkey_id] = accuracies
                    monkey_sequences_2[monkey_id] = m_sequences
                    model_sequences_2[monkey_id] = mod_sequences
                    computer_sequences_2[monkey_id] = comp_sequences
                    delta_Q_sequences_2[monkey_id] = q_sequences
                    print(f"Processed monkey {monkey_id} with Model 2")
                except Exception as exc:
                    print(f'Monkey {monkey} (Model 2) generated an exception: {exc}')
                    import traceback
                    traceback.print_exc()
    else:
        print(f"\nProcessing neural network models sequentially...")
        
        # Process first neural network model
        print(f"Processing {model_names['Model 1']}...")
        for monkey in monkeys:
            print(f"Processing monkey {monkey} with Model 1...")
            monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences, trial_accuracies = process_monkey(
                monkey_data_dict[monkey], model_params1, env_params
            )
            all_monkey_accuracies_1[monkey_id] = accuracies
            monkey_sequences_1[monkey_id] = m_sequences
            model_sequences_1[monkey_id] = mod_sequences
            computer_sequences_1[monkey_id] = comp_sequences
            delta_Q_sequences_1[monkey_id] = q_sequences
        
        # Process second neural network model
        print(f"Processing {model_names['Model 2']}...")
        for monkey in monkeys:
            print(f"Processing monkey {monkey} with Model 2...")
            monkey_id, accuracies, trial_counts, m_sequences, mod_sequences, comp_sequences, q_sequences,trial_accuracies = process_monkey(
                monkey_data_dict[monkey], model_params2, env_params
            )
            all_monkey_accuracies_2[monkey_id] = accuracies
            monkey_sequences_2[monkey_id] = m_sequences
            model_sequences_2[monkey_id] = mod_sequences
            computer_sequences_2[monkey_id] = comp_sequences
            delta_Q_sequences_2[monkey_id] = q_sequences
    
    # Define colors for each model
    model_colors = {
        'Model 1': '#1f77b4',  # blue
        'Model 2': '#ff7f0e',  # orange
        'RL Model': '#2ca02c'  # green
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    # Update the GridSpec to have 7 columns (3 for strategic + 4 for non-strategic monkeys)
    gs = gridspec.GridSpec(2, 7, height_ratios=[3, 1])
    
    # Plot strategic monkeys
    for i, monkey in enumerate(strategic_monkeys):
        ax = plt.subplot(gs[0, i])
        
        # Plot performance for all models in the same subplot
        if monkey in all_monkey_accuracies_rl:
            ax.plot(all_monkey_accuracies_rl[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['RL Model'],
                   label=f"{model_names['RL Model']} (Mean: {np.mean(all_monkey_accuracies_rl[monkey]):.3f})")
        
        if monkey in all_monkey_accuracies_1:
            ax.plot(all_monkey_accuracies_1[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['Model 1'],
                   label=f"{model_names['Model 1']} (Mean: {np.mean(all_monkey_accuracies_1[monkey]):.3f})")
        
        if monkey in all_monkey_accuracies_2:
            ax.plot(all_monkey_accuracies_2[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['Model 2'],
                   label=f"{model_names['Model 2']} (Mean: {np.mean(all_monkey_accuracies_2[monkey]):.3f})")
        
        ax.set_title(f'Monkey {monkey} (Strategic)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=8, loc='lower right')
    
    # Plot non-strategic monkeys
    for i, monkey in enumerate(nonstrategic_monkeys):
        ax = plt.subplot(gs[0, i+len(strategic_monkeys)])
        
        # Plot performance for all models in the same subplot
        if monkey in all_monkey_accuracies_rl:
            ax.plot(all_monkey_accuracies_rl[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['RL Model'],
                   label=f"{model_names['RL Model']} (Mean: {np.mean(all_monkey_accuracies_rl[monkey]):.3f})")
        
        if monkey in all_monkey_accuracies_1:
            ax.plot(all_monkey_accuracies_1[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['Model 1'],
                   label=f"{model_names['Model 1']} (Mean: {np.mean(all_monkey_accuracies_1[monkey]):.3f})")
        
        if monkey in all_monkey_accuracies_2:
            ax.plot(all_monkey_accuracies_2[monkey], 
                   marker='o', 
                   alpha=0.7,
                   color=model_colors['Model 2'],
                   label=f"{model_names['Model 2']} (Mean: {np.mean(all_monkey_accuracies_2[monkey]):.3f})")
        
        ax.set_title(f'Monkey {monkey} (Non-strategic)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(fontsize=8, loc='lower right')
    
    # Calculate performance statistics for all models
    strategic_data = {}
    nonstrategic_data = {}
    all_data = {}
    
    # Get strategic data
    strategic_data['RL Model'] = []
    for monkey in strategic_monkeys:
        if monkey in all_monkey_accuracies_rl:
            strategic_data['RL Model'].extend(all_monkey_accuracies_rl[monkey])
    
    strategic_data['Model 1'] = []
    for monkey in strategic_monkeys:
        if monkey in all_monkey_accuracies_1:
            strategic_data['Model 1'].extend(all_monkey_accuracies_1[monkey])
    
    strategic_data['Model 2'] = []
    for monkey in strategic_monkeys:
        if monkey in all_monkey_accuracies_2:
            strategic_data['Model 2'].extend(all_monkey_accuracies_2[monkey])
    
    # Get non-strategic data
    nonstrategic_data['RL Model'] = []
    for monkey in nonstrategic_monkeys:
        if monkey in all_monkey_accuracies_rl:
            nonstrategic_data['RL Model'].extend(all_monkey_accuracies_rl[monkey])
    
    nonstrategic_data['Model 1'] = []
    for monkey in nonstrategic_monkeys:
        if monkey in all_monkey_accuracies_1:
            nonstrategic_data['Model 1'].extend(all_monkey_accuracies_1[monkey])
    
    nonstrategic_data['Model 2'] = []
    for monkey in nonstrategic_monkeys:
        if monkey in all_monkey_accuracies_2:
            nonstrategic_data['Model 2'].extend(all_monkey_accuracies_2[monkey])
    
    # Combine all data
    for model_key in ['RL Model', 'Model 1', 'Model 2']:
        all_data[model_key] = np.concatenate([strategic_data[model_key], nonstrategic_data[model_key]])
    
    # Create two comparison subplots in the bottom row
    # Subplot 1: Group comparison with violin plots
    ax_violin = plt.subplot(gs[1, :3])
    
    # Prepare data for violin plots
    violin_data = []
    violin_labels = []
    violin_colors = []
    
    for model_key in ['RL Model', 'Model 1', 'Model 2']:
        violin_data.append(strategic_data[model_key])
        violin_labels.append(f"{model_names[model_key]}\nStrategic")
        violin_colors.append(model_colors[model_key])
        
        violin_data.append(nonstrategic_data[model_key])
        violin_labels.append(f"{model_names[model_key]}\nNon-strategic")
        violin_colors.append(model_colors[model_key])
    
    # Create violin plot
    violin_parts = ax_violin.violinplot(violin_data, showmedians=True)
    
    # Customize violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(violin_colors[i//2])
        pc.set_alpha(0.7)
    
    # Set custom x-axis ticks and labels
    ax_violin.set_xticks(range(1, len(violin_labels) + 1))
    ax_violin.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=9)
    ax_violin.set_ylabel('Accuracy Distribution', fontsize=12)
    ax_violin.set_title('Model Performance by Monkey Type', fontsize=14)
    ax_violin.grid(True, axis='y')
    
    # Subplot 2: Bar chart for group averages
    ax_bar = plt.subplot(gs[1, 3:7])
    
    # Calculate averages for bar chart
    avg_data = {}
    for model_key in ['RL Model', 'Model 1', 'Model 2']:
        strategic_mean = np.mean(strategic_data[model_key]) if strategic_data[model_key] else 0
        nonstrategic_mean = np.mean(nonstrategic_data[model_key]) if nonstrategic_data[model_key] else 0
        strategic_err = np.std(strategic_data[model_key]) / np.sqrt(len(strategic_data[model_key])) if strategic_data[model_key] else 0
        nonstrategic_err = np.std(nonstrategic_data[model_key]) / np.sqrt(len(nonstrategic_data[model_key])) if nonstrategic_data[model_key] else 0
        
        avg_data[model_key] = {
            'strategic_avg': strategic_mean,
            'nonstrategic_avg': nonstrategic_mean,
            'strategic_err': strategic_err,
            'nonstrategic_err': nonstrategic_err
        }
    
    # Bar chart positions
    x = np.arange(2)
    width = 0.25
    
    # Plot bar chart for each model
    for i, (model_key, model_name) in enumerate(model_names.items()):
        bars = ax_bar.bar(x + (i-1)*width, 
                       [avg_data[model_key]['strategic_avg'], avg_data[model_key]['nonstrategic_avg']], 
                       width, 
                       yerr=[avg_data[model_key]['strategic_err'], avg_data[model_key]['nonstrategic_err']], 
                       capsize=5,
                       color=model_colors[model_key],
                       label=model_name)
    
    # Add labels and legend
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['Strategic', 'Non-strategic'], fontsize=12)
    ax_bar.set_ylabel('Mean Accuracy', fontsize=12)
    ax_bar.set_title('Average Performance Comparison', fontsize=14)
    ax_bar.legend(fontsize=10)
    ax_bar.set_ylim(0.45, 0.65)
    ax_bar.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Return results using the same dictionary structure as compare_sequence_matching
    return {
        'rl_model': {
            'monkey_sequences': monkey_sequences_rl,
            'model_sequences': model_sequences_rl,
            'computer_sequences': computer_sequences_rl,
            'delta_Q_sequences': delta_Q_sequences_rl,
            'accuracies': all_monkey_accuracies_rl
        },
        'model1': {
            'monkey_sequences': monkey_sequences_1,
            'model_sequences': model_sequences_1,
            'computer_sequences': computer_sequences_1,
            'delta_Q_sequences': delta_Q_sequences_1,
            'accuracies': all_monkey_accuracies_1
        },
        'model2': {
            'monkey_sequences': monkey_sequences_2,
            'model_sequences': model_sequences_2,
            'computer_sequences': computer_sequences_2,
            'delta_Q_sequences': delta_Q_sequences_2,
            'accuracies': all_monkey_accuracies_2
        }
    }

def load_model(env_params, model_params_original):
    """Load a model with the given parameters and environment"""
    environment = make_env(env_params)
    
    # Define keys accepted by RLRNN.__init__
    # Ensure this list is accurate based on the RLRNN class definition
    accepted_rlrnn_keys = [
        'hidden_dim', 'max_episodes', 'max_steps', 'batch_size', 'update_itr', 
        'DETERMINISTIC', 'environment', 'model_path', 'weighting', 'num_rnns', 
        'activation', 'RL', 'l1', 'l2', 'Qalpha', 'Qgamma', 'scaletype', 'gamma', 
        'lambd', 'LBFGS', 'lr', 'leaky_q', 'leaky_policy', 'leaky_tau', 
        'policy_scale', 'learn_RL', 'q_init', 'norm_p'
        # Add any other valid keys for RLRNN.__init__ here
    ]
    
    # Filter model_params for RLRNN constructor, creating a new dict for instantiation
    rlrnn_init_params = {k: v for k, v in model_params_original.items() if k in accepted_rlrnn_keys and k != 'environment'}
    
    # Instantiate the model
    model = RLRNN(environment=environment, **rlrnn_init_params)
    
    # Use the 'model_path' and 'model_name' from the original, unfiltered params for loading logic
    path_to_load_from = model_params_original.get('model_path') # This should be the folder containing model files
    name_for_model_files = model_params_original.get('model_name')

    if not path_to_load_from:
        print(f"Warning: 'model_path' not found in parameters for model '{name_for_model_files if name_for_model_files else 'Unknown'}'. Using initialized model without loading weights.")
        return model

    if not name_for_model_files:
        print(f"Warning: 'model_name' not found in parameters for path '{path_to_load_from}'. Attempting to load with model's default name: '{model.model_name}'.")
        # If name_for_model_files is not set, model.load_model will use model.model_name, which might be a default.
        # This could be problematic if the actual files are named differently.
        # However, discover_models should ensure model_name is set from the folder name if not in pkl.
    else:
        # This is the crucial step: set the model's internal name attribute.
        # model.load_model(path, exact=False) uses self.model_name to construct filenames.
        model.model_name = name_for_model_files 

    print(f"Attempting to load model files for '{model.model_name}' from directory: '{path_to_load_from}'")
    
    # model.load_model with exact=False expects 'path_to_load_from' to be the directory
    # and uses 'model.model_name' (which we just set) to find files like 'model_name_actor'.
    print(f"Trying: model.load_model(path='{path_to_load_from}', exact=False) using model.model_name='{model.model_name}'")
    model.load_model(path_to_load_from, exact=False)
    print(f"Successfully loaded model '{model.model_name}' from '{path_to_load_from}' (exact=False)")
    return model

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pd.read_pickle(f)
    return data

def compare_sequence_matching_parallel(model_params, env_params, data_path=None, num_workers=None, plot=True):
    """
    Convenience function for parallel processing (default behavior).
    """
    return compare_sequence_matching(model_params, env_params, data_path, num_workers, plot, use_multiprocessing=True)

def compare_sequence_matching_sequential(model_params, env_params, data_path=None, plot=True):
    """
    Convenience function for sequential processing (for debugging).
    """
    return compare_sequence_matching(model_params, env_params, data_path, num_workers=1, plot=plot, use_multiprocessing=False)

def compare_all_models_parallel(model_params1, model_params2, env_params, data_path=None, num_workers=None):
    """
    Convenience function for parallel processing of all models (default behavior).
    """
    return compare_all_models(model_params1, model_params2, env_params, data_path, num_workers, use_multiprocessing=True)

def compare_all_models_sequential(model_params1, model_params2, env_params, data_path=None):
    """
    Convenience function for sequential processing of all models (for debugging).
    """
    return compare_all_models(model_params1, model_params2, env_params, data_path, num_workers=1, use_multiprocessing=False)


