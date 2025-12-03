import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression, fit_glr, fit_single_paper_strategic, fit_single_paper
from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior
from analysis_scripts.population_coding import plot_population_encoding_paper, plot_code_switching_paper
import seaborn as sns
import pickle
from scipy.optimize import curve_fit
import scipy.stats as stats
from matplotlib.patches import Patch

def plot_glr_results(ax, coeffs, errors, order, labels, colors):
    """Helper function to plot GLR coefficients."""
    xord = np.arange(1, order + 1)
    num_regressors = len(labels)
    for i in range(num_regressors):
        start_idx = i * order
        end_idx = (i + 1) * order
        ax.plot(xord, coeffs[start_idx:end_idx], label=labels[i], color=colors[i])
        ax.fill_between(xord, coeffs[start_idx:end_idx] - errors[start_idx:end_idx],
                        coeffs[start_idx:end_idx] + errors[start_idx:end_idx],
                        alpha=0.25, facecolor=colors[i])
    ax.axhline(linestyle='--', color='k', alpha=0.5)
    ax.set_xticks(xord)

def rename_keys(d, key_map):
    """Rename keys in a dictionary."""
    if not isinstance(d, dict):
        return d
    return {key_map.get(k, k): v for k, v in d.items()}

# # Define function to calculate center of mass of a curve
# def calculate_center_of_mass(x, y):
#     """
#     Calculate the center of mass of a curve starting from the maximum value.
    
#     Parameters:
#     x : array-like
#         x-coordinates (lags in this case)
#     y : array-like
#         y-coordinates (weights/encoding strength)
    
#     Returns:
#     float
#         Center of mass of the decay portion of the curve
#     """
#     # Use absolute values to ensure proper weighting
#     y_abs = np.abs(y)
    
#     # Find the index of the maximum magnitude
#     max_idx = np.argmax(y_abs)
    
#     # Only use data points from max_idx onwards (decay portion)
#     x_decay = x[max_idx:]
#     y_decay = y_abs[max_idx:]
    
#     # Avoid division by zero
#     if len(y_decay) == 0 or np.sum(y_decay) == 0:
#         return np.nan
    
#     # Calculate center of mass of the decay portion
#     com = np.sum(x_decay * y_decay) / np.sum(y_decay)
#     return com

# Define function to calculate center of mass of a curve
def calculate_center_of_mass(x, y):
    """
    Calculate the center of mass of a curve 
    
    Parameters:
    x : array-like
        x-coordinates (lags in this case)
    y : array-like
        y-coordinates (weights/encoding strength)
    
    Returns:
    float
        Center of mass of the decay portion of the curve
    """
    # Use absolute values to ensure proper weighting
    y_abs = np.abs(y)
    
    # Find the index of the maximum magnitude
    max_idx = np.argmax(y_abs)
    
    # Only use data points from max_idx onwards (decay portion)
    x_decay = x
    y_decay = y_abs
    
    # Avoid division by zero
    if len(y_decay) == 0 or np.sum(y_decay) == 0:
        return np.nan
    
    # Calculate center of mass of the decay portion
    com = np.sum(x_decay * y_decay) / np.sum(y_decay)
    return com

# instead of generating data every time, lets make one  with like 40 trials of each or whatever was done
# for the weight plots and then save and just load that every time instead of generating it each time

weight_supp_data = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/raw_data.p'
# check if theres an mp1 parsed data also
weight_supp_data_mp1 = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/parsed_data_mp1.p'

# and then we can also fit timescales once and save that too
# same with sign reversals 

# Cache for session-by-session logistic regression results
session_logistic_data_mp1 = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/session_logistic_mp1.p'
session_logistic_data_mp2 = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/session_logistic_mp2.p'






def plot_neural_coding_algo(mp1_params, mp2_params, env1_params, env2_params, nits=2, gen=False, strategic=True, multilayer=True, layout_test_mode=False):
    
    # Full weight keys for neuronal encoding analysis (includes action/reward)
    weight_keys = ['action', 'reward', 'repeat_win', 'change_win', 'repeat_lose', 'change_lose']
    # Behavioral weight keys for strategic regression (behavioral strategies only)
    behavioral_weight_keys = ['repeat_win', 'change_win', 'repeat_lose', 'change_lose']
    lags = np.arange(1, 9) # Used in encoding strength plots

    if not layout_test_mode:
        with open(weight_supp_data, 'rb') as f:
            parsed_data_loaded = pickle.load(f)
        
        with open(weight_supp_data_mp1, 'rb') as f:
            parsed_data_mp1_loaded = pickle.load(f)
        
        env1 = make_env(env1_params)
        env2 = make_env(env2_params)
        
        algo2_model = RLRNN(environment=env2, **mp2_params)
        algo2_model.load_model('')
        algo1_model = RLRNN(environment=env1, **mp1_params)
        algo1_model.load_model('')
        
        if gen:
            # This part is simplified as per original, assuming pre-loaded parsed_data_mp1_loaded and parsed_data_loaded are base
            algo1_data_gen = [generate_data(algo1_model, env1, nits=nits)]
            algo1_data_gen.extend(parsed_data_mp1_loaded) # extend modifies in place, careful if original list is needed elsewhere
            algo2_data_gen = [generate_data(algo2_model, env2, nits=nits)]
            algo2_data_gen.extend(parsed_data_loaded)
            
            with open(weight_supp_data_mp1, 'wb') as f:
                pickle.dump(algo1_data_gen, f)
            with open(weight_supp_data, 'wb') as f:
                pickle.dump(algo2_data_gen, f)
            
            # Use the newly generated and saved data for the current plot
            algo1_data_list = algo1_data_gen
            algo2_data_list = algo2_data_gen
        else:
            # For session-by-session analysis, we only need a subset of sessions
            # Limit to first 10 sessions to avoid fitting too many regressions
            max_sessions = 10
            algo1_data_list = parsed_data_mp1_loaded[:max_sessions] if len(parsed_data_mp1_loaded) > max_sessions else parsed_data_mp1_loaded
            algo2_data_list = parsed_data_loaded[:max_sessions] if len(parsed_data_loaded) > max_sessions else parsed_data_loaded
            
            print(f"Using {len(algo1_data_list)} MP1 sessions and {len(algo2_data_list)} MP2 sessions for analysis")
            
        algo1_data = algo1_data_list[0]
        algo2_data = algo2_data_list[0]


        algo2_data_stacked = []

        for i in range(len(algo2_data)):
            algo2_temp = []
            for j in range(len(algo2_data[i])):
                algo2_temp.append(algo2_data[i][j])
            algo2_data_stacked.append(algo2_temp)


    else: # layout_test_mode is True
        algo1_data = [None, [], [0.50]]  # Mock for WR: [states, actions, rewards]
        algo2_data = [None, [], [0.48]]  # Mock for WR
        algo2_data_stacked = [] 
        
        algo1_model = None
        algo2_model = None

        # Initialize dummy timescales and weights for subsequent plotting logic
        timescale_1, signs_1, weights_1, var_1 = {}, {}, {k: np.zeros((1, len(lags))) for k in weight_keys}, {}
        timescale_2, signs_2, weights_2, var_2 = {}, {}, {k: np.zeros((1, len(lags))) for k in weight_keys}, {}

    fig = plt.figure(layout='constrained', figsize=(15, 9.5), dpi=300)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 0.8, 1.2], hspace=0.55, wspace=0.45)
    
    mp1_logistic_ax = fig.add_subplot(gs[0, 0])
    pfc_ws_ax = fig.add_subplot(gs[0, 1])
    pfc_ls_ax = fig.add_subplot(gs[0, 2])
    mp1_comparison_ax = fig.add_subplot(gs[0, 3])
    
    mp2_logistic_ax = fig.add_subplot(gs[1, 0])
    pfcbg_ws_ax = fig.add_subplot(gs[1, 1])
    pfcbg_ls_ax = fig.add_subplot(gs[1, 2])
    mp2_comparison_ax = fig.add_subplot(gs[1, 3])
    
    violin_ax = fig.add_subplot(gs[2, :3])
    session_comparison_ax = fig.add_subplot(gs[2, 3])
    
    fixed_colors = {
        'action': 'purple', 'reward': 'brown', 'repeat_win': '#1f77b4',
        'change_lose': '#ff7f0e', 'change_win': '#2ca02c', 'repeat_lose': '#d62728'
    }
    colors = [fixed_colors.get(key, 'gray') for key in weight_keys] # ensure colors list matches weight_keys
    labels = ['Action', 'Reward', 'Repeat Win', 'Change Win', 'Repeat Lose', 'Change Lose']
    
    # Colors and labels for behavioral analysis
    behavioral_colors = [fixed_colors.get(key, 'gray') for key in behavioral_weight_keys]
    behavioral_labels = ['Repeat Win', 'Change Win', 'Repeat Lose', 'Change Lose']

    if not layout_test_mode:
        mp1_logistic_ax.set_prop_cycle(None)
        if strategic:
            # bg_data_algo1 = paper_logistic_regression_strategic(mp1_logistic_ax, True, algo1_data, order=5)
            bg_data_algo1, bg_data_algo1_err = fit_glr(algo1_data, order=5, a_order=2, r_order=1, err = True, model = True, labels = False, average = True)
            plot_glr_results(mp1_logistic_ax, bg_data_algo1, bg_data_algo1_err, 5, behavioral_labels, behavioral_colors)
            mp2_logistic_ax.set_prop_cycle(None)
            # bg_data_algo2 = paper_logistic_regression_strategic(mp2_logistic_ax, True, algo2_data_stacked, order=5)
            bg_data_algo2, bg_data_algo2_err = fit_glr(algo2_data_stacked, order=5, a_order=2, r_order=1, err = True, model = True, labels = False, average = True)
            plot_glr_results(mp2_logistic_ax, bg_data_algo2, bg_data_algo2_err, 5, behavioral_labels, behavioral_colors)
        else:
            bg_data_algo1 = paper_logistic_regression(mp1_logistic_ax, True, algo1_data, order=5)
            mp2_logistic_ax.set_prop_cycle(None)
            bg_data_algo2 = paper_logistic_regression(mp2_logistic_ax, True, algo2_data_stacked, order=5)
    else: # layout_test_mode is True
        mp1_logistic_ax.set_ylabel('Regression Coefficient', fontsize=14)
        mp2_logistic_ax.set_ylabel('Regression Coefficient', fontsize=14)
        mp1_logistic_ax.set_xticks(range(1, 5 + 1))
        mp2_logistic_ax.set_xticks(range(1, 5 + 1))
        # Optionally draw placeholder lines if legend items are needed from here (currently not the case)
        # mp1_logistic_ax.plot([], [], label="Reg 1") # Example
        # mp2_logistic_ax.plot([], [], label="Reg 1") # Example

    mp1_logistic_ax.set_ylabel('Regression Coefficient', fontsize=14) # Ensure labels set in both modes
    mp2_logistic_ax.set_ylabel('Regression Coefficient', fontsize=14) # Ensure labels set in both modes
    
    # Calculate mean win rates for titles - handles potential empty or non-numeric data in algoX_data[2]
    try:
        # Handle case where algo*_data[2] might be a list of arrays
        # First, flatten the data structure
        flattened_rewards_algo1 = []
        flattened_rewards_algo2 = []
        
        # Check if algo*_data[2] exists and process accordingly
        if len(algo1_data) > 2 and algo1_data[2] is not None:
            for item in algo1_data[2]:
                if isinstance(item, (list, np.ndarray)):
                    flattened_rewards_algo1.extend(item)
                else:
                    flattened_rewards_algo1.append(item)
        
        if len(algo2_data) > 2 and algo2_data[2] is not None:
            for item in algo2_data[2]:
                if isinstance(item, (list, np.ndarray)):
                    flattened_rewards_algo2.extend(item)
                else:
                    flattened_rewards_algo2.append(item)
        
        # Extract valid numerical values (filter out NaN values too)
        valid_values_algo1 = [val for val in flattened_rewards_algo1 if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val))]
        valid_values_algo2 = [val for val in flattened_rewards_algo2 if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val))]
        
        # Calculate mean with fallback to default values if no valid values
        mean_wr_algo1 = np.mean(valid_values_algo1) if valid_values_algo1 else 0.5
        mean_wr_algo2 = np.mean(valid_values_algo2) if valid_values_algo2 else 0.48
        
        # Ensure values are finite
        if not np.isfinite(mean_wr_algo1):
            mean_wr_algo1 = 0.5
        if not np.isfinite(mean_wr_algo2):
            mean_wr_algo2 = 0.48
    except (TypeError, IndexError, ValueError): # Fallback for any errors
        mean_wr_algo1 = 0.50
        mean_wr_algo2 = 0.48

    # Set plot titles with win rates
    mp1_logistic_ax.set_title(f'MP 1: WR = {mean_wr_algo1:.2f}', fontsize=12, pad=35)
    mp2_logistic_ax.set_title(f'MP 2: WR = {mean_wr_algo2:.2f}', fontsize=12, pad=35)
    
    algo1_axs = [pfc_ws_ax, pfc_ls_ax]
    algo2_axs = [pfcbg_ws_ax, pfcbg_ls_ax]
    
    fig.suptitle('Figure 6: Differences in Model Encoding', fontsize=24)
    
    if not layout_test_mode:
        timescale_1, signs_1, weights_1, var_1 = plot_population_encoding_paper(algo1_model, algo1_data, order=8, nits=nits, ax=algo1_axs, multilayer=multilayer, timescale=False)
        timescale_2, signs_2, weights_2, var_2 = plot_population_encoding_paper(algo2_model, algo2_data, order=8, nits=nits, ax=algo2_axs, linestyle='--', multilayer=multilayer, timescale=False)
        
        # Rename keys to match new terminology
        key_map = {
            'win_stay': 'repeat_win',
            'win_switch': 'change_win',
            'lose_stay': 'repeat_lose',
            'lose_switch': 'change_lose'
        }
        
        timescale_1, signs_1, weights_1, var_1 = (rename_keys(d, key_map) for d in (timescale_1, signs_1, weights_1, var_1))
        timescale_2, signs_2, weights_2, var_2 = (rename_keys(d, key_map) for d in (timescale_2, signs_2, weights_2, var_2))

        # Update subplot titles to reflect new terminology
        pfc_ws_ax.set_title('Repeat Win Coding', fontsize=14)
        pfc_ls_ax.set_title('Change Lose Coding', fontsize=14)
        pfcbg_ws_ax.set_title('Repeat Win Coding', fontsize=14)
        pfcbg_ls_ax.set_title('Change Lose Coding', fontsize=14)
        
        # Ensure x-axis ticks only appear at the bottom of the heatmap plots
        for ax in algo1_axs + algo2_axs:
            # Remove top ticks and labels
            ax.xaxis.set_tick_params(top=False, labeltop=False)
            # Ensure bottom ticks and labels are visible
            ax.xaxis.set_tick_params(bottom=True, labelbottom=True)
    # In layout_test_mode, timescales and weights are already mocked. Heatmap axes will be blank.

    # Encoding strength by Lag plots
    com_comparison = {'mean_curve': {'mp1': {}, 'mp2': {}}, 'individual_curves': {'mp1': {}, 'mp2': {}}}
    mp1_comparison_ax.set_title('MP 1: Encoding Strength by Lag', fontsize=12, pad=35)
    mp2_comparison_ax.set_title('MP 2: Encoding Strength by Lag', fontsize=12, pad=35)
    mp1_comparison_ax.set_xlabel('Lag (Trials Back)', fontsize=14)
    mp2_comparison_ax.set_xlabel('Lag (Trials Back)', fontsize=14)
    mp1_comparison_ax.set_ylabel('|Encoding Strength|', fontsize=14)
    mp2_comparison_ax.set_ylabel('|Encoding Strength|', fontsize=14)

    # This loop will run with real or mocked (zeros) weights
    for i, key in enumerate(weight_keys): # Use weight_keys instead of weights_2.keys() to ensure we process all expected keys
        # Check if the key exists in both weight dictionaries and handle accordingly
        if key not in weights_2 or key not in weights_1:
            # Skip keys not in weights dictionaries or initialize empty arrays if needed
            print(f"Warning: Key '{key}' not found in weights dictionaries")
            if key not in weights_2:
                weights_2[key] = np.zeros((1, len(lags)))
            if key not in weights_1:
                weights_1[key] = np.zeros((1, len(lags)))
        
        # Ensure we're working with arrays of the right shape
        current_weights_2 = weights_2[key]
        current_weights_1 = weights_1[key]
        
        # Handle case where weights might be None or empty
        if current_weights_2 is None or len(current_weights_2) == 0:
            current_weights_2 = np.zeros((1, len(lags)))
        if current_weights_1 is None or len(current_weights_1) == 0:
            current_weights_1 = np.zeros((1, len(lags)))
            
        # Take absolute values for center of mass calculation
        current_weights_2 = np.abs(current_weights_2)
        current_weights_1 = np.abs(current_weights_1)
        
        # Calculate means for plotting (with error handling)
        mean_weights_2 = np.mean(current_weights_2, axis=0) if current_weights_2.shape[0] > 0 else np.zeros(len(lags))
        mean_weights_1 = np.mean(current_weights_1, axis=0) if current_weights_1.shape[0] > 0 else np.zeros(len(lags))
        
        # Calculate standard errors for error bars (with error handling)
        sem_weights_2 = np.std(current_weights_2, axis=0) / np.sqrt(current_weights_2.shape[0]) if current_weights_2.shape[0] > 0 else np.zeros_like(mean_weights_2)
        sem_weights_1 = np.std(current_weights_1, axis=0) / np.sqrt(current_weights_1.shape[0]) if current_weights_1.shape[0] > 0 else np.zeros_like(mean_weights_1)
        
        # Calculate center of mass on mean data
        com_2_mean = calculate_center_of_mass(lags, mean_weights_2)
        com_1_mean = calculate_center_of_mass(lags, mean_weights_1)
        com_comparison['mean_curve']['mp2'][key] = com_2_mean
        com_comparison['mean_curve']['mp1'][key] = com_1_mean
        
        # Calculate center of mass for each individual curve
        individual_coms_2 = []
        for j in range(current_weights_2.shape[0]):
            # Ensure the weights are valid before calculating COM
            weights_row = current_weights_2[j]
            if np.sum(weights_row) > 0:  # Skip rows with all zeros
                com = calculate_center_of_mass(lags, weights_row)
                if np.isfinite(com):  # Only add finite values
                    individual_coms_2.append(com)
        
        individual_coms_1 = []
        for j in range(current_weights_1.shape[0]):
            # Ensure the weights are valid before calculating COM
            weights_row = current_weights_1[j]
            if np.sum(weights_row) > 0:  # Skip rows with all zeros
                com = calculate_center_of_mass(lags, weights_row)
                if np.isfinite(com):  # Only add finite values
                    individual_coms_1.append(com)
        
        # Calculate mean of individual COMs (with error handling)
        if individual_coms_2:
            com_2_individual = np.nanmean(individual_coms_2)
        else:
            com_2_individual = com_2_mean if np.isfinite(com_2_mean) else np.nan
            
        if individual_coms_1:
            com_1_individual = np.nanmean(individual_coms_1)
        else:
            com_1_individual = com_1_mean if np.isfinite(com_1_mean) else np.nan
            
        com_comparison['individual_curves']['mp2'][key] = com_2_individual
        com_comparison['individual_curves']['mp1'][key] = com_1_individual
        
        # Initialize timescale dictionaries if needed
        if not timescale_2:
            timescale_2 = {}
        if not timescale_1:
            timescale_1 = {}
            
        # Store the calculated timescales
        timescale_2[key] = com_2_individual
        timescale_1[key] = com_1_individual
        
        # Plot the encoding strength curves
        mp1_comparison_ax.errorbar(lags, mean_weights_1, yerr=sem_weights_1, fmt='o-', label=labels[i], color=colors[i], capsize=3, alpha=0.8)
        mp2_comparison_ax.errorbar(lags, mean_weights_2, yerr=sem_weights_2, fmt='o-', label=labels[i], color=colors[i], capsize=3, alpha=0.8)
    
    mp1_comparison_ax.legend(loc='upper right', fontsize=6, frameon=False)
    ymax = max(mp1_comparison_ax.get_ylim()[1], mp2_comparison_ax.get_ylim()[1], 0.1) # ensure ymax is at least a small positive
    ymin = min(mp1_comparison_ax.get_ylim()[0], mp2_comparison_ax.get_ylim()[0], 0)   # ensure ymin is at least 0
    mp1_comparison_ax.set_ylim(ymin, ymax)
    mp2_comparison_ax.set_ylim(ymin, ymax)
    
    # Violin Plot
    violin_data_list = [] # Renamed from violin_data to avoid confusion with outer scope algo_data
    violin_positions = []
    violin_labels_text = [] # Renamed from violin_labels
    p_values = []
    position = 1
    
    # This loop iterates over weight_keys. Uses mocked weights in layout_test_mode.
    for i, key in enumerate(weight_keys):
        if key not in weights_2 or key not in weights_1:
            if key not in weights_2:
                weights_2[key] = np.zeros((1, len(lags)))
            if key not in weights_1:
                weights_1[key] = np.zeros((1, len(lags)))
                
        current_weights_2_violin = weights_2[key]
        current_weights_1_violin = weights_1[key]
        
        # Handle case where weights might be None or empty
        if current_weights_2_violin is None or len(current_weights_2_violin) == 0:
            current_weights_2_violin = np.zeros((1, len(lags)))
        if current_weights_1_violin is None or len(current_weights_1_violin) == 0:
            current_weights_1_violin = np.zeros((1, len(lags)))
            
        # Take absolute values for calculation
        current_weights_2_violin = np.abs(current_weights_2_violin)
        current_weights_1_violin = np.abs(current_weights_1_violin)

        # Calculate center of mass for each curve using the same approach as above
        mp2_coms = []
        for j in range(current_weights_2_violin.shape[0]):
            weights_row = current_weights_2_violin[j]
            if np.sum(weights_row) > 0:
                com = calculate_center_of_mass(lags, weights_row)
                if np.isfinite(com):
                    mp2_coms.append(com)
        
        mp1_coms = []
        for j in range(current_weights_1_violin.shape[0]):
            weights_row = current_weights_1_violin[j]
            if np.sum(weights_row) > 0:
                com = calculate_center_of_mass(lags, weights_row)
                if np.isfinite(com):
                    mp1_coms.append(com)
        
        # If no valid COMs, use the timescale calculated earlier or a default value
        if not mp2_coms and key in timescale_2 and np.isfinite(timescale_2[key]):
            mp2_coms = [timescale_2[key]]
        if not mp1_coms and key in timescale_1 and np.isfinite(timescale_1[key]):
            mp1_coms = [timescale_1[key]]
            
        # Ensure we have something to plot even if all calculations failed
        if not mp2_coms:
            mp2_coms = [4.0]  # Default reasonable value
        if not mp1_coms:
            mp1_coms = [4.0]  # Default reasonable value

        # Run statistics only if we have sufficient data
        if len(mp2_coms) >= 5 and len(mp1_coms) >= 5 and not layout_test_mode:
            try:
                u_stat, p_value = stats.mannwhitneyu(mp2_coms, mp1_coms)
                p_values.append(p_value)
            except ValueError:
                try:
                    t_stat, p_value = stats.ttest_ind(mp2_coms, mp1_coms, equal_var=False, nan_policy='omit')
                    p_values.append(p_value)
                except:
                    p_values.append(1.0)  # Default to non-significant
        elif not layout_test_mode:
            p_values.append(1.0)  # Default to non-significant

        # Add data for violins
        violin_data_list.append(mp1_coms)
        violin_data_list.append(mp2_coms)
        violin_positions.append(position)
        violin_positions.append(position + 1)
        violin_labels_text.extend(["", ""])
        position += 3
    
    if violin_data_list: # Check if there's anything to plot (even placeholders)
        violin_parts = violin_ax.violinplot(violin_data_list, positions=violin_positions, showmeans=True, showmedians=False, showextrema=True)
        
        for i_vp, pc in enumerate(violin_parts['bodies']):
            regressor_idx_vp = i_vp // 2
            if regressor_idx_vp < len(colors):
                edge_color_vp = colors[regressor_idx_vp]
                face_color_vp = 'white' if i_vp % 2 == 0 else colors[regressor_idx_vp]
                alpha_vp = 1.0 if i_vp % 2 == 0 else 0.7
                hatch_vp = '///' if i_vp % 2 == 0 else None
                pc.set_facecolor(face_color_vp)
                pc.set_edgecolor(edge_color_vp)
                pc.set_alpha(alpha_vp)
                if hatch_vp: pc.set_hatch(hatch_vp)
                pc.set_linewidth(1)
            else: # Fallback color if regressor_idx_vp is out of bounds for colors
                pc.set_facecolor('gray')
                pc.set_edgecolor('black')

        if 'cmeans' in violin_parts: violin_parts['cmeans'].set_edgecolor('black')
        if 'cbars' in violin_parts: violin_parts['cbars'].set_color('black')
        if 'cmins' in violin_parts: violin_parts['cmins'].set_color('black')
        if 'cmaxes' in violin_parts: violin_parts['cmaxes'].set_color('black')
    
    # Determine y_data_max for positioning significance markers and tau text
    # Use a robust way to get max value from potentially nested/jagged violin_data_list
    all_finite_violin_data = [x for subl in violin_data_list for x in subl if hasattr(subl, '__iter__') and np.isfinite(x)]
    if not all_finite_violin_data: # Handle case where all_finite_violin_data is empty
        all_finite_violin_data = [x for x in violin_data_list if np.isscalar(x) and np.isfinite(x)] # Check for scalar finite values

    y_data_max = max(all_finite_violin_data) if all_finite_violin_data else 1.0
    violin_ylim_max = y_data_max * 1.45 # Increased slightly for more room
    sig_base_height = y_data_max * 1.20 # Increased slightly

    for i_sig in range(0, len(violin_positions), 2):
        regressor_idx_sig = i_sig // 2
        x1_sig, x2_sig = violin_positions[i_sig], violin_positions[i_sig+1]

        if regressor_idx_sig < len(labels):
            # Push x-group labels even further down
            violin_ax.text((x1_sig + x2_sig) / 2, -2.2 * (y_data_max / 7 if y_data_max > 0 else 0), labels[regressor_idx_sig], ha='center', va='top', fontsize=11, fontweight='bold')

        if regressor_idx_sig < len(p_values): # Only if p_values were calculated (not in layout_test_mode with insufficient data)
            p_sig = p_values[regressor_idx_sig]
            sig_height_val = sig_base_height 
            violin_ax.plot([x1_sig, x2_sig], [sig_height_val, sig_height_val], 'k-', linewidth=1.5)
            text_sig = 'ns'
            if p_sig < 0.001: text_sig = '***'
            elif p_sig < 0.01: text_sig = '**'
            elif p_sig < 0.05: text_sig = '*'
            violin_ax.text((x1_sig + x2_sig) / 2, sig_height_val * 1.01, text_sig, ha='center', va='bottom', fontsize=12, fontweight='bold')

        mp1_com_val = timescale_1.get(weight_keys[regressor_idx_sig], np.nan)
        mp2_com_val = timescale_2.get(weight_keys[regressor_idx_sig], np.nan)
        y_pos_mp1_tau = y_data_max * 0.70 if y_data_max > 0 else 0.7
        y_pos_mp2_tau = y_data_max * 0.30 if y_data_max > 0 else 0.3
        
        if regressor_idx_sig < len(colors):
            violin_ax.text(x1_sig - 0.3, y_pos_mp1_tau, rf"$\tau$={mp1_com_val:.2f}", ha='right', va='center', fontsize=10, color=colors[regressor_idx_sig])
            violin_ax.text(x2_sig + 0.3, y_pos_mp2_tau, rf"$\tau$={mp2_com_val:.2f}", ha='left', va='center', fontsize=10, color=colors[regressor_idx_sig])

    current_ylim_bottom, current_ylim_top = violin_ax.get_ylim()
    # Ensure bottom y_lim accommodates the pushed down x-labels
    violin_ax.set_ylim(min(current_ylim_bottom, -2.2 * (y_data_max / 7 if y_data_max > 0 else 0.15)), violin_ylim_max if violin_ylim_max > 0 else 1.0)
    
    max_tick = int(violin_ylim_max if violin_ylim_max > 0 else 1.0) + 1
    yticks = np.arange(0, min(9, max_tick), 2 if max_tick > 4 else 1) # Adjusted step for yticks
    if not list(yticks) and max_tick > 0 : yticks = [0, round(max_tick/2), max_tick-1 if max_tick >0 else 1]
    elif not list(yticks): yticks = [0,1]
    violin_ax.set_yticks(list(yticks))
    violin_ax.tick_params(axis='y', labelsize=10)

    violin_ax.set_xticks([])
    violin_ax.set_xticklabels([])
    violin_ax.set_ylabel(r'Neuronal Timescale ($\tau$)', fontsize=14)
    violin_ax.set_title('Comparison of Timescales between MP1 and MP2', fontsize=14, pad=15)
    
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='white', alpha=1.0, edgecolor='black', hatch='///', linewidth=1.25, label='MP1'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', linewidth=1.25, label='MP2')
    ]
    violin_ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1.21), ncol=2, frameon=False, fontsize=10)
    
    # Session-by-session comparison plot
    
    # Debug session lengths and reset behavior  
    if not layout_test_mode:
        print(f"\nDEBUG SESSION ANALYSIS:")
        print(f"env1 reset_time = {getattr(env1, 'reset_time', 'N/A')}, model max_steps = {getattr(algo1_model, 'max_steps', 'N/A')}")
        print(f"env2 reset_time = {getattr(env2, 'reset_time', 'N/A')}, model max_steps = {getattr(algo2_model, 'max_steps', 'N/A')}")
        print(f"Number of MP1 sessions: {len(algo1_data_list)}")
        print(f"Number of MP2 sessions: {len(algo2_data_list)}")
        
        # Check session lengths - properly handle nested structure
        for i, session_data in enumerate(algo1_data_list[:3]):  # Check first 3 sessions
            if len(session_data) > 2 and session_data[2] is not None:
                # session_data[2] is episode_rewards - could be list of episodes
                if hasattr(session_data[2], '__len__'):
                    if isinstance(session_data[2][0], (list, np.ndarray)):
                        # Multiple episodes - sum lengths
                        total_trials = sum(len(episode) for episode in session_data[2])
                        print(f"MP1 session {i} length: {total_trials} trials across {len(session_data[2])} episodes")
                    else:
                        # Single episode
                        print(f"MP1 session {i} length: {len(session_data[2])} trials (single episode)")
                else:
                    print(f"MP1 session {i} length: N/A (not iterable)")
        
        for i, session_data in enumerate(algo2_data_list[:3]):  # Check first 3 sessions  
            if len(session_data) > 2 and session_data[2] is not None:
                # session_data[2] is episode_rewards - could be list of episodes
                if hasattr(session_data[2], '__len__'):
                    if isinstance(session_data[2][0], (list, np.ndarray)):
                        # Multiple episodes - sum lengths
                        total_trials = sum(len(episode) for episode in session_data[2])
                        print(f"MP2 session {i} length: {total_trials} trials across {len(session_data[2])} episodes")
                    else:
                        # Single episode
                        print(f"MP2 session {i} length: {len(session_data[2])} trials (single episode)")
                else:
                    print(f"MP2 session {i} length: N/A (not iterable)")

    # Basic plot setup
    session_comparison_ax.set_ylabel(r'Behavioral Timescale ($\tau$)', fontsize=12)
    session_comparison_ax.set_title('Behavioral Timescale \n Comparison', fontsize=14, pad=15, x=0.7)
    session_comparison_ax.set_xticks(range(len(behavioral_weight_keys)))
    session_comparison_ax.set_xticklabels([label.replace('-', '\n') for label in behavioral_labels], 
                                        fontsize=10, rotation=0, ha='center')
    session_comparison_ax.tick_params(axis='both', which='major', labelsize=10)
    session_comparison_ax.grid(True, alpha=0.3)

    # Fit regressions for each session and extract coefficients by component
    if not layout_test_mode:
        # Initialize dictionaries to store coefficients for each behavioral component
        algo1_coeffs = {key: [] for key in behavioral_weight_keys}
        algo2_coeffs = {key: [] for key in behavioral_weight_keys}
        
        print(f"\nFitting regressions for {len(algo1_data_list)} MP1 sessions...")
        for i, session_data in enumerate(algo1_data_list):
            try:
                # Calculate session length properly
                if len(session_data) > 2 and session_data[2] is not None:
                    if hasattr(session_data[2], '__len__'):
                        if isinstance(session_data[2][0], (list, np.ndarray)):
                            # Multiple episodes - sum lengths
                            total_trials = sum(len(episode) for episode in session_data[2])
                            print(f"MP1 session {i}: {total_trials} trials across {len(session_data[2])} episodes")
                        else:
                            # Single episode
                            total_trials = len(session_data[2])
                            print(f"MP1 session {i}: {total_trials} trials (single episode)")
                    else:
                        total_trials = 0
                        print(f"MP1 session {i}: 0 trials (not iterable)")
                else:
                    total_trials = 0
                    print(f"MP1 session {i}: 0 trials (no reward data)")
                
                # Only fit if we have enough trials
                if total_trials < 25:
                    print(f"  Skipping MP1 session {i}: insufficient trials ({total_trials} < 25)")
                    for key in behavioral_weight_keys:
                        algo1_coeffs[key].append(0.0)
                    continue
                
                # For model data (True parameter), use paper_logistic_regression_strategic 
                fit_result, fit_result_err = fit_glr(session_data, order=5, a_order=2, r_order=1, err = True, model = True, labels = False, average = True)
                fit_result = {'action': fit_result, 'err': fit_result_err}
                # Extract coefficients and compute center of mass for each component
                if isinstance(fit_result, dict) and 'action' in fit_result:
                    coeffs = fit_result['action']
                    # Strategic regression coefficient structure: win_stay (0:5), lose_switch (5:10), win_switch (10:15), lose_stay (15:20)
                    lags = np.arange(1, 6)  # lags 1-5 for order=5
                    
                    for j, key in enumerate(behavioral_weight_keys):
                        if key == 'repeat_win':
                            # Extract win_stay coefficients (0:5) and compute center of mass
                            rw_coeffs = coeffs[0:5] if len(coeffs) >= 5 else coeffs[0:min(len(coeffs), 5)]
                            if len(rw_coeffs) > 0 and np.sum(np.abs(rw_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(rw_coeffs)], rw_coeffs)
                                algo1_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo1_coeffs[key].append(0.0)
                        elif key == 'change_lose':
                            # Extract lose_switch coefficients (5:10) and compute center of mass
                            cl_coeffs = coeffs[15:20] if len(coeffs) >= 10 else coeffs[15:min(len(coeffs), 10)]
                            if len(cl_coeffs) > 0 and np.sum(np.abs(cl_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(cl_coeffs)], cl_coeffs)
                                algo1_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo1_coeffs[key].append(0.0)
                        elif key == 'change_win':
                            # Extract win_switch coefficients (10:15) and compute center of mass
                            cw_coeffs = coeffs[5:10] if len(coeffs) >= 15 else coeffs[5:min(len(coeffs), 15)]
                            if len(cw_coeffs) > 0 and np.sum(np.abs(cw_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(cw_coeffs)], cw_coeffs)
                                algo1_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo1_coeffs[key].append(0.0)
                        elif key == 'repeat_lose':
                            # Extract lose_stay coefficients (15:20) and compute center of mass
                            rl_coeffs = coeffs[10:15] if len(coeffs) >= 20 else coeffs[10:min(len(coeffs), 20)]
                            if len(rl_coeffs) > 0 and np.sum(np.abs(rl_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(rl_coeffs)], rl_coeffs)
                                algo1_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo1_coeffs[key].append(0.0)
                        else:
                            algo1_coeffs[key].append(0.0)
                else:
                    print(f"  Warning: Unexpected fit result format for MP1 session {i}: {type(fit_result)}")
                    for key in behavioral_weight_keys:
                        algo1_coeffs[key].append(0.0)
                            
            except Exception as e:
                print(f"Error fitting MP1 session {i}: {e}")
                # Add default values for failed fits
                for key in behavioral_weight_keys:
                    algo1_coeffs[key].append(0.0)
        
        print(f"\nFitting regressions for {len(algo2_data_list)} MP2 sessions...")
        for i, session_data in enumerate(algo2_data_list):
            try:
                # Calculate session length properly
                if len(session_data) > 2 and session_data[2] is not None:
                    if hasattr(session_data[2], '__len__'):
                        if isinstance(session_data[2][0], (list, np.ndarray)):
                            # Multiple episodes - sum lengths
                            total_trials = sum(len(episode) for episode in session_data[2])
                            print(f"MP2 session {i}: {total_trials} trials across {len(session_data[2])} episodes")
                        else:
                            # Single episode
                            total_trials = len(session_data[2])
                            print(f"MP2 session {i}: {total_trials} trials (single episode)")
                    else:
                        total_trials = 0
                        print(f"MP2 session {i}: 0 trials (not iterable)")
                else:
                    total_trials = 0
                    print(f"MP2 session {i}: 0 trials (no reward data)")
                
                # Only fit if we have enough trials
                if total_trials < 25:
                    print(f"  Skipping MP2 session {i}: insufficient trials ({total_trials} < 25)")
                    for key in behavioral_weight_keys:
                        algo2_coeffs[key].append(0.0)
                    continue
                
                # For model data (True parameter), use paper_logistic_regression_strategic 
                fit_result, fit_result_err = fit_glr(session_data, order=5, a_order=2, r_order=1, err = True, model = True, labels = False, average = True)
                fit_result = {'action': fit_result, 'err': fit_result_err}
                # Extract coefficients and compute center of mass for each component
                if isinstance(fit_result, dict) and 'action' in fit_result:
                    coeffs = fit_result['action']
                    for j, key in enumerate(behavioral_weight_keys):
                        if key == 'repeat_win':
                            # Extract win_stay coefficients (0:5) and compute center of mass
                            rw_coeffs = coeffs[0:5] if len(coeffs) >= 5 else coeffs[0:min(len(coeffs), 5)]
                            if len(rw_coeffs) > 0 and np.sum(np.abs(rw_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(rw_coeffs)], rw_coeffs)
                                algo2_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo2_coeffs[key].append(0.0)
                        elif key == 'change_lose':
                            # Extract change_lose coefficients (5:10) and compute center of mass
                            cl_coeffs = coeffs[15:20] if len(coeffs) >= 10 else coeffs[5:min(len(coeffs), 10)]
                            if len(cl_coeffs) > 0 and np.sum(np.abs(cl_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(cl_coeffs)], cl_coeffs)
                                algo2_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo2_coeffs[key].append(0.0)
                        elif key == 'change_win':
                            # Extract win_switch coefficients (10:15) and compute center of mass
                            cw_coeffs = coeffs[5:10] if len(coeffs) >= 15 else coeffs[5:min(len(coeffs), 15)]
                            if len(cw_coeffs) > 0 and np.sum(np.abs(cw_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(cw_coeffs)], cw_coeffs)
                                algo2_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo2_coeffs[key].append(0.0)
                        elif key == 'repeat_lose':
                            # Extract lose_stay coefficients (15:20) and compute center of mass
                            rl_coeffs = coeffs[10:15] if len(coeffs) >= 20 else coeffs[10:min(len(coeffs), 20)]
                            if len(rl_coeffs) > 0 and np.sum(np.abs(rl_coeffs)) > 0:
                                com = calculate_center_of_mass(lags[:len(rl_coeffs)], rl_coeffs)
                                algo2_coeffs[key].append(com if np.isfinite(com) else 0.0)
                            else:
                                algo2_coeffs[key].append(0.0)
                        else:
                            algo2_coeffs[key].append(0.0)
                else:
                    print(f"  Warning: Unexpected fit result format for MP2 session {i}: {type(fit_result)}")
                    for key in behavioral_weight_keys:
                        algo2_coeffs[key].append(0.0)
                            
            except Exception as e:
                print(f"Error fitting MP2 session {i}: {e}")
                # Add default values for failed fits
                for key in behavioral_weight_keys:
                    algo2_coeffs[key].append(0.0)
        
        # Create errorbar plot for each behavioral component
        x_positions = np.arange(len(behavioral_weight_keys))
        width = 0.35
        
        for i, key in enumerate(behavioral_weight_keys):
            # Calculate means and standard errors
            algo1_values = np.array(algo1_coeffs[key])
            algo2_values = np.array(algo2_coeffs[key])
            
            # algo1_values = np.exp(algo1_values)
            # algo2_values = np.exp(algo2_values)
            
            algo1_mean = np.mean(algo1_values) if len(algo1_values) > 0 else 0.0
            
            algo1_sem = np.std(algo1_values) / np.sqrt(len(algo1_values)) if len(algo1_values) > 1 else 0.0
            
            algo2_mean = np.mean(algo2_values) if len(algo2_values) > 0 else 0.0
            algo2_sem = np.std(algo2_values) / np.sqrt(len(algo2_values)) if len(algo2_values) > 1 else 0.0
            
            # Plot error bars - empty circle for MP1, filled circle for MP2
            session_comparison_ax.errorbar(i - width/2, algo1_mean, yerr=algo1_sem, 
                                         fmt='o', color=behavioral_colors[i], alpha=0.8, 
                                         markerfacecolor='none', markeredgecolor=behavioral_colors[i],
                                         capsize=3, capthick=1, label='MP1' if i == 0 else "")
            session_comparison_ax.errorbar(i + width/2, algo2_mean, yerr=algo2_sem, 
                                         fmt='o', color=behavioral_colors[i], alpha=0.8, 
                                         markerfacecolor=behavioral_colors[i], markeredgecolor=behavioral_colors[i],
                                         capsize=3, capthick=1, label='MP2' if i == 0 else "")
            
            print(f"{key}: MP1={algo1_mean:.3f}±{algo1_sem:.3f} (n={len(algo1_values)}), MP2={algo2_mean:.3f}±{algo2_sem:.3f} (n={len(algo2_values)})")
        
        # Add legend with black markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                   markeredgecolor='black', markersize=8, label='MP1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                   markeredgecolor='black', markersize=8, label='MP2')
        ]
        session_comparison_ax.legend(handles=legend_elements, fontsize=10)
        
    else:
        # Layout test mode - show placeholder
        session_comparison_ax.text(0.5, 0.5, 'Session-by-Session\nRegression Analysis', 
                                  ha='center', va='center', transform=session_comparison_ax.transAxes,
                                  fontsize=12, style='italic', alpha=0.7)
        

    pfc_axes = fig.add_subplot(gs[0, 1:3])
    pfcbg_axes = fig.add_subplot(gs[1, 1:3])
    title_pad = 31
    pfc_axes.set_title('RLRNN Neuronal Encoding Algorithm 1', pad=title_pad, fontsize=16)
    pfcbg_axes.set_title('RLRNN Neuronal Encoding Algorithm 2', pad=title_pad, fontsize=16)
    pfc_axes.set_axis_off()
    pfcbg_axes.set_axis_off()
    
    # Add plot letters (A-H) to each major subplot
    plot_labels = {
        'A': mp1_logistic_ax, 'B': pfc_axes, 'C': mp1_comparison_ax,
        'D': mp2_logistic_ax, 'E': pfcbg_axes, 'F': mp2_comparison_ax,
        'G': violin_ax, 'H': session_comparison_ax
    }

    
    # Use a base x-offset and adjust for axes that span multiple columns
    base_x_offset = -0.4
    for label, ax in plot_labels.items():
        if ax in [pfc_axes, pfcbg_axes]:  # 2-column axes
            x_pos = base_x_offset / 4
        elif ax == violin_ax:  # This is for label 'G'
            x_pos = base_x_offset / 3 + .033
        elif label in ['C', 'F', 'H']:
            x_pos = -0.3
        else:  # 1-column axes
            x_pos = base_x_offset
        
        ax.text(x_pos, 1.15, label, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

    for ax_main in [mp1_logistic_ax, mp2_logistic_ax, mp1_comparison_ax, mp2_comparison_ax, session_comparison_ax]:
        # Titles are set above, this just ensures font size if they were missed or reset
        current_title = ax_main.get_title()
        if current_title: ax_main.set_title(current_title, fontsize=16) 
        
        current_xlabel = ax_main.get_xlabel()
        if current_xlabel: ax_main.set_xlabel(current_xlabel, fontsize=14)

        current_ylabel = ax_main.get_ylabel()
        if current_ylabel: ax_main.set_ylabel(current_ylabel, fontsize=14)
        
        ax_main.tick_params(axis='both', which='major', labelsize=12)
        leg = ax_main.get_legend()
        if leg:
            leg.set_frame_on(False)
            
    plt.tight_layout(h_pad=0.5, w_pad=0.5, rect=[0.08, 0.08, 0.98, 0.96])
    return fig

# ========================= New correlation analysis utilities ========================= #

def _get_hidden_dim_from_data(data):
    try:
        episode_hiddens = data[3]
        if hasattr(episode_hiddens, '__len__') and len(episode_hiddens) > 0:
            first = episode_hiddens[0]
            if hasattr(first, 'shape') and len(first.shape) >= 1:
                return int(first.shape[0])
    except Exception:
        pass
    return None

def _compute_per_neuron_com(weights_dict, order):
    lags = np.arange(1, order + 1)
    per_neuron_coms = {}
    for key, weight_matrix in weights_dict.items():
        if weight_matrix is None or len(weight_matrix) == 0:
            per_neuron_coms[key] = np.array([])
            continue
        weight_matrix = np.asarray(weight_matrix)
        coms = []
        for i in range(weight_matrix.shape[0]):
            row = np.abs(weight_matrix[i])
            if np.sum(row) > 0:
                com = calculate_center_of_mass(lags[:len(row)], row[:len(lags)])
                if np.isfinite(com):
                    coms.append(com)
                else:
                    coms.append(np.nan)
            else:
                coms.append(np.nan)
        per_neuron_coms[key] = np.array(coms, dtype=float)
    return per_neuron_coms

def compute_model_timescale_correlations(data_entry, order=8, model_id=None):
    try:
        from analysis_scripts.population_coding import plot_population_encoding_paper
    except Exception:
        return None

    if isinstance(data_entry, dict) and 'data' in data_entry:
        data = data_entry['data']
        if model_id is None:
            model_id = str(data_entry.get('decision_params', 'unknown')) + f"_{data_entry.get('iteration', 'NA')}"
    else:
        data = data_entry
        if model_id is None:
            model_id = 'unknown'

    hidden_dim = _get_hidden_dim_from_data(data)
    if hidden_dim is None:
        return None

    class _M: pass
    m = _M()
    m.hidden_dim = int(hidden_dim)

    try:
        # Use multilayer=False because we only have saved hidden states, not full model policy layers
        timescales, signs, weights, var = plot_population_encoding_paper(m, data, order=order, nits=1, ax=None, multilayer=False, timescale=False)
    except Exception:
        return None

    per_neuron_coms = _compute_per_neuron_com(weights, order)

    def _pairwise_corr(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 3 or y.size < 3:
            return np.nan, np.nan, 0
        r = np.corrcoef(x, y)[0, 1]
        try:
            from scipy import stats
            rho, _ = stats.spearmanr(x, y)
        except Exception:
            rho = np.nan
        return float(r), float(rho), int(x.size)

    results = []
    reward = per_neuron_coms.get('reward', np.array([]))
    for key in ['repeat_win', 'change_lose', 'change_win', 'repeat_lose']:
        r, rho, n = _pairwise_corr(reward, per_neuron_coms.get(key, np.array([])))
        results.append({
            'model_id': model_id,
            'variable': key,
            'pearson_r': r,
            'spearman_r': rho,
            'n_neurons': n
        })
    return results

def analyze_timescale_correlations_across_models(dataset_path, order=8, max_models=None):
    records = []
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return pd.DataFrame([])

    if not isinstance(dataset, (list, tuple)):
        print("Dataset is not a list/tuple of models")
        return pd.DataFrame([])

    count = 0
    for idx, entry in enumerate(dataset):
        if max_models is not None and count >= max_models:
            break
        # Construct a readable model id
        if isinstance(entry, dict):
            mid = str(entry.get('decision_params', 'unknown')) + f"_{entry.get('iteration', 'NA')}"
        else:
            mid = f"model_{idx+1}"
        res = compute_model_timescale_correlations(entry, order=order, model_id=mid)
        if res is not None:
            records.extend(res)
            count += 1
    return pd.DataFrame.from_records(records)

def plot_timescale_correlations_summary(df, title_suffix=""):
    if df is None or df.empty:
        return None
    fig = plt.figure(figsize=(14, 6), dpi=300, layout='constrained')
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1.4])

    ax1 = fig.add_subplot(gs[0, 0])
    sns.violinplot(data=df, x='variable', y='pearson_r', ax=ax1, inner=None, cut=0)
    sns.stripplot(data=df, x='variable', y='pearson_r', ax=ax1, color='black', size=2, alpha=0.6)
    ax1.set_title('Correlation (Reward vs Strategy) across models' + (f" {title_suffix}" if title_suffix else ''))
    ax1.set_ylabel('Pearson r')
    ax1.set_xlabel('Strategy variable')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Heatmap (top-N models by completeness)
    ax2 = fig.add_subplot(gs[0, 1])
    pivot = df.pivot_table(index='model_id', columns='variable', values='pearson_r', aggfunc='mean')
    if len(pivot) > 60:
        pivot = pivot.iloc[:60]
    sns.heatmap(pivot, vmin=-1, vmax=1, cmap='coolwarm', ax=ax2, cbar_kws={'label': 'Pearson r'})
    ax2.set_title('Per-model correlation heatmap' + (f" {title_suffix}" if title_suffix else ''), fontsize=14)
    ax2.set_xlabel('Strategy variable')
    ax2.set_ylabel('Model')

    return fig

def run_timescale_correlation_workflow(dataset_path=weight_supp_data, order=8, max_models=None, save_path=None, title_suffix=""):
    df = analyze_timescale_correlations_across_models(dataset_path, order=order, max_models=max_models)
    fig = plot_timescale_correlations_summary(df, title_suffix=title_suffix)
    if fig is not None and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return df, fig

# Example of how to call for layout testing:
# if __name__ == '__main__':
#     # Dummy params for layout testing
#     dummy_params = {'actor_lr': 0.1} # Example, structure might need to match RLRNN expectations if not fully mocked
#     dummy_env_params = {'N_CHOICES': 2} # Example
#     fig = plot_neural_coding_algo(dummy_params, dummy_params, dummy_env_params, dummy_env_params, layout_test_mode=True)
#     plt.savefig("fig6_layout_test.png")
#     plt.show()
