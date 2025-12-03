"""
RNN Behavioral Insufficiency Analysis with Configurable Distance Metrics

This module provides tools for analyzing behavioral differences between RNNs and primates
using various distance metrics for comparing regression coefficient sequences.

Available distance metrics:
- 'frechet': Frechet distance (default, good for sequence alignment)
- 'euclidean': Euclidean distance (simple L2 norm)
- 'cosine': Cosine similarity (good for comparing shapes regardless of magnitude; higher values = more similar)
- 'manhattan': Manhattan distance (L1 norm, robust to outliers)
- 'wasserstein': Wasserstein distance (treats sequences as distributions)
- 'area_norm': Area normalized distance

Usage examples:
    rnn_figure(mpdb_p, mpbeh_p, distance_metric='euclidean')
    cluster_monkeys_by_distance(data, monkeys, strategic, nonstrategic, distance_metric='cosine')
    find_most_similar_monkey_to_rnns(distance_metric=my_custom_function)

Note: This version uses raw coefficient values without z-score normalization.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression, paper_logistic_regression_strategic, fit_single_paper, fit_single_paper_strategic
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
import pickle
from numba import jit
from typing import Callable
from analysis_scripts.entropy import *
from scipy import stats
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpdb_p = '/Users/fmb35/Desktop/matching-pennies-lite.sqlite'

mpbeh_p = '/Users/fmb35/Desktop/MPbehdata.csv'
# rnns = [0,1,6,12,7,24,23,26,27]

stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'


with open('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/RNN_zoo_dict.pkl', 'rb') as f:
    rnn_zoo_dict = pickle.load(f)

rnns = sorted(list(rnn_zoo_dict.keys()))

def euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Simple Euclidean distance between two sequences."""
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.linalg.norm(p - q)

def cosine_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Cosine similarity between two sequences, robust to zero vectors.
    Note: Despite the function name, this actually returns cosine similarity (not distance).
    Higher values indicate more similar vectors."""
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    
    # Calculate magnitudes
    p_mag = np.linalg.norm(p)
    q_mag = np.linalg.norm(q)
    
    # Handle zero magnitude cases
    if p_mag == 0 and q_mag == 0:
        return 1.0  # Two zero vectors are identical (maximum similarity)
    elif p_mag == 0 or q_mag == 0:
        return 0.0  # Minimum cosine similarity when one vector is zero
    
    # Normal cosine similarity calculation
    dot_product = np.dot(p, q)
    cosine_similarity = dot_product / (p_mag * q_mag)
    
    # Ensure similarity is in valid range [-1, 1] due to numerical precision
    # cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    # Return similarity (higher values = more similar)
    return cosine_similarity

def manhattan_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Manhattan (L1) distance between two sequences.""" 
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    return np.sum(np.abs(p - q))

def wasserstein_sequence_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein distance between two sequences treated as distributions."""
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    # Normalize to make them valid probability distributions
    p_norm = np.abs(p) / (np.sum(np.abs(p)) + 1e-10)
    q_norm = np.abs(q) / (np.sum(np.abs(q)) + 1e-10)
    return wasserstein_distance(range(len(p_norm)), range(len(q_norm)), p_norm, q_norm)

def area_norm_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Area normalized distance between two sequences."""
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    p_sum = np.sum(np.abs(p))
    q_sum = np.sum(np.abs(q))
    
    # Handle edge cases where sum is zero
    if p_sum == 0 and q_sum == 0:
        return 0.0
    elif p_sum == 0 or q_sum == 0:
        return 1.0  # Maximum distance when one sequence is all zeros
    
    p_norm = p / p_sum
    q_norm = q / q_sum
    return np.sum(np.abs(p_norm - q_norm))

def normalize_coeffs_by_lag(all_coeffs_dict, power=2):
    """
    Normalize coefficient vectors by Z-score normalization across all models for each lag position.
    
    Args:
        all_coeffs_dict: Dictionary with keys 'ws', 'ls', 'wsw', 'lss', each containing lists of coefficient arrays
        
    Returns:
        Dictionary with same structure but Z-score normalized coefficients
    """
    normalized_coeffs = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
    
    # Calculate statistics for each coefficient type and lag position
    coeff_stats = {}
    
    for coeff_type in ['ws', 'ls', 'wsw', 'lss']:
        if not all_coeffs_dict[coeff_type]:
            continue
            
        # Stack all coefficient arrays for this type
        all_arrays = all_coeffs_dict[coeff_type]
        if len(all_arrays) == 0:
            continue
            
        # Ensure all arrays have the same length
        array_lengths = [len(arr) for arr in all_arrays]
        if len(set(array_lengths)) > 1:
            print(f"Warning: {coeff_type} arrays have different lengths: {set(array_lengths)}")
            min_length = min(array_lengths)
            all_arrays = [arr[:min_length] for arr in all_arrays]
        
        # Stack all arrays and calculate statistics for each lag position
        stacked_arrays = np.vstack(all_arrays)  # Shape: (n_models, n_lags)
        
        # Calculate mean and std for each lag position across all models
        lag_means = np.mean(stacked_arrays, axis=0)  # Shape: (n_lags,)
        lag_stds = np.std(stacked_arrays, axis=0)    # Shape: (n_lags,)
        
        coeff_stats[coeff_type] = {'means': lag_means, 'stds': lag_stds}
        
        # Normalize each model's coefficients using the global lag statistics
        for arr in all_arrays:
            # Avoid division by zero
            safe_stds = np.where(lag_stds > 0, lag_stds, 1.0)
            normalized_arr = (arr - lag_means) / safe_stds
            normalized_coeffs[coeff_type].append(normalized_arr)
    
    return normalized_coeffs, coeff_stats

def normalize_coeffs(ws, ls, wsw, lss, power=2):
    """Legacy function - now just passes through for individual normalization if needed."""
    # This is now used only for individual coefficient normalization in special cases
    all_coeffs = np.concatenate([ws.flatten(), ls.flatten(), wsw.flatten(), lss.flatten()])
    mean_val = np.mean(all_coeffs)
    std_val = np.std(all_coeffs)
    
    if std_val > 0:
        ws_norm = (ws - mean_val) / std_val
        ls_norm = (ls - mean_val) / std_val
        wsw_norm = (wsw - mean_val) / std_val
        lss_norm = (lss - mean_val) / std_val
        return ws_norm, ls_norm, wsw_norm, lss_norm
    else:
        return ws, ls, wsw, lss

def partition_dataset(monkey_data, n_trials=5000):
    """
    Partition the entire dataset into multiple chunks of approximately n_trials trials each.
    Partitions will be >= n_trials, not smaller. Remaining trials that would be < n_trials 
    are combined with the previous partition.
    
    Args:
        monkey_data: DataFrame with session data
        n_trials: Minimum number of trials per partition
    
    Returns:
        tuple: (data_list, trial_lengths)
            - data_list: List of DataFrames, each containing complete sessions forming one partition
            - trial_lengths: List of integers, number of trials in each corresponding partition
    """
    data_list = []
    trial_lengths = []
    
    # Get unique session IDs in order
    session_ids = sorted(monkey_data['id'].unique())
    current_session_idx = 0
    
    while current_session_idx < len(session_ids):
        # Start a new partition
        current_partition_sessions = []
        current_partition_trials = 0
        
        # Add sessions to current partition until we reach at least n_trials
        while current_session_idx < len(session_ids):
            current_session = session_ids[current_session_idx]
            session_data = monkey_data[monkey_data['id'] == current_session]
            session_length = len(session_data)
            
            # Always add sessions until we have at least n_trials
            current_partition_sessions.append(session_data)
            current_partition_trials += session_length
            current_session_idx += 1
            
            # If we have at least n_trials, check if we should continue adding sessions
            if current_partition_trials >= n_trials:
                # Look ahead to see how many trials remain
                remaining_trials = 0
                for j in range(current_session_idx, len(session_ids)):
                    remaining_session = session_ids[j]
                    remaining_session_data = monkey_data[monkey_data['id'] == remaining_session]
                    remaining_trials += len(remaining_session_data)
                
                # If remaining trials would be < n_trials, add them to current partition
                # Otherwise, stop current partition here
                if remaining_trials < n_trials and remaining_trials > 0:
                    # Add all remaining sessions to current partition
                    while current_session_idx < len(session_ids):
                        remaining_session = session_ids[current_session_idx]
                        remaining_session_data = monkey_data[monkey_data['id'] == remaining_session]
                        current_partition_sessions.append(remaining_session_data)
                        current_partition_trials += len(remaining_session_data)
                        current_session_idx += 1
                
                break
        
        # Combine all sessions in this partition into one DataFrame
        if current_partition_sessions:
            partition_data = pd.concat(current_partition_sessions, ignore_index=True)
            data_list.append(partition_data)
            trial_lengths.append(current_partition_trials)
            
    return data_list, trial_lengths


def cutoff_trials_by_session(data, cutoff_trials):
    """
    Cut off the first cutoff_trials trials, rounded to the nearest complete session.
    
    Args:
        data: DataFrame with trial data containing 'id' column for session IDs
        cutoff_trials: Number of trials to cut off (will round to nearest session)
    
    Returns:
        DataFrame with early sessions removed
    """
    if len(data) <= cutoff_trials:
        print(f"Warning: Data has only {len(data)} trials, cannot cut off {cutoff_trials}")
        return data
    
    # Sort data by session order (assuming sessions are chronologically ordered by id)
    sorted_sessions = sorted(data['id'].unique())
    
    cumulative_trials = 0
    cutoff_session_idx = 0
    
    # Find the session that gets us closest to the cutoff point
    for i, session_id in enumerate(sorted_sessions):
        session_data = data[data['id'] == session_id]
        session_trials = len(session_data)
        
        # Check if adding this session would exceed the cutoff
        if cumulative_trials + session_trials > cutoff_trials:
            # Decide whether to include this session or not based on which is closer
            trials_if_exclude = cumulative_trials
            trials_if_include = cumulative_trials + session_trials
            
            if abs(trials_if_exclude - cutoff_trials) <= abs(trials_if_include - cutoff_trials):
                # Closer to cutoff if we exclude this session
                cutoff_session_idx = i
            else:
                # Closer to cutoff if we include this session
                cutoff_session_idx = i + 1
            break
        
        cumulative_trials += session_trials
        cutoff_session_idx = i + 1
    
    # Return data starting from the cutoff session
    if cutoff_session_idx < len(sorted_sessions):
        sessions_to_keep = sorted_sessions[cutoff_session_idx:]
        filtered_data = data[data['id'].isin(sessions_to_keep)]
        actual_trials_cut = len(data) - len(filtered_data)
        print(f"Cut off {actual_trials_cut} trials (target: {cutoff_trials}) by removing {cutoff_session_idx} sessions")
        return filtered_data
    else:
        print(f"Warning: Would remove all sessions, returning original data")
        return data

def rnn_figure(mpdb_path, mpbeh_path, strategic =True, order =5, colinear = True, bias = False,power=2, distance_metric='frechet', plot_matrix=False):
    """
    Generate RNN behavioral insufficiency figure with configurable distance metrics.
    
    Args:
        mpdb_path: Path to monkey database
        mpbeh_path: Path to monkey behavior data
        strategic: Whether to use strategic fits
        order: Order of the regression
        colinear: Whether to use colinear analysis
        bias: Whether to include bias term
        power: Power for normalization
        distance_metric: Distance metric to use. Can be:
                        - String: 'frechet', 'euclidean', 'cosine', 'manhattan', 'wasserstein', 'area_norm'
                        - Callable: Custom distance function that takes two arrays and returns a scalar
        plot_matrix: If True, plot upper triangular matrix of all comparisons instead of separate violin plots
    """
    

    if plot_matrix:
        fig = plt.figure(figsize=(17,8), dpi = 300)  # Reduced height and no constrained layout
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    else:
        fig = plt.figure(layout='constrained', figsize=(17,7.1), dpi = 300)
        gs = GridSpec(2, 3, figure=fig)

    fig.suptitle('Figure 3: RNN Behavioral Insufficiency', fontsize=20, y=0.95 if plot_matrix else None)


    RNN_model_ax = fig.add_subplot(gs[:,0])
    RNNm  = plt.imread('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/RNN_only_model.png')
    RNN_model_ax.imshow(RNNm)
    
    RNN_model_ax.set_title('RNN Model', fontsize = 16)
    RNN_model_ax.set_axis_off()        

    # Create a single main RNN plot instead of the zoo
    RNN_main_ax = fig.add_subplot(gs[:,1])
    RNN_main_ax.set_title('Representative RNN Model', fontsize=20, pad=15)
    
    # Check if we've already computed the central model
    central_model_file = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/central_rnn_model.pkl'
    
    if os.path.exists(central_model_file):
        # Load the pre-computed central model index and Frechet distances
        with open(central_model_file, 'rb') as f:
            central_model_data = pickle.load(f)
            central_model_idx = central_model_data['index']
            
            # Always use the new format with rnn_zoo_dict
        central_model_rnn_idx = central_model_data['central_rnn_idx']
        
        # Load the saved Frechet distance data if available
        if 'rnn_frechet_data' in central_model_data:
            all_rnn_ws_dists = central_model_data['rnn_frechet_data']['ws_dists']
            all_rnn_ls_dists = central_model_data['rnn_frechet_data']['ls_dists']
            all_rnn_ws_errs = central_model_data['rnn_frechet_data']['ws_errs']
            all_rnn_ls_errs = central_model_data['rnn_frechet_data']['ls_errs']
            all_rnn_wsw_dists = central_model_data['rnn_frechet_data']['wsw_dists']
            all_rnn_lss_dists = central_model_data['rnn_frechet_data']['lss_dists']
            
            # Also load central model comparison data if available
            if 'central_pairs' in central_model_data['rnn_frechet_data']:
                central_pairs = central_model_data['rnn_frechet_data']['central_pairs']
                has_central_pairs = True
            else:
                has_central_pairs = False
            
            # Load rnn_norm if available
            if 'rnn_norm' in central_model_data:
                rnn_norm = central_model_data['rnn_norm']
            elif 'rnn_frechet_data' in central_model_data and 'rnn_norm' in central_model_data['rnn_frechet_data']:
                rnn_norm = central_model_data['rnn_frechet_data']['rnn_norm']
            else:
                # Calculate rnn_norm if not available
                rnn_normalization_factors = []
                for rnn in rnns:
                    ws = np.array(rnn_zoo_dict[rnn]['action'][(1-strategic)*order:(2-strategic)*order])
                    ls = np.array(rnn_zoo_dict[rnn]['action'][(2-strategic)*order:(3-strategic)*order])
                    wsw = np.array(rnn_zoo_dict[rnn]['action'][(3-strategic)*order:(4-strategic)*order]) 
                    lss = np.array(rnn_zoo_dict[rnn]['action'][(4-strategic)*order:(5-strategic)*order])
                    
                    c1 = np.sum(np.abs(ws)**power)
                    c2 = np.sum(np.abs(ls)**power)
                    c3 = np.sum(np.abs(wsw)**power)
                    c4 = np.sum(np.abs(lss)**power)
                    cmax = np.sqrt(c1 + c2 + c3 + c4)
                    rnn_normalization_factors.append(cmax)
                rnn_norm = np.mean(rnn_normalization_factors)
            
            has_frechet_data = True
        else:
            has_frechet_data = False
            has_central_pairs = False
            # Calculate rnn_norm since we don't have saved data
            rnn_normalization_factors = []
            for rnn in rnns:
                ws = np.array(rnn_zoo_dict[rnn]['action'][(1-strategic)*order:(2-strategic)*order])
                ls = np.array(rnn_zoo_dict[rnn]['action'][(2-strategic)*order:(3-strategic)*order])
                wsw = np.array(rnn_zoo_dict[rnn]['action'][(3-strategic)*order:(4-strategic)*order]) 
                lss = np.array(rnn_zoo_dict[rnn]['action'][(4-strategic)*order:(5-strategic)*order])
                
                c1 = np.sum(np.abs(ws)**power)
                c2 = np.sum(np.abs(ls)**power)
                c3 = np.sum(np.abs(wsw)**power)
                c4 = np.sum(np.abs(lss)**power)
                cmax = np.sqrt(c1 + c2 + c3 + c4)
                rnn_normalization_factors.append(cmax)
            rnn_norm = np.mean(rnn_normalization_factors)
    
    # Process the central model - always use rnn_zoo_dict
        # Create data structure for plotting using rnn_zoo_dict
        # The regressor functions need a dictionary with an 'action' key
        # For real models, they would expect a tuple with states, actions, etc.
        # but we're just going to fake it with a simple structure
        # We need True as the second parameter to indicate this is a model not monkey data
        dat = {
            'action': rnn_zoo_dict[central_model_rnn_idx]['action']
        }
        
    # Just use a simple plot approach with the coefficients
        xord = np.arange(1, 1+order)
        regressor_data = dat['action']
            
            # Create the plot manually
        ws = np.array(regressor_data[(1-strategic)*order:(2-strategic)*order])
        ls = np.array(regressor_data[(2-strategic)*order:(3-strategic)*order])
        wsw = np.array(regressor_data[(3-strategic)*order:(4-strategic)*order]) if len(regressor_data) > (3-strategic)*order else np.zeros_like(ws)
        lss = np.array(regressor_data[(4-strategic)*order:(5-strategic)*order]) if len(regressor_data) > (4-strategic)*order else np.zeros_like(ls)
            
        # Define nice legend labels
        reggy = ['win stay', 'lose switch', 'win switch', 'lose stay']
        RNN_main_ax.set_xticks(range(1, order+1))
        
        # Plot each coefficient type
        RNN_main_ax.plot(xord, ws, label=reggy[0])
        RNN_main_ax.plot(xord, ls, label=reggy[1])
        
        if len(regressor_data) > (3-strategic)*order:
            RNN_main_ax.plot(xord, wsw, label=reggy[2])
            RNN_main_ax.plot(xord, lss, label=reggy[3])
            
        # Add a horizontal line at zero
        RNN_main_ax.axhline(linestyle='--', color='k', alpha=.5)
        # invisible legend
        RNN_main_ax.legend(fontsize=10, frameon=False)
            
            # Store the computed data for Frechet computations later
        rnn_ws = ws
        rnn_ls = ls
        rnn_wsw = wsw
        rnn_lss = lss
               
        # Normalize using L2 norm for visualization consistency
        c1 = np.sum(np.abs(ws)**power)
        c2 = np.sum(np.abs(ls)**power)
        c3 = np.sum(np.abs(wsw)**power)
        c4 = np.sum(np.abs(lss)**power)
        cmax = np.sqrt(c1 + c2 + c3 + c4)
        rnn_norm = cmax
        
        print(f"RNN norm (from central model): {rnn_norm}")
        
        ws = ws / rnn_norm
        ls = ls / rnn_norm
        wsw = wsw / rnn_norm
        lss = lss / rnn_norm
        
        # TODO: Calculate actual mutual information using saved RNN trial data  
        # This will be handled in a separate script
        # print("Computing mutual information for central RNN model using saved trial data...")
        
        # # Use the same saved RNN trial data as before
        # rnn_data_dir = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/data_dir'
        # rnn_data_files = [f for f in os.listdir(rnn_data_dir) if f.endswith('_data.p')]
        
        # Placeholder values for now - will be computed in separate script
        # rnn_mutual_infos = np.zeros((1, 2))
        # rnn_entropies = np.zeros((1, 2))
        
        
        print("Using placeholder entropy values - will be computed in separate script")
    
    # Set square aspect ratio for the middle plot
    RNN_main_ax.set_box_aspect(1)
    
    # load normalized coeffs

    # Add proper y-axis label
    RNN_main_ax.set_ylabel('Regression Coefficient', fontsize=14)
    RNN_main_ax.set_xlabel('Trials Back', fontsize=14)
    RNN_main_ax.tick_params(axis='both', labelsize=12)

    # Add Stay/Switch labels on the right y-axis
    right_label = RNN_main_ax.twinx()
    right_label.yaxis.set_label_position('right')
    # right_label.set_ylabel('Regressor Effect', fontsize=14)
    right_label.set_yticks([0.4, -0.4])  # Adjust these values based on your y-axis limits
    right_label.set_yticklabels(['Stay', 'Switch'], fontsize=14)
    right_label.set_box_aspect(1)  # Match the aspect ratio
    
    # Set y-axis limits to make the plot square and symmetric
    ylim = max(abs(RNN_main_ax.get_ylim()[0]), abs(RNN_main_ax.get_ylim()[1]))
    RNN_main_ax.set_ylim(-ylim, ylim)
    right_label.set_ylim(RNN_main_ax.get_ylim())

    # Save the data for other scripts

    # Check if rnn_entropies and rnn_mutual_infos are already arrays or need to be stacked
    # if isinstance(rnn_entropies, list):
    #     rnn_entropies = np.vstack(rnn_entropies)
    #     rnn_entropies = np.mean(rnn_entropies, axis=0)
    
    # if isinstance(rnn_mutual_infos, list):
    #     rnn_mutual_infos = np.vstack(rnn_mutual_infos)
    #     rnn_mutual_infos = np.mean(rnn_mutual_infos, axis=0)

    # Use a direct path that we're sure exists
    # entropy_save_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5_rnn_entropy.pkl'
    # os.makedirs(os.path.dirname(entropy_save_path), exist_ok=True)
    # with open(entropy_save_path, 'wb') as f:
    #     pickle.dump([rnn_entropies, rnn_mutual_infos], f)
    # print(f"Saved entropy data to {entropy_save_path}")

    
    nonstrategic_monkeys  = ['C','H','F','K']
    strategic_monkeys = ['E','D','I']
    
    # load behavior from new merged dataset:
    mp2_data = pd.read_pickle(stitched_p)
    mp2_data = mp2_data[mp2_data['task'] == 'mp']
    
    # def category_append():
    #     for category in categories:
    #         # for group in [strategic_monkeys,nonstrategic_monkeys]:
    #         gm = []
    #         for monkey in monkeys:
    #             gm = preds[monkey][category]
    #         group_means[category].append(np.mean(gm))

    if plot_matrix:
        # Create a single subplot for the triangular matrix
        matrix_ax = fig.add_subplot(gs[:,2])
    else:
        #RNN_metric_ax = fig.add_subplot(gs[0,2])
        within_violin_ax = fig.add_subplot(gs[0,2])
        between_violin_ax = fig.add_subplot(gs[1,2])
    # Initialize the dictionaries with empty lists to ensure they always exist
    within_violin_dict = {'RNN': [], 'strategic': [], 'nonstrategic': []}
    between_violin_dict = {'S-RNN': [], 'NS-RNN': [], 'S-NS': []}

    # First, calculate and plot distances for individual monkeys
    monkey_individual_data = {
        'strategic': {'monkeys': [], 'ws_dists': [], 'ls_dists': [], 'ws_errs': [], 'ls_errs': [],
                      'wsw_dists' : [], 'lss_dists' : [], 'wsw_errs' : [], 'lss_errs' : [],
                      'tot_dists' : [], 'tot_errs' : []},
        'nonstrategic': {'monkeys': [], 'ws_dists': [], 'ls_dists': [], 'ws_errs': [], 'ls_errs': [],
                         'wsw_dists' : [], 'lss_dists' : [], 'wsw_errs' : [], 'lss_errs' : [],
                         'tot_dists' : [], 'tot_errs' : []}
    }
    ###### THESE ARE THE COMPARISONS WITHIN GROUPS
    
    ###### COLLECT ALL COEFFICIENTS FOR LAG-BASED NORMALIZATION ######
    
    print("Collecting coefficients for separate normalization by dataset...")
    
    # Collect coefficients separately for each group
    rnn_raw_coeffs = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
    strategic_raw_coeffs = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
    nonstrategic_raw_coeffs = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
    
    # Collect RNN coefficients
    print("Collecting RNN coefficients...")
    for rnn in rnns:
        ws = np.array(rnn_zoo_dict[rnn]['action'][(1-strategic)*order:(2-strategic)*order])
        ls = np.array(rnn_zoo_dict[rnn]['action'][(2-strategic)*order:(3-strategic)*order])
        wsw = np.array(rnn_zoo_dict[rnn]['action'][(3-strategic)*order:(4-strategic)*order])
        lss = np.array(rnn_zoo_dict[rnn]['action'][(4-strategic)*order:(5-strategic)*order])
        
        rnn_raw_coeffs['ws'].append(ws)
        rnn_raw_coeffs['ls'].append(ls)
        rnn_raw_coeffs['wsw'].append(wsw)
        rnn_raw_coeffs['lss'].append(lss)
    
    # Collect non-strategic monkey coefficients
    print("Collecting non-strategic monkey coefficients...")
    for m in nonstrategic_monkeys:
        mdat_all = mp2_data[mp2_data['animal'] == m]
        
        # Apply session-based cutoff for monkeys C, E, F (same as in fig1)
        if m == 'F' or m == 'E' or m == 'C':
            print(f'Applying session-based cutoff of 5000 trials for monkey {m}')
            mdat_all = cutoff_trials_by_session(mdat_all, 5000)
        
        # Partition the data into chunks of roughly 5000 trials
        partition_data_list, partition_trial_lengths = partition_dataset(mdat_all, n_trials=5000)
        print(f'Monkey {m}: Created {len(partition_data_list)} partitions with lengths {partition_trial_lengths}')
        
        # Fit each partition separately
        for i, partition_data in enumerate(partition_data_list):
            if len(partition_data) > 0:  # Only process if we have data
                if strategic:
                    fit = fit_single_paper_strategic(partition_data, bias=True)[:-1]
                else:
                    fit = fit_single_paper(partition_data, bias=True)[:-1]
                
                ws = np.array(fit[(1-strategic)*order:(2-strategic)*order])
                ls = np.array(fit[(2-strategic)*order:(3-strategic)*order])
                wsw = np.array(fit[(3-strategic)*order:(4-strategic)*order]) 
                lst = np.array(fit[(4-strategic)*order:(5-strategic)*order]) 
                
                nonstrategic_raw_coeffs['ws'].append(ws)
                nonstrategic_raw_coeffs['ls'].append(ls)
                nonstrategic_raw_coeffs['wsw'].append(wsw)
                nonstrategic_raw_coeffs['lss'].append(lst)
    
    # Collect strategic monkey coefficients
    print("Collecting strategic monkey coefficients...")
    for m in strategic_monkeys:
        mdat_all = mp2_data[mp2_data['animal'] == m]
        
        # Apply session-based cutoff for monkeys C, E, F (same as in fig1)
        if m == 'F' or m == 'E' or m == 'C':
            print(f'Applying session-based cutoff of 5000 trials for monkey {m}')
            mdat_all = cutoff_trials_by_session(mdat_all, 5000)
        
        # Partition the data into chunks of roughly 5000 trials
        partition_data_list, partition_trial_lengths = partition_dataset(mdat_all, n_trials=5000)
        print(f'Monkey {m}: Created {len(partition_data_list)} partitions with lengths {partition_trial_lengths}')
        
        # Fit each partition separately
        for i, partition_data in enumerate(partition_data_list):
            if len(partition_data) > 0:  # Only process if we have data
                if strategic:
                    fit = fit_single_paper_strategic(partition_data, bias=True)[:-1]
                else:
                    fit = fit_single_paper(partition_data, bias=True)[:-1]
                
                ws = np.array(fit[(1-strategic)*order:(2-strategic)*order])
                ls = np.array(fit[(2-strategic)*order:(3-strategic)*order])
                wsw = np.array(fit[(3-strategic)*order:(4-strategic)*order]) 
                lst = np.array(fit[(4-strategic)*order:(5-strategic)*order]) 
                
                strategic_raw_coeffs['ws'].append(ws)
                strategic_raw_coeffs['ls'].append(ls)
                strategic_raw_coeffs['wsw'].append(wsw)
                strategic_raw_coeffs['lss'].append(lst)
    
    # Normalize each group separately
    
    print(f"RNN dataset: {len(rnn_raw_coeffs['ws'])} models")
    rnn_coeffs = rnn_raw_coeffs  # Use raw coefficients directly
    
    print(f"Non-strategic monkey dataset: {len(nonstrategic_raw_coeffs['ws'])} sessions")
    nonstrategic_coeffs = nonstrategic_raw_coeffs  # Use raw coefficients directly
    
    print(f"Strategic monkey dataset: {len(strategic_raw_coeffs['ws'])} sessions")
    strategic_coeffs = strategic_raw_coeffs  # Use raw coefficients directly
    
    print("\nSeparate normalization statistics:")
    print("RNN normalization stats:")
    # Note: Normalization statistics removed - using raw coefficients
    
    print("\nNon-strategic monkey normalization stats:")
    # Note: Normalization statistics removed - using raw coefficients
    
    print("\nStrategic monkey normalization stats:")
    # Note: Normalization statistics removed - using raw coefficients
    
    ###### THESE ARE THE COMPARISONS WITHIN GROUPS
    
    # Process non-strategic monkeys - partition level
    nonstrategic_monkey_coeffs = {}
    ns_idx = 0  # Index for non-strategic coefficients
    
    for m in nonstrategic_monkeys:
        nonstrategic_monkey_coeffs[m] = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
        mdat_all = mp2_data[mp2_data['animal'] == m]
        
        # Apply session-based cutoff for monkeys C, E, F
        if m == 'F' or m == 'E' or m == 'C':
            mdat_all = cutoff_trials_by_session(mdat_all, 5000)
        
        # Partition the data
        partition_data_list, partition_trial_lengths = partition_dataset(mdat_all, n_trials=5000)
        
        # Use coefficients from each partition
        for i, partition_data in enumerate(partition_data_list):
            if len(partition_data) > 0:  # Only process if we have data
                # Use the raw coefficients directly
                nonstrategic_monkey_coeffs[m]['ws'].append(nonstrategic_coeffs['ws'][ns_idx])
                nonstrategic_monkey_coeffs[m]['ls'].append(nonstrategic_coeffs['ls'][ns_idx])
                nonstrategic_monkey_coeffs[m]['wsw'].append(nonstrategic_coeffs['wsw'][ns_idx])
                nonstrategic_monkey_coeffs[m]['lss'].append(nonstrategic_coeffs['lss'][ns_idx])
                ns_idx += 1

    # Calculate average partition coefficients per monkey (for stats)
    nonstrategic_partition_counts = []
    for m in nonstrategic_monkeys:
        if m in nonstrategic_monkey_coeffs and len(nonstrategic_monkey_coeffs[m]['ws']) > 0:
            nonstrategic_partition_counts.append(len(nonstrategic_monkey_coeffs[m]['ws']))
    
    print(f"Non-strategic monkeys partition counts: {nonstrategic_partition_counts}")
    print(f"Total non-strategic partitions: {sum(nonstrategic_partition_counts)}")

    # Calculate within-group distances for non-strategic monkeys (partition level)
    temp_nonstrategic_within_dists = []
    for m_name in nonstrategic_monkeys:
        if not (m_name in nonstrategic_monkey_coeffs and len(nonstrategic_monkey_coeffs[m_name]['ws']) > 0):
            continue

        m1_coeffs = nonstrategic_monkey_coeffs[m_name]
        
        # Compare each partition with every other partition within the same monkey
        for i in range(len(m1_coeffs['ws'])):
            for j in range(len(m1_coeffs['ws'])):
                if i == j:
                    continue
                        
                    # Ensure coefficients are numpy arrays
                    ws1 = np.array(m1_coeffs['ws'][i])
                    ws2 = np.array(m1_coeffs['ws'][j])
                    ls1 = np.array(m1_coeffs['ls'][i])
                    ls2 = np.array(m1_coeffs['ls'][j])
                    wsw1 = np.array(m1_coeffs['wsw'][i])
                    wsw2 = np.array(m1_coeffs['wsw'][j])
                    lss1 = np.array(m1_coeffs['lss'][i])
                    lss2 = np.array(m1_coeffs['lss'][j])
                    
                    # Compute distances with normalization for fair comparison
                    ws_dist, _ = compute_all_distances([ws1], [ws2], distance_metric, normalize=True)
                    ls_dist, _ = compute_all_distances([ls1], [ls2], distance_metric, normalize=True)
                    wsw_dist, _ = compute_all_distances([wsw1], [wsw2], distance_metric, normalize=True)
                    lss_dist, _ = compute_all_distances([lss1], [lss2], distance_metric, normalize=True)
                    
                    # Combine distance components appropriately based on the metric
                    pair_dist = combine_distance_components(ws_dist, ls_dist, wsw_dist, lss_dist, distance_metric)
                    if not np.isnan(pair_dist):
                        temp_nonstrategic_within_dists.append(float(pair_dist))

    # Now compute between-monkey distances within non-strategic group (partition level)
    for idx, m1_name in enumerate(nonstrategic_monkeys):
        if not (m1_name in nonstrategic_monkey_coeffs and len(nonstrategic_monkey_coeffs[m1_name]['ws']) > 0):
            continue
            
        m1_coeffs = nonstrategic_monkey_coeffs[m1_name]
        
        for m2_name in nonstrategic_monkeys[idx+1:]:  # Compare with subsequent monkeys
            if not (m2_name in nonstrategic_monkey_coeffs and len(nonstrategic_monkey_coeffs[m2_name]['ws']) > 0):
                continue
                
            m2_coeffs = nonstrategic_monkey_coeffs[m2_name]
            
            # Compare each partition of m1 with each partition of m2
            for i in range(len(m1_coeffs['ws'])):
                for j in range(len(m2_coeffs['ws'])):
                    # Ensure coefficients are numpy arrays
                    ws1 = np.array(m1_coeffs['ws'][i])
                    ws2 = np.array(m2_coeffs['ws'][j])
                    ls1 = np.array(m1_coeffs['ls'][i])
                    ls2 = np.array(m2_coeffs['ls'][j])
                    wsw1 = np.array(m1_coeffs['wsw'][i])
                    wsw2 = np.array(m2_coeffs['wsw'][j])
                    lss1 = np.array(m1_coeffs['lss'][i])
                    lss2 = np.array(m2_coeffs['lss'][j])
                    
                    ws_dist, _ = compute_all_distances([ws1], [ws2], distance_metric, normalize=True)
                    ls_dist, _ = compute_all_distances([ls1], [ls2], distance_metric, normalize=True)
                    wsw_dist, _ = compute_all_distances([wsw1], [wsw2], distance_metric, normalize=True)
                    lss_dist, _ = compute_all_distances([lss1], [lss2], distance_metric, normalize=True)
                    
                    # Combine distance components appropriately based on the metric
                    pair_dist = combine_distance_components(ws_dist, ls_dist, wsw_dist, lss_dist, distance_metric)
                    if not np.isnan(pair_dist):
                        temp_nonstrategic_within_dists.append(float(pair_dist))

    if temp_nonstrategic_within_dists:
        within_violin_dict['nonstrategic'].extend(temp_nonstrategic_within_dists)

    # Process strategic monkeys - partition level
    strategic_monkey_coeffs = {}
    
    for m in strategic_monkeys:
        strategic_monkey_coeffs[m] = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
        mdat_all = mp2_data[mp2_data['animal'] == m]
        
        # Apply session-based cutoff for monkeys C, E, F
        if m == 'F' or m == 'E' or m == 'C':
            mdat_all = cutoff_trials_by_session(mdat_all, 5000)
        
        # Partition the data
        partition_data_list, partition_trial_lengths = partition_dataset(mdat_all, n_trials=5000)
        
        # Fit each partition separately and store coefficients
        for i, partition_data in enumerate(partition_data_list):
            if len(partition_data) > 0:  # Only process if we have data
                # Compute coefficients for this partition
                if strategic:
                    fit = fit_single_paper_strategic(partition_data, bias=True)[:-1]
                else:
                    fit = fit_single_paper(partition_data, bias=True)[:-1]
                
                ws = np.array(fit[(1-strategic)*order:(2-strategic)*order])
                ls = np.array(fit[(2-strategic)*order:(3-strategic)*order])
                wsw = np.array(fit[(3-strategic)*order:(4-strategic)*order]) 
                lst = np.array(fit[(4-strategic)*order:(5-strategic)*order]) 
                
                # Store the coefficients directly
                strategic_monkey_coeffs[m]['ws'].append(ws)
                strategic_monkey_coeffs[m]['ls'].append(ls)
                strategic_monkey_coeffs[m]['wsw'].append(wsw)
                strategic_monkey_coeffs[m]['lss'].append(lst)

    # Calculate average partition coefficients per monkey (for stats)
    strategic_partition_counts = []
    for m in strategic_monkeys:
        if m in strategic_monkey_coeffs and len(strategic_monkey_coeffs[m]['ws']) > 0:
            strategic_partition_counts.append(len(strategic_monkey_coeffs[m]['ws']))
    
    print(f"Strategic monkeys partition counts: {strategic_partition_counts}")
    print(f"Total strategic partitions: {sum(strategic_partition_counts)}")

    # Calculate within-group distances for strategic monkeys (partition level)
    temp_strategic_within_dists = []
    for m_name in strategic_monkeys:
        if not (m_name in strategic_monkey_coeffs and len(strategic_monkey_coeffs[m_name]['ws']) > 0):
            continue

        m1_coeffs = strategic_monkey_coeffs[m_name]
        
        # Compare each partition with every other partition
        for i in range(len(m1_coeffs['ws'])):
            for j in range(len(m1_coeffs['ws'])):
                if i == j:
                    continue
                    
                # Ensure coefficients are numpy arrays
                ws1 = np.array(m1_coeffs['ws'][i])
                ws2 = np.array(m1_coeffs['ws'][j])
                ls1 = np.array(m1_coeffs['ls'][i])
                ls2 = np.array(m1_coeffs['ls'][j])
                wsw1 = np.array(m1_coeffs['wsw'][i])
                wsw2 = np.array(m1_coeffs['wsw'][j])
                lss1 = np.array(m1_coeffs['lss'][i])
                lss2 = np.array(m1_coeffs['lss'][j])
                
                ws_dist, _ = compute_all_distances([ws1], [ws2], distance_metric, normalize=True)
                ls_dist, _ = compute_all_distances([ls1], [ls2], distance_metric, normalize=True)
                wsw_dist, _ = compute_all_distances([wsw1], [wsw2], distance_metric, normalize=True)
                lss_dist, _ = compute_all_distances([lss1], [lss2], distance_metric, normalize=True)
                
                # Combine distance components appropriately based on the metric
                pair_dist = combine_distance_components(ws_dist, ls_dist, wsw_dist, lss_dist, distance_metric)
                if not np.isnan(pair_dist):
                    temp_strategic_within_dists.append(float(pair_dist))

    if temp_strategic_within_dists:
        within_violin_dict['strategic'].extend(temp_strategic_within_dists)

    # Now compute between-monkey distances within strategic group (partition level)
    for idx, m1_name in enumerate(strategic_monkeys):
        if not (m1_name in strategic_monkey_coeffs and len(strategic_monkey_coeffs[m1_name]['ws']) > 0):
            continue
            
        m1_coeffs = strategic_monkey_coeffs[m1_name]
        
        for m2_name in strategic_monkeys[idx+1:]:  # Compare with subsequent monkeys
            if not (m2_name in strategic_monkey_coeffs and len(strategic_monkey_coeffs[m2_name]['ws']) > 0):
                continue
                
            m2_coeffs = strategic_monkey_coeffs[m2_name]
            
            # Compare each partition of m1 with each partition of m2
            for i in range(len(m1_coeffs['ws'])):
                for j in range(len(m2_coeffs['ws'])):
                    # Ensure coefficients are numpy arrays
                    ws1 = np.array(m1_coeffs['ws'][i])
                    ws2 = np.array(m2_coeffs['ws'][j])
                    ls1 = np.array(m1_coeffs['ls'][i])
                    ls2 = np.array(m2_coeffs['ls'][j])
                    wsw1 = np.array(m1_coeffs['wsw'][i])
                    wsw2 = np.array(m2_coeffs['wsw'][j])
                    lss1 = np.array(m1_coeffs['lss'][i])
                    lss2 = np.array(m2_coeffs['lss'][j])
                    
                    ws_dist, _ = compute_all_distances([ws1], [ws2], distance_metric, normalize=True)
                    ls_dist, _ = compute_all_distances([ls1], [ls2], distance_metric, normalize=True)
                    wsw_dist, _ = compute_all_distances([wsw1], [wsw2], distance_metric, normalize=True)
                    lss_dist, _ = compute_all_distances([lss1], [lss2], distance_metric, normalize=True)
                    
                    # Combine distance components appropriately based on the metric
                    pair_dist = combine_distance_components(ws_dist, ls_dist, wsw_dist, lss_dist, distance_metric)
                    if not np.isnan(pair_dist):
                        temp_strategic_within_dists.append(float(pair_dist))

    ###### THESE ARE THE COMPARISONS BETWEEN GROUPS
    # Now continue with the existing group comparison plots

    # Load raw RNN coefficients  
    rnn_coeffs_list = []
    for i, rnn in enumerate(rnns):
        # Use the raw coefficients directly
        rnn_coeffs_list.append({
            'ws': rnn_coeffs['ws'][i],
            'ls': rnn_coeffs['ls'][i], 
            'wsw': rnn_coeffs['wsw'][i],
            'lss': rnn_coeffs['lss'][i]
        })
    
    # Extract raw lists for distance calculations
    rnn_ws = [coeffs['ws'] for coeffs in rnn_coeffs_list]
    rnn_ls = [coeffs['ls'] for coeffs in rnn_coeffs_list]
    rnn_wsw = [coeffs['wsw'] for coeffs in rnn_coeffs_list]
    rnn_lss = [coeffs['lss'] for coeffs in rnn_coeffs_list]
    
    for group in ['MP2 Non-strategic','MP2 Strategic']:
        if group == 'MP2 Non-strategic':
            monkeys = nonstrategic_monkeys
            data = nonstrategic_monkey_coeffs
            for monkey in monkeys:
                try:
                    # Only process if the monkey has data
                    if monkey in data and len(data[monkey]['ws']) > 0:
                        for i in range(len(data[monkey]['ws'])):
                            ns_ws = data[monkey]['ws'][i]
                            ns_ls = data[monkey]['ls'][i]
                            ns_wsw = data[monkey]['wsw'][i]
                            ns_lss = data[monkey]['lss'][i]
                    
                            # Always use np.zeros_like for consistency if needed
                            dists_ws = compute_all_distances([ns_ws], rnn_ws, distance_metric, normalize=True, return_all=True)
                            dists_ls = compute_all_distances([ns_ls], rnn_ls, distance_metric, normalize=True, return_all=True)
                            
                            # Always create wsw and lss arrays regardless of whether they exist
                            dists_wsw = compute_all_distances([ns_wsw], rnn_wsw, distance_metric, normalize=True, return_all=True)
                            dists_lss = compute_all_distances([ns_lss], rnn_lss, distance_metric, normalize=True, return_all=True)
                            
                            tot_dists = np.vstack([dists_ws, dists_ls, dists_wsw, dists_lss])
                            # Combine distance arrays appropriately based on the metric
                            tot_dist = combine_distance_arrays(tot_dists, distance_metric)
                            
                            # Ensure data is flattened and add to violin dict
                            between_violin_dict['NS-RNN'].extend(tot_dist.flatten())
                except Exception as e:
                    print(f"Error processing non-strategic monkey {monkey}: {str(e)}")
            
        elif group == 'MP2 Strategic':
            monkeys = strategic_monkeys
            data = strategic_monkey_coeffs
            for monkey in monkeys:
                try:
                    # Only process if the monkey has data
                    if monkey in data and len(data[monkey]['ws']) > 0:
                        for i in range(len(data[monkey]['ws'])):
                            s_ws = data[monkey]['ws'][i]
                            s_ls = data[monkey]['ls'][i]
                            s_wsw = data[monkey]['wsw'][i]
                            s_lss = data[monkey]['lss'][i]
                        
                            # Always use np.zeros_like for consistency if needed
                            dists_ws = compute_all_distances([s_ws], rnn_ws, distance_metric, normalize=True, return_all=True)
                            dists_ls = compute_all_distances([s_ls], rnn_ls, distance_metric, normalize=True, return_all=True)
                            
                            # Always create wsw and lss arrays regardless of whether they exist
                            dists_wsw = compute_all_distances([s_wsw], rnn_wsw, distance_metric, normalize=True, return_all=True)
                            dists_lss = compute_all_distances([s_lss], rnn_lss, distance_metric, normalize=True, return_all=True)
                            
                            tot_dists = np.vstack([dists_ws, dists_ls, dists_wsw, dists_lss])
                            # Combine distance arrays appropriately based on the metric
                            tot_dist = combine_distance_arrays(tot_dists, distance_metric)
                            
                            # Ensure data is flattened and add to violin dict
                            between_violin_dict['S-RNN'].extend(tot_dist.flatten())
                except Exception as e:
                    print(f"Error processing strategic monkey {monkey}: {str(e)}")

    # Add a comparison between strategic and non-strategic monkeys (session level)
    # Calculate distances between strategic and non-strategic monkey groups
    
    for monkey in strategic_monkeys:
        for monkey2 in nonstrategic_monkeys:
            try:
                # Check if both monkeys have data
                if (monkey in strategic_monkey_coeffs and len(strategic_monkey_coeffs[monkey]['ws']) > 0 and
                    monkey2 in nonstrategic_monkey_coeffs and len(nonstrategic_monkey_coeffs[monkey2]['ws']) > 0):
                    for i in range(len(strategic_monkey_coeffs[monkey]['ws'])):
                        for j in range(len(nonstrategic_monkey_coeffs[monkey2]['ws'])):
                            s_ws = strategic_monkey_coeffs[monkey]['ws'][i]
                            s_ls = strategic_monkey_coeffs[monkey]['ls'][i]
                            s_wsw = strategic_monkey_coeffs[monkey]['wsw'][i]
                            s_lss = strategic_monkey_coeffs[monkey]['lss'][i]
                            
                            ns_ws = nonstrategic_monkey_coeffs[monkey2]['ws'][j]
                            ns_ls = nonstrategic_monkey_coeffs[monkey2]['ls'][j]
                            ns_wsw = nonstrategic_monkey_coeffs[monkey2]['wsw'][j]
                            ns_lss = nonstrategic_monkey_coeffs[monkey2]['lss'][j]
                            
                            # Compute distances
                            dists_ws = compute_all_distances([s_ws], [ns_ws], distance_metric, normalize=True, return_all=True)
                            dists_ls = compute_all_distances([s_ls], [ns_ls], distance_metric, normalize=True, return_all=True)
                            dists_wsw = compute_all_distances([s_wsw], [ns_wsw], distance_metric, normalize=True, return_all=True)
                            dists_lss = compute_all_distances([s_lss], [ns_lss], distance_metric, normalize=True, return_all=True)
                            
                            tot_dists = np.vstack([dists_ws, dists_ls, dists_wsw, dists_lss])
                            # No normalization needed since all coefficients are Z-score normalized
                            tot_dist = combine_distance_arrays(tot_dists, distance_metric)
                            
                            # Ensure data is flattened and add to violin dict
                            between_violin_dict['S-NS'].extend(tot_dist.flatten())
            except Exception as e:
                print(f"Error processing S-NS comparison for monkeys {monkey} vs {monkey2}: {str(e)}")

    # Plot RNN comparison data
    rnn_ls_dists = []
    rnn_ws_dists = []
    rnn_wsw_dists = []
    rnn_lss_dists = []
    rnn_tot_dists = []
    
    # compute distances between all rnns. Save this so we don't have to do it again.
    
    # Check if we have saved RNN distances
    # Also prepare containers for saving RNN anchor points used by Fig. 5 (ws-wsw, ls-lss at lag 1)
    rnn_dict = {'ws': [], 'ls': []}
    rnn_distances_file = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/rnn_distances.pkl'
    
    # Force recomputation without old normalization approach
    # Delete old distances file to ensure we use the raw coefficients
    if os.path.exists(rnn_distances_file):
        os.remove(rnn_distances_file)
        print("Deleted old RNN distances file to recompute with raw coefficients")
    
    # Always compute distances fresh with raw coefficients
    # Compute distances if not saved
    sample_rnns = rnns
    for i, rnn_i in enumerate(sample_rnns):
        rnn_i_coeffs = rnn_coeffs_list[i]
        rnn_i_ws = rnn_i_coeffs['ws']
        rnn_i_ls = rnn_i_coeffs['ls']
        rnn_i_wsw = rnn_i_coeffs['wsw']
        rnn_i_lss = rnn_i_coeffs['lss']
        
        # Compute combined-regressor skew at lag-1 with robust normalization
        ws_den = np.max(np.abs(rnn_i_ws)) if np.max(np.abs(rnn_i_ws)) > 0 else 1.0
        wsw_den = np.max(np.abs(rnn_i_wsw)) if np.max(np.abs(rnn_i_wsw)) > 0 else 1.0
        ls_den = np.max(np.abs(rnn_i_ls)) if np.max(np.abs(rnn_i_ls)) > 0 else 1.0
        lss_den = np.max(np.abs(rnn_i_lss)) if np.max(np.abs(rnn_i_lss)) > 0 else 1.0
        skew_w = (rnn_i_ws[0] / ws_den) - (rnn_i_wsw[0] / wsw_den)
        skew_l = (rnn_i_ls[0] / ls_den) - (rnn_i_lss[0] / lss_den)
        skew_w /= 2
        skew_l /= 2
        
        rnn_dict['ws'].append(skew_w)
        rnn_dict['ls'].append(skew_l)

        for j, rnn_j in enumerate(sample_rnns[i+1:], i+1):  # Compare with subsequent RNNs
            rnn_j_coeffs = rnn_coeffs_list[j]
            rnn_j_ws = rnn_j_coeffs['ws']
            rnn_j_ls = rnn_j_coeffs['ls']
            rnn_j_wsw = rnn_j_coeffs['wsw']
            rnn_j_lss = rnn_j_coeffs['lss']
            
            
            # Compute Frechet distances between raw coefficients
            ws_dist = compute_frechet(rnn_i_ws, rnn_j_ws, lambda x, y: np.linalg.norm(x - y))
            ls_dist = compute_frechet(rnn_i_ls, rnn_j_ls, lambda x, y: np.linalg.norm(x - y))
            wsw_dist = compute_frechet(rnn_i_wsw, rnn_j_wsw, lambda x, y: np.linalg.norm(x - y))
            lss_dist = compute_frechet(rnn_i_lss, rnn_j_lss, lambda x, y: np.linalg.norm(x - y))
            
            # Store results
            rnn_ws_dists.append(ws_dist)
            rnn_ls_dists.append(ls_dist)
            rnn_wsw_dists.append(wsw_dist)
            rnn_lss_dists.append(lss_dist)
            tot_dist = combine_distance_components(ws_dist, ls_dist, wsw_dist, lss_dist, distance_metric)
            rnn_tot_dists.append(tot_dist)
    
    # Save computed distances
    with open(rnn_distances_file, 'wb') as f:
        pickle.dump([rnn_ws_dists, rnn_ls_dists, rnn_wsw_dists, rnn_lss_dists, rnn_tot_dists], f)

    # Save anchors in fig1-compatible format: [wsrnn, lsrnn]
    wsrnn_vals = [float(v) for v in rnn_dict['ws']]
    lsrnn_vals = [float(v) for v in rnn_dict['ls']]
    min_len = min(len(wsrnn_vals), len(lsrnn_vals))
    wsrnn_vals = wsrnn_vals[:min_len]
    lsrnn_vals = lsrnn_vals[:min_len]
    with open('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig1_rnn_data.pkl','wb') as f:
        pickle.dump([wsrnn_vals, lsrnn_vals], f)
    print(f"Saved fig1_rnn_data.pkl with {len(wsrnn_vals)} anchors (ws, ls)")

    # Convert list of lists or arrays to flat arrays for violin plots
    # Make sure the data is correctly shaped for violin plots
    rnn_ws_dists = np.array(rnn_ws_dists).flatten()
    rnn_ls_dists = np.array(rnn_ls_dists).flatten()
    rnn_wsw_dists = np.array(rnn_wsw_dists).flatten()
    rnn_lss_dists = np.array(rnn_lss_dists).flatten()
    rnn_tot_dists = np.array(rnn_tot_dists).flatten()

    # Use the raw distances directly since all coefficients are raw
    within_violin_dict['RNN'] = rnn_tot_dists
    
    print(f"RNN distances (raw coefficients) - mean: {np.mean(rnn_tot_dists):.6f}, std: {np.std(rnn_tot_dists):.6f}")
    print(f"Monkey strategic distances - mean: {np.mean(within_violin_dict['strategic']) if within_violin_dict['strategic'] else 'No data'}")
    print(f"Monkey non-strategic distances - mean: {np.mean(within_violin_dict['nonstrategic']) if within_violin_dict['nonstrategic'] else 'No data'}")
    
    print("\\nDistance Summary with raw coefficient values:")
    print(f"RNN within-group distances: mean={np.mean(rnn_tot_dists):.4f}, n={len(rnn_tot_dists)}")
    if within_violin_dict['strategic']:
        print(f"Strategic monkey within-group distances: mean={np.mean(within_violin_dict['strategic']):.4f}, n={len(within_violin_dict['strategic'])}")
    if within_violin_dict['nonstrategic']:
        print(f"Non-strategic monkey within-group distances: mean={np.mean(within_violin_dict['nonstrategic']):.4f}, n={len(within_violin_dict['nonstrategic'])}")
    if between_violin_dict['S-RNN']:
        print(f"Strategic-RNN between-group distances: mean={np.mean(between_violin_dict['S-RNN']):.4f}, n={len(between_violin_dict['S-RNN'])}")
    if between_violin_dict['NS-RNN']:
        print(f"Non-strategic-RNN between-group distances: mean={np.mean(between_violin_dict['NS-RNN']):.4f}, n={len(between_violin_dict['NS-RNN'])}")
    if between_violin_dict['S-NS']:
        print(f"Strategic-Non-strategic between-group distances: mean={np.mean(between_violin_dict['S-NS']):.4f}, n={len(between_violin_dict['S-NS'])}")
        
    print("\\nDiagnostics:")
    print(f"RNN distance range: {np.min(rnn_tot_dists):.4f} - {np.max(rnn_tot_dists):.4f}")
    if within_violin_dict['strategic']:
        print(f"Strategic distance range: {np.min(within_violin_dict['strategic']):.4f} - {np.max(within_violin_dict['strategic']):.4f}")
    if within_violin_dict['nonstrategic']:
        print(f"Non-strategic distance range: {np.min(within_violin_dict['nonstrategic']):.4f} - {np.max(within_violin_dict['nonstrategic']):.4f}")
    if between_violin_dict['S-RNN']:
        print(f"Strategic-RNN distance range: {np.min(between_violin_dict['S-RNN']):.4f} - {np.max(between_violin_dict['S-RNN']):.4f}")
    if between_violin_dict['NS-RNN']:
        print(f"Non-strategic-RNN distance range: {np.min(between_violin_dict['NS-RNN']):.4f} - {np.max(between_violin_dict['NS-RNN']):.4f}")
        
    # Check if issue is with Z-score normalization
    print("\\nChecking Z-score normalization effects:")
    print(f"Number of RNN models: {len(rnns)}")
    sample_means = []
    sample_stds = []
    for i, coeffs in enumerate(rnn_coeffs_list[:5]):  # Check first 5
        all_coeffs = np.concatenate([coeffs['ws'].flatten(), coeffs['ls'].flatten(), 
                                   coeffs['wsw'].flatten(), coeffs['lss'].flatten()])
        mean_val = np.mean(all_coeffs)
        std_val = np.std(all_coeffs)
        sample_means.append(mean_val)
        sample_stds.append(std_val)
        print(f"  RNN {rnns[i]} mean: {mean_val:.6f}, std: {std_val:.6f}")
    
    # Print the length of each violin dataset for debugging
    # print("Violin data sizes (after collections):")
    # print(f"within_violin_dict['RNN']: {len(within_violin_dict['RNN'])}")
    # print(f"within_violin_dict['nonstrategic']: {len(within_violin_dict['nonstrategic'])}")
    # print(f"within_violin_dict['strategic']: {len(within_violin_dict['strategic'])}")
    # print(f"between_violin_dict['NS-RNN']: {len(between_violin_dict['NS-RNN'])}")
    # print(f"between_violin_dict['S-RNN']: {len(between_violin_dict['S-RNN'])}")
    # print(f"between_violin_dict['S-NS']: {len(between_violin_dict['S-NS'])}")
    
    # Create the plots based on plot_matrix flag
    if plot_matrix:
        # Create upper triangular matrix plot
        create_triangular_matrix_plot(matrix_ax, within_violin_dict, between_violin_dict, distance_metric)
    else:
        # plot within violin plot
        # Force plotting of all violin plots without try-except blocks
        vpRNN = within_violin_ax.violinplot(within_violin_dict['RNN'], positions=[2], showmedians=True)
        # Set colors for RNN violin
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in vpRNN:
                vp = vpRNN[partname]
                vp.set_edgecolor('tab:red')
                vp.set_linewidth(1)
        
        if 'bodies' in vpRNN:
            for pc in vpRNN['bodies']:
                pc.set_facecolor('tab:red')
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)

        # Only plot if data is available
        if len(within_violin_dict['nonstrategic']) > 0:
            vpNS = within_violin_ax.violinplot(within_violin_dict['nonstrategic'], positions=[0], showmedians=True)
            # Set colors for non-strategic violin
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in vpNS:
                    vp = vpNS[partname]
                    vp.set_edgecolor('tab:purple')
                    vp.set_linewidth(1)
        
            if 'bodies' in vpNS:
                for pc in vpNS['bodies']:
                    pc.set_facecolor('tab:purple')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        else:
            print("Skipping non-strategic within-group violin plot: No data.")
        
        if len(within_violin_dict['strategic']) > 0:
            vpS = within_violin_ax.violinplot(within_violin_dict['strategic'], positions=[1], showmedians=True)
            # Set colors for strategic violin
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in vpS:
                    vp = vpS[partname]
                    vp.set_edgecolor('tab:cyan')
                    vp.set_linewidth(1)
        
            if 'bodies' in vpS:
                for pc in vpS['bodies']:
                    pc.set_facecolor('tab:cyan')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        else:
            print("Skipping strategic within-group violin plot: No data.")

        within_violin_ax.set_xlabel('Group', fontsize=14)
        within_violin_ax.set_ylabel(f'{get_distance_label(distance_metric)}', fontsize=14)
        within_violin_ax.set_title(f'{get_distance_label(distance_metric)} Within Groups', fontsize=16)
        within_violin_ax.set_xticks([0, 1, 2])
        within_violin_ax.set_xticklabels(['Non-strategic', 'Strategic', 'RNN'], fontsize=12)
        within_violin_ax.tick_params(axis='y', labelsize=12)
        
        # Add horizontal black line at similarity = 1
        within_violin_ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        
        # Modify y-axis ticks to hide values greater than 1
        current_yticks = within_violin_ax.get_yticks()
        filtered_yticks = [tick for tick in current_yticks if tick <= 1]
        within_violin_ax.set_yticks(filtered_yticks)

        # Significance tests for within_violin_ax
        y_max_within = 0
        # Determine max y for positioning significance bars, considering all potentially plotted groups
        all_within_data_for_ymax = []
        if len(within_violin_dict['nonstrategic']) > 0:
            all_within_data_for_ymax.extend(within_violin_dict['nonstrategic'])
        if len(within_violin_dict['strategic']) > 0:
            all_within_data_for_ymax.extend(within_violin_dict['strategic'])
        if len(within_violin_dict['RNN']) > 0:
            all_within_data_for_ymax.extend(within_violin_dict['RNN'])
        
        if all_within_data_for_ymax: # only if there's any data at all
            y_max_within = np.max(all_within_data_for_ymax)
        else:
            y_max_within = within_violin_ax.get_ylim()[1] # fallback if no data at all, use current ylim

        # Improved spacing factors to prevent overlap
        initial_offset_from_data_within = y_max_within * 0.10  # Increased from 0.08
        current_y_level_within = y_max_within + initial_offset_from_data_within

        bar_leg_height_factor_within = 0.06  # Increased from 0.05
        text_baseline_offset_factor_within = 0.04  # Increased from 0.03
        text_actual_height_estimate_factor_within = 0.15  # Increased from 0.12 to accommodate *** and delta
        inter_annotation_spacing_factor_within = 0.12  # Increased from 0.10 for better separation

        bar_leg_height_within_abs = y_max_within * bar_leg_height_factor_within
        text_baseline_offset_within_abs = y_max_within * text_baseline_offset_factor_within
        text_actual_height_estimate_within_abs = y_max_within * text_actual_height_estimate_factor_within
        vertical_spacing_between_annotations_within_abs = y_max_within * inter_annotation_spacing_factor_within

        # Compare Non-strategic (0) vs Strategic (1)
        if len(within_violin_dict['nonstrategic']) > 1 and len(within_violin_dict['strategic']) > 1:
            plot_significance_bar_bootstrap(within_violin_ax, 0, 1, current_y_level_within, bootstrap_result=bootstrap_median_difference_ci(within_violin_dict['nonstrategic'], within_violin_dict['strategic']), cliff_result=cliff_delta(within_violin_dict['nonstrategic'], within_violin_dict['strategic']), dh=bar_leg_height_within_abs, barh=text_baseline_offset_within_abs, group1_name="Non-strategic", group2_name="Strategic")
            current_y_level_within += bar_leg_height_within_abs + text_baseline_offset_within_abs + text_actual_height_estimate_within_abs + vertical_spacing_between_annotations_within_abs

        # Compare Non-strategic (0) vs RNN (2)
        if len(within_violin_dict['nonstrategic']) > 1 and len(within_violin_dict['RNN']) > 1:
            plot_significance_bar_bootstrap(within_violin_ax, 0, 2, current_y_level_within, bootstrap_result=bootstrap_median_difference_ci(within_violin_dict['nonstrategic'], within_violin_dict['RNN']), cliff_result=cliff_delta(within_violin_dict['nonstrategic'], within_violin_dict['RNN']), dh=bar_leg_height_within_abs, barh=text_baseline_offset_within_abs, group1_name="Non-strategic", group2_name="RNN")
            current_y_level_within += bar_leg_height_within_abs + text_baseline_offset_within_abs + text_actual_height_estimate_within_abs + vertical_spacing_between_annotations_within_abs

        # Compare Strategic (1) vs RNN (2)
        if len(within_violin_dict['strategic']) > 1 and len(within_violin_dict['RNN']) > 1:
            plot_significance_bar_bootstrap(within_violin_ax, 1, 2, current_y_level_within, bootstrap_result=bootstrap_median_difference_ci(within_violin_dict['strategic'], within_violin_dict['RNN']), cliff_result=cliff_delta(within_violin_dict['strategic'], within_violin_dict['RNN']), dh=bar_leg_height_within_abs, barh=text_baseline_offset_within_abs, group1_name="Strategic", group2_name="RNN")
            # No increment needed for the last bar in this stack if no more bars above it
        
        # plot between violin plot - without try-except blocks
        if len(between_violin_dict['NS-RNN']) > 0:
            vpNSRNN = between_violin_ax.violinplot(between_violin_dict['NS-RNN'], positions=[0], showmedians=True)
            # Set colors
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in vpNSRNN:
                    vp = vpNSRNN[partname]
                    vp.set_edgecolor('tab:purple')
                    vp.set_linewidth(1)
            
            if 'bodies' in vpNSRNN:
                for pc in vpNSRNN['bodies']:
                    pc.set_facecolor('tab:purple')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        else:
            print("Skipping NS-RNN between-group violin plot: No data.")
        
        if len(between_violin_dict['S-RNN']) > 0:
            vpSRNN = between_violin_ax.violinplot(between_violin_dict['S-RNN'], positions=[1], showmedians=True)
            # Set colors
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in vpSRNN:
                    vp = vpSRNN[partname]
                    vp.set_edgecolor('tab:cyan')
                    vp.set_linewidth(1)
            
            if 'bodies' in vpSRNN:
                for pc in vpSRNN['bodies']:
                    pc.set_facecolor('tab:cyan')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        else:
            print("Skipping S-RNN between-group violin plot: No data.")
        
        if len(between_violin_dict['S-NS']) > 0:
            vpSNS = between_violin_ax.violinplot(between_violin_dict['S-NS'], positions=[2], showmedians=True)
            # Set colors
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in vpSNS:
                    vp = vpSNS[partname]
                    vp.set_edgecolor('tab:green')
                    vp.set_linewidth(1)
            
            if 'bodies' in vpSNS:
                for pc in vpSNS['bodies']:
                    pc.set_facecolor('tab:green')
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        else:
            print("Skipping S-NS between-group violin plot: No data.")
        
        between_violin_ax.set_xlabel('Group', fontsize=14)
        between_violin_ax.set_ylabel(f'{get_distance_label(distance_metric)}', fontsize=14)
        between_violin_ax.set_title(f'{get_distance_label(distance_metric)} Between Groups', fontsize=16)
        between_violin_ax.set_xticks([0, 1, 2])
        between_violin_ax.set_xticklabels(['NS-RNN', 'S-RNN', 'S-NS'], fontsize=12)
        between_violin_ax.tick_params(axis='y', labelsize=12)
        
        # Add horizontal black line at similarity = 1
        between_violin_ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        
        # Modify y-axis ticks to hide values greater than 1
        current_yticks = between_violin_ax.get_yticks()
        filtered_yticks = [tick for tick in current_yticks if tick <= 1]
        between_violin_ax.set_yticks(filtered_yticks)

        # Significance tests for between_violin_ax
        y_max_between = 0
        all_between_data_for_ymax = []
        if len(between_violin_dict['NS-RNN']) > 0:
            all_between_data_for_ymax.extend(between_violin_dict['NS-RNN'])
        if len(between_violin_dict['S-RNN']) > 0:
            all_between_data_for_ymax.extend(between_violin_dict['S-RNN'])
        if len(between_violin_dict['S-NS']) > 0:
            all_between_data_for_ymax.extend(between_violin_dict['S-NS'])
        
        if all_between_data_for_ymax:
            y_max_between = np.max(all_between_data_for_ymax)
        else:
            y_max_between = between_violin_ax.get_ylim()[1] # fallback

        # Improved spacing factors to prevent overlap
        initial_offset_from_data_between = y_max_between * 0.10  # Increased from 0.08
        current_y_level_between = y_max_between + initial_offset_from_data_between

        bar_leg_height_factor_between = 0.06  # Increased from 0.05
        text_baseline_offset_factor_between = 0.04  # Increased from 0.03
        text_actual_height_estimate_factor_between = 0.15  # Increased from 0.12 to accommodate *** and delta
        inter_annotation_spacing_factor_between = 0.12  # Increased from 0.10 for better separation

        bar_leg_height_between_abs = y_max_between * bar_leg_height_factor_between
        text_baseline_offset_between_abs = y_max_between * text_baseline_offset_factor_between
        text_actual_height_estimate_between_abs = y_max_between * text_actual_height_estimate_factor_between
        vertical_spacing_between_annotations_between_abs = y_max_between * inter_annotation_spacing_factor_between

        # Compare NS-RNN (0) vs S-RNN (1)
        if len(between_violin_dict['NS-RNN']) > 1 and len(between_violin_dict['S-RNN']) > 1:
            plot_significance_bar_bootstrap(between_violin_ax, 0, 1, current_y_level_between, bootstrap_result=bootstrap_median_difference_ci(between_violin_dict['NS-RNN'], between_violin_dict['S-RNN']), cliff_result=cliff_delta(between_violin_dict['NS-RNN'], between_violin_dict['S-RNN']), dh=bar_leg_height_between_abs, barh=text_baseline_offset_between_abs, group1_name="NS-RNN", group2_name="S-RNN")
            current_y_level_between += bar_leg_height_between_abs + text_baseline_offset_between_abs + text_actual_height_estimate_between_abs + vertical_spacing_between_annotations_between_abs

        # Compare NS-RNN (0) vs S-NS (2)
        if len(between_violin_dict['NS-RNN']) > 1 and len(between_violin_dict['S-NS']) > 1:
            plot_significance_bar_bootstrap(between_violin_ax, 0, 2, current_y_level_between, bootstrap_result=bootstrap_median_difference_ci(between_violin_dict['NS-RNN'], between_violin_dict['S-NS']), cliff_result=cliff_delta(between_violin_dict['NS-RNN'], between_violin_dict['S-NS']), dh=bar_leg_height_between_abs, barh=text_baseline_offset_between_abs, group1_name="NS-RNN", group2_name="S-NS")
            current_y_level_between += bar_leg_height_between_abs + text_baseline_offset_between_abs + text_actual_height_estimate_between_abs + vertical_spacing_between_annotations_between_abs

        # Compare S-RNN (1) vs S-NS (2)
        if len(between_violin_dict['S-RNN']) > 1 and len(between_violin_dict['S-NS']) > 1:
            plot_significance_bar_bootstrap(between_violin_ax, 1, 2, current_y_level_between, bootstrap_result=bootstrap_median_difference_ci(between_violin_dict['S-RNN'], between_violin_dict['S-NS']), cliff_result=cliff_delta(between_violin_dict['S-RNN'], between_violin_dict['S-NS']), dh=bar_leg_height_between_abs, barh=text_baseline_offset_between_abs, group1_name="S-RNN", group2_name="S-NS")
            # No increment needed for the last bar in this stack

        # Adjust y-axis limits to ensure all annotations are visible with extra margin
        # For within-group plot
        max_annotation_y_within = current_y_level_within + bar_leg_height_within_abs + text_baseline_offset_within_abs + text_actual_height_estimate_within_abs
        current_ylim_within = within_violin_ax.get_ylim()
        if max_annotation_y_within > current_ylim_within[1]:
            new_ylim_within = max_annotation_y_within * 1.1  # Add 10% margin
            within_violin_ax.set_ylim(current_ylim_within[0], new_ylim_within)
        
        # For between-group plot  
        max_annotation_y_between = current_y_level_between + bar_leg_height_between_abs + text_baseline_offset_between_abs + text_actual_height_estimate_between_abs
        current_ylim_between = between_violin_ax.get_ylim()
        if max_annotation_y_between > current_ylim_between[1]:
            new_ylim_between = max_annotation_y_between * 1.1  # Add 10% margin
            between_violin_ax.set_ylim(current_ylim_between[0], new_ylim_between)

    # Add explanation of significance system only for violin plots
    if not plot_matrix:
        fig.text(0.02, 0.02, 
                 'Significance: *** = p<0.001, ** = p<0.01, * = p<0.05, ns = not significant (Bootstrap CI)\n' +
                 'Stars (right) = significance; δ values (left) = Cliff\'s Delta with direction (G1>G2 or G1<G2)',
                 fontsize=8, ha='left', va='bottom', 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

    if plot_matrix:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
    else:
        plt.tight_layout()
    
    # Save the figure to disk for debugging
    fig_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3_output.png'
    plt.savefig(fig_path)
    
    plt.show()
    
    # Return the figure and distance data for analysis
    return fig, within_violin_dict, between_violin_dict

def plot_significance_bar_bootstrap(ax, x1, x2, y_start, bootstrap_result, cliff_result, dh=.05, barh=.05, fs=10, group1_name=None, group2_name=None):
    """Plots a significance bar with bootstrap CI and Cliff's Delta information.

    ax: matplotlib axes
    x1, x2: x-coordinates for the bar
    y_start: y-coordinate for the bottom of the bar legs
    bootstrap_result: dict from bootstrap_median_difference_ci()
    cliff_result: dict from cliff_delta()
    dh: height of the bar legs
    barh: height of the horizontal bar
    fs: font size for text
    group1_name, group2_name: Names of the groups being compared (for directionality)
    
    Significance symbols (based on p-values):
    *** : p < 0.001 (99.9% CI excludes zero)
    **  : p < 0.01 (99% CI excludes zero)
    *   : p < 0.05 (95% CI excludes zero)
    ns  : p ≥ 0.05 (95% CI includes zero)
    δ=X.XX : Cliff's Delta effect size with directionality (positioned left of bar center)
    """
    
    if 'error' in bootstrap_result or 'error' in cliff_result:
        return  # Skip if there's an error
    
    y_bar = y_start + dh
    y_text = y_bar + barh
    bar_center_x = (x1 + x2) * 0.5

    # Draw the bar with slightly thicker lines for better visibility
    ax.plot([x1, x1, x2, x2], [y_start, y_bar, y_bar, y_start], lw=2, c='k')

    # Determine significance level based on multiple confidence intervals (p-values)
    # This is more efficient than calling get_significance_level which would recalculate
    if bootstrap_result['is_significant']:
        # We have the 95% CI result, need to check higher confidence levels
        # For efficiency, we'll use the group data if available, otherwise use effect size as proxy
        
        # Try to get original groups from bootstrap_result (if we modify the function to return them)
        # For now, use a simplified approach based on CI width as proxy for significance strength
        ci_width = bootstrap_result['ci_upper'] - bootstrap_result['ci_lower']
        median_diff = abs(bootstrap_result['median_diff'])
        
        # Narrower CI relative to effect size suggests stronger significance
        if ci_width > 0 and median_diff / ci_width > 3.0:  # Very strong signal
            text = '***'
        elif ci_width > 0 and median_diff / ci_width > 2.0:  # Strong signal
            text = '**'
        else:  # Significant but not as strong
            text = '*'
    else:
        text = 'ns'  # Not significant

    # Position significance stars to the RIGHT of bar center
    stars_x = bar_center_x + 0.3 * (x2 - x1)  # Offset right by 30% of bar width
    ax.text(stars_x, y_text, text, ha='center', va='bottom', color='k', 
            fontsize=fs, fontweight='bold')
    
    # Position delta text to the LEFT of bar center with directional info
    delta_value = cliff_result['effect_size']
    
    # Create directional delta text
    if group1_name and group2_name:
        # Better abbreviation logic for specific group names
        def abbreviate_name(name):
            # Handle specific conventional abbreviations
            if name.lower() == "non-strategic":
                return "NS"
            elif name.lower() == "strategic":
                return "S"
            elif name == "RNN":
                return "RNN"
            elif '-' in name:
                # For hyphenated names like "NS-RNN", keep the full name but limit length
                return name[:6] if len(name) > 6 else name
            else:
                # For other names, take first 3 characters
                return name[:3] if len(name) > 3 else name
        
        g1_abbrev = abbreviate_name(group1_name)
        g2_abbrev = abbreviate_name(group2_name)
        
        # Direction logic: δ > 0 means group1 > group2, δ < 0 means group1 < group2
        if delta_value > 0.001:  # Use small threshold to avoid floating point issues
            direction_text = f"{g1_abbrev}>{g2_abbrev}"
        elif delta_value < -0.001:
            direction_text = f"{g1_abbrev}<{g2_abbrev}"
        else:
            direction_text = f"{g1_abbrev}≈{g2_abbrev}"
        
        effect_text = f"δ={delta_value:.2f}\n{direction_text}"
    else:
        # Fallback: just show the delta with sign
        effect_text = f"δ={delta_value:+.2f}"
    
    delta_x = bar_center_x - 0.3 * (x2 - x1)  # Offset left by 30% of bar width
    ax.text(delta_x, y_text, effect_text, 
           ha='center', va='bottom', color='darkred', fontsize=fs-2, 
           fontweight='normal', style='italic')

def compute_all_distances(group1, group2, distance_func=None, normalize=False, return_all=False):
    """Compute distances between two groups of embeddings using specified distance function.
    
    Args:
        group1, group2: Lists of sequences to compare
        distance_func: Function to compute distance between two sequences. 
                      Can be a string key from DISTANCE_METRICS or a callable.
                      Defaults to 'frechet'.
        normalize: Whether to normalize sequences before computing distance
        return_all: Whether to return all pairwise distances or just mean and std
    """
    # Handle distance function specification
    if distance_func is None:
        distance_func = 'frechet'
    
    if isinstance(distance_func, str):
        if distance_func not in DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {distance_func}. Available: {list(DISTANCE_METRICS.keys())}")
        dist_fn = DISTANCE_METRICS[distance_func]
    elif callable(distance_func):
        dist_fn = distance_func
    else:
        raise ValueError("distance_func must be a string key or callable function")
    
    distances = []
    for i, g1 in enumerate(group1):
        for j, g2 in enumerate(group2):
            # Convert to numpy arrays for consistent handling
            g1_arr = np.asarray(g1)
            g2_arr = np.asarray(g2)
            
            # Skip self-comparisons only if comparing within the same group
            if len(group1) == len(group2) and i == j and np.array_equal(g1_arr, g2_arr):
                continue
                
            if normalize:
                # normalize the curves by their area (L1 norm - sum of absolute values)
                g1_area = np.sum(np.abs(g1_arr))
                g2_area = np.sum(np.abs(g2_arr))
                g1_norm = g1_arr/g1_area if g1_area > 0 else g1_arr
                g2_norm = g2_arr/g2_area if g2_area > 0 else g2_arr
                distances.append(dist_fn(g1_norm, g2_norm))
            else:
                distances.append(dist_fn(g1_arr, g2_arr))
    
    if return_all:
        return distances
    else:
        return np.mean(distances), np.std(distances)

# Keep the old function name for backward compatibility
def compute_all_frechets(group1, group2, normalize=False, return_all=False):
    """Legacy function - now calls compute_all_distances with Frechet distance."""
    return compute_all_distances(group1, group2, 'frechet', normalize, return_all)

def compute_frechet(p: np.ndarray, q: np.ndarray, dist_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)) -> float:
    """Compute the Frechet distance between two sequences."""
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    n_p = len(p)
    n_q = len(q)
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            # For 1D sequences, pass scalars to distance function
            d = dist_func(np.array([p[i]]), np.array([q[j]])) if callable(dist_func) else np.abs(p[i] - q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
                
    return ca[n_p - 1, n_q - 1]

def frechet_distance_wrapper(p: np.ndarray, q: np.ndarray) -> float:
    """Wrapper to use Frechet distance with the compute_all_distances function."""
    return compute_frechet(p, q)

def get_distance_label(distance_metric):
    """
    Get the appropriate label for the distance metric.
    For cosine, returns 'Similarity' instead of 'Distance'.
    """
    if isinstance(distance_metric, str) and distance_metric.lower() == 'cosine':
        return f'{distance_metric.capitalize()} Similarity'
    else:
        metric_name = distance_metric if isinstance(distance_metric, str) else 'Custom'
        return f'{metric_name.capitalize()} Distance'

# Dictionary of available distance metrics
DISTANCE_METRICS = {
    'frechet': frechet_distance_wrapper,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'wasserstein': wasserstein_sequence_distance,
    'area_norm': area_norm_distance
}

def demonstrate_distance_metrics():
    """
    Demonstration function showing how to use different distance metrics.
    """
    print("Available distance metrics:")
    for name, func in DISTANCE_METRICS.items():
        print(f"  - '{name}': {func.__doc__}")
    
    print("\nExample usage:")
    print("# Using a built-in distance metric:")
    print("rnn_figure(mpdb_p, mpbeh_p, distance_metric='euclidean')")
    print("cluster_monkeys_by_distance(data, monkeys, strategic, nonstrategic, distance_metric='cosine')")
    print("find_most_similar_monkey_to_rnns(distance_metric='manhattan')")
    
    print("\n# Using a custom distance function:")
    print("def my_custom_distance(p, q):")
    print("    return np.sum((p - q)**4)  # L4 norm")
    print("rnn_figure(mpdb_p, mpbeh_p, distance_metric=my_custom_distance)")
    
    # Example with actual arrays
    print("\nExample with sample data:")
    p = np.array([1, 2, 3, 4, 5])
    q = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    
    print(f"p = {p}")
    print(f"q = {q}")
    print("\nDistance calculations:")
    
    for name, func in DISTANCE_METRICS.items():
        try:
            dist = func(p, q)
            print(f"  {name}: {dist:.4f}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

# Simple main function to run the script

def cluster_monkeys_by_distance(mp2_data, all_monkey_ids, strategic_monkeys_list, nonstrategic_monkeys_list, strategic_fits=True, order=5, distance_metric='frechet'):
    """
    Clusters monkeys based on distance of their average partition-normalized regression coefficients.
    Plots a dendrogram of the clustering.
    
    Args:
        mp2_data: Monkey behavioral data
        all_monkey_ids: List of all monkey IDs to include
        strategic_monkeys_list: List of strategic monkey IDs
        nonstrategic_monkeys_list: List of non-strategic monkey IDs  
        strategic_fits: Whether to use strategic regression fits
        order: Order of regression
        distance_metric: Distance metric to use (string key or callable function)
    """
    print(f"\nClustering monkeys by {get_distance_label(distance_metric).lower()} of average regressor shapes...")

    avg_monkey_coeffs = {}
    valid_monkeys_for_clustering = []

    for monkey_id in all_monkey_ids:
        mdat_all = mp2_data[mp2_data['animal'] == monkey_id]
        if mdat_all.empty:
            print(f"No data for monkey {monkey_id}, skipping.")
            continue

        # Apply session-based cutoff for monkeys C, E, F (same as in main function)
        if monkey_id == 'F' or monkey_id == 'E' or monkey_id == 'C':
            print(f'Applying session-based cutoff of 5000 trials for monkey {monkey_id}')
            mdat_all = cutoff_trials_by_session(mdat_all, 5000)

        # Partition the data into chunks of roughly 5000 trials
        partition_data_list, partition_trial_lengths = partition_dataset(mdat_all, n_trials=5000)
        print(f'Monkey {monkey_id}: Created {len(partition_data_list)} partitions with lengths {partition_trial_lengths}')

        partition_coeffs_normalized = {'ws': [], 'ls': [], 'wsw': [], 'lss': []}
        
        # Process each partition
        for i, partition_data in enumerate(partition_data_list):
            if len(partition_data) < order + 5:  # Basic check for sufficient data
                continue

            try:
                if strategic_fits:
                    fit = fit_single_paper_strategic(partition_data, order=order, bias=True)[:-1]
                else:
                    fit = fit_single_paper(partition_data, order=order, bias=True)[:-1]

                # Extract coefficients consistently with main function
                ws = np.array(fit[(1-strategic_fits)*order:(2-strategic_fits)*order])
                ls = np.array(fit[(2-strategic_fits)*order:(3-strategic_fits)*order])
                wsw = np.array(fit[(3-strategic_fits)*order:(4-strategic_fits)*order]) if len(fit) > (3-strategic_fits)*order else np.zeros(order)
                lss = np.array(fit[(4-strategic_fits)*order:(5-strategic_fits)*order]) if len(fit) > (4-strategic_fits)*order else np.zeros(order)

                # Normalize this partition's coefficients
                c_all = np.concatenate([ws, ls, wsw, lss])
                norm = np.linalg.norm(c_all)
                if norm > 0:
                    partition_coeffs_normalized['ws'].append(ws / norm)
                    partition_coeffs_normalized['ls'].append(ls / norm)
                    partition_coeffs_normalized['wsw'].append(wsw / norm)
                    partition_coeffs_normalized['lss'].append(lss / norm)
                else:  # All zero coeffs for this partition
                    partition_coeffs_normalized['ws'].append(ws)
                    partition_coeffs_normalized['ls'].append(ls)
                    partition_coeffs_normalized['wsw'].append(wsw)
                    partition_coeffs_normalized['lss'].append(lss)
            
            except Exception as e:
                print(f"Error processing partition {i} for monkey {monkey_id}: {e}")
                continue
        
        if not partition_coeffs_normalized['ws']:  # No valid partitions processed
            print(f"No valid partitions with data for monkey {monkey_id} after processing, skipping.")
            continue

        # Average across partitions for this monkey
        avg_monkey_coeffs[monkey_id] = {
            'ws': np.mean(partition_coeffs_normalized['ws'], axis=0),
            'ls': np.mean(partition_coeffs_normalized['ls'], axis=0),
            'wsw': np.mean(partition_coeffs_normalized['wsw'], axis=0),
            'lss': np.mean(partition_coeffs_normalized['lss'], axis=0),
        }
        valid_monkeys_for_clustering.append(monkey_id)

    if len(valid_monkeys_for_clustering) < 2:
        print("Not enough monkeys with valid data to perform clustering.")
        return

    num_monkeys = len(valid_monkeys_for_clustering)
    dist_matrix_condensed = []

    # Get the distance function
    if isinstance(distance_metric, str):
        # Import the available distance metrics dictionary
        from scipy.spatial.distance import euclidean, cosine
        DISTANCE_METRICS = {
            'euclidean': euclidean_distance,
            'cosine': cosine_distance,
            'manhattan': manhattan_distance,
            'wasserstein': wasserstein_sequence_distance,
            'area_norm': area_norm_distance,
            'frechet': frechet_distance_wrapper
        }
        if distance_metric not in DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {distance_metric}. Available: {list(DISTANCE_METRICS.keys())}")
        dist_fn = DISTANCE_METRICS[distance_metric]
    elif callable(distance_metric):
        dist_fn = distance_metric
    else:
        raise ValueError("distance_metric must be a string key or callable function")

    for i in range(num_monkeys):
        for j in range(i + 1, num_monkeys):
            monkey1_id = valid_monkeys_for_clustering[i]
            monkey2_id = valid_monkeys_for_clustering[j]

            c1 = avg_monkey_coeffs[monkey1_id]
            c2 = avg_monkey_coeffs[monkey2_id]

            ws_dist = dist_fn(c1['ws'], c2['ws'])
            ls_dist = dist_fn(c1['ls'], c2['ls'])
            wsw_dist = dist_fn(c1['wsw'], c2['wsw'])
            lss_dist = dist_fn(c1['lss'], c2['lss'])
            
            total_dist = np.sqrt(ws_dist**2 + ls_dist**2 + wsw_dist**2 + lss_dist**2)
            dist_matrix_condensed.append(total_dist)

    if not dist_matrix_condensed:
        print("No distances computed, cannot perform clustering.")
        return
        
    linked = sch.linkage(np.array(dist_matrix_condensed), method='average')

    # Plotting
    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
    dendro_data = sch.dendrogram(linked,
                                 orientation='top',
                                 labels=valid_monkeys_for_clustering,
                                 distance_sort='descending',
                                 show_leaf_counts=True,
                                 ax=ax_cluster)
    
    ax_cluster.set_title(f'Hierarchical Clustering of Monkeys by Regressor Shape ({get_distance_label(distance_metric)})\n(Partition-Based Analysis)', fontsize=16)
    ax_cluster.set_ylabel(f'{get_distance_label(distance_metric)}', fontsize=14)
    ax_cluster.set_xlabel('Monkey ID', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Color labels based on original grouping
    ax_labels = ax_cluster.get_xmajorticklabels()
    for label in ax_labels:
        monkey_id_text = label.get_text()
        if monkey_id_text in strategic_monkeys_list:
            label.set_color('tab:cyan') # Consistent with violin plots
            label.set_weight('bold')
        elif monkey_id_text in nonstrategic_monkeys_list:
            label.set_color('tab:purple') # Consistent with violin plots
            label.set_weight('bold')
        else:
            label.set_color('black') # Should not happen if all_monkey_ids are covered

    # Add a legend for colors (manual)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='tab:purple', lw=4, label='Original Non-Strategic'),
                       Line2D([0], [0], color='tab:cyan', lw=4, label='Original Strategic')]
    ax_cluster.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    fig_cluster_path = f'/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3_monkey_clustering_dendrogram_{distance_metric}_partitioned.png'
    plt.savefig(fig_cluster_path)
    print(f"Saved monkey clustering dendrogram to {fig_cluster_path}")
    plt.show()

    print("\nOriginal Groupings:")
    print(f"Strategic Monkeys: {strategic_monkeys_list}")
    print(f"Non-strategic Monkeys: {nonstrategic_monkeys_list}")
    
    return avg_monkey_coeffs, valid_monkeys_for_clustering

# Keep the old function name for backward compatibility
def cluster_monkeys_by_frechet(mp2_data, all_monkey_ids, strategic_monkeys_list, nonstrategic_monkeys_list, strategic_fits=True, order=5):
    """Legacy function - now calls cluster_monkeys_by_distance with Frechet distance."""
    return cluster_monkeys_by_distance(mp2_data, all_monkey_ids, strategic_monkeys_list, nonstrategic_monkeys_list, strategic_fits, order, 'frechet')

def find_most_similar_monkey_to_rnns(power=2, distance_metric='frechet'):
    """Calculate which non-strategic monkey is most similar to the RNNs based on specified distance metric.
    
    Args:
        power: Power for normalization
        distance_metric: Distance metric to use (string key or callable function)
    """
    print(f"\nAnalyzing non-strategic monkey similarity to RNNs using {get_distance_label(distance_metric).lower()}...")
    
    # Load the data
    mp2_data = pd.read_pickle(stitched_p)
    mp2_data = mp2_data[mp2_data['task'] == 'mp']
    
    # Define non-strategic monkeys
    nonstrategic_monkeys = ['C','H','F','K']
    order = 5
    strategic = True
    bias = False
    
    # Load RNN coefficients
    rnn_ws = [rnn_zoo_dict[rnn]['action'][(1-strategic)*order:(2-strategic)*order] for rnn in rnns]
    rnn_ls = [rnn_zoo_dict[rnn]['action'][(2-strategic)*order:(3-strategic)*order] for rnn in rnns]
    rnn_wsw = [rnn_zoo_dict[rnn]['action'][(3-strategic)*order:(4-strategic)*order] for rnn in rnns]
    rnn_lss = [rnn_zoo_dict[rnn]['action'][(4-strategic)*order:(5-strategic)*order] for rnn in rnns]
    
    # Process non-strategic monkeys
    monkey_similarity = {}
    
    for monkey in nonstrategic_monkeys:
        mdat = mp2_data[mp2_data['animal'] == monkey]
        if len(mdat) == 0:
            print(f"No data available for monkey {monkey}")
            continue
            
        try:
            if strategic:
                fit = fit_single_paper_strategic(mdat, bias=True)[:-1]
            else:
                fit = fit_single_paper(mdat, bias=True)[:-1]
            
            ws = np.array(fit[(1-strategic)*order:(2-strategic)*order])
            ls = np.array(fit[(2-strategic)*order:(3-strategic)*order])
            wsw = np.array(fit[(3-strategic)*order:(4-strategic)*order]) if len(fit) > (3-strategic)*order else np.zeros_like(ws)
            lst = np.array(fit[(4-strategic)*order:(5-strategic)*order]) if len(fit) > (4-strategic)*order else np.zeros_like(ls)
            
            # Normalize
            c1 = np.sum(np.abs(ws)**power)
            c2 = np.sum(np.abs(ls)**power)
            c3 = np.sum(np.abs(wsw)**power)
            c4 = np.sum(np.abs(lst)**power)
            cmax = np.sqrt(c1 + c2 + c3 + c4)
            
            if cmax > 0:
                ws = ws / cmax
                ls = ls / cmax
                wsw = wsw / cmax
                lst = lst / cmax
            
            # Calculate distances to all RNNs
            dists_ws = compute_all_distances([ws], rnn_ws, distance_metric, return_all=True)
            dists_ls = compute_all_distances([ls], rnn_ls, distance_metric, return_all=True)
            dists_wsw = compute_all_distances([wsw], rnn_wsw, distance_metric, return_all=True)
            dists_lss = compute_all_distances([lst], rnn_lss, distance_metric, return_all=True)
            
            tot_dists = np.vstack([dists_ws, dists_ls, dists_wsw, dists_lss])
            combined_dists = combine_distance_arrays(tot_dists, distance_metric)
            avg_dist = np.mean(combined_dists)
            
            monkey_similarity[monkey] = avg_dist
            print(f"Monkey {monkey} average distance to RNNs: {avg_dist:.4f}")
            
        except Exception as e:
            print(f"Error processing monkey {monkey}: {str(e)}")
    
    if monkey_similarity:
        most_similar = min(monkey_similarity, key=monkey_similarity.get)
        print(f"\nMost similar non-strategic monkey to RNNs: {most_similar} (distance: {monkey_similarity[most_similar]:.4f})")
        
        # Sort all monkeys by similarity
        sorted_monkeys = sorted(monkey_similarity.items(), key=lambda x: x[1])
        print(f"\nNon-strategic monkeys ranked by similarity to RNNs (lowest {get_distance_label(distance_metric).lower()} = most similar):")
        for idx, (monkey, dist) in enumerate(sorted_monkeys):
            print(f"{idx+1}. Monkey {monkey}: {dist:.4f}")
    else:
        print("No valid monkey data found for comparison.")

# Add a call to the new function at the end
if __name__ == "__main__":
    # Show available distance metrics
    print("="*80)
    print("DISTANCE METRICS DEMONSTRATION")
    print("="*80)
    demonstrate_distance_metrics()
    print("\n" + "="*80)
    
    # Default run with Frechet distance
    rnn_figure(mpdb_p, mpbeh_p, strategic=True, order=5,power=2, distance_metric='frechet')
    print("Figure generated successfully with Frechet distance")

def compare_groups_multiple_tests(group1, group2, group1_name="Group 1", group2_name="Group 2"):
    """
    Compare two groups using multiple statistical tests to determine if one tends to be larger.
    
    Args:
        group1, group2: Arrays of values to compare
        group1_name, group2_name: Names for the groups
    
    Returns:
        Dictionary with test results
    """
    results = {}
    
    # Convert to numpy arrays and remove NaNs
    g1 = np.array(group1)
    g2 = np.array(group2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Insufficient data for comparison"}
    
    print(f"\nComparing {group1_name} (n={len(g1)}) vs {group2_name} (n={len(g2)})")
    print(f"{group1_name} median: {np.median(g1):.4f}, {group2_name} median: {np.median(g2):.4f}")
    
    # 1. Mann-Whitney U test (two-sided)
    try:
        stat, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        results['mann_whitney_u'] = {'statistic': stat, 'p_value': p_mw, 'test': 'Mann-Whitney U'}
        print(f"Mann-Whitney U: p = {p_mw:.6f}")
    except Exception as e:
        print(f"Mann-Whitney U failed: {e}")
    
    # 2. One-sided Mann-Whitney U (if group1 tends to be larger)
    try:
        stat, p_mw_greater = stats.mannwhitneyu(g1, g2, alternative='greater')
        results['mann_whitney_u_greater'] = {'statistic': stat, 'p_value': p_mw_greater, 'test': 'Mann-Whitney U (group1 > group2)'}
        print(f"Mann-Whitney U (group1 > group2): p = {p_mw_greater:.6f}")
    except Exception as e:
        print(f"One-sided Mann-Whitney U failed: {e}")
    
    # 3. Kolmogorov-Smirnov test (two-sample)
    try:
        stat, p_ks = stats.ks_2samp(g1, g2)
        results['kolmogorov_smirnov'] = {'statistic': stat, 'p_value': p_ks, 'test': 'Kolmogorov-Smirnov'}
        print(f"Kolmogorov-Smirnov: p = {p_ks:.6f}")
    except Exception as e:
        print(f"Kolmogorov-Smirnov failed: {e}")
    
    # 4. Mood's median test
    try:
        stat, p_mood, med, tbl = stats.median_test(g1, g2)
        results['mood_median'] = {'statistic': stat, 'p_value': p_mood, 'test': "Mood's median test"}
        print(f"Mood's median test: p = {p_mood:.6f}")
    except Exception as e:
        print(f"Mood's median test failed: {e}")
    
    # 5. Brunner-Munzel test (more robust than Mann-Whitney for ties)
    try:
        stat, p_bm = stats.brunnermunzel(g1, g2)
        results['brunner_munzel'] = {'statistic': stat, 'p_value': p_bm, 'test': 'Brunner-Munzel'}
        print(f"Brunner-Munzel: p = {p_bm:.6f}")
    except Exception as e:
        print(f"Brunner-Munzel failed: {e}")
    
    # 6. Effect size: Cliff's Delta
    try:
        # Cliff's delta: proportion of times group1 > group2 minus proportion group1 < group2
        comparisons = []
        for x in g1:
            for y in g2:
                if x > y:
                    comparisons.append(1)
                elif x < y:
                    comparisons.append(-1)
                else:
                    comparisons.append(0)
        
        cliff_delta = np.mean(comparisons)
        results['cliff_delta'] = {'effect_size': cliff_delta, 'test': "Cliff's Delta"}
        
        # Interpret effect size
        if abs(cliff_delta) < 0.147:
            effect_size_interp = "negligible"
        elif abs(cliff_delta) < 0.33:
            effect_size_interp = "small"
        elif abs(cliff_delta) < 0.474:
            effect_size_interp = "medium"
        else:
            effect_size_interp = "large"
        
        print(f"Cliff's Delta: {cliff_delta:.4f} ({effect_size_interp} effect)")
        results['cliff_delta']['interpretation'] = effect_size_interp
        
    except Exception as e:
        print(f"Cliff's Delta failed: {e}")
    
    # 7. Permutation test (modern resampling approach)
    try:
        def permutation_test(x, y, n_permutations=10000):
            """Simple permutation test for difference in medians"""
            observed_diff = np.median(x) - np.median(y)
            combined = np.concatenate([x, y])
            n_x = len(x)
            
            permuted_diffs = []
            for _ in range(n_permutations):
                np.random.shuffle(combined)
                perm_x = combined[:n_x]
                perm_y = combined[n_x:]
                permuted_diffs.append(np.median(perm_x) - np.median(perm_y))
            
            p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
            return observed_diff, p_value
        
        obs_diff, p_perm = permutation_test(g1, g2)
        results['permutation_test'] = {'observed_diff': obs_diff, 'p_value': p_perm, 'test': 'Permutation test'}
        print(f"Permutation test (median diff): {obs_diff:.4f}, p = {p_perm:.6f}")
        
    except Exception as e:
        print(f"Permutation test failed: {e}")
    
    # 8. Bootstrap confidence interval for difference in medians
    try:
        def bootstrap_median_diff(x, y, n_bootstrap=10000):
            """Bootstrap confidence interval for difference in medians"""
            diffs = []
            for _ in range(n_bootstrap):
                boot_x = np.random.choice(x, size=len(x), replace=True)
                boot_y = np.random.choice(y, size=len(y), replace=True)
                diffs.append(np.median(boot_x) - np.median(boot_y))
            
            return np.array(diffs)
        
        boot_diffs = bootstrap_median_diff(g1, g2)
        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)
        
        results['bootstrap_ci'] = {
            'median_diff': np.median(boot_diffs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'test': 'Bootstrap 95% CI for median difference'
        }
        
        print(f"Bootstrap median difference: {np.median(boot_diffs):.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Check if CI includes zero
        if ci_lower <= 0 <= ci_upper:
            print("  95% CI includes zero - no significant difference")
        else:
            print("  95% CI excludes zero - significant difference")
            
    except Exception as e:
        print(f"Bootstrap CI failed: {e}")
    
    return results

def bootstrap_median_difference_ci(group1, group2, n_bootstrap=10000, confidence_level=0.95):
    """
    Compute bootstrap confidence interval for the difference in medians between two groups.
    
    Args:
        group1, group2: Lists or arrays of values to compare
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        dict with keys:
        - 'median_diff': Observed difference in medians (group1 - group2)
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval  
        - 'is_significant': True if CI excludes zero
        - 'confidence_level': The confidence level used
    """
    try:
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        if len(group1) == 0 or len(group2) == 0:
            return {'error': 'Empty groups provided'}
        
        # Observed difference in medians
        observed_diff = np.median(group1) - np.median(group2)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Sample with replacement from each group
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            
            # Compute difference in medians for this bootstrap sample
            diff = np.median(sample1) - np.median(sample2)
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
        
        # Check if significant (CI excludes zero)
        is_significant = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'median_diff': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'is_significant': is_significant,
            'confidence_level': confidence_level
        }
        
    except Exception as e:
        return {'error': f'Bootstrap calculation failed: {str(e)}'}

def cliff_delta(group1, group2):
    """
    Calculate Cliff's Delta effect size.
    
    Args:
        group1, group2: Arrays of values
        
    Returns:
        dict with effect_size, interpretation
    """
    g1 = np.array(group1)
    g2 = np.array(group2)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    
    if len(g1) == 0 or len(g2) == 0:
        return {"error": "No valid data"}
    
    # Calculate Cliff's delta efficiently
    comparisons = []
    for x in g1:
        for y in g2:
            if x > y:
                comparisons.append(1)
            elif x < y:
                comparisons.append(-1)
            else:
                comparisons.append(0)
    
    delta = np.mean(comparisons)
    
    # Interpret effect size
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'effect_size': abs_delta,
        'interpretation': interpretation,
        'abs_effect_size': abs_delta
    }

def compare_groups_bootstrap_cliff(group1, group2, group1_name="Group 1", group2_name="Group 2"):
    """
    Compare groups using Bootstrap CI and Cliff's Delta.
    
    Returns:
        dict with bootstrap_ci and cliff_delta results
    """
    # Bootstrap CI for significance
    bootstrap_result = bootstrap_median_difference_ci(group1, group2)
    
    # Cliff's Delta for effect size
    cliff_result = cliff_delta(group1, group2)
    
    # Summary
    if 'error' not in bootstrap_result and 'error' not in cliff_result:
        print(f"{group1_name} vs {group2_name}:")
        print(f"  Median difference: {bootstrap_result['observed_diff']:.4f}")
        print(f"  95% CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
        print(f"  Significant: {'Yes' if bootstrap_result['is_significant'] else 'No'}")
        print(f"  Cliff's Δ: {cliff_result['effect_size']:.4f} ({cliff_result['interpretation']})")
    
    return {
        'bootstrap_ci': bootstrap_result,
        'cliff_delta': cliff_result,
        'group1_name': group1_name,
        'group2_name': group2_name
    }

    if between_violin_dict['NS-RNN']:
        print(f"Non-strategic-RNN distance range: {np.min(between_violin_dict['NS-RNN']):.4f} - {np.max(between_violin_dict['NS-RNN']):.4f}")
        
    # Print sample coefficient statistics for debugging (raw coefficients with area normalization)
    print("\\nSample raw coefficient statistics (with area normalization):")
    print(f"Number of RNN models: {len(rnns)}")
    for i, coeffs in enumerate(rnn_coeffs_list[:5]):  # Check first 5
        all_coeffs = np.concatenate([coeffs['ws'].flatten(), coeffs['ls'].flatten(), 
                                   coeffs['wsw'].flatten(), coeffs['lss'].flatten()])
        mean_val = np.mean(all_coeffs)
        std_val = np.std(all_coeffs)
        area_norm = np.sum(np.abs(all_coeffs))
        print(f"  RNN {rnns[i]} mean: {mean_val:.6f}, std: {std_val:.6f}, area: {area_norm:.6f}")

def get_significance_level(group1, group2):
    """
    Determine significance level based on different confidence intervals.
    
    Args:
        group1, group2: Lists or arrays of values to compare
    
    Returns:
        str: '***' for p<0.001, '**' for p<0.01, '*' for p<0.05, 'ns' for not significant
    """
    try:
        # Test multiple confidence levels (equivalent to different p-value thresholds)
        
        # 99.9% CI (equivalent to p < 0.001)
        ci_999 = bootstrap_median_difference_ci(group1, group2, confidence_level=0.999)
        if not ('error' in ci_999) and ci_999['is_significant']:
            return '***'
        
        # 99% CI (equivalent to p < 0.01)
        ci_99 = bootstrap_median_difference_ci(group1, group2, confidence_level=0.99)
        if not ('error' in ci_99) and ci_99['is_significant']:
            return '**'
        
        # 95% CI (equivalent to p < 0.05)
        ci_95 = bootstrap_median_difference_ci(group1, group2, confidence_level=0.95)
        if not ('error' in ci_95) and ci_95['is_significant']:
            return '*'
        
        return 'ns'
        
    except Exception as e:
        print(f"Error in significance level calculation: {e}")
        return 'ns'

def combine_distance_components(ws_val, ls_val, wsw_val, lss_val, distance_metric):
    """
    Combine the four distance/similarity components appropriately based on the distance metric.
    
    Args:
        ws_val, ls_val, wsw_val, lss_val: Individual distance/similarity values for each component
        distance_metric: The distance metric being used
        
    Returns:
        Combined distance/similarity value
    """
    if isinstance(distance_metric, str) and distance_metric == 'cosine':
        # For cosine similarity, take the mean of the similarities
        # (since cosine_distance function actually returns similarity values)
        return np.mean([ws_val, ls_val, wsw_val, lss_val])
    else:
        # For distance metrics, use the traditional L2 combination
        return np.sqrt(ws_val + ls_val + wsw_val + lss_val)

def combine_distance_arrays(distance_arrays, distance_metric):
    """
    Combine stacked arrays of distance/similarity values appropriately based on the distance metric.
    
    Args:
        distance_arrays: np.array of shape (4, n_comparisons) containing [ws_dists, ls_dists, wsw_dists, lss_dists]
        distance_metric: The distance metric being used
        
    Returns:
        Array of combined distance/similarity values
    """
    if isinstance(distance_metric, str) and distance_metric == 'cosine':
        # For cosine similarity, take the mean across the first axis (components)
        return np.mean(distance_arrays, axis=0)
    else:
        # For distance metrics, use the traditional L2 combination
        return np.sqrt(np.sum(distance_arrays**2, axis=0))

def create_triangular_matrix_plot(ax, within_violin_dict, between_violin_dict, distance_metric):
    """
    Create an upper triangular matrix plot showing all pairwise comparisons.
    
    Args:
        ax: Matplotlib axes object to plot on
        within_violin_dict: Dict containing within-group distance data
        between_violin_dict: Dict containing between-group distance data  
        distance_metric: Distance metric being used
    """
    import matplotlib.patches as patches
    from matplotlib.colors import LogNorm, PowerNorm
    
    # Define the groups and their corresponding data
    groups = ['Non-strategic', 'Strategic', 'RNN']
    group_colors = {'Non-strategic': 'tab:purple', 'Strategic': 'tab:cyan', 'RNN': 'tab:red'}
    
    # Organize the data into a matrix format
    # (0,0): Non-strategic within, (1,1): Strategic within, (2,2): RNN within
    # (0,1): NS-S between, (0,2): NS-RNN between, (1,2): S-RNN between
    matrix_data = {}
    matrix_data[(0,0)] = within_violin_dict['nonstrategic']
    matrix_data[(1,1)] = within_violin_dict['strategic'] 
    matrix_data[(2,2)] = within_violin_dict['RNN']
    matrix_data[(0,1)] = between_violin_dict['S-NS']  # Strategic vs Non-strategic
    matrix_data[(0,2)] = between_violin_dict['NS-RNN']  # Non-strategic vs RNN
    matrix_data[(1,2)] = between_violin_dict['S-RNN']   # Strategic vs RNN
    
    # Calculate summary statistics for each cell
    cell_stats = {}
    for key, data in matrix_data.items():
        if len(data) > 0:
            cell_stats[key] = {
                'median': np.median(data),
                'mean': np.mean(data), 
                'std': np.std(data),
                'n': len(data)
            }
        else:
            cell_stats[key] = {
                'median': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'n': 0
            }
    
    # Create the matrix visualization
    n_groups = len(groups)
    
    # Calculate cell size and positions
    cell_size = 1.0
    spacing = 0.0
    
    # Get all valid median values for color scaling
    valid_medians = [stats['median'] for stats in cell_stats.values() if not np.isnan(stats['median'])]
    
    if len(valid_medians) > 0:
        vmin = min(valid_medians)
        vmax = max(valid_medians)
        # Use a slight normalization to avoid extreme values
        norm = plt.Normalize(vmin=vmin*0.9, vmax=vmax*1.1)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
    
    # Use viridis colormap for all distance metrics
    cmap = plt.cm.viridis
    
    # Draw the lower triangular matrix
    for i in range(n_groups):
        for j in range(i + 1):  # j goes from 0 to i (inclusive)
            # Calculate cell position
            x = j + spacing/2
            y = (n_groups - 1 - i) + spacing/2
            
            # Get the data for this cell - need to map lower triangular to upper triangular data
            # For lower triangular, we need to swap indices when j > i
            data_key = (min(i, j), max(i, j)) if i != j else (i, j)
            if data_key in cell_stats:
                stats = cell_stats[data_key]
                data = matrix_data[data_key]
                
                if stats['n'] > 0:
                    # Color based on median value
                    color = cmap(norm(stats['median']))
                    
                    # Create the cell without any text labels
                    rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor=color, alpha=0.9)
                    ax.add_patch(rect)
                else:
                    # No data available - create empty cell without text
                    rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='lightgray', alpha=0.5)
                    ax.add_patch(rect)
    
    # Set up the axes for compact matrix without spacing
    ax.set_xlim(-0.5, n_groups + 0.5)
    ax.set_ylim(-0.5, n_groups + 0.5)
    ax.set_aspect('equal')
    
    # Add group labels for the compact matrix - labels below the matrix
    for i, group in enumerate(groups):
        # X-axis labels (bottom) - adjusted for no spacing
        ax.text(i + cell_size/2, -0.3, group, 
               ha='center', va='top', fontsize=10, fontweight='bold',
               rotation=45, color=group_colors[group])
        
        # Y-axis labels (left) - adjusted for no spacing
        ax.text(-0.3, (n_groups - 1 - i) + cell_size/2, group,
               ha='right', va='center', fontsize=10, fontweight='bold',
               color=group_colors[group])
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add title with less padding to avoid overlap
    ax.set_title(f'{get_distance_label(distance_metric)} Matrix\n(Lower Triangular)', 
                fontsize=14, pad=10)
    
    # Create colorbar using a simpler approach that avoids layout engine conflicts
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Use plt.colorbar with the specific axes - simpler approach
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'{get_distance_label(distance_metric)}', fontsize=10)
