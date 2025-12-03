import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression, paper_logistic_regression_strategic
from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior, fit_single_paper,fit_single_paper_strategic, WSLS_reward_comparison, monkey_logistic_regression, parse_monkey_behavior_reduced, paper_logistic_accuracy, fit_glr 
from seaborn import kdeplot, violinplot
import random
import pickle
from scipy.optimize import curve_fit
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
# from analysis_scripts.PCA_and_rotation import PCA, rotate_to_PCs
from mpl_toolkits.mplot3d import Axes3D



random.seed(29890)
# general idea for plot: logistic regression for BG only. logistic regression for PFC only. 
# Logistic regression for RLRNN with RNN and RL modules inset smaller. to the right. 
# length of image roughly 3.5x size of a single plot.

mpdb_p = '/Users/fmb35/Desktop/matching-pennies-lite.sqlite'

mpbeh_p = '/Users/fmb35/Desktop/MPbehdata.csv'

# stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'

stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/stitched_monkey_data_safely_cleaned.pkl'

task_image = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/MP_image.png'

#dict of representative sessions for each monkey
session_dict = {'D' : 39, 'E' : 87, 'I' : 23, 'K' :11, 'H':4, 'C': 18, 'F' : 20}


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


def calculate_decay_metric_simple(coeff):
    """Calculates the decay-based skew metric based on user formula."""
    if len(coeff) < 2:
        # Fallback for insufficient coefficients
        return 0
    
    # return (coeff[0] - coeff[1])/coeff[0]
    return (coeff[0] - coeff[1])/max(np.abs(coeff))


def calculate_decay_metric(coeff):
    """Calculates the decay-based skew metric based on user formula."""
    if len(coeff) < 2:
        # Fallback for insufficient coefficients
        return coeff[0] / np.max(np.abs(coeff)) if np.max(np.abs(coeff)) > 0 else 0
    
    max_abs_val = np.max(np.abs(coeff))
    if max_abs_val == 0:
        return 0
        
    decay = np.abs(coeff[1] - coeff[0])
    sign = np.sign(coeff[np.argmax(np.abs(coeff))])
    
    return (decay / max_abs_val) * sign

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


def generate_comparison_figure(mpbeh_path, mpdb_path, pfc_params, env_params, 
                               BG_params, task_image_p, nits=2, order=5, 
                               cutoff=5000, n_trials=5000, drop_nan_cols=False, cached=False):
    """
    Generate comparison figure with monkey behavioral data and model fits.
    
    Args:
        bias (bool): Whether to include bias term in logistic regression. When True,
                    the coefficient vectors include the bias term, resulting in a 
                    43x21 matrix for PCA (instead of 43x20 without bias).
        PCA (bool): Whether to use PCA-based asymmetry calculation.
        old_formatting (bool): Whether to use the old figure layout where asymmetry 
                              plot appears above (before) the monkey plots, similar 
                              to fig1_old.py formatting.
        ... (other parameters)
    """

    # Define modern color palette
    COLORS = {
        'strategic': '#3a86ff',       # Match actual plot blue
        'non-strategic': '#ff006e',   # Match actual plot purple/magenta
        'algorithm': '#2D9D5A',       # Green
        'algorithm1': '#F2B705',      # Amber for Algorithm 1
        'background': '#f8f9fa',      # Light gray background
        'grid': '#e6e6e6',            # Subtle grid lines
        'text': '#333333',            # Dark gray text
        'accent1': '#F2B705',         # Amber accent
        'accent2': '#23A7C1'          # Cyan accent
    }
    
    mp_data = pd.read_pickle(stitched_p)
    
    # Drop time columns with NaN values for monkeys C, E, F since sessions should already be sorted
    time_columns = ['year', 'month', 'day', 'ord', 'datetime']
    existing_time_columns = [col for col in time_columns if col in mp_data.columns]
    
    if existing_time_columns:
        # Check if any of the problematic monkeys have NaN values in time columns
        problematic_monkeys = ['C', 'E', 'F']
        columns_to_drop = set()
        
        for monkey in problematic_monkeys:
            monkey_data = mp_data[mp_data['animal'] == monkey]
            if len(monkey_data) > 0:
                for col in existing_time_columns:
                    if monkey_data[col].isna().any():
                        columns_to_drop.add(col)
        
        if columns_to_drop:
            print(f"Dropping time columns with NaNs for monkeys C, E, F: {sorted(columns_to_drop)}")
            print("  (Sessions should already be sorted)")
            mp_data = mp_data.drop(columns=list(columns_to_drop))


    fig = plt.figure(figsize=(14, 10), dpi=144)
    # Define GridSpec with reduced columns to minimize whitespace
    gs = gridspec.GridSpec(19, 20, figure=fig, wspace=0.5, hspace=1)


    # Define legend elements first to avoid UnboundLocalError
    # Using tab:purple and tab:cyan to match actual plotting colors (consistent with fig5)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:purple', 
               markersize=8, label='Non-Strategic'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='tab:cyan', 
               markersize=10, label='Strategic'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor=COLORS['algorithm1'], 
               markersize=8, label='Algorithm 1'),
    ]
    
    # Add the main title for the figure with modern styling
    fig.suptitle('Figure 1: Monkey Behavior Can Be Categorized Into Strategic and Non-Strategic Based on Behavioral Asymmetry', 
                 fontsize=18, y=0.97, fontweight='bold', color=COLORS['text'])
    
    # Helper for consistent panel labels
    label_props = dict(fontsize=20, fontweight='bold', ha='right', va='bottom')
    def add_panel_label(ax, label, x_offset=-0.02, y_offset=0.012):
        bbox = ax.get_position()
        fig.text(bbox.x0 + x_offset, bbox.y1 + y_offset, label, **label_props)
    
    # Layout based on formatting choice
    # Old formatting: asymmetry before monkeys
    # Task and Sequence panels - compact layout
    task_ax = fig.add_subplot(gs[0:5, 0:7])
    task_ax.set_title('Task', fontsize=16, fontweight='bold', pad=10)
    add_panel_label(task_ax, 'A', x_offset=-0.035)
    
    # Move sequence plot down to fill blank space between top and bottom sections
    WSLS_sequence_ax = fig.add_subplot(gs[6:10, 0:7])
    WSLS_sequence_ax.axis('off')
    add_panel_label(WSLS_sequence_ax, 'B', x_offset=-0.035)
    
    # Setup the asymmetry plot - positioned at the top (before monkeys), shifted right
    stationarity_plot_ax = fig.add_subplot(gs[0:9, 9:19])
    add_panel_label(stationarity_plot_ax, 'C')
    
    # Monkey plots positioned below asymmetry plot
    plot_height = 4
    # Non-strategic monkeys row (top row) with consistent sizes
    Monkey_C = fig.add_subplot(gs[11:11+plot_height, 0:5])
    Monkey_K = fig.add_subplot(gs[11:11+plot_height, 5:10])
    Monkey_H = fig.add_subplot(gs[11:11+plot_height, 10:15])
    Monkey_F = fig.add_subplot(gs[11:11+plot_height, 15:20])
    
    # Strategic monkeys row (bottom row) with consistent sizes
    Monkey_D = fig.add_subplot(gs[15:15+plot_height, 0:5])
    Monkey_E = fig.add_subplot(gs[15:15+plot_height, 5:10])
    Monkey_I = fig.add_subplot(gs[15:15+plot_height, 10:15])
        
        
    stationarity_plot_ax.set_xlabel(r'\xi_{loss}', fontsize=16, labelpad=5)
    stationarity_plot_ax.set_ylabel(r'\xi_{win}', fontsize=16, labelpad=5)
    
    stationarity_plot_ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    stationarity_plot_ax.axvline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    # Add diagonal line from (0,0) to (1,1)
    stationarity_plot_ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    # Add title to the stationarity plot with conditional formatting
    stationarity_plot_ax.set_title('Behavioral Asymmetry', fontsize=16, fontweight='bold', pad=10)
    # Remove grid
    stationarity_plot_ax.grid(False)
    # Expand axis limits to show all monkeys
    stationarity_plot_ax.set_xlim(-1.5, 1.5)
    stationarity_plot_ax.set_ylim(-1.5, 1.5)
    
    # Set ticks with larger range
    stationarity_plot_ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    stationarity_plot_ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    
    # Add dashed box from (0,0) to (1,1) for reference
    stationarity_plot_ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k--', alpha=0.3, lw=1)
    
    # # REMOVED: Example quadrant plots
    # q1_plot = fig.add_subplot(gs[0:2, 19:23])
    # q2_plot = fig.add_subplot(gs[2:4, 19:23])
    # q3_plot = fig.add_subplot(gs[4:6, 19:23])
    # q4_plot = fig.add_subplot(gs[6:8, 19:23])
    # 
    # # Create a list of the example regressor plots
    # plot_list = [q1_plot, q2_plot, q3_plot, q4_plot]
    # 
    # # Add title for the example regressors section
    # q1_plot.set_title('Example Regressors\nfor Each Quadrant', fontsize=14, fontweight='bold', pad=15)

    # Add behavioral asymmetry legend directly to the plot (matching fig5 style)
    stationarity_plot_ax.legend(
        handles=legend_elements,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        fontsize=10,
        frameon=False,  # No frame
        ncol=1  # Arrange vertically
    )
    
    # Create legend axis in the empty space on the bottom row (strategic monkeys) - moved down
    legend_ax = fig.add_subplot(gs[16:20, 15:20])
    legend_ax.axis('off')  # Hide the axis
    
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color='#1f77b4', lw=2, label='repeat win'),      # Blue (colors[0])
            Line2D([0], [0], color='#ff7f0e', lw=2, label='change win'),      # Orange (colors[1])
            Line2D([0], [0], color='#2ca02c', lw=2, label='repeat loss'),     # Green (colors[2])
            Line2D([0], [0], color='#d62728', lw=2, label='change loss')      # Red (colors[3])
        ],
        loc='center',
        fontsize=10,
        frameon=False,  # No frame
        title="Regression Coefficients",
        title_fontsize=12,
        ncol=1  # Arrange vertically
    )

    # # REMOVED: Setup the regressor plots with proper axes
    # for i, ax in enumerate(plot_list):
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_ylabel(f'{i+1}', fontsize=13, rotation=0, labelpad=12, ha='right', fontweight='bold')
    #     ax.yaxis.set_label_position('right')
    #     # Keep all axes visible
    #     for spine in ax.spines.values():
    #         spine.set_visible(True)
    #         spine.set_color('#bdbdbd')
    #         spine.set_linewidth(0.8)
    #     ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=1)
    # 
    # # Define the inset data that was missing with better curve shapes
    # q1_inset_data = [[1, 0.7, 0.4, 0.2, 0], [0.95, 0.8, 0.65, 0.4, 0.2]]  # Top right
    # q2_inset_data = [[1, 0.6, 0.3, 0.1, 0], [-0.6, 0.3, 0.5, 0.2, 0.05]]  # Top left
    # q3_inset_data = [[-0.3, 0.1, 0.3, 0.1, 0.25], [-0.6, 0.2, 0.5, 0.2, 0.05]]  # Bottom left
    # q4_inset_data = [[-0.3, -0.1, 0.2, 0.4, 0.6], [0.95, 0.9*0.6, 0.9*0.3, 0.9*0.1, 0.9*0]]  # Bottom right
    # 
    # # Plot example data in each regressor plot with enhanced styling
    # for i, ax in enumerate(plot_list):
    #     data = q1_inset_data if i == 0 else q2_inset_data if i == 1 else q3_inset_data if i == 2 else q4_inset_data
    #     ax.plot(data[0], linewidth=2.5, color = 'tab:blue',alpha=0.9)
    #     ax.plot(data[1], linewidth=2.5, color = 'tab:orange',alpha=0.9)
    #     ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=1)
    # 
    # # REMOVED: Marker points and quadrant numbers in the asymmetry plot
    # stationarity_plot_ax.scatter(0.8, 0.8, alpha=1.0, marker='x', color='black', s=80, zorder=10, linewidth=2)
    # stationarity_plot_ax.scatter(-0.8, 0.8, alpha=1.0, marker='x', color='black', s=80, zorder=10, linewidth=2)
    # stationarity_plot_ax.scatter(-0.8, -0.8, alpha=1.0, marker='x', color='black', s=80, zorder=10, linewidth=2)
    # stationarity_plot_ax.scatter(0.8, -0.8, alpha=1.0, marker='x', color='black', s=80, zorder=10, linewidth=2)
    # 
    # # Add numbers with white backgrounds for better visibility
    # for i, (x, y) in enumerate([(0.8, 0.8), (-0.8, 0.8), (-0.8, -0.8), (0.8, -0.8)], 1):
    #     # Add white circular background with cleaner styling
    #     # stationarity_plot_ax.scatter(x, y+0.05, s=160, color='white', edgecolor='#bdbdbd', zorder=9, alpha=0.95)
    #     # Add number with better visibility
    #     stationarity_plot_ax.text(x, y+0.08, str(i), ha='center', va='center', 
    #                           fontsize=13, color=COLORS['text'], fontweight='bold', zorder=10)
    
    # Custom titles for strategic and non-strategic monkeys
    strategic_titles = {
        'D': 'Monkey D',
        'I': 'Monkey I',
        'E': 'Monkey E'
    }
    
    nonstrategic_titles = {
        'C': 'Monkey C',
        'F': 'Monkey F',
        'K': 'Monkey K',
        'H': 'Monkey H'
    }
    
    # Set titles according to new layout with consistent styling and better padding
    
    Monkey_E.set_title(strategic_titles['E'] , fontsize=16, fontweight='bold', pad=10)
    Monkey_D.set_title(strategic_titles['D'] , fontsize=16, fontweight='bold', pad=10)
    Monkey_I.set_title(strategic_titles['I'] , fontsize=16, fontweight='bold', pad=10)
    
    Monkey_C.set_title(nonstrategic_titles['C'] , fontsize=16, fontweight='bold', pad=10)
    # Add label D to first monkey plot
    Monkey_C.text(-0.15, 1.05, 'D', transform=Monkey_C.transAxes, 
                  fontsize=20, fontweight='bold', va='top', ha='right')
    Monkey_F.set_title(nonstrategic_titles['F'] , fontsize=16, fontweight='bold', pad=10)
    Monkey_K.set_title(nonstrategic_titles['K'] , fontsize=16, fontweight='bold', pad=10)
    Monkey_H.set_title(nonstrategic_titles['H'] , fontsize=16, fontweight='bold', pad=10)
    
    fig.text(0.05, 0.18, 'Strategic Monkeys', fontsize=14, rotation=90, va='center', ha='center',
                fontweight='bold', color='tab:cyan', bbox=dict(facecolor='white', alpha=0.9, 
                edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Non-strategic monkeys label - positioned for bottom left area with better spacing
    fig.text(0.05, 0.36, 'Non-Strategic\nMonkeys', fontsize=14, rotation=90, va='center', ha='center',
             fontweight='bold', color='tab:purple', bbox=dict(facecolor='white', alpha=0.9,
             edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Set y-axis labels for the different sections with better positioning - removed from stationarity plot
    fig.text(.075, 0.25, 'Regression Coefficient', fontsize=14, rotation=90, va='center', ha='center')

    # Add Stay/Switch labels near Monkey F (positioned based on layout)
    fig.text(0.85, 0.38, 'Stay', fontsize=12, ha='left', va='center', 
            fontweight='bold', color='darkgreen')
    fig.text(0.85, 0.3, 'Switch', fontsize=12, ha='left', va='center',
            fontweight='bold', color='darkred')
    
    # Set x-axis labels - for bottom row non-strategic monkeys only
    for ax in [Monkey_H, Monkey_F]:
        ax.set_xlabel('Trials Back', fontsize=16, labelpad=10)
    
    # Remove x-axis labels from other axes
    for ax in [Monkey_D, Monkey_E, Monkey_I, Monkey_C, Monkey_K]:
        ax.set_xlabel('')
    
    # Remove grid and set axis properties for all plots
    for ax in [Monkey_E, Monkey_D, Monkey_I, Monkey_C, Monkey_F, Monkey_K, Monkey_H]:
        # Remove grid
        ax.grid(False)
        # Set cleaner spines
        for spine in ax.spines.values():
            spine.set_color('#bdbdbd')
            spine.set_linewidth(0.8)
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=1)
    
    # Place all axes in a list for easier iteration - updated to match new layout order
    axs = [Monkey_C, Monkey_K, Monkey_H, Monkey_F, Monkey_D, Monkey_E, Monkey_I]
    monkeys = ['C','K','H','F','D','E','I']
    
    # Hide yticks for all plots except the leftmost in each section
    # Strategic monkeys: only Monkey_D (leftmost) gets yticks
    # Non-strategic monkeys: only Monkey_C and Monkey_H (leftmost in their rows) get yticks
    for ax in [Monkey_E, Monkey_I, Monkey_K, Monkey_F]:
        ax.set_yticks([])
        
    # Original data loading method, adjusted for new monkey order
    mp2_data = mp_data[mp_data['task'] == 'mp']
    print(f'Available animals in mp data: {sorted(mp2_data["animal"].unique())}')
    monkey_data = [mp2_data[(mp2_data['animal'] == monkeys[i])] for i in range(len(monkeys))]
    

    # monkey_data.append(monkey_beh_data[monkey_beh_data['animal']==112])
    
    all_coef_values = []
    glr_results = []  # Store (result, err, labels) for each monkey

    # for i in range(len(monkeys)):
    #     md = monkey_data[i]
    #     # fit glr to monkey data       
    #     result, err, labels = fit_glr(md, order=order, a_order=2, r_order=1, model=False, err=True, labels=True)
    #     glr_results.append((result, err, labels))
    #     # Filter out NaN and infinite values and collect all coefficient values
    #     # if result is not None:
    #     #     valid_coefficients = [c for c in result if not (np.isnan(c) or np.isinf(c))]
    #     #     all_coef_values.extend(valid_coefficients)

    # Calculate global y-limits based on actual data with 10% padding
    if all_coef_values and len(all_coef_values) > 0:
        max_abs_coef = max(abs(min(all_coef_values)), abs(max(all_coef_values)))
        # Add 10% padding, but ensure minimum range for visibility
        max_abs_coef = max(max_abs_coef * 1.1, 0.1)
        y_min, y_max = -max_abs_coef, max_abs_coef
    else:
        # Fallback if no data
        y_min, y_max = -0.4, 0.4
    
    for i in range(len(monkeys)):
        md = monkey_data[i]
    
        
        # Set y-limits for consistency
        axs[i].set_ylim(y_min, y_max)
                
        if monkeys[i] == 'F' or monkeys[i] == 'E' or monkeys[i] == 'C':
            # Use session-based cutoff instead of simple trial-based cutoff
            print(f'Applying session-based cutoff of {cutoff} trials for monkey {monkeys[i]}')
            monkey_data[i] = cutoff_trials_by_session(monkey_data[i], cutoff)
       
        # Fit GLR without averaging to get per-session results
        result, labels = fit_glr(md, order=order, a_order=2, r_order=1, model=False, err=False, labels=True, average=False)

        # Calculate session lengths for weighted average
        session_list = list(md['id'].unique())
        sess_lens = [len(md[md['id'] == session]) for session in session_list]
        
        # Calculate the weighted average of the regression coefficients
        partition_fit = np.average(result, axis=0, weights=sess_lens)
        
        # Plot the regression coefficients for each monkey
        if partition_fit is not None and len(partition_fit) > 0:
            for j in range(len(labels)):
                axs[i].plot(np.arange(1, order+1), partition_fit[j*order:(j+1)*order], label=labels[j])
        
    # Don't override the titles that were already set with session suffix
    # The titles were already set above with the correct session suffix
        
    # Remove ticks from all regression plots except the leftmost in each section
    for ax in [Monkey_E, Monkey_I, Monkey_K, Monkey_F]:
        ax.set_yticks([])
            
    
    nonstrategic_monkeys  = ['C','H','F','K']  # MP2 dataset - C,F are non-strategic in MP2
    strategic_monkeys = ['E','D','I']  # MP2 dataset - E is strategic in MP2
    
    nonstrat_corr = []
    strat_corr = []
    corr_list = []
    
    # Create dictionaries to store all session data for each monkey for plotting
    monkey_session_data = {monkey: {'ls_values': [], 'ws_values': [], 'n_trials': []} for monkey in nonstrategic_monkeys + strategic_monkeys}
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache filenames
    cache_filename = 'fig1_data.pkl'
    regression_cache_filename = 'fig1_regression_data.pkl'
    cache_filepath = os.path.join(cache_dir, cache_filename)
    regression_cache_filepath = os.path.join(cache_dir, regression_cache_filename)
    
    # Check if cached data exists
    cache_exists = os.path.exists(cache_filepath)
    regression_cache_exists = os.path.exists(regression_cache_filepath)
    
    # Load cached regression results if requested and available
    regression_results = {}
    if cached and regression_cache_exists:
        print('Loading cached regression results')
        with open(regression_cache_filepath, 'rb') as f:
            regression_results = pickle.load(f)


    ns_fits = {}
    ns_trials = {}
    s_fits = {}
    s_trials = {}
    mp1_fits = {}
    mp1_trials = {}
    
    # Process data or use cached results
    if not cached or not cache_exists:
        monkey_dict = {}
        # Process non-strategic monkeys
        for monkey in nonstrategic_monkeys:
            data = monkey_data[monkeys.index(monkey)]
            
            # Apply session-based cutoff for monkeys C, E, F (same as in main plotting)
            if monkey == 'F' or monkey == 'E' or monkey == 'C':
                print(f'Applying session-based cutoff of {cutoff} trials for asymmetry calculation of monkey {monkey}')
                data = cutoff_trials_by_session(data, cutoff)
            
            session_list = list(data['id'].unique())

            # number of trials in each session
            sess_lens = [len(data[data['id'] == session]) for session in session_list]
            
            # Create a cache key for this specific partition regression
            cache_key = f"monkey_{monkey}"
            
            # Check if we have cached results for this partition
            if cached and cache_key in regression_results:
                result, labels = regression_results[cache_key]
                partition_fit = np.average(result, axis=0, weights=sess_lens)
                print(f"Using cached regression for {cache_key}")
            else:                   
                result, err, labels = fit_glr(data, order=order, model=False,a_order=2, r_order=1,err=True,labels=True, average=False)
                partition_fit =  np.average(result, axis=0, weights=sess_lens)
                # Store the result and labels in cache
                regression_results[cache_key] = (result, labels)
        
            # Extract coefficients from the partition fit
            
            coeffs = {labels[j]: partition_fit[j*order:(j+1)*order] for j in range(len(labels))}
            # Compute metrics using correct order: repeat - change
            # labels[0] = 'win stay' (repeat win), labels[1] = 'win switch' (change win)
            # labels[2] = 'lose stay' (repeat loss), labels[3] = 'lose switch' (change loss)
            metrics = {}
            metrics['win'] = calculate_decay_metric_simple(coeffs[labels[0]]-coeffs[labels[1]])  # repeat win - change win
            metrics['loss'] = calculate_decay_metric_simple(coeffs[labels[3]]-coeffs[labels[2]])  # repeat loss - change loss
            
            # Use leave-one-session-out cross validation to bootstrap error on the metrics
            metric_errs = {'loss': [], 'win': []}
            for fold in range(len(result)):
                # Delete one session from results
                temp_result = np.delete(result, fold, axis=0)
                temp_sess_lens = np.delete(sess_lens, fold)
                temp_result = np.average(temp_result, axis=0, weights=temp_sess_lens)
                
                # Extract coefficients from this fold
                temp_coeffs = {labels[j]: temp_result[j*order:(j+1)*order] for j in range(len(labels))}
                
                # Compute metrics for this fold
                metric_errs['win'].append(calculate_decay_metric_simple(temp_coeffs[labels[0]]-temp_coeffs[labels[1]]))
                metric_errs['loss'].append(calculate_decay_metric_simple(temp_coeffs[labels[3]]-temp_coeffs[labels[2]]))
            # Use 95% confidence interval to estimate the error on the metrics for each monkey
            asymmetric_metric_errs = {}
            for metric_name in ['win', 'loss']:
                mean_val = metrics[metric_name]
                percentiles = np.percentile(metric_errs[metric_name], [2.5, 97.5])
                asymmetric_metric_errs[metric_name] = np.abs(percentiles - mean_val)
            
            # X-axis: loss metric, Y-axis: win metric
            x_axis_metrics = metrics['loss']
            x_axis_errs = asymmetric_metric_errs['loss']

            y_axis_metrics = metrics['win']
            y_axis_errs = asymmetric_metric_errs['win']
    
                
            stationarity_plot_ax.errorbar(x_axis_metrics, y_axis_metrics, 
                yerr=y_axis_errs.reshape(2, 1), 
                xerr=x_axis_errs.reshape(2, 1), 
                color='tab:purple', fmt='o', markersize=4, capsize=5)
            corr_list.append({'monkey': monkey, 
                            'non-monotonicity': (x_axis_metrics, y_axis_metrics), 
                            'error': (x_axis_errs, y_axis_errs), 
                            'strategy': 'non-strategic',
                            'fit': partition_fit,
                            'labels': labels})
            
            # Add label for this monkey directly on the plot
            # Adjust text position for monkeys at extreme positions to keep labels visible
            text_x = x_axis_metrics + 0.02
            text_y = y_axis_metrics + 0.02
            
            # If the monkey is at extreme positions, adjust label placement
            if x_axis_metrics > 0.9:  # Far right
                text_x = x_axis_metrics - 0.05
                ha = 'right'
            else:
                ha = 'left'
                
            if y_axis_metrics > 0.9:  # Far top
                text_y = y_axis_metrics - 0.05
                va = 'top'
            else:
                va = 'bottom'
                
            stationarity_plot_ax.text(text_x, text_y, monkey, 
                                    ha=ha, va=va, fontsize=12, 
                                    fontweight='bold', color='tab:purple')


        # Process strategic monkeys
        
        for monkey in strategic_monkeys:

            data = monkey_data[monkeys.index(monkey)]
            
            # Apply session-based cutoff for monkeys C, E, F (same as in main plotting)
            if monkey == 'F' or monkey == 'E' or monkey == 'C':
                print(f'Applying session-based cutoff of {cutoff} trials for asymmetry calculation of monkey {monkey}')
                data = cutoff_trials_by_session(data, cutoff)
            
            session_list = list(data['id'].unique())

            # number of trials in each session
            sess_lens = [len(data[data['id'] == session]) for session in session_list]
            
            # Create a cache key for this specific partition regression
            cache_key = f"monkey_{monkey}"
            
            # Check if we have cached results for this partition
            if cached and cache_key in regression_results:
                result, labels = regression_results[cache_key]
                partition_fit = np.average(result, axis=0, weights=sess_lens)
                print(f"Using cached regression for {cache_key}")
            else:                   
                result, err, labels = fit_glr(data, order=order, model=False,a_order=2, r_order=1,err=True,labels=True, average=False)
                partition_fit =  np.average(result, axis=0, weights=sess_lens)
                # Store the result and labels in cache
                regression_results[cache_key] = (result, labels)
        
            # Extract coefficients from the partition fit
            
            coeffs = {labels[j]: partition_fit[j*order:(j+1)*order] for j in range(len(labels))}
            # Compute metrics using correct order: repeat - change
            # labels[0] = 'win stay' (repeat win), labels[1] = 'win switch' (change win)
            # labels[2] = 'lose stay' (repeat loss), labels[3] = 'lose switch' (change loss)
            metrics = {}
            metrics['win'] = calculate_decay_metric_simple(coeffs[labels[0]]-coeffs[labels[1]])  # repeat win - change win
            metrics['loss'] = calculate_decay_metric_simple(coeffs[labels[3]]-coeffs[labels[2]])  # repeat loss - change loss

            # Use leave-one-session-out cross validation to bootstrap error on the metrics
            metric_errs = {'loss': [], 'win': []}
            for fold in range(len(result)):
                # Delete one session from results
                temp_result = np.delete(result, fold, axis=0)
                temp_sess_lens = np.delete(sess_lens, fold)
                temp_result = np.average(temp_result, axis=0, weights=temp_sess_lens)
                
                # Extract coefficients from this fold
                temp_coeffs = {labels[j]: temp_result[j*order:(j+1)*order] for j in range(len(labels))}
                
                # Compute metrics for this fold
                metric_errs['win'].append(calculate_decay_metric_simple(temp_coeffs[labels[0]]-temp_coeffs[labels[1]]))
                metric_errs['loss'].append(calculate_decay_metric_simple(temp_coeffs[labels[3]]-temp_coeffs[labels[2]]))
            # Use 95% confidence interval to estimate the error on the metrics for each monkey
            asymmetric_metric_errs = {}
            for metric_name in ['win', 'loss']:
                mean_val = metrics[metric_name]
                percentiles = np.percentile(metric_errs[metric_name], [2.5, 97.5])
                asymmetric_metric_errs[metric_name] = np.abs(percentiles - mean_val)
            
            # X-axis: loss metric, Y-axis: win metric
            x_axis_metrics = metrics['loss']
            x_axis_errs = asymmetric_metric_errs['loss']

            y_axis_metrics = metrics['win']
            y_axis_errs = asymmetric_metric_errs['win']
    
                
            stationarity_plot_ax.errorbar(x_axis_metrics, y_axis_metrics, 
                yerr=y_axis_errs.reshape(2, 1), 
                xerr=x_axis_errs.reshape(2, 1), 
                color='tab:cyan', fmt='*', markersize=4, capsize=5)
            corr_list.append({'monkey': monkey, 
                            'non-monotonicity': (x_axis_metrics, y_axis_metrics), 
                            'error': (x_axis_errs, y_axis_errs), 
                            'strategy': 'strategic',
                            'fit': partition_fit,
                            'labels': labels})
            
            # Add label for this monkey directly on the plot
            # Adjust text position for monkeys at extreme positions to keep labels visible
            text_x = x_axis_metrics + 0.02
            text_y = y_axis_metrics + 0.02
            
            # If the monkey is at extreme positions, adjust label placement
            if x_axis_metrics > 0.9:  # Far right
                text_x = x_axis_metrics - 0.05
                ha = 'right'
            else:
                ha = 'left'
                
            if y_axis_metrics > 0.9:  # Far top
                text_y = y_axis_metrics - 0.05
                va = 'top'
            else:
                va = 'bottom'
                
            stationarity_plot_ax.text(text_x, text_y, monkey, 
                                    ha=ha, va=va, fontsize=12, 
                                    fontweight='bold', color='tab:cyan')


            
        # save x_axis_metrics and y_axis_metrics for each monkey, as well as x_axis_errs and y_axis_errs
        # as well as regression coefficients
        x_axis_metrics = [item['non-monotonicity'][0] for item in corr_list]
        y_axis_metrics = [item['non-monotonicity'][1] for item in corr_list]
        x_axis_errs = [item['error'][0] for item in corr_list]
        y_axis_errs = [item['error'][1] for item in corr_list]
            
        corr_dict = {item['monkey']: item for item in corr_list}
        
        # Save regression results for individual sessions BEFORE overwriting
        os.makedirs(os.path.dirname(cache_filepath), exist_ok=True)
        with open(regression_cache_filepath, 'wb') as f:
            pickle.dump(regression_results, f)
        print(f'Saved {len(regression_results)} regression results to {regression_cache_filepath}')
        
        # Save corr_list for quick loading of summary statistics
        with open(cache_filepath, 'wb') as f:
            pickle.dump(corr_dict, f)
        print(f'Saved aggregated data to {cache_filepath}')
    
    else:
        print('Loading cached data')
        # load corr_list from fig1_data.pkl
        with open(cache_filepath, 'rb') as f:
            corr_dict = pickle.load(f)
        
        # Convert back to list format if needed for compatibility with existing code
        corr_list = list(corr_dict.values())
        
        # use corr_list to plot stationarity plot
        print(f'Cached monkeys: {sorted(corr_dict.keys())}')
        for monkey, data in corr_dict.items():
            # Handle different strategy types (using tab colors to match fig5)
            if data['strategy'] == 'strategic':
                color = 'tab:cyan'
                marker = '*'
                markersize = 5
            elif data['strategy'] == 'non-strategic':
                color = 'tab:purple'
                marker = 'o'
                markersize = 5
            elif data['strategy'] == '1':
                color = COLORS['algorithm1']
                marker = 'd'
                markersize = 6
            else:
                color = 'gray'
                marker = 'o'
                markersize = 5
                
            stationarity_plot_ax.errorbar(
                data['non-monotonicity'][0], 
                data['non-monotonicity'][1], 
                yerr=[[data['error'][1][0]], [data['error'][1][1]]], 
                xerr=[[data['error'][0][0]], [data['error'][0][1]]], 
                color=color, 
                fmt=marker, 
                markersize=markersize, 
                capsize=5
            )
            print(f'Plotting monkey {monkey} at ({data["non-monotonicity"][0]:.3f}, {data["non-monotonicity"][1]:.3f})')
            
            # Add label for this monkey directly on the plot (using tab colors to match fig5)
            if data['strategy'] == 'strategic':
                text_color = 'tab:cyan'
            elif data['strategy'] == '1':
                text_color = COLORS['algorithm1']
            else:
                text_color = 'tab:purple'
            
            # Adjust text position for monkeys at extreme positions to keep labels visible
            text_x = data['non-monotonicity'][0] + 0.02
            text_y = data['non-monotonicity'][1] + 0.02
            
            # For MP1 (Algorithm 1) monkeys, add extra vertical offset to avoid overlap with diamond markers
            if data['strategy'] == '1':
                text_y += 0.08
            
            # If the monkey is at extreme positions, adjust label placement
            if data['non-monotonicity'][0] > 0.9:  # Far right
                text_x = data['non-monotonicity'][0] - 0.05
                ha = 'right'
            elif data['non-monotonicity'][0] < -0.9:  # Far left
                text_x = data['non-monotonicity'][0] + 0.05
                ha = 'left'
            else:
                ha = 'left'
                
            if data['non-monotonicity'][1] > 0.9:  # Far top
                text_y = data['non-monotonicity'][1] - 0.05
                va = 'top'
            elif data['non-monotonicity'][1] < -0.9:  # Far bottom
                text_y = data['non-monotonicity'][1] + 0.05
                va = 'bottom'
            else:
                va = 'bottom'
            
            stationarity_plot_ax.text(text_x, text_y, monkey, 
                                        ha=ha, va=va, fontsize=12, 
                                        fontweight='bold', color=text_color)

    
    # algo2_early = algo2[(algo2['id'] < session_list[5+5]) & (algo2['id'] >= session_list[5]) ] # first 10 sessions after 5 warmup sessions
    
    # Remove all references to algo2_late_ax
    
    # Update row labels to correspond to where the strategic and non-strategic monkeys actually are
    # Strategic monkeys (E, D, I) are in the top row

    
    # ===================================================================
    # MP1 (Algorithm 1) REGRESSOR COMPUTATION FOR BIAS PLOT
    # ===================================================================
    # This section computes the logistic regression coefficients for monkeys
    # C, F, and E playing against Algorithm 1 opponents. These regressors
    # are used to compute behavioral asymmetry (bias) in the main scatter plot.
    # 
    # Process:
    # 1. Load behavioral data for each monkey vs Algorithm 1
    # 2. Partition data into chunks of ~n_trials each  
    # 3. Fit logistic regression to each partition using fit_single_paper_strategic()
    # 4. Store coefficients in mp1_fits and trial counts in mp1_trials
    # 5. Use in PCA asymmetry calculation for bias plot positioning
    
    C_1 = load_behavior(mpbeh_path,algorithm=1, monkey= 13, drop_nan_cols=drop_nan_cols)
    F_1 = load_behavior(mpbeh_path,algorithm=1, monkey= 112, drop_nan_cols=drop_nan_cols)
    E_1 = load_behavior(mpbeh_path,algorithm=1, monkey= 18, drop_nan_cols=drop_nan_cols)
    # make sure dataframes only contain that monkey
    C_1 = C_1[C_1['animal'] == 13]
    F_1 = F_1[F_1['animal'] == 112]
    E_1 = E_1[E_1['animal'] == 18]
    
    mp1_data = [E_1,C_1, F_1]
    
    
    mp1labels = ['E_MP1','C_MP1','F_MP1']
    mp1colors = ['tab:blue','tab:blue','tab:blue']  # Changed from tab:red to tab:blue
    # corr_list = []
    # Initialize storage for MP1 regression fits and trial counts
    mp1_fits = {}
    mp1_trials = {}
    for i,dat in enumerate(mp1_data):
        monkey_label = mp1labels[i]
        monkey_char = monkey_label.split('_')[0]
        
        # Apply session-based cutoff for monkeys C, E, F (same as in main plotting)
        if monkey_char in ['F', 'E', 'C']:
            print(f'Applying session-based cutoff of {cutoff} trials for asymmetry calculation of monkey {monkey_label}')
            dat = cutoff_trials_by_session(dat, cutoff)
        
        session_list = list(dat['id'].unique())

        # number of trials in each session
        sess_lens = [len(dat[dat['id'] == session]) for session in session_list]
        
        # Create a cache key for this specific partition regression
        cache_key = f"monkey_{monkey_label}"
        


        # Check if we have cached results for this partition
        if cached and cache_key in regression_results:
            result, labels = regression_results[cache_key]
            partition_fit = np.average(result, axis=0, weights=sess_lens)
            print(f"Using cached regression for {cache_key}")
        else:                   
            result, err, labels = fit_glr(dat, order=order, model=False,a_order=2, r_order=1,err=True,labels=True, average=False)
            partition_fit =  np.average(result, axis=0, weights=sess_lens)
            # Store the result and labels in cache
            regression_results[cache_key] = (result, labels)

         
        coeffs = {labels[j]: partition_fit[j*order:(j+1)*order] for j in range(len(labels))}
        # Compute metrics using correct order: repeat - change
        # labels[0] = 'win stay' (repeat win), labels[1] = 'win switch' (change win)
        # labels[2] = 'lose stay' (repeat loss), labels[3] = 'lose switch' (change loss)
        metrics = {}
        metrics['win'] = calculate_decay_metric_simple(coeffs[labels[0]]-coeffs[labels[1]])  # repeat win - change win
        metrics['loss'] = calculate_decay_metric_simple(coeffs[labels[3]]-coeffs[labels[2]])  # repeat loss - change loss
        
        # Use leave-one-session-out cross validation to bootstrap error on the metrics
        metric_errs = {'loss': [], 'win': []}
        for fold in range(len(result)):
            # Delete one session from results
            temp_result = np.delete(result, fold, axis=0)
            temp_sess_lens = np.delete(sess_lens, fold)
            temp_result = np.average(temp_result, axis=0, weights=temp_sess_lens)
            
            # Extract coefficients from this fold
            temp_coeffs = {labels[j]: temp_result[j*order:(j+1)*order] for j in range(len(labels))}
            
            # Compute metrics for this fold
            metric_errs['win'].append(calculate_decay_metric_simple(temp_coeffs[labels[0]]-temp_coeffs[labels[1]]))
            metric_errs['loss'].append(calculate_decay_metric_simple(temp_coeffs[labels[3]]-temp_coeffs[labels[2]]))
        # use 95% confidence interval to estimate the error on the metrics for each monkey
        asymmetric_metric_errs = {}
        for metric_name in ['win', 'loss']:
            mean_val = metrics[metric_name]
            percentiles = np.percentile(metric_errs[metric_name], [2.5, 97.5])
            asymmetric_metric_errs[metric_name] = np.abs(percentiles - mean_val)
        
        # X-axis: loss metric, Y-axis: win metric
        x_axis_metrics = metrics['loss']
        x_axis_errs = asymmetric_metric_errs['loss']

        y_axis_metrics = metrics['win']
        y_axis_errs = asymmetric_metric_errs['win']
        # x_axis_metrics = 1/2 * (metrics[labels[1]] - metrics[labels[2]])
        # err_1 = asymmetric_metric_errs[labels[1]]
        # err_3 = asymmetric_metric_errs[labels[2]]
        # x_axis_errs = 0.5 * np.sqrt(np.vstack([err_1, err_3])**2).sum(axis=0)

        # y_axis_metrics = 1/2 * (metrics[labels[0]] - metrics[labels[3]])
        # err_0 = asymmetric_metric_errs[labels[0]]
        # err_2 = asymmetric_metric_errs[labels[3]]
        # y_axis_errs = 0.5 * np.sqrt(np.vstack([err_0, err_2])**2).sum(axis=0)

        
        
        stationarity_plot_ax.errorbar(x_axis_metrics, y_axis_metrics, 
            yerr=y_axis_errs.reshape(2, 1), 
            xerr=x_axis_errs.reshape(2, 1), 
            color=COLORS['algorithm1'], 
            ecolor=COLORS['algorithm1'], 
            marker='d', ms=6, alpha=.8, elinewidth=1.5)
        corr_list.append({'monkey': monkey_label, 
                        'non-monotonicity': (x_axis_metrics, y_axis_metrics), 
                        'error': (x_axis_errs, y_axis_errs), 
                        'strategy': '1',
                        'fit': partition_fit,
                        'labels': labels})
        
        # Add label for this monkey directly on the plot
        # Adjust text position for monkeys at extreme positions to keep labels visible
        text_x = x_axis_metrics + 0.02
        text_y = y_axis_metrics + 0.02
        
        # If the monkey is at extreme positions, adjust label placement
        if x_axis_metrics > 0.9:  # Far right
            text_x = x_axis_metrics - 0.05
            ha = 'right'
        else:
            ha = 'left'
            
        if y_axis_metrics > 0.9:  # Far top
            text_y = y_axis_metrics - 0.05
            va = 'top'
        else:
            va = 'bottom'
            
        stationarity_plot_ax.text(text_x, text_y + 0.08, monkey_label, 
                                ha=ha, va=va, fontsize=12, 
                                fontweight='bold', color=COLORS['algorithm1'])

    # # REMOVED: Duplicate MP1 plotting loop - already plotted above
    # for monkey in mp1labels:
    #     idx = next((i for i, item in enumerate(corr_list) if item['monkey'] == monkey and item['strategy'] == '1'), None)
    #     if idx is not None:
    #         m = corr_list[idx]
    #         lsm, wsm = m['non-monotonicity']
    #         stationarity_plot_ax.errorbar(lsm, wsm, 
    #                                     yerr=[[m['error'][1][0]], [m['error'][1][1]]], 
    #                                     xerr=[[m['error'][0][0]], [m['error'][0][1]]], 
    #                                     color=COLORS['algorithm1'], 
    #                                     ecolor=COLORS['algorithm1'], 
    #                                     marker='d', ms=6, alpha=.8, elinewidth=1.5)
    #         # Improved text position to prevent overlap - offset vertically
    #         stationarity_plot_ax.text(lsm+.02, wsm + 0.1, monkey, ha='left', va='bottom', 
    #                                 fontsize=12, fontweight='bold', color=COLORS['algorithm1'])

    # # REMOVED: Fix connections between the quadrant markers and example plots
    # # Define inset data again to prevent issue
    # inset_data = [q1_inset_data, q2_inset_data, q3_inset_data, q4_inset_data]
    
    # # REMOVED: Plot data in the example regressor plots
    # for i, ax in enumerate(plot_list):
    #     # Plot both lines for each inset with correct colors
    #     ax.plot(inset_data[i][0], color=COLORS['strategic'], linewidth=2.5, alpha=0.9)
    #     ax.plot(inset_data[i][1], color=COLORS['non-strategic'], linewidth=2.5, alpha=0.9)
    #     ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    # # Make patches to highlight the regions on the main plot
    # rect_x = 0.075
    # rect_y = 0.05
    # 
    # # Use plot_list instead of inset_axes_list
    # quadrant_names = ['', '', '', '']
    # 
    # for i, ax in enumerate(plot_list):
    #     ax.set_title(quadrant_names[i], fontsize=10)
    # 
    # # Define inset data again to prevent issue
    # inset_data = [q1_inset_data, q2_inset_data, q3_inset_data, q4_inset_data]
    # 
    # # Plot data in the example regressor plots
    # for i, ax in enumerate(plot_list):
    #     # Plot both lines for each inset with correct colors
    #     ax.plot(inset_data[i][0], label='repeat-win')  # Strategic color
    #     ax.plot(inset_data[i][1], label='change-loss')  # Non-strategic color
    #     ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # 
    # # For Q1 (Top Right) - Connect from inner corners
    # q1_rect_top_right = (0.8 + rect_x/2, 0.8 + rect_y/2)
    # q1_rect_bottom_right = (0.8 + rect_x/2, 0.8 - rect_y/2)
    # q1_plot.set_title('Example Regressors \n for Each Quadrant', fontsize = 14)

            
    with open(os.path.join(os.path.dirname(__file__),'fig1_data.pkl'),'wb+') as f:
        #average across nonstrategic monkeys and write lsm, wsm
        # f.write(corr_list)
        pickle.dump(corr_list, f)
        # f.write('{}, {},'.format(np.mean(lsns),np.mean(wsns)))
        
    
    
    extract_sequence_and_fit(mpbeh_path,fig,gs)

    task_im = plt.imread(task_image_p)
    task_ax.imshow(task_im)
    task_ax.set_axis_off()
    task_ax.set_title('Task', fontsize=16, fontweight='bold')
    
    # # REMOVED: Add title to sequence panel - commented out to avoid duplication
    # # Ensure titles are visible for the example regressor plots
    # q1_plot.set_ylabel('1', fontsize=12, rotation=0, ha='left')
    # q2_plot.set_ylabel('2', fontsize=12, rotation=0, ha='left')
    # q3_plot.set_ylabel('3', fontsize=12, rotation=0, ha='left')
    # q4_plot.set_ylabel('4', fontsize=12, rotation=0, ha='left')
    
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5, h_pad=3.5, w_pad=3.0)
    plt.show()
    
    # pass monkey E fit and labels
    # search for monkey E in corr_list and get the fit
    for item in corr_list:
        if item['monkey'] == 'E':
            monkey_E_fit = item['fit']
            monkey_E_labels = item['labels']
            break
    fig, ax = supplemental_figure_metric(monkey_E_fit, monkey_E_labels)
    fig.show()
 
def compute_entropy_and_mutual_information(monkey_data):
    # compute entropy of ws and ls
    mdict = {}
    for monkey in monkey_data:
        # data = monkey_data[monkey_data['animal'] == monkey]
        regressors = parse_monkey_behavior_reduced(monkey,1)
        ws = regressors[:,1]
        ls = regressors[:,2]
        # compute joint entropy of ws and ls
        # Hx = (entropy(ws,ls) + entropy(ls,ws))/2
        # joint distribution of ws and ls is given by:
        # ws and ls are mutually exclusive, so we have P(ws) = , P(ls), and then 
        # we can split them further into P(ws,ls) for 5 total pairs
        # joints = [p(-1,0),p(0,0),p(1,0),p(0,-1),p(0,1)]
        # we know that ls must be 0 if ws is nonzero and vice versa
        n = len(ws)
        joints = [len(ws[ws == -1])/n, len(ws[ws == 1])/n, 
                  len(ls[ls == -1])/n, len(ls[ls == 1])/n,
                    len(ws[np.where((ws == 0) * (ls == 0))])/n]  # 
        
        jws = [joints[0],joints[1],joints[-1]]
        pws = [joints[0],joints[1],len(ws[ws==0])/n]
        jls = joints[2:]
        pls = [joints[2],joints[3],len(ls[ls==0])/n]
        Hx = -1*sum([j*np.log2(j) for j in joints])
        
        # conditional entropy = -sum j*log2 (j/p) where p = the corresponding nonjoint prob
        Hws = -1*sum([j*np.log2(j/p) for j,p in zip(jws,pws)])
        Hls = -1*sum([j*np.log2(j/p) for j,p in zip(jls,pls)])
        
        Ix = Hx - Hws - Hls
        # Corr = np.corrcoef(ws,ls)[0,1]
        # cossim = np.dot(ws,ls)/(np.linalg.norm(ws)*np.linalg.norm(ls))
        # Ix = mutual_info_score(ws,ls)
        # mdict[monkey['animal'].iloc[0]] = (Hx,Ix)
        mdict[monkey['animal'].iloc[0]] = Ix
    return mdict   


def extract_sequence_and_fit(mpbeh_path,fig,gs, strategic=False):
    # plt.rc('font', family='serif',size=20)
    ntrials = 12
    # selects a sequence of 20 trials. fits the logistic regression model to the sequence.
    # then maps the sequence to a scatter plot.
    # First, we get monkey E data and select a sequence of 20 trials
    monkey_beh_data = load_behavior(mpbeh_path,algorithm=2, monkey = 18)   
    # select a random session 
    session_list = list(monkey_beh_data['id'].unique())
    selected = random.choice(session_list)
    # select a random 20 trial sequence
    # start = random.choice(range(len(sequence)-ntrials))
    # start = len(sequence) - ntrials - 1
    # find a sequence with roughly 50% win rate and use that as start. Start from the last session and work backwards
    for i in range(len(session_list)-1,-1,-1):
        start = 75
        selected = sorted(session_list)[-i]
        sequence = monkey_beh_data[monkey_beh_data['id'] == selected]

        seq = sequence.iloc[start:start+ntrials]
        while np.mean(seq['reward'].to_numpy()) < .49 or np.mean(seq['reward'].to_numpy()) > .51:
            start += 1
            seq = sequence.iloc[start:start+ntrials]
        if np.mean(seq['reward'].to_numpy()) > .49 and np.mean(seq['reward'].to_numpy()) < .51:
            break
        
        

    
    
    # need to shift these over by one to get the correct sequence mapping over the reward and choice
    wsls_sequence = sequence.iloc[start-2:start+ntrials]
    sequence = sequence.iloc[start-2:start+ntrials-1]
    ws = (wsls_sequence['monkey_choice'].iloc[:-1].to_numpy() == wsls_sequence['monkey_choice'].iloc[1:].to_numpy()) * (sequence['reward'].to_numpy() == 1)
    ls =  (wsls_sequence['monkey_choice'].iloc[:-1].to_numpy() != wsls_sequence['monkey_choice'].iloc[1:].to_numpy()) * (sequence['reward'].to_numpy() == 0)
    wswitch = (wsls_sequence['monkey_choice'].iloc[:-1].to_numpy() != wsls_sequence['monkey_choice'].iloc[1:].to_numpy()) * (sequence['reward'].to_numpy() == 1)
    lstay = (wsls_sequence['monkey_choice'].iloc[:-1].to_numpy() == wsls_sequence['monkey_choice'].iloc[1:].to_numpy()) * (sequence['reward'].to_numpy() == 0)
    
    # Create axis for definitions
    WSLS_definitions_ax = fig.add_subplot(gs[0:2, 0:3])
    WSLS_definitions_ax.set_axis_off()
    
    # Create axis for sequence - use the updated location to fill blank space
    WSLS_sequence_ax = fig.add_subplot(gs[6:10, 0:8])
    WSLS_sequence_ax.set_axis_off()

    
    # Create arrays with colors for WSLS sequence plot
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    
    # Color mapping in correct order: repeat win, change win, repeat loss, change loss
    # ws (win stay) = repeat win -> colors[0]
    # wswitch (win switch) = change win -> colors[1]
    # lstay (lose stay) = repeat loss -> colors[2]
    # ls (lose switch) = change loss -> colors[3]
    
    colors_m = [colors[4] for x in sequence['monkey_choice']]
    colors_c = [colors[6] for x in sequence['computer_choice']]
    colors_r = ['k' if x == 0 else 'k' for x in sequence['reward']]
    
    colors_ws = ['w' if x == 0 else colors[0] for x in ws]
    colors_wswitch = ['w' if x == 0 else colors[1] for x in wswitch]
    colors_lstay = ['w' if x == 0 else colors[2] for x in lstay]
    colors_ls = ['w' if x == 0 else colors[3] for x in ls]
    
    edges_ws = ['k' if x == 0 else colors[0] for x in ws]
    edges_wswitch = ['k' if x == 0 else colors[1] for x in wswitch]
    edges_lstay = ['k' if x == 0 else colors[2] for x in lstay]
    edges_ls = ['k' if x == 0 else colors[3] for x in ls]
    
    styles_m = ['<' if x == 0 else '>' for x in sequence['monkey_choice']]
    styles_c = ['<' if x == 0 else '>' for x in sequence['computer_choice']]
    
    checkmark = 'o'  # Use circle for checkmark
    cross = 'x'      # Use x for cross
    styles_r = [checkmark if x == 1 else cross for x in sequence['reward']] 
    styles_ws = ['.' if ws[i] == 0 else ('>' if sequence['monkey_choice'].iloc[i] == 1 else '<') for i in range(len(ws))]
    styles_ls = ['.' if ls[i] == 0 else ('>' if sequence['monkey_choice'].iloc[i] == 0 else '<') for i in range(len(ls))]
    styles_wswitch = ['.' if wswitch[i] == 0 else ('>' if sequence['monkey_choice'].iloc[i] == 1 else '<') for i in range(len(wswitch))]
    styles_lstay = ['.' if lstay[i] == 0 else ('>' if sequence['monkey_choice'].iloc[i] == 0 else '<') for i in range(len(lstay))]
    #shift the ws and ls over by one to match the sequence
    styles_ws = styles_ws[:-1]
    styles_ls = styles_ls[:-1]
    colors_ws = colors_ws[:-1]
    colors_ls = colors_ls[:-1]
    edges_ws = edges_ws[:-1]
    edges_ls = edges_ls[:-1]
    styles_wswitch = styles_wswitch[:-1]
    styles_lstay = styles_lstay[:-1]
    colors_wswitch = colors_wswitch[:-1]
    colors_lstay = colors_lstay[:-1]
    edges_wswitch = edges_wswitch[:-1]
    edges_lstay = edges_lstay[:-1]
    
    

    full_labels = ['monkey choice', 'computer choice', 'reward', 'repeat win', 'change win', 'repeat loss', 'change loss']
    
    for p in range(len(colors)):
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],5*np.ones(ntrials)[p], c = colors_m[p], marker = styles_m[p])
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],4*np.ones(ntrials)[p], c = colors_c[p], marker = styles_c[p])
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],3*np.ones(ntrials)[p], c = colors_r[p], marker = styles_r[p])
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],2*np.ones(ntrials)[p], c = colors_ws[p], marker = styles_ws[p], edgecolors = edges_ws[p])  # repeat win
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],1*np.ones(ntrials)[p], c = colors_wswitch[p], marker = styles_wswitch[p], edgecolors = edges_wswitch[p])  # change win
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],0*np.ones(ntrials)[p], c = colors_lstay[p], marker = styles_lstay[p], edgecolors = edges_lstay[p])  # repeat loss
        WSLS_sequence_ax.scatter(np.arange(ntrials)[p],-1*np.ones(ntrials)[p], c = colors_ls[p], marker = styles_ls[p], edgecolors = edges_ls[p])  # change loss
        # if strategic:
            

    # label each row by annotating first point in sequence
    # Abbreviated labels for left-side annotations
    abbrev_labels = ['m', 'c', 'r', 'rw', 'cw', 'rl', 'cl']
    for i in range(len(abbrev_labels)):
        WSLS_sequence_ax.annotate(abbrev_labels[i], (0,5-i), xytext=(-1.25, 4.75-i), fontsize = 12)
    plt.rc('text', usetex=True)

    # eqns = [r'$r = (m \land c) \lor (\neg m \land \neg c)$', r'Stay: $m_t \leftrightarrow m_{t-1}$', 
    #         r'win stay: $r_t \land s_t$', r'lose switch: $\neg r_t \land \neg s_t$',
    #         r'ws+ls: $r_t \odot  s_t$']
    
    WSLS_sequence_ax.set_xticks([])
    WSLS_sequence_ax.set_yticks([5,4,3,2,1,0,-1], labels = full_labels) # Use full names for yticks
    # WSLS_sequence_ax.set_axis_off()
    
    # WSLS_sequence_ax.xaxis.set_visible(False)
    # make spines (the box) invisible
    plt.setp(WSLS_sequence_ax.spines.values(), visible=False)
    # remove ticks and labels for the left axis
    WSLS_sequence_ax.yaxis.set_visible(False)
    WSLS_sequence_ax.tick_params(bottom=False, labelbottom=False)
    
    # Add title to the sequence plot
    WSLS_sequence_ax.set_title('Sequence', fontsize=16, fontweight='bold', pad=10)


def supplemental_figure_metric(monkey_fit, labels):
    # plot monkey E regression, then intermediate steps showing how we get to
    # the metric for monkey E
    # maybe make it a 1x2 fig where the left is the regression and the right is the intermediate step
    # of taking the difference between the two coefficients
    
    # load mp2 data for monkey E
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.arange(1, 5+1), monkey_fit)
    ax[0].legend(labels)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax[1].plot(np.arange(1, 5+1), monkey_fit[0] - monkey_fit[1],color=colors[5]) # fifth color in color cycle
    ax[1].plot(np.arange(1, 5+1), monkey_fit[3] - monkey_fit[2],color=colors[6]) # sixth color in color cycle
    ax[1].legend(['repeat win - change win', 'change loss - repeat loss'])
    ax[1].set_xlabel('Trial Lag', fontsize=12)
    ax[1].set_ylabel(r'\Delta\beta', fontsize=12)
    ax[0].set_xlabel('Trial Lag', fontsize=12)
    ax[0].set_ylabel(r'\beta', fontsize=12)
    ax[0].set_title('Monkey Fit', fontsize=14)
    ax[1].set_title('Intermediate Steps', fontsize=14)
    fig.suptitle(r'Monkey E \xi_{win} and \xi_{loss} Computation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return fig, ax