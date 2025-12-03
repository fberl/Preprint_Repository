import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import pickle
import sys
import ast
import itertools
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import fit_glr
from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior, general_logistic_regressors
from analysis_scripts.stationarity_and_randomness import wsls_autocorrelation, wsls_crosscorrelation
from matplotlib.markers import MarkerStyle
import matplotlib as mpl
from analysis_scripts.entropy import *
from analysis_scripts.LLH_behavior_RL import multi_session_fit, cross_validated_performance_sessions, cross_validated_performance_by_monkey_df
import numpy as np
from figure_scripts.fig3_RNN_insufficiency import compute_all_frechets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, ttest_ind
from matplotlib.lines import Line2D
from figure_scripts.fig3_RNN_insufficiency import *
from figure_scripts.fig1_monkey_RL_RNN_comparison import *
from figure_scripts.fig1_monkey_RL_RNN_comparison import partition_dataset, cutoff_trials_by_session, calculate_decay_metric_simple

from figure_scripts.fig3_RNN_insufficiency import cosine_distance
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

RLRNN_PARAM_STYLES = {
    (0.9, 0.0): {
        'cmap': 'Blues',
        'contour_color': '#1f77b4',
        'marker_color': '#1f77b4',
        'scatter_color': 'tab:blue',
        'label': 'RLRNN (0.9, 0)'
    },
    (1.0, 0.0): {
        'cmap': 'Oranges',
        'contour_color': '#d35400',
        'marker_color': '#d35400',
        'scatter_color': 'tab:orange',
        'label': 'RLRNN (1, 0)'
    },
    (1.0, 0.1): {
        'cmap': 'Greens',
        'contour_color': '#2ca02c',
        'marker_color': '#2ca02c',
        'scatter_color': 'tab:green',
        'label': 'RLRNN (1, 0.1)'
    }
}
# RLRNN_500_FITS_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/RLRNN_500_fits.pkl'
RNN_500_ZOO_PATH = os.path.join(os.path.dirname(__file__), '..', 'figure_scripts', 'fig3data', 'RNN_zoo_dict.pkl')
RLRNN_ZOO_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/RLRNN_zoo_dict.pkl'
RLRNN_BEST_FITS_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/strategic_and_nonstrategic_best_param_fits.pkl'
BEST_RUN_ID_OFFSET = 1000

# RLRNN_BEST_FITS_PATH = RLRNN_ZOO_PATH

def load_rlrnn_zoo(path):
    """Load the RLRNN zoo data from a pickle file."""
    pickle_dict = {}
    with open(path, 'rb') as f:
        while True:
            try:
                param_tuple, fits_dict = pickle.load(f)
                pickle_dict[param_tuple] = fits_dict
            except EOFError:
                break
    return pickle_dict

def generate_model_figure(pfcw_params, pfc_params, mpdb_path, env_params,img_path, nits = 2, ep = None, order=5, bias = True, strategic=True, fit_single=False, perf = False, use_median=True, power_scale=2.5, n_top_models=100,
                        distance_metric='area_norm', use_subset='all', combined_regressors=False, skip_model_loading=False, make_extra_figures=False, cv_folds=10, bootstrap_iters=1000, cv_random_state=0,
                        override_strategic_param=None, override_nonstrategic_param=None, layout_debug: bool = False, use_constrained_layout: bool = False,
                        export_asymmetry_progress: bool = False, asymmetry_progress_dir: str = None, cached: bool = True,
                        exclude_monkeys=None,
                        strategic_candidate_params=None,
                        nonstrategic_candidate_params=None,
                        density_param_list=None):
    env = make_env(env_params)
    
    mp_data = query_monkey_behavior(mpdb_path)
    mp_data = mp_data[(mp_data['task'] == 'mp')]

    # BG_model = BG(alpha = .2, gamma = [.3,-.15], env = env, asymmetric= True, load=False)

    
    RLRNN_model = RLRNN(environment = env, **pfc_params)
    if not skip_model_loading:
        if ep is None:
            RLRNN_model.load_model('')
        else:
            RLRNN_model.load_model_ep(ep)
            
    exclude_monkeys = set(exclude_monkeys or [])

    def _normalize_param_list(param_list, default):
        if param_list is None:
            return list(default)
        normalized = []
        for p in param_list:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                normalized.append((float(p[0]), float(p[1])))
            else:
                raise ValueError(f"Parameter specification {p} is not a length-2 tuple/list.")
        return normalized

    strategic_candidate_params = _normalize_param_list(
        strategic_candidate_params,
        default=[(1.0, 0.0), (1.0, 0.1)]
    )
    nonstrategic_candidate_params = _normalize_param_list(
        nonstrategic_candidate_params,
        default=[(0.9, 0.0), (1.0, 0.0)]
    )

    if density_param_list is None:
        density_param_list = [(0.9, 0.0), (1.0, 0.0), (1.0, 0.1)]
    else:
        density_param_list = _normalize_param_list(density_param_list, default=[])
    # Preserve order while ensuring uniqueness
    seen = set()
    density_param_list = [p for p in density_param_list if not (p in seen or seen.add(p))]

    nonstrategic_monkeys = ['C', 'H', 'F', 'K']  # Non-strategic monkeys
    strategic_monkeys = ['E', 'D', 'I']  # Strategic monkeys

    nonstrategic_monkeys = [m for m in nonstrategic_monkeys if m not in exclude_monkeys]
    strategic_monkeys = [m for m in strategic_monkeys if m not in exclude_monkeys]

    monkey_dict = {'strategic': strategic_monkeys, 'nonstrategic': nonstrategic_monkeys}
    # Create figure without constrained layout (constrained will be enabled before returning)
    # Slightly taller to prevent title clipping
    fig = plt.figure(figsize=(28.5, 9.5), dpi=600)
    fig.suptitle('Figure 5: RLRNN Model Yields Monkey-Like Strategic Deviations from RL', fontsize=26, y=0.965)
    # make gridpsec
    # columns: RNN plot, asymmetry plot, cosine similarity, performance plot and maybe MI below? 
    # rows: 1? but do it with 3 in case we want the bottom of RNN plot to be modules and MI
    gs = gridspec.GridSpec(4,4,figure=fig,wspace=0.35,hspace=0.53)
    
    RLRNN_ax_s = fig.add_subplot(gs[:2,0])
    RLRNN_ax_ns = fig.add_subplot(gs[2:,0])
    Asymmetry_ax = fig.add_subplot(gs[:,1])
    Cosine_similarity_ax = fig.add_subplot(gs[:,2])
    Performance_ax = fig.add_subplot(gs[:2,3])
    MI_ax = fig.add_subplot(gs[2:,3])
    
    # Add letter labels to each subplot
    label_props = dict(fontsize=20, fontweight='bold', va='top', ha='left')
    RLRNN_ax_s.text(-0.15, 1.05, 'A', transform=RLRNN_ax_s.transAxes, **label_props)
    RLRNN_ax_ns.text(-0.15, 1.05, 'B', transform=RLRNN_ax_ns.transAxes, **label_props)
    Asymmetry_ax.text(-0.15, 1.02, 'C', transform=Asymmetry_ax.transAxes, **label_props)
    Cosine_similarity_ax.text(-0.15, 1.02, 'D', transform=Cosine_similarity_ax.transAxes, **label_props)
    Performance_ax.text(-0.15, 1.05, 'E', transform=Performance_ax.transAxes, **label_props)
    MI_ax.text(-0.15, 1.05, 'F', transform=MI_ax.transAxes, **label_props)
    
    RLRNN_ax_dict = {'strategic': RLRNN_ax_s, 'nonstrategic': RLRNN_ax_ns}
    for group, ax in RLRNN_ax_dict.items():
        ax.set_xlabel('Trial Lag', fontsize=14, labelpad=5)
        ax.set_ylabel('Coefficient Value', fontsize=14, labelpad=5)
        # ax.set_title(f'Representative RLRNN ({group.capitalize()}, params={params_by_group[group]})', fontsize=16, fontweight='bold', pad=10)
        if group == 'strategic':
            ax.legend(
                loc='upper right',
                frameon=False,
                fontsize=10,
                title='RLRNN Coefficients',
                title_fontsize=12
            )
       
    # for asymmetry plot, need to load monkey data from fig1
    fig1_data = pickle.load(open(os.path.join(os.path.dirname(__file__), 'fig1_data.pkl'), 'rb'))
    # def compute_representative_rlrnn(model_dict, ignore_dict):
    #     # given a list of models, computes the central model
    #     # the central model is that one that maximizes similarity to the other  models 
    #     # use cosine similarity to compute the closest fit
    #     # return the fit and the cosine similarity
    #     best_fit = None
    #     best_model_idx = None
    #     best_avg_similarity = -float('inf')

    #     for model_idx, model_fit in model_dict.items():
    #         for model_idx2, model_fit2 in model_dict.items():
    #             if model_idx2 != model_idx:
    #                 similarity = cosine_distance(model_fit, model_fit2)
    #                 if similarity > best_avg_similarity:
    #                     best_avg_similarity = similarity
    #                     best_fit = model_fit
    #                     best_model_idx = model_idx

    #     return best_fit, best_model_idx, best_avg_similarity


    def compute_representative_rlrnn(model_dict, ignore_dict):
        # given a list of model fits and a list of monkey fits, compute the representative RLRNN fit
        # the representative RLRNN fit is the fit that is most similar to the monkey fits 
        # use cosine similarity to compute the closest fit
        # return the fit and the cosine similarity
        best_fit = None
        best_model_idx = None
        best_avg_similarity = -float('inf')

        for model_idx, model_fit in model_dict.items():
            try:
                similarities = [cosine_distance(model_fit, monkey_fit) for monkey_fit in monkey_fits]
            except:
                similarities = [cosine_distance(model_fit[:-1], monkey_fit) for monkey_fit in monkey_fits]
            avg_similarity = np.mean(similarities)
            
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_fit = model_fit
                best_model_idx = model_idx
        print(f"Best model idx: {best_avg_similarity}")

        return best_fit, best_model_idx, best_avg_similarity
 

    RLRNN_zoo_dict = load_rlrnn_zoo(RLRNN_ZOO_PATH)
    RLRNN_best_param_fits = load_rlrnn_zoo(RLRNN_BEST_FITS_PATH)
    if not RLRNN_best_param_fits:
        raise RuntimeError(
            "strategic_and_nonstrategic_best_param_fits.pkl is empty. "
            "Regenerate the strategic/nonstrategic best-fit cache before running Fig. 5."
        )
    
    monkey_fits_by_group = {'strategic': [], 'nonstrategic': []}
    for data in fig1_data:
        strategy = data['strategy']
        if data.get('monkey') in exclude_monkeys:
            continue
        if strategy == 'strategic':
            monkey_fits_by_group['strategic'].append(data['fit'][:-1])
        elif strategy == 'non-strategic':
            monkey_fits_by_group['nonstrategic'].append(data['fit'][:-1])
    
    # set up the asymmetry plot like in fig1
    Asymmetry_ax.set_xlabel('Loss Deviation', fontsize=16, labelpad=5)
    Asymmetry_ax.set_ylabel('Win Deviation', fontsize=16, labelpad=5)
    
    Asymmetry_ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    Asymmetry_ax.axvline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    # Add diagonal line from (0,0) to (1,1)
    Asymmetry_ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
    # Add title to the stationarity plot with conditional formatting
    Asymmetry_ax.set_title('Behavioral Asymmetry', fontsize=16, fontweight='bold', pad=10)
    # Remove grid
    Asymmetry_ax.grid(False)
    # Expand axis limits to show all monkeys
    Asymmetry_ax.set_xlim(-1.5, 1.5)
    Asymmetry_ax.set_ylim(-1.5, 1.5)
    # Create light green shaded region for RL model space (rectangle)
    Asymmetry_ax.fill_between([0, 1], [0, 0], [1, 1], color='lightgreen', alpha=0.3, label='RL')
    
    #plot on the asymmetry plot the monkeys from fig1
    # Track which groups have been plotted for legend
    strategic_plotted = False
    nonstrategic_plotted = False
    algorithm_plotted = False
    
    for data in fig1_data:
        monkey = data['monkey']
        if monkey in exclude_monkeys:
            continue
        if data['strategy'] == 'non-strategic':
            # Use circle marker for non-strategic monkeys (consistent with fig1)
            label = 'Non-strategic' if not nonstrategic_plotted else None
            Asymmetry_ax.scatter(data['non-monotonicity'][0], data['non-monotonicity'][1], label=label, color='tab:purple', marker='o', s=120)
            # Add individual monkey label
            Asymmetry_ax.text(data['non-monotonicity'][0] + 0.02, data['non-monotonicity'][1] + 0.02, monkey, 
                            ha='left', va='bottom', fontsize=10, color='tab:purple')
            nonstrategic_plotted = True
        elif data['strategy'] == 'strategic':
            # Use star marker for strategic monkeys (consistent with fig1)
            label = 'Strategic' if not strategic_plotted else None
            Asymmetry_ax.scatter(data['non-monotonicity'][0], data['non-monotonicity'][1], label=label, color='tab:cyan', marker='*', s=120)
            # Add individual monkey label
            Asymmetry_ax.text(data['non-monotonicity'][0] + 0.02, data['non-monotonicity'][1] + 0.02, monkey, 
                            ha='left', va='bottom', fontsize=10, color='tab:cyan')
            strategic_plotted = True
        elif data['strategy'] == '1':
            # Use diamond marker for algorithm monkeys (consistent with fig1)
            label = 'Algorithm' if not algorithm_plotted else None
            Asymmetry_ax.scatter(data['non-monotonicity'][0], data['non-monotonicity'][1], label=label, color='#F2B705', marker='d', s=120)
            # Add individual monkey label with extra offset for algorithm monkeys
            Asymmetry_ax.text(data['non-monotonicity'][0] + 0.02, data['non-monotonicity'][1] + 0.08, monkey, 
                            ha='left', va='bottom', fontsize=10, color='#F2B705')
            algorithm_plotted = True
    
    # load BG model data
    
    # Load RL data from fig1
    rl_fit_path = os.path.join(os.path.dirname(__file__), 'fig5_rl_data.p')
    #assume these are raw coefficients and errors on the raw coefficients.  If they don't exist, we can create it
        

    # if os.path.exists(rl_fit_path) and cached==True:
    #     print(f"Loading existing BG/RL data from {rl_fit_path}")
    #     with open(rl_fit_path, 'rb') as f:
    #         rl_dict = pickle.load(f)
    # else:
    #     # create BG object fit to all monkey data
    #     bg_env_params = env_params.copy() if env_params else {}
    #     bg_env_params["opponents"] = ["all"]
    #     bg_env_params["opponent"] = "all"
    #     bg_env_params["opponents_params"] = {"all": {"bias": [0], "depth": 4}}
    #     # Fix: ensure fixed_length and reset_time are consistent
    #     bg_env_params["fixed_length"] = True
    #     # Fix: Use proper session length instead of bg_nits for reset_time
    #     # bg_nits should only control number of sessions, not session length
    #     bg_env_params["reset_time"] = 200  # Each session should be 200 trials long
        
    #     bg_env = make_env(bg_env_params)
                
    #     # RL_params = {'alpha' : 1-.911, 'gamma' : [.137,-.07], 'asymmetric' : True, 'load' : False, 'beta' : .93, 'forgetting' : True}
    #     RL_params = {'alpha' : 1, 'gamma' : 0, 'asymmetric' : False, 'load' : False, 'beta' : 1, 'forgetting' : False, 'deterministic' : False}
    #     # Create BG model with parameters
    #     BG_model = BG(env=bg_env, **BG_params)
        
    #     # Generate data from BG model
    #     print(f"Generating BG model data with {nits} sessions...")
    #     BG_data, masks = BG_model.generate_data(nits=250)
    #     rl_fit, err_rl_fit, labels_rl_fit = fit_glr(BG_data, order=order, a_order=2, r_order=1, model=False, err=True, labels=True, average=True)
    #     # save fit. split BG into components we can use through labels
    #     coeffs = {labels_rl_fit[j]: rl_fit[j*order:(j+1)*order] for j in range(len(labels_rl_fit))}

    #     rl_dict = {'BG_fit': coeffs, 'err_BG_fit': err_rl_fit, 'labels_BG_fit': labels_rl_fit}
    #     with open(rl_fit_path, 'wb') as f:
    #         pickle.dump(rl_dict, f)
    #     print(f"Saved BG data to {rl_fit_path}")
        

    # # compute RL metric
    # labels_rl_fit = rl_dict['labels_BG_fit']
    # rl_metric_win = calculate_decay_metric_simple(rl_dict['BG_fit'][labels_rl_fit[0]]-rl_dict['BG_fit'][labels_rl_fit[1]])
    # rl_metric_loss = calculate_decay_metric_simple(rl_dict['BG_fit'][labels_rl_fit[3]]-rl_dict['BG_fit'][labels_rl_fit[2]])
        
    # # Plot RL point
    # Asymmetry_ax.scatter(rl_metric_loss, rl_metric_win, label='RL Model (Fitted)', color='tab:blue', marker='D', s=120)
        
    # #load RNN model
    # rnn_model_path = os.path.join(os.path.dirname(__file__), 'fig5_rnn_model.p')
    # if os.path.exists(rnn_model_path): 
    #     with open(rnn_model_path, 'rb') as f:
    #         rnn_dict = pickle.load(f)
    #     rnn_metric_win = calculate_decay_metric_simple(rnn_dict['BG_fit'][labels_rl_fit[0]]-rnn_dict['BG_fit'][labels_rl_fit[1]])
    #     rnn_metric_loss = calculate_decay_metric_simple(rnn_dict['BG_fit'][labels_rl_fit[3]]-rnn_dict['BG_fit'][labels_rl_fit[2]])
    #     # plot RNN point
    #     Asymmetry_ax.scatter(rnn_metric_loss, rnn_metric_win, label='RNN Model (Fitted)', color='tab:red', marker='X', s=120)
    # else:
    #     print(f"RNN model not found at {rnn_model_path}")
    
    
    # load RLRNN
    # first check if precomputed RLRNN data exists
    # fig5dir = os.path.join(os.path.dirname(__file__), 'fig5data')
    # rep_rlrnn_coeffs_path = os.path.join(os.path.dirname(__file__), 'fig5_rep_rlnn_coeffs.p')
    # if os.path.exists(rep_rlrnn_coeffs_path) and cached==True:
    #     with open(rep_rlrnn_coeffs_path, 'rb') as f:
    #         rep_rlrnn_coeffs = pickle.load(f)
    # else: # fit regression to data and save 
    #     rep_RLRNN_path= os.path.join(fig5dir, '(0.5,0)_226_data.p')
    #     with open(rep_RLRNN_path, 'rb') as f:
    #         rep_RLRNN_data = pickle.load(f)
        
    #     RL_module_data = [rep_RLRNN_data[0],np.array(rep_RLRNN_data[-1]),rep_RLRNN_data[2],rep_RLRNN_data[3],None,None]
    #     RNN_module_data = [rep_RLRNN_data[0],np.array(rep_RLRNN_data[-2]),rep_RLRNN_data[2],rep_RLRNN_data[3],None,None]
    #     rep_rlrnn_coeffs_raw, err_rep_rlrnn_coeffs_raw, labels_rep_rlrnn_coeffs = fit_glr(rep_RLRNN_data, order=order, a_order=2, r_order=1, model=True, err=True, labels=True, average=True)
    #     fit_rlrnn_coeffs = {labels_rep_rlrnn_coeffs[j]: rep_rlrnn_coeffs_raw[j*order:(j+1)*order] for j in range(len(labels_rep_rlrnn_coeffs))}
    #     err_rlrnn_coeffs = {labels_rep_rlrnn_coeffs[j]: err_rep_rlrnn_coeffs_raw[j*order:(j+1)*order] for j in range(len(labels_rep_rlrnn_coeffs))}
    #     rep_rlrnn_coeffs = {'RL_module_coeffs': fit_rlrnn_coeffs, 'err_RL_module_coeffs': err_rlrnn_coeffs, 'labels_RL_module_coeffs': labels_rep_rlrnn_coeffs}
    #     with open(rep_rlrnn_coeffs_path, 'wb') as f:
    #         pickle.dump(rep_rlrnn_coeffs, f)

    # labels_rep_rlrnn_coeffs = rep_rlrnn_coeffs['labels_RL_module_coeffs']
    # rep_rlrnn_metric_win = calculate_decay_metric_simple(rep_rlrnn_coeffs['RL_module_coeffs'][labels_rep_rlrnn_coeffs[0]]-rep_rlrnn_coeffs['RL_module_coeffs'][labels_rep_rlrnn_coeffs[1]])
    # rep_rlrnn_metric_loss = calculate_decay_metric_simple(rep_rlrnn_coeffs['RL_module_coeffs'][labels_rep_rlrnn_coeffs[3]]-rep_rlrnn_coeffs['RL_module_coeffs'][labels_rep_rlrnn_coeffs[2]])
    # Asymmetry_ax.scatter(rep_rlrnn_metric_loss, rep_rlrnn_metric_win, label='RLRNN Model', color='tab:olive', marker='P', s=120)
    
    group_mi = {'strategic': None, 'nonstrategic': None}
    prepared_group_data = {}
    available_params = sorted(RLRNN_best_param_fits.keys(), key=lambda p: (p[0], p[1]))
    if len(available_params) < 2:
        raise RuntimeError(
            "strategic_and_nonstrategic_best_param_fits.pkl must contain at least two parameter tuples "
            "(one for nonstrategic, one for strategic)."
        )
        
    # plot rnn zoo density
    # load RNN zoo data from RNN_zoo_dict.pkl (fig3 format)
    rnn_zoo_path = RNN_500_ZOO_PATH
    if os.path.exists(rnn_zoo_path):
        with open(rnn_zoo_path, 'rb') as f:
            rnn_zoo_dict = pickle.load(f)
        
        # Initialize lists to store metrics for all RNNs
        rnn_ls_values = []
        rnn_ws_values = []
        
        # RNN_zoo_dict format: {rnn_idx: {'action': [coefficients]}}
        # The action array has the same structure as RLRNN
        for rnn_key, rnn_data in rnn_zoo_dict.items():
            if not isinstance(rnn_data, dict) or 'action' not in rnn_data:
                continue
            
            action = np.asarray(rnn_data['action'])
            ws = action[0:order]
            wsw = action[order:2*order]
            lst = action[2*order:3*order]
            ls = action[3*order:4*order]
            # Compute combined regressor metric
            ws_metric = calculate_decay_metric_simple(ws - wsw)
            ls_metric = calculate_decay_metric_simple(ls - lst)
            rnn_ws_values.append(ws_metric)
            rnn_ls_values.append(ls_metric)
        
        print(f"Loaded {len(rnn_ls_values)} RNN models from zoo (RNN_zoo_dict.pkl format)")
    
    
    # Overlay RNN zoo density
    n_rnn = overlay_model_density(
        ax=Asymmetry_ax,
        model_type='rnn',
        ls_vals=rnn_ls_values,  # Array of loss metric for each RNN
        ws_vals=rnn_ws_values,  # Array of win metric for each RNN
        alpha=0.7,
        mask_percentile=95.0,
        scatter_points=True,
        n_contours=4,
        plot_representative=True
    )
    
    
    
    # plot cosine similarity
    # Cosine_similarity_ax.plot(RLRNN_model.model_history)
    # 
    Cosine_similarity_ax.set_title('Model Parameter Sweep', fontsize=16, fontweight='bold', pad=10)
    Cosine_similarity_ax.set_xlabel('Decision Layer Loss Parameter', fontsize=16, labelpad=5)
    Cosine_similarity_ax.set_ylabel('Decision Layer Win Parameter', fontsize=16, labelpad=5)

    def _offset_run_ids(run_dict, offset):
        adjusted = {}
        for run_idx, run_data in run_dict.items():
            if isinstance(run_idx, int):
                adjusted[run_idx + offset] = run_data
            else:
                adjusted[run_idx] = run_data
        return adjusted

    combined_model_data = {}
    for param, run_dict in RLRNN_zoo_dict.items():
        if isinstance(run_dict, dict):
            combined_model_data[param] = dict(run_dict)
    for param, run_dict in RLRNN_best_param_fits.items():
        if not isinstance(run_dict, dict):
            continue
        combined_runs = combined_model_data.setdefault(param, {})
        combined_runs.update(_offset_run_ids(run_dict, BEST_RUN_ID_OFFSET))

    _, best_params_for_rlrnn, monkey_best_params = similarity_comparison_plot(
        Cosine_similarity_ax,
        combined_model_data,
        cached=cached,
        performance=False,
        exclude_monkeys=exclude_monkeys
    )
    
    if monkey_best_params:
        strategic_assignments = [monkey_best_params.get(m) for m in strategic_monkeys if monkey_best_params.get(m) is not None]
        nonstrategic_assignments = [monkey_best_params.get(m) for m in nonstrategic_monkeys if monkey_best_params.get(m) is not None]

        strategic_summary = Counter(strategic_assignments)
        nonstrategic_summary = Counter(nonstrategic_assignments)

        print("Per-monkey best parameter assignments (strategic):")
        for param, count in strategic_summary.most_common():
            print(f"  {param}: {count} monkey(s)")
        print("Per-monkey best parameter assignments (nonstrategic):")
        for param, count in nonstrategic_summary.most_common():
            print(f"  {param}: {count} monkey(s)")

    params_by_group = {}
    candidate_params_by_group = {
        'strategic': strategic_candidate_params,
        'nonstrategic': nonstrategic_candidate_params
    }

    if best_params_for_rlrnn:
        for group, candidate_list in candidate_params_by_group.items():
            candidate_set = set(candidate_list)
            chosen = best_params_for_rlrnn.get(group)
            if chosen in candidate_set:
                params_by_group[group] = chosen
            elif candidate_list:
                params_by_group[group] = candidate_list[0]
    else:
        for group, candidate_list in candidate_params_by_group.items():
            if candidate_list:
                params_by_group[group] = candidate_list[0]

    param_cache = {}
    param_metrics = {}
    param_idx_metadata = {}

    def fetch_param_data(param):
        if param in param_cache:
            return param_cache[param]

        raw_runs = RLRNN_best_param_fits.get(param)
        source = 'best'
        offset = BEST_RUN_ID_OFFSET

        if not isinstance(raw_runs, dict) or not raw_runs:
            raw_runs = RLRNN_zoo_dict.get(param, {})
            source = 'zoo'
            offset = 0

        if not isinstance(raw_runs, dict) or not raw_runs:
            raise RuntimeError(
                f"No RLRNN runs found for parameter tuple {param} in either best-fit cache or zoo dictionary."
            )

        cleaned = {}
        metadata = {}
        for idx, data in raw_runs.items():
            if not isinstance(idx, int):
                continue
            new_idx = idx + offset if offset else idx
            cleaned[new_idx] = data
            metadata[new_idx] = {
                'source': source,
                'original_run_idx': idx,
                'offset': offset
            }

        if not cleaned:
            raise RuntimeError(
                f"No integer-indexed runs available for parameter tuple {param}."
            )

        param_cache[param] = cleaned
        param_idx_metadata[param] = metadata
        return cleaned

    def compute_param_metrics(param):
        cleaned = fetch_param_data(param)
        if param not in param_metrics:
            ls_vals = []
            ws_vals = []
            mi_vals = []
            for run_data in cleaned.values():
                action = np.asarray(run_data['action'])
                ws = action[0:order]
                wsw = action[order:2*order]
                lst = action[2*order:3*order]
                ls = action[3*order:4*order]
                ws_vals.append(calculate_decay_metric_simple(ws - wsw))
                ls_vals.append(calculate_decay_metric_simple(ls - lst))
                mi_vals.append(run_data['mutual_information']['mean_mutual_info'][-3])
            param_metrics[param] = {'ls': ls_vals, 'ws': ws_vals, 'mi': mi_vals}
        return cleaned, param_metrics[param]

    for group, candidate_list in candidate_params_by_group.items():
        combined_coeffs = {}
        combined_errs = {}
        id_map = {}
        mi_values = []

        for param in candidate_list:
            cleaned, metrics = compute_param_metrics(param)
            print(f"Loaded {len(cleaned)} RLRNN runs for group '{group}' candidate {param}")
            mi_values.extend(metrics['mi'])
            metadata_lookup = param_idx_metadata.get(param, {})
            for run_idx, run_data in cleaned.items():
                key = len(combined_coeffs)
                combined_coeffs[key] = np.asarray(run_data['action'])
                combined_errs[key] = np.asarray(run_data['err']) if 'err' in run_data else np.zeros_like(run_data['action'])
                run_meta = metadata_lookup.get(run_idx, {})
                id_map[key] = {
                    'param': param,
                    'run_idx': run_idx,
                    'source': run_meta.get('source', 'unknown'),
                    'original_run_idx': run_meta.get('original_run_idx', run_idx)
                }

        if not combined_coeffs:
            raise RuntimeError(
                f"No RLRNN runs available for group '{group}' using candidate parameters {candidate_list}."
            )

        prepared_group_data[group] = {
            'coeffs': combined_coeffs,
            'errs': combined_errs,
            'id_map': id_map
        }
        group_mi[group] = float(np.mean(mi_values)) if mi_values else float('nan')

    # Overlay parameter-specific densities on the asymmetry plot
    for param in density_param_list:
        _, metrics = compute_param_metrics(param)
        model_label = f"rlrnn_param:{param[0]:.3f},{param[1]:.3f}"
        overlay_model_density(
            ax=Asymmetry_ax,
            model_type=model_label,
            ls_vals=metrics['ls'],
            ws_vals=metrics['ws'],
            alpha=0.6,
            mask_percentile=95.0,
            scatter_points=True,
            scatter_alpha=0.08,
            n_contours=4,
            plot_representative=True
        )

    Asymmetry_ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        frameon=False,
        ncol=1,
        fontsize=8,
        title='Model Distributions',
        title_fontsize=10
    )

    fig5_data_dir = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data'

    for group, ax in RLRNN_ax_dict.items():
        data_info = prepared_group_data.get(group)
        if not data_info:
            raise RuntimeError(f"No prepared RLRNN runs available for group '{group}'.")

        group_model_coeffs = data_info['coeffs']
        group_model_errs = data_info['errs']
        id_map = data_info['id_map']

        monkey_fits = monkey_fits_by_group[group]

        best_fit_coeffs, best_model_idx, best_similarity = compute_representative_rlrnn(group_model_coeffs, monkey_fits)

        if best_model_idx is None:
            print(f"Could not find a representative RLRNN for {group} group using the specified candidate parameters.")
            continue

        selected_info = id_map[best_model_idx]
        selected_param = selected_info['param']
        selected_run_idx = selected_info['run_idx']
        selected_source = selected_info.get('source', 'unknown')
        original_run_idx = selected_info.get('original_run_idx', selected_run_idx)
        params_by_group[group] = selected_param

        model_path = os.path.join(fig5_data_dir, f'({selected_param[0]},{selected_param[1]})_{original_run_idx}_data.p')

        fit_rlrnn_coeffs = best_fit_coeffs
        fit_rlrnn_errs = group_model_errs[best_model_idx]

        print(
            f"Representative RLRNN ({group}): param={selected_param}, run={selected_run_idx}"
            f" [{selected_source}, original={original_run_idx}], similarity={best_similarity:.4f}"
        )

        y = fit_rlrnn_coeffs
        err = fit_rlrnn_errs
        
        ax.plot(y[0:order], label='repeat win')
        ax.fill_between(list(range(order)), y[0:order] - err[0:order], y[0:order] + err[0:order], alpha=0.25)
        ax.plot(y[order:2*order], label='change win')
        ax.fill_between(list(range(order)), y[order:2*order] - err[order:2*order], y[order:2*order] + err[order:2*order], alpha=0.25)
        ax.plot(y[2*order:3*order], label='repeat lose')
        ax.fill_between(list(range(order)), y[2*order:3*order] - err[2*order:3*order], y[2*order:3*order] + err[2*order:3*order], alpha=0.25)
        ax.plot(y[3*order:4*order], label='change lose')
        ax.fill_between(list(range(order)), y[3*order:4*order] - err[3*order:4*order], y[3*order:4*order] + err[3*order:4*order], alpha=0.25)
        
        if group == 'strategic':
            ax.legend(
                loc='upper right',
                frameon=False,
                fontsize=10,
                title='RLRNN Coefficients',
                title_fontsize=12
            )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
        ax.set_xlabel('Trial Lag', fontsize=14, labelpad=5)
        ax.set_ylabel('Coefficient Value', fontsize=14, labelpad=5)
        ax.set_title(
            f'Representative RLRNN ({group.capitalize()}) params={selected_param}',
            fontsize=14,
            fontweight='bold',
            pad=6
        )
    
    # Plot performance histogram
    print("Adding performance histogram...")
    add_performance_histogram(
        Performance_ax, 
        bootstrap_iters=bootstrap_iters, 
        random_state=cv_random_state, 
        cached=cached,
        strategic_candidate_params=strategic_candidate_params,
        nonstrategic_candidate_params=nonstrategic_candidate_params,
        rlrnn_zoo_dict=RLRNN_zoo_dict,
        rlrnn_best_param_fits=RLRNN_best_param_fits
    )
    print("Performance histogram completed.")
    
    # plot MI histogram for computer sequence of length 4 for completely random agent, strategic monkeys,
    #           nonstrategic monkeys, RNN, and (0.5,0) RLRNNs
    # once we have (0.9,0) and (0.2,0.1) RLRNNs, we can replace them as the proxies for strategic and nonstrategic
    
    mi_path = os.path.join(os.path.dirname(__file__), 'fig5data', 'monkey_mi_data.pkl')
    if os.path.exists(mi_path) and cached==True:
        with open(mi_path, 'rb') as f:
            mi_data = pickle.load(f)
            random_mi = mi_data['random']
            random_mi_err = mi_data['random_err']
            strategic_mi = mi_data['strategic']
            strategic_mi_err = mi_data['strategic_err']
            nonstrategic_mi = mi_data['nonstrategic']
            nonstrategic_mi_err = mi_data['nonstrategic_err']
            rnn_mi = mi_data['rnn']
            rnn_mi_err = mi_data['rnn_err']
            rlrnn_strategic_mi = mi_data['rlrnn_strategic']
            rlrnn_strategic_mi_err = mi_data['rlrnn_strategic_err']
            rlrnn_nonstrategic_mi = mi_data['rlrnn_nonstrategic']
            rlrnn_nonstrategic_mi_err = mi_data['rlrnn_nonstrategic_err']
    else:
        random_mi = compute_mutual_information(np.random.randint(0,2,1000000),np.random.randint(0,2,1000000),minlen=8,maxlen=8)
        random_mi_err = np.std(random_mi, axis=0)
        random_mi = np.mean(random_mi, axis=0)
        strategic_mi = []
        nonstrategic_mi = []
        for monkey in strategic_monkeys:
            monkey_data = mp_data[mp_data['animal'] == monkey]
            for sess in monkey_data['id'].unique():
                sess_df = monkey_data[monkey_data['id'] == sess]
                mi = compute_mutual_information(sess_df['monkey_choice'].values, sess_df['reward'].values, minlen=8,maxlen=8)
                strategic_mi.append(mi)
        for monkey in nonstrategic_monkeys:
            monkey_data = mp_data[mp_data['animal'] == monkey]
            for sess in monkey_data['id'].unique():
                sess_df = monkey_data[monkey_data['id'] == sess]
                mi = compute_mutual_information(sess_df['monkey_choice'].values, sess_df['reward'].values, minlen=8,maxlen=8)
                nonstrategic_mi.append(mi)
        strategic_mi_err = np.std(strategic_mi, axis=0)
        nonstrategic_mi_err = np.std(nonstrategic_mi, axis=0)
        strategic_mi = np.mean(strategic_mi, axis=0)[0]
        nonstrategic_mi = np.mean(nonstrategic_mi, axis=0)[0]
        with open(os.path.join(os.path.dirname(__file__), 'fig3data', 'mi_values_for_model.txt'), 'rb') as f:
            rnn_mi = f.readlines()
        rnn_mi = float(rnn_mi[0][:-2])# want the mi of length 8. this is in b'\n' format.
        # average mi of length 8 for the rlrnn models with strategic and nonstrategic parameters
        rlrnn_strategic_mi = group_mi['strategic']
        rlrnn_nonstrategic_mi = group_mi['nonstrategic']
        
        with open(mi_path, 'wb') as f:
            mi_data = pickle.dump({
                'random': random_mi,
                'random_err': random_mi_err,
                'strategic': strategic_mi,
                'strategic_err': strategic_mi_err,
                'nonstrategic': nonstrategic_mi,
                'nonstrategic_err': nonstrategic_mi_err,
                'rnn': rnn_mi,
                'rnn_err': 0,
                'rlrnn_strategic': rlrnn_strategic_mi,
                'rlrnn_strategic_err': 0,
                'rlrnn_nonstrategic': rlrnn_nonstrategic_mi,
                'rlrnn_nonstrategic_err': 0,
                
            }, f)
        
    # plot histogram for the MI of random, RNN, Strategic RLRNN, Nonstrategic RLRNN, and group of monkeys
    # MI_ax.hist(random, bins=20, alpha=0.5, label='Random')
    # MI_ax.hist(strategic_mi, bins=20, alpha=0.5, label='Strategic Monkeys')
    # MI_ax.hist(nonstrategic_mi, bins=20, alpha=0.5, label='Nonstrategic Monkeys')
    # MI_ax.hist(rnn_mi, bins=20, alpha=0.5, label='RNN')
    # MI_ax.hist(rlrnn_strategic_mi, bins=20, alpha=0.5, label='Strategic RLRNN')
    # MI_ax.hist(rlrnn_nonstrategic_mi, bins=20, alpha=0.5, label='Nonstrategic RLRNN')
    

    # heights = [random_mi, strategic_mi, nonstrategic_mi, rnn_mi, rlrnn_strategic_mi, rlrnn_nonstrategic_mi]
    # labels = ['Random', 'Strategic Monkeys', 'Nonstrategic Monkeys', 'RNN', 'Strategic RLRNN', 'Nonstrategic RLRNN']
    # for i in range(len(heights)):
    #     print(f"{labels[i]}: {heights[i]}")
        

    # MI_ax.bar(x = [0, 1, 2, 3, 4, 5], height = [random_mi, strategic_mi, nonstrategic_mi, rnn_mi, rlrnn_strategic_mi, rlrnn_nonstrategic_mi],
    #         alpha = 1)
    # MI_ax.set_xticks([0, 1, 2, 3, 4, 5])
    # MI_ax.set_xticklabels(['Random', 'Strategic Monkeys', 'Nonstrategic Monkeys', 'RNN', 'Strategic RLRNN', 'Nonstrategic RLRNN'])

    # MI_ax.legend()
    # MI_ax.set_title('Mutual Information Histogram')
    # MI_ax.set_xlabel('Mutual Information')

    # plot histogram for the MI of random, RNN, Strategic RLRNN, Nonstrategic RLRNN, and group of monkeys
    bar_labels = ['Strategic\nMonkeys', 'Non-strategic\nMonkeys', 'RNN', 'Strategic\nRLRNN', 'Non-strategic\nRLRNN']
    bar_heights = [strategic_mi, nonstrategic_mi, rnn_mi, rlrnn_strategic_mi, rlrnn_nonstrategic_mi]
    # Colors from other plots for consistency
    bar_colors = ['tab:cyan', 'tab:purple', 'tab:red', 'tab:olive', 'tab:olive']
    x_pos = np.arange(len(bar_labels))

    # Plot bars for different models/groups
    MI_ax.bar(x_pos, bar_heights, color=bar_colors, alpha=0.85)
    
    # Plot random MI as a dashed line for baseline
    MI_ax.axhline(y=random_mi, color='gray', linestyle='--', linewidth=2, label='Random Agent')

    # Style the plot
    MI_ax.set_ylabel('Mutual Information (bits)', fontsize=14)
    MI_ax.set_title('Mutual Information Comparison (seq_len=8)', fontsize=16, fontweight='bold')
    MI_ax.set_xticks(x_pos)
    MI_ax.set_xticklabels(bar_labels, fontsize=12)
    MI_ax.legend(frameon=False, fontsize=12, loc='upper left')
    MI_ax.grid(True, alpha=0.3, axis='y')
        
        
    # mi for strategic models, nonstrategic models, and RNN should have been precomputed, load from corresponding files
        
    #     rnn_mi = compute_mutual_information(rnn_data['monkey_choice'].values, rnn_data['reward'].values, minlen=8,maxlen=8)

        
        # mi for strategic models, nonstrategic models, and RNN should have been precomputed, load from corresponding files
        
    #     rnn_mi = compute_mutual_information(rnn_data['monkey_choice'].values, rnn_data['reward'].values, minlen=8,maxlen=8)

        
    #     mi_data = {
    #         'random': random,
    #         'strategic': [strategic_mi, strategic_mi_err],
    #         'nonstrategic': [nonstrategic_mi, nonstrategic_mi_err],
    #         'rnn': [rnn_mi, rnn_mi_err]
    #         'rlrnn_strategic': [rlrnn_strategic_mi, rlrnn_strategic_mi_err],
    #         'rlrnn_nonstrategic': [rlrnn_nonstrategic_mi, rlrnn_nonstrategic_mi_err]
    #     }
        

        
        
    #     strategic = compute_mutual_information(np.random.randint(0,2,10000),np.random.randint(0,2,10000),minlen=8,maxlen=8)
    
    # mi_dict = {
    #     'completely random agent': mi_data['random'],
    #     'strategic monkeys': mi_data['strategic'],
    #     'nonstrategic monkeys': mi_data['nonstrategic'],
    #     'RNN': mi_data['rnn'],
    #     'RLRNN (0.5,0)': mi_data['rlrnn_0.5_0']
    # }
    
    
def overlay_model_density(ax, model_type='rnn', 
                         ls_vals=None, ws_vals=None,
                         alpha=0.5, mask_percentile=95.0,
                         scatter_points=False, scatter_alpha=0.05,
                         scatter_size=28, n_contours=3,
                         plot_representative=True):
    """
    Overlay KDE density for model behavioral asymmetry on axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    model_type : str, default='rnn'
        Either 'rnn' or 'rlrnn' - determines colors and styling
    ls_vals : list or array
        Loss-switch asymmetry values for each model
    ws_vals : list or array  
        Win-stay asymmetry values for each model
    alpha : float, default=0.5
        Transparency of the density overlay
    mask_percentile : float, default=95.0
        Percentile threshold for masking low-density regions
    scatter_points : bool, default=False
        Whether to scatter individual model points
    plot_representative : bool, default=True
        Whether to plot the representative (medoid) model
        
    Returns
    -------
    n_models : int
        Number of models included in the density
    """
    from scipy.stats import gaussian_kde
    
    # Model-specific styling
    model_type_lower = model_type.lower()

    if model_type_lower == 'rnn':
        cmap = 'Reds'
        contour_color = 'darkred'
        marker_color = 'darkred'
        scatter_color = 'tab:red'
        label = 'Representative RNN'
    elif model_type_lower == 'rlrnn_strategic':
        cmap = 'YlOrBr'
        contour_color = '#8B4513'
        marker_color = '#C8A951'
        scatter_color = '#C8A951'
        label = 'Representative RLRNN (Strategic)'
    elif model_type_lower == 'rlrnn_nonstrategic':
        cmap = 'Purples'
        contour_color = 'purple'
        marker_color = 'purple'
        scatter_color = 'tab:purple'
        label = 'Representative RLRNN (Non-Strategic)'
    elif model_type_lower.startswith('rlrnn_param'):
        try:
            param_str = model_type.split(':', 1)[1]
            param_tuple = tuple(float(x) for x in param_str.split(','))
        except (IndexError, ValueError):
            param_tuple = None

        style = RLRNN_PARAM_STYLES.get(param_tuple, {
            'cmap': 'YlGnBu',
            'contour_color': '#004466',
            'marker_color': '#006699',
            'scatter_color': 'tab:cyan',
            'label': f'RLRNN {param_tuple}' if param_tuple else 'RLRNN (param)'
        })

        cmap = style['cmap']
        contour_color = style['contour_color']
        marker_color = style['marker_color']
        scatter_color = style['scatter_color']
        label = style.get('label', f'RLRNN {param_tuple}')
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")
    
    # Validate inputs
    if ls_vals is None or ws_vals is None or len(ls_vals) == 0:
        print(f"No data provided for {model_type} density")
        return 0
    
    ls_vals = np.array(ls_vals, dtype=float)
    ws_vals = np.array(ws_vals, dtype=float)
    
    # Filter out non-finite values that can cause huge canvas sizes
    finite_mask = np.isfinite(ls_vals) & np.isfinite(ws_vals)
    ls_vals = ls_vals[finite_mask]
    ws_vals = ws_vals[finite_mask]
    
    n_models = len(ls_vals)
    
    if n_models <= 5:
        print(f"{model_type}: not enough models ({n_models}) for density")
        return n_models
    
    # Compute density grid
    x_min, x_max = np.min(ls_vals), np.max(ls_vals)
    y_min, y_max = np.min(ws_vals), np.max(ws_vals)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    pad = 0.1
    x_min -= x_range * pad
    x_max += x_range * pad
    y_min -= y_range * pad
    y_max += y_range * pad
    
    # Create grid
    grid_size = 100
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Compute KDE
    values = np.vstack([ls_vals, ws_vals])
    kernel = gaussian_kde(values, bw_method='silverman')
    Z = np.reshape(kernel(positions), X.shape)
    
    # Threshold low-density regions
    positive = Z[Z > 0]
    if positive.size:
        min_threshold = np.percentile(positive, mask_percentile)
        Z[Z < min_threshold] = np.nan
    
    # Plot density heatmap
    extent = [x_min, x_max, y_min, y_max]
    ax.imshow(Z, origin='lower', extent=extent, aspect='auto', 
             cmap=cmap, alpha=alpha, interpolation='bilinear')
    
    # Plot contours
    levels = np.linspace(np.nanpercentile(Z, 5), np.nanmax(Z), n_contours)
    ax.contour(X, Y, Z, levels=levels, colors=[contour_color], 
              alpha=0.5, linewidths=0.5)
    
    # Optionally scatter individual points
    if scatter_points:
        ax.scatter(ls_vals, ws_vals, c=scatter_color, s=scatter_size, 
                  alpha=scatter_alpha, marker='o', linewidths=0)
    
    # Plot representative (medoid in cosine space)
    if plot_representative:
        anchors = np.stack([ls_vals, ws_vals], axis=1)
        norms = np.linalg.norm(anchors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized = anchors / norms
        centroid = np.mean(normalized, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm != 0:
            sims = normalized @ (centroid / centroid_norm)
            midx = int(np.argmax(sims))
            m_ls, m_ws = anchors[midx, 0], anchors[midx, 1]
            ax.scatter(m_ls, m_ws, color=marker_color, marker='X', s=140, 
                      edgecolor='k', linewidths=1.2, label=label, zorder=10)
    
    print(f"Added {model_type} density with {n_models} models")
    return n_models

def similarity_comparison_plot(ax, model_data, cached=False, block = False, performance = False, weighted = False, exclude_monkeys=None):
    """
    Plot comparison of model cosine similarity to monkey strategic behavior.
    
    Args:
        ax: Matplotlib axis for plotting
        model_data: Dictionary of RLRNN fits loaded from zoo
        cached: Whether to use cached block fits (default: False)
    """
    # Load data is now handled outside
    
    # Note: monkey fits loaded from fig1 are already in canonical order [WS, WSW, LST, LS, bias]
    # Model fits from RLRNN zoo need to be reordered via reorder_to_canonical
    # i want to make sure this reordering is correct. Can we plot model 226 using this reordering and see if it is correct?
    # load model 226
    # model_226_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/(0.5,0)_226_data.p'
    # with open(model_226_path, 'rb') as f:
    #     model_226_data = pickle.load(f)
    # # model_226_fit = model_226_data['action']
    # # model_226_fit = reorder_coefficients(model_226_fit)
    # model_226_fit, _labels = fit_glr(model_226_data, order=5, a_order=2, r_order=1, model=True, err=False, labels=True, average=True)
    # plt.plot(model_226_fit[0:5], label='win stay')
    # plt.plot(model_226_fit[5:10], label='lose switch')
    # plt.plot(model_226_fit[10:15], label='win switch')
    # plt.plot(model_226_fit[15:20], label='lose stay')
    # plt.legend()
    # plt.show()
    
    model_names = list(sorted(set(model_data.keys())))
    model_frechet_dict = {'strategic': {}, 'nonstrategic': {}}
    
    # Load monkey data and compute block fits
    mp2_data = pd.read_pickle(stitched_p)
    mp2_data = mp2_data[mp2_data['task'] == 'mp']
    
    # Categorize monkeys
    exclude_monkeys = set(exclude_monkeys or [])
    nonstrategic_monkeys = [m for m in ['C', 'H', 'F', 'K'] if m not in exclude_monkeys]
    strategic_monkeys = [m for m in ['E', 'D', 'I'] if m not in exclude_monkeys]
    
    # Cache file for block fits
    cache_file = os.path.join(os.path.dirname(__file__), 'monkey_block_fits_cache.pkl')
    
    
    fig1_data = pickle.load(open(os.path.join(os.path.dirname(__file__), 'fig1_data.pkl'), 'rb'))
    monkey_data_dict = {}
    for data in fig1_data:
        monkey = data.get('monkey')
        if not monkey:
            continue
        if monkey in exclude_monkeys:
            continue
        strategy_flag = 'strategic' if data.get('strategy') == 'strategic' else 'nonstrategic'
        entry = monkey_data_dict.setdefault(monkey, {
            'type': strategy_flag,
            'fits': []
        })
        entry['fits'].append(data['fit'][:-1])

    # Initialize similarity tracking
    monkey_similarity_tracker = {}
    for monkey in monkey_data_dict.keys():
        monkey_similarity_tracker[monkey] = {
            'idx': None,
            'weight': None,
            'fit': None,
            'similarity': -float('inf'),
            'monkey_fit': None
        }

    # Compute cosine similarity for each model
    

    # Load or compute block fits
    if not performance:
        if block:
            if cached and os.path.exists(cache_file):
                print(f"Loading cached block fits from {cache_file}")
                with open(cache_file, 'rb') as f:
                    monkey_data_dict = pickle.load(f)
            else:
                print("Computing block fits for each monkey...")
                
                # Build per-block fits for every monkey
                def compute_block_fits(df):
                    parts, lengths = partition_dataset(df, 5000)
                    # parts, lengths = [df], [len(df)]
                    block_fits = []
                    for pdata in parts:
                        # Fit GLR on this block and take the averaged coefficients
                        coeffs, _labels = fit_glr(pdata, order=5, a_order=2, r_order=1, model=False, err=False, labels=True, average=True)
                        fit_vec = coeffs[:-1] if coeffs is not None and len(coeffs) > 0 else None
                        if fit_vec is not None:
                            block_fits.append(fit_vec)
                    return block_fits, lengths
                
                monkey_data_dict = {}
                for monkey in nonstrategic_monkeys + strategic_monkeys:
                    mdat = mp2_data[mp2_data['animal'] == monkey]
                    if len(mdat) == 0:
                        continue
                    if monkey in ['F', 'E', 'C']:
                        mdat = cutoff_trials_by_session(mdat, 5000)
                    
                    block_fits, block_lengths = compute_block_fits(mdat)
                    if not block_fits:
                        continue
                    
                    mtype = 'strategic' if monkey in strategic_monkeys else 'nonstrategic'
                    monkey_data_dict[monkey] = {
                        'type': mtype,
                        'blocks': (block_fits),
                        'block_lengths': block_lengths[:len(block_fits)]
                    }
                    print(f"  Monkey {monkey}: {len(block_fits)} blocks, {int(sum(block_lengths[:len(block_fits)]))} trials")
                
                # Save cache
                if cached:
                    print(f"Saving block fits to cache: {cache_file}")
                    with open(cache_file, 'wb') as f:
                        pickle.dump(monkey_data_dict, f)
            
        # don't do block fits, just load the monkey data from fig1 and reorder the coefficients to match the model data
        
        # load monkey data from fig1
        
            
            for model in model_names:
                model_fits = []
                for run_data in model_data[model].values():
                    if isinstance(run_data, dict) and run_data.get('avg_reward', 0) > 0.45 and 'action' in run_data:
                        fit = np.array(run_data['action'])
                        model_fits.append(fit)

                # Accumulate weighted similarities for this parameter
                sum_w_s = 0.0
                sum_dw_s = 0.0
                sum_w_ns = 0.0
                sum_dw_ns = 0.0

                for i, model_fit in enumerate(model_fits):
                    # Compute cosine similarity for each monkey across their blocks
                    for monkey, m_data in monkey_data_dict.items():
                        block_fits = m_data['blocks']
                        block_weights = m_data['block_lengths']
                        
                        # Compute weighted average cosine similarity across blocks
                        dsum = 0.0
                        wsum = 0.0
                        norm_block_fits = []
                        
                        for bfit, bw in zip(block_fits, block_weights):
                            # Asymmetry-style differences in canonical order
                            # Win metric: WS - WSW; Loss metric: LS - LST (map from old order)
                            # model_win_diff = model_fit[0:5] - model_fit[5:10]
                            # model_loss_diff = model_fit[15:20] - model_fit[10:15]
                            
                            # bfit_win_diff = bfit[0:5] - bfit[5:10]
                            # bfit_loss_diff = bfit[15:20] - bfit[10:15]
                            
                            # Cosine similarity on difference vectors
                            # win_sim = np.dot(model_win_diff, bfit_win_diff) / (np.linalg.norm(model_win_diff) * np.linalg.norm(bfit_win_diff))
                            # loss_sim = np.dot(model_loss_diff, bfit_loss_diff) / (np.linalg.norm(model_loss_diff) * np.linalg.norm(bfit_loss_diff))
                            # similarity = (win_sim + loss_sim) / 2
                            
                            
                            
                            mws_sim = np.dot(model_fit[0:5], bfit[0:5]) / (np.linalg.norm(model_fit[0:5]) * np.linalg.norm(bfit[0:5]))
                            msw_sim = np.dot(model_fit[5:10], bfit[5:10]) / (np.linalg.norm(model_fit[5:10]) * np.linalg.norm(bfit[5:10]))
                            mls_sim = np.dot(model_fit[10:15], bfit[10:15]) / (np.linalg.norm(model_fit[10:15]) * np.linalg.norm(bfit[10:15]))
                            msl_sim = np.dot(model_fit[15:20], bfit[15:20]) / (np.linalg.norm(model_fit[15:20]) * np.linalg.norm(bfit[15:20]))
                            similarity = (mws_sim + msw_sim + mls_sim + msl_sim) / 4
                            
                            
                            # similarity = cosine_distance(model_fit, bfit)
                            
                            if np.isnan(similarity) or np.isinf(similarity):
                                continue
                            
                            norm_block_fits.append(bfit)
                            if weighted:
                                dsum += float(bw) * similarity
                                wsum += float(bw)
                            else:
                                dsum += similarity
                                wsum += 1

                        if wsum == 0:
                            continue
                        avg_similarity = dsum / wsum

                        # Track best model per monkey (higher cosine similarity is better)
                        if avg_similarity > monkey_similarity_tracker[monkey]['similarity']:
                            monkey_similarity_tracker[monkey]['idx'] = i
                            monkey_similarity_tracker[monkey]['weight'] = model
                            monkey_similarity_tracker[monkey]['fit'] = model_fit
                            monkey_similarity_tracker[monkey]['similarity'] = avg_similarity

                        # Accumulate by strategic group
                        if m_data['type'] == 'nonstrategic':
                            sum_w_ns += wsum
                            sum_dw_ns += dsum
                        else:
                            sum_w_s += wsum
                            sum_dw_s += dsum

                # Store weighted averages
                if sum_w_s > 0:
                    model_frechet_dict['strategic'][model] = sum_dw_s / sum_w_s
                if sum_w_ns > 0:
                    model_frechet_dict['nonstrategic'][model] = sum_dw_ns / sum_w_ns
        else:
        
            # Initialize similarity tracking
            monkey_similarity_tracker = {}
            for monkey in monkey_data_dict.keys():
                monkey_similarity_tracker[monkey] = {
                    'idx': None, 
                    'weight': None, 
                    'fit': None,
                    'similarity': -float('inf'),
                    'monkey_fit': None
                }

            # Compute cosine similarity for each model
            
            for model in model_names:
                model_fits = []
                for run_data in model_data[model].values():
                    if isinstance(run_data, dict) and run_data.get('avg_reward', 0) > 0.45 and 'action' in run_data:
                        fit = np.array(run_data['action'])
                        model_fits.append(fit)
                # Accumulate weighted similarities for this parameter
                sum_w_s = 0.0
                sum_dw_s = 0.0
                sum_w_ns = 0.0
                sum_dw_ns = 0.0

                for i, model_fit in enumerate(model_fits):
                    # Compute cosine similarity for each monkey
                    for monkey, m_data in monkey_data_dict.items():
                        for monkey_fit in m_data.get('fits', []):
                            dsum = 0.0
                            wsum = 0.0
                            norm_block_fits = []
                            # Cosine similarity per canonical segment
                            repeat_win_sim = np.dot(model_fit[0:5], monkey_fit[0:5]) / (np.linalg.norm(model_fit[0:5]) * np.linalg.norm(monkey_fit[0:5]))
                            change_win_sim = np.dot(model_fit[5:10], monkey_fit[5:10]) / (np.linalg.norm(model_fit[5:10]) * np.linalg.norm(monkey_fit[5:10]))
                            repeat_lose_sim = np.dot(model_fit[10:15], monkey_fit[10:15]) / (np.linalg.norm(model_fit[10:15]) * np.linalg.norm(monkey_fit[10:15]))
                            change_lose_sim = np.dot(model_fit[15:20], monkey_fit[15:20]) / (np.linalg.norm(model_fit[15:20]) * np.linalg.norm(monkey_fit[15:20]))

                            similarity = (repeat_win_sim + change_lose_sim + change_win_sim + repeat_lose_sim) / 4
                            norm_block_fits.append(monkey_fit)
                            dsum += similarity
                            wsum += 1

                            if wsum == 0:
                                continue
                            avg_similarity = dsum / wsum

                            norm_block_fits = np.array(norm_block_fits)
                            if norm_block_fits.ndim == 2 and len(norm_block_fits) > 0:
                                monkey_avg_fit = np.average(norm_block_fits, axis=0)
                            else:
                                monkey_avg_fit = norm_block_fits[0] if len(norm_block_fits) > 0 else None

                            if avg_similarity > monkey_similarity_tracker[monkey]['similarity']:
                                monkey_similarity_tracker[monkey]['idx'] = i
                                monkey_similarity_tracker[monkey]['weight'] = model
                                monkey_similarity_tracker[monkey]['fit'] = model_fit
                                monkey_similarity_tracker[monkey]['monkey_fit'] = monkey_avg_fit
                                monkey_similarity_tracker[monkey]['similarity'] = avg_similarity

                            if m_data['type'] == 'nonstrategic':
                                sum_w_ns += wsum
                                sum_dw_ns += dsum
                            else:
                                sum_w_s += wsum
                                sum_dw_s += dsum

                # Store weighted averages
                if sum_w_s > 0:
                    model_frechet_dict['strategic'][model] = sum_dw_s / sum_w_s
                if sum_w_ns > 0:
                    model_frechet_dict['nonstrategic'][model] = sum_dw_ns / sum_w_ns
    else:
        # instead of using similarity, use performance to find best parameter combination for each monkey
        # this data should be present inside the RLRNN zoo dictionary
        

        # Iterate through each parameter combination in the loaded model data
        for param_tuple, fits_dict in model_data.items():
            
            strategic_perfs = []
            nonstrategic_perfs = []
            
            # Iterate through each model run for the current parameter combination
            for run_id, run_data in fits_dict.items():
                if 'sequence_prediction' in run_data and run_data.get('avg_reward', 0) > 0.45:
                    seq_pred = run_data['sequence_prediction']
                    
                    # The key for params in seq_pred does not have parentheses and uses 'g' formatting for floats
                    if isinstance(param_tuple[0], int) and isinstance(param_tuple[1], int):
                        param_str = f"{param_tuple[0]},{param_tuple[1]}"
                    else:
                        param_str = f"{param_tuple[0]:g}, {param_tuple[1]:g}"

                    # Convert run_id to string for consistent key matching
                    run_id_str = str(run_id)

                    # Check for strategic performance
                    if 'strategic' in seq_pred and param_str in seq_pred.get('strategic', {}) and run_id_str in seq_pred['strategic'][param_str]:
                        s_data = seq_pred['strategic'][param_str][run_id_str]
                        if s_data:
                            # s_data is a list of boolean numpy arrays (one for each session)
                            total_correct = sum(np.sum(arr) for arr in s_data if isinstance(arr, np.ndarray) and arr.size > 0)
                            total_trials = sum(arr.size for arr in s_data if isinstance(arr, np.ndarray) and arr.size > 0)
                            if total_trials > 0:
                                strategic_perfs.append(total_correct / total_trials)
                    
                    # Check for non-strategic performance
                    if 'nonstrategic' in seq_pred and param_str in seq_pred.get('nonstrategic', {}) and run_id_str in seq_pred['nonstrategic'][param_str]:
                        ns_data = seq_pred['nonstrategic'][param_str][run_id_str]
                        if ns_data:
                            total_correct = sum(np.sum(arr) for arr in ns_data if isinstance(arr, np.ndarray) and arr.size > 0)
                            total_trials = sum(arr.size for arr in ns_data if isinstance(arr, np.ndarray) and arr.size > 0)
                            if total_trials > 0:
                                nonstrategic_perfs.append(total_correct / total_trials)
            
            # Average performance across all runs for this parameter combination
            if strategic_perfs:
                model_frechet_dict['strategic'][param_tuple] = np.mean(strategic_perfs)
            if nonstrategic_perfs:
                model_frechet_dict['nonstrategic'][param_tuple] = np.mean(nonstrategic_perfs)
    monkey_best_params = {}
    if isinstance(monkey_similarity_tracker, dict) and monkey_similarity_tracker:
        for monkey, info in monkey_similarity_tracker.items():
            monkey_best_params[monkey] = {
                'param': info.get('weight'),
                'similarity': info.get('similarity'),
                'fit_index': info.get('idx')
            }

    # Build grid for visualization
    grid_size = 11
    msarr = np.full((grid_size, grid_size), np.nan)
    mnsarr = np.full((grid_size, grid_size), np.nan)
    
    # Debug: Check parameter ranges
    strategic_params = list(model_frechet_dict['strategic'].keys())
    nonstrategic_params = list(model_frechet_dict['nonstrategic'].keys())
    all_params = strategic_params + nonstrategic_params
    if all_params:
        x_vals = [p[0] for p in all_params]
        y_vals = [p[1] for p in all_params]
        print(f"Parameter ranges - X: {min(x_vals):.3f} to {max(x_vals):.3f}, Y: {min(y_vals):.3f} to {max(y_vals):.3f}")
        print(f"Number of strategic models: {len(strategic_params)}, non-strategic: {len(nonstrategic_params)}")
        print(f"Unique X values: {sorted(set(x_vals))}")
        print(f"Unique Y values: {sorted(set(y_vals))}")
        print(f"X value count: {len(set(x_vals))}, Y value count: {len(set(y_vals))}")
    
    for model in model_frechet_dict['strategic']:
        x_idx = min(int(round(model[0] * (grid_size - 1))), grid_size - 1)
        y_idx = min(int(round(model[1] * (grid_size - 1))), grid_size - 1)
        print(f"Strategic model {model} -> grid[{y_idx}, {x_idx}]")
        msarr[y_idx, x_idx] = model_frechet_dict['strategic'][model]
    
    for model in model_frechet_dict['nonstrategic']:
        x_idx = min(int(round(model[0] * (grid_size - 1))), grid_size - 1)
        y_idx = min(int(round(model[1] * (grid_size - 1))), grid_size - 1)
        print(f"Non-strategic model {model} -> grid[{y_idx}, {x_idx}]")
        mnsarr[y_idx, x_idx] = model_frechet_dict['nonstrategic'][model]
    
    # Debug: Check final array shapes and non-NaN counts
    print(f"Strategic array shape: {msarr.shape}, non-NaN count: {np.sum(~np.isnan(msarr))}")
    print(f"Non-strategic array shape: {mnsarr.shape}, non-NaN count: {np.sum(~np.isnan(mnsarr))}")
    
    # Debug: Check which rows/columns have data
    strategic_rows_with_data = np.any(~np.isnan(msarr), axis=1)
    strategic_cols_with_data = np.any(~np.isnan(msarr), axis=0)
    nonstrategic_rows_with_data = np.any(~np.isnan(mnsarr), axis=1)
    nonstrategic_cols_with_data = np.any(~np.isnan(mnsarr), axis=0)
    
    print(f"Strategic - rows with data: {np.where(strategic_rows_with_data)[0]}")
    print(f"Strategic - cols with data: {np.where(strategic_cols_with_data)[0]}")
    print(f"Non-strategic - rows with data: {np.where(nonstrategic_rows_with_data)[0]}")
    print(f"Non-strategic - cols with data: {np.where(nonstrategic_cols_with_data)[0]}")
    
    # Find best parameters before normalization
    best_strategic_param = None
    if model_frechet_dict['strategic']:
        best_strategic_param = max(model_frechet_dict['strategic'], key=model_frechet_dict['strategic'].get)

    best_nonstrategic_param = None
    if model_frechet_dict['nonstrategic']:
        best_nonstrategic_param = max(model_frechet_dict['nonstrategic'], key=model_frechet_dict['nonstrategic'].get)
    
    best_params = {
        'strategic': best_strategic_param,
        'nonstrategic': best_nonstrategic_param
    }
    
    # Normalize to 0-1 range
    def _normalize(arr):
        if arr.size == 0 or np.all(np.isnan(arr)):
            return arr
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if amax - amin == 0:
            return np.where(np.isnan(arr), np.nan, 0.5)
        norm = np.full_like(arr, np.nan)
        mask = ~np.isnan(arr)
        norm[mask] = (arr[mask] - amin) / (amax - amin)
        return norm

    sim_s = _normalize(msarr)
    sim_ns = _normalize(mnsarr)
    
    # Debug: Check specific position (0.2, 0.1) which corresponds to grid[1, 2]
    print(f"\nDebug for position (0.2, 0.1) -> grid[1, 2]:")
    print(f"  Raw strategic value: {msarr[1, 2]}")
    print(f"  Normalized strategic value: {sim_s[1, 2]}")
    print(f"  Is NaN? {np.isnan(sim_s[1, 2])}")
    print(f"  Raw non-strategic value: {mnsarr[1, 2]}")
    print(f"  Normalized non-strategic value: {sim_ns[1, 2]}")
    
    # Setup grid
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    grid_spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
    square_size = grid_spacing
    circle_radius = square_size * 0.35
    
    # Find maximum values for highlighting
    strategic_max = np.nanmax(sim_s) if np.any(~np.isnan(sim_s)) else None
    nonstrategic_max = np.nanmax(sim_ns) if np.any(~np.isnan(sim_ns)) else None
    
    # Plot grid elements
    for i in range(grid_size):
        for j in range(grid_size):
            # Plot square (strategic model)
            if not np.isnan(sim_s[j, i]):
                is_max_strategic = (strategic_max is not None and 
                                  np.isclose(sim_s[j, i], strategic_max, rtol=1e-9))
                color = 'red' if is_max_strategic else plt.cm.cividis(sim_s[j, i])
                square = plt.Rectangle(
                    (x_grid[i] - square_size/2, y_grid[j] - square_size/2),
                    square_size, square_size,
                    facecolor=color, edgecolor='k', alpha=0.8, linewidth=0.5
                )
                ax.add_patch(square)
            else:
                # No data for strategic - show color for 0 value
                square = plt.Rectangle(
                    (x_grid[i] - square_size/2, y_grid[j] - square_size/2),
                    square_size, square_size,
                    facecolor=plt.cm.cividis(0.0), edgecolor='k', alpha=0.8, linewidth=0.5
                )
                ax.add_patch(square)
                
            # Plot circle (non-strategic model)
            if not np.isnan(sim_ns[j, i]):
                is_max_nonstrategic = (nonstrategic_max is not None and 
                                      np.isclose(sim_ns[j, i], nonstrategic_max, rtol=1e-9))
                color = 'red' if is_max_nonstrategic else plt.cm.cividis(sim_ns[j, i])
                circle = plt.Circle(
                    (x_grid[i], y_grid[j]), circle_radius,
                    facecolor=color, edgecolor='k', alpha=0.9, linewidth=0.5
                )
                ax.add_patch(circle)
            else:
                # No data for non-strategic - show color for 0 value
                circle = plt.Circle(
                    (x_grid[i], y_grid[j]), circle_radius,
                    facecolor=plt.cm.cividis(0.0), edgecolor='k', alpha=0.9, linewidth=0.5
                )
                ax.add_patch(circle)
    
    # Create divider for legend
    divider = make_axes_locatable(ax)
    
    # Create colorbar
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    colorbar_axes = inset_axes(ax, width="3%", height="96%", loc='center left',
                               bbox_to_anchor=(1.04, 0.02, 1, 1),
                               bbox_transform=ax.transAxes, borderpad=0)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=colorbar_axes)
    cbar.set_label('Cosine Similarity', fontsize=12)
    colorbar_axes.yaxis.set_label_position('right')
    cbar.ax.tick_params(labelsize=12)

    # Add legend
    legend_ax = divider.append_axes("bottom", size="12%", pad=0.30)
    legend_ax.axis('off')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=15, label='Strategic (square)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=15, label='Non-strategic (circle)')
    ]
    legend_ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.10))
    
    # Add red outline markers at max-similarity coordinates
    if strategic_max is not None:
        s_idx = np.nanargmax(sim_s)
        s_j, s_i = np.unravel_index(s_idx, sim_s.shape)
        ax.add_patch(plt.Rectangle((x_grid[s_i]-square_size/2, y_grid[s_j]-square_size/2),
                                   square_size, square_size,
                                   facecolor='none', edgecolor='red', linewidth=2.0))
    if nonstrategic_max is not None:
        ns_idx = np.nanargmax(sim_ns)
        ns_j, ns_i = np.unravel_index(ns_idx, sim_ns.shape)
        ax.add_patch(plt.Circle((x_grid[ns_i], y_grid[ns_j]),
                                circle_radius*1.2, facecolor='none', edgecolor='red', linewidth=2.0))
    
    # Set axis properties
    ax.set_xlim(-square_size, 1 + square_size)
    ax.set_ylim(-square_size, 1 + square_size)
    ax.set_xlabel('Decision Layer Loss Parameter', fontsize=12)
    ax.set_ylabel('Decision Layer Win Parameter', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    
    # Add model type labels
    ax.text(1.1, 1.15, 'RNN', ha='left', va='top', fontsize=16)
    ax.text(1.05, -0.15, 'Switching', rotation=0, ha='right', va='top', fontsize=16)
    ax.text(-0.1, -0.15, 'RL', ha='left', va='bottom', fontsize=16)
    
    
    return ax, best_params, {k: v['param'] for k, v in monkey_best_params.items()}

def add_performance_histogram(ax, bootstrap_iters=1000, random_state=0, cached=False, 
                             strategic_candidate_params=None, nonstrategic_candidate_params=None,
                             rlrnn_zoo_dict=None, rlrnn_best_param_fits=None):
    """
    Plot performance bars for Hybrid, RNN, RL, and LR models.
    
    LR performance represents out-of-sample prediction accuracy, computed via
    cross-validation. This serves as a baseline for how well a logistic
    regression model can generalize to unseen data.
    
    Args:
        strategic_candidate_params: List of (gamma_loss, gamma_win) tuples for strategic models
        nonstrategic_candidate_params: List of (gamma_loss, gamma_win) tuples for nonstrategic models
        rlrnn_zoo_dict: Dictionary of RLRNN models by parameter tuple
        rlrnn_best_param_fits: Dictionary of best-fit RLRNN models by parameter tuple
    """
    # Data paths
    rnn_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/rnn_sequence_prediction_summary.pkl'
    monkey_path = '/Users/fmb35/Desktop/BG-PFC-RNN/stitched_monkey_data_safely_cleaned.pkl'
    lr_cache_path = os.path.join(os.path.dirname(__file__), 'fig5_lr_performance_cache.pkl')
    bootstrap_cache_path = os.path.join(os.path.dirname(__file__), 'fig5_bootstrap_cache.pkl')
    rl_perf_path = os.path.join(os.path.dirname(__file__), 'fig5_rl_performance_cache.pkl')
    
    strategic_monkeys = ['E', 'D', 'I']
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']
    
    # Default parameter sets (matching MI comparison)
    if strategic_candidate_params is None:
        strategic_candidate_params = [(1.0, 0.0), (1.0, 0.1)]
    if nonstrategic_candidate_params is None:
        nonstrategic_candidate_params = [(1.0, 0.0), (0.9, 0.0)]
    
    # --- Helper: Bootstrap SEM ---
    def boot_sem(vals, n=1000, seed=0):
        if not vals: return 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(vals)
        means = [np.mean(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
        return np.std(means, ddof=1)
    
    def boot_sem_weighted(accs, lengths, n=1000, seed=0):
        if not accs: return 0.0
        rng = np.random.default_rng(seed)
        accs = np.array(accs)
        lengths = np.array(lengths)
        
        boot_means = []
        for _ in range(n):
            indices = rng.integers(0, len(accs), len(accs))
            sample_accs = accs[indices]
            sample_lengths = lengths[indices]
            if np.sum(sample_lengths) > 0:
                weighted_mean = np.sum(sample_accs * sample_lengths) / np.sum(sample_lengths)
            else:
                weighted_mean = np.mean(sample_accs) if len(sample_accs) > 0 else 0
            boot_means.append(weighted_mean)
            
        return np.std(boot_means, ddof=1)

    # --- Load Hybrid (RLRNN) performance from multiple parameter tuples ---
    def get_hybrid_perf_from_params(param_list, monkeys, model_data_dict, best_fits_dict):
        """Extract performance across multiple parameter tuples."""
        perfs = []
        
        for param_tuple in param_list:
            # Try best fits first, then zoo
            fits_dict = best_fits_dict.get(param_tuple, {})
            if not fits_dict:
                fits_dict = model_data_dict.get(param_tuple, {})
            
            if not fits_dict:
                print(f"Warning: No RLRNN data found for parameter {param_tuple}")
                continue
            
            # Create proper param_key format
            if isinstance(param_tuple[0], int) and isinstance(param_tuple[1], int):
                param_key = f"{param_tuple[0]},{param_tuple[1]}"
            else:
                param_key = f"{param_tuple[0]:g}, {param_tuple[1]:g}"
            
            for run_idx, fd in fits_dict.items():
                if not isinstance(fd, dict) or fd.get('avg_reward', 0) <= 0.45:
                    continue
                if 'sequence_prediction' not in fd:
                    continue
                    
                sp = fd['sequence_prediction']
                for m in monkeys:
                    if m in sp and sp[m] and param_key in sp[m]:
                        run_id_str = str(run_idx)
                        if run_id_str in sp[m][param_key]:
                            arrs = sp[m][param_key][run_id_str]
                            if arrs:
                                total = sum(np.sum(a) for a in arrs if isinstance(a, np.ndarray) and a.size > 0)
                                count = sum(a.size for a in arrs if isinstance(a, np.ndarray) and a.size > 0)
                                if count > 0:
                                    perfs.append(total / count)
        
        return perfs
    
    # Load RLRNN data if not provided
    if rlrnn_zoo_dict is None or rlrnn_best_param_fits is None:
        print("Loading RLRNN zoo and best fits...")
        RLRNN_ZOO_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/RLRNN_zoo_dict.pkl'
        RLRNN_BEST_FITS_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/strategic_and_nonstrategic_best_param_fits.pkl'
        
        if rlrnn_zoo_dict is None:
            rlrnn_zoo_dict = load_rlrnn_zoo(RLRNN_ZOO_PATH)
        if rlrnn_best_param_fits is None:
            rlrnn_best_param_fits = load_rlrnn_zoo(RLRNN_BEST_FITS_PATH)
    
    strat_hybrid = get_hybrid_perf_from_params(
        strategic_candidate_params, strategic_monkeys, rlrnn_zoo_dict, rlrnn_best_param_fits
    )
    nonstrat_hybrid = get_hybrid_perf_from_params(
        nonstrategic_candidate_params, nonstrategic_monkeys, rlrnn_zoo_dict, rlrnn_best_param_fits
    )
    
    print(f"Hybrid performance samples - Strategic: {len(strat_hybrid)} (params: {strategic_candidate_params})")
    print(f"Hybrid performance samples - Non-strategic: {len(nonstrat_hybrid)} (params: {nonstrategic_candidate_params})")
    
    if cached and os.path.exists(bootstrap_cache_path) and os.path.exists(lr_cache_path) and os.path.exists(rl_perf_path):
        # load all cached results
        print(f"Loading cached bootstrap results from {bootstrap_cache_path}")
        with open(bootstrap_cache_path, 'rb') as f:
            bootstrap_cache = pickle.load(f)
        
        # Extract cached values
        strat_hybrid_sem = bootstrap_cache.get('strat_hybrid_sem', 0)
        nonstrat_hybrid_sem = bootstrap_cache.get('nonstrat_hybrid_sem', 0)
        strat_rnn_sem = bootstrap_cache.get('strat_rnn_sem', 0)
        nonstrat_rnn_sem = bootstrap_cache.get('nonstrat_rnn_sem', 0)
        strat_lr_sem = bootstrap_cache.get('strat_lr_sem', 0)
        nonstrat_lr_sem = bootstrap_cache.get('nonstrat_lr_sem', 0)
        strat_lr_insample_sem = bootstrap_cache.get('strat_lr_insample_sem', 0)
        nonstrat_lr_insample_sem = bootstrap_cache.get('nonstrat_lr_insample_sem', 0)
        
        print(f"Loading cached LR performance from {lr_cache_path}")
        with open(lr_cache_path, 'rb') as f:
            lr_results = pickle.load(f)
        strat_lr_accuracies = lr_results['strategic']
        nonstrat_lr_accuracies = lr_results['nonstrategic']
        strat_lr_lengths = lr_results.get('strategic_lengths', [])
        nonstrat_lr_lengths = lr_results.get('nonstrategic_lengths', [])
        strat_lr_insample_accs = lr_results.get('strategic_insample_accs', [])
        nonstrat_lr_insample_accs = lr_results.get('nonstrategic_insample_accs', [])
        strat_lr_insample_lengths = lr_results.get('strategic_insample_lengths', [])
        nonstrat_lr_insample_lengths = lr_results.get('nonstrategic_insample_lengths', [])
        
        with open(rl_perf_path, 'rb') as f:
            rl_perf = pickle.load(f)
        strat_rl_mean = rl_perf.get('strat_rl_mean', 0)
        nonstrat_rl_mean = rl_perf.get('nonstrat_rl_mean', 0)
        strat_rl_sem = rl_perf.get('strat_rl_sem', 0)
        nonstrat_rl_sem = rl_perf.get('nonstrat_rl_sem', 0)
        
        with open(rnn_path, 'rb') as f:
            rnn = pickle.load(f)
        strat_rnn_mean = rnn.get('strategic_monkeys', {}).get('mean_accuracy', np.nan)
        nonstrat_rnn_mean = rnn.get('non_strategic_monkeys', {}).get('mean_accuracy', np.nan)
        rnn_strat_vals = rnn.get('aggregate', {}).get('strategic', {}).get('model_means', [])
        rnn_nonstrat_vals = rnn.get('aggregate', {}).get('nonstrategic', {}).get('model_means', [])
        
        
    else:

        # --- Load RNN performance ---
        with open(rnn_path, 'rb') as f:
            rnn = pickle.load(f)
        strat_rnn_mean = rnn.get('strategic_monkeys', {}).get('mean_accuracy', np.nan)
        nonstrat_rnn_mean = rnn.get('non_strategic_monkeys', {}).get('mean_accuracy', np.nan)
        rnn_strat_vals = rnn.get('aggregate', {}).get('strategic', {}).get('model_means', [])
        rnn_nonstrat_vals = rnn.get('aggregate', {}).get('nonstrategic', {}).get('model_means', [])
        
        # --- Compute RL performance (simplified CV) ---
        from analysis_scripts.LLH_behavior_RL import cross_validated_performance_sessions
        
        df = pd.read_pickle(monkey_path)
        df = df[df['task'] == 'mp']
        
        # --- In-sample LR performance helper ---
        def get_lr_perf_in_sample(monkeys, df):
            from analysis_scripts.logistic_regression import parse_monkey_behavior_strategic, create_order_data
            def _expit(x):
                return 1 / (1 + np.exp(-x))

            group_df = df[df['animal'].isin(monkeys)]
            if group_df.empty:
                return 0.0

            coeffs, _labels = fit_glr(group_df, order=5, a_order=2, r_order=1, model=False, err=False, labels=True, average=True)
            if coeffs is None:
                return 0.0

            # Build X and y with the same GLR design for consistency
            sessions = group_df['id'].unique()
            X_list, y_list = [], []
            for s in sessions:
                sess = group_df[group_df['id'] == s].sort_values(by=['id','trial'])
                X_i, y_i, _ = general_logistic_regressors(sess['monkey_choice'].values, sess['reward'].values, regression_order=5, a_order=2, r_order=1)
                X_list.append(X_i)
                y_list.append(y_i.squeeze())
            if not X_list:
                return 0.0
            X_data = np.vstack(X_list)
            y_data = np.hstack(y_list)

            if X_data is None or y_data is None or X_data.shape[0] != y_data.shape[0] or len(coeffs) != X_data.shape[1] + 1:
                return 0.0

            logits = X_data @ coeffs[:-1] + coeffs[-1]
            preds = (_expit(logits) > 0.5).astype(int)
            accuracy = np.mean(preds == y_data)
            return accuracy

        def get_rl_perf(monkeys, seed):
            result = cross_validated_performance_by_monkey_df(
                df, monkeys, model='simple', n_folds=10, random_state=seed,
                punitive=False, decay=False, const_beta=False, const_gamma=True,
                disable_abs=False, n_bootstrap=bootstrap_iters, greedy=True
            )
            return result.get('mean_accuracy', 0.0), result.get('bootstrap_sem', 0.0)
        
        strat_rl_mean, strat_rl_sem = get_rl_perf(strategic_monkeys, random_state)
        nonstrat_rl_mean, nonstrat_rl_sem = get_rl_perf(nonstrategic_monkeys, random_state + 4)
        
        # save strat_rl_mean, nonstrat_rl_mean, strat_rl_sem, nonstrat_rl_sem to rl_perf_path
        with open(rl_perf_path, 'wb') as f:
            rl_perf = pickle.dump({
                'strat_rl_mean': strat_rl_mean,
                'nonstrat_rl_mean': nonstrat_rl_mean,
                'strat_rl_sem': strat_rl_sem,
                'nonstrat_rl_sem': nonstrat_rl_sem,
            }, f)
        
        # --- Compute LR performance (cross-validated) ---
        if cached and os.path.exists(lr_cache_path):
            print(f"Loading cached LR performance from {lr_cache_path}")
            with open(lr_cache_path, 'rb') as f:
                lr_results = pickle.load(f)
            strat_lr_accuracies = lr_results['strategic']
            nonstrat_lr_accuracies = lr_results['nonstrategic']
            strat_lr_lengths = lr_results.get('strategic_lengths', [])
            nonstrat_lr_lengths = lr_results.get('nonstrategic_lengths', [])
            strat_lr_insample_accs = lr_results.get('strategic_insample_accs', [])
            nonstrat_lr_insample_accs = lr_results.get('nonstrategic_insample_accs', [])
            strat_lr_insample_lengths = lr_results.get('strategic_insample_lengths', [])
            nonstrat_lr_insample_lengths = lr_results.get('nonstrategic_insample_lengths', [])
        else:
            print("Computing LR performance per session (per-monkey cross-validated)...")
            from analysis_scripts.logistic_regression import parse_monkey_behavior_strategic, create_order_data
            from sklearn.model_selection import KFold

            def _expit(x):
                return 1 / (1 + np.exp(-x))

            def get_lr_perf_cv_per_monkey(monkeys, n_folds=10, seed=0):
                all_session_accuracies = []
                all_session_lengths = []
                all_insample_accuracies = []
                all_insample_lengths = []
                rng = np.random.default_rng(seed)

                for mk in monkeys:
                    mk_df = df[df['animal'] == mk]
                    sessions = mk_df['id'].unique()
                    if len(sessions) == 0:
                        continue
                    
                    n_splits = min(max(2, n_folds), len(sessions))
                    folds = np.array_split(rng.permutation(sessions), n_splits)
                    
                    for fold in folds:
                        test_ids = set(fold.tolist())
                        train_ids = [sid for sid in sessions if sid not in test_ids]

                        if not train_ids:
                            continue

                        train_df = mk_df[mk_df['id'].isin(train_ids)]
                        
                        coeffs, _labels = fit_glr(train_df, order=5, a_order=2, r_order=1, model=False, err=False, labels=True, average=True)
                        
                        # --- In-sample performance for this fold ---
                        if not train_df.empty:
                            X_train_list, y_train_list = [], []
                            for train_session_id in train_ids:
                                train_sess_df = train_df[train_df['id'] == train_session_id]
                                if train_sess_df.empty: continue
                                X_i_train, y_i_train, _ = general_logistic_regressors(train_sess_df['monkey_choice'].values, train_sess_df['reward'].values, regression_order=5, a_order=2, r_order=1)
                                X_train_list.append(X_i_train)
                                y_train_list.append(y_i_train.squeeze())
                            
                            if X_train_list:
                                X_train = np.vstack(X_train_list)
                                y_train = np.hstack(y_train_list)
                                if X_train.shape[0] > 0 and len(coeffs) == X_train.shape[1] + 1:
                                    logits_train = X_train @ coeffs[:-1] + coeffs[-1]
                                    preds_train = (_expit(logits_train) > 0.5).astype(int)
                                    accuracy_train = np.mean(preds_train == y_train)
                                    all_insample_accuracies.append(accuracy_train)
                                    all_insample_lengths.append(len(y_train))

                        for test_session_id in test_ids:
                            test_df = mk_df[mk_df['id'] == test_session_id]
                            if test_df.empty:
                                continue
                            
                            # Build X_test and y_test using GLR design for consistency
                            X_list, y_list = [], []
                            X_i, y_i, _ = general_logistic_regressors(test_df['monkey_choice'].values, test_df['reward'].values, regression_order=5, a_order=2, r_order=1)
                            X_list.append(X_i)
                            y_list.append(y_i.squeeze())
                            if not X_list:
                                continue
                            X_test = np.vstack(X_list)
                            y_test = np.hstack(y_list)

                            if X_test is None or y_test is None or X_test.shape[0] != y_test.shape[0] or len(coeffs) != X_test.shape[1] + 1:
                                continue

                            logits = X_test @ coeffs[:-1] + coeffs[-1]
                            preds = (_expit(logits) > 0.5).astype(int)
                            accuracy = np.mean(preds == y_test)
                            all_session_accuracies.append(accuracy)
                            all_session_lengths.append(len(y_test))

                return all_session_accuracies, all_session_lengths, all_insample_accuracies, all_insample_lengths

            strat_lr_accuracies, strat_lr_lengths, strat_lr_insample_accs, strat_lr_insample_lengths = get_lr_perf_cv_per_monkey(strategic_monkeys, seed=random_state)
            nonstrat_lr_accuracies, nonstrat_lr_lengths, nonstrat_lr_insample_accs, nonstrat_lr_insample_lengths = get_lr_perf_cv_per_monkey(nonstrategic_monkeys, seed=random_state)

            # Cache results
            lr_results = {
                'strategic': strat_lr_accuracies,
                'nonstrategic': nonstrat_lr_accuracies,
                'strategic_lengths': strat_lr_lengths,
                'nonstrategic_lengths': nonstrat_lr_lengths,
                'strategic_insample_accs': strat_lr_insample_accs,
                'nonstrategic_insample_accs': nonstrat_lr_insample_accs,
                'strategic_insample_lengths': strat_lr_insample_lengths,
                'nonstrategic_insample_lengths': nonstrat_lr_insample_lengths,
            }
            with open(lr_cache_path, 'wb') as f:
                pickle.dump(lr_results, f)
            print(f"Saved LR performance to cache: {lr_cache_path}")
        
        print(f"LR sessions (cross-validated) - Strategic: {len(strat_lr_accuracies)}, Non-strategic: {len(nonstrat_lr_accuracies)}")
        
        # --- Compute or load bootstrap SEMs ---
        if cached and os.path.exists(bootstrap_cache_path):
            print(f"Loading cached bootstrap results from {bootstrap_cache_path}")
            with open(bootstrap_cache_path, 'rb') as f:
                bootstrap_cache = pickle.load(f)
            
            # Extract cached values
            strat_hybrid_sem = bootstrap_cache.get('strat_hybrid_sem', 0)
            nonstrat_hybrid_sem = bootstrap_cache.get('nonstrat_hybrid_sem', 0)
            strat_rnn_sem = bootstrap_cache.get('strat_rnn_sem', 0)
            nonstrat_rnn_sem = bootstrap_cache.get('nonstrat_rnn_sem', 0)
            strat_lr_sem = bootstrap_cache.get('strat_lr_sem', 0)
            nonstrat_lr_sem = bootstrap_cache.get('nonstrat_lr_sem', 0)
            strat_lr_insample_sem = bootstrap_cache.get('strat_lr_insample_sem', 0)
            nonstrat_lr_insample_sem = bootstrap_cache.get('nonstrat_lr_insample_sem', 0)
        else:
            print(f"Computing bootstrap SEMs (n={bootstrap_iters})...")
            # Compute all bootstrap SEMs
            strat_hybrid_sem = boot_sem(strat_hybrid, bootstrap_iters, random_state)
            nonstrat_hybrid_sem = boot_sem(nonstrat_hybrid, bootstrap_iters, random_state+1)
            strat_rnn_sem = boot_sem(rnn_strat_vals, bootstrap_iters, random_state+2)
            nonstrat_rnn_sem = boot_sem(rnn_nonstrat_vals, bootstrap_iters, random_state+3)
            strat_lr_sem = boot_sem_weighted(strat_lr_accuracies, strat_lr_lengths, bootstrap_iters, random_state+5) if strat_lr_accuracies else 0
            nonstrat_lr_sem = boot_sem_weighted(nonstrat_lr_accuracies, nonstrat_lr_lengths, bootstrap_iters, random_state+6) if nonstrat_lr_accuracies else 0
            strat_lr_insample_sem = boot_sem_weighted(strat_lr_insample_accs, strat_lr_insample_lengths, bootstrap_iters, random_state+7) if strat_lr_insample_accs else 0
            nonstrat_lr_insample_sem = boot_sem_weighted(nonstrat_lr_insample_accs, nonstrat_lr_insample_lengths, bootstrap_iters, random_state+8) if nonstrat_lr_insample_accs else 0
            
            # Cache bootstrap results
            bootstrap_cache = {
                'strat_hybrid_sem': strat_hybrid_sem,
                'nonstrat_hybrid_sem': nonstrat_hybrid_sem,
                'strat_rnn_sem': strat_rnn_sem,
                'nonstrat_rnn_sem': nonstrat_rnn_sem,
                'strat_lr_sem': strat_lr_sem,
                'nonstrat_lr_sem': nonstrat_lr_sem,
                'strat_lr_insample_sem': strat_lr_insample_sem,
                'nonstrat_lr_insample_sem': nonstrat_lr_insample_sem,
                'bootstrap_iters': bootstrap_iters,
                'random_state': random_state
            }
            with open(bootstrap_cache_path, 'wb') as f:
                pickle.dump(bootstrap_cache, f)
            print(f"Saved bootstrap results to cache: {bootstrap_cache_path}")
        
    # --- Plot ---
    x = np.arange(2)  # Strategic, Non-strategic
    width = 0.2
    colors = ['tab:olive', 'tab:red', 'tab:blue', 'tab:green']
    
    def weighted_avg(accs, lens):
        if not accs:
            return 0.5
        accs, lens = np.array(accs), np.array(lens)
        return np.sum(accs * lens) / np.sum(lens)

    means = [
        [np.mean(strat_hybrid), strat_rnn_mean, strat_rl_mean, weighted_avg(strat_lr_accuracies, strat_lr_lengths)],
        [np.mean(nonstrat_hybrid), nonstrat_rnn_mean, nonstrat_rl_mean, weighted_avg(nonstrat_lr_accuracies, nonstrat_lr_lengths)]
    ]
    sems = [
        [strat_hybrid_sem, strat_rnn_sem, strat_rl_sem, strat_lr_sem],
        [nonstrat_hybrid_sem, nonstrat_rnn_sem, nonstrat_rl_sem, nonstrat_lr_sem]
    ]
    
    for i, (label, color) in enumerate(zip(['Hybrid', 'RNN', 'RL', 'LR (out-of-sample)'], colors)):
        ax.bar(x + (i-1.5)*width, [m[i] for m in means], width, 
               yerr=[s[i] for s in sems], label=label, color=color, 
               alpha=0.85, capsize=4)
    
    # --- Plot in-sample LR performance ---
    strat_lr_in_sample = weighted_avg(strat_lr_insample_accs, strat_lr_insample_lengths)
    nonstrat_lr_in_sample = weighted_avg(nonstrat_lr_insample_accs, nonstrat_lr_insample_lengths)
    
    line_extent = 2 * width  # Span across the four bars
    ax.plot([x[0] - line_extent, x[0] + line_extent], [strat_lr_in_sample, strat_lr_in_sample],
            color=colors[3], linestyle='--', label='LR (in-sample)')
    ax.plot([x[1] - line_extent, x[1] + line_extent], [nonstrat_lr_in_sample, nonstrat_lr_in_sample],
            color=colors[3], linestyle='--')

    # Style
    ax.set_ylabel('Sequence Prediction Accuracy', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Strategic', 'Non-strategic'], fontsize=14)
    ax.legend(frameon=False, fontsize=12)
    ax.set_ylim(0.45, 0.68)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Print summary
    print(f"Performance: Strategic - Hybrid: {means[0][0]:.3f}, RNN: {means[0][1]:.3f}, RL: {means[0][2]:.3f}, LR: {means[0][3]:.3f}")
    print(f"Performance: Non-strategic - Hybrid: {means[1][0]:.3f}, RNN: {means[1][1]:.3f}, RL: {means[1][2]:.3f}, LR: {means[1][3]:.3f}")


def mp1_supplemental_figure(mp1_model_fits_path, mp1_monkey_data_path):
    """
    Creates a supplemental figure showing the parameter sweep for MP1.
    Plots where strategic and non-strategic optimal parameters are based on 
    cosine similarity to monkey GLM coefficients.
    """
    with open(mp1_model_fits_path, 'rb') as f:
        loaded_data = pickle.load(f)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            mp1_model_fits = loaded_data[1]
        else:
            mp1_model_fits = loaded_data

    # Load and process monkey data to get average coefficients
    C_1 = load_behavior(mp1_monkey_data_path, algorithm=1, monkey=13)
    F_1 = load_behavior(mp1_monkey_data_path, algorithm=1, monkey=112)
    E_1 = load_behavior(mp1_monkey_data_path, algorithm=1, monkey=18)
    all_monkey_data = pd.concat([C_1, F_1, E_1])
    avg_monkey_coeffs, _ = fit_glr(all_monkey_data, order=5, a_order=2, r_order=1, err=False, labels=True, average=True)

    # Correlate integer keys with (alpha, beta) pairs and calculate similarities
    param_map = {}
    similarities = {}
    for key, data in mp1_model_fits.items():
        if isinstance(data, dict) and 'policy_weights' in data and len(data['policy_weights']) == 2:
            alpha, beta = data['policy_weights']
            param_tuple = (alpha, beta)
            param_map[key] = param_tuple
            
            model_coeffs = data.get('action', data.get('fit'))
            if model_coeffs is not None:
                min_len = min(len(avg_monkey_coeffs), len(model_coeffs))
                cos_sim = np.dot(avg_monkey_coeffs[:min_len], model_coeffs[:min_len]) / \
                          (np.linalg.norm(avg_monkey_coeffs[:min_len]) * np.linalg.norm(model_coeffs[:min_len]))
                similarities[param_tuple] = cos_sim

    if not similarities:
        raise ValueError("Could not compute cosine similarities. Check data structure and 'policy_weights' key.")

    best_param_tuple = max(similarities, key=similarities.get)
    
    # Find the original integer key for the best parameter tuple
    best_param_key = [k for k, v in param_map.items() if v == best_param_tuple][0]

    # Create the figure
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.3)
    ax_model = fig.add_subplot(gs[0])
    ax_sweep = fig.add_subplot(gs[1])
    ax_monkey = fig.add_subplot(gs[2])

    # Plot sweep heatmap
    unique_alphas = sorted(list(set(p[0] for p in param_map.values())))
    unique_betas = sorted(list(set(p[1] for p in param_map.values())))

    if len(unique_alphas) > 1 and len(unique_betas) > 1:
        similarity_grid = np.full((len(unique_betas), len(unique_alphas)), np.nan)
        for i, beta in enumerate(unique_betas):
            for j, alpha in enumerate(unique_alphas):
                param_tuple = (alpha, beta)
                if param_tuple in similarities:
                    similarity_grid[i, j] = similarities[param_tuple]

        im = ax_sweep.imshow(similarity_grid, cmap='viridis', aspect='auto', origin='lower',
                             extent=[min(unique_alphas), max(unique_alphas), min(unique_betas), max(unique_betas)])
        fig.colorbar(im, ax=ax_sweep, label='Cosine Similarity')
        ax_sweep.set_xlabel('Alpha')
        ax_sweep.set_ylabel('Beta')
        ax_sweep.set_title('Parameter Sweep: Cosine Similarity to Monkeys')
        
        # Mark the best parameter
        ax_sweep.plot(best_param_tuple[0], best_param_tuple[1], 'r*', markersize=15, label=f'Best Fit ({best_param_tuple[0]:.2f}, {best_param_tuple[1]:.2f})')
        ax_sweep.legend()
    else:
        ax_sweep.text(0.5, 0.5, 'Not enough data for a 2D sweep', ha='center', va='center')


    # Plot coefficients for the best model
    best_model_data = mp1_model_fits[best_param_key]
    best_coeffs = best_model_data.get('action', best_model_data.get('fit'))
    
    plot_model_coefficients(ax_model, best_coeffs, 
                          title=f'Best Fit RLRNN\nparams={best_param_tuple}',
                          order=5)
    
    # Plot the average monkey coefficients for comparison
    plot_model_coefficients(ax_monkey, avg_monkey_coeffs,
                          title='Average Monkey Coefficients',
                          order=5)

    plt.tight_layout()
    return fig


def plot_model_coefficients(ax, coefficients, title='', order=5):
    """Helper function to plot RLRNN coefficients."""
    # Assume canonical order: [WS, LST, WSW, LS, bias]
    ws = coefficients[0:order]
    lst = coefficients[order:2*order]
    wsw = coefficients[2*order:3*order]
    ls = coefficients[3*order:4*order]
    
    x = np.arange(1, order + 1)
    
    ax.plot(x, ws, 'o-', label='Win-Stay', color='blue', linewidth=2, markersize=6)
    ax.plot(x, lst, 's-', label='Lose-Switch', color='red', linewidth=2, markersize=6)
    ax.plot(x, wsw, '^-', label='Win-Switch', color='orange', linewidth=2, markersize=6)
    ax.plot(x, ls, 'v-', label='Lose-Stay', color='green', linewidth=2, markersize=6)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax.set_xlabel('Trial Lag', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax
    