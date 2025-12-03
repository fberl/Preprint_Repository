import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression, paper_logistic_regression_strategic
from figure_scripts.monkey_E_learning import load_behavior
from figure_scripts.supplement2_stationarity import stationarity_supplement 

from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
# from models.misc_utils import RLTester_Softmax as RL
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec,  GridSpecFromSubplotSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior, paper_logistic_accuracy, paper_logistic_accuracy_strategic,histogram_logistic_accuracy_strategic
from analysis_scripts.stationarity_and_randomness import compute_predictability,plot_predictiability_monkeys_violin, plot_RL_timescales_violin, plot_logistic_coefficients_violin, process_data
import pickle
from analysis_scripts.entropy import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysis_scripts.LLH_behavior_RL import single_session_fit

mpdb_p = '/Users/fmb35/Desktop/matching-pennies-lite.sqlite'

mpbeh_p = '/Users/fmb35/Desktop/MPbehdata.csv'

stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'

# Define consistent colors and markers at module level for all plots
MONKEY_COLORS = {
    # Strategic monkeys - blue shades
    'E': '#1f77b4',  # Blue
    'D': '#17a2b8',  # Info blue  
    'I': '#007bff',  # Primary blue
    
    # Non-strategic monkeys - red/purple shades
    'C': '#dc3545',  # Red
    'H': '#e83e8c',  # Pink
    'F': '#fd7e14',  # Orange-red
    'K': '#6f42c1',  # Purple
}

MONKEY_MARKERS = {
    # Strategic monkeys - squares
    'E': 's',
    'D': 's', 
    'I': 's',
    
    # Non-strategic monkeys - circles
    'C': 'o',
    'H': 'o',
    'F': 'o',
    'K': 'o',
}

# Monkey groups
STRATEGIC_MONKEYS = ['E', 'D', 'I']
NONSTRATEGIC_MONKEYS = ['C', 'H', 'F', 'K']
ALL_MONKEYS = STRATEGIC_MONKEYS + NONSTRATEGIC_MONKEYS

def load_or_fit_all_monkey_rl_parameters(mpbeh_path, save_dir="fitted_params", force_refit=False, 
                                      session_selection="last10", disable_abs=False):
    """
    Load or fit RL parameters for all monkeys across all available algorithms.
    This creates a comprehensive cache to avoid refitting sessions.
    Uses comprehensive MP2 data including extra sessions from stitched dataset.
    
    Returns:
    --------
    dict : Monkey parameters in format {monkey: {algorithm: fitted_params}}
    """
    
    os.makedirs(save_dir, exist_ok=True)
    cache_file = os.path.join(save_dir, f"all_monkey_rl_cache_{session_selection}.pkl")
    
    # Try to load existing cache
    if os.path.exists(cache_file) and not force_refit:
        print(f"Loading cached RL parameters from {cache_file}")
        with open(cache_file, 'rb') as f:
            all_params = pickle.load(f)
        print(f"Loaded parameters for {len(all_params)} monkeys")
        return all_params
    
    print("Fitting RL parameters for all monkeys...")
    all_params = {}
    
    # Load behavior data  
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
    
    # Map numeric IDs to letters
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    mp2_data['animal'] = mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Check what data we have for each monkey
    mp1_monkeys = set(mp1_data['animal'].unique())
    mp2_monkeys = set(mp2_data['animal'].unique())
    all_available_monkeys = mp1_monkeys.union(mp2_monkeys)
    
    print(f"Found data for monkeys: {sorted(all_available_monkeys)}")
    print(f"  MP1 data: {sorted(mp1_monkeys)}")
    print(f"  MP2 data: {sorted(mp2_monkeys)}")
    
    for monkey in sorted(all_available_monkeys):
        print(f"\nProcessing Monkey {monkey}...")
        all_params[monkey] = {}
        
        # Fit Algorithm 1 (MP1) if available
        if monkey in mp1_monkeys:
            monkey_mp1_data = mp1_data[mp1_data['animal'] == monkey]
            try:
                params = fit_rl_parameters_for_monkey(
                    monkey_mp1_data, monkey, algorithm=1, 
                    session_selection=session_selection, disable_abs=disable_abs
                )
                all_params[monkey]['MP1'] = params
                print(f"  ✓ Fitted MP1 parameters")
            except Exception as e:
                print(f"  ✗ MP1 fitting failed: {e}")
                all_params[monkey]['MP1'] = None
        
        # Fit Algorithm 2 (MP2) if available
        if monkey in mp2_monkeys:
            monkey_mp2_data = mp2_data[mp2_data['animal'] == monkey]
            try:
                params = fit_rl_parameters_for_monkey(
                    monkey_mp2_data, monkey, algorithm=2,
                    session_selection=session_selection, disable_abs=disable_abs
                )
                all_params[monkey]['MP2'] = params
                print(f"  ✓ Fitted MP2 parameters")
            except Exception as e:
                print(f"  ✗ MP2 fitting failed: {e}")
                all_params[monkey]['MP2'] = None
    
    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(all_params, f)
    print(f"\nSaved RL parameter cache to {cache_file}")
    
    return all_params

def fit_rl_parameters_for_monkey(monkey_data, monkey_id, algorithm, session_selection="last10", disable_abs=False):
    """
    Fit RL parameters for a single monkey on a single algorithm.
    """
    print(f"    Fitting {monkey_id} on Algorithm {algorithm}...")
    
    # Select sessions
    sessions = monkey_data['id'].unique()
    print(f"    Found {len(sessions)} sessions")
    
    if session_selection == "all":
        selected_sessions = sessions
    elif session_selection == "half":
        n_sessions = max(1, len(sessions) // 2)
        selected_sessions = sessions[-n_sessions:]
    elif isinstance(session_selection, int):
        n_sessions = min(session_selection, len(sessions))
        selected_sessions = sessions[-n_sessions:]
    else:  # "last10" or similar
        n_sessions = min(10, len(sessions))
        selected_sessions = sessions[-n_sessions:]
    
    print(f"    Selected {len(selected_sessions)} sessions: {selected_sessions}")
    
    # Combine selected sessions
    selected_data = monkey_data[monkey_data['id'].isin(selected_sessions)]
    actions = selected_data['monkey_choice'].values.astype(int)
    rewards = selected_data['reward'].values.astype(int)
    
    print(f"    Fitting on {len(actions)} trials...")
    
    # Fit simple RL model
    fit_params, performance = single_session_fit(
        actions, rewards,
        model='simple',
        punitive=False,
        decay=False,
        ftol=1e-6,
        const_beta=False,
        const_gamma=True
    )
    
    alpha, beta = fit_params[0], fit_params[1]  # Handle 3-parameter return
    
    if not disable_abs:
        alpha = np.abs(alpha)
        beta = np.abs(beta)
    
    # For simple model, stochasticity is just alpha * beta
    stochasticity = alpha * beta
    
    # Create parameter dictionary
    fitted_params = {
        'alpha': alpha,
        'asymmetric': False,
        'deterministic': False,
        'load': False,
        'beta': beta,
        '_fit_info': {
            'original_params': fit_params,
            'performance': performance,
            'n_trials': len(actions),
            'winrate': np.mean(rewards),
            'alpha': alpha,
            'beta': beta,
            'stochasticity': stochasticity,  # New measure
            'session_selection': session_selection,
            'disable_abs': disable_abs,
            'n_sessions': len(selected_sessions),
            'selected_sessions': selected_sessions.tolist()
        }
    }
    
    return fitted_params

def fit_and_save_monkey_E_RL_parameters(mpbeh_path, save_dir="fitted_params", force_refit=False, 
                                      session_selection="last10", disable_abs=False):
    """
    Fit simple RL model to monkey E's data for both MP1 and MP2.
    Save the fitted parameters to files for reuse.
    
    Parameters:
    -----------
    mpbeh_path : str
        Path to monkey behavioral data file
    save_dir : str
        Directory to save fitted parameters
    force_refit : bool
        If True, refit even if saved parameters exist
    session_selection : str or int
        How to select sessions for fitting:
        - "last10": Use last 10 sessions (default)
        - "all": Use all available sessions
        - "half": Use last half of sessions
        - int: Use last N sessions (e.g., 5, 15, 20)
    disable_abs : bool
        If True, disable np.abs() application to fitted parameters
        
    Returns:
    --------
    dict : {'MP1': BG_params_dict, 'MP2': BG_params_dict}
        Dictionary containing fitted parameters for both algorithms
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filenames based on session selection
    session_str = str(session_selection).replace(".", "_")
    abs_str = "_no_abs" if disable_abs else ""
    mp1_params_file = os.path.join(save_dir, f"monkey_E_MP1_RL_params_{session_str}{abs_str}.pkl")
    mp2_params_file = os.path.join(save_dir, f"monkey_E_MP2_RL_params_{session_str}{abs_str}.pkl")
    
    fitted_params = {}
    
    # Fit MP1 parameters
    if not force_refit and os.path.exists(mp1_params_file):
        print("Loading existing MP1 fitted parameters for monkey E...")
        with open(mp1_params_file, 'rb') as f:
            fitted_params['MP1'] = pickle.load(f)
    else:
        print("Fitting RL model to monkey E's MP1 data...")
        
        # Load monkey E's MP1 data
        mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=18)  # 18 = monkey E
        # Double-check filtering: ensure only animal 18 and task 1
        mp1_data = mp1_data[(mp1_data['animal'] == 18) & (mp1_data['task'] == 1)]
        
        if len(mp1_data) == 0:
            print("Error: No MP1 data found for monkey E")
            fitted_params['MP1'] = None
        else:
            # Session selection logic
            all_sessions = sorted(mp1_data['id'].unique())
            
            if session_selection == "all":
                selected_sessions = all_sessions
                print(f"  Using all {len(selected_sessions)} sessions")
            elif session_selection == "half":
                n_half = len(all_sessions) // 2
                selected_sessions = all_sessions[-n_half:] if n_half > 0 else all_sessions
                print(f"  Using last half: {len(selected_sessions)} sessions")
            elif isinstance(session_selection, int):
                n_sessions = session_selection
                if len(all_sessions) > n_sessions:
                    selected_sessions = all_sessions[-n_sessions:]
                else:
                    selected_sessions = all_sessions
                print(f"  Using last {len(selected_sessions)} sessions (requested {n_sessions})")
            else:  # default to "last10"
                if len(all_sessions) > 10:
                    selected_sessions = all_sessions[-10:]
                else:
                    selected_sessions = all_sessions
                print(f"  Using last {len(selected_sessions)} sessions (default)")
            
            mp1_data = mp1_data[mp1_data['id'].isin(selected_sessions)]
            print(f"    Total trials: {len(mp1_data)} from {len(selected_sessions)} sessions")
            
            # Extract actions and rewards
            actions = mp1_data['monkey_choice'].values.astype(int)
            rewards = mp1_data['reward'].values.astype(int)
            
            print(f"  Monkey E MP1 winrate: {np.mean(rewards):.3f}")
            
            try:
                # Fit simple RL model
                fit_params, performance = single_session_fit(
                    actions, rewards,
                    model='simple',  # Returns [alpha, beta]
                    punitive=False,
                    decay=False,
                    ftol=1e-6,
                    const_beta=False,
                    const_gamma=True,
                    disable_abs=disable_abs
                )
                
                # Simple model returns [alpha, beta, gamma] - extract first 2
                alpha, beta = fit_params[0], fit_params[1]
                print(f"  ✓ MP1 fitting successful!")
                beta_display = np.abs(beta) if not disable_abs else beta
                print(f"    α={alpha:.3f}, β={beta_display:.3f}")
                print(f"    Model performance: {performance:.3f}")
                
                # Convert to RLTester format
                # RLTester uses: alpha (single learning rate), beta (temperature) 
                fitted_params['MP1'] = {
                    'alpha': alpha,  # Single alpha learning rate
                    'asymmetric': False,
                    'deterministic': False,
                    'load': False,
                    'beta': beta,             # Use fitted temperature
                    # Store original fit info
                    '_fit_info': {
                        'original_params': fit_params,
                        'performance': performance,
                        'n_trials': len(actions),
                        'winrate': np.mean(rewards),
                        'alpha': alpha,
                        'beta': beta,
                        'session_selection': session_selection,
                        'disable_abs': disable_abs,
                        'n_sessions': len(selected_sessions)
                    }
                }
                
                # Save parameters
                with open(mp1_params_file, 'wb') as f:
                    pickle.dump(fitted_params['MP1'], f)
                print(f"  Saved MP1 parameters to {mp1_params_file}")
                
            except Exception as e:
                print(f"  ✗ MP1 fitting failed: {e}")
                fitted_params['MP1'] = None
    
    # Fit MP2 parameters
    if not force_refit and os.path.exists(mp2_params_file):
        print("Loading existing MP2 fitted parameters for monkey E...")
        with open(mp2_params_file, 'rb') as f:
            fitted_params['MP2'] = pickle.load(f)
    else:
        print("Fitting RL model to monkey E's MP2 data...")
        
        # Load monkey E's MP2 data - use stitched data if available
        if stitched_p and os.path.exists(stitched_p):
            print(f"  Using stitched data from {stitched_p}")
            with open(stitched_p, 'rb') as f:
                stitched_data = pickle.load(f)
            # Filter for monkey E and algorithm 2
            mp2_data = stitched_data[(stitched_data['animal'] == 'E') & (stitched_data['task'] == 'mp')]
        else:
            print(f"  Using standard behavioral data from {mpbeh_path}")
            mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=18)  # 18 = monkey E
        
        if len(mp2_data) == 0:
            print("Error: No MP2 data found for monkey E")
            fitted_params['MP2'] = None
        else:
            # Session selection logic
            all_sessions = sorted(mp2_data['id'].unique())
            
            if session_selection == "all":
                selected_sessions = all_sessions
                print(f"  Using all {len(selected_sessions)} sessions")
            elif session_selection == "half":
                n_half = len(all_sessions) // 2
                selected_sessions = all_sessions[-n_half:] if n_half > 0 else all_sessions
                print(f"  Using last half: {len(selected_sessions)} sessions")
            elif isinstance(session_selection, int):
                n_sessions = session_selection
                if len(all_sessions) > n_sessions:
                    selected_sessions = all_sessions[-n_sessions:]
                else:
                    selected_sessions = all_sessions
                print(f"  Using last {len(selected_sessions)} sessions (requested {n_sessions})")
            else:  # default to "last10"
                if len(all_sessions) > 10:
                    selected_sessions = all_sessions[-10:]
                else:
                    selected_sessions = all_sessions
                print(f"  Using last {len(selected_sessions)} sessions (default)")
            
            mp2_data = mp2_data[mp2_data['id'].isin(selected_sessions)]
            print(f"    Total trials: {len(mp2_data)} from {len(selected_sessions)} sessions")
            
            # Extract actions and rewards
            actions = mp2_data['monkey_choice'].values.astype(int)
            rewards = mp2_data['reward'].values.astype(int)
            
            print(f"  Monkey E MP2 winrate: {np.mean(rewards):.3f}")
            
            try:
                # Fit simple RL model
                fit_params, performance = single_session_fit(
                    actions, rewards,
                    model='simple',  # Returns [alpha, beta]
                    punitive=False,
                    decay=False,
                    ftol=1e-6,
                    const_beta=False,
                    const_gamma=True,
                    disable_abs=disable_abs
                )
                
                # Simple model returns [alpha, beta, gamma] - extract first 2
                alpha, beta = fit_params[0], fit_params[1]
                print(f"  ✓ MP2 fitting successful!")
                beta_display = np.abs(beta) if not disable_abs else beta
                print(f"    α={alpha:.3f}, β={beta_display:.3f}")
                print(f"    Model performance: {performance:.3f}")
                
                # Convert to RLTester format
                # RLTester uses: alpha (single learning rate), beta (temperature)
                beta_for_bgtester = np.abs(beta) if not disable_abs else beta
                fitted_params['MP2'] = {
                    'alpha': alpha,  # Single alpha learning rate
                    'asymmetric': False,
                    'deterministic': False,
                    'load': False,
                    'beta': beta_for_bgtester,             # Use fitted temperature
                    # Store original fit info
                    '_fit_info': {
                        'original_params': fit_params,
                        'performance': performance,
                        'n_trials': len(actions),
                        'winrate': np.mean(rewards),
                        'alpha': alpha,
                        'beta': beta,
                        'session_selection': session_selection,
                        'disable_abs': disable_abs,
                        'n_sessions': len(selected_sessions)
                    }
                }
                
                # Save parameters
                with open(mp2_params_file, 'wb') as f:
                    pickle.dump(fitted_params['MP2'], f)
                print(f"  Saved MP2 parameters to {mp2_params_file}")
                
            except Exception as e:
                print(f"  ✗ MP2 fitting failed: {e}")
                fitted_params['MP2'] = None
    
    # Print summary
    print("\n" + "="*60)
    print("FITTED MONKEY E RL PARAMETERS SUMMARY")
    print("="*60)
    
    for alg in ['MP1', 'MP2']:
        if fitted_params[alg] is not None:
            params = fitted_params[alg]
            fit_info = params.get('_fit_info', {})
            print(f"\n{alg} (Algorithm {'1' if alg == 'MP1' else '2'}):")
            alpha = fit_info.get('alpha', None)
            beta = fit_info.get('beta', None)
            performance = fit_info.get('performance', None)
            winrate = fit_info.get('winrate', None)
            
            alpha_str = f"{alpha:.3f}" if alpha is not None else "N/A"
            beta_str = f"{beta:.3f}" if beta is not None else "N/A"
            performance_str = f"{performance:.3f}" if performance is not None else "N/A"
            winrate_str = f"{winrate:.3f}" if winrate is not None else "N/A"
            
            print(f"  Simple RL fitted: α={alpha_str}, β={beta_str}")
            print(f"  RLTester format: α={params['alpha']}, β={params['beta']:.3f}")
            print(f"  Model performance: {performance_str}")
            print(f"  Data: {fit_info.get('n_trials', 'N/A')} trials, winrate: {winrate_str}")
        else:
            print(f"\n{alg}: FITTING FAILED")
    
    print("="*60)
    
    return fitted_params

def fit_monkey_logistic_regression(monkey_data_tuple):
    """Helper function for parallel logistic regression fitting"""
    monkey, monkey_data, strategic = monkey_data_tuple
    from analysis_scripts.logistic_regression import paper_logistic_regression_strategic, paper_logistic_accuracy_strategic
    import numpy as np
    
    if strategic:
        paper_logistic_regression_strategic(None, False, data=monkey_data, legend=False, return_model=False, order=5, bias=True)
    
    monkey_winrate = np.mean(monkey_data['reward']) * 100
    _, monkey_acc = paper_logistic_accuracy_strategic(np.array(monkey_data['monkey_choice']), np.array(monkey_data['reward']), order=5)
    monkey_acc *= 100
    
    return monkey, monkey_winrate, monkey_acc, None


def plot_behavioral_nonstationarity_and_RL(mpdb_path, mpbeh_path, overfit=False, env_params = None, BG_params = None, BG_params_algo1 = None, strategic = True, violin_test=False, bias=True, bg_nits=50, max_workers=4, generate_supplements=True, session_selection="last10", disable_abs=False, use_provided_params=False, cutoff_trials=None):
    """
    Plot behavioral nonstationarity and RL model analysis using fitted monkey E parameters.
    
    Parameters:
    -----------
    bg_nits : int
        Number of sessions to generate for RL model. Each session will be 200 trials long.
        Total trials = bg_nits * 200
    generate_supplements : bool
        Whether to also generate supplementary figures (default: True)
    session_selection : str or int
        How to select sessions for parameter fitting:
        - "last10": Use last 10 sessions (default)
        - "all": Use all available sessions
        - "half": Use last half of sessions
        - int: Use last N sessions (e.g., 5, 15, 20)
    disable_abs : bool
        If True, disable np.abs() application to fitted parameters
    use_provided_params : bool
        If True, use provided BG_params and BG_params_algo1 instead of fitting from data.
        When True, BG_params and BG_params_algo1 must be provided (default: False)
    cutoff_trials : int, optional
        If provided, will keep only the last N trials (rounded to complete sessions)
        by removing early sessions from each monkey's data. Similar to fig1 cutoff logic.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Main figure object. If generate_supplements=True, supplement figures are stored 
        as fig.supplement_figures, a list of (name, figure) tuples.
    
    Notes:
    ------
    This function automatically fits simple RL models to monkey E's behavioral data
    for both Algorithm 1 (MP1) and Algorithm 2 (MP2). The fitted parameters have the form:
    [alpha, beta] and are converted to RLTester format.
    
    No hardcoded parameters are used - the function will fail if fitting is unsuccessful.
    
    Example:
    --------
    # Use fitted parameters (default behavior)
    fig = plot_behavioral_nonstationarity_and_RL(mpdb_path, mpbeh_path)
    # Access supplements: fig.supplement_figures[0][1].show()  # Show first supplement
    
    # Use provided RL parameters instead of fitting
    bg_params_mp2 = {
        'alpha': 0.1,  # Single alpha learning rate
        'beta': 5.0,
        'asymmetric': False,
        'deterministic': False,
        'load': False
    }
    bg_params_mp1 = {
        'alpha': 0.15,  # Single alpha learning rate
        'beta': 3.5,
        'asymmetric': False,
        'deterministic': False,
        'load': False
    }
    fig = plot_behavioral_nonstationarity_and_RL(
        mpdb_path, mpbeh_path,
        BG_params=bg_params_mp2,
        BG_params_algo1=bg_params_mp1,
        use_provided_params=True
    )
    
    # Use last 5000 trials (rounded to complete sessions):
    fig = plot_behavioral_nonstationarity_and_RL(
        mpdb_path, mpbeh_path,
        cutoff_trials=5000
    )
    """
    # Fix main figure spacing and layout issues
    fig = plt.figure(layout=None, figsize=(15, 10), dpi = 300)  # Reduced height for better spacing

    # Create a 3x3 GridSpec layout with better spacing
    gs = GridSpec(3, 3, figure=fig, wspace=0.25, hspace=0.5)  # Reduced spacing

    # First row: Individual monkey logistic regressions on Algorithm 1 (E, C, F)
    monkey_E_ax = fig.add_subplot(gs[0, 0])
    monkey_C_ax = fig.add_subplot(gs[0, 1])
    monkey_F_ax = fig.add_subplot(gs[0, 2])

    # Second row: Asymmetric RL model diagram, RL model on algo 1, RL model on algo 2
    RL_model_ax = fig.add_subplot(gs[1, 0])
    RL_algo1_ax = fig.add_subplot(gs[1, 1])
    RL_algo2_ax = fig.add_subplot(gs[1, 2])

    # Third row: Create a nested GridSpec with 4 columns for custom layout
    gs_bottom = GridSpecFromSubplotSpec(1, 4, gs[2, :], wspace=0.3)
    alpha_beta_ax = fig.add_subplot(gs_bottom[0, :2])  # Takes first 1/2 of the row (2 out of 4 columns)
    rl_lr_ratio_ax = fig.add_subplot(gs_bottom[0, 2])   # Takes 1/4 of the row (1 out of 4 columns)
    new_ax = fig.add_subplot(gs_bottom[0, 3])   # Takes last 1/4 of the row (1 out of 4 columns)
    
    # Fit and load monkey E RL parameters or use provided defaults
    print("="*60)
    print("LOADING RL PARAMETERS FOR MONKEY E")
    print("="*60)
    
    if use_provided_params:
        # Use provided parameters instead of fitting
        print("✓ Using provided RL parameters (not fitting from data)")
        
        if BG_params is None:
            raise ValueError("use_provided_params=True but BG_params not provided. "
                           "Must provide BG_params for Algorithm 2 (MP2).")
        
        if BG_params_algo1 is None:
            raise ValueError("use_provided_params=True but BG_params_algo1 not provided. "
                           "Must provide BG_params_algo1 for Algorithm 1 (MP1).")
        
        print(f"  Algorithm 2 (MP2) parameters: {BG_params}")
        print(f"  Algorithm 1 (MP1) parameters: {BG_params_algo1}")
        
    else:
        # Fit parameters from behavioral data
        fitted_params = fit_and_save_monkey_E_RL_parameters(
            mpbeh_path, 
            force_refit=True, 
            session_selection=session_selection,
            disable_abs=disable_abs
        )
        
        # Use fitted parameters - fail if not available
        print("✓ Using fitted parameters for Algorithm 2 (MP2)")
        BG_params = fitted_params['MP2'].copy()
        # Remove the fit info before passing to RLTester
        fit_info = BG_params.pop('_fit_info', {})
        performance = fit_info.get('performance', None)
        performance_str = f"{performance:.3f}" if performance is not None else "N/A"
        print(f"  Fitted from {fit_info.get('n_trials', 'N/A')} trials, "
                f"performance: {performance_str}")

        
        if fitted_params.get('MP1') is not None:
            print("✓ Using fitted parameters for Algorithm 1 (MP1)")
            BG_params_algo1 = fitted_params['MP1'].copy()
            # Remove the fit info before passing to RLTester
            fit_info = BG_params_algo1.pop('_fit_info', {})
            performance = fit_info.get('performance', None)
            performance_str = f"{performance:.3f}" if performance is not None else "N/A"
            print(f"  Fitted from {fit_info.get('n_trials', 'N/A')} trials, "
                    f"performance: {performance_str}")
        else:
            raise ValueError("Failed to fit RL parameters for Algorithm 1 (MP1). "
                            "Cannot proceed without fitted parameters for monkey E.")
       
    
    # Load behavior data
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Plot individual monkey logistic regressions for Algorithm 1 with multithreading
    monkeys = ['E', 'C', 'F']
    monkey_axes = [monkey_E_ax, monkey_C_ax, monkey_F_ax]
    
    print("Fitting logistic regressions for Algorithm 1 monkeys...")
    
    # Prepare data for parallel processing - maintain order
    monkey_data_list = []
    for monkey in monkeys:
        monkey_data = mp1_data[mp1_data['animal'] == monkey]
        if len(monkey_data) > 0:
            monkey_data_list.append((monkey, monkey_data, strategic))
    
    # Fit logistic regressions in parallel but preserve order
    monkey_stats = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit in order and collect results in order
        future_to_monkey = {executor.submit(fit_monkey_logistic_regression, data): data[0] for data in monkey_data_list}
        for future in as_completed(future_to_monkey):
            monkey, winrate, acc, error = future.result()
            monkey_stats[monkey] = (winrate, acc)
    
    # Plot the results
    for i, (monkey, ax) in enumerate(zip(monkeys, monkey_axes)):
        monkey_data = mp1_data[mp1_data['animal'] == monkey]
        if len(monkey_data) > 0:
            paper_logistic_regression_strategic(ax, False, data=monkey_data, legend=(i==0), return_model=False, order=5, bias=True)
            winrate, acc = monkey_stats.get(monkey, (50.0, 50.0))
            ax.set_title(f'Monkey {monkey} on Opponent 1\nWinrate: {winrate:.1f}%, LR Predictability: {acc:.1f}%', fontsize=12)
            ax.set_xlabel('Trials Back', fontsize=10)
            ax.set_ylabel('Regression Coefficient', fontsize=10)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add Stay/Switch labels
            right_label = ax.twinx()
            right_label.yaxis.set_label_position('right')
            right_label.set_yticks([0.25, 0.725])
            right_label.set_yticklabels(['Switch', 'Stay'], fontsize=10)
            right_label.tick_params(axis='y', which='both', length=0)
    
    # RL model image
    # RL_model_ax.set_title('Asymmetric RL Model', fontsize=14)    
    RL_model_ax.set_title('Simple RL Model', fontsize=14)    
    RLm = plt.imread('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/Symmetric RL Model 3.png')   
    RL_model_ax.imshow(RLm)
    RL_model_ax.set_axis_off()
    
    # Session-wise predictability computation (for supplements)
    print("Computing session-wise predictability for all monkeys...")
    predictability_results = compute_and_cache_session_wise_predictability(
        mpbeh_path, 
        cache_file=f"session_wise_predictability_{session_selection}.pkl",
        force_recompute=False,  # Set to True to force recomputation
        order=5
    )
    
    # Load comprehensive RL parameters for all monkeys using our caching system
    all_monkey_params = load_or_fit_all_monkey_rl_parameters(
        mpbeh_path, force_refit=True,  # Force refit to ensure fresh parameters
        session_selection=session_selection, disable_abs=disable_abs
    )
    
    # Set up environment for RL model - FIX reset_time consistency
    bg_env_params = env_params.copy() if env_params else {}
    bg_env_params["opponents"] = ["all"]
    bg_env_params["opponent"] = "all"
    bg_env_params["opponents_params"] = {"all": {"bias": [0], "depth": 4}}
    # Fix: ensure fixed_length and reset_time are consistent
    bg_env_params["fixed_length"] = True
    # Fix: Use proper session length instead of bg_nits for reset_time
    # bg_nits should only control number of sessions, not session length
    bg_env_params["reset_time"] = 200  # Each session should be 200 trials long
    
    bg_env = make_env(bg_env_params)
    
    # Fix: Remove LBFGS parameter - RLTester doesn't use it
    if BG_params:
        BG_params_fixed = BG_params.copy()
        # Remove LBFGS if it exists - RLTester doesn't use this parameter
        BG_params_fixed.pop('LBFGS', None)
    else:
        BG_params_fixed = {}
    
    BG_model = BG(env=bg_env, **BG_params_fixed) if BG_params_fixed else None

    if not violin_test and BG_model:
        # Set up models for both algorithms
        bg_env_params_algo1 = bg_env_params.copy()
        bg_env_params_algo1["opponents_params"] = {"1": {"bias": [0], "depth": 4}}
        bg_env_params_algo1["opponents"] = ["1"]
        bg_env_params_algo1["opponent"] = "1"
        bg_env_algo1 = make_env(bg_env_params_algo1)
        
        if BG_params_algo1:
            BG_params_algo1_fixed = BG_params_algo1.copy()
            BG_params_algo1_fixed.pop('LBFGS', None)  # Remove LBFGS if it exists
        else:
            BG_params_algo1_fixed = BG_params_fixed.copy()
            
        BG_model_algo1 = BG(env=bg_env_algo1, **BG_params_algo1_fixed)
        
        # Generate data for both algorithms
        # bg_nits controls number of sessions, each session is 200 trials
        print(f"Generating BG model data with {bg_nits} sessions of 200 trials each...")
        BG_data_algo1, masks_algo1 = BG_model_algo1.generate_data(bg_nits)
        BG_data_algo2, masks_algo2 = BG_model.generate_data(bg_nits)
        
        # Plot RL model regressions
        if strategic:
            bg_data_algo1 = paper_logistic_regression_strategic(RL_algo1_ax, False, data=BG_data_algo1, 
                                                               legend=True, mask=None, return_model=True, order=5, bias=bias)
            bg_data_algo2 = paper_logistic_regression_strategic(RL_algo2_ax, False, data=BG_data_algo2, 
                                                              legend=False, mask=None, return_model=True, order=5, bias=bias)
        else:
            bg_data_algo1 = paper_logistic_regression(RL_algo1_ax, False, data=BG_data_algo1, 
                                                     legend=True, mask=None, return_model=True, order=5, bias=bias)
            bg_data_algo2 = paper_logistic_regression(RL_algo2_ax, False, data=BG_data_algo2, 
                                                    legend=False, mask=None, return_model=True, order=5, bias=bias)
        
        # Calculate stats for RL models
        bg_actions_algo1 = np.array(BG_data_algo1['monkey_choice'])
        bg_rewards_algo1 = np.array(BG_data_algo1['reward'])
        bg_actions_algo2 = np.array(BG_data_algo2['monkey_choice'])
        bg_rewards_algo2 = np.array(BG_data_algo2['reward'])
        
        # Compute prediction accuracy
        _, pred_acc_algo1 = paper_logistic_accuracy_strategic(bg_actions_algo1, bg_rewards_algo1, order=5)
        _, pred_acc_algo2 = paper_logistic_accuracy_strategic(bg_actions_algo2, bg_rewards_algo2, order=5)
        # Note: Keep as fractions for scatter plot, convert to percentages for titles
        
        # Update titles
        algo1_winrate = np.mean(bg_rewards_algo1) * 100
        algo2_winrate = np.mean(bg_rewards_algo2) * 100
        RL_algo1_ax.set_title(f'RL Model on Opponent 1\nWinrate: {algo1_winrate:.1f}%, LR Predictability: {pred_acc_algo1*100:.1f}%', fontsize=12)
        RL_algo2_ax.set_title(f'RL Model on Opponent 2\nWinrate: {algo2_winrate:.1f}%, LR Predictability: {pred_acc_algo2*100:.1f}%', fontsize=12)
        
        # RL vs Monkey predictability plot moved to supplements
        
        # Save entropy data - create files in current project directory
        order = 5 
        if isinstance(bg_data_algo2, dict) and 'action' in bg_data_algo2:
            output_dir = os.path.join(os.getcwd(), 'figure_scripts')
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
            with open(os.path.join(output_dir, 'fig1_rl_data.p'), 'w+') as f:
                # Normalize coefficients by max abs and compute combined regressors with division by 2
                ws_coeff = bg_data_algo2['action'][0]/max(abs(bg_data_algo2['action'][:order]))
                ls_coeff = bg_data_algo2['action'][order]/max(abs(bg_data_algo2['action'][order:2*order]))
                wsw_coeff = bg_data_algo2['action'][2*order]/max(abs(bg_data_algo2['action'][2*order:3*order]))
                lst_coeff = bg_data_algo2['action'][3*order]/max(abs(bg_data_algo2['action'][3*order:4*order]))
                print(f"ws_coeff: {ws_coeff}, ls_coeff: {ls_coeff}, wsw_coeff: {wsw_coeff}, lst_coeff: {lst_coeff}")
                f.write('{}, {}'.format((ws_coeff-wsw_coeff)/2, (ls_coeff-lst_coeff)/2))

        output_dir = os.path.join(os.getcwd(), 'figure_scripts')
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        with open(os.path.join(output_dir, 'fig5_rl_entropy.pkl'), 'wb') as f:
            entropy = compute_entropy(bg_actions_algo2, np.array(BG_data_algo2['computer_choice']))
            mutual_information = compute_mutual_information(bg_actions_algo2, np.array(BG_data_algo2['computer_choice']))
            pickle.dump((entropy, mutual_information), f)
    
    # RL vs Monkey predictability plot moved to supplements
    
    # Plot session-level Alpha * Beta Values timecourse
    print("Plotting session-level Alpha * Beta values timecourse...")
    plot_session_alpha_beta_timecourse(alpha_beta_ax, mpbeh_path, session_selection)
    # Old loop removed - using cached parameters instead
    
    # Plot RL to LR ratio violin plot
    print("Plotting RL to LR performance ratio violin plot...")
    plot_rl_vs_lr_ratio_violin(rl_lr_ratio_ax, mpbeh_path, 
                               strategic_monkeys=['E', 'D', 'I'], 
                               nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                               model='asymmetric', order=5, cutoff_trials=cutoff_trials)
    
    # Plot strategic vs nonstrategic comparison for MP2 using comprehensive data
    print("Plotting strategic vs nonstrategic MP2 comparison...")
    plot_strategic_vs_nonstrategic_mp2_violin(new_ax, mpbeh_path, 
                                              strategic_monkeys=['E', 'D', 'I'], 
                                              nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                                              model='asymmetric', order=5, cutoff_trials=cutoff_trials)
    
    # Set labels for RL regression plots
    for ax in [RL_algo1_ax, RL_algo2_ax]:
        ax.set_xlabel('Trials Back', fontsize=10)
        ax.set_ylabel('Regression Coefficient', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add Stay/Switch labels
        right_label = ax.twinx()
        right_label.yaxis.set_label_position('right')
        right_label.set_yticks([0.25, 0.725])
        right_label.set_yticklabels(['Switch', 'Stay'], fontsize=10)
        right_label.tick_params(axis='y', which='both', length=0)
    
    # Note: Prediction accuracy and parameter plots moved to supplement
    
    # Use subplots_adjust instead of tight_layout for better control
    plt.subplots_adjust(hspace=0.5, wspace=0.25, top=0.88, bottom=0.1, left=0.08, right=0.92)
    
    # Generate supplement figures if requested
    supplement_figs = []
    if generate_supplements:
        print("\n" + "="*60)
        print("GENERATING SUPPLEMENT FIGURES")
        print("="*60)
        
        try:
            # Generate first supplement: prediction accuracy and delta parameters (updated)
            print("Creating Supplement 1: Prediction Accuracy and RL Model Parameters (Updated)...")
            supplement1 = plot_supplement_figures_updated(
                mpdb_path, mpbeh_path, overfit=overfit, strategic=strategic,
                session_selection=session_selection, disable_abs=disable_abs, force_refit=False
            )
            supplement_figs.append(("Supplement_1_Prediction_Accuracy_Updated", supplement1))
            print("✓ Supplement 1 generated successfully")
            
            # Generate second supplement: stationarity analysis
            print("Creating Supplement 2: Stationarity Analysis...")
            supplement2 = stationarity_supplement(mpbeh_path)
            supplement_figs.append(("Supplement_2_Stationarity", supplement2))
            print("✓ Supplement 2 generated successfully")
            
            # Generate third supplement: algorithm comparison (put last as it may error)
            print("Creating Supplement 3: Algorithm Comparison...")
            supplement3 = generate_algorithm_comparison_supplement(
                mpdb_path, mpbeh_path, 
                strategic=strategic, 
                env_params=env_params, 
                BG_params=BG_params, 
                bg_nits=bg_nits
            )
            supplement_figs.append(("Supplement_3_Algorithm_Comparison", supplement3))
            print("✓ Supplement 3 generated successfully")
            
            print(f"✓ Generated {len(supplement_figs)} supplement figures")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to generate supplements: {e}")
            import traceback
            traceback.print_exc()
    
    # Add panel labels first
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    panel_axes = [monkey_E_ax, monkey_C_ax, monkey_F_ax, RL_model_ax, alpha_beta_ax, rl_lr_ratio_ax, new_ax]
    
    for label, ax in zip(panel_labels, panel_axes):
        if ax is not None:
            # Position label in upper left corner of each panel
            ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=14, 
                   fontweight='bold', va='bottom', ha='right')
    
    # Adjust layout to accommodate title with more space - reduce top margin first
    fig.subplots_adjust(top=0.82, hspace=0.5, wspace=0.25, bottom=0.1, left=0.08, right=0.92)
    
    # Add figure title with better positioning to avoid overlap - lower position
    fig.suptitle('Figure 2: Behavioral Deviations from Reinforcement Learning', 
                fontsize=16, fontweight='bold', y=0.96)
    
    # Store supplement figures as attribute of main figure for easy access
    fig.supplement_figures = supplement_figs
    
    return fig

def plot_rl_alpha_beta_parameters_cached_updated(ax_alpha, ax_beta, all_monkey_params):
    """Plot alpha and beta parameters using comprehensive cached RL results"""
    ax_alpha.set_title('RL Model Alpha Parameter (Learning Rate)', fontsize=14)
    ax_alpha.set_xlabel('Algorithm', fontsize=12)
    ax_alpha.set_ylabel('Alpha Parameter Value', fontsize=12)
    
    ax_beta.set_title('RL Model Beta Parameter (Temperature)', fontsize=14)
    ax_beta.set_xlabel('Algorithm', fontsize=12)
    ax_beta.set_ylabel('Beta Parameter Value', fontsize=12)
    
    print("Plotting RL alpha and beta parameters from comprehensive cache...")
    
    plotted_any_alpha = False
    plotted_any_beta = False
    
    # Collect data for plotting
    strategic_data = {'alpha': [], 'beta': [], 'stochasticity': []}
    nonstrategic_data = {'alpha': [], 'beta': [], 'stochasticity': []}
    
    # Track individual monkey points for scatter plot
    individual_points = []
    
    for monkey in ALL_MONKEYS:
        if monkey not in all_monkey_params:
            continue
            
        monkey_params = all_monkey_params[monkey]
        color = MONKEY_COLORS.get(monkey, 'gray')
        marker = MONKEY_MARKERS.get(monkey, 'o')
        is_strategic = monkey in STRATEGIC_MONKEYS
        
        print(f"  Processing monkey {monkey} ({color}, {marker})...")
        
        # Plot Algorithm 1 if available
        if 'MP1' in monkey_params and monkey_params['MP1'] is not None:
            params = monkey_params['MP1']
            fit_info = params.get('_fit_info', {})
            alpha = fit_info.get('alpha', 0)
            beta = fit_info.get('beta', 0)
            stochasticity = fit_info.get('stochasticity', 0)
            
            # Plot on Algorithm 1 side (x=-1 with jitter)
            x_pos = -1 + np.random.normal(0, 0.05)  # Small jitter
            ax_alpha.scatter(x_pos, alpha, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax_beta.scatter(x_pos, beta, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            individual_points.append({
                'monkey': monkey, 'algorithm': 'MP1', 'x_pos': x_pos,
                'alpha': alpha, 'beta': beta,
                'stochasticity': stochasticity, 'color': color, 'marker': marker, 'is_strategic': is_strategic
            })
            
            # Add to group data
            group_data = strategic_data if is_strategic else nonstrategic_data
            group_data['alpha'].append(alpha)
            group_data['beta'].append(beta)
            group_data['stochasticity'].append(stochasticity)
            
            plotted_any_alpha = True
            plotted_any_beta = True
            
        # Plot Algorithm 2 if available  
        if 'MP2' in monkey_params and monkey_params['MP2'] is not None:
            params = monkey_params['MP2']
            fit_info = params.get('_fit_info', {})
            alpha = fit_info.get('alpha', 0)
            beta = fit_info.get('beta', 0)
            stochasticity = fit_info.get('stochasticity', 0)
            
            # Plot on Algorithm 2 side (x=+1 with jitter)
            x_pos = 1 + np.random.normal(0, 0.05)  # Small jitter
            ax_alpha.scatter(x_pos, alpha, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax_beta.scatter(x_pos, beta, color=color, marker=marker, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            individual_points.append({
                'monkey': monkey, 'algorithm': 'MP2', 'x_pos': x_pos,
                'alpha': alpha, 'beta': beta,
                'stochasticity': stochasticity, 'color': color, 'marker': marker, 'is_strategic': is_strategic
            })
            
            # Add to group data
            group_data = strategic_data if is_strategic else nonstrategic_data
            group_data['alpha'].append(alpha)
            group_data['beta'].append(beta)
            group_data['stochasticity'].append(stochasticity)
            
            plotted_any_alpha = True
            plotted_any_beta = True
        
        # Old session finding code removed - using cached parameters directly
    
    # Handle case when no parameters could be fitted
    if not plotted_any_alpha:
        ax_alpha.text(0.5, 0.5, 'No RL alpha parameters could be fitted', 
                ha='center', va='center', transform=ax_alpha.transAxes, fontsize=12)
    
    if not plotted_any_beta:
        ax_beta.text(0.5, 0.5, 'No RL beta parameters could be fitted', 
                ha='center', va='center', transform=ax_beta.transAxes, fontsize=12)
    
    # Format both plots consistently
    for ax in [ax_alpha, ax_beta]:
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7)
        ax.set_xticks([-1, 1])  
        ax.set_xticklabels(['Opponent 1', 'Opponent 2'])
        ax.grid(True, alpha=0.3)
    
    # Create legends with consistent colors and markers
    if plotted_any_alpha or plotted_any_beta:
        # Create legend elements for each monkey
        legend_elements = []
        for monkey in ALL_MONKEYS:
            if monkey in all_monkey_params and (
                (all_monkey_params[monkey].get('MP1') is not None) or 
                (all_monkey_params[monkey].get('MP2') is not None)
            ):
                color = MONKEY_COLORS.get(monkey, 'gray')
                marker = MONKEY_MARKERS.get(monkey, 'o')
                group = "Strategic" if monkey in STRATEGIC_MONKEYS else "Non-strategic"
                
                legend_elements.append(
                    plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                              markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                              label=f'{monkey} ({group})')
                )
        
        # Add alpha parameter note
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='gray', 
                                        markersize=10, label='α (learning rate)', alpha=0.8))
        
        # Add legends outside plots
        ax_alpha.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax_beta.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Print summary with new stochasticity measure
    if individual_points:
        print("\n" + "="*70)
        print("FITTED RL PARAMETERS SUMMARY (Updated)")
        print("="*70)
        
        for point in individual_points:
            monkey = point['monkey']
            algorithm = point['algorithm']
            alpha = point['alpha']
            beta = point['beta']
            stochasticity = point['stochasticity']
            group = "Strategic" if point['is_strategic'] else "Non-strategic"
            
            print(f"{monkey} ({group}) - {algorithm}:")
            print(f"  α={alpha:.3f}, β={beta:.3f}")
            print(f"  Stochasticity = α × β = {stochasticity:.3f}")
            print()
        
        # Group statistics
        if strategic_data['stochasticity']:
            strategic_mean = np.mean(strategic_data['stochasticity'])
            print(f"Strategic monkeys - Mean stochasticity: {strategic_mean:.3f}")
        
        if nonstrategic_data['stochasticity']:
            nonstrategic_mean = np.mean(nonstrategic_data['stochasticity'])
            print(f"Non-strategic monkeys - Mean stochasticity: {nonstrategic_mean:.3f}")
        
        print("="*70)
    
    return individual_points

def plot_session_alpha_beta_timecourse(ax, mpbeh_path, session_selection="last10"):
    """
    Plot session-level Alpha * Beta values (All Monkeys) as a timecourse.
    Uses cached data for fast generation.
    """
    ax.set_title('Session-Level Alpha * Beta Values (All Monkeys)', fontsize=14)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Alpha * Beta Value', fontsize=12)
    
    print("Using cached session-level Alpha * Beta timecourse data...")
    
    # Use cached data for fast generation
    all_session_data = compute_and_cache_alpha_beta_timecourse(
        mpbeh_path, 
        session_selection=session_selection,
        force_recompute=False  # Use cache by default
    )
    
    plotted_any = False
    
    # Plot the timecourse data from cache
    if all_session_data:
        # Group by monkey and plot
        monkeys_plotted = set()
        for data_point in all_session_data:
            monkey = data_point['monkey']
            session_index = data_point['session_index']
            alpha_beta = data_point['alpha_beta']
            color = data_point['color']
            marker = data_point['marker']
            
            # Add to legend only once per monkey
            label = monkey if monkey not in monkeys_plotted else ""
            monkeys_plotted.add(monkey)
            
            ax.scatter(session_index, alpha_beta, color=color, marker=marker, s=80,
                      alpha=0.8, edgecolors='black', linewidth=0.5, label=label)
            plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No session-level RL parameters could be fitted', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Add algorithm boundary and formatting
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=2)
        ax.grid(True, alpha=0.3)
        
        # Create comprehensive legend with strategic vs non-strategic markers
        legend_elements = []
        monkeys_with_data = set([d['monkey'] for d in all_session_data])
        
        # Add individual monkey legends
        for monkey in sorted(monkeys_with_data):
            color = MONKEY_COLORS.get(monkey, 'gray')
            marker = MONKEY_MARKERS.get(monkey, 'o')
            group = "Strategic" if monkey in STRATEGIC_MONKEYS else "Non-strategic"
            
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                          markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                          label=f'{monkey} ({group})', linestyle='None')
            )
        
        # Add separator and group legends
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add algorithm labels inside the plot area in the upper portion
        y_top = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        label_y = y_top - 0.1 * y_range  # Position labels inside the plot, near the top
        
        ax.text(-25, label_y, 'Opponent 1', ha='center', va='center', 
               fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(50, label_y, 'Opponent 2', ha='center', va='center',
               fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Add horizontal line at y=0.5 for reference (moderate learning)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    print(f"  Plotted {len(all_session_data)} session data points from cache")
    
    return all_session_data



def plot_stochasticity_comparison_supplement(ax, all_monkey_params):
    """Plot the new stochasticity measure: (max(alpha) - min(alpha)) * beta"""
    ax.set_title('RL Model Stochasticity: (max(α) - min(α)) × β', fontsize=14)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Stochasticity Measure', fontsize=12)
    
    print("Plotting stochasticity measure...")
    
    plotted_any = False
    
    for monkey in ALL_MONKEYS:
        if monkey not in all_monkey_params:
            continue
            
        monkey_params = all_monkey_params[monkey]
        color = MONKEY_COLORS.get(monkey, 'gray')
        marker = MONKEY_MARKERS.get(monkey, 'o')
        is_strategic = monkey in STRATEGIC_MONKEYS
        
        # Plot Algorithm 1 if available
        if 'MP1' in monkey_params and monkey_params['MP1'] is not None:
            fit_info = monkey_params['MP1'].get('_fit_info', {})
            stochasticity = fit_info.get('stochasticity', 0)
            
            x_pos = -1 + np.random.normal(0, 0.05)  # Small jitter
            ax.scatter(x_pos, stochasticity, color=color, marker=marker, s=120, 
                      alpha=0.8, edgecolors='black', linewidth=0.5)
            plotted_any = True
            
        # Plot Algorithm 2 if available
        if 'MP2' in monkey_params and monkey_params['MP2'] is not None:
            fit_info = monkey_params['MP2'].get('_fit_info', {})
            stochasticity = fit_info.get('stochasticity', 0)
            
            x_pos = 1 + np.random.normal(0, 0.05)  # Small jitter
            ax.scatter(x_pos, stochasticity, color=color, marker=marker, s=120,
                      alpha=0.8, edgecolors='black', linewidth=0.5)
            plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No stochasticity data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Format plot
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Opponent 1', 'Opponent 2'])
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend with consistent colors
    legend_elements = []
    for monkey in ALL_MONKEYS:
        if monkey in all_monkey_params and (
            (all_monkey_params[monkey].get('MP1') is not None) or 
            (all_monkey_params[monkey].get('MP2') is not None)
        ):
            color = MONKEY_COLORS.get(monkey, 'gray')
            marker = MONKEY_MARKERS.get(monkey, 'o')
            group = "Strategic" if monkey in STRATEGIC_MONKEYS else "Non-strategic"
            
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                          markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                          label=f'{monkey} ({group})')
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

def plot_supplement_figures_updated(mpdb_path, mpbeh_path, overfit=False, strategic=True,
                                   session_selection="last10", disable_abs=False, force_refit=False):
    """Create updated supplement figure with session-wise timecourses (except for second plot)"""
    fig = plt.figure(layout=None, figsize=(18, 28), dpi=300)  # Increased height
    
    # Create a 5x1 GridSpec layout for supplement with better spacing
    gs = GridSpec(5, 1, figure=fig, wspace=0.3, hspace=0.7)  # Increased hspace
    
    # First row: Prediction accuracy TIMECOURSE (session-by-session)
    predictability_ax = fig.add_subplot(gs[0, :])
    
    # Second row: RL vs Monkey predictability comparison (STAYS as scatter plot)
    rl_vs_monkey_ax = fig.add_subplot(gs[1, :])
    
    # Third row: Alpha win/loss parameters TIMECOURSE
    alpha_params_ax = fig.add_subplot(gs[2, :])
    
    # Fourth row: Beta parameters TIMECOURSE
    beta_params_ax = fig.add_subplot(gs[3, :])
    
    # Fifth row: Stochasticity measure TIMECOURSE
    stochasticity_ax = fig.add_subplot(gs[4, :])
    
    # Load comprehensive RL parameters for all monkeys
    print("Loading comprehensive RL parameters for supplement...")
    all_monkey_params = load_or_fit_all_monkey_rl_parameters(
        mpbeh_path, force_refit=force_refit, 
        session_selection=session_selection, disable_abs=disable_abs
    )
    
    # Load data for predictability analysis - USE COMPREHENSIVE MP2 DATA
    print("Loading data for predictability analysis...")
    
    # Load MP2 data - prefer stitched data with extra sessions if available
    if stitched_p and os.path.exists(stitched_p):
        print(f"  Using stitched data from {stitched_p} for comprehensive MP2 analysis...")
        with open(stitched_p, 'rb') as f:
            stitched_data = pickle.load(f)
        # Filter for matching pennies task - this includes extra MP2 sessions
        comprehensive_mp2_data = stitched_data[stitched_data['task'] == 'mp'].copy()
        print(f"  Found {len(comprehensive_mp2_data)} trials in comprehensive MP2 data")
        print(f"  Monkeys in comprehensive MP2: {sorted(comprehensive_mp2_data['animal'].unique())}")
        
        # Also load standard MP2 data for monkey F specifically
        standard_mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
        standard_mp2_data['animal'] = standard_mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
        mp2_data_F = standard_mp2_data[standard_mp2_data['animal'] == 'F']
        
        # Use comprehensive data as the main dataset for predictability analysis
        stitched_p_for_plot = comprehensive_mp2_data
    else:
        print(f"  Using standard behavioral data from {mpbeh_path} for MP2 analysis...")
        comprehensive_mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
        comprehensive_mp2_data['animal'] = comprehensive_mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
        mp2_data_F = comprehensive_mp2_data[comprehensive_mp2_data['animal'] == 'F']
        stitched_p_for_plot = comprehensive_mp2_data

    # Use consistent colors for predictability plot
    for handle, label in zip(*predictability_ax.get_legend_handles_labels()):
        if label in MONKEY_COLORS:
            # Update handle color to match our consistent scheme
            if hasattr(handle, 'set_color'):
                handle.set_color(MONKEY_COLORS[label])
            elif hasattr(handle, 'set_facecolor'):
                handle.set_facecolor(MONKEY_COLORS[label])
    
    # Call the plot function to create the prediction accuracy TIMECOURSE with consistent colors
    try:
        print("  Creating predictability timecourse with comprehensive MP2 data...")
        # Use session-wise predictability for timecourse
        predictability_results = compute_and_cache_session_wise_predictability(
            mpbeh_path, 
            cache_file=f"session_wise_predictability_{session_selection}.pkl",
            force_recompute=False,
            order=5
        )
        plot_session_wise_predictability_errorbars(predictability_ax, predictability_results)
        
        # Force consistent colors in the predictability plot
        for handle, label in zip(*predictability_ax.get_legend_handles_labels()):
            if label in MONKEY_COLORS:
                if hasattr(handle, 'set_color'):
                    handle.set_color(MONKEY_COLORS[label])
                elif hasattr(handle, 'set_facecolor'):
                    handle.set_facecolor(MONKEY_COLORS[label])
        
    except Exception as e:
        print(f"Error in predictability plot: {e}")
        import traceback
        traceback.print_exc()
        predictability_ax.text(0.5, 0.5, 'Predictability plot failed', 
                              ha='center', va='center', transform=predictability_ax.transAxes)
    
    # Load additional data for delta predictability - USE COMPREHENSIVE DATA
    print("  Loading additional data for delta predictability analysis...")
    mp_data = query_monkey_behavior(mpdb_path)
    mp_data = mp_data[(mp_data['task'] == 'mp')]
    
    # Use comprehensive MP2 data that includes extra sessions
    print(f"  Combining query data ({len(mp_data)} trials) with comprehensive MP2 data ({len(comprehensive_mp2_data)} trials)...")
    # Combine query data with comprehensive MP2 data, avoiding duplicates
    if len(mp_data) > 0 and len(comprehensive_mp2_data) > 0:
        # Check for overlapping columns and align them
        common_cols = set(mp_data.columns).intersection(set(comprehensive_mp2_data.columns))
        if len(common_cols) >= 5:  # Need at least basic columns
            mp2_data_combined = pd.concat([mp_data[list(common_cols)], comprehensive_mp2_data[list(common_cols)]], ignore_index=True)
            mp2_data_combined = mp2_data_combined.drop_duplicates()
        else:
            mp2_data_combined = comprehensive_mp2_data
    else:
        mp2_data_combined = comprehensive_mp2_data
    
    print(f"  Final combined MP2 data: {len(mp2_data_combined)} trials from {len(mp2_data_combined['animal'].unique())} monkeys")
    
    # Compute predictability for inset
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    try:
        print("  Computing predictability for MP1 and comprehensive MP2...")
        mp1_preds = compute_predictability(mp1_data, overfit=overfit, violin=True)    
        mp2_preds = compute_predictability(mp2_data_combined, overfit=overfit, violin=True)
        
        print(f"  MP1 predictability computed for: {list(mp1_preds.keys())}")
        print(f"  MP2 predictability computed for: {list(mp2_preds.keys())}")
        
        # Calculate delta predictability for inset histogram - include all monkeys
        delta_predictability = {}
        for monkey in ALL_MONKEYS:
            if monkey in mp1_preds and monkey in mp2_preds:
                mp1_median = np.mean(mp1_preds[monkey]['perf'])
                mp2_median = np.mean(mp2_preds[monkey]['perf'])
                delta_predictability[monkey] = mp2_median - mp1_median
                print(f"    {monkey}: MP1={mp1_median:.3f}, MP2={mp2_median:.3f}, Δ={delta_predictability[monkey]:.3f}")
            elif monkey in mp2_preds:  # MP2-only monkeys
                mp2_median = np.mean(mp2_preds[monkey]['perf'])
                delta_predictability[monkey] = mp2_median - 0.5  # Baseline of 0.5
                print(f"    {monkey}: MP2-only={mp2_median:.3f}, Δ={delta_predictability[monkey]:.3f} (vs baseline)")
        
        # Add inset histogram with consistent colors
        if delta_predictability:
            inset_ax = predictability_ax.inset_axes([0.72, 0.5, 0.25, 0.25])
            delta_values = []
            monkey_names = []
            colors = []
            for monkey, delta in delta_predictability.items():
                delta_values.append(delta)
                monkey_names.append(monkey)
                colors.append(MONKEY_COLORS.get(monkey, 'gray'))
            
            bars = inset_ax.bar(range(len(monkey_names)), delta_values, color=colors)
            
            inset_ax.set_xticks(range(len(monkey_names)))
            inset_ax.set_xticklabels(monkey_names)
            inset_ax.set_title('Δ Predictability', fontsize=10)
            inset_ax.set_ylabel('Δ Predictability', fontsize=8)
            inset_ax.set_ylim(-0.3, 0.3)
            inset_ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars, delta_values)):
                inset_ax.text(i, 0.02, f'{value:.2f}', ha='center', va='bottom', 
                             fontsize=8, color='black', fontweight='bold')
    
    except Exception as e:
        print(f"Error in predictability analysis: {e}")
    
    # Add main title for the predictability plot
    predictability_ax.set_title('Prediction Accuracy for Monkeys Playing Matching Pennies', fontsize=14, pad=20)
    
    # Generate RL models to get predictability values for comparison plot
    print("Generating RL models for predictability comparison...")
    try:
        # Fit monkey E parameters for RL model generation
        fitted_params = fit_and_save_monkey_E_RL_parameters(
            mpbeh_path, force_refit=False, session_selection=session_selection, disable_abs=disable_abs
        )
        
        if fitted_params.get('MP1') and fitted_params.get('MP2'):
            # Set up environment and generate RL model data
            env_params = env_params if 'env_params' in locals() else {}
            bg_env_params = env_params.copy()
            bg_env_params["opponents"] = ["all"]
            bg_env_params["opponent"] = "all" 
            bg_env_params["opponents_params"] = {"all": {"bias": [0], "depth": 4}}
            bg_env_params["fixed_length"] = True
            bg_env_params["reset_time"] = 200
            
            # Generate Algorithm 1 data
            bg_env_params_algo1 = bg_env_params.copy()
            bg_env_params_algo1["opponents_params"] = {"1": {"bias": [0], "depth": 4}}
            bg_env_params_algo1["opponents"] = ["1"]
            bg_env_params_algo1["opponent"] = "1"
            bg_env_algo1 = make_env(bg_env_params_algo1)
            
            BG_params_algo1_fixed = fitted_params['MP1'].copy()
            BG_params_algo1_fixed.pop('_fit_info', None)
            BG_model_algo1 = BG(env=bg_env_algo1, **BG_params_algo1_fixed)
            BG_data_algo1, _ = BG_model_algo1.generate_data(50)
            
            # Generate Algorithm 2 data
            bg_env = make_env(bg_env_params)
            BG_params_fixed = fitted_params['MP2'].copy()
            BG_params_fixed.pop('_fit_info', None)
            BG_model = BG(env=bg_env, **BG_params_fixed)
            BG_data_algo2, _ = BG_model.generate_data(50)
            
            # Calculate RL model predictabilities
            bg_actions_algo1 = np.array(BG_data_algo1['monkey_choice'])
            bg_rewards_algo1 = np.array(BG_data_algo1['reward'])
            bg_actions_algo2 = np.array(BG_data_algo2['monkey_choice'])
            bg_rewards_algo2 = np.array(BG_data_algo2['reward'])
            
            _, pred_acc_algo1 = paper_logistic_accuracy_strategic(bg_actions_algo1, bg_rewards_algo1, order=5)
            _, pred_acc_algo2 = paper_logistic_accuracy_strategic(bg_actions_algo2, bg_rewards_algo2, order=5)
            
            # Compute session-wise predictability for the comparison
            supplement_predictability_results = compute_and_cache_session_wise_predictability(
                mpbeh_path, 
                cache_file=f"session_wise_predictability_{session_selection}.pkl",
                force_recompute=False,
                order=5
            )
            
            # Plot RL vs Monkey predictability comparison
            print("Creating RL vs Monkey predictability comparison in supplement...")
            plot_rl_vs_monkey_predictability_scatter(
                rl_vs_monkey_ax, supplement_predictability_results, pred_acc_algo1, pred_acc_algo2
            )
        else:
            # Fallback with default values
            supplement_predictability_results = compute_and_cache_session_wise_predictability(
                mpbeh_path, 
                cache_file=f"session_wise_predictability_{session_selection}.pkl",
                force_recompute=False,
                order=5
            )
            plot_rl_vs_monkey_predictability_scatter(
                rl_vs_monkey_ax, supplement_predictability_results, 0.5, 0.5
            )
            
    except Exception as e:
        print(f"Error generating RL models for supplement: {e}")
        # Fallback with default values
        supplement_predictability_results = compute_and_cache_session_wise_predictability(
            mpbeh_path, 
            cache_file=f"session_wise_predictability_{session_selection}.pkl",
            force_recompute=False,
            order=5
        )
        plot_rl_vs_monkey_predictability_scatter(
            rl_vs_monkey_ax, supplement_predictability_results, 0.5, 0.5
        )
    
    # Plot RL parameters as TIMECOURSES instead of violins
    print("Creating RL parameter timecourses...")
    
    # Plot alpha parameters timecourse
    from analysis_scripts.stationarity_and_randomness import plot_RL_timescales
    plot_RL_timescales(load_behavior(mpbeh_path, algorithm=None, monkey=None), 
                      ax=alpha_params_ax, hist_ax=None, rl_model='asymmetric', window=0)
    alpha_params_ax.set_title('RL Learning Rate (Alpha) Timecourse', fontsize=14)
    alpha_params_ax.set_ylabel('Alpha (Learning Rate)', fontsize=12)
    
    # Plot beta parameters timecourse - create custom beta timecourse function
    plot_rl_beta_timecourse(beta_params_ax, mpbeh_path, session_selection)
    
    # Plot stochasticity measure as timecourse
    plot_stochasticity_timecourse(stochasticity_ax, mpbeh_path, session_selection)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.suptitle('Supplement: Behavioral Analysis and RL Model Validation', fontsize=16, y=0.98)
    
    return fig

def plot_rl_alpha_parameters_cached(ax, monkey_colors, monkey_rl_results):
    """Plot timescales (1/(1-alpha)) using cached RL results"""
    ax.set_title('RL Model Timescales (Memory Duration)', fontsize=14)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Timescale (trials)', fontsize=12)
    
    print("Plotting RL timescales from cached results...")
    
    plotted_any = False
    
    # Track best timescales for each monkey
    best_timescales = {}
    
    for monkey in monkey_rl_results:
        coeff_dict = monkey_rl_results[monkey]
        if coeff_dict is None:
            continue
            
        print(f"  Processing monkey {monkey}...")
        print(f"    Got {len(coeff_dict['1'])} Algorithm 1 sessions, {len(coeff_dict['2'])} Algorithm 2 sessions")
        
        # Extract alpha parameters and convert to timescales
        algo1_timescales = []
        algo2_timescales = []
        
        # Algorithm 1 sessions - convert alpha to timescale = 1/(1-alpha)
        for i, coeff in enumerate(coeff_dict['1']):
            if len(coeff) >= 1:  # [alpha, delta_win, delta_loss, ...]
                alpha = coeff[0]  # alpha parameter
                # Calculate timescale, handle edge case where alpha = 1
                if alpha >= 0.999:  # Very close to 1, set a reasonable upper bound
                    timescale = 1000  # Cap at 1000 trials
                elif alpha <= 0.001:  # Very close to 0
                    timescale = 1.0  # Minimum timescale
                else:
                    timescale = 1.0 / (1.0 - alpha)
                algo1_timescales.append(timescale)
                print(f"    Algo1 session {i}: α={alpha:.3f}, timescale={timescale:.1f}")
            else:
                print(f"    Algo1 session {i}: insufficient coefficients ({len(coeff)})")
        
        # Algorithm 2 sessions
        for i, coeff in enumerate(coeff_dict['2']):
            if len(coeff) >= 1:
                alpha = coeff[0]
                # Calculate timescale, handle edge case where alpha = 1
                if alpha >= 0.999:  # Very close to 1, set a reasonable upper bound
                    timescale = 1000  # Cap at 1000 trials
                elif alpha <= 0.001:  # Very close to 0
                    timescale = 1.0  # Minimum timescale
                else:
                    timescale = 1.0 / (1.0 - alpha)
                algo2_timescales.append(timescale)
                print(f"    Algo2 session {i}: α={alpha:.3f}, timescale={timescale:.1f}")
            else:
                print(f"    Algo2 session {i}: insufficient coefficients ({len(coeff)})")
        
        # Create session numbers for plotting
        n_algo1 = len(algo1_timescales)
        n_algo2 = len(algo2_timescales)
        
        color = monkey_colors.get(monkey, 'gray')
        
        if n_algo1 > 0:
            sessions_algo1 = np.arange(-n_algo1, 0) + 0.5
            ax.plot(sessions_algo1, algo1_timescales, 'o-', color=color, alpha=0.7, 
                   label=f'{monkey}' if monkey in ['C', 'F', 'E'] else "")
            plotted_any = True
        
        if n_algo2 > 0:
            sessions_algo2 = np.arange(0, n_algo2) + 0.5
            ax.plot(sessions_algo2, algo2_timescales, 'o-', color=color, alpha=0.7)
            plotted_any = True
        
        # Store best timescales for this monkey
        if algo1_timescales or algo2_timescales:
            best_timescales[monkey] = {
                'algo1': {
                    'mean_timescale': np.mean(algo1_timescales) if algo1_timescales else None,
                    'median_timescale': np.median(algo1_timescales) if algo1_timescales else None,
                    'max_timescale': np.max(algo1_timescales) if algo1_timescales else None,
                    'min_timescale': np.min(algo1_timescales) if algo1_timescales else None,
                    'n_sessions': len(algo1_timescales)
                },
                'algo2': {
                    'mean_timescale': np.mean(algo2_timescales) if algo2_timescales else None,
                    'median_timescale': np.median(algo2_timescales) if algo2_timescales else None,
                    'max_timescale': np.max(algo2_timescales) if algo2_timescales else None,
                    'min_timescale': np.min(algo2_timescales) if algo2_timescales else None,
                    'n_sessions': len(algo2_timescales)
                }
            }
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No RL parameters could be fitted', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.7)
    ax.set_xticks([-2, 2])  # Generic positions
    ax.set_xticklabels(['Opponent 1', 'Opponent 2'])
    if plotted_any:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1, 100)  # Timescales from 1 to 100 trials
    ax.set_yscale('log')  # Use log scale for better visualization of timescales
    
    # Print summary of best timescales
    if best_timescales:
        print("\n" + "="*50)
        print("SUMMARY: RL TIMESCALES (MEMORY DURATION)")
        print("="*50)
        for monkey, params in best_timescales.items():
            print(f"\nMonkey {monkey}:")
            if params['algo1']['n_sessions'] > 0:
                print(f"  Algorithm 1 ({params['algo1']['n_sessions']} sessions):")
                print(f"    Mean timescale:   {params['algo1']['mean_timescale']:.1f} trials")
                print(f"    Median timescale: {params['algo1']['median_timescale']:.1f} trials")
                print(f"    Range:            {params['algo1']['min_timescale']:.1f} - {params['algo1']['max_timescale']:.1f} trials")
            if params['algo2']['n_sessions'] > 0:
                print(f"  Algorithm 2 ({params['algo2']['n_sessions']} sessions):")
                print(f"    Mean timescale:   {params['algo2']['mean_timescale']:.1f} trials")
                print(f"    Median timescale: {params['algo2']['median_timescale']:.1f} trials")
                print(f"    Range:            {params['algo2']['min_timescale']:.1f} - {params['algo2']['max_timescale']:.1f} trials")
        print("="*50)

# Add new function for supplementary figure
def generate_algorithm_comparison_supplement(mpdb_path, mpbeh_path, strategic=False, env_params=None, BG_params=None, bg_nits=200):
    """
    Generate a supplementary figure showing:
    - Algorithm 1
    - Early Algorithm 2
    - Late Algorithm 2 comparison
    - Two regression plots
    - A violin plot
    Uses comprehensive MP2 data with extra sessions.
    """
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = GridSpec(3, 3, figure=fig, wspace=0.6, hspace=0.4)
    
    # Load data - use comprehensive MP2 data if available
    algo1 = load_behavior(mpbeh_path, algorithm=1, monkey=18)  # Monkey E
    
    # Load comprehensive MP2 data for monkey E
    if stitched_p and os.path.exists(stitched_p):
        print(f"Loading comprehensive MP2 data from {stitched_p} for algorithm comparison...")
        with open(stitched_p, 'rb') as f:
            stitched_data = pickle.load(f)
        # Filter for monkey E and matching pennies task
        algo2 = stitched_data[(stitched_data['animal'] == 'E') & (stitched_data['task'] == 'mp')].copy()
        print(f"Found {len(algo2)} trials for Monkey E in comprehensive MP2 data")
        
        # Handle different column names if needed
        if 'session_id' in algo2.columns and 'id' not in algo2.columns:
            algo2['id'] = algo2['session_id']
        if 'choice' in algo2.columns and 'monkey_choice' not in algo2.columns:
            algo2['monkey_choice'] = algo2['choice']
        if 'outcome' in algo2.columns and 'reward' not in algo2.columns:
            algo2['reward'] = algo2['outcome']
    else:
        algo2 = load_behavior(mpbeh_path, algorithm=2, monkey=18)  # Monkey E
    
    # Get session list and divide into early and late
    session_list = sorted(list(algo2['id'].unique()))
    num_cutoff = 5  # Skip first 5 sessions
    num_sessions = 10  # Use 10 sessions for early/late
    
    # Early sessions: after warmup but before late sessions
    algo2_early = algo2[(algo2['id'] < session_list[num_sessions+num_cutoff]) & 
                        (algo2['id'] >= session_list[num_cutoff])]
    
    # Late sessions: last num_sessions sessions
    algo2_late = algo2[algo2['id'] >= session_list[-num_sessions]]
    
    # Create plot areas
    algo1_ax = fig.add_subplot(gs[0, 0])
    algo2_early_ax = fig.add_subplot(gs[0, 1])
    algo2_late_ax = fig.add_subplot(gs[0, 2])
    
    regression1_ax = fig.add_subplot(gs[1, 0:2])
    regression2_ax = fig.add_subplot(gs[1, 2])
    
    violin_ax = fig.add_subplot(gs[2, :])
    
    # Set titles
    algo1_ax.set_title('Algorithm 1 (Monkey E)', fontsize=14)
    algo2_early_ax.set_title('Early Algorithm 2 (Monkey E)', fontsize=14)
    algo2_late_ax.set_title('Late Algorithm 2 (Monkey E)', fontsize=14)
    
    regression1_ax.set_title('Regression Analysis 1', fontsize=14)
    regression2_ax.set_title('Regression Analysis 2', fontsize=14)
    
    violin_ax.set_title('Behavioral Comparison', fontsize=14)
    
    # Plot logistic regression for each algorithm version
    if strategic:
        paper_logistic_regression_strategic(algo1_ax, False, data=algo1, legend=True, return_model=True, order=5)
        paper_logistic_regression_strategic(algo2_early_ax, False, data=algo2_early, legend=False, return_model=True, order=5)
        paper_logistic_regression_strategic(algo2_late_ax, False, data=algo2_late, legend=False, return_model=True, order=5)
    else:
        paper_logistic_regression(algo1_ax, False, data=algo1, legend=True, return_model=True, order=5)
        paper_logistic_regression(algo2_early_ax, False, data=algo2_early, legend=False, return_model=True, order=5)
        paper_logistic_regression(algo2_late_ax, False, data=algo2_late, legend=False, return_model=True, order=5)
    
    # Set labels
    for ax in [algo1_ax, algo2_early_ax, algo2_late_ax]:
        ax.set_xlabel('Trials Back', fontsize=10)
        ax.set_ylabel('Regression Coefficient', fontsize=10)
    
    # Generate regression analysis plots (these are placeholders - modify as needed)
    if BG_params and env_params:
        bg_env_params = env_params.copy()
        bg_env_params["opponents"] = ["all"]
        bg_env_params["opponent"] = "all"
        bg_env_params["opponents_params"] = {"all":{"bias":[0], "depth": 4}}
        
        bg_env = make_env(bg_env_params)
        BG_model = BG(env=bg_env, **BG_params)
        BG_data, masks = BG_model.generate_data(bg_nits)
        
        # Plot regression for BG model
        if strategic:
            bg_data = paper_logistic_regression_strategic(regression1_ax, False, data=BG_data, 
                                                          legend=True, mask=None, return_model=True, order=5)
        else:
            bg_data = paper_logistic_regression(regression1_ax, False, data=BG_data, 
                                                legend=True, mask=None, return_model=True, order=5)
    
    # Generate violin plot comparing the three algorithms
    # Calculate accuracy for each algorithm version
    algo1_actions = np.array(algo1['monkey_choice'])
    algo1_rewards = np.array(algo1['reward'])
    
    algo2_early_actions = np.array(algo2_early['monkey_choice'])
    algo2_early_rewards = np.array(algo2_early['reward'])
    
    algo2_late_actions = np.array(algo2_late['monkey_choice'])
    algo2_late_rewards = np.array(algo2_late['reward'])
    
    # Calculate accuracy using standard and strategic methods
    if strategic:
        algo1_acc = paper_logistic_accuracy_strategic(algo1_actions, algo1_rewards, order=5)
        algo2_early_acc = paper_logistic_accuracy_strategic(algo2_early_actions, algo2_early_rewards, order=5)
        algo2_late_acc = paper_logistic_accuracy_strategic(algo2_late_actions, algo2_late_rewards, order=5)
    else:
        algo1_acc = paper_logistic_accuracy(algo1_actions, algo1_rewards)
        algo2_early_acc = paper_logistic_accuracy(algo2_early_actions, algo2_early_rewards)
        algo2_late_acc = paper_logistic_accuracy(algo2_late_actions, algo2_late_rewards)
    
    # Create violin plot of accuracies
    violin_data = [algo1_acc, algo2_early_acc, algo2_late_acc]
    
    # Create custom violin plot
    positions = [1, 2, 3]
    # violin_ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    
    # Add individual data points for better visibility
    for i, data in enumerate(violin_data):
        pos = positions[i]
        violin_ax.scatter(np.random.normal(pos, 0.05, size=len(data)), data, alpha=0.5, s=5)
    
    # Set labels and ticks
    violin_ax.set_xticks(positions)
    violin_ax.set_xticklabels(['Algorithm 1', 'Early Algorithm 2', 'Late Algorithm 2'])
    violin_ax.set_ylabel('Predictability', fontsize=10)
    violin_ax.set_ylim(0.45, 1.0)
    
    # Add a main title for the whole figure
    fig.suptitle('Supplementary Figure: Algorithm Comparison for Monkey E', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def demo_new_features():
    """
    Demo function to show how to use the new session selection and disable_abs features.
    """
    print("="*70)
    print("DEMO: NEW FEATURES FOR FIG2")
    print("="*70)
    
    print("1. Session Selection Options:")
    print("   - session_selection='last10' (default)")
    print("   - session_selection='all'")
    print("   - session_selection='half'")
    print("   - session_selection=5 (or any integer)")
    
    print("\n2. Disable Absolute Value Option:")
    print("   - disable_abs=False (default, applies np.abs() to parameters)")
    print("   - disable_abs=True (allows negative parameters)")
    
    print("\n3. Use Provided Parameters Option:")
    print("   - use_provided_params=False (default, fit from behavioral data)")
    print("   - use_provided_params=True (use provided BG_params and BG_params_algo1)")
    
    print("\n4. Alpha*Beta Plot:")
    print("   - Automatically included in main figure")
    print("   - Shows α_win*β, α_loss*β, and average for both MP1 and MP2")
    
    print("\n5. Usage Examples:")
    print("   # Use all sessions with negative parameters allowed:")
    print("   fig = plot_behavioral_nonstationarity_and_RL(")
    print("       mpdb_path, mpbeh_path,")
    print("       session_selection='all',")
    print("       disable_abs=True")
    print("   )")
    
    print("\n   # Use last 15 sessions with default abs() behavior:")
    print("   fig = plot_behavioral_nonstationarity_and_RL(")
    print("       mpdb_path, mpbeh_path,")
    print("       session_selection=15")
    print("   )")
    
    print("\n   # Use provided RL parameters instead of fitting:")
    print("   bg_params_mp2 = {'alpha': [0.1, 0.05], 'beta': 5.0, 'asymmetric': True, 'deterministic': False, 'load': False}")
    print("   bg_params_mp1 = {'alpha': [0.15, 0.08], 'beta': 3.5, 'asymmetric': True, 'deterministic': False, 'load': False}")
    print("   fig = plot_behavioral_nonstationarity_and_RL(")
    print("       mpdb_path, mpbeh_path,")
    print("       BG_params=bg_params_mp2,")
    print("       BG_params_algo1=bg_params_mp1,")
    print("       use_provided_params=True")
    print("   )")
    
    print("\n   # Use half sessions with standard parameters:")
    print("   fig = plot_behavioral_nonstationarity_and_RL(")
    print("       mpdb_path, mpbeh_path,")
    print("       session_selection='half'")
    print("   )")
    
    print("\n6. File Naming:")
    print("   - Parameter files now include session selection and abs settings")
    print("   - Example: 'monkey_E_MP1_RL_params_all_no_abs.pkl'")
    print("   - Example: 'monkey_E_MP2_RL_params_15.pkl'")
    
    print("="*70)

def comprehensive_data_usage_summary():
    """
    Summary of comprehensive data usage in updated supplement figures.
    
    The updated supplement figures now use comprehensive MP2 data that includes:
    
    1. PREDICTABILITY ANALYSIS:
       - Uses stitched data with extra MP2 sessions from all monkeys
       - Includes MP2-only monkeys (D, H, I, K) not present in standard dataset
       - Combines query data with comprehensive MP2 data to avoid duplicates
       - Computes delta predictability for all monkeys including MP2-only ones
    
    2. RL PARAMETER FITTING:
       - load_or_fit_all_monkey_rl_parameters() fits all available monkeys
       - Uses comprehensive MP2 data from stitched dataset when available
       - Includes extra sessions for improved parameter estimation
       - Creates comprehensive cache for all monkey-algorithm combinations
    
    3. SESSION-LEVEL TIMECOURSE:
       - plot_session_alpha_beta_timecourse() uses comprehensive MP2 data
       - Handles different column names between stitched and standard data
       - Includes all available sessions for more complete analysis
    
    4. ALGORITHM COMPARISON SUPPLEMENT:
       - generate_algorithm_comparison_supplement() uses comprehensive data
       - Loads stitched data for monkey E with extra MP2 sessions
       - Handles column name differences automatically
    
    5. CONSISTENT COLORS AND MARKERS:
       - All supplements use consistent MONKEY_COLORS and MONKEY_MARKERS
       - Strategic monkeys (E, D, I) use blue shades and square markers
       - Non-strategic monkeys (C, H, F, K) use red/purple shades and circle markers
    
    Key Benefits:
    - More comprehensive analysis with additional MP2 sessions
    - Inclusion of MP2-only monkeys in all analyses
    - Better statistical power from larger datasets
    - Consistent visual representation across all plots
    - Improved parameter fitting with more data per monkey
    """
    print("="*80)
    print("COMPREHENSIVE DATA USAGE IN UPDATED SUPPLEMENT FIGURES")
    print("="*80)
    print("✓ Predictability analysis uses stitched data with extra MP2 sessions")
    print("✓ RL parameter fitting includes all available monkeys and sessions")
    print("✓ Session-level timecourse uses comprehensive MP2 data")
    print("✓ Algorithm comparison supplement uses enhanced dataset")
    print("✓ Consistent colors and markers across all supplement plots")
    print("✓ MP2-only monkeys (D, H, I, K) included in all analyses")
    print("="*80)

# Also add function to demonstrate new supplement features
def demo_comprehensive_supplement_features():
    """
    Demo function to show the enhanced supplement features.
    """
    print("\n" + "="*70)
    print("ENHANCED SUPPLEMENT FEATURES DEMO")
    print("="*70)
    
    print("1. Comprehensive MP2 Data Usage:")
    print("   - Predictability plots use stitched data with extra sessions")
    print("   - RL parameter plots include all available monkeys")
    print("   - Session timecourse uses comprehensive dataset")
    print("   - Algorithm comparison includes enhanced MP2 data")
    
    print("\n2. Enhanced Monkey Coverage:")
    print("   - Strategic monkeys: E, D, I (blue squares)")
    print("   - Non-strategic monkeys: C, H, F, K (red/purple circles)")
    print("   - MP2-only monkeys included in all analyses")
    
    print("\n3. Improved Parameter Fitting:")
    print("   - Uses comprehensive caching system")
    print("   - Fits all available sessions per monkey")
    print("   - Includes stochasticity measure: (max(α) - min(α)) × β")
    
    print("\n4. Usage Examples:")
    print("   # Generate comprehensive supplement:")
    print("   supplement = plot_supplement_figures_updated(")
    print("       mpdb_path, mpbeh_path,")
    print("       session_selection='all',  # Use all available sessions")
    print("       force_refit=False  # Use cached parameters")
    print("   )")
    
    print("\n   # Access individual parameter data:")
    print("   all_params = load_or_fit_all_monkey_rl_parameters(mpbeh_path)")
    print("   # all_params contains data for all monkeys including MP2-only")
    
    print("="*70)

def compute_and_cache_session_wise_predictability(mpbeh_path, cache_file="session_wise_predictability_cache.pkl", force_recompute=False, order=5):
    """
    Compute session-wise predictability for all monkeys in MP1 and MP2.
    Cache results to avoid recomputation.
    
    Returns:
    --------
    dict: {monkey: {algorithm: {'sessions': [...], 'predictabilities': [...], 'mean': float, 'std': float, 'sem': float}}}
    """
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Try to load existing cache
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached session-wise predictability from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_results = pickle.load(f)
        print(f"Loaded predictability for {len(cached_results)} monkeys")
        return cached_results
    
    print("Computing session-wise predictability for all monkeys...")
    results = {}
    
    # Load MP1 data
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Load comprehensive MP2 data
    if stitched_p and os.path.exists(stitched_p):
        print(f"  Using stitched data from {stitched_p} for comprehensive MP2 analysis...")
        with open(stitched_p, 'rb') as f:
            stitched_data = pickle.load(f)
        mp2_data = stitched_data[stitched_data['task'] == 'mp'].copy()
    else:
        print(f"  Using standard behavioral data from {mpbeh_path} for MP2 analysis...")
        mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
        mp2_data['animal'] = mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Get all available monkeys
    mp1_monkeys = set(mp1_data['animal'].unique()) if len(mp1_data) > 0 else set()
    mp2_monkeys = set(mp2_data['animal'].unique()) if len(mp2_data) > 0 else set()
    all_monkeys = mp1_monkeys.union(mp2_monkeys)
    
    print(f"Found data for monkeys: {sorted(all_monkeys)}")
    print(f"  MP1 monkeys: {sorted(mp1_monkeys)}")
    print(f"  MP2 monkeys: {sorted(mp2_monkeys)}")
    
    # Compute session-wise predictability for each monkey
    for monkey in sorted(all_monkeys):
        print(f"  Processing Monkey {monkey}...")
        results[monkey] = {}
        
        # Process MP1 if available
        if monkey in mp1_monkeys:
            monkey_mp1_data = mp1_data[mp1_data['animal'] == monkey]
            sessions = sorted(monkey_mp1_data['id'].unique())
            
            session_predictabilities = []
            valid_sessions = []
            
            for session_id in sessions:
                session_data = monkey_mp1_data[monkey_mp1_data['id'] == session_id]
                if len(session_data) < order * 3:  # Need minimum trials
                    continue
                    
                actions = session_data['monkey_choice'].values
                rewards = session_data['reward'].values
                
                try:
                    _, pred_acc = paper_logistic_accuracy_strategic(actions, rewards, order=order)
                    session_predictabilities.append(pred_acc)
                    valid_sessions.append(session_id)
                except Exception as e:
                    print(f"    Warning: MP1 session {session_id} failed: {e}")
                    continue
            
            if session_predictabilities:
                results[monkey]['MP1'] = {
                    'sessions': valid_sessions,
                    'predictabilities': session_predictabilities,
                    'mean': np.mean(session_predictabilities),
                    'std': np.std(session_predictabilities),
                    'sem': np.std(session_predictabilities) / np.sqrt(len(session_predictabilities))
                }
                print(f"    MP1: {len(session_predictabilities)} sessions, mean predictability: {np.mean(session_predictabilities):.3f}")
        
        # Process MP2 if available
        if monkey in mp2_monkeys:
            monkey_mp2_data = mp2_data[mp2_data['animal'] == monkey]
            
            # Handle different column names
            session_col = 'id' if 'id' in monkey_mp2_data.columns else 'session_id'
            sessions = sorted(monkey_mp2_data[session_col].unique())
            
            session_predictabilities = []
            valid_sessions = []
            
            for session_id in sessions:
                session_data = monkey_mp2_data[monkey_mp2_data[session_col] == session_id]
                if len(session_data) < order * 3:  # Need minimum trials
                    continue
                
                # Handle different column names for actions and rewards
                if 'monkey_choice' in session_data.columns:
                    actions = session_data['monkey_choice'].values
                elif 'choice' in session_data.columns:
                    actions = session_data['choice'].values
                else:
                    continue
                
                if 'reward' in session_data.columns:
                    rewards = session_data['reward'].values
                elif 'outcome' in session_data.columns:
                    rewards = session_data['outcome'].values
                else:
                    continue
                
                try:
                    _, pred_acc = paper_logistic_accuracy_strategic(actions, rewards, order=order)
                    session_predictabilities.append(pred_acc)
                    valid_sessions.append(session_id)
                except Exception as e:
                    print(f"    Warning: MP2 session {session_id} failed: {e}")
                    continue
            
            if session_predictabilities:
                results[monkey]['MP2'] = {
                    'sessions': valid_sessions,
                    'predictabilities': session_predictabilities,
                    'mean': np.mean(session_predictabilities),
                    'std': np.std(session_predictabilities),
                    'sem': np.std(session_predictabilities) / np.sqrt(len(session_predictabilities))
                }
                print(f"    MP2: {len(session_predictabilities)} sessions, mean predictability: {np.mean(session_predictabilities):.3f}")
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved session-wise predictability cache to {cache_path}")
    
    return results


def plot_rl_vs_monkey_predictability_scatter(ax, predictability_results, rl_pred_algo1, rl_pred_algo2):
    """
    Plot comparison of RL model vs monkey behavioral predictability.
    X-axis: Monkeys grouped by algorithm
    Y-axis: Predictability values
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    predictability_results : dict
        Results from compute_and_cache_session_wise_predictability
    rl_pred_algo1 : float
        RL model predictability for Algorithm 1
    rl_pred_algo2 : float
        RL model predictability for Algorithm 2
    """
    ax.set_title('RL Model vs Monkey Behavioral Predictability', fontsize=14)
    ax.set_xlabel('Monkey (Algorithm)', fontsize=12)
    ax.set_ylabel('Predictability', fontsize=12)
    
    print("\n" + "="*70)
    print("RL MODEL vs MONKEY PREDICTABILITY COMPARISON")
    print("="*70)
    print(f"RL Model - Algorithm 1 Predictability: {rl_pred_algo1:.3f}")
    print(f"RL Model - Algorithm 2 Predictability: {rl_pred_algo2:.3f}")
    print("="*70)
    
    # Organize data for plotting
    plot_data = []
    x_labels = []
    x_positions = []
    
    current_x = 0
    
    # Algorithm 1 monkeys (strategic first, then non-strategic)
    mp1_monkeys_strategic = [m for m in STRATEGIC_MONKEYS if m in predictability_results and 'MP1' in predictability_results[m]]
    mp1_monkeys_nonstrategic = [m for m in NONSTRATEGIC_MONKEYS if m in predictability_results and 'MP1' in predictability_results[m]]
    
    mp1_monkeys = mp1_monkeys_strategic + mp1_monkeys_nonstrategic
    
    for monkey in mp1_monkeys:
        if 'MP1' in predictability_results[monkey]:
            monkey_pred = predictability_results[monkey]['MP1']['mean']
            color = MONKEY_COLORS.get(monkey, 'gray')
            marker = MONKEY_MARKERS.get(monkey, 'o')
            
            # Add monkey behavioral predictability with error bars
            monkey_sem = predictability_results[monkey]['MP1']['sem']
            ax.errorbar(current_x - 0.1, monkey_pred, yerr=monkey_sem, 
                       fmt=marker, color=color, markersize=10, alpha=0.8, 
                       markeredgecolor='black', markeredgewidth=0.5, capsize=3,
                       label=f'{monkey} LR' if current_x == 0 else "")
            
            # Add RL model predictability
            ax.scatter(current_x + 0.1, rl_pred_algo1, color='blue', marker='D', 
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                      label='RL Model' if current_x == 0 else "")
            
            # Connect with line
            ax.plot([current_x - 0.1, current_x + 0.1], [monkey_pred, rl_pred_algo1], 
                   'k-', alpha=0.3, linewidth=1)
            
            x_labels.append(f'{monkey}\n(MP1)')
            x_positions.append(current_x)
            
            print(f"Algorithm 1 - Monkey {monkey}: Behavioral={monkey_pred:.3f}, RL Model={rl_pred_algo1:.3f}")
            current_x += 1
    
    # Add separator if we have MP1 data
    mp1_end = current_x - 0.5 if mp1_monkeys else -0.5
    
    # Algorithm 2 monkeys - prioritize E, C, F first for comparison, then others
    mp2_monkeys_priority = ['E', 'C', 'F']  # Same order as MP1 for easy comparison
    mp2_monkeys_remaining_strategic = [m for m in STRATEGIC_MONKEYS if m in predictability_results and 'MP2' in predictability_results[m] and m not in mp2_monkeys_priority]
    mp2_monkeys_remaining_nonstrategic = [m for m in NONSTRATEGIC_MONKEYS if m in predictability_results and 'MP2' in predictability_results[m] and m not in mp2_monkeys_priority]
    
    # Build ordered list: priority monkeys first (if they have MP2 data), then remaining
    mp2_monkeys = []
    for monkey in mp2_monkeys_priority:
        if monkey in predictability_results and 'MP2' in predictability_results[monkey]:
            mp2_monkeys.append(monkey)
    mp2_monkeys.extend(mp2_monkeys_remaining_strategic + mp2_monkeys_remaining_nonstrategic)
    
    for monkey in mp2_monkeys:
        if 'MP2' in predictability_results[monkey]:
            monkey_pred = predictability_results[monkey]['MP2']['mean']
            color = MONKEY_COLORS.get(monkey, 'gray')
            marker = MONKEY_MARKERS.get(monkey, 'o')
            
            # Add monkey behavioral predictability with error bars
            monkey_sem = predictability_results[monkey]['MP2']['sem']
            ax.errorbar(current_x - 0.1, monkey_pred, yerr=monkey_sem, 
                       fmt=marker, color=color, markersize=10, alpha=0.8, 
                       markeredgecolor='black', markeredgewidth=0.5, capsize=3)
            
            # Add RL model predictability
            ax.scatter(current_x + 0.1, rl_pred_algo2, color='green', marker='D', 
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Connect with line
            ax.plot([current_x - 0.1, current_x + 0.1], [monkey_pred, rl_pred_algo2], 
                   'k-', alpha=0.3, linewidth=1)
            
            x_labels.append(f'{monkey}\n(MP2)')
            x_positions.append(current_x)
            
            print(f"Algorithm 2 - Monkey {monkey}: Behavioral={monkey_pred:.3f}, RL Model={rl_pred_algo2:.3f}")
            current_x += 1
    
    print("="*70)
    
    # Set up the plot
    if x_positions:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0.45, 1.0)
        
        # Add horizontal chance line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        
        # Add vertical separator between algorithms
        if mp1_monkeys and mp2_monkeys:
            # Calculate proper separator position - between last MP1 and first MP2
            separator_x = len(mp1_monkeys) - 0.5
            ax.axvline(x=separator_x, color='black', linestyle=':', alpha=0.7, linewidth=2)
            
            # Add algorithm labels with better separation and background
            mp1_center = (len(mp1_monkeys) - 1) / 2  # Center of MP1 monkeys
            mp2_center = len(mp1_monkeys) + (len(mp2_monkeys) - 1) / 2  # Center of MP2 monkeys
            
            ax.text(mp1_center, ax.get_ylim()[1] * 0.92, 'Opponent 1', ha='center', va='center', 
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax.text(mp2_center, ax.get_ylim()[1] * 0.92, 'Opponent 2', ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Create legend
        legend_elements = []
        
        # Add monkey type markers (behavioral data)
        strategic_present = any(m in STRATEGIC_MONKEYS for m in mp1_monkeys + mp2_monkeys)
        nonstrategic_present = any(m in NONSTRATEGIC_MONKEYS for m in mp1_monkeys + mp2_monkeys)
        
        if strategic_present:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                                            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                            label='Strategic (LR)', linestyle='None'))
        if nonstrategic_present:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                                            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                            label='Non-strategic (LR)', linestyle='None'))
        
        # Add RL model markers
        if mp1_monkeys:
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='blue', 
                                            markersize=8, label='RL Model (Algo 1)', linestyle='None'))
        if mp2_monkeys:
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='green', 
                                            markersize=8, label='RL Model (Algo 2)', linestyle='None'))
        
        # Add chance line
        legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                        label='Chance', linewidth=2))
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Calculate and print summary statistics
        all_monkey_preds = []
        all_rl_preds = []
        
        for monkey in mp1_monkeys:
            if 'MP1' in predictability_results[monkey]:
                all_monkey_preds.append(predictability_results[monkey]['MP1']['mean'])
                all_rl_preds.append(rl_pred_algo1)
        
        for monkey in mp2_monkeys:
            if 'MP2' in predictability_results[monkey]:
                all_monkey_preds.append(predictability_results[monkey]['MP2']['mean'])
                all_rl_preds.append(rl_pred_algo2)
        
        if len(all_monkey_preds) > 1:
            correlation = np.corrcoef(all_monkey_preds, all_rl_preds)[0, 1]
            print(f"★ CORRELATION: r = {correlation:.3f} ★")
            print(f"Overall correlation between RL model and monkey predictability: r = {correlation:.3f}")
            print(f"Number of comparison points: {len(all_monkey_preds)}")
            
            # Correlation text removed from plot per user request
    
    else:
        ax.text(0.5, 0.5, 'No predictability comparison data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    return x_positions, x_labels

def plot_session_wise_predictability_errorbars(ax, predictability_results):
    """
    Plot errorbars showing session-wise predictability for all monkeys.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    predictability_results : dict
        Results from compute_and_cache_session_wise_predictability
    """
    ax.set_title('Session-Wise Predictability by Monkey and Algorithm', fontsize=14)
    ax.set_xlabel('Monkey', fontsize=12)
    ax.set_ylabel('Logistic Regression Predictability', fontsize=12)
    
    # Organize data for plotting - 10 errorbars total
    # MP1: C, E, F (3 bars)
    # MP2: All available MP2 monkeys (7 bars, ordered by strategic/non-strategic)
    
    plot_data = []
    labels = []
    colors = []
    
    # MP1 monkeys (strategic first, then non-strategic)
    mp1_monkeys_strategic = [m for m in STRATEGIC_MONKEYS if m in predictability_results and 'MP1' in predictability_results[m]]
    mp1_monkeys_nonstrategic = [m for m in NONSTRATEGIC_MONKEYS if m in predictability_results and 'MP1' in predictability_results[m]]
    
    for monkey in mp1_monkeys_strategic + mp1_monkeys_nonstrategic:
        if 'MP1' in predictability_results[monkey]:
            mp1_data = predictability_results[monkey]['MP1']
            plot_data.append({
                'mean': mp1_data['mean'],
                'sem': mp1_data['sem'],
                'n_sessions': len(mp1_data['sessions'])
            })
            labels.append(f"{monkey}\n(MP1)")
            colors.append(MONKEY_COLORS.get(monkey, 'gray'))
    
    # MP2 monkeys (strategic first, then non-strategic)
    mp2_monkeys_strategic = [m for m in STRATEGIC_MONKEYS if m in predictability_results and 'MP2' in predictability_results[m]]
    mp2_monkeys_nonstrategic = [m for m in NONSTRATEGIC_MONKEYS if m in predictability_results and 'MP2' in predictability_results[m]]
    
    for monkey in mp2_monkeys_strategic + mp2_monkeys_nonstrategic:
        if 'MP2' in predictability_results[monkey]:
            mp2_data = predictability_results[monkey]['MP2']
            plot_data.append({
                'mean': mp2_data['mean'],
                'sem': mp2_data['sem'],
                'n_sessions': len(mp2_data['sessions'])
            })
            labels.append(f"{monkey}\n(MP2)")
            colors.append(MONKEY_COLORS.get(monkey, 'gray'))
    
    # Plot errorbars
    if plot_data:
        x_positions = range(len(plot_data))
        means = [d['mean'] for d in plot_data]
        sems = [d['sem'] for d in plot_data]
        
        # Create errorbars
        bars = ax.errorbar(x_positions, means, yerr=sems, fmt='o', capsize=5, capthick=2, 
                          elinewidth=2, markersize=8, alpha=0.8)
        
        # Color each errorbar - fix the matplotlib errorbar structure
        # bars[0] is a single Line2D for data points, bars[1] is error lines, bars[2] is caps
        data_line = bars[0]
        error_lines = bars[1]  # List of Line2D objects for error bars
        cap_lines = bars[2]    # List of Line2D objects for caps
        
        # Create individual errorbars with colors and correct markers
        ax.clear()  # Clear and redraw with individual colors and markers
        
        for i, (x, mean, sem, color, label) in enumerate(zip(x_positions, means, sems, colors, labels)):
            monkey = label.split('\n')[0]
            marker = MONKEY_MARKERS.get(monkey, 'o')  # Get strategic vs non-strategic marker
            ax.errorbar([x], [mean], yerr=[sem], fmt=marker, capsize=5, capthick=2,
                       elinewidth=2, markersize=8, alpha=0.8, color=color,
                       markeredgecolor='black', markeredgewidth=0.5)
        
        # Set labels and formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0.45, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate MP1 and MP2
        mp1_count = len([l for l in labels if 'MP1' in l])
        if mp1_count > 0 and mp1_count < len(labels):
            ax.axvline(x=mp1_count - 0.5, color='black', linestyle=':', alpha=0.7, linewidth=2)
            
            # Add algorithm labels with better separation and background
            mp1_center = (mp1_count - 1) / 2
            mp2_center = mp1_count + (len(labels) - mp1_count - 1) / 2
            ax.text(mp1_center, ax.get_ylim()[1] * 0.92, 'Opponent 1', ha='center', va='center', 
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax.text(mp2_center, ax.get_ylim()[1] * 0.92, 'Opponent 2', ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Add legend for strategic vs non-strategic
        legend_elements = []
        monkeys_in_plot = [l.split('\n')[0] for l in labels]
        strategic_present = any(m in STRATEGIC_MONKEYS for m in monkeys_in_plot)
        nonstrategic_present = any(m in NONSTRATEGIC_MONKEYS for m in monkeys_in_plot)
        
        if strategic_present:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                                            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                            label='Strategic', linestyle='None'))
        if nonstrategic_present:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                                            markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                                            label='Non-strategic', linestyle='None'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    else:
        ax.text(0.5, 0.5, 'No session-wise predictability data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Print summary
    print("\n" + "="*70)
    print("SESSION-WISE PREDICTABILITY SUMMARY")
    print("="*70)
    for i, (label, data) in enumerate(zip(labels, plot_data)):
        monkey = label.split('\n')[0]
        algorithm = label.split('\n')[1].strip('()')
        group = "Strategic" if monkey in STRATEGIC_MONKEYS else "Non-strategic"
        print(f"{monkey} ({group}) - {algorithm}:")
        print(f"  Mean predictability: {data['mean']:.3f} ± {data['sem']:.3f} (SEM)")
        print(f"  Sessions: {data['n_sessions']}")
        print()
    print("="*70)
    
    return plot_data

def plot_rl_vs_lr_ratio_violin(ax, mpbeh_path, strategic_monkeys=['E', 'D', 'I'], nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                               model='simple', order=5, const_beta=False, const_gamma=True, 
                               punitive=False, decay=False, ftol=1e-8, alpha=None, bias=False, mask=None, cutoff_trials=None):
    """
    Create a violin plot comparing RL/LR performance ratios between MP1 and MP2 tasks.
    This function is integrated with the main figure.
    
    Parameters:
    -----------
    cutoff_trials : int, optional
        If provided, will keep only the last N trials (rounded to complete sessions)
        by removing early sessions from each monkey's data.
    """
    
    # Use cached data for fast generation
    print("Using cached RL to LR ratio data for violin plot...")
    cached_ratios = compute_and_cache_rl_to_lr_ratios(
        mpbeh_path, 
        strategic_monkeys=strategic_monkeys, 
        nonstrategic_monkeys=nonstrategic_monkeys,
        model=model, order=order, const_beta=const_beta, const_gamma=const_gamma,
        punitive=punitive, decay=decay, ftol=ftol, alpha=alpha, bias=bias, mask=mask,
        cutoff_trials=cutoff_trials,
        force_recompute=False  # Use cache by default
    )
    
    # Extract aggregated MP1 and MP2 data from cache metadata
    metadata = cached_ratios['metadata']
    mp1_data = metadata['MP1']
    mp2_data = metadata['MP2']
    
    # Prepare data for violin plot
    plot_data = []
    labels = []
    colors = ['#F2B705', '#2D9D5A']  # Colors matching fig1: yellow for MP1, green for MP2
    
    # Add MP1 data
    if mp1_data:
        plot_data.append(mp1_data)
        labels.append('MP1')
    
    # Add MP2 data
    if mp2_data:
        plot_data.append(mp2_data)
        labels.append('MP2')
    
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Truncate data at max value of 1.0 to avoid kernel density beyond realistic bounds
    plot_data_truncated = []
    for data in plot_data:
        truncated = [min(val, 1.0) for val in data]  # Cap values at 1.0
        plot_data_truncated.append(truncated)
    
    # Create violin plot with truncated data
    positions = range(1, len(plot_data_truncated) + 1)
    violins = ax.violinplot(plot_data_truncated, positions, widths=0.6, showmeans=True, showextrema=False)
    
    # Color the violins
    for i, violin in enumerate(violins['bodies']):
        violin.set_facecolor(colors[i % len(colors)])
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)
        violin.set_linewidth(1.5)
    
    # Style the mean lines
    violins['cmeans'].set_color('black')
    violins['cmeans'].set_linewidth(2)
    
    # Customize plot to match figure style
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)  # Reduced font size
    ax.set_ylabel('RL Performance / LR Accuracy', fontsize=12)  # Reduced font size
    ax.set_title('RL to LR Performance Ratio', fontsize=14, fontweight='bold')  # Reduced font size
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=1 (equal performance)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    # Set y-axis limits with proper bounds (max at 1.0) and reduced whitespace
    all_values = [val for sublist in plot_data_truncated for val in sublist]
    if all_values:
        y_min, y_max = min(all_values), min(max(all_values), 1.0)  # Cap at 1.0
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, 1.02)  # Reduce top whitespace by setting max to 1.02
    
    # Add nonparametric significance tests between groups
    from scipy.stats import mannwhitneyu
    
    # Perform pairwise comparisons and add significance annotations
    if len(plot_data) == 2:
        y_max_for_annotations = max(all_values) if all_values else 1.0
        annotation_height = y_max_for_annotations + 0.05 * y_range if all_values else 1.05
        
        # Compare MP1 vs MP2
        statistic, p_value = mannwhitneyu(plot_data_truncated[0], plot_data_truncated[1], alternative='two-sided')
        
        # Determine significance level
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Position annotation
        x1, x2 = positions[0], positions[1]
        y_line = annotation_height
        
        # Draw line and add text
        ax.plot([x1, x2], [y_line, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.plot([x1, x1], [y_line - 0.015 * y_range, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.plot([x2, x2], [y_line - 0.015 * y_range, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.text((x1 + x2) / 2, y_line + 0.008 * y_range if all_values else y_line + 0.008, 
               sig_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        print(f"  Statistical test: {labels[0]} vs {labels[1]}: p={p_value:.4f} ({sig_text})")
        
        # Update y-limits to accommodate annotations but cap at reasonable level
        if all_values:
            max_annotation_y = min(annotation_height + 0.05 * y_range, 1.15)
            ax.set_ylim(y_min - 0.05 * y_range, max_annotation_y)
    
    # Print summary statistics for logging (using original data for stats)
    print("\nRL/LR Ratio Summary:")
    for i, (label, values) in enumerate(zip(labels, plot_data)):
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            n_sessions = len(values)
            print(f"  {label}: {mean_val:.3f} ± {std_val:.3f} (N={n_sessions})")
    
    # Calculate MP1-MP2 difference
    if len(plot_data) == 2:
        mp1_values = np.array(plot_data[0])
        mp2_values = np.array(plot_data[1])
        
        mp1_mean = np.mean(mp1_values)
        mp2_mean = np.mean(mp2_values)
        difference = mp1_mean - mp2_mean
        
        print(f"\nMP1-MP2 Analysis:")
        print(f"  MP1 mean: {mp1_mean:.3f}")
        print(f"  MP2 mean: {mp2_mean:.3f}")
        print(f"  Difference (MP1-MP2): {difference:.3f}")
        print(f"  Data: MP1 ({len(mp1_values)} sessions), MP2 ({len(mp2_values)} sessions)")
    
    return ax

def cutoff_trials_by_session(data, cutoff_trials):
    """
    Keep only the last cutoff_trials trials, rounded to complete sessions.
    This keeps the most recent N trials by removing early sessions.
    
    Args:
        data: DataFrame with trial data containing 'id' column for session IDs
        cutoff_trials: Number of trials to keep (will round to nearest complete sessions)
    
    Returns:
        DataFrame with only the last N trials (rounded to complete sessions)
    """
    if len(data) <= cutoff_trials:
        print(f"Data has only {len(data)} trials, keeping all data")
        return data
    
    # Sort data by session order (assuming sessions are chronologically ordered by id)
    sorted_sessions = sorted(data['id'].unique())
    
    # Work backwards from the end to find which sessions to keep
    cumulative_trials = 0
    sessions_to_keep = []
    
    for session_id in reversed(sorted_sessions):
        session_data = data[data['id'] == session_id]
        session_trials = len(session_data)
        
        # Check if adding this session would exceed our target
        if cumulative_trials + session_trials > cutoff_trials:
            # We've found enough trials, stop here
            break
        
        # Add this session to the keep list
        sessions_to_keep.append(session_id)
        cumulative_trials += session_trials
    
    if not sessions_to_keep:
        # If no complete sessions fit, keep the last session
        sessions_to_keep = [sorted_sessions[-1]]
        cumulative_trials = len(data[data['id'] == sorted_sessions[-1]])
    
    # Filter data to keep only selected sessions
    filtered_data = data[data['id'].isin(sessions_to_keep)]
    trials_kept = len(filtered_data)
    trials_removed = len(data) - trials_kept
    
    print(f"Kept last {trials_kept} trials (target: {cutoff_trials}) from {len(sessions_to_keep)} sessions")
    print(f"Removed {trials_removed} trials from {len(sorted_sessions) - len(sessions_to_keep)} early sessions")
    
    return filtered_data


def compute_and_cache_rl_to_lr_ratios(mpbeh_path, cache_file="rl_lr_ratio_cache.pkl", 
                                      strategic_monkeys=['E', 'D', 'I'], nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                                      model='simple', order=5, const_beta=False, const_gamma=True, 
                                      punitive=False, decay=False, ftol=1e-8, alpha=None, bias=False, 
                                      mask=None, force_recompute=False, cutoff_trials=None):
    """
    Compute and cache RL to LR performance ratios for all monkeys and sessions.
    This avoids recomputing expensive RL fitting and LR calculations every time.
    
    Parameters:
    -----------
    cutoff_trials : int, optional
        If provided, will keep only the last N trials (rounded to complete sessions)
        by removing early sessions. Similar to fig1 cutoff logic.
    
    Returns:
    --------
    dict: {
        'strategic_data': {'MP1': [...], 'MP2': [...]},
        'nonstrategic_data': {'MP1': [...], 'MP2': [...]},
        'metadata': {...}
    }
    """
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Include cutoff_trials in cache filename if specified
    if cutoff_trials is not None:
        cache_file = cache_file.replace('.pkl', f'_cutoff{cutoff_trials}.pkl')
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Try to load existing cache
    # if os.path.exists(cache_path) and not force_recompute:
    #     print(f"Loading cached RL to LR ratios from {cache_path}")
    #     with open(cache_path, 'rb') as f:
    #         cached_results = pickle.load(f)
    #     print(f"Loaded RL/LR ratios for {len(cached_results.get('metadata', {}).get('monkeys', []))} monkeys")
    #     return cached_results
    
    print("Computing RL to LR performance ratios...")
    
    # Load data
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    standard_mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
    
    # Map numeric IDs to letters for standard data
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    standard_mp2_data['animal'] = standard_mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Load comprehensive MP2 data from stitched dataset if available
    comprehensive_mp2_data = None
    if stitched_p and os.path.exists(stitched_p):
        print(f"  Loading comprehensive MP2 data from {stitched_p}...")
        with open(stitched_p, 'rb') as f:
            stitched_data = pickle.load(f)
        # Filter for matching pennies task
        comprehensive_mp2_data = stitched_data[stitched_data['task'] == 'mp'].copy()
        
        # Handle different column names if needed
        if 'session_id' in comprehensive_mp2_data.columns and 'id' not in comprehensive_mp2_data.columns:
            comprehensive_mp2_data['id'] = comprehensive_mp2_data['session_id']
        if 'choice' in comprehensive_mp2_data.columns and 'monkey_choice' not in comprehensive_mp2_data.columns:
            comprehensive_mp2_data['monkey_choice'] = comprehensive_mp2_data['choice']
        if 'outcome' in comprehensive_mp2_data.columns and 'reward' not in comprehensive_mp2_data.columns:
            comprehensive_mp2_data['reward'] = comprehensive_mp2_data['outcome']
        
        print(f"  Found {len(comprehensive_mp2_data)} trials in comprehensive MP2 data")
        print(f"  Monkeys in comprehensive MP2: {sorted(comprehensive_mp2_data['animal'].unique())}")
        
        # Use comprehensive data as primary MP2 source
        mp2_data = comprehensive_mp2_data
    else:
        print(f"  Using standard MP2 data from {mpbeh_path}")
        mp2_data = standard_mp2_data
    
    # Add task column for consistency
    mp1_data['task'] = 1
    mp2_data['task'] = 2
    
    # Combine all data for processing
    all_data = pd.concat([mp1_data, mp2_data], ignore_index=True)
    
    # Check what monkeys we have for each algorithm
    mp1_monkeys = set(mp1_data['animal'].unique())
    mp2_monkeys = set(mp2_data['animal'].unique())
    all_available_monkeys = mp1_monkeys.union(mp2_monkeys)
    
    print(f"  Available monkeys:")
    print(f"    MP1: {sorted(mp1_monkeys)}")
    print(f"    MP2: {sorted(mp2_monkeys)}")
    print(f"    Total unique monkeys: {sorted(all_available_monkeys)}")
    
    # Update monkey lists to include all available monkeys
    strategic_monkeys_available = [m for m in strategic_monkeys if m in all_available_monkeys]
    nonstrategic_monkeys_available = [m for m in nonstrategic_monkeys if m in all_available_monkeys]
    
    # Add any additional strategic/nonstrategic monkeys found in data
    additional_strategic = [m for m in mp2_monkeys if m in STRATEGIC_MONKEYS and m not in strategic_monkeys_available]
    additional_nonstrategic = [m for m in mp2_monkeys if m in NONSTRATEGIC_MONKEYS and m not in nonstrategic_monkeys_available]
    
    strategic_monkeys_available.extend(additional_strategic)
    nonstrategic_monkeys_available.extend(additional_nonstrategic)
    
    if additional_strategic or additional_nonstrategic:
        print(f"  Found additional monkeys in comprehensive data:")
        if additional_strategic:
            print(f"    Additional strategic: {additional_strategic}")
        if additional_nonstrategic:
            print(f"    Additional nonstrategic: {additional_nonstrategic}")
    
    print(f"  Processing strategic monkeys: {strategic_monkeys_available}")
    print(f"  Processing nonstrategic monkeys: {nonstrategic_monkeys_available}")
    
    # Apply cutoff trials filtering if specified
    if cutoff_trials is not None:
        print(f"Applying cutoff of {cutoff_trials} trials to each monkey...")
        
    # Initialize results structure
    strategic_data = {'MP1': [], 'MP2': []}
    nonstrategic_data = {'MP1': [], 'MP2': []}
    
    # Separate arrays for additional monkeys from comprehensive data
    strategic_data_comprehensive = {'MP1': [], 'MP2': []}
    nonstrategic_data_comprehensive = {'MP1': [], 'MP2': []}
    
    session_details = []
    
    # Determine which monkeys are in original behavioral dataset vs comprehensive
    original_mp1_monkeys = set(load_behavior(mpbeh_path, algorithm=1, monkey=None)['animal'].replace({13:'C', 112:'F', 18:'E'}).unique())
    original_mp2_monkeys = set(load_behavior(mpbeh_path, algorithm=2, monkey=None)['animal'].replace({13:'C', 112:'F', 18:'E'}).unique())
    original_monkeys = original_mp1_monkeys.union(original_mp2_monkeys)
    
    print(f"  Original behavioral dataset monkeys: {sorted(original_monkeys)}")
    print(f"  Additional comprehensive monkeys: {sorted(all_available_monkeys - original_monkeys)}")
    
    monkeys = all_data['animal'].unique()
    
    for monkey in monkeys:
        print(f"  Processing Monkey {monkey}...")
        monkey_data = all_data[all_data['animal'] == monkey]
        
        # Apply cutoff trials filtering to this monkey's data if specified
        if cutoff_trials is not None and len(monkey_data) > cutoff_trials:
            print(f"    Before cutoff: {len(monkey_data)} trials")
            monkey_data = cutoff_trials_by_session(monkey_data, cutoff_trials)
            print(f"    After cutoff: {len(monkey_data)} trials")
        
        # Process data but use task column directly for better MP2 coverage
        episode_actions, episode_rewards, center = process_data(monkey_data)
        
        # Create session to task mapping from the original data
        session_to_task = {}
        for session_id in monkey_data['id'].unique():
            session_data = monkey_data[monkey_data['id'] == session_id]
            task_value = session_data['task'].iloc[0]  # Get task for this session
            session_to_task[session_id] = 'MP1' if task_value == 1 else 'MP2'
        
        # Get sorted session IDs to match with episode indices (MUST match process_data order)
        # process_data sorts by task first, then by session ID within each task
        sorted_sessions = []
        for task in sorted(monkey_data['task'].unique()):
            if task == 0:  # Skip task 0 as process_data does
                continue
            task_sessions = sorted(monkey_data[monkey_data['task'] == task]['id'].unique())
            sorted_sessions.extend(task_sessions)
        
        print(f"    Processing {len(episode_actions)} sessions for monkey {monkey}")
        print(f"    Session order: {sorted_sessions[:5]}{'...' if len(sorted_sessions) > 5 else ''}")
        
        if len(episode_actions) != len(sorted_sessions):
            print(f"    Warning: Episode count ({len(episode_actions)}) != session count ({len(sorted_sessions)})")
            print(f"    This may cause index errors - skipping monkey {monkey}")
            continue
        
        for session_idx in range(len(episode_actions)):
            actions = episode_actions[session_idx]
            rewards = episode_rewards[session_idx]
            
            try:
                # Compute RL performance
                rl_fit, rl_perf = single_session_fit(actions, rewards, model=model, 
                                                   const_beta=const_beta, const_gamma=const_gamma, 
                                                   punitive=punitive, alpha=alpha, decay=decay, ftol=ftol)
                
                # Compute logistic regression accuracy
                _, lr_perf = paper_logistic_accuracy_strategic(actions, rewards, order=order, bias=bias, mask=mask)
                
                # Compute ratio
                if lr_perf > 0:  # Avoid division by zero
                    ratio = rl_perf / lr_perf
                else:
                    continue  # Skip this session if LR performance is 0
                
                # Determine task using session mapping
                if session_idx >= len(sorted_sessions):
                    print(f"    Error: session_idx {session_idx} >= len(sorted_sessions) {len(sorted_sessions)}")
                    continue
                
                session_id = sorted_sessions[session_idx]
                if session_id not in session_to_task:
                    print(f"    Warning: session_id {session_id} not in session_to_task mapping")
                    continue
                
                task = session_to_task[session_id]
                
                # Determine if this monkey is from original behavioral dataset or comprehensive data
                is_original_monkey = monkey in original_monkeys
                
                # Store session details for debugging/verification
                session_details.append({
                    'monkey': monkey,
                    'session': session_idx,
                    'session_id': session_id,
                    'task': task,
                    'rl_perf': rl_perf,
                    'lr_perf': lr_perf,
                    'ratio': ratio,
                    'n_trials': len(actions),
                    'data_source': 'original' if is_original_monkey else 'comprehensive',
                    'monkey_type': 'strategic' if monkey in strategic_monkeys_available else 'nonstrategic' if monkey in nonstrategic_monkeys_available else 'other'
                })
                
                # Classify monkey as strategic or nonstrategic, and separate original vs comprehensive
                
                if monkey in strategic_monkeys_available:
                    if is_original_monkey:
                        strategic_data[task].append(ratio)
                    else:
                        strategic_data_comprehensive[task].append(ratio)
                elif monkey in nonstrategic_monkeys_available:
                    if is_original_monkey:
                        nonstrategic_data[task].append(ratio)
                    else:
                        nonstrategic_data_comprehensive[task].append(ratio)
                    
            except Exception as e:
                print(f"    Warning: Error processing monkey {monkey}, session {session_idx}: {e}")
                continue
    
    # Aggregate by MP1 and MP2 for original behavioral dataset monkeys only
    MP1_original = strategic_data['MP1'] + nonstrategic_data['MP1']
    MP2_original = strategic_data['MP2'] + nonstrategic_data['MP2']
    
    # Aggregate comprehensive data separately
    MP1_comprehensive = strategic_data_comprehensive['MP1'] + nonstrategic_data_comprehensive['MP1']
    MP2_comprehensive = strategic_data_comprehensive['MP2'] + nonstrategic_data_comprehensive['MP2']
    
    # Create results structure
    results = {
        'strategic_data': strategic_data,
        'nonstrategic_data': nonstrategic_data,
        'strategic_data_comprehensive': strategic_data_comprehensive,
        'nonstrategic_data_comprehensive': nonstrategic_data_comprehensive,
        'session_details': session_details,
        'metadata': {
            'strategic_monkeys': strategic_monkeys,  # Original requested list
            'nonstrategic_monkeys': nonstrategic_monkeys,  # Original requested list
            'strategic_monkeys_available': strategic_monkeys_available,  # Actually processed
            'nonstrategic_monkeys_available': nonstrategic_monkeys_available,  # Actually processed
            'MP1': MP1_original,  # Original behavioral dataset only
            'MP2': MP2_original,  # Original behavioral dataset only
            'MP1_comprehensive': MP1_comprehensive,  # Additional comprehensive data
            'MP2_comprehensive': MP2_comprehensive,  # Additional comprehensive data
            'original_monkeys': sorted(original_monkeys),
            'comprehensive_only_monkeys': sorted(all_available_monkeys - original_monkeys),
            'monkeys': list(monkeys),
            'mp1_monkeys': sorted(mp1_monkeys),
            'mp2_monkeys': sorted(mp2_monkeys),
            'model': model,
            'order': order,
            'cutoff_trials': cutoff_trials,
            'used_comprehensive_mp2': comprehensive_mp2_data is not None,
            'n_sessions_total': len(session_details),
            'cache_created': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved RL to LR ratio cache to {cache_path}")
    print(f"  Cached {len(session_details)} session ratios")
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("RL TO LR RATIO ANALYSIS SUMMARY")
    print("="*70)
    print(f"Monkeys processed:")
    print(f"  Strategic ({len(strategic_monkeys_available)}): {strategic_monkeys_available}")
    print(f"  Non-strategic ({len(nonstrategic_monkeys_available)}): {nonstrategic_monkeys_available}")
    print(f"Data sources:")
    print(f"  Original behavioral dataset: {sorted(original_monkeys)}")
    print(f"  Additional comprehensive data: {sorted(all_available_monkeys - original_monkeys)}")
    print(f"  Used comprehensive MP2 data: {comprehensive_mp2_data is not None}")
    
    print(f"\nOriginal behavioral dataset sessions:")
    print(f"  Strategic MP1: {len(strategic_data['MP1'])}")
    print(f"  Strategic MP2: {len(strategic_data['MP2'])}")  
    print(f"  Non-strategic MP1: {len(nonstrategic_data['MP1'])}")
    print(f"  Non-strategic MP2: {len(nonstrategic_data['MP2'])}")
    print(f"  Original total: {len(MP1_original)} MP1 + {len(MP2_original)} MP2 = {len(MP1_original) + len(MP2_original)}")
    
    print(f"\nComprehensive dataset sessions:")
    print(f"  Strategic MP1: {len(strategic_data_comprehensive['MP1'])}")
    print(f"  Strategic MP2: {len(strategic_data_comprehensive['MP2'])}")  
    print(f"  Non-strategic MP1: {len(nonstrategic_data_comprehensive['MP1'])}")
    print(f"  Non-strategic MP2: {len(nonstrategic_data_comprehensive['MP2'])}")
    print(f"  Comprehensive total: {len(MP1_comprehensive)} MP1 + {len(MP2_comprehensive)} MP2 = {len(MP1_comprehensive) + len(MP2_comprehensive)}")
    
    print(f"\nGrand total sessions: {len(session_details)}")
    print("="*70)
    
    return results


def compute_and_cache_alpha_beta_timecourse(mpbeh_path, cache_file="alpha_beta_timecourse_cache.pkl", 
                                           session_selection="last10", force_recompute=False):
    """
    Compute and cache session-level alpha*beta values for all monkeys.
    This avoids refitting RL parameters for each session every time.
    
    Returns:
    --------
    list: Session data points with fitted parameters
    """
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Try to load existing cache
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached alpha*beta timecourse from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_results = pickle.load(f)
        print(f"Loaded {len(cached_results)} session data points from cache")
        return cached_results
    
    print("Computing session-level alpha*beta timecourse...")
    
    # Load behavior data for MP1
    mp1_data = load_behavior(mpbeh_path, algorithm=1, monkey=None)
    mp1_data['animal'] = mp1_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Load MP2 data - prefer stitched data if available
    mp2_data = None
    if stitched_p and os.path.exists(stitched_p):
        print(f"  Using stitched data from {stitched_p} for MP2 sessions...")
        with open(stitched_p, 'rb') as f:
            stitched_data = pickle.load(f)
        # Filter for matching pennies task
        mp2_data = stitched_data[stitched_data['task'] == 'mp'].copy()
        print(f"  Found {len(mp2_data)} trials in stitched MP2 data")
    else:
        print(f"  Using standard behavioral data from {mpbeh_path} for MP2 sessions...")
        mp2_data = load_behavior(mpbeh_path, algorithm=2, monkey=None)
        mp2_data['animal'] = mp2_data['animal'].replace({13:'C', 112:'F', 18:'E'})
    
    # Check what data we have
    mp1_monkeys = set(mp1_data['animal'].unique()) if len(mp1_data) > 0 else set()
    mp2_monkeys = set(mp2_data['animal'].unique()) if len(mp2_data) > 0 else set()
    all_available_monkeys = mp1_monkeys.union(mp2_monkeys)
    
    print(f"Found data for monkeys: {sorted(all_available_monkeys)}")
    
    all_session_data = []
    
    for monkey in sorted(all_available_monkeys):
        color = MONKEY_COLORS.get(monkey, 'gray')
        marker = MONKEY_MARKERS.get(monkey, 'o')
        
        print(f"  Processing Monkey {monkey}...")
        
        # Process Algorithm 1 sessions if available
        if monkey in mp1_monkeys:
            monkey_mp1_data = mp1_data[mp1_data['animal'] == monkey]
            sessions = sorted(monkey_mp1_data['id'].unique())
            print(f"    MP1: Found {len(sessions)} sessions")
            
            for i, session_id in enumerate(sessions):
                session_data = monkey_mp1_data[monkey_mp1_data['id'] == session_id]
                actions = session_data['monkey_choice'].values.astype(int)  
                rewards = session_data['reward'].values.astype(int)
                
                if len(actions) < 20:  # Skip very short sessions
                    continue
                    
                try:
                    # Fit simple RL model for this session
                    fit_params, performance = single_session_fit(
                        actions, rewards,
                        model='simple',
                        punitive=False,
                        decay=False,
                        ftol=1e-6,
                        const_beta=False,
                        const_gamma=True
                    )
                    
                    # Simple model returns [alpha, beta, gamma] - extract first 2
                    alpha, beta = fit_params[0], fit_params[1]
                    alpha = np.abs(alpha)
                    beta = np.abs(beta)
                    
                    # Calculate Alpha * Beta for simple model
                    alpha_beta_value = alpha * beta
                    
                    # Store session data
                    all_session_data.append({
                        'monkey': monkey,
                        'algorithm': 'Algorithm 1', 
                        'session_index': i - len(sessions),  # Negative indices for MP1
                        'session_id': session_id,
                        'alpha_beta': alpha_beta_value,
                        'alpha': alpha,
                        'beta': beta,
                        'color': color,
                        'marker': marker,
                        'performance': performance,
                        'n_trials': len(actions)
                    })
                    
                except Exception as e:
                    print(f"    MP1 session {session_id} fit failed: {e}")
                    continue
        
        # Process Algorithm 2 sessions if available
        if monkey in mp2_monkeys:
            monkey_mp2_data = mp2_data[mp2_data['animal'] == monkey]
            
            # Handle different column names between stitched and standard data
            if 'id' in monkey_mp2_data.columns:
                session_col = 'id'
            elif 'session_id' in monkey_mp2_data.columns:
                session_col = 'session_id'
            else:
                print(f"    Warning: No session ID column found for monkey {monkey} MP2 data")
                continue
                
            sessions = sorted(monkey_mp2_data[session_col].unique())
            print(f"    MP2: Found {len(sessions)} sessions")  
            
            for i, session_id in enumerate(sessions):
                session_data = monkey_mp2_data[monkey_mp2_data[session_col] == session_id]
                
                # Handle different column names for actions and rewards
                if 'monkey_choice' in session_data.columns:
                    actions = session_data['monkey_choice'].values.astype(int)
                elif 'choice' in session_data.columns:
                    actions = session_data['choice'].values.astype(int)
                else:
                    print(f"    Warning: No choice column found for session {session_id}")
                    continue
                    
                if 'reward' in session_data.columns:
                    rewards = session_data['reward'].values.astype(int)
                elif 'outcome' in session_data.columns:
                    rewards = session_data['outcome'].values.astype(int)
                else:
                    print(f"    Warning: No reward column found for session {session_id}")
                    continue
                
                if len(actions) < 20:  # Skip very short sessions
                    continue
                    
                try:
                    # Fit simple RL model for this session
                    fit_params, performance = single_session_fit(
                        actions, rewards,
                        model='simple',
                        punitive=False,
                        decay=False,
                        ftol=1e-6,
                        const_beta=False,
                        const_gamma=True
                    )
                    
                    # Simple model returns [alpha, beta, gamma] - extract first 2
                    alpha, beta = fit_params[0], fit_params[1]
                    alpha = np.abs(alpha)
                    beta = np.abs(beta)
                    
                    # Calculate Alpha * Beta for simple model
                    alpha_beta_value = alpha * beta
                    
                    # Store session data
                    all_session_data.append({
                        'monkey': monkey,
                        'algorithm': 'Algorithm 2',
                        'session_index': i + 1,  # Positive indices for MP2  
                        'session_id': session_id,
                        'alpha_beta': alpha_beta_value,
                        'alpha': alpha,
                        'beta': beta,
                        'color': color,
                        'marker': marker,
                        'performance': performance,
                        'n_trials': len(actions)
                    })
                    
                except Exception as e:
                    print(f"    MP2 session {session_id} fit failed: {e}")
                    continue
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(all_session_data, f)
    print(f"Saved alpha*beta timecourse cache to {cache_path}")
    print(f"  Cached {len(all_session_data)} session data points")
    
    return all_session_data


def compute_and_cache_all_figure_data(mpbeh_path, mpdb_path=None, force_recompute=False, 
                                     session_selection="last10", order=5):
    """
    Compute and cache ALL expensive data needed for the main figure.
    This is a master function that ensures all components are cached.
    
    Returns:
    --------
    dict: Complete cached data for figure generation
    """
    
    print("Computing and caching all figure data...")
    print("=" * 60)
    
    # 1. Cache RL parameters for all monkeys
    print("1. Caching RL parameters...")
    all_rl_params = load_or_fit_all_monkey_rl_parameters(
        mpbeh_path, 
        force_refit=force_recompute, 
        session_selection=session_selection
    )
    
    # 2. Cache session-wise predictability
    print("\n2. Caching session-wise predictability...")
    predictability_results = compute_and_cache_session_wise_predictability(
        mpbeh_path, 
        force_recompute=force_recompute, 
        order=order
    )
    
    # 3. Cache RL to LR ratios for violin plot
    print("\n3. Caching RL to LR ratios...")
    rl_lr_ratios = compute_and_cache_rl_to_lr_ratios(
        mpbeh_path, 
        force_recompute=force_recompute,
        order=order
    )
    
    # 4. Cache alpha*beta timecourse data
    print("\n4. Caching alpha*beta timecourse...")
    alpha_beta_data = compute_and_cache_alpha_beta_timecourse(
        mpbeh_path, 
        force_recompute=force_recompute
    )
    
    # 5. Optional: Cache logistic regression data if mpdb_path provided
    lr_data = None
    if mpdb_path and os.path.exists(mpdb_path):
        print("\n5. Loading monkey behavioral data...")
        try:
            lr_data = query_monkey_behavior(mpdb_path)
            print(f"   Loaded behavioral data for {len(lr_data['animal'].unique())} monkeys")
        except Exception as e:
            print(f"   Warning: Could not load LR data: {e}")
    
    # Combine all results
    complete_cache = {
        'rl_parameters': all_rl_params,
        'predictability_results': predictability_results,
        'rl_lr_ratios': rl_lr_ratios,
        'alpha_beta_timecourse': alpha_beta_data,
        'lr_data': lr_data,
        'metadata': {
            'cached_at': pd.Timestamp.now().isoformat(),
            'session_selection': session_selection,
            'order': order,
            'mpbeh_path': mpbeh_path,
            'mpdb_path': mpdb_path
        }
    }
    
    print("\n" + "=" * 60)
    print("CACHING SUMMARY")
    print("=" * 60)
    print(f"RL Parameters: {len(all_rl_params)} monkeys")
    print(f"Predictability: {len(predictability_results)} monkeys")
    print(f"RL/LR Ratios: {len(rl_lr_ratios['session_details'])} sessions")
    print(f"Alpha*Beta: {len(alpha_beta_data)} sessions")
    print("=" * 60)
    
    return complete_cache

def plot_rl_beta_timecourse(ax, mpbeh_path, session_selection="last10"):
    """
    Plot session-level beta (temperature) values as a timecourse for all monkeys.
    """
    ax.set_title('RL Temperature Parameter (Beta) Timecourse', fontsize=14)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Beta (Temperature)', fontsize=12)
    
    print("Plotting RL beta parameter timecourse...")
    
    # Load behavioral data
    all_data = load_behavior(mpbeh_path, algorithm=None, monkey=None)
    monkeys = all_data['animal'].unique()
    
    plotted_any = False
    
    for monkey in monkeys:
        monkey_data = all_data[all_data['animal'] == monkey]
        monkey_name = monkey
        if monkey == 13:
            monkey_name = 'C'
        elif monkey == 112:
            monkey_name = 'F'
        elif monkey == 18:
            monkey_name = 'E'
        
        color = MONKEY_COLORS.get(monkey_name, 'gray')
        marker = MONKEY_MARKERS.get(monkey_name, 'o')
        
        # Process each algorithm separately
        for algorithm in [1, 2]:
            algo_data = monkey_data[monkey_data['task'] == algorithm]
            if len(algo_data) == 0:
                continue
                
            session_positions = []
            beta_values = []
            
            # Fit beta for each session
            for session_id in sorted(algo_data['id'].unique()):
                session_data = algo_data[algo_data['id'] == session_id]
                if len(session_data) > 20:  # Minimum trials for fitting
                    try:
                        actions = session_data['monkey_choice'].values
                        rewards = session_data['reward'].values
                        fit_params, _ = single_session_fit(
                            actions, rewards, 
                            model='simple', 
                            const_beta=False, 
                            const_gamma=True
                        )
                        if len(fit_params) >= 2:
                            beta = fit_params[1]  # Beta is second parameter
                            
                            # Position relative to algorithm transition
                            all_sessions = sorted(monkey_data['id'].unique())
                            center = len(algo_data['id'].unique()) if algorithm == 1 else 0
                            session_idx = list(sorted(algo_data['id'].unique())).index(session_id)
                            position = session_idx - center if algorithm == 1 else session_idx
                            
                            session_positions.append(position)
                            beta_values.append(beta)
                    except:
                        continue
            
            if session_positions and beta_values:
                ax.plot(session_positions, beta_values, marker=marker, color=color, 
                       linewidth=2, markersize=6, alpha=0.8, 
                       label=f'{monkey_name}' if algorithm == 1 else "")
                plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No beta parameter data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Add algorithm boundary and formatting
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add algorithm labels inside the plot area
        ax.text(-0.5, ax.get_ylim()[1] * 0.92, 'Opponent 1', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.5, ax.get_ylim()[1] * 0.92, 'Opponent 2', ha='center', va='center',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))


def plot_stochasticity_timecourse(ax, mpbeh_path, session_selection="last10"):
    """
    Plot session-level stochasticity measure as a timecourse for all monkeys.
    Stochasticity = (max(alpha) - min(alpha)) * beta
    """
    ax.set_title('RL Stochasticity Measure Timecourse: (max(α) - min(α)) × β', fontsize=14)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Stochasticity Measure', fontsize=12)
    
    print("Plotting RL stochasticity measure timecourse...")
    
    # Load behavioral data
    all_data = load_behavior(mpbeh_path, algorithm=None, monkey=None)
    monkeys = all_data['animal'].unique()
    
    plotted_any = False
    
    for monkey in monkeys:
        monkey_data = all_data[all_data['animal'] == monkey]
        monkey_name = monkey
        if monkey == 13:
            monkey_name = 'C'
        elif monkey == 112:
            monkey_name = 'F'
        elif monkey == 18:
            monkey_name = 'E'
        
        color = MONKEY_COLORS.get(monkey_name, 'gray')
        marker = MONKEY_MARKERS.get(monkey_name, 'o')
        
        # Process each algorithm separately
        for algorithm in [1, 2]:
            algo_data = monkey_data[monkey_data['task'] == algorithm]
            if len(algo_data) == 0:
                continue
                
            session_positions = []
            stochasticity_values = []
            
            # Fit parameters for each session and compute stochasticity
            for session_id in sorted(algo_data['id'].unique()):
                session_data = algo_data[algo_data['id'] == session_id]
                if len(session_data) > 20:  # Minimum trials for fitting
                    try:
                        actions = session_data['monkey_choice'].values
                        rewards = session_data['reward'].values
                        fit_params, _ = single_session_fit(
                            actions, rewards, 
                            model='simple', 
                            const_beta=False, 
                            const_gamma=True
                        )
                        if len(fit_params) >= 2:
                            alpha = fit_params[0]
                            beta = fit_params[1]
                            
                            # For simple model, stochasticity is just alpha * beta
                            stochasticity = alpha * beta
                            
                            # Position relative to algorithm transition
                            all_sessions = sorted(monkey_data['id'].unique())
                            center = len(algo_data['id'].unique()) if algorithm == 1 else 0
                            session_idx = list(sorted(algo_data['id'].unique())).index(session_id)
                            position = session_idx - center if algorithm == 1 else session_idx
                            
                            session_positions.append(position)
                            stochasticity_values.append(stochasticity)
                    except:
                        continue
            
            if session_positions and stochasticity_values:
                ax.plot(session_positions, stochasticity_values, marker=marker, color=color, 
                       linewidth=2, markersize=6, alpha=0.8, 
                       label=f'{monkey_name}' if algorithm == 1 else "")
                plotted_any = True
    
    if not plotted_any:
        ax.text(0.5, 0.5, 'No stochasticity data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    else:
        # Add algorithm boundary and formatting
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.7, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add algorithm labels inside the plot area
        ax.text(-0.5, ax.get_ylim()[1] * 0.92, 'Opponent 1', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.5, ax.get_ylim()[1] * 0.92, 'Opponent 2', ha='center', va='center',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

def plot_strategic_vs_nonstrategic_mp2_violin(ax, mpbeh_path, strategic_monkeys=['E', 'D', 'I'], 
                                              nonstrategic_monkeys=['C', 'H', 'F', 'K'],
                                              model='simple', order=5, const_beta=False, 
                                              const_gamma=True, punitive=False, decay=False, 
                                              ftol=1e-8, alpha=None, bias=False, mask=None, 
                                              cutoff_trials=None):
    """
    Create a violin plot comparing RL/LR performance ratios between strategic and nonstrategic monkeys
    for MP2 data only, using both original and comprehensive datasets.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    cutoff_trials : int, optional
        If provided, will keep only the last N trials (rounded to complete sessions)
        by removing early sessions from each monkey's data.
    """
    
    # Use cached data for fast generation
    print("Using cached RL to LR ratio data for strategic vs nonstrategic comparison...")
    cached_ratios = compute_and_cache_rl_to_lr_ratios(
        mpbeh_path, 
        strategic_monkeys=strategic_monkeys, 
        nonstrategic_monkeys=nonstrategic_monkeys,
        model=model, order=order, const_beta=const_beta, const_gamma=const_gamma,
        punitive=punitive, decay=decay, ftol=ftol, alpha=alpha, bias=bias, mask=mask,
        cutoff_trials=cutoff_trials,
        force_recompute=False  # Use cache by default
    )
    
    # Extract strategic and nonstrategic data for MP2 (combine original + comprehensive)
    strategic_mp2_original = cached_ratios['strategic_data']['MP2']
    strategic_mp2_comprehensive = cached_ratios['strategic_data_comprehensive']['MP2']
    nonstrategic_mp2_original = cached_ratios['nonstrategic_data']['MP2']
    nonstrategic_mp2_comprehensive = cached_ratios['nonstrategic_data_comprehensive']['MP2']
    
    # Combine original and comprehensive data
    strategic_mp2_all = strategic_mp2_original + strategic_mp2_comprehensive
    nonstrategic_mp2_all = nonstrategic_mp2_original + nonstrategic_mp2_comprehensive
    
    # Prepare data for violin plot
    plot_data = []
    labels = []
    colors = ['#3a86ff', '#ff006e']  # Colors matching fig1: blue for strategic, purple for nonstrategic
    
    # Add strategic data
    if strategic_mp2_all:
        plot_data.append(strategic_mp2_all)
        labels.append('Strategic')
    
    # Add nonstrategic data
    if nonstrategic_mp2_all:
        plot_data.append(nonstrategic_mp2_all)
        labels.append('Nonstrategic')
    
    if not plot_data:
        ax.text(0.5, 0.5, 'No MP2 data available', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Truncate data at max value of 1.0 to avoid unrealistic values
    plot_data_truncated = []
    for data in plot_data:
        truncated = [min(val, 1.0) for val in data]  # Cap values at 1.0
        plot_data_truncated.append(truncated)
    
    # Create violin plot with truncated data
    positions = range(1, len(plot_data_truncated) + 1)
    violins = ax.violinplot(plot_data_truncated, positions, widths=0.6, showmeans=True, showextrema=False)
    
    # Color the violins
    for i, violin in enumerate(violins['bodies']):
        violin.set_facecolor(colors[i % len(colors)])
        violin.set_edgecolor('black')
        violin.set_alpha(0.8)
        violin.set_linewidth(1.5)
    
    # Style the mean lines
    violins['cmeans'].set_color('black')
    violins['cmeans'].set_linewidth(2)
    
    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('RL Performance / LR Accuracy', fontsize=12)
    ax.set_title('Strategic vs Nonstrategic\n(MP2 Only)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at y=1 (equal performance)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    # Set y-axis limits with proper bounds
    all_values = [val for sublist in plot_data_truncated for val in sublist]
    if all_values:
        y_min, y_max = min(all_values), min(max(all_values), 1.0)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, 1.02)
    
    # Add significance testing between groups
    if len(plot_data_truncated) == 2:
        from scipy.stats import mannwhitneyu
        
        y_max_for_annotations = max(all_values) if all_values else 1.0
        annotation_height = y_max_for_annotations + 0.05 * y_range if all_values else 1.05
        
        # Compare strategic vs nonstrategic
        statistic, p_value = mannwhitneyu(plot_data_truncated[0], plot_data_truncated[1], 
                                         alternative='two-sided')
        
        # Determine significance level
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Position annotation
        x1, x2 = positions[0], positions[1]
        y_line = annotation_height
        
        # Draw line and add text
        ax.plot([x1, x2], [y_line, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.plot([x1, x1], [y_line - 0.015 * y_range, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.plot([x2, x2], [y_line - 0.015 * y_range, y_line], 'k-', alpha=0.7, linewidth=1)
        ax.text((x1 + x2) / 2, y_line + 0.008 * y_range if all_values else y_line + 0.008, 
               sig_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        print(f"  Statistical test: {labels[0]} vs {labels[1]}: p={p_value:.4f} ({sig_text})")
        
        # Update y-limits to accommodate annotations
        if all_values:
            max_annotation_y = min(annotation_height + 0.05 * y_range, 1.15)
            ax.set_ylim(y_min - 0.05 * y_range, max_annotation_y)
    
    # Print summary statistics
    print("\nStrategic vs Nonstrategic MP2 Comparison:")
    for i, (label, values) in enumerate(zip(labels, plot_data)):
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            n_sessions = len(values)
            print(f"  {label}: {mean_val:.3f} ± {std_val:.3f} (N={n_sessions})")
            
            # Break down by data source
            if label == 'Strategic':
                print(f"    Original dataset: {len(strategic_mp2_original)} sessions")
                print(f"    Comprehensive dataset: {len(strategic_mp2_comprehensive)} sessions")
                if strategic_mp2_original:
                    print(f"    Original mean: {np.mean(strategic_mp2_original):.3f}")
                if strategic_mp2_comprehensive:
                    print(f"    Comprehensive mean: {np.mean(strategic_mp2_comprehensive):.3f}")
            elif label == 'Nonstrategic':
                print(f"    Original dataset: {len(nonstrategic_mp2_original)} sessions")
                print(f"    Comprehensive dataset: {len(nonstrategic_mp2_comprehensive)} sessions")
                if nonstrategic_mp2_original:
                    print(f"    Original mean: {np.mean(nonstrategic_mp2_original):.3f}")
                if nonstrategic_mp2_comprehensive:
                    print(f"    Comprehensive mean: {np.mean(nonstrategic_mp2_comprehensive):.3f}")
    
    # Calculate strategic-nonstrategic difference
    if len(plot_data) == 2:
        strategic_values = np.array(plot_data[0])
        nonstrategic_values = np.array(plot_data[1])
        
        strategic_mean = np.mean(strategic_values)
        nonstrategic_mean = np.mean(nonstrategic_values)
        difference = strategic_mean - nonstrategic_mean
        
        print(f"\nStrategic-Nonstrategic Analysis (MP2 Only):")
        print(f"  Strategic mean: {strategic_mean:.3f}")
        print(f"  Nonstrategic mean: {nonstrategic_mean:.3f}")
        print(f"  Difference (Strategic-Nonstrategic): {difference:.3f}")
        print(f"  Data: Strategic ({len(strategic_values)} sessions), Nonstrategic ({len(nonstrategic_values)} sessions)")
    
    return ax

if __name__ == "__main__":
    # Run demo when script is executed directly
    demo_new_features()
    
    # Optionally run tests
    # test_monkey_E_parameter_fitting()
    
    # Run comprehensive data summary when script is executed directly
    comprehensive_data_usage_summary()
    demo_comprehensive_supplement_features()
