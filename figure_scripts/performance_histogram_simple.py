"""
Super-simplified performance histogram for fig5_RLRNN_fixed.py
"""

import numpy as np
import pickle
import pandas as pd
import os

def add_performance_histogram(ax, bootstrap_iters=1000, random_state=0, cached=True):
    """
    Plot performance bars for Hybrid, RNN, and RL models.
    """
    # Data paths
    rlrnn_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/RLRNN_500_fits.pkl'
    rnn_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/rnn_sequence_prediction_summary.pkl'
    rl_cache_path = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/rl_group_sequence_performance.pkl'
    monkey_path = '/Users/fmb35/Desktop/BG-PFC-RNN/stitched_monkey_data_safely_cleaned.pkl'
    
    strategic_monkeys = ['E', 'D', 'I']
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']
    
    # --- Helper: Bootstrap SEM ---
    def boot_sem(vals, n=1000, seed=0):
        if not vals: return 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(vals)
        means = [np.mean(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)]
        return np.std(means, ddof=1)
    
    # --- Load Hybrid (RLRNN) performance ---
    with open(rlrnn_path, 'rb') as f:
        param, fits = pickle.load(f)
    param_key = f"{param[0]}, {param[1]:g}"
    
    def get_hybrid_perf(monkeys):
        perfs = []
        for fd in fits.values():
            if fd.get('avg_reward', 0) <= 0.45 or 'sequence_prediction' not in fd:
                continue
            sp = fd['sequence_prediction']
            for m in monkeys:
                if m in sp and sp[m] and param_key in sp[m]:
                    for arrs in sp[m][param_key].values():
                        if arrs:
                            total = sum(np.sum(a) for a in arrs if isinstance(a, np.ndarray))
                            count = sum(len(a) for a in arrs if isinstance(a, np.ndarray))
                            if count > 0:
                                perfs.append(total / count)
        return perfs
    
    strat_hybrid = get_hybrid_perf(strategic_monkeys)
    nonstrat_hybrid = get_hybrid_perf(nonstrategic_monkeys)
    
    # --- Load RNN performance ---
    with open(rnn_path, 'rb') as f:
        rnn = pickle.load(f)
    strat_rnn_mean = rnn.get('strategic_monkeys', {}).get('mean_accuracy', np.nan)
    nonstrat_rnn_mean = rnn.get('non_strategic_monkeys', {}).get('mean_accuracy', np.nan)
    rnn_strat_vals = rnn.get('aggregate', {}).get('strategic', {}).get('model_means', [])
    rnn_nonstrat_vals = rnn.get('aggregate', {}).get('nonstrategic', {}).get('model_means', [])
    
    # --- RL performance (with caching for speed) ---
    rl_cache_valid = False
    if cached and os.path.exists(rl_cache_path):
        with open(rl_cache_path, 'rb') as f:
            rl_cache = pickle.load(f)
        # Check if cache matches params
        if (rl_cache.get('n_bootstrap') == bootstrap_iters and 
            rl_cache.get('random_state') == random_state):
            strat_rl_mean = rl_cache['strategic']['mean_accuracy']
            strat_rl_sem = rl_cache['strategic']['bootstrap_sem']
            nonstrat_rl_mean = rl_cache['nonstrategic']['mean_accuracy']
            nonstrat_rl_sem = rl_cache['nonstrategic']['bootstrap_sem']
            rl_cache_valid = True
            print("Loaded RL performance from cache (fast!)")
    
    if not rl_cache_valid:
        print("Computing RL performance (slow, caching for next time)...")
        from analysis_scripts.LLH_behavior_RL import cross_validated_performance_sessions
        df = pd.read_pickle(monkey_path)
        df = df[df['task'] == 'mp']
        
        def get_rl_perf(monkeys, seed):
            episodes = [(s['monkey_choice'].to_numpy(), s['reward'].to_numpy()) 
                        for _, s in df[df['animal'].isin(monkeys)].groupby('id') if len(s) > 1]
            actions, rewards = zip(*episodes) if episodes else ([], [])
            result = cross_validated_performance_sessions(
                actions, rewards, model='simple', n_folds=10, random_state=seed,
                punitive=False, decay=False, const_beta=False, const_gamma=True,
                disable_abs=False, n_bootstrap=bootstrap_iters, greedy=True
            )
            return result.get('mean_accuracy', 0.0), result.get('bootstrap_sem', 0.0)
        
        strat_rl_mean, strat_rl_sem = get_rl_perf(strategic_monkeys, random_state)
        nonstrat_rl_mean, nonstrat_rl_sem = get_rl_perf(nonstrategic_monkeys, random_state + 4)
        
        # Save cache
        with open(rl_cache_path, 'wb') as f:
            pickle.dump({
                'n_bootstrap': bootstrap_iters,
                'random_state': random_state,
                'strategic': {'mean_accuracy': strat_rl_mean, 'bootstrap_sem': strat_rl_sem},
                'nonstrategic': {'mean_accuracy': nonstrat_rl_mean, 'bootstrap_sem': nonstrat_rl_sem}
            }, f)
        print(f"Saved RL cache to {rl_cache_path}")
    
    # --- Plot ---
    x = np.arange(2)  # Strategic, Non-strategic
    width = 0.25
    colors = ['tab:olive', 'tab:red', 'tab:blue']
    
    means = [
        [np.mean(strat_hybrid), strat_rnn_mean, strat_rl_mean],
        [np.mean(nonstrat_hybrid), nonstrat_rnn_mean, nonstrat_rl_mean]
    ]
    sems = [
        [boot_sem(strat_hybrid, bootstrap_iters, random_state), 
         boot_sem(rnn_strat_vals, bootstrap_iters, random_state+2), 
         strat_rl_sem],
        [boot_sem(nonstrat_hybrid, bootstrap_iters, random_state+1), 
         boot_sem(rnn_nonstrat_vals, bootstrap_iters, random_state+3), 
         nonstrat_rl_sem]
    ]
    
    for i, (label, color) in enumerate(zip(['Hybrid', 'RNN', 'RL'], colors)):
        ax.bar(x + (i-1)*width, [m[i] for m in means], width, 
               yerr=[s[i] for s in sems], label=label, color=color, 
               alpha=0.85, capsize=4)
    
    # Style
    ax.set_ylabel('Sequence Prediction Accuracy', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Strategic', 'Non-strategic'], fontsize=14)
    ax.legend(frameon=False, fontsize=12)
    ax.set_ylim(0.45, 0.68)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Add LR upper bound dashed lines ---
    try:
        from analysis_scripts.LLH_behavior_RL import cross_validated_performance_by_monkey_df
        df = pd.read_pickle(monkey_path)
        df = df[df['task'] == 'mp']
        
        res_s = cross_validated_performance_by_monkey_df(df, strategic_monkeys, model='simple', n_folds=10, random_state=random_state)
        res_ns = cross_validated_performance_by_monkey_df(df, nonstrategic_monkeys, model='simple', n_folds=10, random_state=random_state)
        
        lr_strat = res_s.get('LR_mean_accuracy')
        lr_nonstrat = res_ns.get('LR_mean_accuracy')
        
        if lr_strat is not None:
            ax.axhline(lr_strat, xmin=0, xmax=0.45, color='mediumslateblue', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(x[0], lr_strat + 0.005, f'LR {lr_strat:.3f}', ha='center', va='bottom', fontsize=10, color='mediumslateblue')
        if lr_nonstrat is not None:
            ax.axhline(lr_nonstrat, xmin=0.55, xmax=1.0, color='mediumslateblue', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(x[1], lr_nonstrat + 0.005, f'LR {lr_nonstrat:.3f}', ha='center', va='bottom', fontsize=10, color='mediumslateblue')
    except Exception as e:
        print(f"Warning: could not add LR lines: {e}")
    
    # Print summary
    print(f"Performance: Strategic - Hybrid: {means[0][0]:.3f}, RNN: {means[0][1]:.3f}, RL: {means[0][2]:.3f}")
    print(f"Performance: Non-strategic - Hybrid: {means[1][0]:.3f}, RNN: {means[1][1]:.3f}, RL: {means[1][2]:.3f}")

