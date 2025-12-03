import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import pickle
import sys
import ast
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from analysis_scripts.logistic_regression import paper_logistic_regression,paper_logistic_regression_strategic
from figure_scripts.monkey_E_learning import load_behavior
from models.misc_utils import make_env, load_model
from models.misc_utils import RLTester as RL
from models.rnn_model import RLRNN
from models.rl_model import QAgentAsymmetric
from analysis_scripts.test_suite import generate_data
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from figure_scripts.monkey_E_learning import load_behavior
from analysis_scripts.logistic_regression import query_monkey_behavior,fit_bundle_paper, fit_single_paper,fit_single_paper_strategic
from analysis_scripts.logistic_regression import parse_monkey_behavior_strategic, create_order_data, logistic_regression_colinear
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
from figure_scripts.fig1_monkey_RL_RNN_comparison import partition_dataset, cutoff_trials_by_session

def compute_distance(p: np.ndarray, q: np.ndarray, metric = 'frechet') -> float:
    """
    Optimized distance computation for single vectors.
    This is much faster than the general compute_all_frechets function
    when we only need to compare two single vectors.
    """
    
    dist = DISTANCE_METRICS[metric](p, q)
       
    return dist

def get_distance_label(distance_metric):
    """Get human-readable label for distance metric."""
    labels = {
        'euclidean': 'Euclidean Distance',
        'cosine': 'Cosine Similarity',
        'frechet': 'Frechet Distance',
        'area_norm': 'Normalized Area Distance'
    }
    return labels.get(distance_metric, distance_metric.title())

# Old add_performance_histogram function removed - replaced with add_performance_histogram_from_data

mpdb_p = '/Users/fmb35/Desktop/matching-pennies-lite.sqlite'

mpbeh_p = '/Users/fmb35/Desktop/MPbehdata.csv'

# weight_supp_data = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/parsed_data.p'

# weight_supp_data_strategic = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/parsed_data_strategic.p'


weight_supp_data_strategic = '/Users/fmb35/Desktop/BG-PFC-RNN/cluster_scripts/RNN_weighting_comparisons/processed_all_fits.p'

RNN_mis = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/mi_values_for_model.txt'

stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'

# Add at the top, after imports

RLRNN_500_FITS_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/RLRNN_500_fits.pkl'
RLRNN_500_DENSITY_CACHE = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/rlrnn_500_density_cache.pkl'
# RNN_500_ZOO_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig3data/RNN_zoo_dict.pkl'
RNN_500_DENSITY_CACHE = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig5data/rnn_500_density_cache.pkl'
RNN_500_ZOO_PATH = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig1_rnn_data.pkl'

# Consistent typography for axes
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
MODEL_DISTANCE_TITLE_FONTSIZE = 20
MODEL_DISTANCE_TICK_FONTSIZE = 15
COLORBAR_FONTSIZE = 15
MODEL_DISTANCE_AXISLABEL_FONTSIZE = 18

def _ensure_param_tuple(key):
    """
    Normalize a PKL key that may be a tuple or a stringified tuple.
    Returns a (after_loss, after_win) tuple when possible; otherwise returns the key unchanged.
    """
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        try:
            parsed = ast.literal_eval(key)
            if isinstance(parsed, tuple):
                return parsed
        except Exception:
            pass
    return key

def load_all_fits_pkl():
    """Load the new all-fits PKL and return (param_tuple, fits_dict)."""
    pickle_dict = {}
    with open('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/weight_fig_all_fits.pkl', 'rb') as f:
        while True:
            try:
                param_tuple, fits_dict = pickle.load(f)
                param_tuple = _ensure_param_tuple(param_tuple)
                pickle_dict[param_tuple] = fits_dict
            except:
                break
    return pickle_dict

def load_rlrnn_500_fits(path: str = RLRNN_500_FITS_PATH):
    """Load the consolidated 500-fit RLRNN PKL.
    Returns (param_tuple, fits_dict) or (None, {}).
    """
    try:
        with open(path, 'rb') as f:
            param_tuple, fits_dict = pickle.load(f)
        return _ensure_param_tuple(param_tuple), fits_dict
    except Exception as e:
        print(f"Warning: failed to load RLRNN 500 fits from {path}: {e}")
        return None, {}

def _compute_ws_ls_skew_from_action(action: np.ndarray, combined_regressors: bool = False, strategic: bool = True):
    """Compute (ls_skew, ws_skew) from a 20-length action vector.
    Follows the same normalization and combined-regressor logic as the figure.
    """
    if action is None or len(action) < 10:
        return None, None
    if strategic:
        ws = np.array(action[0:5], dtype=float)
        ls = np.array(action[5:10], dtype=float)
        if combined_regressors and len(action) >= 20:
            wsw = np.array(action[10:15], dtype=float)
            lst = np.array(action[15:20], dtype=float)
            if np.max(np.abs(ws)) == 0 or np.max(np.abs(ls)) == 0 or np.max(np.abs(wsw)) == 0 or np.max(np.abs(lst)) == 0:
                return None, None
            ws_skew = (ws[0]/np.max(np.abs(ws)) - wsw[0]/np.max(np.abs(wsw))) / 2.0
            ls_skew = (ls[0]/np.max(np.abs(ls)) - lst[0]/np.max(np.abs(lst))) / 2.0
        else:
            if np.max(np.abs(ws)) == 0 or np.max(np.abs(ls)) == 0:
                return None, None
            ws_skew = ws[0]/np.max(np.abs(ws))
            ls_skew = ls[0]/np.max(np.abs(ls))
    else:
        # Non-strategic indexing: ws at 5:10, ls at 10:15
        if len(action) < 15:
            return None, None
        ws = np.array(action[5:10], dtype=float)
        ls = np.array(action[10:15], dtype=float)
        if np.max(np.abs(ws)) == 0 or np.max(np.abs(ls)) == 0:
            return None, None
        ws_skew = ws[0]/np.max(np.abs(ws))
        ls_skew = ls[0]/np.max(np.abs(ls))
    return float(ls_skew), float(ws_skew)

def overlay_rlrnn_500_density(
    ax,
    combined_regressors: bool = False,
    strategic: bool = True,
    alpha: float = 0.55,
    include_param_pair_models: bool = True,
    use_cache: bool = False,
    scatter_points: bool = False,
    scatter_alpha: float = 0.35,
    scatter_size: float = 28,
    rlrnn_mask_percentile: float = 60.0,
):
    """Overlay KDE density for RLRNN (0.5,0) models using the 500-fits file on the asymmetry axis.
    Uses the same KDE method and grid size as the normal RNN overlay, with caching for speed.
    """
    param_tuple, fits = load_rlrnn_500_fits()
    if not fits:
        return 0
    # Try to load from cache
    try:
        import os
        if use_cache and os.path.exists(RLRNN_500_DENSITY_CACHE):
            with open(RLRNN_500_DENSITY_CACHE, 'rb') as f:
                cache = pickle.load(f)
            key = (bool(combined_regressors), bool(strategic))
            if isinstance(cache, dict) and key in cache:
                payload = cache[key]
                X = payload.get('X'); Y = payload.get('Y'); Z = payload.get('Z')
                extent = payload.get('extent')
                if X is not None and Y is not None and Z is not None and extent is not None:
                    ax.imshow(Z, origin='lower', extent=extent, aspect='auto', cmap='YlOrBr', alpha=alpha, interpolation='bilinear')
                    levels = np.linspace(np.nanpercentile(Z, 5), np.nanmax(Z), 4)
                    ax.contour(X, Y, Z, levels=levels, colors=['#8B4513'], alpha=0.5, linewidths=0.5)
                    print("Used cached RLRNN (0.5,0) density overlay")
                    return int(payload.get('n_models', 0))
    except Exception as e:
        print(f"Warning: failed reading RLRNN 500 density cache: {e}")
    ls_vals = []
    ws_vals = []
    for _idx, v in fits.items():
        if v.get('avg_reward', 0.0) <= 0.45:
            continue
        ls_ws = _compute_ws_ls_skew_from_action(v.get('action'), combined_regressors=combined_regressors, strategic=strategic)
        if ls_ws[0] is None:
            continue
        ls_vals.append(ls_ws[0])
        ws_vals.append(ls_ws[1])
    # Optionally include the ~30 models with the same parameter pair (0.5, 0)
    if include_param_pair_models:
        try:
            sweep = load_all_fits_pkl()
            # find exact or nearest key to (0.5, 0.0)
            target = (0.5, 0.0)
            if target not in sweep:
                # nearest by euclidean in param space
                best_k = None
                best_d = float('inf')
                for k in sweep.keys():
                    try:
                        d = ((float(k[0]) - 0.5)**2 + (float(k[1]) - 0.0)**2) ** 0.5
                    except Exception:
                        continue
                    if d < best_d:
                        best_d = d
                        best_k = k
                if best_k is not None:
                    target = best_k
            fits_30 = sweep.get(target, {})
            for _idx, v in fits_30.items():
                if v.get('avg_reward', 0.0) <= 0.45:
                    continue
                ls_ws = _compute_ws_ls_skew_from_action(v.get('action'), combined_regressors=combined_regressors, strategic=strategic)
                if ls_ws[0] is None:
                    continue
                ls_vals.append(ls_ws[0])
                ws_vals.append(ls_ws[1])
        except Exception as e:
            print(f"Warning: failed to include param-pair models in RLRNN density: {e}")

    if len(ls_vals) <= 5:
        print(f"RLRNN 500-fits: not enough models ({len(ls_vals)}) for density overlay")
        return len(ls_vals)
    try:
        # KDE overlay similar to existing ochre region logic
        x_min, x_max = min(ls_vals), max(ls_vals)
        y_min, y_max = min(ws_vals), max(ws_vals)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        grid_size = 100  # match the normal RNN grid
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([ls_vals, ws_vals])
        kernel = gaussian_kde(values, bw_method='silverman')
        Z = np.reshape(kernel(positions), X.shape)
        # Threshold low-density regions for transparency
        positive = Z[Z > 0]
        if positive.size:
            min_threshold = np.percentile(positive, rlrnn_mask_percentile)
            Z[Z < min_threshold] = np.nan
        extent = [x_min, x_max, y_min, y_max]
        ax.imshow(Z, origin='lower', extent=extent, aspect='auto', cmap='YlOrBr', alpha=alpha, interpolation='bilinear')
        levels = np.linspace(np.nanpercentile(Z, 5), np.nanmax(Z), 4)
        ax.contour(X, Y, Z, levels=levels, colors=['#8B4513'], alpha=0.5, linewidths=0.5)
        if scatter_points:
            try:
                ax.scatter(ls_vals, ws_vals, c='#C8A951', s=scatter_size, alpha=scatter_alpha, marker='o', linewidths=0)
            except Exception:
                pass
        # Save to cache
        try:
            import os
            os.makedirs(os.path.dirname(RLRNN_500_DENSITY_CACHE), exist_ok=True)
            payload = {'X': X, 'Y': Y, 'Z': Z, 'extent': extent, 'n_models': len(ls_vals)}
            try:
                with open(RLRNN_500_DENSITY_CACHE, 'rb') as f:
                    cache = pickle.load(f)
            except Exception:
                cache = {}
            cache[(bool(combined_regressors), bool(strategic))] = payload
            with open(RLRNN_500_DENSITY_CACHE, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: failed to write RLRNN 500 density cache: {e}")
        print(f"Added RLRNN (0.5,0) density from 500-fits with {len(ls_vals)} models")
    except Exception as e:
        print(f"Error overlaying RLRNN 500-fits density: {e}")
    return len(ls_vals)

def overlay_rnn_500_density(ax, alpha: float = 0.5, rnn_contours: int = 3, rnn_mask_percentile: float = 95.0, use_cache: bool = False, scatter_points: bool = False):
    """Overlay KDE density for the 500 RNNs from RNN_zoo_dict.pkl on the asymmetry axis.
    Uses the same KDE/grid as others and caches results for speed.
    """
    import os
    used_cache = False
    try:
        if use_cache and os.path.exists(RNN_500_DENSITY_CACHE):
            with open(RNN_500_DENSITY_CACHE, 'rb') as f:
                payload = pickle.load(f)
            X = payload.get('X'); Y = payload.get('Y'); Z = payload.get('Z')
            extent = payload.get('extent')
            if X is not None and Y is not None and Z is not None and extent is not None:
                ax.imshow(Z, origin='lower', extent=extent, aspect='auto', cmap='Reds', alpha=alpha, interpolation='bilinear')
                levels = np.linspace(np.nanpercentile(Z, 30), np.nanmax(Z), rnn_contours)
                ax.contour(X, Y, Z, levels=levels, colors=['darkred'], alpha=0.45, linewidths=0.5)
                used_cache = True
                print("Used cached RNN 500 density overlay")
    except Exception as e:
        print(f"Warning: failed reading RNN 500 density cache: {e}")
    # Load RNN anchors (ws, ls) from fig1_rnn_data.pkl or fallback to zoo if dict
    try:
        with open(RNN_500_ZOO_PATH, 'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load RNN anchors from {RNN_500_ZOO_PATH}: {e}")
        return 0
    ls_vals = []
    ws_vals = []
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        wsrnn, lsrnn = obj[0], obj[1]
        for ws_val, ls_val in zip(wsrnn, lsrnn):
            ws_vals.append(float(ws_val))
            ls_vals.append(float(ls_val))

    elif isinstance(obj, dict):
        # Fallback: old zoo structure
        for k, v in obj.items():
            act = v.get('action') if isinstance(v, dict) else None
            if act is None or len(act) < 15:
                continue
            ws = np.array(act[0:5], dtype=float)
            ls = np.array(act[5:10], dtype=float)
            wsw = np.array(act[10:15], dtype=float)
            lst = np.array(act[15:20], dtype=float)
            if np.max(np.abs(ws)) == 0 or np.max(np.abs(ls)) == 0 or np.max(np.abs(wsw)) == 0 or np.max(np.abs(lst)) == 0:
                continue
            ws_vals.append(float(ws[0]/np.max(np.abs(ws))) - float(wsw[0]/np.max(np.abs(wsw))))
            ls_vals.append(float(ls[0]/np.max(np.abs(ls))) - float(lst[0]/np.max(np.abs(lst)))) 
       
    else:
        print("Warning: unrecognized RNN anchor structure; skipping density")
        return 0
    if len(ls_vals) <= 5:
        print(f"RNN zoo: not enough models ({len(ls_vals)}) for density overlay")
        return len(ls_vals)
    try:
        x_min, x_max = min(ls_vals), max(ls_vals)
        y_min, y_max = min(ws_vals), max(ws_vals)
        x_range = x_max - x_min
        y_range = y_max - y_min
        # Smaller padding to reduce visual spread
        pad = 0.05
        x_min -= x_range * pad
        x_max += x_range * pad
        y_min -= y_range * pad
        y_max += y_range * pad
        grid_size = 100
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([ls_vals, ws_vals])
        if not used_cache:
            try:
                kernel = gaussian_kde(values, bw_method='silverman')
                Z = np.reshape(kernel(positions), X.shape)
            except Exception:
                # Fallback to simple 2D histogram density if KDE fails
                H, xedges, yedges = np.histogram2d(ls_vals, ws_vals, bins=50, range=[[x_min, x_max],[y_min, y_max]])
                Z = H.T
                X, Y = np.meshgrid((xedges[:-1]+xedges[1:])/2, (yedges[:-1]+yedges[1:])/2)
            positive = Z[Z > 0]
            if positive.size:
                min_threshold = np.percentile(positive, rnn_mask_percentile)
                Z[Z < min_threshold] = np.nan
            extent = [x_min, x_max, y_min, y_max]
            ax.imshow(Z, origin='lower', extent=extent, aspect='auto', cmap='Reds', alpha=alpha, interpolation='bilinear')
            levels = np.linspace(np.nanpercentile(Z, 30), np.nanmax(Z), 4)
            ax.contour(X, Y, Z, levels=levels, colors=['darkred'], alpha=0.6, linewidths=0.7)
            # cache
            try:
                os.makedirs(os.path.dirname(RNN_500_DENSITY_CACHE), exist_ok=True)
                payload = {'X': X, 'Y': Y, 'Z': Z, 'extent': extent, 'n_models': len(ls_vals)}
                with open(RNN_500_DENSITY_CACHE, 'wb') as f:
                    pickle.dump(payload, f)
            except Exception as e:
                print(f"Warning: failed to write RNN 500 density cache: {e}")
            print(f"Added RNN (zoo) density with {len(ls_vals)} models")
    except Exception as e:
        print(f"Error overlaying RNN 500 density: {e}")
    # Optionally scatter anchors
    if scatter_points:
        ax.scatter(ls_vals, ws_vals, c='tab:red', s=14, alpha=0.15, marker='o', linewidths=0)
    # Plot the medoid (closest to mean in cosine space) of the RNN anchors
    anchors = np.stack([np.array(ls_vals), np.array(ws_vals)], axis=1)
    norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    nanchors = anchors / norms
    centroid = np.mean(nanchors, axis=0)
    sims = nanchors @ (centroid / (np.linalg.norm(centroid) if np.linalg.norm(centroid) != 0 else 1e-9))
    midx = int(np.argmax(sims))
    m_ls, m_ws = anchors[midx, 0], anchors[midx, 1]
    ax.scatter(m_ls, m_ws, color='darkred', marker='X', s=120, edgecolor='k', linewidths=0.7, label='Representative RNN')
    
    return len(ls_vals)

def summarize_rlrnn_500_performance(bootstrap_iters: int = 1000, random_state: int = 0):
    """Compute mean accuracy and bootstrap SEM for strategic and nonstrategic groups
    from the 500-fits file if available.
    """
    param_tuple, fits = load_rlrnn_500_fits()
    if not fits:
        return None
    param_key = f"{float(param_tuple[0]):g}, {float(param_tuple[1]):g}"
    perf = {'strategic': [], 'nonstrategic': []}
    for k, v in fits.items():
        if v.get('avg_reward', 0.0) <= 0.45:
            continue
        sp = v.get('sequence_prediction', {})
        idx_str = str(v.get('model_idx', k))
        for group in ['strategic', 'nonstrategic']:
            gdict = sp.get(group)
            if not isinstance(gdict, dict):
                continue
            by_param = gdict.get(param_key)
            if not isinstance(by_param, dict):
                continue
            arrs = by_param.get(idx_str)
            if not isinstance(arrs, (list, tuple)) or len(arrs) == 0:
                continue
            tot = 0.0
            n = 0
            for a in arrs:
                if isinstance(a, np.ndarray):
                    # Arrays are boolean correctness; take mean
                    tot += float(np.sum(a))
                    n += int(a.size)
            if n > 0:
                perf[group].append(tot / n)
    def _bootstrap_sem(values, n_boot, seed):
        if not values:
            return 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n == 0:
            return 0.0
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[i] = float(np.mean(arr[idx]))
        return float(np.std(means, ddof=1)) if n_boot > 1 else 0.0
    results = {}
    for g in ['strategic', 'nonstrategic']:
        vals = perf[g]
        mean = float(np.mean(vals)) if vals else float('nan')
        sem = _bootstrap_sem(vals, bootstrap_iters, random_state + (11 if g=='strategic' else 13))
        results[g] = {'mean': mean, 'sem': sem, 'n_models': int(len(vals))}
    return results

def _setup_asymmetry_axis(ax, combined_regressors: bool, strategic: bool):
    """Configure the behavioral asymmetry axis consistently with Fig 5."""
    if combined_regressors:
        ax.set_xlabel(r'Win Bias: $\frac{1}{2}(\frac{ws_1}{max(|ws|)} - \frac{wsw_1}{max(|wsw|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
        ax.set_ylabel(r'Lose Bias: $\frac{1}{2}(\frac{ls_1}{max(|ls|)} - \frac{lst_1}{max(|lst|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
    else:
        ax.set_xlabel(r'Win Bias: $\frac{1}{2}(\frac{ws_1}{max(|ws|)} - \frac{wsw_1}{max(|wsw|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
        ax.set_ylabel(r'Lose Bias: $\frac{1}{2}(\frac{ls_1}{max(|ls|)} - \frac{lst_1}{max(|lst|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax.grid(False)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 'k--', alpha=0.5, lw=1)
    # Diagonal reference
    try:
        ax.axline((0, 0), slope=1, linestyle='--', color='k')
    except Exception:
        pass

def export_asymmetry_progress_images(corr_list, s_ls, s_ws, combined_regressors=False, strategic=True, out_dir=None, rl_point=None):
    """
    Save a sequence of PNGs showing the incremental build-up of the behavioral
    asymmetry plot in this order: empty, MP1, MP2, RNN, RLRNN.

    MP1 corresponds to Algorithm 1 monkeys (strategy == '1');
    MP2 corresponds to the strategic monkeys (strategy == '0').
    """
    try:
        import os
        from matplotlib.lines import Line2D
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), 'fig5data', 'asymmetry_progress')
        os.makedirs(out_dir, exist_ok=True)

        strategic_monkeys = ['E', 'D', 'I']
        nonstrategic_monkeys = ['C', 'H', 'F', 'K']

        def _new_fig_ax():
            fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2), dpi=600)
            try:
                fig.patch.set_alpha(1.0)
            except Exception:
                pass
            _setup_asymmetry_axis(ax, combined_regressors, strategic)
            return fig, ax

        def _save(fig, name):
            p = os.path.join(out_dir, name)
            # Ensure directory exists and save robustly
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
            except Exception:
                pass
            try:
                fig.savefig(p, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving asymmetry progress image to {p}: {e}")
            plt.close(fig)
            print(f"Saved asymmetry progress image: {p}")

        # 0) Empty
        fig, ax = _new_fig_ax()
        _save(fig, '00_empty.png')

        # Helper to scatter a list of (lsm, wsm, label, color, marker)
        def _scatter_points(ax, items):
            for lsm, wsm, label, color, marker in items:
                ax.scatter(lsm, wsm, c=color, marker=marker, s=120)
                ax.text(lsm, wsm, label, ha='right', va='bottom', fontsize=12,
                        bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=2))

        # Prepare groups:
        # - MP1: Algorithm 1 baseline (yellow circles)
        # - MP2: Non-strategic monkeys (purple circles)
        # - strategic: Strategic monkeys (cyan stars) [not part of this sequence, but kept for parity]
        mp1_points = []
        mp2_points = []  # will hold non-strategic (purple) points
        strategic_points = []
        for m in corr_list:
            lsm, wsm = m['non-monotonicity']
            if m['strategy'] == '1':  # Algorithm 1 group (MP1)
                mp1_points.append((lsm, wsm, m['monkey'], '#F2B705', 'o'))
            else:  # Real monkey groups
                if m['monkey'] in nonstrategic_monkeys:
                    mp2_points.append((lsm, wsm, m['monkey'], 'tab:purple', 'o'))
                elif m['monkey'] in strategic_monkeys:
                    strategic_points.append((lsm, wsm, m['monkey'], 'tab:cyan', '*'))

        # 1) MP1 only
        fig, ax = _new_fig_ax()
        _scatter_points(ax, mp1_points)
        _save(fig, '01_mp1.png')

        # 2) MP2 added (both real monkey groups: non-strategic + strategic)
        fig, ax = _new_fig_ax()
        _scatter_points(ax, mp1_points)
        _scatter_points(ax, mp2_points)
        _scatter_points(ax, strategic_points)
        _save(fig, '02_mp1_mp2.png')

        # 3) Add RL point (MP1+MP2+RL)
        fig, ax = _new_fig_ax()
        _scatter_points(ax, mp1_points)
        _scatter_points(ax, mp2_points)
        _scatter_points(ax, strategic_points)
        try:
            if rl_point is not None:
                ax.scatter(rl_point[0], rl_point[1], color='tab:blue', marker='D', s=120)
        except Exception:
            pass
        _save(fig, '03_mp1_mp2_rl.png')

        # 4) Add RNN overlay (MP1+MP2+RL+RNN)
        fig, ax = _new_fig_ax()
        _scatter_points(ax, mp1_points)
        _scatter_points(ax, mp2_points)
        _scatter_points(ax, strategic_points)
        try:
            if rl_point is not None:
                ax.scatter(rl_point[0], rl_point[1], color='tab:blue', marker='D', s=120)
        except Exception:
            pass
        try:
            overlay_rnn_500_density(ax, alpha=0.5, rnn_contours=3, rnn_mask_percentile=95.0, use_cache=False, scatter_points=False)
        except Exception as e:
            print(f"Warning: could not overlay RNN density in progress export: {e}")
        _save(fig, '04_mp1_mp2_rl_rnn.png')

        # 5) Add RLRNN overlay and representative point (MP1+MP2+RL+RNN+RLRNN)
        fig, ax = _new_fig_ax()
        _scatter_points(ax, mp1_points)
        _scatter_points(ax, mp2_points)
        _scatter_points(ax, strategic_points)
        try:
            if rl_point is not None:
                ax.scatter(rl_point[0], rl_point[1], color='tab:blue', marker='D', s=120)
        except Exception:
            pass
        # Ensure the RNN overlay is also present in the final frame
        try:
            overlay_rnn_500_density(ax, alpha=0.5, rnn_contours=3, rnn_mask_percentile=95.0, use_cache=False, scatter_points=False)
        except Exception as e:
            print(f"Warning: could not overlay RNN density in final progress export: {e}")
        try:
            overlay_rlrnn_500_density(
                ax,
                combined_regressors=combined_regressors,
                strategic=strategic,
                alpha=0.7,
                scatter_points=False,
                rlrnn_mask_percentile=60.0,
            )
        except Exception as e:
            print(f"Warning: could not overlay RLRNN density in progress export: {e}")
        try:
            ax.scatter(s_ls, s_ws, color='tab:olive', marker='P', s=120)
            mvec, midx_id, mparams = find_rlrnn_medoid_for_param((0.5, 0.0), strategic=strategic, use_subset='all')
            if mvec is not None:
                if strategic:
                    ws = mvec[0:5]; ls = mvec[5:10]
                else:
                    ws = mvec[5:10]; ls = mvec[10:15]
                m_ws = ws[0]/(np.max(np.abs(ws)) if np.max(np.abs(ws))!=0 else 1e-9)
                m_ls = ls[0]/(np.max(np.abs(ls)) if np.max(np.abs(ls))!=0 else 1e-9)
                ax.scatter(m_ls, m_ws, color='#C8A951', marker='X', s=140, edgecolor='#5C3D00', linewidths=1.2)
        except Exception as e:
            print(f"Warning: could not plot representative RLRNN in progress export: {e}")
        _save(fig, '05_mp1_mp2_rl_rnn_rlrnn.png')

    except Exception as e:
        print(f"Warning: export_asymmetry_progress_images failed: {e}")

def merge_rlrnn_500_into(data_dict: dict) -> dict:
    """Merge the consolidated RLRNN (0.5,0) 500-fits into a data_dict of the form
    { (after_loss, after_win): { idx: fit_data_dict } } used throughout fig5.
    Ensures unique idx keys by offsetting indices when needed.
    """
    param_tuple, fits = load_rlrnn_500_fits()
    if not fits or param_tuple is None:
        return data_dict
    try:
        pt = (float(param_tuple[0]), float(param_tuple[1]))
    except Exception:
        pt = param_tuple
    if pt in data_dict:
        existing = data_dict[pt]
        # compute offset for new indices
        try:
            existing_keys = [int(k) for k in existing.keys() if str(k).isdigit()]
            offset = (max(existing_keys) + 1) if existing_keys else 0
        except Exception:
            offset = 0
        for k, v in fits.items():
            try:
                base_idx = int(k) if str(k).isdigit() else 0
            except Exception:
                base_idx = 0
            new_idx = base_idx + offset
            # avoid rare collisions
            while new_idx in existing:
                new_idx += 1
            existing[new_idx] = v
    else:
        data_dict[pt] = fits
    return data_dict

def generate_model_figure(pfcw_params, pfc_params, mpdb_path, env_params,img_path, nits = 2, ep = None, order=5, bias = True, strategic=True, fit_single=False, perf = False, use_median=True, power_scale=2.5, n_top_models=100,
                        distance_metric='area_norm', use_subset='all', combined_regressors=False, skip_model_loading=False, make_extra_figures=False, cv_folds=10, bootstrap_iters=1000, cv_random_state=0,
                        override_strategic_param=None, override_nonstrategic_param=None, layout_debug: bool = False, use_constrained_layout: bool = False,
                        export_asymmetry_progress: bool = False, asymmetry_progress_dir: str = None):
    """
    Generate the complete Figure 5 visualization.
    
    Args:
        pfcw_params: PFC weighted parameters
        pfc_params: PFC parameters  
        mpdb_path: Path to monkey database
        env_params: Environment parameters
        img_path: Image path for saving
        nits: Number of iterations
        ep: Episode number
        order: Order for regression analysis
        bias: Whether to include bias in regression
        strategic: Whether to use strategic regression
        fit_single: Whether to fit single sessions
        perf: Whether to use performance instead of distance
        use_median: Whether to use median aggregation
        power_scale: Power scaling factor
        n_top_models: Number of top models to consider
        distance_metric: Distance metric to use
        use_subset: Which regression components to analyze ('all', 'ws_ls')
        combined_regressors: Whether to use combined regressor asymmetry calculation
    """
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
    
    # RLRNN_model_w = RLRNN(environment = env, **pfcw_params)
    # if epw == None:
    #     RLRNN_model_w.load_model('')
    # else:
    #     RLRNN_model_w.load_model_ep(epw)
        

    # Define correct strategic and non-strategic monkeys
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']  # Non-strategic monkeys
    strategic_monkeys = ['E', 'D', 'I']  # Strategic monkeys

    # Create figure without constrained layout (constrained will be enabled before returning)
    # Slightly taller to prevent title clipping
    fig = plt.figure(figsize=(26.5, 12.5), dpi=600)
    
    # Independent GridSpecs per block with explicit figure coords, so each block is sized individually
    # Dialable proportions
    gap = 0.05
    # Slightly enlarge the center-right grid block; keep within bounds
    left_w, c1_w, c2_w, right_w = 0.20, 0.24, 0.26, 0.10
    left_l = 0.02
    c1_l = left_l + left_w + gap
    c2_l = c1_l + c1_w + gap
    right_l = c2_l + c2_w + gap
    # Compute vertical bounds so center blocks are square given fig aspect
    fig_ratio = fig.get_figwidth() / fig.get_figheight()  # W/H
    center_h = min(0.78, c1_w * fig_ratio)
    top = 0.90  # a bit lower to increase space under suptitle
    bottom = max(0.04, top - center_h)

    gs_left = GridSpec(2, 2, figure=fig, left=left_l, right=left_l + left_w, bottom=bottom, top=top,
                       height_ratios=[2, 1], width_ratios=[1, 1])
    gs_c1 = GridSpec(1, 1, figure=fig, left=c1_l, right=c1_l + c1_w, bottom=bottom, top=top)
    gs_c2 = GridSpec(1, 1, figure=fig, left=c2_l, right=c2_l + c2_w, bottom=bottom, top=top)
    # Ensure the right block keeps its intended width; shift left edge if it would be clipped by the 0.98 boundary
    if right_l + right_w > 0.98:
        right_l = max(0.98 - right_w, 0.02)
    gs_right = GridSpec(1, 1, figure=fig, left=right_l, right=right_l + right_w, bottom=bottom, top=top)

    # Left block: top spans both columns, bottom has two columns
    RLRNN_ax = fig.add_subplot(gs_left[0, 0:2])
    BG_ax = fig.add_subplot(gs_left[1, 0])
    PFC_ax = fig.add_subplot(gs_left[1, 1])

    # Center-left and center-right
    asymmetry_ax = fig.add_subplot(gs_c1[0, 0])
    model_distance_ax = fig.add_subplot(gs_c2[0, 0])

    # Enforce minimum width for the performance panel (at least 12% of fig width)
    right_left = gs_right.get_geometry()[0] if hasattr(gs_right, 'get_geometry') else right_l
    # Create a new GridSpec if current width is too small
    if (min(right_l + right_w, 0.98) - right_l) < 0.12:
        adj_left = max(0.98 - 0.12, right_l)
        gs_right = GridSpec(1, 1, figure=fig, left=adj_left, right=0.98, bottom=bottom, top=top)
    perf_hist_ax = fig.add_subplot(gs_right[0, 0])

    # Optional fast preview mode: draw only frames/titles to iterate on proportions
    if layout_debug:
        # Titles
        titles = ['RLRNN Model', 'RL Module', 'RNN Module', 'Behavioral Asymmetry', 'Model Cosine Similarity', 'Performance by Group']
        sizes = [20, 12, 12, 20, 16, 16]
        axs = [RLRNN_ax, BG_ax, PFC_ax, asymmetry_ax, model_distance_ax, perf_hist_ax]
        for i, ax in enumerate(axs):
            ax.set_title(titles[i], fontsize=sizes[i])
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines['top'].set_alpha(0.4)
            ax.spines['right'].set_alpha(0.4)
            ax.spines['bottom'].set_alpha(0.4)
            ax.spines['left'].set_alpha(0.4)
        fig.suptitle('Figure 5: RLRNN Model Yields Monkey-Like Behavior', fontsize=26, y=0.965)
        if use_constrained_layout:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
        return fig

    # Keep center plots square; allow left top to be slightly flexible for legend
    try:
        asymmetry_ax.set_box_aspect(1)
        model_distance_ax.set_box_aspect(1)
    except Exception:
        pass
    
    # Setup axes list for plotting (no MI axis)
    axs = [RLRNN_ax, BG_ax, PFC_ax, asymmetry_ax, model_distance_ax, perf_hist_ax]
   
    # Load RLRNN data from precomputed pickle; prefer idx=226, fall back to 450
    fig5dir = os.path.join(os.path.dirname(__file__), 'fig5data')
    candidate_files = [
        os.path.join(fig5dir, '(0.5,0)_226_data.p'),
        os.path.join(fig5dir, '(0.5,0)_450_data.p'),
    ]
    data = None
    used_path = None
    for dp in candidate_files:
        if os.path.exists(dp):
            try:
                with open(dp, 'rb') as f:
                    data = pickle.load(f)
                used_path = dp
                break
            except Exception:
                continue
    if data is None:
        raise FileNotFoundError(
            "No precomputed RLRNN data file found. Looked for: " + ", ".join(candidate_files)
        )
    print(f"Successfully loaded RLRNN data from pickle file: {used_path}")
    
    BG_data = [data[0],np.array(data[-1]),data[2],data[3],None,None]
    PFC_data = [data[0],np.array(data[-2]),data[2],data[3],None,None]
    
    if strategic:
        fit_s = paper_logistic_regression_strategic(axs[0],True, data=data,legend=True, return_model=True, bias = bias)
        paper_logistic_regression_strategic(axs[1],True,data=BG_data, legend = False, bias = bias)
        paper_logistic_regression_strategic(axs[2],True, data=PFC_data,legend=False, bias = bias)
    else:        
        fit_s = paper_logistic_regression(axs[0],True, data=data,legend=True, return_model=True, bias = bias)
        paper_logistic_regression(axs[1],True,data=BG_data, legend = False, bias = bias)
        paper_logistic_regression(axs[2],True, data=PFC_data,legend=False, bias = bias)
    
    # data_w = generate_data(RLRNN_model_w,env,nits)
    # BG_data_w = [data_w[0],np.array(data_w[-1]),data_w[2],data_w[3],None,None]
    # PFC_data_w = [data_w[0],np.array(data_w[-2]),data_w[2],data_w[3],None,None]
    # fit_w = paper_logistic_regression(axs[4],True, data=data_w,legend=False, return_model=True)
    # paper_logistic_regression(axs[5],True,data=BG_data_w, legend = False)
    # paper_logistic_regression(axs[6],True, data=PFC_data_w,legend=False)
    
    # switching_fit = fit_bundle_paper(data_w,order=order)
    # weighted_fit = fit_bundle_paper(data,order=order)
    
    # s_ws = switching_fit[order:2*order]
    # s_ls = switching_fit[2*order:3*order]
    # w_ws = weighted_fit[order:2*order]
    # w_ls = weighted_fit[2*order:3*order]
    
    if strategic:
        s_ws = fit_s['action'][0:order]
        s_ls = fit_s['action'][order:2*order]
        if combined_regressors:
            s_wsw = fit_s['action'][2*order:3*order]  # Win-Switch
            s_lst = fit_s['action'][3*order:4*order]  # Lose-Stay
    
    else:
        s_ws = fit_s['action'][order:2*order]
        s_ls = fit_s['action'][2*order:3*order]
    # w_ws = fit_w['action'][order:2*order]
    # w_ls = fit_w['action'][2*order:3*order]
    
    # Update skew calculation to use max(abs()) as in fig1
    if combined_regressors and strategic:
        # Combined regressors formula: (ls - lst)/2 and (ws - wsw)/2
        s_ws_skew = s_ws[0]/max(np.abs(s_ws)) if max(np.abs(s_ws)) > 0 else 0
        s_wsw_skew = s_wsw[0]/max(np.abs(s_wsw)) if max(np.abs(s_wsw)) > 0 else 0
        s_ls_skew = s_ls[0]/max(np.abs(s_ls)) if max(np.abs(s_ls)) > 0 else 0
        s_lst_skew = s_lst[0]/max(np.abs(s_lst)) if max(np.abs(s_lst)) > 0 else 0
        
        s_ws = (s_ws_skew - s_wsw_skew) / 2
        s_ls = (s_ls_skew - s_lst_skew) / 2
    else:
        s_ws = s_ws[0]/max(np.abs(s_ws))
        s_ls = s_ls[0]/max(np.abs(s_ls))
    
    rl_anchor_path = os.path.join(os.path.dirname(os.getcwd()), 'figure_scripts', 'fig1_rl_data.p')
    lsbg, wsbg = [1, 1]
    if os.path.exists(rl_anchor_path):
        with open(rl_anchor_path, 'r+') as f:
            s = f.read()
            s = s.replace(', ', ',')
            s = s.replace(' ', ',')
            s = s.replace('-', ',-')
            s = s.split(',')
            s = [float(i) for i in s if i != '']
            if len(s) >= 4:
                lsbg, wsbg = s[:2]

        
    # with open(os.path.join(os.path.dirname(os.getcwd()),'figure_scripts','fig1_data.p'),'w+') as f:
    #     #average across nonstrategic monkeys and write lsm, wsm
    #     f.write('{}, {},'.format(np.mean(lsns),np.mean(wsns)))
    #     f.write('{}, {},'.format(np.mean(lss), np.mean(wss)))
    #     f.write('{}, {},'.format(np.mean(rnn_ls),np.mean(rnn_ws))) # rnn family regressors
    # actually i want to plot ls and ws individually for the strategic and nonstrategic monkeys
    
    lsnon = []
    wsnon = []
    lss = []
    wss = []
    
    nonstrat_corr = []
    strat_corr = []
    corr_list = []
    
    
    
    monkey_beh_data = load_behavior(mpbeh_p,algorithm=2)
    monkeys = ['E','C','D','I','K','H','F']
    # monkeys = ['E','D','I','C','K','H','F']

    #['E', 'D', 'C', 'H', 'I', 'K','F']
    monkey_data = [mp_data[mp_data['animal'] == monkeys[i]] for i in range(len(monkeys)-1)]
    monkey_data.append(monkey_beh_data[monkey_beh_data['animal']==112])
    
    # Load monkey correlation data - if empty, populate it
    target_fig1 = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/fig1_data.pkl'
    try:
        with open(target_fig1,'rb') as f:
            corr_list = pickle.load(f)
            if len(corr_list) == 0:
                print("fig1_data.pkl is empty, computing monkey regression data...")
                # Compute monkey regression data
                for i, m_data in enumerate(monkey_data):
                    if len(m_data) > 0:
                        monkey_name = monkeys[i] if i < len(monkeys)-1 else 'F'
                        try:
                            if strategic:
                                result = paper_logistic_regression_strategic(None, False, data=m_data, order=5, bias=True, return_model=True)
                                if result is not None and 'action' in result:
                                    coeffs = result['action'][:-1]  # Remove bias term
                                    ws_coeffs = coeffs[0:5]
                                    ls_coeffs = coeffs[5:10]
                                    # Calculate non-monotonicity (skew)
                                    if combined_regressors:
                                        wsw_coeffs = coeffs[10:15]  # Win-Switch
                                        lst_coeffs = coeffs[15:20]  # Lose-Stay
                                        ws_skew = ws_coeffs[0] / max(np.abs(ws_coeffs)) if max(np.abs(ws_coeffs)) > 0 else 0
                                        wsw_skew = wsw_coeffs[0] / max(np.abs(wsw_coeffs)) if max(np.abs(wsw_coeffs)) > 0 else 0
                                        ls_skew = ls_coeffs[0] / max(np.abs(ls_coeffs)) if max(np.abs(ls_coeffs)) > 0 else 0
                                        lst_skew = lst_coeffs[0] / max(np.abs(lst_coeffs)) if max(np.abs(lst_coeffs)) > 0 else 0
                                        # Combined regressors formula: (ls - lst)/2 and (ws - wsw)/2
                                        ws_skew = (ws_skew - wsw_skew) / 2
                                        ls_skew = (ls_skew - lst_skew) / 2
                                    else:
                                        ws_skew = ws_coeffs[0] / max(np.abs(ws_coeffs)) if max(np.abs(ws_coeffs)) > 0 else 0
                                        ls_skew = ls_coeffs[0] / max(np.abs(ls_coeffs)) if max(np.abs(ls_coeffs)) > 0 else 0
                                    
                                    # Determine strategy type
                                    is_strategic = monkey_name in strategic_monkeys
                                    strategy = '0' if is_strategic else '1'
                                    
                                    monkey_entry = {
                                        'monkey': monkey_name,
                                        'non-monotonicity': [ls_skew, ws_skew],
                                        'strategy': strategy
                                    }
                                    corr_list.append(monkey_entry)
                                    print(f"Added data for monkey {monkey_name}: ls={ls_skew:.3f}, ws={ws_skew:.3f}")
                            else:
                                result = paper_logistic_regression(None, False, data=m_data, order=5, bias=True, return_model=True)
                                if result is not None and 'action' in result:
                                    coeffs = result['action']
                                    ws_coeffs = coeffs[5:10]
                                    ls_coeffs = coeffs[10:15]
                                    # Calculate non-monotonicity (skew)
                                    ws_skew = ws_coeffs[0] / max(np.abs(ws_coeffs)) if max(np.abs(ws_coeffs)) > 0 else 0
                                    ls_skew = ls_coeffs[0] / max(np.abs(ls_coeffs)) if max(np.abs(ls_coeffs)) > 0 else 0
                                    
                                    # Determine strategy type
                                    is_strategic = monkey_name in strategic_monkeys
                                    strategy = '0' if is_strategic else '1'
                                    
                                    monkey_entry = {
                                        'monkey': monkey_name,
                                        'non-monotonicity': [ls_skew, ws_skew],
                                        'strategy': strategy
                                    }
                                    corr_list.append(monkey_entry)
                                    print(f"Added data for monkey {monkey_name}: ls={ls_skew:.3f}, ws={ws_skew:.3f}")
                        except Exception as e:
                            print(f"Error processing monkey {monkey_name}: {e}")
                            continue
                
                # Save the computed data back to the file
                with open(target_fig1,'wb') as f:
                    pickle.dump(corr_list, f)
                print(f"Saved {len(corr_list)} monkey entries to fig1_data.pkl")
    except Exception as e:
        raise FileNotFoundError(f"Could not load fig1_data.pkl at expected path: {target_fig1}. Error: {e}")
    
    # Optionally export a step-by-step build-up of the asymmetry plot
    if export_asymmetry_progress:
        try:
            # If no directory provided, prefer alongside img_path when available
            _progress_dir = asymmetry_progress_dir
            try:
                if _progress_dir is None and isinstance(img_path, str) and len(img_path) > 0:
                    import os as _os
                    _progress_dir = _os.path.join(_os.path.dirname(img_path), 'asymmetry_progress')
            except Exception:
                pass
            export_asymmetry_progress_images(
                corr_list=corr_list,
                s_ls=s_ls,
                s_ws=s_ws,
                combined_regressors=combined_regressors,
                strategic=strategic,
                out_dir=_progress_dir,
                rl_point=(lsbg, wsbg)
            )
        except Exception as _e:
            print(f"Warning: failed to export asymmetry progress images: {_e}")

    
    # asymmetry_ax.invert_xaxis()
    # asymmetry_ax.plot([0, 1], [0, 1], transform=asymmetry_ax.transAxes,color = 'black', linestyle = '--',alpha=.5)
    asymmetry_ax.clear()
    
    # Set the axis labels to match fig1 bias figure
    if combined_regressors:
        asymmetry_ax.set_xlabel(r'Win Bias: $\frac{1}{2}(\frac{ws_1}{max(|ws|)} - \frac{wsw_1}{max(|wsw|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
        asymmetry_ax.set_ylabel(r'Lose Bias: $\frac{1}{2}(\frac{ls_1}{max(|ls|)} - \frac{lst_1}{max(|lst|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
        # fig.suptitle('Behavioral Asymmetry (Combined WS+LS)', fontsize=26, y=0.97)
        asymmetry_ax.set_title('Behavioral Asymmetry', fontsize=18, fontweight='bold', pad=18)
    else:
        asymmetry_ax.set_xlabel(r'Win Bias: $\frac{1}{2}(\frac{ws_1}{max(|ws|)} - \frac{wsw_1}{max(|wsw|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
        asymmetry_ax.set_ylabel(r'Lose Bias: $\frac{1}{2}(\frac{ls_1}{max(|ls|)} - \frac{lst_1}{max(|lst|)})$', fontsize=LABEL_FONTSIZE, labelpad=6)
    # Add horizontal and vertical lines at y=0 and x=0
    asymmetry_ax.axhline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    asymmetry_ax.axvline(0, color='gray', linestyle='--', alpha=0.5, lw=1)
    
    # Turn off grid
    asymmetry_ax.grid(False)
    
    # Set axis limits and ticks to match fig1
    asymmetry_ax.set_xlim(-1.1, 1.1)
    asymmetry_ax.set_ylim(-1.1, 1.1)
    asymmetry_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    asymmetry_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    asymmetry_ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    
    # Add dashed box at boundaries
    asymmetry_ax.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], 'k--', alpha=0.5, lw=1)
    
    # Initialize empty lists for strategic and non-strategic data
    lsnon = []
    wsnon = []
    lss = []
    wss = []
    
    # Process each monkey with correct labeling
    for m in corr_list:
        lsm, wsm = m['non-monotonicity']
        # Check if monkey is strategic or non-strategic
        if m['monkey'] in strategic_monkeys and m['strategy'] != '1':
            marker = '*'
            color = 'tab:cyan'
            lss.append(m['non-monotonicity'][0])
            wss.append(m['non-monotonicity'][1])
            asymmetry_ax.scatter(lsm, wsm, c=color, marker=marker, s=128, label=f"Strategic" if m['monkey'] == strategic_monkeys[0] else "")
            asymmetry_ax.text(lsm, wsm, m['monkey'], ha='right', va='bottom', fontsize=12, 
                           bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=2))
        elif m['monkey'] in nonstrategic_monkeys and m['strategy'] != '1':
            marker = 'o'
            color = 'tab:purple'
            lsnon.append(m['non-monotonicity'][0])
            wsnon.append(m['non-monotonicity'][1])
            asymmetry_ax.scatter(lsm, wsm, c=color, marker=marker, s=120, label=f"Non-Strategic" if m['monkey'] == nonstrategic_monkeys[0] else "")
            asymmetry_ax.text(lsm, wsm, m['monkey'], ha='right', va='bottom', fontsize=12,
                           bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=2))
        else:
            marker = 'o'
            color = '#F2B705'
            lsnon.append(m['non-monotonicity'][0])
            wsnon.append(m['non-monotonicity'][1])
            asymmetry_ax.scatter(lsm, wsm, c=color, marker=marker, s=120, label=f"Algorithm 1" if m['monkey'] == nonstrategic_monkeys[0] else "")
            asymmetry_ax.text(lsm, wsm, m['monkey'], ha='right', va='bottom', fontsize=12,
                           bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=2))
    
    # Representative RLRNN point (default from current data)
    asymmetry_ax.scatter(s_ls, s_ws, label='Representative RLRNN', color='tab:olive', marker='P', s=120)

    # Plot models with clear labels and overlay requested densities
    asymmetry_ax.scatter(s_ls, s_ws, label='Representative RLRNN', color='tab:olive', marker='P', s=120)
    try:
        n_rlrnn = overlay_rlrnn_500_density(
            asymmetry_ax,
            combined_regressors=combined_regressors,
            strategic=strategic,
            alpha=0.7,
            scatter_points=True,
            scatter_alpha=0.35,
            scatter_size=30,
            rlrnn_mask_percentile=60.0,
        )
        if n_rlrnn:
            print(f"RLRNN 500 density overlay applied with {n_rlrnn} models")
    except Exception as e:
        print(f"Warning: could not overlay RLRNN 500 density: {e}")
    # Overlay RNN bias contours using fig1_rnn_data.pkl
    try:
        n_rnn = overlay_rnn_500_density(asymmetry_ax, alpha=0.5, rnn_contours=3, rnn_mask_percentile=95.0, use_cache=False, scatter_points=True)
        if n_rnn:
            print(f"RNN density/contours overlay applied with {n_rnn} models")
    except Exception as e:
        print(f"Warning: could not overlay RNN density: {e}")
    # Ensure medoid marker is drawn on top (no try/except)
    mvec, midx_id, mparams = find_rlrnn_medoid_for_param((0.5,0.0), strategic=strategic, use_subset=use_subset)
    if mvec is not None:
        if strategic:
            ws = mvec[0:5]
            ls = mvec[5:10]
        else:
            ws = mvec[5:10]
            ls = mvec[10:15]
        m_ws = ws[0]/(np.max(np.abs(ws)) if np.max(np.abs(ws))!=0 else 1e-9)
        m_ls = ls[0]/(np.max(np.abs(ls)) if np.max(np.abs(ls))!=0 else 1e-9)
        asymmetry_ax.scatter(m_ls, m_ws, label='Representative RLRNN', color='#C8A951', marker='X', s=140, edgecolor='#5C3D00', linewidths=1.2)
        # # Print and annotate winrate for the representative model if available
        # try:
        #     all_fits = merge_rlrnn_500_into(load_all_fits_pkl())
        #     pkey = (0.5, 0.0)
        #     fits = all_fits.get(pkey, {})
        #     winrates = []
        #     for k, v in fits.items():
        #         try:
        #             if 'avg_reward' in v and v['avg_reward'] > 0:
        #                 winrates.append(float(v['avg_reward']))
        #         except Exception:
        #             continue
        #     if winrates:
        #         wr = float(np.mean(winrates))
        #         print(f"Representative RLRNN approximate mean winrate across (0.5,0) models: {wr:.3f}")
        #         try:
        #             asymmetry_ax.text(m_ls, m_ws+0.05, f"winrate~{wr:.3f}", ha='center', va='bottom', fontsize=10, color='#5C3D00')
        #         except Exception:
        #             pass
        # except Exception as e:
        #     print(f"Warning: could not compute/print RLRNN winrate: {e}")
    
    asymmetry_ax.scatter(lsbg, wsbg, label='RL', color='tab:blue', marker='D', s=120)
    
    # Add diagonal line. Diagonal line should have slope 1 and pass through (0,0). THis should be origin not axis coordinates
    asymmetry_ax.axline((0,0), slope=1, linestyle='--', color='k')
    
    # Add legend with clear groupings - moved to upper right to avoid covering data
    handles, labels = asymmetry_ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Make legend transparent and position in upper right
    asymmetry_ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=12, frameon=False)
    # Prevent legend and labels from being clipped
    asymmetry_ax.margins(x=0.02, y=0.02)
    
    # Ensure consistent heights across subplots (avoid forcing square aspect)
    
    # Set axis labels and title
    axs[0].set_xlabel('Trials Back', fontsize=LABEL_FONTSIZE)
    axs[0].set_ylabel('Logistic Regression Coefficient', fontsize=LABEL_FONTSIZE)
    axs[0].tick_params(axis='both', labelsize=TICK_FONTSIZE)
    # Tighten left block: remove exterior whitespace and standardize placement
    for _ax in [RLRNN_ax, BG_ax, PFC_ax, asymmetry_ax, model_distance_ax, perf_hist_ax]:
        try:
            _ax.set_anchor('C')
        except Exception:
            pass
    for _ax in [BG_ax, PFC_ax]:
        _ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    
    # Update titles
    # Slightly lower the title to avoid overlapping square plots
    fig.suptitle('Figure 5: RLRNN Model Yields Monkey-Like Behavior', fontsize=26, y=0.975)
    titles = ['RLRNN Model', 'RL Module', 'RNN Module', 
              'Behavioral Asymmetry', 
              'Model Cosine Similarity from Monkey Behavior (All Components)',
              'Performance by Group (Hybrid/RNN/RL)']
    sizes = [20, 12, 12, 20, 16, 16]

    for i in range(len(axs)):
        if axs[i] is not None:
            if i == 3:
                axs[i].set_title(titles[i], fontsize=sizes[i], pad=18)
            elif i == 4:
                axs[i].set_title(titles[i], fontsize=MODEL_DISTANCE_TITLE_FONTSIZE, pad=6)
            else:
                axs[i].set_title(titles[i], fontsize=sizes[i], pad=8)
    
    # Add new combined visualization for model distances (frechet_comparison_plot merges RLRNN_500 internally)
    model_distance_ax, extreme_models = frechet_comparison_plot(
        model_distance_ax, strategic=strategic, fit_single=fit_single, use_performance=perf,
        use_median=use_median, power_scale=power_scale, distance_metric=distance_metric,
        use_subset=use_subset, plot_examples=make_extra_figures)
    if perf:
        model_distance_ax.set_title("Model Sequence Matching \n of Monkey Behavior", fontsize=16)
    else:
        subset_label = "Win-Stay/Lose-Switch" if use_subset == 'ws_ls' else "All Components" 
        name = 'Cosine Similarity' if distance_metric=='cosine' else get_distance_label(distance_metric)
        model_distance_ax.set_title(f"Model {name} \n from Monkey Behavior ({subset_label})", fontsize=MODEL_DISTANCE_TITLE_FONTSIZE)
    # Increase tick label size for the model distance axis
    try:
        model_distance_ax.tick_params(axis='both', labelsize=MODEL_DISTANCE_TICK_FONTSIZE)
    except Exception:
        pass
    # Ensure consistent heights across subplots (avoid forcing square aspect)
    
    # Plot extreme models for visualization
    if make_extra_figures and extreme_models:
        # Use the consolidated fits PKL; the helper ignores data_file but we keep the arg for signature
        plot_extreme_model_regressors(extreme_models, weight_supp_data_strategic, strategic=strategic, distance_metric=distance_metric)
    
    # Before building the histogram, summarize the (0.5,0) consolidated performance if present
    try:
        perf_summary = summarize_rlrnn_500_performance(bootstrap_iters=bootstrap_iters, random_state=cv_random_state)
        if perf_summary:
            s = perf_summary.get('strategic', {})
            ns = perf_summary.get('nonstrategic', {})
            print(f"RLRNN (0.5,0) 500-fits Strategic: mean={s.get('mean', float('nan')):.3f}, sem={s.get('sem', 0.0):.3f}, n_models={s.get('n_models', 0)}")
            print(f"RLRNN (0.5,0) 500-fits Non-strategic: mean={ns.get('mean', float('nan')):.3f}, sem={ns.get('sem', 0.0):.3f}, n_models={ns.get('n_models', 0)}")
    except Exception as e:
        print(f"Warning: failed to summarize RLRNN 500-fits performance: {e}")
    
    # Add performance histogram using data (now positioned in the far-right column)
    print("Calling performance histogram function...")
    # Force both groups to use all (0.5, 0) hybrid models for performance aggregation
    add_performance_histogram_from_data(
        perf_hist_ax,
        strategic=strategic,
        cv_folds=cv_folds,
        bootstrap_iters=bootstrap_iters,
        random_state=cv_random_state,
        override_strategic_param=(0.5, 0.0),
        override_nonstrategic_param=(0.5, 0.0),
    )
    print("Performance histogram function completed.")

    # MI calculations removed per request to speed up figure generation
    # (previous MI computation, weighting, and printing were here)
    
    # Prefer constrained layout only if explicitly requested
    if use_constrained_layout:
        try:
            fig.set_constrained_layout(True)
            fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06, hspace=0.06, wspace=0.055)
        except Exception:
            pass

    # If requested, export each main panel individually for presentations
    if export_asymmetry_progress:
        try:
            import os as _os
            from matplotlib.transforms import Bbox
            _single_dir = asymmetry_progress_dir
            if _single_dir is None and isinstance(img_path, str) and len(img_path) > 0:
                _single_dir = _os.path.join(_os.path.dirname(img_path), 'single_panels')
            _os.makedirs(_single_dir, exist_ok=True)
            # Save each axis as its own PNG
            panels = [
                (RLRNN_ax, 'panel_r lrnn_model.png'),
                (BG_ax, 'panel_rl_module.png'),
                (PFC_ax, 'panel_rnn_module.png'),
                (asymmetry_ax, 'panel_asymmetry.png'),
                (model_distance_ax, 'panel_model_distance.png'),
                (perf_hist_ax, 'panel_performance.png'),
            ]
            # Helper to save only the region occupied by the axis
            def _save_axis(fig, ax, path, dpi=600, pad=0.02):
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
                try:
                    renderer = fig.canvas.get_renderer()
                    bbox = ax.get_tightbbox(renderer).expanded(1.0 + pad, 1.0 + pad)
                    fig.savefig(path, bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()), dpi=dpi)
                except Exception as e:
                    # Fallback: save full fig if tightbbox route fails
                    fig.savefig(path, bbox_inches='tight', dpi=dpi)
                    print(f"Warning: fallback to full-figure save for {path}: {e}")

            for _ax, _name in panels:
                try:
                    p = _os.path.join(_single_dir, _name)
                    _save_axis(fig, _ax, p, dpi=600)
                    print(f"Saved panel: {p}")
                except Exception as e:
                    print(f"Warning: failed to save panel {_name}: {e}")
        except Exception as e:
            print(f"Warning: failed to export single panels: {e}")
    # Redundant safeguard: export asymmetry progress at the end as well
    if export_asymmetry_progress:
        try:
            _progress_dir = asymmetry_progress_dir
            try:
                if _progress_dir is None and isinstance(img_path, str) and len(img_path) > 0:
                    import os as _os
                    _progress_dir = _os.path.join(_os.path.dirname(img_path), 'asymmetry_progress')
            except Exception:
                pass
            export_asymmetry_progress_images(
                corr_list=corr_list,
                s_ls=s_ls,
                s_ws=s_ws,
                combined_regressors=combined_regressors,
                strategic=strategic,
                out_dir=_progress_dir,
                rl_point=(lsbg, wsbg)
            )
        except Exception as _e:
            print(f"Warning: failed to export asymmetry progress images (final): {_e}")
    return fig
    
    # Comment out the separate entropy/MI plots since we've integrated them
    # into the main figure
    '''
    # Create a bar plot comparing entropies
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    # plt.bar(['Monkeys', 'RLRNN', 'RNN', 'RL'], 
            # [weighted_monkey_entropy, RLRNN_entropy, RNN_entropy, RL_entropy])
    entropies = [*monkey_entropy, RLRNN_entropy, RNN_entropy, RL_entropy]
    labels = [*monkey_labels, 'RLRNN', 'RNN', 'RL']

    plt.xticks([0,1,2,3,4,5],list(range(8, 2,-1)))
    plt.xlabel('Sequence Length')
    for i in range(len(entropies)):
        if i < len(monkey_entropy): # make it dashed
            plt.plot(entropies[i], label = labels[i], linestyle = '--')
        else:
            plt.plot(entropies[i], label = labels[i])
    # plot theoretical max entropy
    # first generate two random sequences of length 100000
    seq1 = np.random.randint(0,2,100000)
    seq2 = np.random.randint(0,2,100000)
    ent = compute_entropy(seq1,seq2)
    plt.plot(ent[::-1], label = 'Max Entropy', linestyle = '--', color = 'k', alpha = .5)
    plt.title('Monkey Action Entropy')
    plt.legend()
    plt.ylabel('Entropy (bits)')
    
    # Create a bar plot comparing mutual information
    plt.subplot(1, 2, 2)
    mutual_informations = [*monkey_mi, RLRNN_mi, RNN_mi[::-1], RL_mi[::-1]]
    for i in range(len(mutual_informations)):
        if i < len(monkey_mi): # make it dashed
            plt.plot(mutual_informations[i], label = labels[i], linestyle = '--')
        else:
            plt.plot(mutual_informations[i], label = labels[i])
    # plot theoretical max mutual information
    # mi = compute_mutual_information(seq1,seq2)
    # plt.plot(mi[::-1], label = 'Theoretical Min Mutual Information', linestyle = '--', color = 'k', alpha = .5)
    plt.xticks([0,1,2,3,4,5],list(range(8, 2,-1)))
    plt.xlabel('Sequence Length')
    plt.title('Monkey-Computer Mutual Information')
    plt.ylabel('Mutual Information (bits)')
    plt.legend()
    plt.show()
    '''

def add_performance_histogram_from_data(ax, strategic=True, cv_folds=10, bootstrap_iters=1000, random_state=0,
                                        override_strategic_param=None, override_nonstrategic_param=None):
    """
    Add performance histogram using data from the frechet distance data file.
    Only uses (0.5,0) models for strategic monkeys and (0.2,0) models for non-strategic monkeys.
    """
    print("Starting performance histogram data loading...")
    # Load performance data from the frechet distance file
    data_dict = merge_rlrnn_500_into(load_all_fits_pkl())
    print(f"Loaded data_dict with {len(data_dict)} parameter combinations")
    
    # Define monkey groups
    strategic_monkeys = ['E', 'D', 'I']
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']
    
    # Initialize performance collections
    strategic_performances = []   # Hybrid (RLRNN) models matching strategic group
    nonstrategic_performances = []  # Hybrid models matching nonstrategic group
    
    # Only look at the specific parameter combinations we want
    strategic_param_found = False
    nonstrategic_param_found = False
    
    # Consistent parameter selection using cosine similarity (separate for each group)
    def _extract_fit(vec):
        arr = np.array(vec)
        return arr[:-1] if len(arr) > 20 else arr

    def _normalize_full_coeffs(arr):
        arr = np.array(arr, dtype=float)
        segments = len(arr) // 5
        if segments <= 0:
            nrm = np.linalg.norm(arr)
            return arr / (nrm if nrm != 0 else 1e-9)
        sums = []
        for i_seg in range(segments):
            s0, s1 = i_seg * 5, min((i_seg + 1) * 5, len(arr))
            if s1 > s0:
                seg = arr[s0:s1]
                sums.append(np.sum(np.abs(seg) ** 2))
        cmax = np.sqrt(np.sum(sums)) if sums else 1e-9
        return arr / (cmax if cmax != 0 else 1e-9)

    # Build monkey fits for strategic and nonstrategic groups
    mp2 = pd.read_pickle(stitched_p)
    mp2 = mp2[mp2['task'] == 'mp']

    def build_group_monkey_fits(monkeys):
        fits = {}
        for m in monkeys:
            mdat = mp2[mp2['animal'] == m]
            if len(mdat) == 0:
                continue
            res = paper_logistic_regression_strategic(None, False, data=mdat, order=5, bias=True, return_model=True)
            if res and 'action' in res:
                fits[m] = _normalize_full_coeffs(_extract_fit(res['action']))
        return fits

    fits_strat = build_group_monkey_fits(strategic_monkeys)
    fits_nonstrat = build_group_monkey_fits(nonstrategic_monkeys)

    def select_best_param_for_group(group_fits):
        best_param = None
        best_score = -float('inf')
        for p, fits in data_dict.items():
            scores = []
            for idx, fd in fits.items():
                if fd.get('avg_reward', 0) <= .45:
                    continue
                mod = _normalize_full_coeffs(_extract_fit(fd.get('action', [])))
                cos_vals = []
                for mf in group_fits.values():
                    cv = compute_distance(mod, mf, metric='cosine')
                    if not (np.isnan(cv) or np.isinf(cv)):
                        cos_vals.append(cv)
                if cos_vals:
                    scores.append(float(np.mean(cos_vals)))
            if scores:
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_param = p
        return best_param, best_score

    best_strat_param, strat_score = select_best_param_for_group(fits_strat)
    best_nonstrat_param, nonstrat_score = select_best_param_for_group(fits_nonstrat)

    # If overrides provided, prefer them (fall back to nearest available if exact not found)
    def _nearest_available(target):
        if target is None:
            return None
        try:
            tx, ty = float(target[0]), float(target[1])
        except Exception:
            return None
        avail = list(data_dict.keys())
        if not avail:
            return None
        best = None
        best_d = float('inf')
        for p in avail:
            dx = float(p[0]) - tx
            dy = float(p[1]) - ty
            d = (dx*dx + dy*dy) ** 0.5
            if d < best_d:
                best_d = d
                best = p
        return best

    if override_strategic_param is not None:
        if override_strategic_param in data_dict:
            print(f"Using override strategic param: {override_strategic_param}")
            best_strat_param = override_strategic_param
        else:
            near = _nearest_available(override_strategic_param)
            print(f"Strategic override {override_strategic_param} not found; using nearest available {near}")
            if near is not None:
                best_strat_param = near
    if override_nonstrategic_param is not None:
        if override_nonstrategic_param in data_dict:
            print(f"Using override non-strategic param: {override_nonstrategic_param}")
            best_nonstrat_param = override_nonstrategic_param
        else:
            near = _nearest_available(override_nonstrategic_param)
            print(f"Non-strategic override {override_nonstrategic_param} not found; using nearest available {near}")
            if near is not None:
                best_nonstrat_param = near

    print(f"Chosen param (strategic): {best_strat_param} (cos~={strat_score:.3f})")
    print(f"Chosen param (non-strategic): {best_nonstrat_param} (cos~={nonstrat_score:.3f})")

    # Compute performance for the chosen parameters
    def compute_group_performance(target_param, target_monkeys):
        perfs = []
        if target_param is None or target_param not in data_dict:
            return perfs
        fits = data_dict[target_param]
        param_key = f"{target_param[0]}, {target_param[1]:g}"
        for idx, fd in fits.items():
            if 'sequence_prediction' not in fd or fd.get('avg_reward', 0) <= .45:
                continue
            sp = fd['sequence_prediction']
            for m in target_monkeys:
                if m not in sp or sp[m] is None:
                    continue
                md = sp[m]
                if isinstance(md, dict) and param_key in md:
                    pdct = md[param_key]
                    if isinstance(pdct, dict):
                        mk = str(idx)
                        if mk in pdct:
                            arrs = pdct[mk]
                            if isinstance(arrs, list) and arrs:
                                import numpy as _np
                                tot = 0
                                n = 0
                                for a in arrs:
                                    if isinstance(a, _np.ndarray):
                                        tot += _np.sum(a)
                                        n += len(a)
                                if n > 0:
                                    perfs.append(tot / n)
        return perfs

    strategic_performances = compute_group_performance(best_strat_param, strategic_monkeys)
    nonstrategic_performances = compute_group_performance(best_nonstrat_param, nonstrategic_monkeys)
    strategic_param_found = best_strat_param is not None
    nonstrategic_param_found = best_nonstrat_param is not None
    
    # Calculate means and standard errors for Hybrid models
    print(f"Parameter search results: Strategic (0.5,0) found: {strategic_param_found}, Non-strategic (0.2,0) found: {nonstrategic_param_found}")
    print(f"Collected strategic performances: {len(strategic_performances)} values")
    print(f"Collected non-strategic performances: {len(nonstrategic_performances)} values")
    
    strategic_mean = np.mean(strategic_performances) if strategic_performances else 0.0
    nonstrategic_mean = np.mean(nonstrategic_performances) if nonstrategic_performances else 0.0

    # Bootstrap SEMs for Hybrid (RLRNN) using model-level distributions
    def _bootstrap_sem(values, n_boot, seed):
        if not values:
            return 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n == 0:
            return 0.0
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[i] = float(np.mean(arr[idx]))
        return float(np.std(means, ddof=1)) if n_boot > 1 else 0.0

    strategic_sem = _bootstrap_sem(strategic_performances, bootstrap_iters, random_state + 11)
    nonstrategic_sem = _bootstrap_sem(nonstrategic_performances, bootstrap_iters, random_state + 13)
    
    print(f"Strategic mean: {strategic_mean:.3f}, Non-strategic mean: {nonstrategic_mean:.3f}")
    
    # Also load RNN model performance from pre-aggregated summary
    rnn_strat_mean = None; rnn_strat_sem = 0.0; rnn_strat_n = 0
    rnn_nonstrat_mean = None; rnn_nonstrat_sem = 0.0; rnn_nonstrat_n = 0
    try:
        # Prefer the new consolidated RNN summary in fig5data
        new_summ_path = os.path.join(os.path.dirname(__file__), 'fig5data', 'rnn_sequence_prediction_summary.pkl')
        alt_summ_path = os.path.join(os.path.dirname(__file__), '..', 'cluster_scripts', 'RNN_Family', 'rnn_sequence_prediction_summary.pkl')
        summ_path = new_summ_path if os.path.exists(new_summ_path) else alt_summ_path
        with open(summ_path, 'rb') as f:
            summ = pickle.load(f)

        # New schema produced by compute_rnn_sequence_performance.py
        strat = summ.get('strategic_monkeys', {})
        nonstrat = summ.get('non_strategic_monkeys', {})

        # Strategic group
        rnn_strat_n = int(strat.get('n_models', 0))
        if rnn_strat_n > 0:
            rnn_strat_mean = float(strat.get('mean_accuracy', 0.0))
            # Prefer bootstrap over analytic SEM if per-model means available
            rnn_means_s_full = summ.get('aggregate', {}).get('strategic', {}).get('model_means', [])
            if rnn_means_s_full:
                rnn_strat_sem = _bootstrap_sem(rnn_means_s_full, bootstrap_iters, random_state + 17)
            else:
                std_s = float(strat.get('std_accuracy', 0.0))
                rnn_strat_sem = (std_s / (rnn_strat_n ** 0.5)) if rnn_strat_n > 0 else 0.0

        # Non-strategic group
        rnn_nonstrat_n = int(nonstrat.get('n_models', 0))
        if rnn_nonstrat_n > 0:
            rnn_nonstrat_mean = float(nonstrat.get('mean_accuracy', 0.0))
            rnn_means_ns_full = summ.get('aggregate', {}).get('nonstrategic', {}).get('model_means', [])
            if rnn_means_ns_full:
                rnn_nonstrat_sem = _bootstrap_sem(rnn_means_ns_full, bootstrap_iters, random_state + 19)
            else:
                std_ns = float(nonstrat.get('std_accuracy', 0.0))
                rnn_nonstrat_sem = (std_ns / (rnn_nonstrat_n ** 0.5)) if rnn_nonstrat_n > 0 else 0.0
    except Exception:
        # Keep None -> bars will be skipped
        pass

    # Compute and cache RL group performance using session-level K-fold CV with bootstrap
    rl_cache_path = os.path.join(os.path.dirname(__file__), 'fig5data', 'rl_group_sequence_performance.pkl')
    rl_group_perf = None
    try:
        if os.path.exists(rl_cache_path):
            with open(rl_cache_path, 'rb') as f:
                rl_group_perf = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load RL group performance cache: {e}")
        rl_group_perf = None

    def build_group_episodes(df, monkeys):
        ep_actions = []
        ep_rewards = []
        sub = df[df['animal'].isin(monkeys)]
        for sid, s in sub.groupby('id'):
            a = s['monkey_choice'].to_numpy()
            r = s['reward'].to_numpy()
            if len(a) > 1:
                ep_actions.append(a)
                ep_rewards.append(r)
        return ep_actions, ep_rewards

    needs_recompute = (
        rl_group_perf is None or
        rl_group_perf.get('model', '') != 'simple_cv' or
        int(rl_group_perf.get('n_folds', -1)) != int(cv_folds) or
        int(rl_group_perf.get('n_bootstrap', -1)) != int(bootstrap_iters) or
        int(rl_group_perf.get('random_state', -1)) != int(random_state)
    )

    if needs_recompute:
        print(f"Computing RL group performance (simple RL CV) with {cv_folds}-fold CV and {bootstrap_iters} bootstrap iters ...")
        mp2_df = pd.read_pickle(stitched_p)
        mp2_df = mp2_df[mp2_df['task'] == 'mp']
        # Strategic group episodes
        ep_a_s, ep_r_s = build_group_episodes(mp2_df, strategic_monkeys)
        cv_s = cross_validated_performance_sessions(
            ep_a_s, ep_r_s,
            model='simple', n_folds=cv_folds, random_state=random_state,
            punitive=False, decay=False, const_beta=False, const_gamma=True,
            disable_abs=False, n_bootstrap=bootstrap_iters, greedy=True
        )
        # Non-strategic group episodes
        ep_a_ns, ep_r_ns = build_group_episodes(mp2_df, nonstrategic_monkeys)
        cv_ns = cross_validated_performance_sessions(
            ep_a_ns, ep_r_ns,
            model='simple', n_folds=cv_folds, random_state=random_state,
            punitive=False, decay=False, const_beta=False, const_gamma=True,
            disable_abs=False, n_bootstrap=bootstrap_iters, greedy=True
        )
        rl_group_perf = {
            'model': 'simple_cv',
            'n_folds': int(cv_folds),
            'n_bootstrap': int(bootstrap_iters),
            'random_state': int(random_state),
            'strategic': {
                'mean_accuracy': float(cv_s.get('mean_accuracy', 0.0)),
                'bootstrap_sem': float(cv_s.get('bootstrap_sem', 0.0)),
                'bootstrap_means': cv_s.get('bootstrap_means', []),
                'per_session_accuracy': cv_s.get('per_session_accuracy', []),
            },
            'nonstrategic': {
                'mean_accuracy': float(cv_ns.get('mean_accuracy', 0.0)),
                'bootstrap_sem': float(cv_ns.get('bootstrap_sem', 0.0)),
                'bootstrap_means': cv_ns.get('bootstrap_means', []),
                'per_session_accuracy': cv_ns.get('per_session_accuracy', []),
            }
        }
        try:
            os.makedirs(os.path.join(os.path.dirname(__file__), 'fig5data'), exist_ok=True)
            with open(rl_cache_path, 'wb') as f:
                pickle.dump(rl_group_perf, f)
            print(f"Saved RL group CV performance cache to: {rl_cache_path}")
        except Exception as e:
            print(f"Warning: failed to save RL group CV performance cache: {e}")
    else:
        print("Loaded RL group CV performance from cache")

    rl_strat_mean = float(rl_group_perf.get('strategic', {}).get('mean_accuracy', 0.0)) if rl_group_perf else 0.0
    rl_nonstrat_mean = float(rl_group_perf.get('nonstrategic', {}).get('mean_accuracy', 0.0)) if rl_group_perf else 0.0
    rl_strat_sem = float(rl_group_perf.get('strategic', {}).get('bootstrap_sem', 0.0)) if rl_group_perf else 0.0
    rl_nonstrat_sem = float(rl_group_perf.get('nonstrategic', {}).get('bootstrap_sem', 0.0)) if rl_group_perf else 0.0

    # Build grouped bars by group: each group shows Hybrid, RNN, and RL (RL via CV)
    groups = ['Strategic', 'Non-strategic']
    model_names = ['Hybrid', 'RNN', 'RL']
    x = np.arange(len(groups))
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = {'Hybrid': 'tab:olive', 'RNN': 'tab:red', 'RL': 'tab:blue'}

    # Prepare means and sems per group/model using all (0.5,0) hybrid models for both groups
    def _bootstrap_sem(values, n_boot, seed):
        if not values:
            return 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(values, dtype=float)
        n = len(arr)
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            means[i] = float(np.mean(arr[idx]))
        return float(np.std(means, ddof=1))
    strategic_mean = float(np.mean(strategic_performances)) if strategic_performances else np.nan
    nonstrategic_mean = float(np.mean(nonstrategic_performances)) if nonstrategic_performances else np.nan
    strategic_sem = _bootstrap_sem(strategic_performances, bootstrap_iters, random_state+101) if strategic_performances else 0.0
    nonstrategic_sem = _bootstrap_sem(nonstrategic_performances, bootstrap_iters, random_state+103) if nonstrategic_performances else 0.0
    group_model_means = {
        'Strategic': [strategic_mean, rnn_strat_mean if rnn_strat_mean is not None else np.nan, rl_strat_mean],
        'Non-strategic': [nonstrategic_mean, rnn_nonstrat_mean if rnn_nonstrat_mean is not None else np.nan, rl_nonstrat_mean]
    }
    group_model_sems = {
        'Strategic': [strategic_sem, rnn_strat_sem if rnn_strat_mean is not None else 0.0, rl_strat_sem],
        'Non-strategic': [nonstrategic_sem, rnn_nonstrat_sem if rnn_nonstrat_mean is not None else 0.0, rl_nonstrat_sem]
    }

    for mi, mname in enumerate(model_names):
        means = [group_model_means[g][mi] for g in groups]
        sems = [group_model_sems[g][mi] for g in groups]
        # Skip RNN if missing
        if mname == 'RNN' and (rnn_strat_mean is None and rnn_nonstrat_mean is None):
            continue
        ax.bar(x + offsets[mi], means, width, yerr=sems, alpha=0.85, capsize=4, color=colors[mname], label=mname)

    # (Logistic regression upper-bound overlay added later after significance bars)

    # Styling
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Mean Sequence Prediction Accuracy', fontsize=12)
    title = 'Performance by Group (Hybrid/RNN/RL)\nHybrid params: (0.5,0) for both groups'
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(frameon=False, fontsize=10)

    # Set y-axis limits to show meaningful range
    ax.set_ylim(0.45, 0.68)

    # Add horizontal line at chance level (0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Significance bars: within-group pairwise comparisons (Hybrid vs RNN, Hybrid vs RL, RNN vs RL)
    # We retrieve RNN distributions from the summary if present
    rnn_means_s = []
    rnn_means_ns = []
    try:
        new_summ_path = os.path.join(os.path.dirname(__file__), 'fig5data', 'rnn_sequence_prediction_summary.pkl')
        alt_summ_path = os.path.join(os.path.dirname(__file__), '..', 'cluster_scripts', 'RNN_Family', 'rnn_sequence_prediction_summary.pkl')
        _summ_path = new_summ_path if os.path.exists(new_summ_path) else alt_summ_path
        with open(_summ_path, 'rb') as _f:
            _s = pickle.load(_f)
        # try to read lists of per-model means if available
        rnn_means_s = _s.get('aggregate', {}).get('strategic', {}).get('model_means', [])
        rnn_means_ns = _s.get('aggregate', {}).get('nonstrategic', {}).get('model_means', [])
    except Exception:
        pass

    def draw_sig(ax, x1, x2, y, pval, h=0.01, text_offset=0.005):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color='k', linewidth=1)
        if pval is None:
            label = 'ns'
        else:
            if pval < 0.001:
                label = '***'
            elif pval < 0.01:
                label = '**'
            elif pval < 0.05:
                label = '*'
            else:
                label = 'ns'
        ax.text((x1+x2)/2, y+h+text_offset, label, ha='center', va='bottom', fontsize=10)

    # RL distributions from CV bootstrap
    rl_boot_s = rl_group_perf.get('strategic', {}).get('bootstrap_means', []) if rl_group_perf else []
    rl_boot_ns = rl_group_perf.get('nonstrategic', {}).get('bootstrap_means', []) if rl_group_perf else []

    ymax = ax.get_ylim()[1]
    # Strategic comparisons
    if strategic_performances and rnn_means_s:
        try:
            pv = ttest_ind(np.array(strategic_performances), np.array(rnn_means_s), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[0] + offsets[0], x[0] + offsets[1], ymax - 0.03, pv)
    if strategic_performances and rl_boot_s:
        try:
            pv = ttest_ind(np.array(strategic_performances), np.array(rl_boot_s), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[0] + offsets[0], x[0] + offsets[2], ymax - 0.06, pv)
    if rnn_means_s and rl_boot_s:
        try:
            pv = ttest_ind(np.array(rnn_means_s), np.array(rl_boot_s), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[0] + offsets[1], x[0] + offsets[2], ymax - 0.09, pv)

    # Non-strategic comparisons
    if nonstrategic_performances and rnn_means_ns:
        try:
            pv = ttest_ind(np.array(nonstrategic_performances), np.array(rnn_means_ns), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[1] + offsets[0], x[1] + offsets[1], ymax - 0.03, pv)
    if nonstrategic_performances and rl_boot_ns:
        try:
            pv = ttest_ind(np.array(nonstrategic_performances), np.array(rl_boot_ns), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[1] + offsets[0], x[1] + offsets[2], ymax - 0.06, pv)
    if rnn_means_ns and rl_boot_ns:
        try:
            pv = ttest_ind(np.array(rnn_means_ns), np.array(rl_boot_ns), equal_var=False).pvalue
        except Exception:
            pv = None
        draw_sig(ax, x[1] + offsets[1], x[1] + offsets[2], ymax - 0.09, pv)

    # Plot-only hook for logistic baseline lines (values should be provided externally)
    def plot_logistic_upper_bound_lines(ax, ub_s, ub_ns, x, offsets, width):
        cluster_min = min(offsets) - width/2
        cluster_max = max(offsets) + width/2
        ymin, ymax = ax.get_ylim()
        for gi, ub in enumerate([ub_s, ub_ns]):
            if ub is None or (isinstance(ub, float) and np.isnan(ub)):
                continue
            yval = float(ub)
            if yval > ymax:
                ymax = yval + 0.01
            ax.hlines(y=yval, xmin=x[gi] + cluster_min, xmax=x[gi] + cluster_max,
                      colors='mediumslateblue', linestyles='--', linewidth=1.6, zorder=3, label=None)
        ax.set_ylim(ymin, max(ymax, ax.get_ylim()[1]))
        handles, labels = ax.get_legend_handles_labels()
        if 'Logistic (CV) baseline' not in labels:
            from matplotlib.lines import Line2D
            ub_handle = Line2D([0], [0], color='mediumslateblue', linestyle='--', linewidth=1.6, label='Logistic (CV) baseline')
            ax.legend(handles + [ub_handle], labels + ['Logistic (CV) baseline'], frameon=False, fontsize=10)

    # Compute out-of-sample LR upper bounds (per-monkey 10-fold CV) and draw dashed lines
    try:
        mp_df = pd.read_pickle(stitched_p)
        mp_df = mp_df[mp_df['task'] == 'mp']
        strat_monkeys = ['E', 'D', 'I']
        nonstrat_monkeys = ['C', 'H', 'F', 'K']
        try:
            _cvrs = int(locals().get('cv_random_state', 0))
        except Exception:
            _cvrs = 0
        res_s = cross_validated_performance_by_monkey_df(mp_df, strat_monkeys, model='simple', n_folds=cv_folds, random_state=_cvrs)
        res_ns = cross_validated_performance_by_monkey_df(mp_df, nonstrat_monkeys, model='simple', n_folds=cv_folds, random_state=_cvrs)
        ub_s = res_s.get('LR_mean_accuracy', None)
        ub_ns = res_ns.get('LR_mean_accuracy', None)
        plot_logistic_upper_bound_lines(ax, ub_s, ub_ns, x, offsets, width)
        # Add LR labels next to dashed lines
        if ub_s is not None:
            ax.text(x[0], ub_s + 0.002, f"LR {ub_s:.3f}", ha='center', va='bottom', fontsize=10, color='mediumslateblue')
        if ub_ns is not None:
            ax.text(x[1], ub_ns + 0.002, f"LR {ub_ns:.3f}", ha='center', va='bottom', fontsize=10, color='mediumslateblue')
    except Exception as e:
        print(f"Warning: failed to compute LR CV upper bounds for dashed lines: {e}")

def weighing_comparison_plot(ax, comparison_ax=None, strategic=False):
    # load data from new PKL
    param_tuple, fits_dict = load_all_fits_pkl()
    # ... rest of the function ...
    # When iterating over models, use:
    # for idx, fit_data in fits_dict.items():
    #     fit = fit_data['action']
    #     ...
    
def weighing_comparison_plot_strategic(ax, comparison_ax=None):
    # load data from new PKL
    param_tuple, fits_dict = load_all_fits_pkl()
    # ... rest of the function ...
    # When iterating over models, use:
    # for idx, fit_data in fits_dict.items():
    #     fit = fit_data['action']
    #     ...

def plot_most_similar_models(monkey_similarity_tracker, strategic=True, use_performance=False, distance_metric='euclidean', use_block_level=False):
    """
    Plot logistic regression coefficients for the models most similar to each monkey.
    
    Args:
        monkey_similarity_tracker: Dictionary tracking most similar models for each monkey
        strategic: Whether using strategic regression analysis
        use_performance: Whether using performance-based similarity
        distance_metric: Distance metric used for similarity
    """
    # Filter out monkeys with no similar models found
    valid_monkeys = [monkey for monkey in monkey_similarity_tracker 
                    if monkey_similarity_tracker[monkey]['fit'] is not None]
    
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']
    strategic_monkeys = ['E', 'D', 'I']
    
    if not valid_monkeys:
        print("No valid similar models found for any monkeys")
        return
    
    # Set up subplot grid - use 2 columns, calculate rows needed
    n_monkeys = len(valid_monkeys)
    n_cols = min(3, n_monkeys)  # Max 3 columns for readability
    n_rows = (n_monkeys + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), dpi=600, gridspec_kw={'wspace': 0.2, 'hspace': 0.55})
    
    # Handle single subplot case
    if n_monkeys == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Set figure title based on similarity metric
    similarity_type = "Performance" if use_performance else get_distance_label(distance_metric)
    scope = "Block-Level" if use_block_level else "Averaged"
    fig.suptitle(f'Most Similar Models to Each Monkey ({similarity_type}, {scope})', fontsize=16)
    
    # Define regressor labels and colors based on strategic flag
    if strategic:
        regressor_labels = ['Win Stay', 'Lose Switch', 'Win Switch', 'Lose Stay']
    else:
        regressor_labels = ['Agent Choice', 'Win Stay', 'Lose Switch']
    
    # Get matplotlib color cycle to match normal logistic regression plots
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    order = 5  # Coefficient order (trials back)
    
    for idx, monkey in enumerate(valid_monkeys):
        # Calculate subplot position
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Get model data for this monkey
        if use_block_level and monkey_similarity_tracker[monkey]['best_block_model_fit'] is not None:
            model_fit = monkey_similarity_tracker[monkey]['best_block_model_fit']
            monkey_fit = monkey_similarity_tracker[monkey]['best_block_monkey_fit']
            model_weight = monkey_similarity_tracker[monkey]['best_block_weight']
            model_idx = monkey_similarity_tracker[monkey]['best_block_model_idx']
            similarity_value = monkey_similarity_tracker[monkey]['best_block_similarity']
        else:
            model_fit = monkey_similarity_tracker[monkey]['fit']
            monkey_fit = monkey_similarity_tracker[monkey]['monkey_fit']
            model_weight = monkey_similarity_tracker[monkey]['weight']
            model_idx = monkey_similarity_tracker[monkey]['idx']
            similarity_value = monkey_similarity_tracker[monkey]['similarity']
        
        # Do not filter by parameter tuple; plot whichever model best matches this monkey
        
        # Create x-axis for trials back
        x_trials = np.arange(1, order + 1)
        ax.set_xticks(range(1, order + 1))
        
        # Plot each regressor type
        n_regressors = len(regressor_labels)
        for reg_idx in range(n_regressors):
            start_idx = reg_idx * order
            end_idx = (reg_idx + 1) * order
            
            # Plot model coefficients (solid line)
            if end_idx <= len(model_fit):
                model_coeffs = model_fit[start_idx:end_idx]
                ax.plot(x_trials, model_coeffs, 
                       label=f'Model: {regressor_labels[reg_idx]}',
                       color=colors[reg_idx % len(colors)], 
                       linewidth=2.5, alpha=0.8)
            
            # Plot monkey coefficients (dashed line) 
            if end_idx <= len(monkey_fit):
                monkey_coeffs = monkey_fit[start_idx:end_idx]
                ax.plot(x_trials, monkey_coeffs, 
                       label=f'Monkey: {regressor_labels[reg_idx]}',
                       color=colors[reg_idx % len(colors)], 
                       linewidth=2, linestyle='--', alpha=0.7)
        
        # Add horizontal line at y=0
        ax.axhline(0, linestyle='-', color='black', alpha=0.3, linewidth=0.8)
        
        # Format title with model parameters and similarity
        if use_performance:
            title = f'Monkey {monkey}\nModel {model_weight} (idx: {model_idx})\nPerformance: {similarity_value:.3f}'
        else:
            title = f'Monkey {monkey}\nModel {model_weight} (idx: {model_idx})\n{get_distance_label(distance_metric)}: {similarity_value:.3f}'
        ax.set_title(title, fontsize=12, pad=10)
        
        # Set axis labels
        ax.set_xlabel('Trials Back', fontsize=11)
        ax.set_ylabel('Logistic Regression Coefficient', fontsize=11)
        
        # Add legend, but only show one set (model or monkey) to avoid clutter
        handles, labels = ax.get_legend_handles_labels()
        # Keep only model coefficients in legend for clarity
        model_handles = [h for h, l in zip(handles, labels) if 'Model:' in l]
        model_labels = [l.replace('Model: ', '') for l in labels if 'Model:' in l]
        ax.legend(model_handles, model_labels, fontsize=9, frameon=True, loc='best')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Make y-axis symmetric around 0 for better comparison
        ylim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_ylim(-ylim*1.1, ylim*1.1)
    
    # Hide empty subplots if any
    total_subplots = n_rows * n_cols
    for idx in range(n_monkeys, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    # Add a bit more vertical breathing room to avoid title/xlabel collisions
    plt.tight_layout(h_pad=1.8, rect=[0, 0.03, 1, 0.96])
    
    # Print summary information
    print(f"\nMost Similar Models Summary:")
    print("=" * 50)
    for monkey in valid_monkeys:
        model_weight = monkey_similarity_tracker[monkey]['weight']
        model_idx = monkey_similarity_tracker[monkey]['idx']
        similarity_value = monkey_similarity_tracker[monkey]['similarity']
        
        if use_performance:
            print(f"Monkey {monkey}: Model {model_weight} (idx: {model_idx}) - Performance: {similarity_value:.4f}")
        else:
            print(f"Monkey {monkey}: Model {model_weight} (idx: {model_idx}) - {get_distance_label(distance_metric)}: {similarity_value:.4f}")

# Add a new function for the updated visualization using Frechet distance
def frechet_comparison_plot(ax, strategic = True, fit_single=False, use_performance=False, use_median=False, power_scale=1, distance_metric='euclidean', use_subset='all', plot_examples=False, plot_overlays=True):
    """
    Plot comparison of model distances from monkey behavior.
    
    Args:
        ax: Matplotlib axis for plotting
        strategic: Whether to use strategic regression analysis
        fit_single: Whether to fit single sessions (deprecated)
        use_performance: Whether to use performance data instead of distance metrics
        use_median: Whether to use median instead of mean for aggregation
        power_scale: Power scaling factor (deprecated)
        distance_metric: Distance metric to use ('euclidean', 'cosine', etc.)
        use_subset: Which regression components to use ('all', 'ws_ls' for win-stay/lose-switch only)
    """
    # load data from new PKL
    data = load_all_fits_pkl()
    # model_names = list(fits_dict.keys())
    
    
    
    # load monkey data and categorize into strategic and non-strategic monkeys
    nonstrategic_monkeys = ['C', 'H', 'F', 'K']
    strategic_monkeys = ['E', 'D', 'I']
    
    # Initialize similarity tracking for all monkeys (works for both performance and distance modes)
    monkey_similarity_tracker = {}
    for monkey in nonstrategic_monkeys + strategic_monkeys:
        if use_performance:
            initial_value = -float('inf')  # For performance, higher is better
        elif distance_metric == 'cosine':
            initial_value = -float('inf')  # For cosine similarity, higher is better
        else:
            initial_value = float('inf')   # For distance metrics, lower is better
        monkey_similarity_tracker[monkey] = {
            'idx': None, 
            'weight': None, 
            'fit': None,
            'similarity': initial_value, 
            'monkey_fit': None,
            # block-level best tracking
            'best_block_model_idx': None,
            'best_block_weight': None,
            'best_block_model_fit': None,
            'best_block_monkey_fit': None,
            'best_block_similarity': initial_value,
            'best_block_idx': None
        }
    
    # load mp2 stitched
    mp2_data = pd.read_pickle(stitched_p)
    mp2_data = mp2_data[mp2_data['task'] == 'mp']
    
    # Process all models to get their names
    # for datum in data:
    #     model_names.append(_ensure_param_tuple(datum))
    model_names = list(sorted(set(data.keys())))
    
    # Function to extract relevant coefficients based on use_subset
    def extract_coefficients(fit_coeffs, strategic_mode, subset_mode):
        """Extract relevant coefficient subset based on parameters."""
        if subset_mode == 'ws_ls':
            if strategic_mode:
                # Strategic: win-stay (0:5) + lose-switch (5:10) = first 10 coefficients
                return fit_coeffs[:10]
            else:
                # Non-strategic: win-stay (5:10) + lose-switch (10:15)
                return fit_coeffs[5:15]
        else:  # 'all'
            if strategic_mode:
                # Strategic: exclude bias term (last coefficient)
                return fit_coeffs[:-1] if len(fit_coeffs) > 20 else fit_coeffs
            else:
                # Non-strategic: use all coefficients
                return fit_coeffs
    
    if use_performance:
        # Use performance data instead of distance calculation
        print("Using performance mode for model evaluation (including RLRNN_500 if present)...")
        
        # Initialize dictionaries for strategic and non-strategic performance
        model_perf_dict = {'strategic': {}, 'nonstrategic': {}}
        
        # Function to get performance for a model
        def get_performance_for_model(model_name, target_monkeys):
            # data dict is structured as follows:
            # model_params : {model_idx : {sequence_prediction : {monkey : performance}}}
            # so we need to get the model_idx for the given model_name
            # so loop over all idxs in data to aggregate performance for each monkey, assuming
            # that 'avg_reward' > .45 for each model
            nonlocal data
            model_perf = []
            
            for idx in data[model_name]:
                # if data[model_name][idx]['avg_reward'] > .45:
                if data[model_name][idx]['avg_reward'] > .45:
                    for monkey in target_monkeys:
                        performance = data[model_name][idx]['sequence_prediction'][monkey]
                        if performance is not None:
                            model_perf.append(performance)
            return model_perf
        
        # Performance mode: similarity values will be updated during model evaluation
        
        # Compute performance for each model and track best for each monkey
        # Only look at specific parameter combinations: (0.5,0) for strategic, (0.2,0) for non-strategic
        for model in model_names:
            # Skip models that aren't the ones we want
            if model not in [(0.5, 0), (0.2, 0)]:
                continue
                
            for idx in data[model]:
                if data[model][idx]['avg_reward'] > .45:
                    # Determine which monkeys to check based on model parameters
                    if model == (0.5, 0):
                        target_monkeys = strategic_monkeys
                    elif model == (0.2, 0):
                        target_monkeys = nonstrategic_monkeys
                    else:
                        continue
                    
                    # Check performance for each monkey in the target group
                    for monkey in target_monkeys:
                        if 'sequence_prediction' in data[model][idx] and monkey in data[model][idx]['sequence_prediction']:
                            performance = data[model][idx]['sequence_prediction'][monkey]
                            # For performance mode, higher is better (similarity, not distance)
                            if performance is not None and performance > monkey_similarity_tracker[monkey]['similarity']:
                                # Get the model fit for this specific model instance
                                model_fit = data[model][idx]['action'] if 'action' in data[model][idx] else None
                                if model_fit is not None:
                                    # Extract relevant coefficients based on use_subset
                                    subset_fit = extract_coefficients(model_fit, strategic, use_subset)
                                    
                                    monkey_similarity_tracker[monkey]['idx'] = idx
                                    monkey_similarity_tracker[monkey]['weight'] = model
                                    monkey_similarity_tracker[monkey]['fit'] = subset_fit  # Model's coefficients
                                    monkey_similarity_tracker[monkey]['monkey_fit'] = None  # Will be filled later with actual monkey data
                                    monkey_similarity_tracker[monkey]['similarity'] = performance
            
            # Also compute aggregate performance for visualization
            if model == (0.5, 0):
                strategic_perf = get_performance_for_model(model, strategic_monkeys)
                model_perf_dict['strategic'][model] = strategic_perf
            elif model == (0.2, 0):
                nonstrategic_perf = get_performance_for_model(model, nonstrategic_monkeys)
                model_perf_dict['nonstrategic'][model] = nonstrategic_perf
        
        # Note: full-sweep aggregation removed in favor of targeted params and merged RLRNN_500
        
        # Load monkey regression data for performance mode comparison at monkey level
        for monkey in nonstrategic_monkeys + strategic_monkeys:
            if monkey_similarity_tracker[monkey]['fit'] is not None:
                # Load monkey data to get the actual regression fit
                mdat = mp2_data[mp2_data['animal'] == monkey]
                if len(mdat) > 0:
                    # Apply session-based cutoff like in fig1
                    if monkey == 'F' or monkey == 'E' or monkey == 'C':
                        mdat = cutoff_trials_by_session(mdat, 5000)
                    
                    # load session fits fro
                    try:
                        if strategic:
                            result = paper_logistic_regression_strategic(None, False, data=mdat, order=5, bias=True, return_model=True)
                            monkey_fit = result['action'][:-1] if result is not None and 'action' in result else None
                        else:
                            result = paper_logistic_regression(None, False, data=mdat, order=5, bias=True, return_model=True)
                            monkey_fit = result['action'] if result is not None and 'action' in result else None
                        
                        if monkey_fit is not None:
                            # Extract relevant coefficients based on use_subset
                            subset_fit = extract_coefficients(monkey_fit, strategic, use_subset)
                            monkey_similarity_tracker[monkey]['monkey_fit'] = subset_fit
                            print(f'Performance mode: Fitted monkey {monkey} with {len(mdat)} trials at monkey level')
                    except Exception as e:
                        print(f"Error fitting monkey {monkey} in performance mode: {e}")
                        continue

        # Extract data for visualization
        x_perf = []
        y_perf = []
        msarr = []
        mnsarr = []
        for model in model_names:
            if model in model_perf_dict['strategic'] and model in model_perf_dict['nonstrategic']:
                ms = model_perf_dict['strategic'][model]
                mns = model_perf_dict['nonstrategic'][model]
                msarr.append(ms)
                mnsarr.append(mns)
                x_perf.append(model[0])
                y_perf.append(model[1])
        
        print(f"Performance mode: Found {len(x_perf)} models with both strategic and non-strategic data")
        print(f"Strategic performance range: {np.nanmin(msarr):.3f} - {np.nanmax(msarr):.3f}")
        print(f"Non-strategic performance range: {np.nanmin(mnsarr):.3f} - {np.nanmax(mnsarr):.3f}")
        
        # Convert to numpy arrays - performance values are already in correct format
        msarr = np.array(msarr)
        mnsarr = np.array(mnsarr)
        
        # For performance, higher values are better, so no need to invert
        # Normalize to 0-1 range for consistent visualization
        if msarr.size > 0:
            min_msarr, max_msarr = np.nanmin(msarr), np.nanmax(msarr)
            if max_msarr - min_msarr != 0:
                msarr = (msarr - min_msarr) / (max_msarr - min_msarr)
            else:
                msarr = np.full_like(msarr, 0.5, dtype=float)

        if mnsarr.size > 0:
            min_mnsarr, max_mnsarr = np.nanmin(mnsarr), np.nanmax(mnsarr)
            if max_mnsarr - min_mnsarr != 0:
                mnsarr = (mnsarr - min_mnsarr) / (max_mnsarr - min_mnsarr)
            else:
                mnsarr = np.full_like(mnsarr, 0.5, dtype=float)
        
        # Set x and y for the final plotting section
        x = x_perf
        y = y_perf
        
    else:
        # Distance mode: similarity tracker was already initialized above
        # Use distance calculation (original behavior) and include RLRNN_500
        subset_label = "Win-Stay/Lose-Switch" if use_subset == 'ws_ls' else "All Components"
        print(f"Using distance mode for model evaluation with {subset_label} (including RLRNN_500 if present)...")
        trial_counts = {}

        # Build per-block fits for every monkey, plus an averaged fit (weighted by block length)
        def compute_block_fits(df):
            parts, lengths = partition_dataset(df, 5000)
            block_fits = []
            for pdata in parts:
                if strategic:
                    res = paper_logistic_regression_strategic(None, False, data=pdata, order=5, bias=True, return_model=True)
                    fit_vec = res['action'][:-1] if res is not None and 'action' in res else None
                else:
                    res = paper_logistic_regression(None, False, data=pdata, order=5, bias=True, return_model=True)
                    fit_vec = res['action'] if res is not None and 'action' in res else None
                if fit_vec is None:
                    continue
                block_fits.append(extract_coefficients(fit_vec, strategic, use_subset))
            return block_fits, lengths
        
        monkey_data_dict = {}
        for monkey in nonstrategic_monkeys + strategic_monkeys:
            mdat = mp2_data[mp2_data['animal'] == monkey]
            if len(mdat) == 0:
                continue
            if monkey in ['F', 'E', 'C']:
                    print(f'Applying session-based cutoff of 5000 trials for monkey {monkey}')
                    mdat = cutoff_trials_by_session(mdat, 5000)
            block_fits, block_lengths = compute_block_fits(mdat)
            if not block_fits:
                continue
            # Weighted average fit across blocks (for model-matching visualization)
            weights = np.array(block_lengths[:len(block_fits)], dtype=float)
            weights[weights <= 0] = 1.0
            avg_fit = np.average(np.stack(block_fits, axis=0), axis=0, weights=weights[:len(block_fits)])
            mtype = 'strategic' if monkey in strategic_monkeys else 'nonstrategic'
            monkey_data_dict[monkey] = {
                'type': mtype,
                'blocks': block_fits,
                'block_lengths': weights[:len(block_fits)],
                'avg_fit': avg_fit
            }
            trial_counts[monkey] = int(np.sum(weights))
            print(f'Monkey {monkey}: {len(block_fits)} blocks, {trial_counts[monkey]} trials (post-cutoff)')

        
        # Initialize dictionaries for strategic and non-strategic distances
        model_frechet_dict = {'strategic': {}, 'nonstrategic': {}}
        
        # Compute distances for each model
        high_param_models_found = 0
        high_param_models_passed = 0
        
        tcs_ns = []
        tcs_s = []
        for model in model_names:
            model_fits = []
            for datum in data[model]:
                if data[model][datum]['avg_reward'] > 0.45:
                    fit = data[model][datum]['action']
                    subset_fit = extract_coefficients(fit, strategic, use_subset)
                    model_fits.append(subset_fit)

            # Accumulate weighted distances for this parameter across both groups
            sum_w_s = 0.0
            sum_dw_s = 0.0
            sum_w_ns = 0.0
            sum_dw_ns = 0.0

            for i in range(len(model_fits)):
                model_data = model_fits[i]
                
                # Normalize model data based on subset choice
                if use_subset == 'ws_ls':
                    norm = np.linalg.norm(model_data)
                    if norm == 0:
                        norm = 1e-9
                    model_data = model_data / norm
                else:
                    if strategic:
                        cs = []
                        expected_segments = len(model_data) // 5
                        for i_seg in range(expected_segments):
                            start_idx = i_seg * 5
                            end_idx = min((i_seg + 1) * 5, len(model_data))
                            if end_idx > start_idx:
                                segment = model_data[start_idx:end_idx]
                                cs.append(np.sum(np.abs(segment) ** 2))
                        if cs:
                            cmax = np.sqrt(np.sum(cs))
                            if cmax == 0:
                                cmax = 1e-9
                            model_data = model_data / cmax
                    else:
                        norm = np.linalg.norm(model_data)
                        if norm == 0:
                            norm = 1e-9
                        model_data = model_data / norm
                                    
                # Distances to each monkey, averaged over that monkey's blocks
                for monkey, m_data in monkey_data_dict.items():
                    block_fits = m_data['blocks']
                    block_weights = m_data['block_lengths']
                    # Compute weighted average distance across blocks for this monkey
                    dsum = 0.0
                    wsum = 0.0
                    # Also build a weighted-average normalized monkey fit for plotting later
                    # (normalize each block, then weight-average them)
                    norm_block_fits = []
                    for bfit, bw in zip(block_fits, block_weights):
                        if use_subset == 'ws_ls':
                            nrm = np.linalg.norm(bfit)
                            if nrm == 0:
                                nrm = 1e-9
                            n_bfit = bfit / nrm
                        else:
                            if strategic:
                                csm = []
                                segs = len(bfit) // 5
                                for j_seg in range(segs):
                                    s0 = j_seg * 5
                                    s1 = min((j_seg + 1) * 5, len(bfit))
                                    if s1 > s0:
                                        seg = bfit[s0:s1]
                                        csm.append(np.sum(np.abs(seg) ** 2))
                                if csm:
                                    cmaxm = np.sqrt(np.sum(csm))
                                    if cmaxm == 0:
                                        cmaxm = 1e-9
                                    n_bfit = bfit / cmaxm
                                else:
                                    n_bfit = bfit
                            else:
                                n_bfit = bfit / (np.linalg.norm(bfit) if np.linalg.norm(bfit) != 0 else 1e-9)
                        norm_block_fits.append(n_bfit)
                        fval = compute_distance(model_data, n_bfit, distance_metric)
                        if np.isnan(fval) or np.isinf(fval):
                            continue
                        dsum += float(bw) * fval
                        wsum += float(bw)

                    if wsum == 0:
                        continue
                    avg_dist_for_monkey = dsum / wsum

                    # Weighted-average normalized monkey fit across blocks
                    norm_block_fits = np.array(norm_block_fits)
                    if norm_block_fits.ndim == 2:
                        monkey_norm_avg_fit = np.average(norm_block_fits, axis=0, weights=block_weights[:len(norm_block_fits)])
                    else:
                        monkey_norm_avg_fit = norm_block_fits
                                
                    # Track best model per monkey (averaged across blocks)
                    better = avg_dist_for_monkey > monkey_similarity_tracker[monkey]['similarity'] if distance_metric == 'cosine' else avg_dist_for_monkey < monkey_similarity_tracker[monkey]['similarity']
                    if better:
                        monkey_similarity_tracker[monkey]['idx'] = i
                        monkey_similarity_tracker[monkey]['weight'] = model
                        monkey_similarity_tracker[monkey]['fit'] = model_data
                        monkey_similarity_tracker[monkey]['monkey_fit'] = monkey_norm_avg_fit
                        monkey_similarity_tracker[monkey]['similarity'] = avg_dist_for_monkey

                    # Track best block-level match as well
                    # Use the minimum (or maximum for cosine) block distance among this monkey's blocks
                    block_dists = []
                    for n_bfit in norm_block_fits:
                        fval_block = compute_distance(model_data, n_bfit, distance_metric)
                        if not (np.isnan(fval_block) or np.isinf(fval_block)):
                            block_dists.append(fval_block)
                    if block_dists:
                        best_block_val = max(block_dists) if distance_metric == 'cosine' else min(block_dists)
                        is_better_block = best_block_val > monkey_similarity_tracker[monkey]['best_block_similarity'] if distance_metric == 'cosine' else best_block_val < monkey_similarity_tracker[monkey]['best_block_similarity']
                        if is_better_block:
                            monkey_similarity_tracker[monkey]['best_block_model_idx'] = i
                            monkey_similarity_tracker[monkey]['best_block_weight'] = model
                            monkey_similarity_tracker[monkey]['best_block_model_fit'] = model_data
                            # pick the block fit that achieved the best value
                            # recompute to find arg best
                            arg_best = np.argmax(block_dists) if distance_metric == 'cosine' else np.argmin(block_dists)
                            monkey_similarity_tracker[monkey]['best_block_monkey_fit'] = norm_block_fits[arg_best]
                            monkey_similarity_tracker[monkey]['best_block_similarity'] = best_block_val
                            monkey_similarity_tracker[monkey]['best_block_idx'] = int(arg_best)

                    # Accumulate by strategic group (averaged across all blocks and monkeys)
                    if m_data['type'] == 'nonstrategic':
                        sum_w_ns += wsum
                        sum_dw_ns += dsum
                    else:
                        sum_w_s += wsum
                        sum_dw_s += dsum

            # Store weighted averages if any
            if sum_w_s > 0:
                model_frechet_dict['strategic'][model] = sum_dw_s / sum_w_s
            if sum_w_ns > 0:
                model_frechet_dict['nonstrategic'][model] = sum_dw_ns / sum_w_ns
            
            # Old averaging path removed; weighted averages already computed above
        
        # Print summary of high-parameter model filtering
        print(f"\nHigh-parameter model summary (>=0.8, >=0.8):")
        print(f"  Found {high_param_models_found} unique high-parameter models")
        print(f"  {high_param_models_passed} model instances passed performance threshold")
        

        
        # Extract data for visualization
        x_dist = []
        y_dist = []
        grid_size = 11
        msarr = np.full((grid_size, grid_size), np.nan)  # Use NaN instead of zeros
        mnsarr = np.full((grid_size, grid_size), np.nan)  # Use NaN instead of zeros

        model_count = 0
        # Populate arrays for all strategic models
        for model in model_frechet_dict['strategic']:
            ms = model_frechet_dict['strategic'][model]
            x_dist.append(model[0])
            y_dist.append(model[1])
            model_count += 1
            x_idx = min(int(model[0] * (grid_size - 1)), grid_size - 1)
            y_idx = min(int(model[1] * (grid_size - 1)), grid_size - 1)
            msarr[y_idx, x_idx] = ms
        
        # Populate arrays for all non-strategic models
        for model in model_frechet_dict['nonstrategic']:
            mns = model_frechet_dict['nonstrategic'][model]
            if model not in [(x_dist[i], y_dist[i]) for i in range(len(x_dist))]:
                x_dist.append(model[0])
                y_dist.append(model[1])
                model_count += 1
            x_idx = min(int(model[0] * (grid_size - 1)), grid_size - 1)
            y_idx = min(int(model[1] * (grid_size - 1)), grid_size - 1)
            mnsarr[y_idx, x_idx] = mns
        
        # # make msarr and mnsarr increment from 0 to 1 to debug. Should be a 2d array
        # # where (0,0) is 0 and (1,1) is 1
        # msarr = np.linspace(0, 1, grid_size**2)
        # mnsarr = np.linspace(0, 1, grid_size**2)
        # msarr = np.array(msarr)
        # mnsarr = np.array(mnsarr)
        # msarr = msarr.reshape((grid_size, grid_size))
        # mnsarr = mnsarr.reshape((grid_size, grid_size))
        
        
        # Debug: Check parameter ranges and coverage
        if x_dist and y_dist:
            print(f"Parameter ranges: x_dist (after_loss) = {min(x_dist):.2f} to {max(x_dist):.2f}")
            print(f"Parameter ranges: y_dist (after_win) = {min(y_dist):.2f} to {max(y_dist):.2f}")
        else:
            print("Warning: No valid models found for visualization")
        print(f"Model data summary:")
        for model in [(0.5, 0), (0.2, 0)]:
            strategic_data = model_frechet_dict['strategic'].get(model, None)
            nonstrategic_data = model_frechet_dict['nonstrategic'].get(model, None)
            if strategic_data is not None or nonstrategic_data is not None:
                x_idx = min(int(model[0] * (grid_size-1)), grid_size-1)
                y_idx = min(int(model[1] * (grid_size-1)), grid_size-1)
                s_val = f"{strategic_data:.3f}" if strategic_data is not None else "None"
                ns_val = f"{nonstrategic_data:.3f}" if nonstrategic_data is not None else "None"
                print(f"  Model {model} -> grid[{y_idx}, {x_idx}]: Strategic={s_val}, Non-strategic={ns_val}")
        print(f"Grid coverage: {np.sum(~np.isnan(msarr))} strategic positions, {np.sum(~np.isnan(mnsarr))} non-strategic positions out of {grid_size*grid_size}")
        
        # Debug: Check what's actually in the corners of the grid arrays
        print(f"\nGrid corner values:")
        # Fix format string issues by separating the conditional logic
        def format_val(arr, i, j):
            return f"{arr[i,j]:.3f}" if not np.isnan(arr[i,j]) else "NaN"
        
        print(f"  Bottom-left [0,0]: Strategic={format_val(msarr,0,0)}, Non-strategic={format_val(mnsarr,0,0)}")
        print(f"  Bottom-right [0,{grid_size-1}]: Strategic={format_val(msarr,0,grid_size-1)}, Non-strategic={format_val(mnsarr,0,grid_size-1)}")
        print(f"  Top-left [{grid_size-1},0]: Strategic={format_val(msarr,grid_size-1,0)}, Non-strategic={format_val(mnsarr,grid_size-1,0)}")
        print(f"  Top-right [{grid_size-1},{grid_size-1}]: Strategic={format_val(msarr,grid_size-1,grid_size-1)}, Non-strategic={format_val(mnsarr,grid_size-1,grid_size-1)}")
        
        
        
        # Normalize values to 0-1 while preserving NaNs, then map to similarity
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

        msarr = _normalize(msarr)
        mnsarr = _normalize(mnsarr)

        # Convert to similarity scale uniformly so higher means better
        if use_performance or distance_metric == 'cosine':
            sim_s = msarr
            sim_ns = mnsarr
        else:
            sim_s = np.where(np.isnan(msarr), np.nan, 1.0 - msarr)
            sim_ns = np.where(np.isnan(mnsarr), np.nan, 1.0 - mnsarr)
        
        # Set x and y for the final plotting section
        x = x_dist
        y = y_dist
    
    # Debug: Print data distribution information
    print(f"Strategic data: {np.sum(~np.isnan(sim_s))} valid values out of {grid_size}x{grid_size} grid")
    print(f"Non-strategic data: {np.sum(~np.isnan(sim_ns))} valid values out of {grid_size}x{grid_size} grid")
    print(f"Strategic similarity range: {np.nanmin(sim_s):.3f} - {np.nanmax(sim_s):.3f}" if np.any(~np.isnan(sim_s)) else "No strategic data")
    print(f"Non-strategic similarity range: {np.nanmin(sim_ns):.3f} - {np.nanmax(sim_ns):.3f}" if np.any(~np.isnan(sim_ns)) else "No non-strategic data")
    
    # Plot directly at model coordinates without interpolation
    # Ensure x and y are not empty before plotting
    if not x or not y:
        # Handle empty data case but still plot overlays of best-matching models vs monkeys
        print("Warning: No data points to plot for comparison.")
        if plot_overlays:
            plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=False)
            plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=True)
        return ax, None

    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    
    # Debug: Print coordinate grid ranges
    print(f"x_grid range: {x_grid[0]:.2f} to {x_grid[-1]:.2f}")
    print(f"y_grid range: {y_grid[0]:.2f} to {y_grid[-1]:.2f}")
    print(f"Expected top-right plot coordinate: ({x_grid[-1]:.2f}, {y_grid[-1]:.2f})")
    
    # Calculate element sizes based on fixed grid range (0 to 1)
    x_range = 1.0  # Fixed range from 0 to 1
    y_range = 1.0  # Fixed range from 0 to 1
    
    # Size elements to fill grid completely (no gaps)
    grid_spacing_x = x_range / (grid_size - 1) if grid_size > 1 else x_range
    grid_spacing_y = y_range / (grid_size - 1) if grid_size > 1 else y_range
    square_size = min(grid_spacing_x, grid_spacing_y)
    circle_radius = square_size * 0.35
    
    # Debug: Check what's being plotted in the corners
    corner_positions = [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]
    corner_names = ["bottom-left", "bottom-right", "top-left", "top-right"]
    
    
    # Find maximum values for hybrid coloring (red for max, cividis for others)
    strategic_max = np.nanmax(sim_s) if np.any(~np.isnan(sim_s)) else None
    nonstrategic_max = np.nanmax(sim_ns) if np.any(~np.isnan(sim_ns)) else None
    
    for i in range(grid_size):
        for j in range(grid_size):
                         # Debug corner positions
            if (j, i) in corner_positions:
                 corner_idx = corner_positions.index((j, i))
                 corner_name = corner_names[corner_idx]
                 plot_coords = (x_grid[i], y_grid[j])
                 s_val = f"{msarr[j,i]:.3f}" if not np.isnan(msarr[j,i]) else "NaN"
                 ns_val = f"{mnsarr[j,i]:.3f}" if not np.isnan(mnsarr[j,i]) else "NaN"
                 print(f"Corner {corner_name} - Grid[{j},{i}] -> Plot coords {plot_coords}: Strategic={s_val}, Non-strategic={ns_val}")
            
            # Plot square (strategic model) if data available (including zero distance = perfect match)
            if not np.isnan(sim_s[j, i]):  # Plot if there's actual data (including 0.0)
                # Use red for maximum similarity, cividis for others
                is_max_strategic = (strategic_max is not None and 
                                  np.isclose(sim_s[j, i], strategic_max, rtol=1e-9))
                color = 'red' if is_max_strategic else plt.cm.cividis(sim_s[j, i])
                
                square = plt.Rectangle(
                    (x_grid[i] - square_size/2, y_grid[j] - square_size/2),
                    square_size, square_size,
                    facecolor=color,
                    edgecolor='k', alpha=0.8, linewidth=0.5
                )
                ax.add_patch(square)
                
            # Plot circle (non-strategic model) if data available (including zero distance = perfect match)
            if not np.isnan(sim_ns[j, i]):  # Plot if there's actual data (including 0.0)
                # Use red for maximum similarity, cividis for others
                is_max_nonstrategic = (nonstrategic_max is not None and 
                                      np.isclose(sim_ns[j, i], nonstrategic_max, rtol=1e-9))
                color = 'red' if is_max_nonstrategic else plt.cm.cividis(sim_ns[j, i])
                
                circle = plt.Circle(
                    (x_grid[i], y_grid[j]),
                    circle_radius,
                    facecolor=color,
                    edgecolor='k', alpha=0.9, linewidth=0.5
                )
                ax.add_patch(circle)
    
    # Get current figure and axes position
    fig = ax.figure
    ax_position = ax.get_position()
    
    # Create a divider for adding other axes components like the legend
    divider = make_axes_locatable(ax)

    # Create colorbar using inset_axes for more reliable positioning
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    # Remove any existing colorbar axis
    if hasattr(ax, 'colorbar_axis') and ax.colorbar_axis in fig.axes:
        ax.colorbar_axis.remove()
    
    # Create inset axes for colorbar with better positioning
    colorbar_axes = inset_axes(ax,
                         width="3%",      # width: slim colorbar
                         height="96%",   # slightly shorter to leave room near titles
                         loc='center left',
                         bbox_to_anchor=(1.04, 0.02, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
    
    # Store reference to the new axis
    ax.colorbar_axis = colorbar_axes
    
    # Create colorbar in the new axes with appropriate label; push it slightly outward to avoid overlap
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=colorbar_axes)
    
    if use_performance:
        cbar.set_label('Performance (normalized)', fontsize=COLORBAR_FONTSIZE)
    else:
        subset_label = "WS/LS" if use_subset == 'ws_ls' else "All"
        cbar.set_label(f'{get_distance_label(distance_metric)} ({subset_label})', fontsize=COLORBAR_FONTSIZE)
    
    # Keep label readable when on the left
    colorbar_axes.yaxis.set_label_position('right')
    try:
        cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE)
    except Exception:
        pass

    # Add legend as a separate axes outside the plot
    # Check if there's already a legend axis for this plot to avoid duplicates
    if hasattr(ax, 'legend_axis'):
        legend_ax = ax.legend_axis
    else:
        legend_ax = divider.append_axes("bottom", size="12%", pad=0.30)
        legend_ax.axis('off')
        ax.legend_axis = legend_ax  # Store reference
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=15, label='Strategic (square)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=15, label='Non-strategic (circle)')
    ]
    # Move the legend further down and shrink slightly to avoid any overlap
    legend_ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.10))
    
    # Add explicit red markers at the max-similarity coordinates (dynamic placement)
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
    
    # Set axis limits with fixed 0 to 1 range
    ax.set_xlim(-square_size, 1 + square_size)
    ax.set_ylim(-square_size, 1 + square_size)
    
    ax.set_xlabel('Decision Layer Loss Parameter', fontsize=MODEL_DISTANCE_AXISLABEL_FONTSIZE)
    ax.set_ylabel('Decision Layer Win Parameter', fontsize=MODEL_DISTANCE_AXISLABEL_FONTSIZE)
    try:
        ax.tick_params(axis='both', labelsize=MODEL_DISTANCE_TICK_FONTSIZE)
    except Exception:
        pass
    
    # Add text outside the plot area
    left, width = 0, 1
    bottom, height = 0, 1
    
    # Update positions for model type labels with fixed coordinates
    # (0,0) = (after_loss=0, after_win=0) = (x=0, y=0) = bottom left = RNN
    # (1,0) = (after_loss=1, after_win=0) = (x=1, y=0) = bottom right = Switching
    # (0,1) = (after_loss=0, after_win=1) = (x=0, y=1) = top left = RL
    ax.text(1.1, 1.15, 'RNN', 
            ha='left', va='top', fontsize=16)
    
    ax.text(1.05, -0.15, 'Switching', 
            rotation=0, ha='right', va='top', fontsize=16)
    
    ax.text(-0.1, -0.15, 'RL', 
            ha='left', va='bottom', fontsize=16)
    
    # Identify extreme models for visualization
    if plot_examples:
        if use_performance:
            extreme_models = identify_extreme_performance_models(model_perf_dict, model_names, distance_metric)
        else:
            extreme_models = identify_extreme_distance_models(model_frechet_dict, model_names, distance_metric)
        # Plot the most similar models for each monkey (averaged and block-level)
        plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=False)
        plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=True)
    else:
        extreme_models = None

    # Always offer overlays when requested
    if plot_overlays:
        plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=False)
        plot_most_similar_models(monkey_similarity_tracker, strategic=strategic, use_performance=use_performance, distance_metric=distance_metric, use_block_level=True)

    # Ensure larger tick labels on the colorbar
    try:
        if hasattr(ax, 'colorbar_axis'):
            for cb in ax.figure.axes:
                if hasattr(cb, 'get_images') and cb in [ax.colorbar_axis]:
                    pass
        # cbar created above already set to COLORBAR_FONTSIZE
    except Exception:
        pass
    
    
    return ax, extreme_models

def identify_extreme_performance_models(model_perf_dict, model_names, distance_metric, n_extreme=5):
    """
    Identify models with extreme performance values for visualization.
    
    Args:
        model_perf_dict: Dictionary with strategic/nonstrategic performance data
        model_names: List of model parameter tuples
        distance_metric: Distance metric being used (for consistency, not used in performance mode)
        n_extreme: Number of extreme models to identify
    
    Returns:
        Dictionary with extreme model information
    """
    extreme_models = {
        'strategic': {'highest': [], 'lowest': []},
        'nonstrategic': {'highest': [], 'lowest': []}
    }
    
    for group in ['strategic', 'nonstrategic']:
        if not model_perf_dict[group]:
            continue
            
        # Get all models and their performance values for this group
        models_and_performance = [(model, perf) for model, perf in model_perf_dict[group].items()]
        
        # Sort by performance (higher is better for performance)
        models_and_performance.sort(key=lambda x: x[1])
        
        # For performance, highest values are best (most accurate)
        extreme_models[group]['highest'] = models_and_performance[-n_extreme:]  # Best performance
        extreme_models[group]['lowest'] = models_and_performance[:n_extreme]    # Worst performance
    
    return extreme_models

def identify_extreme_distance_models(model_frechet_dict, model_names, distance_metric, n_extreme=5):
    """
    Identify models with extreme distance values for visualization.
    
    Args:
        model_frechet_dict: Dictionary with strategic/nonstrategic distance data
        model_names: List of model parameter tuples
        distance_metric: Distance metric being used
        n_extreme: Number of extreme models to identify
    
    Returns:
        Dictionary with extreme model information
    """
    extreme_models = {
        'strategic': {'highest': [], 'lowest': []},
        'nonstrategic': {'highest': [], 'lowest': []}
    }
    
    for group in ['strategic', 'nonstrategic']:
        if not model_frechet_dict[group]:
            continue
            
        # Get all models and their distances for this group
        models_and_distances = [(model, dist) for model, dist in model_frechet_dict[group].items()]
        
        # Sort by distance
        models_and_distances.sort(key=lambda x: x[1])
        
        # For cosine similarity, highest values are best (most similar)
        # For other distances, lowest values are best (closest)
        if distance_metric == 'cosine':
            # Highest cosine similarity = most similar
            extreme_models[group]['lowest'] = models_and_distances[-n_extreme:]  # Highest similarity
            extreme_models[group]['highest'] = models_and_distances[:n_extreme]  # Lowest similarity
        else:
            # Lowest distance = closest
            extreme_models[group]['lowest'] = models_and_distances[:n_extreme]   # Closest
            extreme_models[group]['highest'] = models_and_distances[-n_extreme:] # Furthest
    
    return extreme_models

def plot_extreme_model_regressors(extreme_models, data_file, strategic=True, distance_metric='euclidean'):
    """
    Plot regression coefficients for models with extreme distance values.
    
    Args:
        extreme_models: Dictionary from identify_extreme_distance_models
        data_file: Path to the model data file
        strategic: Whether using strategic regression
        distance_metric: Distance metric being used
    """
    # Load the model data from new PKL
    data_dict = load_all_fits_pkl()
    # Create a mapping from model params to fit data
    model_fits = {}
    for param_tuple, fits_dict in data_dict.items():
        model_fits[param_tuple] = [fit_data['action'] for fit_data in fits_dict.values()]
    
    # Create figure for extreme model visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=600)
    fig.suptitle(f'Regression Coefficients for Extreme {get_distance_label(distance_metric)} Models', fontsize=16)
    
    plot_titles = {
        ('strategic', 'lowest'): f'Strategic: {"Highest Similarity" if distance_metric == "cosine" else "Closest"}',
        ('strategic', 'highest'): f'Strategic: {"Lowest Similarity" if distance_metric == "cosine" else "Furthest"}',
        ('nonstrategic', 'lowest'): f'Non-strategic: {"Highest Similarity" if distance_metric == "cosine" else "Closest"}',
        ('nonstrategic', 'highest'): f'Non-strategic: {"Lowest Similarity" if distance_metric == "cosine" else "Furthest"}'
    }
    
    # Define regressor labels based on strategic flag
    if strategic:
        reggy = ['win stay', 'lose switch', 'win switch', 'lose stay']
    else:
        reggy = ['agent choice', 'win stay', 'lose switch']
    
    # Get matplotlib color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    ax_idx = 0
    order = 5  # Order for regression coefficients
    
    for group in ['strategic', 'nonstrategic']:
        for extreme_type in ['lowest', 'highest']:
            if ax_idx >= 4:
                break
                
            ax = axes[ax_idx // 2, ax_idx % 2]
            ax.set_title(plot_titles[(group, extreme_type)])
            
            models_list = extreme_models[group][extreme_type]
            if not models_list:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax_idx += 1
                continue
            
            # We'll plot the average fit for all extreme models
            all_fits = []
            model_labels = []
            
            for model_params, distance_val in models_list:
                if model_params in model_fits:
                    # Average across all instances of this model
                    avg_fit = np.mean(model_fits[model_params], axis=0)
                    all_fits.append(avg_fit)
                    model_labels.append(f'{model_params}')
            
            if all_fits:
                # Average across all extreme models for this group/type
                grand_avg_fit = np.mean(all_fits, axis=0)
                
                # Calculate standard error across models
                if len(all_fits) > 1:
                    std_fit = np.std(all_fits, axis=0)
                    err_fit = 1.96 * std_fit / np.sqrt(len(all_fits))  # 95% CI
                else:
                    err_fit = np.zeros_like(grand_avg_fit)
                
                # Set up x-axis for trials back
                xord = np.arange(1, order + 1)
                ax.set_xticks(range(1, order + 1))
                
                # Plot each regressor type with error bars, similar to paper_logistic_regression
                num_regressors = len(reggy)
                for i in range(num_regressors):
                    start_idx = i * order
                    end_idx = (i + 1) * order
                    
                    if end_idx <= len(grand_avg_fit):
                        coeffs = grand_avg_fit[start_idx:end_idx]
                        errors = err_fit[start_idx:end_idx]
                        
                        # Plot the main line
                        ax.plot(xord, coeffs, label=reggy[i], color=colors[i % len(colors)], linewidth=2)
                        
                        # Add error bars with fill_between
                        ax.fill_between(xord, coeffs - errors, coeffs + errors, 
                                      alpha=0.25, facecolor=colors[i % len(colors)])
                
                # Add horizontal dashed line at y=0
                ax.axhline(linestyle='--', color='k', alpha=0.5)
                
                # Set labels and styling to match logistic regression plots
                ax.set_xlabel('Trials Back')
                ax.set_ylabel('Logistic Regression Coefficient')
                ax.legend(frameon=False, fontsize=10)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No model data available', ha='center', va='center', transform=ax.transAxes)
            
            ax_idx += 1
    
    plt.tight_layout()
    
    # Print detailed information about extreme models
    print(f"\nExtreme {get_distance_label(distance_metric)} Models:")
    print("=" * 60)
    
    for group in ['strategic', 'nonstrategic']:
        print(f"\n{group.capitalize()} Group:")
        for extreme_type in ['lowest', 'highest']:
            type_label = "Highest Similarity" if (distance_metric == 'cosine' and extreme_type == 'lowest') else \
                        "Lowest Similarity" if (distance_metric == 'cosine' and extreme_type == 'highest') else \
                        "Closest" if extreme_type == 'lowest' else "Furthest"
            
            print(f"\n  {type_label}:")
            models_list = extreme_models[group][extreme_type]
            for model_params, distance_val in models_list:
                # Extract x,y coordinates: x=after_win, y=after_loss
                x_coord = model_params[1]  # after_win (x-axis)
                y_coord = model_params[0]  # after_loss (y-axis)
                print(f"    {model_params} (x={x_coord}, y={y_coord}): {distance_val:.4f}")

    return fig
    
def plot_models_in_region(data_file, strategic=True, region_bounds=((0.8, 1.0), (0.8, 1.0)), max_models=6):
    """
    Plot logistic regression coefficients for models within a specified parameter region.
    
    Args:
        data_file: Path to the model data file
        strategic: Whether using strategic regression
        region_bounds: ((min_after_loss, max_after_loss), (min_after_win, max_after_win))
        max_models: Maximum number of models to plot
    """
    # Load the model data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract region bounds
    (min_after_loss, max_after_loss), (min_after_win, max_after_win) = region_bounds
    
    # Find models in the specified region
    models_in_region = []
    model_fits = {}
    
    for datum in data:
        params = _ensure_param_tuple(datum['decision_params'])
        if type(params) != tuple:
            params = (params, params)
        
        after_loss, after_win = params
        
        # Check if model is in the specified region
        if (min_after_loss <= after_loss <= max_after_loss and 
            min_after_win <= after_win <= max_after_win):
            
            if params not in model_fits:
                model_fits[params] = []
            model_fits[params].append(datum['fit']['action'])
    
    # Select representative models (up to max_models)
    selected_models = list(model_fits.keys())[:max_models]
    
    if not selected_models:
        print(f"No models found in region {region_bounds}")
        return None
    
    print(f"Found {len(model_fits)} unique models in region {region_bounds}")
    print(f"Plotting first {len(selected_models)} models: {selected_models}")
    
    # Create figure
    n_cols = min(3, len(selected_models))
    n_rows = (len(selected_models) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=600)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    region_str = f"({min_after_loss:.1f}-{max_after_loss:.1f}, {min_after_win:.1f}-{max_after_win:.1f})"
    fig.suptitle(f'Logistic Regression Coefficients for Models in Region {region_str}', fontsize=16)
    
    # Define regressor labels based on strategic flag
    if strategic:
        reggy = ['win stay', 'lose switch', 'win switch', 'lose stay']
    else:
        reggy = ['agent choice', 'win stay', 'lose switch']
    
    # Get matplotlib color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    order = 5  # Order for regression coefficients
    
    for idx, model_params in enumerate(selected_models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Average across all instances of this model
        all_fits = model_fits[model_params]
        avg_fit = np.mean(all_fits, axis=0)
        
        # Calculate standard error if multiple fits
        if len(all_fits) > 1:
            std_fit = np.std(all_fits, axis=0)
            err_fit = 1.96 * std_fit / np.sqrt(len(all_fits))  # 95% CI
        else:
            err_fit = np.zeros_like(avg_fit)
        
        # Set up x-axis for trials back
        xord = np.arange(1, order + 1)
        ax.set_xticks(range(1, order + 1))
        
        # Plot each regressor type with error bars
        num_regressors = len(reggy)
        for i in range(num_regressors):
            start_idx = i * order
            end_idx = (i + 1) * order
            
            if end_idx <= len(avg_fit):
                coeffs = avg_fit[start_idx:end_idx]
                errors = err_fit[start_idx:end_idx]
                
                # Plot the main line
                ax.plot(xord, coeffs, label=reggy[i], color=colors[i % len(colors)], linewidth=2)
                
                # Add error bars with fill_between
                ax.fill_between(xord, coeffs - errors, coeffs + errors, 
                              alpha=0.25, facecolor=colors[i % len(colors)])
        
        # Add horizontal dashed line at y=0
        ax.axhline(linestyle='--', color='k', alpha=0.5)
        
        # Set labels and styling
        ax.set_xlabel('Trials Back')
        ax.set_ylabel('Logistic Regression Coefficient')
        ax.set_title(f'Model {model_params}\n(after_loss={model_params[0]:.1f}, after_win={model_params[1]:.1f})')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    total_plots = n_rows * n_cols
    for idx in range(len(selected_models), total_plots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig
    
    
    
    
    
    
    
    
    
    
def find_rlrnn_medoid_for_param(target_param=(0.5, 0.0), strategic: bool = True, use_subset: str = 'all'):
    def extract_coeffs(vec):
        if use_subset == 'ws_ls':
            return vec[:10] if strategic else vec[5:15]
        return vec[:-1] if (strategic and len(vec) > 20) else vec
    def norm_vec(arr):
        arr = np.array(arr, dtype=float)
        if use_subset == 'ws_ls':
            n = np.linalg.norm(arr); return arr / (n if n != 0 else 1e-9)
        if strategic:
            segs = len(arr) // 5; s = []
            for i in range(segs):
                s0 = i * 5; s1 = min((i + 1) * 5, len(arr))
                if s1 > s0:
                    seg = arr[s0:s1]; s.append(np.sum(np.abs(seg) ** 2))
            c = np.sqrt(np.sum(s)) if s else 1e-9
            return arr / (c if c != 0 else 1e-9)
        n = np.linalg.norm(arr); return arr / (n if n != 0 else 1e-9)
    candidates = []
    # Normalize target param to floats for robust matching
    try:
        target_param = (float(target_param[0]), float(target_param[1]))
    except Exception:
        pass
    # Include from 500-fits if same or very close
    pt_500, fits_500 = load_rlrnn_500_fits()
    try:
        if pt_500 is not None:
            p500 = (float(pt_500[0]), float(pt_500[1]))
            d500 = ((p500[0]-target_param[0])**2 + (p500[1]-target_param[1])**2) ** 0.5
            if d500 < 1e-6 or d500 <= 0.05:
                for idx, v in (fits_500 or {}).items():
                    if v.get('avg_reward', 0.0) <= 0.45: continue
                    act = v.get('action');
                    if act is None: continue
                    candidates.append((p500, idx, norm_vec(extract_coeffs(np.array(act)))))
    except Exception:
        pass
    # Include from sweep; if exact not present, use nearest available
    sweep = load_all_fits_pkl()
    fits = sweep.get(target_param, {})
    if not fits:
        try:
            best_k = None
            best_d = float('inf')
            for k in sweep.keys():
                try:
                    kf = (float(k[0]), float(k[1]))
                except Exception:
                    continue
                d = ((kf[0]-target_param[0])**2 + (kf[1]-target_param[1])**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_k = k
            fits = sweep.get(best_k, {}) if best_k is not None else {}
        except Exception:
            fits = {}
    for idx, v in fits.items():
        if v.get('avg_reward', 0.0) <= 0.45: continue
        act = v.get('action');
        if act is None: continue
        candidates.append((target_param, idx, norm_vec(extract_coeffs(np.array(act)))))
    if not candidates:
        return None, None, None
    vecs = np.stack([c[2] for c in candidates], axis=0)
    sims = vecs @ vecs.T
    sums = np.sum(sims, axis=1)
    midx = int(np.argmax(sums))
    return candidates[midx][2], candidates[midx][1], candidates[midx][0]
    
    
    
    
    
    
    
    
    
    
