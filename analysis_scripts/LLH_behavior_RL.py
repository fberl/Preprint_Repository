import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import expit as _expit
from analysis_scripts.logistic_regression import (
    logistic_regression_masked,
    paper_logistic_regression_strategic,
    histogram_logistic_accuracy_strategic,
    parse_monkey_behavior_strategic,
    create_order_data,
)
from envs.mp_env import MPEnv
import pandas as pd
from models.misc_utils import make_env
from typing import List, Tuple, Dict, Any, Optional


def fit_function_asymmetric(x, *args):
    alpha1, alpha2, beta, gamma = x
    actions, rewards, mask, decay, disable_abs = args
    
    # Conditionally apply abs() based on disable_abs parameter
    if not disable_abs:
        # Ensure alpha1, alpha2, and beta are always positive
        alpha1 = np.abs(alpha1)
        alpha2 = np.abs(alpha2)
        beta = np.abs(beta)
    
    alphas = [alpha1, alpha2]
    n = len(actions)

    # Create a list with the Q values of each trial
    Qs = np.ones((n, 2), dtype="float64") * 0.5
    for t, (a, r) in enumerate(
        zip(actions[:-1], rewards[:-1])
    ):  # The last Q values were never used, so there is no need to compute them
        r_index = int(max(0, r))
        Qs[t + 1, a] = Qs[t, a] + alphas[r_index] * (r - Qs[t, a] + gamma * max(Qs[t]))
        # Qs[t + 1, 1 - a] = (1-np.mean(alphas)*decay)*Qs[t, 1 - a]

    # Apply the softmax transformation in a vectorized way
    Qs_ = Qs * beta  # beta is now conditionally guaranteed to be positive
    return Qs_, actions, mask

def fit_function_simple(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    alpha, beta, gamma = x
    actions, rewards, mask, decay = args
    n = len(actions)

    # Create a list with the Q values of each trial
    Qs = np.ones((n, 2), dtype="float64") * 0.5
    # Qs[0] = 0.5  # Initialize Q values to 0.5 for both actions
    
    for t, (a, r) in enumerate(
        zip(actions[:-1], rewards[:-1])
    ):  # The last Q values were never used, so there is no need to compute them
        Qs[t + 1, a] = Qs[t, a] + alpha * (r - Qs[t, a] + gamma * max(Qs[t]))
        Qs[t + 1, 1 - a] = (1-alpha*decay) * Qs[t, 1 - a] 

    # Apply the softmax transformation in a vectorized way
    Qs_ = Qs * beta
    return Qs_, actions, mask

def fit_function_forgetting(x, *args):
    alpha, delta_win, delta_loss, beta = x
    actions, rewards, mask, decay = args
    n = len(actions)

    # Create a list with the Q values of each trial
    Qs = np.ones((n, 2), dtype="float64") * 0
    # Qs[0] = 0.5  # Initialize Q values to 0.5 for both actions
    
    for t, (a, r) in enumerate(
        zip(actions[:-1], rewards[:-1])
    ):  # The last Q values were never used, so there is no need to compute them
        Qs[t+1] = alpha * Qs[t]
        if r:
            Qs[t+1,a] += delta_win
        elif 1-r:
            Qs[t+1,a] += delta_loss
    # Apply the softmax transformation in a vectorized way
    # Qs_ = Qs * beta
    Qs_ = Qs * beta
    return Qs_, actions, mask

def compute_llik(Qs, actions, mask = None):
    logp_actions = Qs - scipy.special.logsumexp(Qs, axis=1)[:, None]
    # Return the logp_actions for the observed actions
    logp_actions = logp_actions[np.arange(len(actions)), actions]
    if mask is None:
        return -np.sum(logp_actions[1:])
    else:
        return -np.sum(mask[1:]*logp_actions[1:])

def test_performance(Qs, actions, mask = None, greedy=True):
    logp_actions = Qs - scipy.special.logsumexp(Qs, axis=1)[:, None]
    model_choices = np.argmax(logp_actions,axis=1)
    perf = np.squeeze(model_choices[1:]) == np.squeeze(actions[1:])
    # use the likelihood instead
    if not greedy:
        return np.exp(-compute_llik(Qs, actions, mask)* (1/len(actions))) 
    if mask is not None:
        mask =  mask[1:].reshape(perf.shape)
        mask = mask.astype(bool)
        perf = perf[mask] #apply mask
    perf = np.mean(perf)
    return perf

def llik_td_asymmetric(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    Qs_, actions, mask = fit_function_asymmetric(x, *args)
    return compute_llik(Qs_, actions, mask)

def llik_td_simple(x, *args):
    Qs_, actions, mask = fit_function_simple(x, *args)
    return compute_llik(Qs_, actions, mask)

    
def test_performance_simple(x, *args):
    Qs_, actions, mask = fit_function_simple(x, *args)
    return test_performance(Qs_, actions, mask)

def test_performance_asymmetric(x, *args):
    Qs_, actions, mask = fit_function_asymmetric(x, *args)
    return test_performance(Qs_, actions, mask)


def llik_td_forgetting(x, *args):
    Qs_, actions, mask = fit_function_forgetting(x, *args)
    return compute_llik(Qs_, actions, mask)

def test_performance_forgetting(x, *args):
    Qs_, actions, mask = fit_function_forgetting(x, *args)
    return test_performance(Qs_, actions, mask)

def single_session_fit(actions, rewards, model = 'forgetting', punitive=False, decay =False,
                     ftol = 1e-8, const_beta = True, const_gamma = True, alpha=None, mask = None, disable_abs=False):
    if model  == 'asymmetric':
        x0 = [.2,.2,1,0]  # Better initial values: both alphas positive
        if const_beta:
            constraint = [(0,1),(0,1),(1,1),(0,0)]  # Constrain alphas to (0,1)
        else:
            constraint = [(0,1),(0,1),(1e-3,10),(0,0)]  # Constrain alphas to (0,1)
        f = llik_td_asymmetric
        perf_func = test_performance_asymmetric
    elif model == 'simple':
        x0 = [.2,1,0]
        if const_beta:
            constraint = [(0.0005,1),(.999999,1.000001),(0,1)]
        else:
            constraint = [(0.0005,1),(1e-3,1000),(0,1)]
        f = llik_td_simple
        perf_func = test_performance_simple
    elif model == 'forgetting':
        if alpha != None:
            constraint = [(alpha-.0001,alpha+.0001),(-1,1),(-1,1),(1e-3,1000)]
            # cap the alpha to be a minimum
            # constraint = [(alpha,1),(-1,1),(-1,1)]

        else:
            constraint = [(0,1,),(-1,1),(-1,1),(1e-3,1000)]
        f = llik_td_forgetting
        perf_func = test_performance_forgetting
        x0 = [.5,0,0,1]
    if const_gamma and model != 'forgetting':
        constraint[-1] = (0,0)


    if mask != None:
        mask = np.array(mask).astype(bool)
    if punitive:
        rewards = 2 * rewards - 1
    
    # Pass disable_abs parameter to the fitting functions
    if model == 'asymmetric':
        result = scipy.optimize.minimize(f, x0, args=(actions, rewards, mask, decay, disable_abs),
                        bounds = constraint, tol = ftol)
        perf = perf_func(result.x, actions, rewards, mask, decay, disable_abs)
    else:
        result = scipy.optimize.minimize(f, x0, args=(actions, rewards, mask, decay),
                        bounds = constraint, tol = ftol)
        perf = perf_func(result.x, actions, rewards, mask, decay)
    
    fit_res = list(result.x)
    
    # For asymmetric model, conditionally ensure all parameters are positive where appropriate
    if model == 'asymmetric' and not disable_abs:
        fit_res[0] = np.abs(fit_res[0])  # alpha1 (alpha_win)
        fit_res[1] = np.abs(fit_res[1])  # alpha2 (alpha_loss) 
        fit_res[2] = np.abs(fit_res[2])  # beta (temperature)
        # gamma (fit_res[3]) can remain as-is since it's often 0
    
    
    return fit_res, perf


def multi_session_fit(actions, rewards, model = 'forgetting', punitive=False, decay =False,
                     ftol = 1e-8, alpha = None,const_beta = False, const_gamma = True, mask = None, disable_abs=False):
    if model  == 'asymmetric':
        x0 = [.2,.2,1,0]  # Better initial values: both alphas positive
        if const_beta:
            constraint = [(0,1),(0,1),(1,1),(0,0)]  # Constrain alphas to (0,1)
        else:
            constraint = [(0,1),(0,1),(1e-3,10),(0,0)]  # Constrain alphas to (0,1)
        f = llik_td_asymmetric
        perf_func = test_performance_asymmetric
    elif model == 'simple':
        x0 = [.2,1,0]
        if const_beta:
            constraint = [(0.0005,1),(.999999,1.000001),(0,1)]
        else:
            constraint = [(0.0005,1),(1e-3,1000),(0,1)]
        f = llik_td_simple
        perf_func = test_performance_simple
    elif model == 'forgetting':
        constraint = [(0,1,),(-1,1),(-1,1),(1e-3,1000)]
        f = llik_td_forgetting
        perf_func = test_performance_forgetting
        x0 = [.5,0,0,1]
    if const_gamma and model != 'forgetting':
        constraint[-1] = (0,0)

    if alpha != None:
        constraint[0] = (alpha-.0001,alpha+.0001)

    if mask != None:
        mask = np.array(mask).astype(bool)
    if punitive:
        rewards = [2 * np.array(r) - 1 for r in rewards]
        # rewards = 2 * rewards - 1
        
        
    def LLH_wrapper(x,f,ep_a, ep_r, mask,decay):
        tot = 0
        for i in range(len(ep_a)):
            if model == 'asymmetric':
                tot += f(x,ep_a[i],ep_r[i],mask[i],decay,disable_abs)
            else:
                tot += f(x,ep_a[i],ep_r[i],mask[i],decay)
        return tot
    
    # generate mask for each :(
    fullfits = []
    result = scipy.optimize.minimize(LLH_wrapper, x0, args=(f,actions, rewards, [None]*len(actions),decay),
                            bounds = constraint,tol=ftol)
    # perf = [np.average([perf_func(avg[i],episode_actions[j], episode_rewards[j], masks[i][j],decay)
    #                                for j in range(len(episode_actions))],weights = weights) for i in range(len(avg))]
    fit_res = list(result.x)
    
    # For asymmetric model, conditionally ensure all parameters are positive where appropriate
    if model == 'asymmetric' and not disable_abs:
        fit_res[0] = np.abs(fit_res[0])  # alpha1 (alpha_win)
        fit_res[1] = np.abs(fit_res[1])  # alpha2 (alpha_loss) 
        fit_res[2] = np.abs(fit_res[2])  # beta (temperature)
        # gamma (fit_res[3]) can remain as-is since it's often 0
    
    def perf_wrapper(fit_res,actions, rewards,mask,decay):
        # if model == 'asymmetric':
        #     pf = np.average([perf_func(fit_res,actions[i], rewards[i], mask,decay,disable_abs) for i in range(len(actions))])
        # else:
        #     pf = np.average([perf_func(fit_res,actions[i], rewards[i], mask,decay) for i in range(len(actions))])
        
        # should be a geometric mean weighted by the length of the episode. 
        total_length = sum([len(actions[i]) for i in range(len(actions))])
        pf = np.prod([perf_func(fit_res,actions[i], rewards[i], mask,decay)**(len(actions[i])/total_length) for i in range(len(actions))])
        return pf
    
    perf = perf_wrapper(fit_res,actions, rewards,mask,decay)
    return fit_res, perf
    
def fit_params_model(episode_actions, episode_rewards, model = 'forgetting', punitive=False, decay =False,
                     ftol = 1e-8, const_beta = True, const_gamma = True):
    if model  == 'asymmetric':
        x0 = [.2,.2,1,0]
        if const_beta:
            constraint = [(0,1),(0,1),(1,1),(0,0)]
        else:
            constraint = [(0,1),(0,1),(1e-3,10),(0,0)]
        param_names = [r'$\alpha_{win}$',r'$\alpha_{loss}$', r'$\beta$',r'$\gamma$', r'perf']
        f = llik_td_asymmetric
        perf_func = test_performance_asymmetric
    elif model == 'simple':
        x0 = [.2,1,0]
        if const_beta:
            constraint = [(0.025,1),(.999999,1.000001),(0,1)]
        else:
            constraint = [(0.025,1),(1e-3,1000),(0,1)]
        param_names = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'perf']
        f = llik_td_simple
        perf_func = test_performance_simple
    elif model == 'forgetting':
        constraint = [(0,1,),(-1,1),(-1,1),(1e-3,5)]
        param_names = [r'$\alpha$', r'$\Delta_{win}$',r'$\Delta_{loss}$', r'perf' ]
        f = llik_td_forgetting
        perf_func = test_performance_forgetting
        x0 = [.5,0,0,1]
    if const_gamma and model != 'forgetting':
        constraint[-1] = (0,1e-12)
    # fig, axs = plt.subplots(num_rows = 3 num_cols = len(param_names), dpi=200, constrained_layout=True)
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(4*len(param_names), 10))
    fig.suptitle('MLE for Q-Learning Parameters', fontsize = 24)
    subfigs = fig.subfigures(nrows=3, ncols=1)

    # want to plot a histogram of the distributions for each parameter,
    # as well as means and standard deviation
    weights = []
    post_win_p = []
    post_loss_p = []
    full_p = []
    post_win_masks = []
    post_loss_masks = []
    
    full_performances = []
    post_win_performances = []
    post_loss_performances = []
    
    for sess, (actions,rewards) in enumerate(zip(episode_actions,episode_rewards)):
        postWin = np.roll(rewards,1)
        postLoss = 1 - postWin
        masks = [None, postWin, postLoss]
        post_win_masks.append(postWin)
        post_loss_masks.append(postLoss)

        # actions = actions.ravel()
        # rewards = rewards.ravel()

        weights.append(len(actions))
        
        if punitive:
            rewards = 2 * rewards - 1
        for i in range(len(masks)):
            result = scipy.optimize.minimize(f, x0, args=(actions, rewards, masks[i],decay),
                            bounds = constraint, tol = ftol)
            # actions = actions.ravel()

            perf = perf_func(result.x,actions, rewards,masks[i],decay)
            fit_res = list(result.x)
            fit_res.append(perf)

            if i == 0:
                full_p.append(fit_res)
                # full_performances.append(perf_func(result.x,actions, rewards,masks[i],decay))
            elif i == 1:
                post_win_p.append(fit_res)
                # post_win_performances.append(perf_func(result.x,actions, rewards,masks[i],decay))
            elif i == 2:
                post_loss_p.append(fit_res)
                # post_loss_performances.append(perf_func(result.x,actions, rewards,masks[i],decay))
    full_p = np.array(full_p)
    post_loss_p = np.array(post_loss_p)
    post_win_p = np.array(post_win_p)
    

    
    results = [full_p, post_win_p, post_loss_p]
    return results
    

# what about a model that uses the above in order to 
class RLModelFits:
    def __init__(self) -> None:
        pass

    def fitModel(self,episode_actions, episode_rewards, model_type = 'forgetting', punitive=False, decay =False,
                     ftol = 1e-8, const_beta = True, const_gamma = True, reset_time = 200):
        model_fits, masks_win, masks_loss = fit_params_model(episode_actions, episode_rewards, model = model_type, punitive=punitive, decay =decay,
                     ftol = ftol, const_beta = const_beta, const_gamma = const_gamma)
        self.fig = plt.figure(constrained_layout=True, dpi=300, figsize=(4*3, 10))
        self.fig.suptitle('Logistic Regressions for MLE Fitted Models', fontsize = 24)
        self.subfigs = self.fig.subfigures(nrows=3, ncols=1)
        self.subfigs[0].suptitle('Full Model', fontsize = 18)
        self.subfigs[1].suptitle('Post Win Model',fontsize = 18)
        self.subfigs[2].suptitle('Post Loss Model', fontsize = 18)
        
        # for (fit,mask) in zip(model_fits,masks):
        #     model = RLModel(fit, model_type, reset_time=reset_time)
        #     model.fit_logistic()


# def test_train_model()

class RLModel():
    def __init__(self, params, model_type, reset_time = 100, nits = 500):
        func_dict = {'forgetting' : self.forward_forgetting, 'simple' : self.forward_simple, 'asymmetric' : self.forward_asymmetric}

        self.params = params
        self.model_type = model_type
        self.forward = func_dict[model_type]
        self.reset()
        self.Qs = np.zeros((2))
        self.nits = nits
        self.reset_time = reset_time
        # generate data:
        # first, create the environment
        env_kwargs = {
            "show_opp" : False,
            "fixed_length" : True,
            "reset_time" : 200,
            "opponents_params" : {"all":{
                "bias":[0],
                "depth": 4
                }},
            "opponents" : ["all"],
            "opponent" : "all"

        }
        # env_kwargs = {
        #     "show_opp" : False,
        #     "fixed_length" : True,
        #     "reset_time" : 200,
        #     "opponents_params" : {"1":{
        #         "bias":[0],
        #         "depth": 4
        #         }},
        #     "opponents" : ["1"],
        #     "opponent" : "1"

        # }
        self.env = make_env(env_kwargs)
    def reset(self):
        if self.model_type == 'forgetting':
            self.Qs = np.zeros(2)
    
    def fit_logistic(self):
        episode_actions = []
        episode_rewards = []
        mask_win = []
        mask_loss = []
        for i in range(self.nits):
            self.reset()
            state = self.env.reset()
            last_action = self.env.action_space.sample()
            reward = np.random.randint(0,2)
            actions = []
            rewards = []
            self.Qs = self.forward(self.params,self.Qs,last_action, reward, None, None)
            for j in range(self.reset_time):
                action = np.argmax(self.Qs)
                actions.append(action)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                self.Qs = self.forward(self.params,self.Qs,action, reward, None, None)
                
            episode_actions.append(actions)
            episode_rewards.append(rewards)
        mask_win = [np.roll(episode_rewards[i], 1) for i in range(len(episode_rewards))]
        mask_loss = [1-mask for mask in mask_win]
        masks = [None, mask_win, mask_loss]  
        episode_actions = np.array(episode_actions)
        episode_rewards = np.array(episode_rewards)
        
        # create df from these data
        bundle_dict = {'id' : [], 'trial' : [], 'monkey_choice' : [], 'reward' : [], 'computer_choice' : []}
        for epi in range(len(episode_actions)):
            for tr in range(len(episode_actions[epi])):
                bundle_dict['id'].append(epi)
                bundle_dict['trial'].append(tr)
                bundle_dict['monkey_choice'].append(episode_actions[epi][tr])
                bundle_dict['reward'].append(episode_rewards[epi][tr])
                bundle_dict['computer_choice'].append(-1)
        bundle = pd.DataFrame.from_dict(bundle_dict)
        # bundle = (None, episode_actions, episode_rewards, None, None, None)
        print('Win Rate: {}'.format(np.mean(episode_rewards)))
        for mask in masks:
            logistic_regression_masked(False, bundle, mask = mask, err = True, fitted_RL=True)
        
    def forward_forgetting(self,x,Qs, *args):
        alpha, delta_win, delta_loss = x
        action, reward, mask, decay = args
        if Qs is None:
            Qs = np.zeros( (2), dtype="float64")
        Qs = Qs * alpha
        if reward:
            Qs[action] += delta_win
        elif 1-reward:
            Qs[action] += delta_loss    
        return Qs
    
    def forward_simple(x, Qs, *args):
        pass
    
    def forward_asymmetric(x, Qs, *args):
        pass


def cross_validated_performance_sessions(
    episode_actions: List[np.ndarray],
    episode_rewards: List[np.ndarray],
    model: str = 'simple',
    n_folds: int = 10,
    random_state: int = 0,
    punitive: bool = False,
    decay: bool = False,
    const_beta: bool = False,
    const_gamma: bool = True,
    disable_abs: bool = False,
    n_bootstrap: int = 1000,
    greedy: bool = True,
    mask: Optional[List[np.ndarray]] = None,
    session_selection: str = "last10",
    per_monkey: bool = True,
) -> Dict[str, Any]:
    """
    Session-level K-fold cross-validation for RL and LR model performance.

    - Splits sessions (episodes) into K folds.
    - Trains on K-1 folds, evaluates accuracy on held-out sessions.
    - Returns overall test accuracy (trial-weighted), per-session accuracies, and
      bootstrap SEM computed by resampling sessions.

    Args:
        episode_actions: list of arrays of actions, one per session
        episode_rewards: list of arrays of rewards, one per session
        model: 'simple' | 'asymmetric' | 'forgetting'
        n_folds: number of folds (defaults to 10), clipped to number of sessions
        random_state: RNG seed for reproducibility
        punitive, decay, const_beta, const_gamma, disable_abs: RL fit options
        n_bootstrap: number of bootstrap replicates for SEM
        greedy: pass-through to test_performance (True = argmax accuracy)
        mask: optional masks per session (unused for most cases)

    Returns:
        dict with keys: 'mean_accuracy', 'per_session_accuracy', 'per_session_lengths',
        'bootstrap_means', 'bootstrap_sem', 'fold_assignments', 'params_per_fold'
    """
    num_sessions = len(episode_actions)
    if num_sessions == 0:
        return {
            'mean_accuracy': np.nan,
            'per_session_accuracy': [],
            'per_session_lengths': [],
            'bootstrap_means': [],
            'bootstrap_sem': np.nan,
            'fold_assignments': [],
            'params_per_fold': [],
            'LR_mean_accuracy': np.nan,
            'LR_bootstrap_means': [],
            'LR_bootstrap_sem': np.nan,
        }

    # Clip folds to at most number of sessions, ensure at least 2 if possible
    k = int(max(1, min(n_folds, num_sessions)))
    # If k==1, we cannot do CV; fall back to train on all and test on all (not ideal)
    if k == 1:
        k = 2 if num_sessions >= 2 else 1

    rng = np.random.default_rng(random_state)
    indices = np.arange(num_sessions)
    rng.shuffle(indices)

    # Optionally perform per-monkey selection like Fig 2: last N sessions per monkey
    # The caller should pre-slice episodes per monkey to enforce selection; here we proceed with given episodes
    # Partition indices into k folds as evenly as possible
    folds: List[np.ndarray] = np.array_split(indices, k)

    # Select perf function based on model
    if model == 'asymmetric':
        perf_func = test_performance_asymmetric
    elif model == 'forgetting':
        perf_func = test_performance_forgetting
    else:
        perf_func = test_performance_simple

    # Storage
    per_session_acc = np.full(num_sessions, np.nan, dtype=float)
    LR_per_session_acc = np.full(num_sessions, np.nan, dtype=float)
    per_session_len = np.array([len(a) for a in episode_actions], dtype=int)
    fold_assignments = np.full(num_sessions, -1, dtype=int)
    params_per_fold: List[List[float]] = []

    for fold_id, test_idx in enumerate(folds):
        test_idx = np.array(test_idx, dtype=int)
        train_idx = np.array([i for i in indices if i not in test_idx], dtype=int)
        if train_idx.size == 0:
            # if no train data (can happen when num_sessions<k), skip this fold
            continue

        train_actions = [episode_actions[i] for i in train_idx]
        train_rewards = [episode_rewards[i] for i in train_idx]

        # Fit on training sessions
        fit_params, _ = multi_session_fit(
            train_actions,
            train_rewards,
            model=model,
            punitive=punitive,
            decay=decay,
            ftol=1e-8,
            alpha=None,
            const_beta=const_beta,
            const_gamma=const_gamma,
            mask=None,
            disable_abs=disable_abs,
        )
        params_per_fold.append(fit_params)

        # Fit logistic regression (strategic) on training sessions using provided helper
        # Build a pandas DataFrame in the expected schema
        order_lr = 5
        bias_lr = True
        train_df = {'id': [], 'trial': [], 'monkey_choice': [], 'reward': [], 'computer_choice': []}
        for idx, (aa, rr) in enumerate(zip(train_actions, train_rewards)):
            for t in range(len(aa)):
                train_df['id'].append(idx)
                train_df['trial'].append(t)
                train_df['monkey_choice'].append(int(aa[t]))
                train_df['reward'].append(int(rr[t]))
                train_df['computer_choice'].append(0)
        train_df = pd.DataFrame(train_df)
        try:
            sols = paper_logistic_regression_strategic(
                None, False, data=train_df, order=order_lr, bias=bias_lr, return_model=True, colinear=True
            )
            theta_lr = sols['action'] if isinstance(sols, dict) and 'action' in sols else None
        except Exception:
            theta_lr = None
        # Evaluate on test sessions (both RL model and LR baseline)
        for j in test_idx:
            a = episode_actions[j]
            r = episode_rewards[j]
            if model == 'asymmetric':
                acc = perf_func(fit_params, a, r, None, decay, disable_abs)
            elif model == 'forgetting':
                acc = perf_func(fit_params, a, r, None, decay)
            else:
                acc = perf_func(fit_params, a, r, None, decay)
            per_session_acc[j] = float(acc)
            fold_assignments[j] = fold_id

            # LR baseline per-session accuracy on held-out session
            try:
                if theta_lr is not None and len(a) > order_lr:
                    # Build test DataFrame for this session
                    te_df = pd.DataFrame({
                        'id': [0]*len(a),
                        'trial': list(range(len(a))),
                        'monkey_choice': [int(x) for x in a],
                        'reward': [int(x) for x in r],
                        'computer_choice': [0]*len(a),
                    })
                    X_te = parse_monkey_behavior_strategic(te_df, order_lr, vif=False, err=False)
                    y_te = create_order_data(te_df, order_lr, err=False)
                    if X_te is not None and X_te.shape[0] == y_te.shape[0] and X_te.shape[0] > 0:
                        logits = X_te @ theta_lr[:-1] + theta_lr[-1] if bias_lr else X_te @ theta_lr
                        preds = (_expit(logits) > 0.5).astype(int)
                        LR_per_session_acc[j] = float(np.mean(preds == y_te))
            except Exception:
                pass

    # Compute overall mean accuracy weighted by session length
    valid_mask = ~np.isnan(per_session_acc)
    if not np.any(valid_mask):
        mean_acc = np.nan
        boot_means = []
        boot_sem = np.nan
    else:
        tot_trials = float(np.sum(per_session_len[valid_mask]))
        if tot_trials > 0:
            mean_acc = float(np.sum(per_session_acc[valid_mask] * per_session_len[valid_mask]) / tot_trials)
        else:
            mean_acc = float(np.nanmean(per_session_acc[valid_mask]))

        # Bootstrap SEM by resampling sessions with replacement
        S = int(np.sum(valid_mask))
        valid_indices = np.where(valid_mask)[0]
        va = per_session_acc[valid_mask]
        vl = per_session_len[valid_mask]
        boot_means_arr = np.zeros(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            samp = rng.integers(0, S, size=S)
            # Map to global indices if needed
            acc_s = va[samp]
            len_s = vl[samp]
            denom = float(np.sum(len_s))
            if denom == 0:
                boot_means_arr[b] = float(np.mean(acc_s))
            else:
                boot_means_arr[b] = float(np.sum(acc_s * len_s) / denom)
        boot_means = boot_means_arr.tolist()
        boot_sem = float(np.std(boot_means_arr, ddof=1)) if n_bootstrap > 1 else 0.0

    # LR aggregate metrics
    LR_valid_mask = ~np.isnan(LR_per_session_acc)
    if np.any(LR_valid_mask):
        tot_trials_lr = float(np.sum(per_session_len[LR_valid_mask]))
        if tot_trials_lr > 0:
            LR_mean_accuracy = float(np.sum(LR_per_session_acc[LR_valid_mask] * per_session_len[LR_valid_mask]) / tot_trials_lr)
        else:
            LR_mean_accuracy = float(np.nanmean(LR_per_session_acc[LR_valid_mask]))
        # Bootstrap SEM for LR
        S_lr = int(np.sum(LR_valid_mask))
        va_lr = LR_per_session_acc[LR_valid_mask]
        vl_lr = per_session_len[LR_valid_mask]
        boot_means_arr_lr = np.zeros(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            samp = rng.integers(0, S_lr, size=S_lr)
            acc_s = va_lr[samp]
            len_s = vl_lr[samp]
            denom = float(np.sum(len_s))
            if denom == 0:
                boot_means_arr_lr[b] = float(np.mean(acc_s))
            else:
                boot_means_arr_lr[b] = float(np.sum(acc_s * len_s) / denom)
        LR_bootstrap_means = boot_means_arr_lr.tolist()
        LR_bootstrap_sem = float(np.std(boot_means_arr_lr, ddof=1)) if n_bootstrap > 1 else 0.0
    else:
        LR_mean_accuracy = np.nan
        LR_bootstrap_means = []
        LR_bootstrap_sem = np.nan

    return {
        'mean_accuracy': mean_acc,
        'per_session_accuracy': per_session_acc.tolist(),
        'per_session_lengths': per_session_len.tolist(),
        'bootstrap_means': boot_means,
        'bootstrap_sem': boot_sem,
        'fold_assignments': fold_assignments.tolist(),
        'params_per_fold': params_per_fold,
        'LR_mean_accuracy': LR_mean_accuracy,
        'LR_bootstrap_means': LR_bootstrap_means,
        'LR_bootstrap_sem': LR_bootstrap_sem,
    }


def cross_validated_performance_by_monkey_df(
    df: pd.DataFrame,
    monkey_ids: list,
    model: str = 'simple',
    n_folds: int = 10,
    random_state: int = 0,
    order_lr: int = 5,
    bias_lr: bool = True,
    punitive: bool = False,
    decay: bool = False,
    const_beta: bool = False,
    const_gamma: bool = True,
    disable_abs: bool = False,
    n_bootstrap: int = 1000,
    greedy: bool = True,
) -> Dict[str, Any]:
    """
    Per-monkey K-fold CV: for each monkey, split their sessions into folds,
    train RL and LR on that monkey's training sessions, and evaluate on that monkey's held-out sessions.
    Aggregates trial-weighted accuracies across all monkeys.
    """
    rng = np.random.default_rng(random_state)
    all_session_acc_rl: list = []
    all_session_acc_lr: list = []
    all_session_len: list = []

    for mk in monkey_ids:
        mk_df = df[df['animal'] == mk].sort_values(['id','trial'])
        sessions = mk_df['id'].unique()
        if len(sessions) == 0:
            continue
        folds = np.array_split(rng.permutation(sessions), min(max(2, n_folds), len(sessions)))

        # Build episodes once
        mk_episodes = {}
        for sid, sub in mk_df.groupby('id'):
            sub = sub.sort_values('trial')
            a = sub['monkey_choice'].to_numpy().astype(int)
            r = sub['reward'].to_numpy().astype(int)
            mk_episodes[sid] = (a, r)

        for fold in folds:
            test_ids = set(fold.tolist())
            train_ids = [sid for sid in sessions if sid not in test_ids]
            if len(train_ids) == 0:
                continue

            # Build RL train data
            train_actions = [mk_episodes[sid][0] for sid in train_ids]
            train_rewards = [mk_episodes[sid][1] for sid in train_ids]

            # Fit RL on training sessions
            fit_params, _ = multi_session_fit(
                train_actions,
                train_rewards,
                model=model,
                punitive=punitive,
                decay=decay,
                ftol=1e-8,
                alpha=None,
                const_beta=const_beta,
                const_gamma=const_gamma,
                mask=None,
                disable_abs=disable_abs,
            )

            # Fit LR on training sessions using paper helper
            train_df = {'id': [], 'trial': [], 'monkey_choice': [], 'reward': [], 'computer_choice': []}
            for row_id, sid in enumerate(train_ids):
                a, r = mk_episodes[sid]
                for t in range(len(a)):
                    train_df['id'].append(row_id)
                    train_df['trial'].append(t)
                    train_df['monkey_choice'].append(int(a[t]))
                    train_df['reward'].append(int(r[t]))
                    train_df['computer_choice'].append(0)
            train_df = pd.DataFrame(train_df)
            try:
                sols = paper_logistic_regression_strategic(
                    None, False, data=train_df, order=order_lr, bias=bias_lr, return_model=True, colinear=True
                )
                theta_lr = sols['action'] if isinstance(sols, dict) and 'action' in sols else None
            except Exception:
                theta_lr = None

            # Evaluate on held-out sessions for this monkey
            if model == 'asymmetric':
                perf_func = test_performance_asymmetric
            elif model == 'forgetting':
                perf_func = test_performance_forgetting
            else:
                perf_func = test_performance_simple

            for sid in test_ids:
                a, r = mk_episodes[sid]
                # RL
                rl_acc = perf_func(fit_params, a, r, None, decay) if model != 'asymmetric' else perf_func(fit_params, a, r, None, decay, disable_abs)
                all_session_acc_rl.append(float(rl_acc))
                all_session_len.append(len(a))
                # LR
                try:
                    if theta_lr is not None and len(a) > order_lr:
                        te_df = pd.DataFrame({
                            'id': [0]*len(a),
                            'trial': list(range(len(a))),
                            'monkey_choice': [int(x) for x in a],
                            'reward': [int(x) for x in r],
                            'computer_choice': [0]*len(a),
                            'animal': [mk]*len(a),
                        })
                        X_te = parse_monkey_behavior_strategic(te_df, order_lr, vif=False, err=False)
                        y_te = create_order_data(te_df, order_lr, err=False)
                        logits = X_te @ theta_lr[:-1] + theta_lr[-1] if bias_lr else X_te @ theta_lr
                        preds = (_expit(logits) > 0.5).astype(int)
                        lr_acc = float(np.mean(preds == y_te))
                    else:
                        lr_acc = np.nan
                except Exception:
                    lr_acc = np.nan
                all_session_acc_lr.append(lr_acc)

    # Aggregate trial-weighted means
    L = np.array(all_session_len, dtype=float)
    RL = np.array(all_session_acc_rl, dtype=float)
    LR = np.array(all_session_acc_lr, dtype=float)
    mask_rl = np.isfinite(RL)
    mask_lr = np.isfinite(LR)
    rl_mean = float(np.sum(RL[mask_rl]*L[mask_rl])/np.sum(L[mask_rl])) if np.any(mask_rl) else float('nan')
    lr_mean = float(np.sum(LR[mask_lr]*L[mask_lr])/np.sum(L[mask_lr])) if np.any(mask_lr) else float('nan')
    return {
        'mean_accuracy': rl_mean,
        'LR_mean_accuracy': lr_mean,
        'n_sessions': int(len(L)),
    }
    