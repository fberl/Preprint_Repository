"""
Script to train RNN networks on the given tasks
Takes an arg and trains based on that opponent
"""
import sys
import os
# Get the path to the repository root (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))  # cluster_scripts/regraining/
cluster_dir = os.path.dirname(script_dir)  # cluster_scripts/
repo_root = os.path.dirname(cluster_dir)  # repository root
sys.path.append(repo_root)
import random
import torch
import numpy as np
from collections import namedtuple, deque
from models.rnn_model import RLRNN
from models.misc_utils import make_env
import json
import analysis_scripts.logistic_regression as logistic_regression
from analysis_scripts.test_suite import generate_data
import pickle
from analysis_scripts.model_prediction_sequence_matching import compute_single_model_perf, load_data
from analysis_scripts.entropy import compute_mutual_information 
import glob
import re

stitched_p = 'stitched_monkey_data.pkl'

device = torch.device("cpu")

params = json.loads(sys.argv[1])
# params = json.loads('{"seed": 28980, "net_params": {"hidden_dim": 64,"weighting" : [1,0], "batch_size": 8, "model_path": "reee_(420,69)_420", "max_episodes": 65536, "max_steps": 200, "l2": 0.05, "l1": 0.01, "RL": true, "leaky_tau": 1, "leaky_q": true, "leaky_policy": true, "Qalpha": 0.2, "gamma": 0.75, "lambd": 0, "q_init": 0.5, "use_linear2": false, "scaletype": "adaptive"}, "env_params": {"show_opp": false, "fixed_length": true, "reset_time": 200, "opponents_params": {"all": {"bias": [0], "depth": 4}}, "opponents": ["all"], "opponent": "all"}}')

env_kwargs = params["env_params"]
net_kwargs = params["net_params"]
seed = params["seed"]

random.seed(seed)
torch.manual_seed(seed)

env = make_env(env_kwargs)

def find_latest_checkpoint(model_path):
    """
    Find the latest checkpoint in the model directory.
    Returns the episode number of the latest checkpoint, 'completed' if job is done, or None if no checkpoints found.
    """
    if not os.path.exists(model_path):
        return None
    
    # Check if job is already completed by looking for fit_all_sess.p
    # fit_all_sess_file = os.path.join(model_path, 'fit_all_sess.p')
    # if os.path.exists(fit_all_sess_file):
    #     print(f"Job already completed (found {fit_all_sess_file})")
    #     return "completed"
    
    # Get the model_name directly from the model_path (same logic as RLRNN.__init__)
    model_name = os.path.basename(model_path)
    print(f"Searching for checkpoints with model_name: '{model_name}' in: {model_path}")
    
    # Check if final models already exist (without episode numbers) - PRIORITY 1
    final_critic = os.path.join(model_path, f"{model_name}_critic1")
    final_actor = os.path.join(model_path, f"{model_name}_actor")
    
    if os.path.exists(final_critic) and os.path.exists(final_actor):
        print(f"Final model already exists at {model_path}")
        return "final"
    
    # If no final model, look for checkpoint files with episode numbers - PRIORITY 2
    critic_pattern = os.path.join(model_path, f"{model_name}_critic1_*")
    actor_pattern = os.path.join(model_path, f"{model_name}_actor_*")
    
    critic_files = glob.glob(critic_pattern)
    actor_files = glob.glob(actor_pattern)
    
    print(f"Found {len(critic_files)} critic files: {[os.path.basename(f) for f in critic_files]}")
    print(f"Found {len(actor_files)} actor files: {[os.path.basename(f) for f in actor_files]}")
    
    if not critic_files or not actor_files:
        print(f"No checkpoint files found in {model_path}")
        return None
    
    # Extract episode numbers from checkpoint filenames
    critic_episodes = []
    actor_episodes = []
    
    # Escape the model name for regex since it may contain special characters like parentheses
    escaped_model_name = re.escape(model_name)
    
    for file in critic_files:
        basename = os.path.basename(file)
        match = re.search(rf"{escaped_model_name}_critic1_(\d+)$", basename)
        if match:
            critic_episodes.append(int(match.group(1)))
    
    for file in actor_files:
        basename = os.path.basename(file)
        match = re.search(rf"{escaped_model_name}_actor_(\d+)$", basename)
        if match:
            actor_episodes.append(int(match.group(1)))
    
    # Find common episode numbers (both critic and actor exist)
    common_episodes = set(critic_episodes) & set(actor_episodes)
    
    print(f"Critic episodes: {sorted(critic_episodes)}")
    print(f"Actor episodes: {sorted(actor_episodes)}")
    print(f"Common episodes: {sorted(common_episodes)}")
    
    if not common_episodes:
        print(f"No matching critic/actor checkpoint pairs found in {model_path}")
        return None
    
    latest_episode = max(common_episodes)
    print(f"Found latest checkpoint at episode {latest_episode} in {model_path}")
    return latest_episode

if os.path.exists(net_kwargs['model_path']) == False:
    os.makedirs(net_kwargs['model_path'], exist_ok=True)
    
# Check for existing models
latest_checkpoint = find_latest_checkpoint(net_kwargs['model_path'])

print(params)
sac = RLRNN(environment=env, **net_kwargs)

# Load existing model if found
# if latest_checkpoint == "completed":
#     print("Job already completed. Exiting.")
#     sys.exit(0)
if latest_checkpoint == "final" or latest_checkpoint is 'completed':
    print("Loading final model...")
    sac.load_model(net_kwargs['model_path'])
    print("Final model loaded successfully. Training will be skipped.")
    # Skip the optimization since final model already exists
    skip_training = True
elif latest_checkpoint is not None:
    print(f"Loading checkpoint from episode {latest_checkpoint}...")
    sac.load_model_ep(latest_checkpoint, net_kwargs['model_path'])
    print(f"Checkpoint loaded successfully. Training will resume from episode {sac.training_iter}.")
    print(f"Model state - training_iter: {sac.training_iter}, patience: {sac.patience}, prev_max: {sac.prev_max}")
    skip_training = False
else:
    print("No existing model found. Starting training from scratch.")
    skip_training = False

#imgpth = os.path.join(imgpth,sac.model_name + '_')

if not skip_training:
    sac.optimize_model(nits=8*sac.batch_size,save=net_kwargs['model_path'], max_patience=2500, prune = True, save_every=5)

# make env have longer dependencies to more realistically mimic the monkey experience
net_kwargs['max_steps'] = 500
env.reset_time = 500
sac.environment.reset_time = 500
sac.max_steps = 500




data = generate_data(sac, env,nits=1250)
# data = generate_data(sac, env,nits=5)


# compute winrate from data
episode_states, episode_actions, episode_rewards, episode_hiddens, RNNChoices, RLChoices = data

# Compute mutual information for model sequences
print("Computing mutual information for model sequences...")
model_mutual_infos = []

for i in range(len(episode_actions)):
    # Compute mutual information between model actions and computer actions for each episode
    monkey_actions = episode_actions[i].flatten()  # Model actions
    computer_actions = episode_states[i].flatten()  # Computer actions (environment states)
    
    # Compute mutual information with computer actions (interspersed sequences)
    mi_values = compute_mutual_information(monkey_actions, computer_actions, N=2, minlen=3, maxlen=10)
    model_mutual_infos.append(mi_values)

# Average mutual information across episodes
if model_mutual_infos:
    model_mutual_infos = np.array(model_mutual_infos)
    mean_mutual_info = np.mean(model_mutual_infos, axis=0)
    std_mutual_info = np.std(model_mutual_infos, axis=0)
    
    print(f"Model mutual information (mean ± std) across sequence lengths 3-10:")
    for i, (mean_mi, std_mi) in enumerate(zip(mean_mutual_info, std_mutual_info)):
        seq_len = i + 3  # Sequence lengths start from 3
        print(f"  Length {seq_len}: {mean_mi:.4f} ± {std_mi:.4f}")
else:
    mean_mutual_info = []
    std_mutual_info = []
    print("Warning: No mutual information computed (no episodes generated)")

# if np.mean(episode_rewards) <= .40:
#     # delete saved models   
#     os.system(f"rm -rf {net_kwargs['model_path']}")
#     exit()

pickle.dump(data, open(os.path.join(net_kwargs['model_path'],'data.p'), 'wb'))
order = 5


#instead of looking at win vs loss, what about trial immediately after a win vs after a loss


# fit_all_sess = logistic_regression.paper_logistic_regression_strategic(None,True, data=data,legend=True, return_model=True)
    
fit_all_sess, fit_errs = logistic_regression.fit_glr(data, order=order, a_order=2, r_order=1, err = True, model = True, labels = False, average = True)

fit_all_sess = {'action': fit_all_sess, 'err': fit_errs}
# save fit_all_sess in a text file in folder
# with open(os.path.join(net_kwargs['model_path'],'fit_all_sess.p'),'wb') as f: 
#     pickle.dump(fit_all_sess, f)

# now fit the sequence prediction for each monkey
print("Computing sequence prediction matching...")

# Since training is commented out, save the model first so sequence matching can load it
sac.save_model(net_kwargs['model_path'])

# Prepare model parameters for sequence matching
model_params = {
    'model_path': net_kwargs['model_path'],
    'model_name': sac.model_name,
    **net_kwargs
}


# Load monkey data to get trial counts (using the relative path)
if not os.path.exists(stitched_p):
    # Try absolute path as fallback
    # check if this exists
    if os.path.exists('/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'):
        stitched_p = '/Users/fmb35/Desktop/BG-PFC-RNN/figure_scripts/stitched_monkey_data.pkl'
    elif os.path.exists('/home/fmb35/project/BG-PFC-RNN/cluster_scripts/regraining/stitched_monkey_data.pkl'):
        stitched_p = '/home/fmb35/project/BG-PFC-RNN/cluster_scripts/regraining/stitched_monkey_data.pkl'
    elif os.path.exists('/home/fmb35/project_pi_jdm54/fmb35/BG-PFC-RNN/cluster_scripts/regraining/stitched_monkey_data.pkl'):
        stitched_p = '/home/fmb35/project_pi_jdm54/fmb35/BG-PFC-RNN/cluster_scripts/regraining/stitched_monkey_data.pkl'
    else:
        print("No stitched monkey data found")
        sys.exit(1)
    

data_monkeys = load_data(stitched_p)
data_monkeys = data_monkeys[data_monkeys['task'] == 'mp']


# Save mutual information results to individual model folder
mutual_info_results = {
    'mean_mutual_info': mean_mutual_info.tolist() if len(mean_mutual_info) > 0 else [],
    'std_mutual_info': std_mutual_info.tolist() if len(std_mutual_info) > 0 else [],
    'sequence_lengths': list(range(3, 11)),
    'raw_mutual_infos': [mi.tolist() for mi in model_mutual_infos] if len(model_mutual_infos) > 0 else [],
    'n_episodes': len(model_mutual_infos) if len(model_mutual_infos) > 0 else 0
}

with open(os.path.join(net_kwargs['model_path'],'mutual_information.p'),'wb') as f:
    pickle.dump(mutual_info_results, f)

# Extract model policy weights (alpha values from the mixing weights)
if hasattr(sac, 'policy') and hasattr(sac.policy, 'weights'):
    print(f"Model scaletype: {getattr(sac.policy, 'scaletype', 'Not found')}")
    print(f"Weights structure: {type(sac.policy.weights)}")
    print(f"Input weighting parameter: {net_kwargs['weighting']}")
    print(f"Original alphas stored in model: {getattr(sac.policy, 'alphas', 'Not found')}")
    
    if hasattr(sac.policy, 'scaletype') and sac.policy.scaletype == 'adaptive':
        # For adaptive case: extract alpha from each weight in ModuleList
        # Structure: MultLayerAdaptiveSimple -> ModuleList -> MultLayer objects
        print(f"Weights.weight type: {type(sac.policy.weights.weight)}")
        print(f"Number of weight modules: {len(sac.policy.weights.weight)}")
        
        policy_weights = []
        for i, w in enumerate(sac.policy.weights.weight):
            print(f"Weight module {i}: {type(w)}")
            
            # Try to get the original/raw weights before parametrization
            if hasattr(w, 'parametrizations') and 'weight' in w.parametrizations:
                # Access the original weight before SumOne parametrization
                raw_weight = w.parametrizations.weight.original
                print(f"Weight module {i} raw weight tensor: {raw_weight}")
                alpha = raw_weight[0].detach().cpu().numpy()
            else:
                # Fallback to parametrized weight
                print(f"Weight module {i} parametrized weight tensor: {w.weight}")
                alpha = w.weight[0].detach().cpu().numpy()
            
            alpha_rounded = round(float(alpha), 3)
            policy_weights.append(alpha_rounded)
            print(f"Extracted alpha for module {i}: {alpha_rounded}")
            
        fit_all_sess['policy_weights'] = policy_weights
        print(f"Added adaptive policy weights (alphas): {policy_weights}")
    else:
        # For non-adaptive case: extract single alpha value
        print(f"Non-adaptive weights object: {type(sac.policy.weights)}")
        
        # Try to get the original/raw weights before parametrization
        if hasattr(sac.policy.weights, 'parametrizations') and 'weight' in sac.policy.weights.parametrizations:
            raw_weight = sac.policy.weights.parametrizations.weight.original
            print(f"Non-adaptive raw weight tensor: {raw_weight}")
            alpha = raw_weight[0].detach().cpu().numpy()
        else:
            # Fallback to parametrized weight
            print(f"Non-adaptive parametrized weight tensor: {sac.policy.weights.weight}")
            alpha = sac.policy.weights.weight[0].detach().cpu().numpy()
            
        alpha_rounded = round(float(alpha), 3)
        fit_all_sess['policy_weights'] = alpha_rounded
        print(f"Added policy weight (alpha): {alpha_rounded}")
else:
    print("Warning: Could not find policy.weights in model")
    fit_all_sess['policy_weights'] = None

# Add mutual information results to fit_all_sess
fit_all_sess['mutual_information'] = {
    'mean': mean_mutual_info.tolist() if len(mean_mutual_info) > 0 else [],
    'std': std_mutual_info.tolist() if len(std_mutual_info) > 0 else [],
    'sequence_lengths': list(range(3, 11)),  # Lengths 3-10
    'raw_values': [mi.tolist() for mi in model_mutual_infos] if len(model_mutual_infos) > 0 else []
}
print(f"Added mutual information results: mean MI = {mean_mutual_info.tolist() if len(mean_mutual_info) > 0 else 'None'}")

# Create regrained directory if it doesn't exist
regrained_dir = 'regrained'
if not os.path.exists(regrained_dir):
    os.makedirs(regrained_dir)


model_perf = compute_single_model_perf(sac, stitched_p = data_monkeys)

with open(os.path.join(net_kwargs['model_path'],'model_perf.p'),'wb') as f:
    pickle.dump(model_perf, f)


# get idx and params from model_path
param_str = '(' + sac.model_name.split('(')[1].split(')')[0] + ')'
id_str = sac.model_name.split(')')[-1].split('_')[1:]


print(id_str) # blank, but shouldnt be

# now save this except in a larger file that's more centralized


with open(os.path.join(regrained_dir,'fit_all_sess.p'),'ab') as f:
    fit_all_sess['weight'] = net_kwargs['weighting']
    fit_all_sess['model_idx'] = param_str
    fit_all_sess['model_params'] = id_str
    # Add sequence prediction results if available
    fit_all_sess['sequence_prediction'] = model_perf
    fit_all_sess['mutual_information'] = mutual_info_results
    fit_all_sess['avg_reward'] = np.mean(episode_rewards)
    pickle.dump(fit_all_sess, f)


with open(os.path.join(net_kwargs['model_path'],'fit_all_sess.p'),'wb') as f:
    fit_all_sess['weight'] = net_kwargs['weighting']
    fit_all_sess['model_idx'] = param_str
    fit_all_sess['model_params'] = id_str
    # Add sequence prediction results if available
    fit_all_sess['sequence_prediction'] = model_perf
    fit_all_sess['mutual_information'] = mutual_info_results
    fit_all_sess['avg_reward'] = np.mean(episode_rewards)
    pickle.dump(fit_all_sess, f)
# Mark job as completed in centralized directory
# Create completed_models directory if it doesn't exist
completed_models_dir = 'completed_models'
if not os.path.exists(completed_models_dir):
    os.makedirs(completed_models_dir)

# Calculate average reward for completion summary
avg_reward = np.mean([np.mean(ep_rewards) for ep_rewards in episode_rewards])

# Create completion record as text file
model_basename = os.path.basename(net_kwargs['model_path'])
completion_file = os.path.join(completed_models_dir, f"{model_basename}.txt")

with open(completion_file, 'w') as f:
    f.write(f"Job completed: {model_basename}\n")
    f.write(f"Timestamp: {np.datetime64('now')}\n")
    f.write(f"Seed: {seed}\n")
    f.write(f"Model path: {net_kwargs['model_path']}\n")
    f.write(f"Model name: {sac.model_name}\n")
    f.write(f"Average reward: {avg_reward:.4f}\n")
    f.write(f"Training skipped: {skip_training}\n")
    f.write(f"Checkpoint loaded: {latest_checkpoint if latest_checkpoint not in ['final', 'completed'] else 'None'}\n")
    f.write(f"Final model existed: {latest_checkpoint == 'final'}\n")
    f.write(f"Episodes generated: {len(episode_rewards)}\n")
    f.write(f"Mutual info computed: {len(model_mutual_infos) > 0}\n")
    f.write(f"Weighting: {net_kwargs['weighting']}\n")

print(f"Job completed successfully!")
print(f"Completion record saved to: {completion_file}")
print(f"Average reward: {avg_reward:.4f}")
