Repository for the draft of my preprint.

Included is the current working draft of the paper, the package requirements, and a script to train and test these models (rnn_script.py). 

In each folder, we have:

# models

## rl_model.py
Implementations for the RL agents used in the paper. These implementations are a little different than standard, i.e. they must remember the history Q-values in order to be compatible with RNN model learning.

## rnn_model.py
Implementation for the RNN and RLRNN models used. The RNN is very similar to the 'PFC Meta RL' paper except with leaky vanilla RNNs and not LSTMs. The RLRNN additionally has an interface that works with any of the RL models in order to make decisions.

## misc_utils.py
Various utilities and wrappers for the aforementioned models.

# envs
Environments for the agents.

## binocdf.py
Python implementation of the binomial CDF that is faster than scipy.

## matching_pennies.py 
Python implementation of the matching pennies game. 

## matching_pennies_numba.py
Implementation of MP with numba JIT compilation. Use this one if possible, it is significantly faster as the bottleneck for learning seems to be the environment once sequence lengths get long.

## misc_opponents.py
[Deprecated] A couple of other RL-based opponents for the RNNs/RLRNNs to play against.

## mp_env.py
Environment for playing against the matching pennies opponent, as well as some (deprecated) alternative games and opponents..

# analysis_scripts
Contains code for the various analyses performed. 

## LLH_behavior_RL.py
Fits RL models to the data and analyzes their performance.

## WSLS_Analysis.py 
[Deprecated] Looks at Win-stay Lose-switch tendencies in behavior.

## entropy.py
Computes entropy and mutual information of decision sequences.

## logistic_regression.py
Fits logistic regressions to behavioral data using different schemes. Many different types of regressions are included for compatability with old analyses, but fit_glr() and general_logistic_regressors() are the ones to use for any future analyses.

## logistic_regression_archive.py
[Deprecated] Contains helpers for logistic_regression.py, included for compatability.

## model_prediction_sequence_matching.py
Computes the log likelihood of the next decision in a sequence under a greedy policy for the decisions for models. Contains single-threaded and multi-threaded implementations.

## population_coding.py
Contains various analyses at the neuronal level for the models.

## stationarity_and_randomness.py
Analyzes how stationary monkey and model behavior are.

## test_suite.py
Contains functions for generating data from RNNs/RLRNNs as well as some other deprecated analyses.

## yule_walker.py
Implementation of the yule-walker equations for fitting autoregressions.

# figure_scripts

These are mostly self explanatory, and additionally include some auxiliary and supplemental analyses. Note that some of these will not work as as the monkey data and model data are not included in this repo.
 



