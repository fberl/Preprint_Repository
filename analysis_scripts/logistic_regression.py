from scipy.optimize import minimize,least_squares
import numpy as np
import pandas as pd
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import analysis_scripts.yule_walker as yw
import sqlite3
import os
import matplotlib as mpl
from scipy import stats
from scipy.ndimage import median_filter
from matplotlib.ticker import MultipleLocator  
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['image.cmap'] = 'coolwarm'


#NUMBER OF UNIQUE REGRESSORS
numvars = 4



# Restored functions (used by reference)

def objective_function(theta, X, y, mask):
    m = len(y)
    l2 = .05
    l1 = .01
    if mask is not None:
        y = y[mask]
        X = X[mask]
    h = logit(X,theta[:-1],theta[-1])
    
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta**2) + l1*np.abs(theta).sum()
    try:
        Xaugmented = np.concatenate((X, np.ones((m, 1))), axis=1) + l2 * 2*theta + l1*np.sign(theta) #extra terms are for computing the gradient
    except: 
        Xaugmented = np.stack((X, np.ones((m))),axis=1) + l2 * 2*theta + l1*np.sign(theta)
    grad = (1 / m) * Xaugmented.T.dot(h - y)
    return cost, grad


# theta_fit and X_fit are the coefficients and regressors from the previous fit
# so that we don't modify them again


def objective_function_colinear_multiepisode(theta, X, theta_fit, bias, X_fit, y):
    l2 = .05
    l1 = .01

    theta_all = np.concatenate((theta_fit, theta)) if theta_fit.size > 0 else theta
    if len(X_fit) > 0:
        X_all = [np.hstack((X_fit[i], X[i])) if X_fit[i].size > 0 else X[i] for i in range(len(X))]
    else:
        X_all = X

    h = logit_multiepisode(X_all, theta_all, bias)
    cost = 0
    grad = np.zeros_like(theta)

    m = sum([len(y_i) for y_i in y])
    if m == 0:
        return 0, np.zeros_like(theta)

    for i in range(len(X)):
        # Clip values to avoid log(0)
        h[i] = np.clip(h[i], 1e-12, 1 - 1e-12)
        
        cost_i = (-y[i].T.dot(np.log(h[i])) - (1 - y[i]).T.dot(np.log(1 - h[i])))
        cost += cost_i

        if bias:
            Xaugmented = np.concatenate((X[i], np.ones((len(X[i]), 1))), axis=1)
            grad += Xaugmented.T.dot(h[i] - y[i])
        else:
            grad += X[i].T.dot(h[i] - y[i])

    # Normalize cost and gradient by total number of samples
    cost = cost / m
    grad = grad / m

    # Add regularization cost and gradient (only once)
    cost += l2 * np.sum(theta_all**2) + l1 * np.abs(theta_all).sum()
    grad += l2 * 2 * theta + l1 * np.sign(theta)

    return cost, grad


def objective_function_colinear_no_bias(theta, X, theta_fit,X_fit, y, mask):
    m = len(y)
    l2 = .05
    l1 = .01
    if mask is not None:
        y = y[mask]
        X = X[mask]
        X_fit = X_fit[mask]
    theta_all = np.concatenate((theta_fit,theta))
    X_all = np.concatenate((X_fit,X),axis=1)
    h = logit(X_all,theta_all,0) 
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta_all**2) + l1*np.abs(theta_all).sum()
    try:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta) # extra terms are for computing the gradient
    except:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta)
    
    
    grad = (1 / m) * Xaugmented.T.dot(h - y) # do the dimensions work out, or do i need to pad with zeros?
    return cost, grad


def objective_function_no_bias(theta, X, y, mask):
    m = len(y)
    l2 = .01
    l1 = 0
    if mask is not None:
        y = y[mask]
        X = X[mask]
        X_fit = X_fit[mask]
    h = logit(X,theta,0)
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta**2) + l1*np.abs(theta).sum()
    try:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta) # extra terms are for computing the gradient
    except:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta)
    
    
    grad = (1 / m) * Xaugmented.T.dot(h - y)
    return cost, grad

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(x,w,b):
    try:
        return sigmoid(np.dot(x,w) + b)
    except:
        try:
            return sigmoid(x*w + b)
        except ValueError:
            return logit_multiepisode(x,w,b)
def logit_multiepisode(x,w,b):
    try:
        y = [sigmoid(np.dot(x[i],w) + b) for i in range(len(x))]
    except:
        y = [sigmoid(x[i]*w + b) for i in range(len(x))]
    # Keep as list if arrays have different lengths, otherwise convert to numpy array
    try:
        return np.array(y)
    except ValueError:
        return np.array(y, dtype=object)

def objective_function_colinear(theta, X, theta_fit, bias,X_fit, y, mask):
    m = len(y)
    l2 = .05
    l1 = .01
    if mask is not None:
        y = y[mask]
        X = X[mask]
        X_fit = X_fit[mask]
    theta_all = np.concatenate((theta_fit,theta)) if theta_fit.size > 0 else theta
    X_all = np.hstack((X_fit,X)) if X_fit.size > 0 else X
    h = logit(X_all,theta_all,bias)
    
    # these are jagged arrays,

    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta_all**2) + l1*np.abs(theta_all).sum()
    # try:
    #     Xaugmented = np.concatenate((X, np.ones((m, 1))), axis=1) + l2 * 2*theta + l1*np.sign(theta) #extra terms are for computing the gradient
    # except: 
    #     Xaugmented = np.stack((X, np.ones((m))),axis=1) + l2 * 2*theta + l1*np.sign(theta)
    try:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta) # extra terms are for computing the gradient
    except:
        Xaugmented = X + l2 * 2*theta + l1*np.sign(theta)
    
    grad = (1 / m) * Xaugmented.T.dot(h - y)
    return cost, grad
    # return cost
    
# def objective_function_colinear_multiepisode(theta, X, theta_fit, bias,X_fit, y, mask):
#     m = len(y)
#     l2 = .05
#     l1 = .01
#     if mask is not None:
#         y = y[mask]
#         X = X[mask]
#         X_fit = X_fit[mask]
#     theta_all = np.concatenate((theta_fit,theta)) if theta_fit.size > 0 else theta
#     X_all = np.hstack((X_fit,X)) if X_fit.size > 0 else X
#     h = logit(X_all,theta_all,bias)
    
#     # these are jagged arrays,
    
#     cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + l2 * np.sum(theta_all**2) + l1*np.abs(theta_all).sum()
#     # try:
#     #     Xaugmented = np.concatenate((X, np.ones((m, 1))), axis=1) + l2 * 2*theta + l1*np.sign(theta) #extra terms are for computing the gradient
#     # except: 
#     #     Xaugmented = np.stack((X, np.ones((m))),axis=1) + l2 * 2*theta + l1*np.sign(theta)
#     try:
#         Xaugmented = X + l2 * 2*theta + l1*np.sign(theta) # extra terms are for computing the gradient
#     except:
#         Xaugmented = X + l2 * 2*theta + l1*np.sign(theta)
    
#     grad = (1 / m) * Xaugmented.T.dot(h - y)
#     return cost, grad
#     # return cost
def logistic_regression_no_bias(X, y,mask = None):
    try:
        n = X.shape[1]
    except:
        n = 1
    
    
    initial_theta = np.zeros(n)
    
    result = minimize(objective_function_no_bias, initial_theta, args=(X, y,mask), jac=True, method='CG')
    # result = least_squares(objective_function_nojac, initial_theta, args=(X, y), jac=True, method='lm')
    return result


def logistic_regression(X, y, bias=True, mask = None, colinear = True,order = 5):
    try:
        n = X.shape[1]
        # n = min(X.shape)
    except:
        n = 1
    
    if mask is not None:
        if isinstance(mask,list):
            mask = np.hstack(mask)
        mask = mask.astype(bool)
        
    initial_theta = np.zeros(n+bias)
    if colinear:
        if bias:
            result = logistic_regression_colinear(X, y, bias, mask, order)
        else:
            result = logistic_regression_colinear_nobias(X, y, bias, mask, order)
    else:
        if bias == True:
            result = minimize(objective_function, initial_theta, args=(X, y,mask), jac=True, method='BFGS')
        else:
            result = minimize(objective_function_no_bias, initial_theta, args=(X, y,mask), jac=True, method='BFGS')
    # result = least_squares(objective_function_nojac, initial_theta, args=(X, y), jac=True, method='lm')
    return result

def logistic_regression_colinear(X, y, bias=False, mask = None, order = 5):
    # fit regressors in order

    try:
        n = X.shape[1]
    except:
        n = 1
    
    
    if mask is not None:
        if isinstance(mask,list):
            mask = np.hstack(mask)
        mask = mask.astype(bool)
        
    initial_theta = np.zeros(n+bias)
    
    
    # needs to be fit sequentially. first bias term, then all terms at lag 1, then all terms at lag 2, etc
    # if bias == True:
    #     objf = objective_function_colinear
    # else:
    #     objf = objective_function_colinear_no_bias
    objf = objective_function_colinear
    
    
    # objective_function_colinear(theta, X, theta_fit, bias,X_fit, y, mask)
        
    res_list = []
    if bias:
        # first fit the bias term
        xi = np.ones((len(y),1))
        initial_theta = np.zeros(1)
        res = minimize(objf, initial_theta, args=(xi,np.array([]),0,np.array([]), y, mask), jac=True, method='BFGS').x
        # res_list.extend(res)
        bias_term = res
    else:
        bias_term = 0
    
    x_prev = np.array([])
    for i in range(0,order):
        # xi = X[:,i*(n//4):(i+1)*(n//4)] 
        # want to fit all terms with lag 0, then all terms with lag 1, etc
        # there are n sets of regressors, each with order terms. So we need the function to take in
        # this information,
        xi = X[:,i::order] # 

        # initial_theta = np.zeros((bias+n//4))
        initial_theta = np.zeros((n//order)) # this bias is for right vs left. What about strategy bias?
        res = minimize(objf, initial_theta, args=(xi,np.array(res_list),bias_term,x_prev, y, mask), jac=True, method='BFGS').x
        x_prev = np.concatenate([x_prev,xi],axis=1) if x_prev.size > 0 else xi
        res_list.extend(res)
        # result = minimize(objective_function, initial_theta, args=(X, y), jac=True, method='CG')

    result_col = {}
    # results are in the wrong order because of the way the regressors are fit. We need to reorder them
    res_list = list(np.array(res_list).reshape(order, -1).T.flatten())
        

    if bias:
        # bias should be the last term, so we need to add it to the end of the list
        # and remove the first term
        # res_list.append(res_list.pop(0))
        res_list.extend(bias_term) 
        
        objf = objective_function
    else:
        objf = objective_function_no_bias

    result_col['x'] = np.array(res_list)

    # now fit using this as a guess
    result = minimize(objf, np.array(res_list), args=(X, y,mask), jac=True, method='BFGS')
    result = result_col
    
    return result


def logistic_regression_colinear_multiepisode(X, y, bias=False, order = 5):
    
    # fit regressors in order
    # Keep y and X as lists if they contain variable-length arrays
    if not isinstance(y, np.ndarray):
        try:
            y = np.array(y, dtype=object)
        except:
            pass  # Keep as list if conversion fails
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X, dtype=object)
        except:
            pass  # Keep as list if conversion fails
    try:

        d = X.shape[0] # number of episodes
        n = X[0].shape[1]

    except:
        d = 1
        n = 1

        
    
    initial_theta = np.zeros(n+bias)
    
    
    # needs to be fit sequentially. first bias term, then all terms at lag 1, then all terms at lag 2, etc
    # if bias == True:
    #     objf = objective_function_colinear
    # else:
    #     objf = objective_function_colinear_no_bias
    if d > 1:
        objf = objective_function_colinear_multiepisode
    else:
        objf = objective_function_colinear
        mask = None
    
    
    objf = objective_function_colinear_multiepisode

    # objective_function_colinear(theta, X, theta_fit, bias,X_fit, y, mask)
        
    res_list = []
    if bias:
        # first fit the bias term
        # xi = np.ones((len(y),d))
        xi = [np.ones((len(y[i]))) for i in range(len(y))]
        initial_theta = np.zeros(1)
        res = minimize(objf, initial_theta, args=(xi,np.array([]),0,np.array([]), y), jac=True, method='BFGS').x
        # res_list.extend(res)
        bias_term = res
    else:
        bias_term = 0
    
    x_prev = np.array([])
    for i in range(0,order):
        xi = [X[j][:,i*(n//order):(i+1)*(n//order)] for j in range(len(X))]
        # want to fit all terms with lag 0, then all terms with lag 1, etc
        # there are n sets of regressors, each with order terms. So we need the function to take in
        # this information,
        # xi = X[:,i::order] # 

        # initial_theta = np.zeros((bias+n//4))
        initial_theta = np.zeros((n//order)) # this bias is for right vs left. What about strategy bias?
        res = minimize(objf, initial_theta, args=(xi,np.array(res_list),bias_term,x_prev, y), jac=True, method='BFGS').x
        if len(x_prev) > 0:
            x_prev = [np.concatenate([x_prev[j],xi[j]],axis=1) for j in range(len(x_prev))]
        else:
            x_prev = xi
        res_list.extend(res)
        # result = minimize(objective_function, initial_theta, args=(X, y), jac=True, method='CG')

    result_col = {}
    # results are in the wrong order because of the way the regressors are fit. We need to reorder them
    res_list = list(np.array(res_list).reshape(order, -1).T.flatten())
        

    if bias:
        # bias should be the last term, so we need to add it to the end of the list
        # and remove the first term
        # res_list.append(res_list.pop(0))
        res_list.extend(bias_term) 
        
        objf = objective_function
    else:
        objf = objective_function_no_bias

    result_col['x'] = np.array(res_list)

    # now fit using this as a guess
    # result = minimize(objf, np.array(res_list), args=(xi,np.array(res_list),bias_term,x_prev, y), jac=True, method='BFGS').x

    # result = minimize(objf, np.array(res_list), args=(X, y,None), jac=True, method='BFGS')
    result = result_col
    
    return result

def logistic_regression_colinear_nobias(X, y, bias=False, mask = None, order = 5):
    # fit regressors in order
    try:
        n = X.shape[1]
    except:
        n = 1
    
    if mask is not None:
        if isinstance(mask,list):
            mask = np.hstack(mask)
        mask = mask.astype(bool)
        
    initial_theta = np.zeros(n+bias)
    
    
    # needs to be fit sequentially. first bias term, then all terms at lag 1, then all terms at lag 2, etc
    if bias == True:
        objf = objective_function
    else:
        objf = objective_function_no_bias

        
    res_list = []
    for i in range(0,order):
        # xi = X[:,i*(n//4):(i+1)*(n//4)] 
        # want to fit all terms with lag 0, then all terms with lag 1, etc
        # there are n sets of regressors, each with order terms. So we need the function to take in
        # this information,
        xi = X[:,i::order]
        # initial_theta = np.zeros((bias+n//4))
        initial_theta = np.zeros((n//order+bias))
        res = minimize(objf, initial_theta, args=(xi, y, mask), jac=True, method='BFGS').x
        res_list.extend(res)
        # result = minimize(objective_function, initial_theta, args=(X, y), jac=True, method='CG')
    result_col = {}
    result_col['x'] = np.array(res_list)
    result = minimize(objf, np.array(res_list), args=(X, y,mask), jac=True, method='BFGS')



    # result = minimize(objf, initial_theta, args=(X, y,mask), jac=True, method='CG')
    
    # do this sequentially; fit each column then pass that back as the next initial theta
    # res_list = []
    # for i in range(0,n//5):
    #     xi = X[:,i*(n//4):(i+1)*(n//4)]
    #     initial_theta = np.zeros((bias+n//4))
    #     res = minimize(objf, initial_theta, args=(xi, y, mask), jac=True, method='BFGS').x
    #     res_list.extend(res)
    #     # result = minimize(objective_function, initial_theta, args=(X, y), jac=True, method='CG')
    # result_col = {}
    # result_col['x'] = np.array(res_list)
    # result = minimize(objf, np.array(res_list), args=(X, y,mask), jac=True, method='BFGS')
    return result

def logistic_regression_reduced(X, y, theta_a, mask = None):
    try:
        n = X.shape[1]
    except:
        n = 1
    
    if mask is not None:
        if isinstance(mask,list):
            mask = np.hstack(mask)
        mask = mask.astype(bool)
    
    initial_theta = np.zeros(n-len(theta_a))
    initial_theta = np.concatenate([theta_a,initial_theta])
    # make bounds theta_a for the ones with theta a
    bounds = [(theta_a[i]-1e-6,theta_a[i]+1e-6) for i in range(len(theta_a))] + [(None,None) for i in range(n-len(theta_a))]
    result = minimize(objective_function_reduced, initial_theta, args=(X, y,mask), jac=True, bounds=bounds, method='CG')

    # result = least_squares(objective_function_nojac, initial_theta, args=(X, y), jac=True, method='lm')
    return result


def logistic_regression_l2(X, y):
    n = X.shape[1]
    initial_theta = np.zeros(n+1)
    result = minimize(objective_function_l2, initial_theta, args=(X, y), method='CG')
    # result = least_squares(objective_function_nojac, initial_theta, args=(X, y), jac=True, method='lm')
    return result

#computes error from a fit 
def data_parse(actions, state, rewards, order=1):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {1, 0, -1} if left and not rewarded on prior trial, rewarded, or right not rewarded (basically, swap on win)
    numvars = 4
    
    
    data = np.zeros((len(actions) - order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #opponent choice
                    if state[i-k-1] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==2: #rewarded and stay
                    if rewards[i-k] == 1:
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==3: #rewarded and swap
                    if rewards[i-k] == 0:
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    data = data.reshape(data.shape[0],-1, order='C')
    return data

def data_parse_two_regressors(actions, state, rewards, order=1):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)

    numvars = 2
    
    data = np.zeros((len(actions) - order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #opponent choice
                    if state[i-k-1] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    data = data.reshape(data.shape[0],-1, order='C')
    return data


# take a model, generate data over many trials, and provide a logistic regression for RNN, RL, and combined
def query_behavioral(task="mp_trials"):
    query = """
    SELECT sessions.id, t.trial, t.monkey_choice, t.reward, t.computer_choice, sessions.animal, sessions.name, sessions.ord, sessions.year, sessions.month, sessions.day
        FROM {task} as t, {task} as tp, sessions WHERE
            tp.session_id == t.session_id AND -- Alignment
            t.session_id == sessions.id AND
            tp.trial == t.trial - 1 AND -- Select previous trial
            tp.session_id == t.session_id AND
            t.tr_good == 1 
    ;
    """.format(task=task)
    return query

def query_monkey_behavior(path = None):
    '''Gets monkey behavior for MP and Search for each opponent type face and each monkey'''    
    if path is None:
        con = sqlite3.connect("matching-pennies-lite.sqlite")
    else:
        con = sqlite3.connect(path)

    pdMP = pd.read_sql_query(query_behavioral(),con)
    pdMP['task'] = 'mp'
    
    pdS = pd.read_sql_query(query_behavioral(task='search_trials'),con)
    pdS['task'] = 'search'
    
    
    df = pd.concat([pdMP,pdS]).sort_values(by=['id','trial'])

    return df

def parse_monkey_behavior(df, order):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_monkey(sessdat['monkey_choice'].to_numpy(),sessdat['computer_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order)
        regressors.append(data)
    reg = np.vstack(regressors)
    return reg


def parse_monkey_behavior_reduced(df, order, vif = False, err = False):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_monkey_reduced(sessdat['monkey_choice'].to_numpy(),sessdat['computer_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order, vif = vif)
        regressors.append(data)
    if err == False:
        regressors = np.vstack(regressors)
    return regressors

def parse_monkey_behavior_strategic(df, order, vif = False, err = False):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_monkey_strategic(sessdat['monkey_choice'].to_numpy(),sessdat['computer_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order, vif = vif)
        regressors.append(data)
    if err == False:
        regressors = np.vstack(regressors)
    return regressors

def parse_monkey_behavior_minimal(df, order, vif = False, err = False):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_monkey_minimal(sessdat['monkey_choice'].to_numpy(),sessdat['computer_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order, vif = vif)
        regressors.append(data)
    if err == False:
        regressors = np.vstack(regressors)
    return regressors

def parse_monkey_behavior_combinatorial(df, order, err = False):
    # need to pad s.t. they're all the same length. if 
    regressors = []
    for session in df['id'].unique():
        sessdat = df[df['id'] == session].sort_values(by=['id','trial'])
        data = data_parse_combinatorial(sessdat['monkey_choice'].to_numpy(),
                          sessdat['reward'].to_numpy(), order = order)
        regressors.append(data)
    if err == False:
        regressors = np.vstack(regressors)
    return regressors


def data_parse_monkey(actions, state, rewards, order=1):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {-1, 0, 1} if left and not rewarded on prior trial, rewarded, or right rewarded (basically, swap on win)
    
    
    data = np.zeros(( len(actions)- order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #opponent choice
                    if state[i-k-1] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==2: #rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==3: #not rewarded and swap
                    if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1

    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    data = data.reshape(data.shape[0],-1, order='C')
    return data

def data_parse_monkey_reduced(actions, state, rewards, order=1, vif = False):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {-1, 0, 1} if left and not rewarded on prior trial, rewarded, or right rewarded (basically, swap on win)
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    numvars = 3
    data = np.zeros(( len(actions)- order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==2: #not rewarded and swap
                    if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1

    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    if vif == True:
        for order in range(order):
            vif, rsq = compute_vif(data[:,:,order], bias=True)
            print('order: {}, Rsqs : {}'.format(order, rsq))
            print(vif)
    data = data.reshape(data.shape[0],-1, order='C')
    return data

def data_parse_monkey_strategic(actions, state, rewards, order=1, vif = False):
    
    # variables of interest: 
    # {-1, 0, 1} if left and not rewarded on prior trial,  rewarded, or right not rewarded (basically, stay on win)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right not rewarded (basically, switch on loss)
    # {-1, 0, 1} if left and not rewarded on prior trial,  rewarded, or right not rewarded (basically, switch on win)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right not rewarded (basically, stay on loss)
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    numvars = 4
    data = np.zeros(( len(actions)- order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==3: #not rewarded and stay
                    if rewards[i-k] == 0 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j == 2: # rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==0: #rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==1: #not rewarded and swap
                    if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1

    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    if vif == True:
        for order in range(order):
            vif, rsq = compute_vif(data[:,:,order], bias=True)
            print('order: {}, Rsqs : {}'.format(order, rsq))
            print(vif)
    data = data.reshape(data.shape[0],-1, order='C')
    return data


def data_parse_monkey_minimal(actions, state, rewards, order=1, vif = False):
    
    # variables of interest: 
    # {-1,1} for agent choice in previous trial (order 1-n) 
    # {-1,1} for computer choice in previous trial (order 1-n)
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {-1, 0, 1} if left and not rewarded on prior trial, rewarded, or right rewarded (basically, swap on win)
    
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    numvars = 2
    data = np.zeros(( len(actions)- order,numvars,order))
    for i in range(order,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #rewarded and stay
                    if state[i-k-1] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1


    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    if vif == True:
        for order in range(order):
            vif, rsq = compute_vif(data[:,:,order], bias=True)
            print('order: {}, Rsqs : {}'.format(order, rsq))
            print(vif)
    data = data.reshape(data.shape[0],-1, order='C')
    return data

def create_padded_matrix(df, max_len):
    '''creates a padded matrix from a dataframe'''
    sessions = df['id'].unique()
    padded = np.zeros((len(sessions),max_len))
    for i in range(padded.shape[0]):
        padded[i,:len(df[df['id'] == sessions[i]])] = df[df['id'] == sessions[i]]['monkey_choice'].to_numpy() * 2 - 1
    return padded
        
# def create_stacked_matrix(df, max_len):
#     '''creates a padded matrix from a dataframe'''
#     sessions = df['id'].unique()
#     padded = np.zeros((len(sessions),max_len))
#     sess = 
#     for i in range(padded.shape[0]):
#         padded[i,:len(df[df['id'] == sessions[i]])] = df[df['id'] == sessions[i]]['monkey_choice'].to_numpy() * 2 - 1
#     return padded
        

def create_order_data(df,order, err = False):
    '''creates a padded matrix from a dataframe'''
    sessions = df['id'].unique()
    data = []
    for s in sessions:
        data.append(df[df['id'] == s]['monkey_choice'].to_numpy()[order:])
    if err == False:
        data = np.concatenate(data)
    return data
        

def create_order_data_combinatorial(df,order, err=False):
    '''creates a padded matrix from a dataframe'''
    sessions = df['id'].unique()
    data = []
    for s in sessions:
        data.append(df[df['id'] == s]['monkey_choice'].to_numpy()[order:])
    if err == False:
        data = np.concatenate(data)
    return data
        

def create_choice_list(df):
    '''creates a padded matrix from a dataframe'''
    sessions = df['id'].unique()
    data = []
    for s in sessions:
        data.append(df[df['id'] == s]['monkey_choice'].to_numpy())
    return data
        
    
'''fits logistic regressions for each monkey'''
def monkey_logistic_regression(df = None, db_path=None, order=5, monkey=None, task = None, plot=True):
    
    if df is None:
        monkey_dat = query_monkey_behavior(db_path)
    else:
        monkey_dat = df
    if not isinstance(task,list):
        task = [task]
    if not isinstance(monkey,list):
        monkey = [monkey]
    
    if isinstance(order,int):
        order = [order]
    
    
    task_dict = {}
    for t in task:
        task_dat = monkey_dat[monkey_dat['task'] == t]
        monkey_dict = {}
        for mk in monkey:
            iter_dat = task_dat[(task_dat['animal'] == mk)]
            all_sols = {}
            max_sess_len  = iter_dat['id'].value_counts().max()
            if plot == True:
                plot_correlation_analysis_monkey(iter_dat, max_ord = 12)
            # monkeyChoices_stacked = create_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            # monkeyChoices_stacked = creatSe_padded_matrix(iter_dat, max_sess_len) #cant use 0 for padding if we're going to use it as a variable
            monkeyChoices_all = create_choice_list(iter_dat)
            for ord in order:
                regressors = parse_monkey_behavior(iter_dat,ord)
                sols = {}
                all_actions = create_order_data(iter_dat,ord)
                
                action_regression = logistic_regression(regressors, all_actions)
                sols['action'] = action_regression.x
                
                # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                # ax1.set_xlabel(r'\theta^T x + b')
                # action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
                # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                # ax1.set_title(r'Monkey Actions regression. $accuracy = {}$'.format(acc))
                # ax1.scatter(action_fit, all_actions)
                # smigmoid = np.linspace(min(action_fit),max(action_fit),100)
                # ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                # ax1.set_ylabel('action')
                
                if plot == False:
                    return action_regression.x
                
                fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))   
                ax1.set_xlabel(r'$\theta^T x + b$')
                action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
                # action_fit = action_fit[~np.isnan(action_fit)]
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                ax1.set_title(r'Monkey {} Actions regression: accuracy = {:.3f}'.format(mk,acc))
                smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
                ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
                ax1.set_ylabel('action')
                bins = np.linspace(min(action_fit),max(action_fit),20)
                hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
                bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
                hist1 = hist1[0]
                hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
                bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
                hist2 = hist2[0]
                kde1 = stats.gaussian_kde(action_fit[all_actions == 0])(smigmoid) * .1
                kde2 = stats.gaussian_kde(action_fit[all_actions == 1])(smigmoid) *.1  
                ax1.set_xlim(min(bins),max(bins))
                ax1.plot(smigmoid,kde1, color = 'b')
                ax3 = ax1.twinx()
                ax3.invert_yaxis()
                ax3.plot(smigmoid,kde2, color = 'b')        
                ax1.axvline(0,0)        
                ax1.set_ylim(0,1)
                ax3.set_ylim(1,0)
                
                ax2.set_xlabel(r'trial number')
                action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
                acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
                xord = np.arange(1,1+ord)
                ax2.plot(xord,sols['action'][:ord])
                ax2.plot(xord,sols['action'][ord:2*ord])
                ax2.plot(xord,sols['action'][2*ord:3*ord])
                ax2.plot(xord,sols['action'][3*ord:4*ord])
                ax2.set_title(r'Monkey {} Action Regression Coefficients'.format(mk))
                ax2.set_ylabel('coefficient')
                ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
                ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

                
                
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
                ax1.set_title('monkey {} {} behavior AR coefficients'.format(mk,t))
                ax2.set_title('monkey {} {} behavior PACF'.format(mk,t))   
                ax1.set_xlabel('lag')
                ax2.set_xlabel('lag')
                ax1.set_ylabel('coefficient')
                for i in range(len(monkeyChoices_all)):
                    try:
                        acf = yw.yule_walker(monkeyChoices_all[i],12)
                        if np.abs(sum(acf)) < 1e4:
                            ax1.plot(xord,acf,alpha=.02, c='b')
                    except:
                        continue
                    try:
                        pacf = yw.pacf(monkeyChoices_all[i],12)
                        if np.abs(sum(pacf)) < 1e4:
                            ax2.plot(xord,pacf,alpha=.02, c='b')
                    except:
                        continue
        
                if len(order) == 1:
                    all_sols = sols
                else:
                    all_sols[ord] = sols
            monkey_dict[mk] = all_sols
        task_dict[t] = monkey_dict
        
    plt.show()

    return task_dict
def fit_vif(X,y,bias):
    if bias:
        X = np.hstack([X,np.ones((X.shape[0],1))])
    fit = np.linalg.inv(X.T @ X) @ X.T @ y
    Rsq = 1 - np.sum((y - X @ fit)**2) / np.sum((y - np.mean(y))**2)
    vif = 1 / (1-Rsq)
    return vif, fit, Rsq
    

def compute_vif(regressors, bias = True):
    # step one: For each regressor, compute a regression using the other regessors
    # step two: VIF_i = 1 / (1-R^2_i)
    vifs = []
    rsqs = []
    for i in range(regressors.shape[1]):
        xind = np.ones(regressors.shape[1], dtype='bool')
        xind[i] = False
        x = regressors[:,xind]
        y = regressors[:,i]
        vif, fit, rsq  = fit_vif(x,y,bias)
        rsqs.append(rsq)
        vifs.append(vif)
    return vifs, rsqs
        
    

# def vif_analysis(df = None, order =5, monkey = None, task = None, combinatorial = False, model = None, env = None):
#     '''computes variation inflation factor for each subsequent regressor'''
#     assert (df is not None) or ((model is not None) and (env is not None)), 'need either data or a model and environment'
    

def plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards,  max_ord = 12, save = ''):
    regressors = []
    for i in range(len(episode_actions)):
        # regressors.append(data_parse(episode_actions[i], episode_states[i], episode_rewards[i], max_ord))
        regressors.append(data_parse_sliding_WSLS(episode_actions[i],  episode_rewards[i], max_ord))

    regressors = np.vstack(regressors)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    fig.suptitle('Correlation Analysis of Model Regressors')
    ax1.set_title('Agent Choice')
    ax2.set_title('Computer Choice')
    ax3.set_title('Win Stay')
    ax4.set_title('Lose Switch')
    
    ax1.set_xlabel('order')
    ax1.set_ylabel('order')
    ax2.set_xlabel('order')
    ax2.set_ylabel('order')
    ax3.set_xlabel('order')
    ax3.set_ylabel('order')
    ax4.set_xlabel('order')
    ax4.set_ylabel('order')
    
    all_regressors = []
    
    labels = [str(i) for i in range(1,max_ord+1)]
    axs = [ax1, ax3, ax4]
    for i in range(len(axs)):
        corrmat = np.zeros((max_ord,max_ord))
        all_regressors.append(regressors[:,i*max_ord:(i+1)*max_ord])
        for j in range(max_ord):
            for k in range(max_ord):
                corrmat[j,k] = np.corrcoef(regressors[:,i*max_ord+j].ravel(),regressors[:,i*max_ord+k].ravel())[0,1]
        im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
        axs[i].set_xticklabels(['']+ labels)
        axs[i].set_yticklabels(['']+ labels)    
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im,cax= cbar_ax)
    
    if save != '':
        plt.savefig(save + '_RegressorAutoCorrelation.png')
    
    # fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
    
    # axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    # labels1 = ['Agent Choice', 'Agent Choice', 'Agent Choice', 'Computer Choice','Computer Choice', 'Win Stay']
    # labels2 = ['Computer Choice', 'Win Stay', 'Lose Switch', 'Win Stay', 'Lose Switch', 'Lose Switch']
    # correlator1 = [all_regressors[0], all_regressors[0], all_regressors[0], all_regressors[1], all_regressors[1], all_regressors[2]]
    # correlator2 = [all_regressors[1], all_regressors[2], all_regressors[3], all_regressors[2], all_regressors[3], all_regressors[3]]
    # for i in range(len(axs)):
    #     axs[i].set_xlabel('order')
    #     axs[i].set_ylabel('order')
    #     axs[i].set_title('{} vs {}'.format(labels1[i],labels2[i]))
    #     corrmat = np.zeros((max_ord,max_ord))
    #     for j in range(max_ord):
    #         for k in range(max_ord):
    #             corrmat[j,k] = np.corrcoef(correlator1[i][:,j].ravel(),correlator2[i][:,k].ravel())[0,1]
    #     im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
    #     axs[i].set_xticklabels(['']+ labels)
    #     axs[i].set_yticklabels(['']+ labels)
    #     axs[i].xaxis.set_major_locator(MultipleLocator(1))
    #     axs[i].yaxis.set_major_locator(MultipleLocator(1))
    
    # if save != '':
    #     plt.savefig(save+'_RegressorCrossCorrelation.png')
        
def plot_correlation_analysis_monkey(dataframe,  max_ord = 12):
    regressors = parse_monkey_behavior(dataframe, max_ord)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    fig.suptitle('Correlation Analysis of Monkey {} Regressors'.format(dataframe['animal'].unique()[0]))
    ax1.set_title('Agent Choice')
    ax2.set_title('Computer Choice')
    ax3.set_title('win stay')
    ax4.set_title('Lose Switch')
    
    ax1.set_xlabel('order')
    ax1.set_ylabel('order')
    ax2.set_xlabel('order')
    ax2.set_ylabel('order')
    ax3.set_xlabel('order')
    ax3.set_ylabel('order')
    ax4.set_xlabel('order')
    ax4.set_ylabel('order')

    all_regressors = []

    
    labels = [str(i) for i in range(1,max_ord+1)]
    axs = [ax1, ax2, ax3, ax4]
    for i in range(len(axs)):
        corrmat = np.zeros((max_ord,max_ord))
        all_regressors.append(regressors[:,i*max_ord:(i+1)*max_ord])
        for j in range(max_ord):
            for k in range(max_ord):
                corrmat[j,k] = np.corrcoef(regressors[:,i*max_ord+j].ravel(),regressors[:,i*max_ord+k].ravel())[0,1]
        im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
        axs[i].set_xticklabels(['']+ labels)
        axs[i].set_yticklabels(['']+ labels)
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im,cax= cbar_ax)
    
    
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
    
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    labels1 = ['Agent Choice', 'Agent Choice', 'Agent Choice', 'Computer Choice','Computer Choice', 'Win Stay']
    labels2 = ['Computer Choice', 'Win Stay', 'Lose Switch', 'Win Stay', 'Lose Switch', 'Lose Switch']
    correlator1 = [all_regressors[0], all_regressors[0], all_regressors[0], all_regressors[1], all_regressors[1], all_regressors[2]]
    correlator2 = [all_regressors[1], all_regressors[2], all_regressors[3], all_regressors[2], all_regressors[3], all_regressors[3]]
    for i in range(len(axs)):
        axs[i].set_xlabel('order')
        axs[i].set_ylabel('order')
        axs[i].set_title('{} vs {}'.format(labels1[i],labels2[i]))
        corrmat = np.zeros((max_ord,max_ord))
        for j in range(max_ord):
            for k in range(max_ord):
                corrmat[j,k] = np.corrcoef(correlator1[i][:,j].ravel(),correlator2[i][:,k].ravel())[0,1]
        im = axs[i].matshow(corrmat,vmin=-1,vmax=1)
        axs[i].set_xticklabels(['']+ labels)
        axs[i].set_yticklabels(['']+ labels)
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
def model_logistic_regression_pregenerated(model, data, order=5, save = '', bias = False, err = False, sliding = None):

    episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, RLChoices = data

    plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards, save = save)
    all_sols = {}
    if isinstance(order,int):
        order = [order]
    for ord in order:
        sols = {}
        ord = min(ord, len(episode_actions[0]) - 1)        
        if model.policy.RL is not None:
            pfC = np.vstack(PFCChoices)[:,ord:]
            bgC = np.vstack(RLChoices)[:,ord:]
        all_actions = np.hstack(episode_actions).T[:,ord:]
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))        
        if not err:
            all_actions = all_actions.ravel()
            pfC = pfC.ravel()
            bgC = bgC.ravel()
            regressors = np.vstack(regressors)
            action_regression = logistic_regression(regressors, all_actions,bias=bias)
            sols['action'] = action_regression.x
            if model.policy.RL is not None:
                PFCChoices = np.vstack(PFCChoices)
                RLChoices = np.vstack(RLChoices)
                
                pfc_regression = logistic_regression(regressors,pfC, bias = bias)
                bg_regression = logistic_regression(regressors,bgC, bias = bias)
                sols['pfc'] = pfc_regression.x
                sols['bg'] = bg_regression.x
        else:
            if sliding is not None:
                fits = [] 
                bgF = []
                pfcF = []
                subregs = []
                subacts = []
                subPFCs = []
                subBGs = []
                for index in range(len(regressors)):
                    regs = regressors[index]
                    acts = all_actions[index]
                    bgs = bgC[index]
                    pfcs =pfC[index]
                    
                    for i in range(len(acts) - sliding):
                        fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=bias).x)
                        subregs.append(regs[i:i+sliding])
                        subacts.append(acts[i:i+sliding])
                        subBGs.append(bgs[i:i+sliding])
                        subPFCs.append(pfcs[i:i+sliding])
                        pfcF.append(logistic_regression(subregs[-1],subPFCs[-1],bias=bias).x)
                        bgF.append(logistic_regression(subregs[-1], subBGs[-1],bias=bias).x)
                fits = np.array(fits)
                bgF = np.array(bgF)
                pfcF = np.array(pfcF)
                sols['action'] = np.mean(fits,axis=0).squeeze()
                sols['err'] = np.std(fits,axis=0).squeeze()
                sols['pfc'] = np.mean(pfcF,axis=0).squeeze()
                sols['pfc_err'] = np.std(pfcF,axis=0).squeeze()
                sols['bg'] = np.mean(bgF, axis=0).squeeze()
                sols['bg_err'] = np.std(bgF,axis=0).squeeze()
                regressors = np.array(subregs)
                all_actions = np.array(subacts) # do i need to weight this somehow
                

                    
            else:
                fits = []
                lens = []
                pfcF = []
                bgF = []
                for index in range(len(regressors)):
                    fits.append(logistic_regression(regressors[index],all_actions[index],bias=bias).x)
                    lens.append(len(all_actions[index]))
                    if model.policy.RL is not None:
                        pfcF.append(logistic_regression(regressors[index],pfC[index],bias=bias).x)
                        bgF.append(logistic_regression(regressors[index],bgC[index],bias=bias).x) 
                fits = np.array(fits)
                bgF = np.array(bgF)
                pfcF = np.array(pfcF)
                sols['action'] = np.average(fits,axis=0,weights=lens).squeeze()
                sols['err'] = np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens)).squeeze()
                sols['pfc'] = np.average(pfcF,axis=0,weights=lens).squeeze()
                sols['pfc_err'] = np.sqrt(np.average((sols['pfc'] - pfcF)**2,axis=0,weights=lens)).squeeze()
                sols['bg'] = np.average(bgF,axis=0,weights=lens).squeeze()
                sols['bg_err'] = np.sqrt(np.average((sols['bg'] - bgF)**2,axis=0,weights=lens)).squeeze()
            if bias:
                action_fit = np.hstack([regressors[index] @ fits[index][:-1] + fits[index][-1] for index in range(len(fits))])
                if model.policy.RL is not None:
                    pfc_fit = np.hstack([regressors[index] @ pfcF[index][:-1] + pfcF[index][-1] for index in range(len(fits))])
                    bg_fit = np.hstack([regressors[index] @ bgF[index][:-1] +bgF[index][-1]  for index in range(len(fits))])
            else:
                action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
                if model.policy.RL is not None:
                    pfc_fit = np.hstack([regressors[index] @ pfcF[index] for index in range(len(fits))])
                    bg_fit = np.hstack([regressors[index] @ bgF[index] for index in range(len(fits))])
            all_actions = np.hstack(all_actions)
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
        
        if bias:  
            ax1.set_xlabel(r'$\theta^T x + b$')
            if not err:
                action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        else:
            ax1.set_xlabel(r'$\theta^T x$')
            if not err:
                action_fit = regressors @ sols['action']
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
        smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
        ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        ax1.set_ylabel('action')
        bins = np.linspace(min(action_fit),max(action_fit),20)
        hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        hist1 = hist1[0]
        hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        hist2 = hist2[0]
        kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
        kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
        ax1.set_xlim(min(bins),max(bins))
        ax1.plot(smigmoid,kde1, color = 'b')
        ax3 = ax1.twinx()
        ax3.invert_yaxis()
        ax3.plot(smigmoid,kde2, color = 'b')        
        ax1.axvline(0,0)        
        ax1.set_ylim(0,1)
        ax3.set_ylim(1,0)
        ax3.set_yticks([0,1])
        ax3.set_yticklabels(['Right','Left'])
        
        xord = np.arange(1,1+ord)

        ax2.set_xlabel(r'trial number')
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        reggy = ['agent choice', 'win stay', 'lose switch']
        if not err:
            for i in range(len(sols['action'])//ord):
                ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
        else:
            for i in range(len(sols['action'])//ord):
                prop_cycle = plt.rcParams['axes.prop_cycle']
                colors = prop_cycle.by_key()['color']
                ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
                ax2.fill_between(xord,sols['action'][i*ord:(i+1)*ord] - sols['err'][i*ord:(i+1)*ord], sols['action'][i*ord:(i+1)*ord]+sols['err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
        ax2.set_title(r'action regression coefficients'.format(acc))
        ax2.set_ylabel('coefficient')
        ax2.legend()
        # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
        # ax2.legend(['agent choice', 'win stay', 'lose switch'])
        ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
        if save != '':
            plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
        
        if model.policy.RL is not None:
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
            if bias:
                ax1.set_xlabel(r'$\theta^T x + b$')
                if not err:
                    pfc_fit = regressors @ sols['pfc'][:-1] + sols['pfc'][-1]
            else:
                ax1.set_xlabel(r'$\theta^T x$')
                if not err:
                    pfc_fit = regressors @ sols['pfc']
            acc = np.mean(np.round(sigmoid(pfc_fit)) == all_actions)
            ax1.set_title(r'Network PFC Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(pfc_fit),max(pfc_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(pfc_fit),max(pfc_fit),20)
            hist1 = np.histogram(pfc_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(pfc_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(pfc_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(pfc_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])
            ax2.set_xlabel(r'trial number')
            # action_fit = regressors @ sols['pfc'][:-1] + sols['pfc'][-1]
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            reggy = ['agent choice', 'win stay', 'lose switch']
            if not err:
                for i in range(len(sols['pfc'])//ord):
                    ax2.plot(xord,sols['pfc'][i*ord:(i+1)*ord], label = reggy[i])
            else:
                for i in range(len(sols['pfc'])//ord):
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    colors = prop_cycle.by_key()['color']
                    ax2.plot(xord,sols['pfc'][i*ord:(i+1)*ord], label = reggy[i])
                    ax2.fill_between(xord,sols['pfc'][i*ord:(i+1)*ord] - sols['pfc_err'][i*ord:(i+1)*ord], sols['pfc'][i*ord:(i+1)*ord]+sols['pfc_err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
            ax2.legend()
            ax2.set_title(r'PFC regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            # ax2.legend(['agent choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

            # fig.show()

            if save != '':
                plt.savefig(save+'_PFCLogisticRegression_{}.png'.format(ord))
        
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
            if bias:
                ax1.set_xlabel(r'$\theta^T x + b$')
                if not err:
                    bg_fit = regressors @ sols['bg'][:-1] + sols['bg'][-1]
            else:
                ax1.set_xlabel(r'$\theta^T x$')
                if not err:
                    bg_fit = regressors @ sols['bg']
            acc = np.mean(np.round(sigmoid(bg_fit)) == all_actions)
            ax1.set_title(r'Network RL Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(bg_fit),max(bg_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(bg_fit),max(bg_fit),20)
            hist1 = np.histogram(bg_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(bg_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(bg_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(bg_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])   
            
            ax2.set_xlabel(r'trial number')
            # acc = np.mean(np.round(sigmoid(bg_fit)) == all_actions)                
            if not err:
                for i in range(len(sols['bg'])//ord):
                    ax2.plot(xord,sols['bg'][i*ord:(i+1)*ord], label = reggy[i])
            else:
                for i in range(len(sols['bg'])//ord):
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    colors = prop_cycle.by_key()['color']
                    ax2.plot(xord,sols['bg'][i*ord:(i+1)*ord], label = reggy[i])
                    ax2.fill_between(xord,sols['bg'][i*ord:(i+1)*ord] - sols['bg_err'][i*ord:(i+1)*ord], sols['bg'][i*ord:(i+1)*ord]+sols['bg_err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
            ax2.legend()
            ax2.set_title(r'RL regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            # ax2.legend(['agent choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
            if save != '':
                plt.savefig(save+'_BGLogisticRegression_{}.png'.format(ord))
                
        if len(order) == 1:
            all_sols = sols
        else:
            all_sols[ord] = sols
        if save == '':
            plt.show()
        
               
def model_logistic_regression_combinatorial(model, data, order=5, save = '', bias = False):

    episode_states, episode_actions, episode_rewards, episode_hiddens, PFCChoices, RLChoices = data

    plot_correlation_analysis_model(episode_actions,episode_states,episode_rewards, save = save)
    all_sols = {}
    if isinstance(order,int):
        order = [order]
    for ord in order:
        sols = {}
        ord = min(ord, len(episode_actions[0]) - 1)
        if model.policy.RL is not None:
            pfC = np.vstack(PFCChoices)[:,ord:].ravel()
            bgC = np.vstack(RLChoices)[:,ord:].ravel()
        all_actions = np.hstack(episode_actions).T[:,ord:].ravel()
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_combinatorial(episode_actions[i], episode_rewards[i], ord))
        regressors = np.vstack(regressors)
        action_regression = logistic_regression(regressors, all_actions,bias=bias)
        sols['action'] = action_regression.x

        if model.policy.RL is not None:
            PFCChoices = np.vstack(PFCChoices)
            RLChoices = np.vstack(RLChoices)
            
            pfc_regression = logistic_regression(regressors,pfC, bias = bias)
            bg_regression = logistic_regression(regressors,bgC, bias = bias)
            sols['pfc'] = pfc_regression.x
            sols['bg'] = bg_regression.x
            

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
        if bias:  
            ax1.set_xlabel(r'$\theta^T x + b$')
            action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
        else:
            ax1.set_xlabel(r'$\theta^T x$')
            action_fit = regressors @ sols['action']
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
        smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
        ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
        ax1.set_ylabel('action')
        bins = np.linspace(min(action_fit),max(action_fit),20)
        hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
        bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
        hist1 = hist1[0]
        hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
        bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
        hist2 = hist2[0]
        kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
        kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
        ax1.set_xlim(min(bins),max(bins))
        ax1.plot(smigmoid,kde1, color = 'b')
        ax3 = ax1.twinx()
        ax3.invert_yaxis()
        ax3.plot(smigmoid,kde2, color = 'b')        
        ax1.axvline(0,0)        
        ax1.set_ylim(0,1)
        ax3.set_ylim(1,0)
        ax3.set_yticks([0,1])
        ax3.set_yticklabels(['Right','Left'])
        
        xord = np.arange(1,1+ord)

        ax2.set_xlabel(r'trial number')
        acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
        for i in range(len(sols['action'])//ord):
            ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], alpha = .3)
        ax2.set_title(r'action regression coefficients'.format(acc))
        ax2.set_ylabel('coefficient')
        # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
        # ax2.legend(['agent choice', 'win stay', 'lose switch'])
        ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
        if save != '':
            plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
        
        if model.policy.RL is not None:


            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))   
            if bias:
                ax1.set_xlabel(r'$\theta^T x + b$')
                action_fit = regressors @ sols['pfc'][:-1] + sols['pfc'][-1]
            else:
                ax1.set_xlabel(r'$\theta^T x$')
                action_fit = regressors @ sols['pfc']
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax1.set_title(r'Network PFC Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(action_fit),max(action_fit),20)
            hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])   
            
            ax2.set_xlabel(r'trial number')
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            for i in range(len(sols['pfc'])//ord):
                ax2.plot(xord,sols['pfc'][i*ord:(i+1)*ord], alpha = .3)
            ax2.set_title(r'PFC regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            # ax2.legend(['agent choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)

            # fig.show()

            if save != '':
                plt.savefig(save+'_PFCLogisticRegression_{}.png'.format(ord))
        
            
            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
            if bias:
                ax1.set_xlabel(r'$\theta^T x + b$')
                action_fit = regressors @ sols['bg'][:-1] + sols['bg'][-1]
            else:
                ax1.set_xlabel(r'$\theta^T x$')
                action_fit = regressors @ sols['bg']
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            ax1.set_title(r'Network RL Implied Action regression: accuracy = {:.3f}'.format(acc))
            smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
            ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
            ax1.set_ylabel('action')
            bins = np.linspace(min(action_fit),max(action_fit),20)
            hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
            bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
            hist1 = hist1[0]
            hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
            bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
            hist2 = hist2[0]
            kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
            kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
            ax1.set_xlim(min(bins),max(bins))
            ax1.plot(smigmoid,kde1, color = 'b')
            ax3 = ax1.twinx()
            ax3.invert_yaxis()
            ax3.plot(smigmoid,kde2, color = 'b')        
            ax1.axvline(0,0)        
            ax1.set_ylim(0,1)
            ax3.set_ylim(1,0)
            ax3.set_yticks([0,1])
            ax3.set_yticklabels(['Right','Left'])   
            
            ax2.set_xlabel(r'trial number')
            acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
            for i in range(len(sols['bg'])//ord):
                ax2.plot(xord,sols['bg'][i*ord:(i+1)*ord], alpha = .3)
            ax2.set_title(r'RL regression coefficients'.format(acc))
            ax2.set_ylabel('coefficient')
            # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
            # ax2.legend(['agent choice', 'win stay', 'lose switch'])
            ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
            if save != '':
                plt.savefig(save+'_BGLogisticRegression_{}.png'.format(ord))
                
        if len(order) == 1:
            all_sols = sols
        else:
            all_sols[ord] = sols
    plt.show()

    return all_sols

def data_parse_sliding_WSLS(actions, rewards,order=1, vif = False):
    
    # variables of interest: 
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {1, 0, -1} if left and not rewarded on prior trial, rewarded, or right not rewarded (basically, swap on win)
    numvars = 3
    state = np.zeros(len(actions))
    for i in range(len(actions)):
        state[i] = actions[i] if rewards[i] == 1 else not actions[i]

    data = np.zeros((len(actions) - order,numvars,order))

    for i in range(order+1,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                #action
                if j==0: #agent choice
                    if actions[i-k] == 0:
                        data[i_offset,j,k_offset] = -1
                    else:
                        data[i_offset,j,k_offset] = 1
                if j==1: #rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==2: #not rewarded and swap
                    if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                
                # if j==0: #agent choice
                #     if actions[i-k] == 0:
                #         data[i_offset,j,k_offset] = -1
                #     else:
                #         data[i_offset,j,k_offset] = 1
                # if j==1: #opponent choice
                #     if state[i-k] == 0:
                #         data[i_offset,j,k_offset] = -1
                #     else:
                #         data[i_offset,j,k_offset] = 1
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    # data = np.flip(data,axis=-1)
    if vif == True:
        for order in range(order):
            vif, rsq = compute_vif(data[:,:,order], bias=True)
            print('order: {}, Rsqs : {}'.format(order, rsq))
            print(vif) 
    data = data.reshape(data.shape[0],-1, order='C')
    return data
def data_parse_sliding_strategic(actions, rewards,order=1, vif = False):
    
    # variables of interest: 
    # {-1, 0, 1} if left and rewarded on prior trial, not rewarded, or right rewarded (basically, stay on win)
    # {1, 0, -1} if left and not rewarded on prior trial, rewarded, or right not rewarded (basically, swap on win)
    numvars = 4
    state = np.zeros(len(actions))
    for i in range(len(actions)):
        state[i] = actions[i] if rewards[i] == 1 else not actions[i]

    data = np.zeros((len(actions) - order,numvars,order))

    for i in range(order+1,len(actions)):
        for j in range(numvars):
            for k in range(1,order+1):
                i_offset = i - order
                k_offset = k - 1
                if j==3: #not rewarded and stay
                    if rewards[i-k] == 0 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j == 2: # rewarded and switch # win 
                    if rewards[i-k] == 1 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==0: #rewarded and stay
                    if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1
                if j==1: #not rewarded and swap
                    if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
                        if actions[i-k-1] == 0:
                            data[i_offset,j,k_offset] = -1
                        else:
                            data[i_offset,j,k_offset] = 1


    # for i in range(order,len(actions)):
    #     for j in range(numvars):
    #         for k in range(1,order+1):
    #             i_offset = i - order
    #             k_offset = k - 1
    #             #action
    #             if j==0: #agent choice
    #                 if actions[i-k] == 0:
    #                     data[i_offset,j,k_offset] = -1
    #                 else:
    #                     data[i_offset,j,k_offset] = 1
    #             if j==1: #rewarded and stay
    #                 if rewards[i-k] == 1 and (actions[i-k] == actions[i-k-1]):
    #                     if actions[i-k-1] == 0:
    #                         data[i_offset,j,k_offset] = -1
    #                     else:
    #                         data[i_offset,j,k_offset] = 1
    #             if j==2: #not rewarded and swap
    #                 if rewards[i-k] == 0 and (actions[i-k] != actions[i-k-1]):
    #                     if actions[i-k-1] == 0:
    #                         data[i_offset,j,k_offset] = -1
    #                     else:
    #                         data[i_offset,j,k_offset] = 1
                
                # if j==0: #agent choice
                #     if actions[i-k] == 0:
                #         data[i_offset,j,k_offset] = -1
                #     else:
                #         data[i_offset,j,k_offset] = 1
                # if j==1: #opponent choice
                #     if state[i-k] == 0:
                #         data[i_offset,j,k_offset] = -1
                #     else:
                #         data[i_offset,j,k_offset] = 1
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    # or the regressors will get mixed up
    # data = np.flip(data,axis=-1)
    if vif == True:
        for order in range(order):
            vif, rsq = compute_vif(data[:,:,order], bias=True)
            print('order: {}, Rsqs : {}'.format(order, rsq))
            print(vif) 
    data = data.reshape(data.shape[0],-1, order='C')
    return data


def general_logistic_regressors(actions, rewards,regression_order=1, a_order =1, r_order =0,r_offset=0):
    '''
    Composes regressors for a general logistic regression
    The regressors are the kron product of a_order-1 onehots times action and r_order onehots times reward
    for each trial
    
    Stay -> (1,0)
    Switch -> (0,1)
    Win -> (1,0)
    Lose -> (0,1)
    
    Args:
        actions: list of actions, 0 or 1 -> mapped to -1 or 1 in function
        rewards: list of rewards, 0 or 1 -> mapped to a onehot in function
        regression_order: order of the regression
        a_order: order of the actions -> regressors are the kron product of a_order-1 onehots times action
        r_order: order of the rewards -> regressors are the kron product of r_order onehots times reward
    
    Returns:
        regressors: list of regressors, each is a onehot of length max_order
        y: the target variable for the regression
        labels: list of labels for the regressors
    '''
    assert r_offset in [0, 1], 'r_offset must be 0 or 1'
    
    original_actions = np.array(actions)
    actions = np.array(actions)
    actions = actions * 2 - 1 # convert to -1 or 1
    rewards = np.array([[1,0] if i == 1 else [0,1] for i in rewards]) # convert to list of onehots where 1 -> (1,0) and 0 -> (0,1)
    
    if r_offset == 0:
        a_labels = ['repeat', 'change']
        r_labels = ['win', 'lose']
    elif r_offset == 1:
        a_labels = ['stay', 'switch']
        r_labels = ['win', 'lose']
    
    labels = []

    # first generate list of all regressors, then stack them so that we perform a regression of order regression_order
    regressors = [] 
    
    # if x_order = 1, then at time t we need values t-1, .., t-regression_order
    # if x_order = X, then at time t we need values t-1 -> t-X, .., t-regression_order -> t-X
    # so max_order = max(X) + regression_order 
    max_order = max(a_order-1, r_order)   # - 1 because we zero index

    assert a_order > 0, 'a_order must be greater than 0'
    assert r_order >= 0, 'r_order must be greater than or equal to 0'

    import itertools
    # Define factors with their time lags (j=0 is t-1, j=1 is t-2, etc.)
    action_factors_def = [(j, 'a') for j in range(a_order - 1)]
    reward_factors_def = [(j, 'r') for j in range(r_order)]
    
    # Sort factors by time lag, descending (e.g., t-2 before t-1).
    all_factors = action_factors_def + reward_factors_def
    sorted_factors = sorted(all_factors, key=lambda x: x[0], reverse=True)

    for i in range(max_order,len(actions)-1):
        # build up regressors using np.kron
        if a_order == 1:
            r = actions[i]
        else:
            r = actions[i - a_order + 1]
    
        # Collect factor values for the current timestep
        factors_to_kron = []
        for lag, factor_type in sorted_factors:
            if factor_type == 'a':
                a_onehot = np.zeros(2)
                # stay/switch for lag t-(lag+1)
                a_tf = int(actions[i-lag] != actions[i-lag-1])
                a_onehot[a_tf] = 1
                factors_to_kron.append(a_onehot)
            else: # factor_type == 'r'
                # win/lose for lag t-(lag+1)
                factors_to_kron.append(rewards[i-lag-r_offset])
        
        # Apply kron product in the new sorted, interleaved order
        for factor in factors_to_kron:
            r = np.kron(r, factor)
            
        regressors.append(r)

    # Generate labels using the same sorted logic to ensure they match the regressors.
    # This replaces the entire label_recurse block.
    label_parts = []
    
    if r_offset == 1: #try this, might be wrong:
        for lag, factor_type in sorted_factors:
            if factor_type == 'a':
                label_parts.append(r_labels)
            else:
                label_parts.append(a_labels)
            
        
    else:
        for lag, factor_type in sorted_factors:
            if factor_type == 'a':
                label_parts.append(a_labels)
            else:
                label_parts.append(r_labels)
            
    if label_parts:
        labels = [" ".join(p) for p in itertools.product(*label_parts)]
    elif a_order >= 1:
        labels = ["action"]
    else:
        labels = []
    
    regressors = np.array(regressors)
    
    # Apply sliding window operation - handle all cases uniformly
    if (2**(a_order-1)*2**r_order) == 1:
        regressors = regressors[:,np.newaxis]
    W = np.lib.stride_tricks.sliding_window_view(regressors.T, window_shape=regression_order, axis=1)  # (F, S-N+1, N)
    W = W.transpose(1, 0, 2)  # (S-N+1, F, N) → features-major, lags oldest→newest
    W = W[:, :, ::-1] # Flip lags to be newest -> oldest, to match strategic parser
    regressors = W.reshape(W.shape[0], -1, order='C')  # (S-N+1, F*N)

    start_index = max_order + regression_order
    y = original_actions[start_index:start_index + regressors.shape[0]]


    return regressors, y, labels


    # x = [[0,1],[0,2],[3,0],[4,0],[0,5],[6,0],[7,0],[8,0],[0,9],[10,0],[11,0],[0,12]]
    
    # y = [[0,5,4,0,3,0,0,2,0,1], [6,0,0,5,4,0,3,0,0,2,],[7,0,6,0,0,5,4,0,3,0],[8,0,7,0,6,0,0,5,4,0]]


def fit_glr(data, order=5, a_order=2, r_order=1, err = False, model = False, labels = True, average = False):
    ''' 
    Fit a general logistic regression and return the coefficients.
    If err is True, then return the error bars computed by leave one out cross validation for sessions
    If average is True, then return the average of the fits, otherwise return the fits for each session
    ''' 
 
    
    
    regressors = []
    ys = []
    
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
        for i in range(len(episode_actions)):
            regressors_i, y_i, labels_i = general_logistic_regressors(episode_actions[i], episode_rewards[i], regression_order=order, a_order=a_order, r_order=r_order)
            regressors.append(regressors_i)
            ys.append(y_i)
    
    else:
        episode_actions = data['monkey_choice'].values
        episode_rewards = data['reward'].values
        # have to iterate over sessions
        sessions = data['id'].unique()
        for s in sessions:
            sessdat = data[data['id'] == s].sort_values(by=['id','trial'])
            regressors_i, y_i, labels_i = general_logistic_regressors(sessdat['monkey_choice'].values, sessdat['reward'].values, regression_order=order, a_order=a_order, r_order=r_order)
            regressors.append(regressors_i)
            ys.append(y_i)
    



    fits = [logistic_regression_colinear(regressors[j],ys[j].squeeze(),bias=True)['x'] for j in range(len(regressors))]
    lens = [len(y_episode) for y_episode in ys]
    if average:
        sol = np.average(fits, axis=0, weights=lens).squeeze()
    else:
        sol = fits
    # compute error bars using leave one out cross validation on fits
    if err:
        loo_sols = []
        for i in range(len(fits)):
            # Leave out the i-th session
            fits_loo = [fits[j] for j in range(len(fits)) if j != i]
            lens_loo = [lens[j] for j in range(len(lens)) if j != i]
            loo_sols.append(np.average(fits_loo, axis=0, weights=lens_loo).squeeze())
        loo_sols = np.stack(loo_sols, axis=0)
        err_bars = np.std(loo_sols, axis=0, ddof=1) * np.sqrt(len(loo_sols) / (len(loo_sols) - 1))
    if err:
        if labels:
            return sol, err_bars, labels_i
        else:
            return sol, err_bars
    else:
        if labels:
            return sol, labels_i
        else:
            return sol
    


def data_parse_combinatorial(actions, rewards,order=1):
    
    # variables of interest:
    # {-1,0,1} if {condition met and left, condition not met, condition met and right}
    # for all zeroth and first order interactions between agent  choice and computer choice
    # regressors take the form:
    # rewarded / not rewarded and action/state[t-i] (4N here)
    # rewarded / not rewarded and {action/state[t]} == {action/state[t-i]} (8N here) or is it 4N*(N+1)?
    # where N = order
    
    #ignore first 100 trials so the model reaches a steady state ?
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # numvars = 12*order
    numvars = 8*order +  8*order*(order+1)
    state = np.zeros(len(actions))
    
    for i in range(len(actions)):
        state[i] = actions[i] if rewards[i] == 1 else not actions[i]

    data = np.zeros((len(actions) - order,numvars))
    #map 1 -> 0, 0 -> 1
    not_rewards = 1 - rewards
    
    cond1 = [rewards, not_rewards]
    cond2 = [actions, state]
    
                                    
    for i in range(order,len(actions)):
        i_offset = i - order
        j = 0
        for k in range(1,order+1):
            k_offset = k - 1
            for case in range(4):
                if case == 0:
                    for l in range(2):
                        for m in range(2):
                            if cond1[l][i-k] == 1 and cond2[m][i-k] == 0:
                                if actions[i-k] == 0:
                                    data[i_offset,j] = -1
                                else:
                                    data[i_offset,j] = 1
                            j += 1
                if case == 1:
                    for l in range(2):
                        for m in range(2):
                            if cond1[l][i-k] == 1 and cond2[m][i-k] != 0:
                                if actions[i-k] == 0:
                                    data[i_offset,j] = -1
                                else:
                                    data[i_offset,j] = 1
                            j += 1
                if case == 2:
                    for l in range(2):
                        for m in range(2):
                            for n in range(2):    
                                for z in range(k,order+1):
                                    if cond1[l][i-k] == 1 and cond2[m][i-k] == cond2[n][i-z]:
                                        if actions[i-k-1] == 0:
                                            data[i_offset,j] = -1
                                        else:
                                            data[i_offset,j] = 1
                                    j+=1
                if case == 3:
                    for l in range(2):
                        for m in range(2):
                            for n in range(2):
                                for z in range(k, order+1):
                                    if cond1[l][i-k] == 1 and cond2[m][i-k] == 1-cond2[n][i-z]:
                                        if actions[i-k-1] == 0:
                                            data[i_offset,j] = -1
                                        else:
                                            data[i_offset,j] = 1
                                    j+=1
        # print(j)

    
    # for i in range(order,len(actions)):
    #     for k in range(1,order+1):
    #         j = 0
    #         i_offset = i - order
    #         k_offset = k - 1
    #         if j < 4:
    #             for l in range(2):
    #                 for m in range(2):
    #                     i_offset = i - order
    #                     k_offset = k - 1
    #                     if cond1[l][i-k] == 1 and cond2[m][i-k] == 0:
    #                         if actions[i-k] == 0:
    #                             data[i_offset,j,k_offset] = -1
    #                         else:
    #                             data[i_offset,j,k_offset] = 1
    #                     j += 1
    #         elif j < 8:
    #             for l in range(2):
    #                 for m in range(2):
    #                     if cond1[l][i-k] == 1 and cond2[m][i-k] != 0:
    #                         if actions[i-k] == 0:
    #                             data[i_offset,j,k_offset] = -1
    #                         else:
    #                             data[i_offset,j,k_offset] = 1
    #                     j += 1
                        
    #         elif j < 8 + 4*order*(order+1):
    #             for l in range(2):
    #                 for m in range(2):
    #                     for n in range(2):    
    #                         for z in range(k,order+1):
    #                             if cond1[l][i-k] == 1 and cond2[m][i-k] == cond2[n][i-z]:
    #                                 if actions[i-k-1] == 0:
    #                                     data[i_offset,j,k_offset] = -1
    #                                 else:
    #                                     data[i_offset,j,k_offset] = 1
    #                             j+=1
                        
    #         elif j < numvars:
    #             for l in range(2):
    #                 for m in range(2):
    #                     for n in range(2):
    #                         for z in range(k, order+1):
    #                             if cond1[l][i-k] == 1 and cond2[m][i-k] == 1-cond2[n][i-z]:
    #                                 if actions[i-k-1] == 0:
    #                                     data[i_offset,j,k_offset] = -1
    #                                 else:
    #                                     data[i_offset,j,k_offset] = 1
    #                             j+=1
                                
                        
                        
    # flatten last two dimensions to combine them. I think this needs to be done carefully
    data = data.reshape(data.shape[0],-1, order='C')
    return data



# def model_logistic_regression_sliding(actions, rewards, window_size= 10, order=2, save = ''):
    '''creates a sliding window and regresses on that'''
#     assert(order < len(actions), 'order must be less than the length of the data')
#     assert(window_size < len(actions), 'window size must be less than the length of the data')
#     assert(len(actions) == len(rewards), 'actions and rewards must be the same length')

    
    
#     data = data_parse_sliding_WSLS(actions, rewards, order=order)

#     regressors = []
#     actions = np.array(actions)
    
#     for window in range(len(actions)-window_size-order):
#         reg = logistic_regression(data[window:window+window_size], actions[window+order:order+window+window_size], bias=False)
#         regressors.append(reg.x)

#     return np.vstack(regressors)

# def model_logistic_regression_sliding(actions, rewards,dominance, window_size= 10, order=2, save = ''):
    ''' This version fits 3 different regressions, one to the PFC data, one to BG, and one to both'''
def logistic_regression_masked(model, data, order=5, save = '', bias = False, err = False, sliding = None, mask = None, fitted_RL = False,):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    sols = {}
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
        ord = min(order, len(episode_actions[0]) - 1)    
        if fitted_RL: 
            all_actions = np.vstack(episode_actions).T[:,ord:]
        else:
            all_actions = np.hstack(episode_actions).T[:,ord:]
        if mask is not None:
            mask = np.hstack(mask).T[:,ord:]
        else:
            mask = [None] * all_actions.shape[0]
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))
    else:
        ord = order
        regressors = parse_monkey_behavior_reduced(data,ord,False,err)
        all_actions = create_order_data(data,ord, err)
        if mask is not None:
            mask = [mask[i][ord:] for i in range(len(mask))]
        else:
            if err:
                mask = [None] * len(all_actions)
     
    if not err:
        all_actions = all_actions.ravel()
        regressors = np.vstack(regressors)
        action_regression = logistic_regression(regressors, all_actions,bias=bias,mask=mask)
        sols['action'] = action_regression.x
    else:
        if sliding is not None:
            fits = [] 
            subregs = []
            subacts = []

            for index in range(len(regressors)):
                regs = regressors[index]
                acts = all_actions[index]

                for i in range(len(acts) - sliding):
                    fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=bias).x)
                    subregs.append(regs[i:i+sliding])
                    subacts.append(acts[i:i+sliding])
            fits = np.array(fits)
            sols['action'] = np.mean(fits,axis=0).squeeze()
            sols['err'] = np.std(fits,axis=0).squeeze()
            regressors = np.array(subregs)
            all_actions = np.array(subacts) # do i need to weight this somehow
                
        else:
            fits = []
            lens = []

            for index in range(len(regressors)):
                fits.append(logistic_regression(regressors[index],all_actions[index],bias=bias, mask=mask[index]).x)
                lens.append(len(all_actions[index]))
            fits = np.array(fits)
            sols['action'] = np.average(fits,axis=0,weights=lens).squeeze()
            sols['err'] = np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens)).squeeze()
        
        if bias:
            action_fit = np.hstack([regressors[index] @ fits[index][:-1] + fits[index][-1] for index in range(len(fits))])
        else:
            action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
        all_actions = np.hstack(all_actions)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
    
    if bias:  
        ax1.set_xlabel(r'$\theta^T x + b$')
        if not err:
            action_fit = regressors @ sols['action'][:-1] + sols['action'][-1]
    else:
        ax1.set_xlabel(r'$\theta^T x$')
        if not err:
            action_fit = regressors @ sols['action']
    acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    ax1.set_title(r'Network Actions regression: accuracy = {:.3f}'.format(acc))
    smigmoid = np.linspace(min(action_fit),max(action_fit),1000)
    ax1.plot(smigmoid, sigmoid(smigmoid), label='sigmoid', color='r')
    ax1.set_ylabel('action')
    bins = np.linspace(min(action_fit),max(action_fit),20)
    hist1 = np.histogram(action_fit[all_actions == 0], bins=bins,density=True)
    bins1 = (hist1[1][1:] + hist1[1][:-1]) / 2
    hist1 = hist1[0]
    hist2 = np.histogram(action_fit[all_actions == 1], bins=bins,density=True)
    bins2 = (hist2[1][1:] + hist2[1][:-1]) / 2
    hist2 = hist2[0]
    kde1 = stats.gaussian_kde(action_fit[all_actions == 0],bw_method=.1)(smigmoid) * .1
    kde2 = stats.gaussian_kde(action_fit[all_actions == 1],bw_method=.1)(smigmoid) *.1  
    ax1.set_xlim(min(bins),max(bins))
    ax1.plot(smigmoid,kde1, color = 'b')
    ax3 = ax1.twinx()
    ax3.invert_yaxis()
    ax3.plot(smigmoid,kde2, color = 'b')        
    ax1.axvline(0,0)        
    ax1.set_ylim(0,1)
    ax3.set_ylim(1,0)
    ax3.set_yticks([0,1])
    ax3.set_yticklabels(['Right','Left'])
    
    xord = np.arange(1,1+ord)

    ax2.set_xlabel(r'trial number')
    acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    # sols['action'] = [sols['action'][i] if sols['action'][i] != 0 else np.nan for i in range(len(sols['action']))]

    reggy = ['agent choice', 'win stay', 'lose switch']
    if not err:
        for i in range(len(sols['action'])//ord):
            ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
    else:
        for i in range(len(sols['action'])//ord):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            ax2.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
            ax2.fill_between(xord,sols['action'][i*ord:(i+1)*ord] - sols['err'][i*ord:(i+1)*ord], sols['action'][i*ord:(i+1)*ord]+sols['err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
    ax2.set_title(r'action regression coefficients'.format(acc))
    ax2.set_ylabel('coefficient')
    ax2.legend()
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    # ax2.legend(['agent choice', 'win stay', 'lose switch'])
    ax2.axhline(linestyle = '--', color = 'k', alpha = .5)
    if save != '':
        plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
    
    if save == '':
        plt.show()
        
        
        
# like logistic_regression_masked, but returns a handle for the figure
def paper_logistic_regression(ax,model, data, order=5, save = '', bias = False, err = True, sliding = None, mask = None, fitted_RL = False,legend = False, return_model=False, colinear=True):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    sols = {}
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
        ord = min(order, len(episode_actions[0]) - 1)    
        if fitted_RL: 
            all_actions = np.vstack(episode_actions).T[:,ord:]
        else:
            all_actions = np.hstack(episode_actions).T[:,ord:]
        if mask is not None:
            mask = np.hstack(mask).T[:,ord:]
        else:
            mask = [None] * all_actions.shape[0]
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))
    else:
        ord = order
        regressors = parse_monkey_behavior_reduced(data,ord,False,err)
        all_actions = create_order_data(data,ord, err)
        if mask is not None:
            mask = [mask[i][ord:] for i in range(len(mask))]
        else:
            if err:
                mask = [None] * len(all_actions)
     
    if not err:
        all_actions = all_actions.ravel()
        regressors = np.vstack(regressors)
        action_regression = logistic_regression(regressors, all_actions,bias=bias,mask=None, colinear=colinear)
        sols['action'] = action_regression['x']
        return action_regression.x
    else:
        if sliding is not None:
            fits = [] 
            subregs = []
            subacts = []

            for index in range(len(regressors)):
                regs = regressors[index]
                acts = all_actions[index]

                for i in range(len(acts) - sliding):
                    fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=bias, colinear=colinear).x)
                    subregs.append(regs[i:i+sliding])
                    subacts.append(acts[i:i+sliding])
            fits = np.array(fits)
            sols['action'] = np.mean(fits,axis=0).squeeze()
            sols['err'] =1.96* (np.std(fits,axis=0)/len(fits)).squeeze()
            regressors = np.array(subregs)
            all_actions = np.array(subacts) # do i need to weight this somehow
                
        else:
            fits = []
            lens = []

            for index in range(len(regressors)):
                fits.append(logistic_regression(regressors[index],all_actions[index],bias=bias, mask=mask[index])['x'])
                lens.append(len(all_actions[index]))
            fits = np.array(fits)
            sols['action'] = np.average(fits,axis=0,weights=lens).squeeze()
            sols['err'] = 1.96*np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens)/sum(lens)).squeeze()
        
        if bias:
            action_fit = np.hstack([regressors[index] @ fits[index][:-1] + fits[index][-1] for index in range(len(fits))])
        else:
            action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
        all_actions = np.hstack(all_actions)
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
    
    
    xord = np.arange(1,1+ord)

    # ax.set_xlabel(r'trial number')
    
    # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    # sols['action'] = [sols['action'][i] if sols['action'][i] != 0 else np.nan for i in range(len(sols['action']))]
    if ax is None:
        return sols
    
    reggy = ['agent choice', 'win stay', 'lose switch']
    
    ax.set_xticks(range(1,ord+1))

    if not err:
        for i in range(len(sols['action'])//ord):
            ax.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
    else:
        for i in range(len(sols['action'])//ord):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            ax.plot(xord,sols['action'][i*ord:(i+1)*ord], label = reggy[i])
            ax.fill_between(xord,sols['action'][i*ord:(i+1)*ord] - sols['err'][i*ord:(i+1)*ord], sols['action'][i*ord:(i+1)*ord]+sols['err'][i*ord:(i+1)*ord], alpha = .25, facecolor = colors[i])
    # ax.set_title(r'action regression coefficients'.format(acc))
    # ax.set_ylabel('coefficient')
    if legend:
        ax.legend()
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    # ax2.legend(['agent choice', 'win stay', 'lose switch'])
    ax.axhline(linestyle = '--', color = 'k', alpha = .5)
    if save != '':
        plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(ord))
    
    if return_model:
        return sols
    return ax

def paper_logistic_regression_strategic(ax,model, data, order=5, save = '', bias = False, err = True, sliding = None, mask = None, fitted_RL = False,legend = False, return_model=False, colinear=True):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    sols = {}
    if model:
        episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
        # ord = min(order, len(episode_actions[0]) - 1)    
        if fitted_RL: 
            all_actions = np.vstack(episode_actions).T[:,order:]
        else:
            all_actions = np.hstack(episode_actions).T[:,order:]
        if mask is not None:
            mask = np.hstack(mask).T[:,order:]
        else:
            mask = [None] * all_actions.shape[0]
        regressors = []
        for i in range(len(episode_actions)):
            regressors.append(data_parse_sliding_strategic(episode_actions[i], episode_rewards[i], order))
    else:
        regressors = parse_monkey_behavior_strategic(data,order,False,err)
        all_actions = create_order_data(data,order, err)
        if mask is not None:
            mask = [mask[i][order:] for i in range(len(mask))]
        else:
            if err:
                mask = [None] * len(all_actions)
     
    if not err:
        all_actions = all_actions.ravel()
        regressors = np.vstack(regressors)
        mask = None
        action_regression = logistic_regression(regressors, all_actions,bias=bias,mask=mask, colinear=colinear)
        # action_regression = logistic_regression_colinear(regressors, all_actions,bias=bias,mask=mask)
        sols['action'] = action_regression['x']
        return action_regression['x']
    else:
        if sliding is not None:
            fits = [] 
            subregs = []
            subacts = []

            for index in range(len(regressors)):
                regs = regressors[index]
                acts = all_actions[index]

                for i in range(len(acts) - sliding):
                    fits.append(logistic_regression(regs[i:i+sliding],acts[i:i+sliding], bias=bias, colinear=colinear).x)
                    subregs.append(regs[i:i+sliding])
                    subacts.append(acts[i:i+sliding])
            fits = np.array(fits)
            sols['action'] = np.mean(fits,axis=0).squeeze()
            sols['err'] =1.96* (np.std(fits,axis=0)/len(fits)).squeeze()
            regressors = np.array(subregs)
            all_actions = np.array(subacts) # do i need to weight this somehow
                
        else:
            fits = []
            lens = []

            for index in range(len(regressors)):
                # fits.append(logistic_regression(regressors[index],all_actions[index],bias=bias, mask=mask[index]).x)
                fits.append(logistic_regression_colinear(regressors[index],all_actions[index],bias=bias, mask=mask[index])['x'])
                lens.append(len(all_actions[index]))
            fits = np.array(fits)
            sols['action'] =np.average(fits,axis=0,weights=lens).squeeze()
            sols['err'] = 1.96*np.sqrt(np.average((sols['action'] - fits)**2,axis=0,weights=lens)/sum(lens)).squeeze()
        
        if bias:
            action_fit = np.hstack([regressors[index] @ fits[index][:-1] + fits[index][-1] for index in range(len(fits))])
        else:
            action_fit = np.hstack([regressors[index] @ fits[index] for index in range(len(fits))])
        all_actions = np.hstack(all_actions)
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4)) 
    
    
    xord = np.arange(1,1+order)

    # ax.set_xlabel(r'trial number')
    
    # acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    # sols['action'] = [sols['action'][i] if sols['action'][i] != 0 else np.nan for i in range(len(sols['action']))]
    if ax is None:
        return sols
    
    # reggy = ['agent choice', 'win stay', 'lose switch']
    reggy = ['win stay', 'lose switch', 'win switch', 'lose stay']    
    ax.set_xticks(range(1,order+1))

    if not err:
        for i in range(len(sols['action'])//order):
            ax.plot(xord,sols['action'][i*order:(i+1)*order], label = reggy[i])
    else:
        for i in range(len(sols['action'])//order):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            ax.plot(xord,sols['action'][i*order:(i+1)*order], label = reggy[i])
            ax.fill_between(xord,sols['action'][i*order:(i+1)*order] - sols['err'][i*order:(i+1)*order], sols['action'][i*order:(i+1)*order]+sols['err'][i*order:(i+1)*order], alpha = .25, facecolor = colors[i])
    # ax.set_title(r'action regression coefficients'.format(acc))
    # ax.set_ylabel('coefficient')
    if legend:
        ax.legend(frameon=False)
    # ax2.legend(['agent choice', 'computer choice', 'win stay', 'lose switch'])
    # ax2.legend(['agent choice', 'win stay', 'lose switch'])
    ax.axhline(linestyle = '--', color = 'k', alpha = .5)
    if save != '':
        plt.savefig(save+'_ActionLogisticRegression_{}.png'.format(order))
    
    if return_model:
        return sols
    return ax
def paper_logistic_accuracy(actions, rewards,order = 5, bias = False, mask = None, greedy=True):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''

    ord = min(order, len(actions) - 1)    
    regressors = data_parse_sliding_WSLS(actions, rewards, ord)
    actions = actions[ord:]
    action_regression = logistic_regression(regressors, actions,bias=bias,mask=mask)
    fit = action_regression['x']
    if greedy:
        if bias:
            action_fit = regressors @ fit[:-1] + fit[-1]
        else:
            action_fit = regressors @ fit
        acc = np.mean(np.round(sigmoid(action_fit)) == actions)
        return fit, acc    

    else:
        likelihood = compute_likelihood(actions, regressors, fit, bias)
        return fit, likelihood

def compute_likelihood(actions, regressors, fit, bias = False):
    '''computes the likelihood of the model given the actions in a single session and the fit'''
    # find the propbabilities output by the model at each trial
    order = min(regressors.shape) #min dimension of regressors
    probabilities = sigmoid(regressors @ fit) if not bias else sigmoid(regressors @ fit[:-1] + fit[-1])
    # find the likelihood of the model given the actions
    likelihoods = []
    for i in range(len(actions)):
        if actions[i] == 1:
            likelihoods.append(probabilities[i])
        else:
            likelihoods.append(1 - probabilities[i])
    # likelihood = np.prod(likelihoods)**(1/len(actions))
    likelihood = np.exp(np.sum(np.log(likelihoods)/len(actions)))
    # likelihood = np.prod(probabilities ** actions * (1 - probabilities) ** (1 - actions))
    return likelihood

def paper_logistic_accuracy_strategic(actions, rewards,order = 5, bias = False, mask = None, greedy=True):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''

    ord = min(order, len(actions) - 1)    
    # regressors = data_parse_sliding_WSLS(actions, rewards, ord)
    regressors = data_parse_sliding_strategic(actions, rewards, ord)
    actions = actions[ord:]
    action_regression = logistic_regression(regressors, actions,bias=bias,mask=mask)
    fit = action_regression['x']
    if bias:
        action_fit = regressors @ fit[:-1] + fit[-1]
    else:
        action_fit = regressors @ fit
    if greedy:
        acc = np.mean(np.round(sigmoid(action_fit)) == actions)
        return fit, acc
    else:
        likelihood = compute_likelihood(actions, regressors, fit, bias)
        return fit, likelihood
    
    # return fit, acc


def paper_logistic_accuracy_combinatorial(actions, rewards,order = 5, bias = False, mask = None):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''

    ord = min(order, len(actions) - 1)    
    regressors = data_parse_combinatorial(actions, rewards, ord)
    actions = actions[ord:]
    action_regression = logistic_regression(regressors, actions,bias=bias,mask=mask)
    fit = action_regression.x
    if bias:
        action_fit = regressors @ fit[:-1] + fit[-1]
    else:
        action_fit = regressors @ fit
    acc = np.mean(np.round(sigmoid(action_fit)) == actions)
    return acc


def histogram_logistic_accuracy(data,order = 5, bias = False, mask = None):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    # regressors = data_parse_sliding_WSLS(actions, rewards, ord)
    # actions = actions[ord:]
    # action_regression = logistic_regression(regressors, actions,bias=bias,mask=mask)
    # fit = action_regression.x
    
    regressors = np.vstack(parse_monkey_behavior_reduced(data,order,False,False))
    all_actions = create_order_data(data,order, False).ravel()
    fit = logistic_regression(regressors, all_actions,bias=bias,mask=mask).x
        
    if bias:
        action_fit = regressors @ fit[:-1] + fit[-1]
    else:
        action_fit = regressors @ fit
    acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    return acc

def histogram_logistic_accuracy_strategic(data,order = 5, bias = False, mask = None):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    # regressors = data_parse_sliding_WSLS(actions, rewards, ord)
    # actions = actions[ord:]
    # action_regression = logistic_regression(regressors, actions,bias=bias,mask=mask)
    # fit = action_regression.x
    
    regressors = np.vstack(parse_monkey_behavior_strategic(data,order,False,False))
    all_actions = create_order_data(data,order, False).ravel()
    fit =  logistic_regression(regressors, all_actions,bias=bias,mask=mask)
    try:
        fit = fit.x
    except:
        fit = fit['x']
    if bias:
        action_fit = regressors @ fit[:-1] + fit[-1]
    else:
        action_fit = regressors @ fit
    acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    return acc

def histogram_logistic_accuracy_combinatorial(data,order = 5, bias = False, mask = None):
    '''Given input data and a boolean whether the data came from monkeys or a model, as well as a mask,
    Fits a logistic regression'''
    regressors = np.vstack(parse_monkey_behavior_combinatorial(data,order,False))
    all_actions = create_order_data(data,order, False).ravel()
    fit = logistic_regression(regressors, all_actions,bias=bias,mask=mask).x
    if bias:
        action_fit = regressors @ fit[:-1] + fit[-1]
    else:
        action_fit = regressors @ fit
    acc = np.mean(np.round(sigmoid(action_fit)) == all_actions)
    return acc

def fit_single_paper(data, order=5, save='', bias=False, colinear=True):
    '''
        Fit a single logistic regression for the paper
        Takes a tuple of data where data = (episode_action, episode_reward)
    '''
    sols = {}
    ord = order
    regressors = parse_monkey_behavior_reduced(data,ord,False,False)
    all_actions = create_order_data(data,ord, False)
    all_actions = all_actions.ravel()
    regressors = np.vstack(regressors)
    action_regression = logistic_regression(regressors, all_actions,bias=bias, mask=None,colinear=colinear)
    return action_regression.x
 
 
def fit_single_paper_strategic(data, order=5, save='', bias=False, colinear=True):
    '''
        Fit a single logistic regression for the paper
        Takes a tuple of data where data = (episode_action, episode_reward)
    '''
    sols = {}
    ord = order
    regressors = parse_monkey_behavior_strategic(data,ord,False,False)
    all_actions = create_order_data(data,ord, False)
    all_actions = all_actions.ravel()
    regressors = np.vstack(regressors)
    action_regression = logistic_regression(regressors, all_actions,bias=bias, mask=None,colinear=colinear)
    return action_regression['x']
 
 
def fit_bundle_paper(data, order=5, save='', bias=False):
    '''
        Fit a single logistic regression for the paper
        Takes a tuple of data where data = (episode_action, episode_reward)
    '''
    sols = {}

    ep_a, _, ep_r, _, _, _ = data
    
    episode_states, episode_actions, episode_rewards, episode_hiddens, _, _ = data
    ord = min(order, len(episode_actions[0]) - 1)    

    all_actions = np.hstack(episode_actions).T[:,ord:]
    mask = [None] * all_actions.shape[0]
    regressors = []
    regs = np.zeros((len(ep_a),order*3))

    for i in range(len(episode_actions)):
        regressors.append(data_parse_sliding_WSLS(episode_actions[i], episode_rewards[i], ord))
    
    
    for row in range(len(ep_a)):
        regressors = data_parse_sliding_WSLS(episode_actions[row],episode_rewards[row],order)
        # all_actions = create_order_data(data,order, False)
        actions = all_actions[row]
        # all_actions = all_actions.ravel()
        regressors = np.vstack(regressors)
        action_regression = logistic_regression(regressors, actions,bias=bias, mask=None)
        regs[row] = action_regression.x
    return np.mean(regs,axis=0)


# def WSLS_comparison_paper(data, order=5):
#     sols = {}
#     ord = order
#     regressors = parse_monkey_behavior_reduced(data,ord,False,False)
#     regressors_reward = parse_monkey_behavior_minimal(data,ord,False)
#     all_actions = create_order_data(data,ord, False)
#     all_actions = all_actions.ravel()
#     regressors = np.vstack(regressors)
#     action_regression = logistic_regression(regressors, all_actions,bias=bias, mask=None)
#     reward_regression = logistic_regression_reduced(regressors_reward, all_actions, action_regression.x[:ord])
#     return action_regression.x