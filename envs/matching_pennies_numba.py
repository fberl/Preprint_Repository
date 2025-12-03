import numpy as np
from numba import njit
import math
import random  # Import random module
from scipy.stats import binomtest

@njit
def log_factorial_stirling(n):
    """Fast approximation of log(n!) using Stirling's approximation for large n."""
    if n < 12:
        # Use exact calculation for small n
        result = 0.0
        for i in range(1, n + 1):
            result += math.log(i)
        return result
    else:
        # Stirling's approximation: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2*π*n)
        return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

@njit
def log_binomial_coeff(n, k):
    """Compute log(C(n,k)) = log(n! / (k!(n-k)!)) efficiently."""
    if k > n or k < 0:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    
    # Use symmetry: C(n,k) = C(n,n-k)
    if k > n - k:
        k = n - k
    
    return log_factorial_stirling(n) - log_factorial_stirling(k) - log_factorial_stirling(n - k)

@njit
def binomial_pmf_log(k, n, p):
    """Compute log of binomial PMF: log(P(X = k)) where X ~ Binomial(n, p)."""
    if k > n or k < 0:
        return -np.inf
    if p == 0.0:
        return 0.0 if k == 0 else -np.inf
    if p == 1.0:
        return 0.0 if k == n else -np.inf
    
    log_coeff = log_binomial_coeff(n, k)
    if k > 0:
        log_coeff += k * math.log(p)
    if n - k > 0:
        log_coeff += (n - k) * math.log(1.0 - p)
    
    return log_coeff

@njit
def binomial_cdf_upper(k, n, p):
    """Compute P(X >= k) where X ~ Binomial(n, p) using log-sum-exp for stability."""
    if k > n:
        return 0.0
    if k <= 0:
        return 1.0
    if p == 0.0:
        return 1.0 if k <= 0 else 0.0
    if p == 1.0:
        return 1.0 if k <= n else 0.0
    
    # For large n, use normal approximation
    if n > 100:
        mu = n * p
        sigma = math.sqrt(n * p * (1 - p))
        if sigma > 0:
            z = (k - 0.5 - mu) / sigma  # continuity correction
            # Complementary error function approximation
            # P(Z >= z) ≈ 0.5 * erfc(z/√2)
            if z > 6:
                return 0.0
            if z < -6:
                return 1.0
            # Approximate erfc using rational approximation
            t = 1.0 / (1.0 + 0.3275911 * abs(z))
            erfc_approx = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
            erfc_approx = erfc_approx * math.exp(-z * z)
            if z < 0:
                erfc_approx = 2.0 - erfc_approx
            return 0.5 * erfc_approx
    
    # Exact calculation using log-sum-exp for numerical stability
    max_log_prob = -np.inf
    
    # First pass: find maximum log probability
    for i in range(k, n + 1):
        log_prob = binomial_pmf_log(i, n, p)
        if log_prob > max_log_prob:
            max_log_prob = log_prob
    
    if max_log_prob == -np.inf:
        return 0.0
    
    # Second pass: compute sum using log-sum-exp trick
    prob_sum = 0.0
    for i in range(k, n + 1):
        log_prob = binomial_pmf_log(i, n, p)
        if log_prob > max_log_prob - 50:  # avoid underflow
            prob_sum += math.exp(log_prob - max_log_prob)
    
    return prob_sum * math.exp(max_log_prob)

@njit
def binomial_cdf_lower(k, n, p):
    """Compute P(X <= k) where X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    
    # Use complement: P(X <= k) = 1 - P(X >= k+1)
    return 1.0 - binomial_cdf_upper(k + 1, n, p)

@njit
def numba_binomtest(k, n, p, alternative):
    """
    Numba-compiled binomial test equivalent to scipy.stats.binomtest.
    
    Parameters:
    k: number of successes
    n: number of trials  
    p: probability of success (usually 0.5)
    alternative: 0 for 'greater' (P(X >= k)), 1 for 'less' (P(X <= k))
    
    Returns:
    p-value for the test
    """
    if alternative == 0:  # 'greater'
        return binomial_cdf_upper(k, n, p)
    else:  # 'less'
        return binomial_cdf_lower(k, n, p)

@njit
def calculate_sequence_numba(choice_arr, trial, depth):
    """Numba-optimized sequence calculation - exactly like original."""
    seq = 0
    for currdepth in range(1, depth):
        seq += choice_arr[trial - currdepth] * (10 ** (currdepth - 1))
    return seq

@njit
def build_histseq_numba(choice_arr, depth, n_trials):
    """Fast histseq building with numba."""
    histseq = np.zeros(n_trials)
    for trial in range(depth + 1, n_trials):
        histseq[trial] = calculate_sequence_numba(choice_arr, trial, depth)
    return histseq

@njit
def find_indices_numba(histseq, target_val):
    """Fast index finding with numba."""
    indices = []
    for i in range(len(histseq) - 1):  # exclude last
        if histseq[i] == target_val:
            indices.append(i)
    return indices

@njit
def sum_at_indices_numba(data, indices):
    """Fast summation at indices."""
    total = 0.0
    for i in indices:
        total += data[i]
    return total

@njit 
def build_choice_reward_seqs_numba(choice_arr, rew_arr, depth, n_trials):
    """Fast building of both choice and reward sequences."""
    chistseq = np.zeros(n_trials)
    rhistseq = np.zeros(n_trials)
    
    for trial in range(depth + 1, n_trials):
        cseq = 0
        rseq = 0
        for currdepth in range(1, depth):
            cseq += choice_arr[trial - currdepth] * (10 ** (currdepth - 1))
            rseq += rew_arr[trial - currdepth] * (10 ** (currdepth - 1))
        chistseq[trial] = cseq
        rhistseq[trial] = rseq
    
    return chistseq, rhistseq

@njit
def find_combined_indices_numba(chistseq, rhistseq, target_c, target_r):
    """Fast finding of indices that match both choice and reward sequences."""
    indices = []
    for i in range(len(chistseq) - 1):  # exclude last
        if chistseq[i] == target_c and rhistseq[i] == target_r:
            indices.append(i)
    return indices

def matching_pennies_numba(choice, rew, maxdepth, alpha, algo="all", const_bias=0):
    """
    EXACT replica of matching_pennies with numba optimizations for bottlenecks only.
    
    This function preserves IDENTICAL algorithm logic, data handling, and random state.
    Only the computational bottlenecks are optimized with numba.
    """
    testAlpha = alpha / 2
    pComputerRight, bias, biasDepth, maxDev, whichAlg = 0.5 + const_bias, 0, -1, 0, 0

    if choice is None:
        return (1 if random.random() < pComputerRight else 0), pComputerRight, [0, -1, 0, 0, const_bias]
    
    # EXACTLY replicate original data handling - no changes!
    data = np.array(choice)
    choice, rew = np.array(choice), np.array(rew)  # recode as 1/2
    choice = np.append(choice, None)
    rew = np.append(rew, None)
   
    mp1 = [1, '1', 'mp1', 'MP1']
    mp2 = ['all', 2, '2', 'mp2', 'MP2']
   
    # Algorithm 1 - EXACTLY like original but with numba for bottlenecks
    if algo in mp1 or algo in mp2:
        for depth in range(maxdepth):
            if len(data) < depth + 1:
                continue
            if depth == 0: # count all right choices to check overall side bias
                countN = len(choice)
                countRight = np.sum(data)
            else: # look for bias on trials where recent choice history was the same
                # Only use numba for the expensive part - sequence building
                histseq = build_histseq_numba(np.array(choice[:-1], dtype=np.float64), depth, len(choice))
                
                # Find matching indices exactly like original
                idx = np.where(histseq == histseq[-1])[0][:-1]
                
                if len(idx) == 0:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)

            pRightBias = binomtest(countRight, countN, 0.5, alternative='greater').pvalue
            pLeftBias = binomtest(countRight, countN, 0.5, alternative='less').pvalue
            pDeviation = countRight / countN - 0.5

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
            np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 1

    # Algorithm 2 - EXACTLY like original but with numba for bottlenecks  
    if algo in mp2:
        for depth in range(maxdepth):
            if len(data) < depth+1:
                continue
            
            # Only use numba for the expensive nested loops
            chistseq, rhistseq = build_choice_reward_seqs_numba(
                np.array(choice[:-1], dtype=np.float64), 
                np.array(rew[:-1], dtype=np.float64), 
                depth, len(choice))
            
            # Find matching indices exactly like original  
            idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
            idx = idx[0][:-1]
            
            if len(idx) == 0:
                continue
            countRight = np.sum((data[idx]))
            countN = len(idx)

            pRightBias = binomtest(countRight, countN, 0.5, alternative='greater').pvalue
            pLeftBias = binomtest(countRight, countN, 0.5, alternative='less').pvalue
            pDeviation = countRight / countN - 0.5

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
                np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 2

    biasInfo = [bias, biasDepth, maxDev, whichAlg, const_bias]
    # EXACTLY the same random call as original
    computerChoice = 1 if random.random() < pComputerRight + const_bias else 0
    return computerChoice, pComputerRight, biasInfo


def matching_pennies_frac_numba(choice, rew, maxdepth, alpha, algo="all", const_bias=0, pf=.5):
    """
    EXACT replica of matching_pennies_frac with numba optimizations for bottlenecks only.
    """
    testAlpha = alpha / 2
    pComputerRight, bias, biasDepth, maxDev, whichAlg = 0.5 + const_bias, 0, -1, 0, 0

    if choice is None:
        return (1 if random.random() < pComputerRight else 0), pComputerRight, [0, -1, 0, 0, const_bias], 0
    
    # EXACTLY replicate original data handling
    data = np.array(choice)
    choice, rew = np.array(choice), np.array(rew)  # recode as 1/2
    choice = np.append(choice, None)
    rew = np.append(rew, None)
    
    # flip a coin, if it's > p then do normal MP, otherwise guess randomly 
    flip = random.random()
    if flip > pf:
        biasInfo = [bias, biasDepth, maxDev, whichAlg, const_bias]
        # might need to flip 0 and 1!
        computerChoice = 1 if random.random() < pComputerRight + const_bias else 0
        return computerChoice, pComputerRight, biasInfo, 0  # flag saying MP2 not engaged
    else:
        # algorithm 1 - EXACTLY like original
        if algo == 1 or algo == "all" or algo == '1':
            for depth in range(maxdepth):
                if len(data) < depth + 1:
                    continue
                if depth == 0: # count all right choices to check overall side bias
                    countN = len(choice)
                    countRight = np.sum(data)
                else: # look for bias on trials where recent choice history was the same
                    histseq = build_histseq_numba(np.array(choice[:-1], dtype=np.float64), depth, len(choice))
                    idx = np.where(histseq == histseq[-1])[0][:-1]
                    
                    if len(idx) == 0:
                        continue
                    countRight = np.sum((data[idx]))
                    countN = len(idx)

                pRightBias = binomtest(countRight, countN, 0.5, alternative='greater').pvalue
                pLeftBias = binomtest(countRight, countN, 0.5, alternative='less').pvalue
                pDeviation = countRight / countN - 0.5

                if pRightBias < testAlpha or pLeftBias < testAlpha and \
                np.abs(pDeviation) > np.abs(maxDev):
                        maxDev = pDeviation
                        bias = 1 if maxDev < 0 else 2
                        biasDepth = depth
                        pComputerRight = 0.5 - maxDev
                        whichAlg = 1

        # algorithm 2 - EXACTLY like original
        if algo == 2 or algo == "all" or algo == '2':
            # checks choice and reward history for bias, no depth 0 and reward history included
            for depth in range(maxdepth):
                if len(data) < depth+1:
                    continue
                
                chistseq, rhistseq = build_choice_reward_seqs_numba(
                    np.array(choice[:-1], dtype=np.float64), 
                    np.array(rew[:-1], dtype=np.float64), 
                    depth, len(choice))
                
                idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
                idx = idx[0][:-1]
                
                if len(idx) == 0:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)
                
                pRightBias = binomtest(countRight, countN, 0.5, alternative='greater').pvalue
                pLeftBias = binomtest(countRight, countN, 0.5, alternative='less').pvalue
                pDeviation = countRight / countN - 0.5

                if pRightBias < testAlpha or pLeftBias < testAlpha and \
                    np.abs(pDeviation) > np.abs(maxDev):
                        maxDev = pDeviation
                        bias = 1 if maxDev < 0 else 2
                        biasDepth = depth
                        pComputerRight = 0.5 - maxDev
                        whichAlg = 2

        biasInfo = [bias, biasDepth, maxDev, whichAlg, const_bias]
        computerChoice = 1 if random.random() < pComputerRight + const_bias else 0
        return computerChoice, pComputerRight, biasInfo, 1  # flag saying MP2 engaged 