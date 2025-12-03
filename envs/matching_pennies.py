import numpy as np
from random import random
from scipy.stats import binomtest

def matching_pennies(choice,rew,maxdepth,alpha,algo = "all",const_bias = 0):
    '''
     matching pennies algorithms
    check previous choice and reward history for any choice bias 
    if no bias, pComputerRight = 0.5
    if bias detected, picks to minimize the subject's reward 
       (pComputerRight = 1 - p(subject choosing right)

    inputs: 
    data: choices/outcomes from all completed trials so far
        choice - choices from session (0 Left / 1 Right) 
        rew - outcomes from session (0 Unrewarded / 1 Rewarded)
    max depth - number from 1-4 with maximum length of recent choice
        sequences to test 
    alpha - significance threshold for two-sided tests

    outputs:
    computerChoice: 0 - left / 1 - right
    pComputerRight: probability of computer picking right on a given trial 
        (used to draw the computer's choice)
    biasInfo: vector containing information about the computer's bias, if any
        (1): bias detected 
            0 - no bias / 1 if left bias / 2 if right bias
        (2): depth of bias detected 
            -1 - no bias / 0-4 - bias depth with maximum deviation
                deviation used to bias computer choice
        (3): magnitude of bias (pRight - 0.5)
            0 - no bias / <0 - left bias / >0 - right bias
        (4): which algorithm was used to bias the computer's choice 
            0 - no bias / 1 - algorithm 1 / 2 - algorithm 2
            (5): constant bias term from argument, that adds or subtracts a certain probability to pComputerRight
    '''
    testAlpha = alpha/2
    pComputerRight,bias,biasDepth,maxDev,whichAlg = 0.5 + const_bias ,0,-1,0,0

    if choice is None:
        return (1 if random() < pComputerRight else 0), pComputerRight, [0,-1,0,0,const_bias]
    data = np.array(choice)
    choice, rew = np.array(choice) , np.array(rew) #recode as 1/2
    choice = np.append(choice,None)
    rew = np.append(rew,None)
   
    mp1 = [1, '1', 'mp1', 'MP1']
    mp2 = ['all', 2, '2', 'mp2', 'MP2']
   
    #algorithm 1
    if algo in mp1 or algo in mp2 :
        for depth in range(maxdepth):
            if len(data) < depth + 1:
                continue
            if depth == 0: #count all right choices to check overall side bias
                countN = len(choice)
                countRight = np.sum(data)
            else: #look for bias on trials where recent choice history was the same
                histseq = np.zeros(len(choice))
                for trial in range(depth+1,len(choice)):
                    seq = 0
                    for currdepth in range(1,depth):
                        seq += choice[trial-currdepth]* (10**(currdepth-1))
                    histseq[trial] = seq
                idx = np.where(histseq == histseq[-1])[0][:-1]
                # if not idx:
                if len(idx) == 0:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)

            pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
            pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
            np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 1

    #algorithm 2
    if algo in mp2:
        #checks choice and reward history for bias, no depth 0 and reward history included
        for depth in range(maxdepth):
            if len(data) < depth+1:
                continue
            chistseq = np.empty(len(choice))
            rhistseq = np.empty(len(rew))

            for trial in range(depth+1,len(choice)):
                cseq, rseq = 0,0
                for currdepth in range(1,depth):
                    cseq += choice[trial-currdepth] * 10 ** (currdepth-1)
                    rseq += rew[trial-currdepth] * 10 ** (currdepth-1)
                chistseq[trial] = cseq
                rhistseq[trial] = rseq
            idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
            idx = idx[0][:-1]
            # if not idx:
            if len(idx) == 0:
                continue
            countRight = np.sum((data[idx]))
            countN = len(idx)

            pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
            pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
            pDeviation = countRight / countN - 0.5 #p(Right)

            if pRightBias < testAlpha or pLeftBias < testAlpha and \
                np.abs(pDeviation) > np.abs(maxDev):
                    maxDev = pDeviation
                    bias = 1 if maxDev < 0 else 2
                    biasDepth = depth
                    pComputerRight = 0.5 - maxDev
                    whichAlg = 2

    biasInfo = [bias,biasDepth,maxDev,whichAlg,const_bias]
    #might need to flip 0 and 1!
    computerChoice = 1 if random() < pComputerRight + const_bias else 0
    return computerChoice,pComputerRight,biasInfo


def matching_pennies_frac(choice,rew,maxdepth,alpha,algo = "all",const_bias = 0, pf = .5):
    '''
    Like MP but guesses randomly with probability 1-p
    '''
    testAlpha = alpha/2
    pComputerRight,bias,biasDepth,maxDev,whichAlg = 0.5 + const_bias ,0,-1,0,0

    if choice is None:
        return (1 if random() < pComputerRight else 0), pComputerRight, [0,-1,0,0,const_bias]
    data = np.array(choice)
    choice, rew = np.array(choice) , np.array(rew) #recode as 1/2
    choice = np.append(choice,None)
    rew = np.append(rew,None)
    # flip a coin, if it's > p then do normal MP, otherwise guess randomly 
    flip = random()
    if flip > pf:
        biasInfo = [bias,biasDepth,maxDev,whichAlg,const_bias]
        #might need to flip 0 and 1!
        computerChoice = 1 if random() < pComputerRight + const_bias else 0
        return computerChoice,pComputerRight,biasInfo, 0 # flag saying MP2 not engaged
    else:
        #algorithm 1
        if algo == 1 or algo == "all" or algo == '1':
            for depth in range(maxdepth):
                if len(data) < depth + 1:
                    continue
                if depth == 0: #count all right choices to check overall side bias
                    countN = len(choice)
                    countRight = np.sum(data)
                else: #look for bias on trials where recent choice history was the same
                    histseq = np.zeros(len(choice))
                    for trial in range(depth+1,len(choice)):
                        seq = 0
                        for currdepth in range(1,depth):
                            seq += choice[trial-currdepth]* (10**(currdepth-1))
                        histseq[trial] = seq
                    idx = np.where(histseq == histseq[-1])[0][:-1]
                    # if not idx:
                    if len(idx) == 0:
                        continue
                    countRight = np.sum((data[idx]))
                    countN = len(idx)

                pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
                pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
                pDeviation = countRight / countN - 0.5 #p(Right)

                if pRightBias < testAlpha or pLeftBias < testAlpha and \
                np.abs(pDeviation) > np.abs(maxDev):
                        maxDev = pDeviation
                        bias = 1 if maxDev < 0 else 2
                        biasDepth = depth
                        pComputerRight = 0.5 - maxDev
                        whichAlg = 1

        #algorithm 2
        if algo == 2 or algo == "all" or algo == '2':
            #checks choice and reward history for bias, no depth 0 and reward history included
            for depth in range(maxdepth):
                if len(data) < depth+1:
                    continue
                chistseq = np.empty(len(choice))
                rhistseq = np.empty(len(rew))

                for trial in range(depth+1,len(choice)):
                    cseq, rseq = 0,0
                    for currdepth in range(1,depth):
                        cseq += choice[trial-currdepth] * 10 ** (currdepth-1)
                        rseq += rew[trial-currdepth] * 10 ** (currdepth-1)
                    chistseq[trial] = cseq
                    rhistseq[trial] = rseq
                idx = np.where(np.logical_and(chistseq == chistseq[-1], rhistseq == rhistseq[-1]))
                idx = idx[0][:-1]
                # if not idx:
                if len(idx) == 0:
                    continue
                countRight = np.sum((data[idx]))
                countN = len(idx)
                
                pRightBias = binomtest(countRight,countN,0.5,alternative='greater').pvalue #p(X>=x)
                pLeftBias = binomtest(countRight,countN,0.5,alternative='less').pvalue #p(X<=x)
                pDeviation = countRight / countN - 0.5 #p(Right)

                if pRightBias < testAlpha or pLeftBias < testAlpha and \
                    np.abs(pDeviation) > np.abs(maxDev):
                        maxDev = pDeviation
                        bias = 1 if maxDev < 0 else 2
                        biasDepth = depth
                        pComputerRight = 0.5 - maxDev
                        whichAlg = 2

        biasInfo = [bias,biasDepth,maxDev,whichAlg,const_bias]
        computerChoice = 1 if random() < pComputerRight + const_bias else 0
        return computerChoice,pComputerRight,biasInfo, 1 # flag saying MP2 engaged
