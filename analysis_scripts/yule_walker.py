import numpy as np
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt

def cov(x,y):
    assert len(x) == len(y)
    xp = x - np.mean(x)
    yp = y - np.mean(y)
    return np.sum(xp*yp)/len(x)

def yule_walker(data, order):
    
    corrs = [1]
    for p in range(1,order+1):
        x = data[order:]
        y = data[order-p:-p]
        sig_x = np.sqrt(cov(x,x))
        sig_y = np.sqrt(cov(y,y))
        covar = cov(x,y)
        corr = covar/(sig_x * sig_y)
        corrs.append(corr)
        
    
    corrs = np.array(corrs)
    row = corrs[:-1]
    col  = corrs[:-1]
    r = corrs[1:]
    
    R = scipy.linalg.toeplitz(col, row)
    
    Rinv = np.linalg.inv(R)
    
    return Rinv @ r

def pacf(data,order):
    
    pacfs = []
    for i in range(1,order+1):
        ars = yule_walker(data,i)
        pacfs.append(ars[-1])
    return pacfs
    
def AR2_sim(n, ar1, ar2, sd):
    Xt = [0, 0]
    for i in range(2, n):
        Xt.append(ar1 * Xt[i - 1] + ar2 * Xt[i - 2] + np.random.normal(loc=0, scale=sd))
    return Xt



def ARN_sim(n, ar, sd):
    Xt = [0] * len(ar)
    for i in range(len(ar), n):
        # Xt.append(ar1 * Xt[i - 1] + ar2 * Xt[i - 2] + np.random.normal(loc=0, scale=sd))
        Xt.append(np.sum(np.flip(np.array(ar)) * np.array(Xt[i-len(ar):i])) + np.random.normal(loc=0, scale=sd))
    return Xt

def test():
    ar = [.15,.24,-.3,.3]
    dat = ARN_sim(500,ar,2)
    # plt.plot([.15,.73])
    plt.plot(ar)
    plt.plot(yule_walker(dat,len(ar)))
    plt.show()
    plt.plot(pacf(dat,len(ar)))
    plt.show()

