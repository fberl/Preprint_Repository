import numpy as np
from math import lgamma as gammln


def binocdf(x, n, p):
    """
    binocdf(k, n, p)

    Binomial cumulative distribution function.

    Returns the sum of the terms 0 through k of the Binomial probability
    density function:

    .. math:: F(k) = \\sum_{j=0}^{k} \\binom{n}{j} p^j (1-p)^{n-j}

    Parameters
    ----------
    k : array_like
        Number of successes (0 through n).
    n : array_like
        Number of trials.
    p : array_like
        Probability of success in each trial.

    Returns
    -------
    F : ndarray
        Cumulative probability of k successes in n trials each with
        probability p of success.

    Notes
    -----
    The probability density function for `binocdf` is

    .. math:: P(k) = \\binom{n}{k} p^k (1-p)^{n-k}

    for k in {0, 1,..., n}.

    `binocdf` is identical to the `binom.cdf` function in scipy.stats.

    Examples
    --------
    Find the cumulative probability of 3 or fewer successes in 5 trials
    with a success probability of 0.3:

    >>> binocdf(3, 5, 0.3)
    0.8369199999999999

    """
    x = np.asarray(x)
    n = np.asarray(n)
    p = np.asarray(p)

    if np.any(x < 0) or np.any(x > n):
        raise ValueError("x must be in the range [0, n]")

    return betai(n-x,x+1,1-p)

def betai(a, b, x):
    bt = 0
    if (x < 0) or (x > 1): return .5
    if (x == 0) or (x == 1): bt = 0
    else:
        bt = np.exp(gammln(a+b)-gammln(a)-gammln(b)+a*np.log(x)+b*np.log(1-x))
    if (x < (a+1)/(a+b+2)):
        return bt*betacf(a,b,x)/a
    else:
        return 1-bt*betacf(b,a,1-x)/b

def betacf(a ,b ,x):
    qab = a+b
    qap = a+1
    qam = a-1
    c=1
    d=1-qab*x/qap
    if (np.abs(d) < 1e-30): d=1e-30
    d = 1/d
    h = d
    for m in range(1,10000):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1+aa*d
        if (np.abs(d) < 1e-30): d=1e-30
        c = 1+aa/c
        if (np.abs(c) < 1e-30): c=1e-30
        d = 1/d
        h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1+aa*d
        if (np.abs(d) < 1e-30): d=1e-30
        c = 1+aa/c
        if (np.abs(c) < 1e-30): c=1e-30
        d = 1/d
        del_ = d*c
        h *= del_
        if (np.abs(del_-1) < 3e-7): break
    return h


    
        
 