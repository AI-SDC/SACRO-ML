"""Method for computing extreme case metrics"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def min_max_disc(
    y_true: np.ndarray, pred_probs: np.ndarray, x_prop: float = 0.1, log_p: bool = True
) -> tuple[float, float, float, float]:  # pylint: disable = line-too-long
    """
    Non-average-case methods for MIA attacks. Considers actual frequency of membership
    amongst samples with highest- and lowest- assessed probability of membership. If an
    MIA method confidently asserts that 5% of samples are members and 5% of samples are
    not, but cannot tell for the remaining 90% of samples, then these metrics will flag
    this behaviour, but AUC/advantage may not. Since the difference may be noisy, a
    p-value against a null of independence of true membership and assessed membership
    probability (that is, membership probabilities are essentially random) is also used
    as a metric (using a usual Gaussian approximation to binomial). If the p-value is
    low and the frequency difference is high (>0.5) then the MIA attack is successful
    for some samples.

    Parameters
    ----------
        y: np.ndarray
            true labels
        yp: np.ndarray
            probabilities of labels, or monotonic transformation of probabilties
        xprop: float
            proportion of samples with highest- and lowest- probability of membership to be
            considered
        logp: bool
            convert p-values to log(p).

    Returns
    -------
        maxd: float
            frequency of y=1 amongst proportion xprop of individuals with highest assessed
            membership probability
        mind: float
            frequency of y=1 amongst proportion xprop of individuals with lowest assessed
            membership probability
        mmd: float
            difference between maxd and mind
        pval: float
            p-value or log-p value corresponding to mmd against null hypothesis that random
            variables corresponding to y and yp are independent.

    Notes
    -----

    Examples
    --------
    >>> y = np.random.choice(2, 100)
    >>> yp = np.random.rand(100)
    >>> maxd, mind, mmd, pval = min_max_desc(y, yp, xprop=0.2, logp=True)

    """

    n_examples = int(np.ceil(len(y_true) * x_prop))
    pos_frequency = np.mean(y_true)  # average frequency
    y_order = np.argsort(pred_probs)  # ordering permutation

    # Frequencies
    # y values corresponding to lowest k values of yp
    y_lowest_n = y_true[y_order[:n_examples]]
    # y values corresponding to highest k values of yp
    y_highest_n = y_true[y_order[-(n_examples):]]
    maxd = np.mean(y_highest_n)
    mind = np.mean(y_lowest_n)
    mmd = maxd - mind

    # P-value
    # mmd is asymptotically distributed as N(0,sdm^2) under null.
    sdm = np.sqrt(2 * pos_frequency * (1 - pos_frequency) / n_examples)
    pval = 1 - norm.cdf(mmd, loc=0, scale=sdm)  # normal CDF
    if log_p:
        if pval < 1e-50:
            pval = -115.13
        else:
            pval = np.log(pval)

    # Return
    return maxd, mind, mmd, pval
