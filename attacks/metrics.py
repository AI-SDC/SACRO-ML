'''
Calculate metrics.
'''

from typing import Iterable#, Optional, Any
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import norm
from .mia_extremecase import min_max_disc

VAR_THRESH = 1e-2

def div(x,y, default):
    '''
    Solve the problem of division by 0 and round up.
    '''
    if y!=0:
        return round(float(x/y),8)
    else:
        #print('Warning: division by 0', x,y)
        return float(default)

def tpr_at_fpr(y_true: Iterable[float], y_score: Iterable[float], fpr: float=0.001, fpr_perc: bool=False) -> float:
    '''
    Compute the TPR at a fixed FPR.
    In particular, returns the TPR value at idx where idx is the first location
    at which the FPR is >= the passsed value fpr. Does not use interpolation.
    '''

    if fpr_perc:
        fpr /= 100.


    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_score)
    exact_pos = np.where(fpr_vals == fpr)[0]
    if len(exact_pos) > 0:
        return tpr_vals[exact_pos[-1]]

    idx = np.where(fpr_vals >= fpr)[0][0]
    return tpr_vals[idx]

def expected_auc_var(auc: float, num_pos: int, num_neg: int) -> float:
    '''
    Compute variance of AUC under assumption of uniform probabilities
    uses the expression given as eqn (2) in  https://cs.nyu.edu/~mohri/pub/area.pdf
    '''
    p_xxy = p_xyy = 1/3
    var = (auc * (1 - auc) + (num_pos - 1) * (p_xxy - auc**2) + (num_neg - 1) * (p_xyy - auc**2)) /\
        (num_pos * num_neg)
    return var

def get_metrics(clf,
                X_test:Iterable[float],
                y_test:Iterable[float],
                permute_rows:bool=True):
    """
    Calculate metrics, including attacker advantage for MIA binary.
    Implemented as Definition 4 on https://arxiv.org/pdf/1709.01604.pdf
    which is also implemented in tensorFlow-privacy https://github.com/tensorflow/privacy

    clf: fitted model.
    X_test: test data.
    y_test: test data labels.

    returns a dictionary with several metrics.

    True positive rate or recall (TPR)
    False positive rate (FPR), proportion of negative examples incorrectly classified as positives
    False alarm rate (FAR), proportion of objects classified as positives that are incorrect, also known as false discovery rate
    True neagative rate (TNR)
    Positive predictive value or precision (PPV)
    Negative predictive value (NPV)
    False neagative rate (FNR)
    Accuracy (ACC)
    F1 Score - harmonic mean of precision and recall.
    Advantage
    """
    metrics = {}
    if permute_rows:
        N, _ = X_test.shape
        order = np.random.RandomState(seed=10).permutation(N)
        X_test = X_test[order, :]
        y_test = y_test[order]
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #print('tn', tn, 'fp',fp,'fn', fn,'tp', tp)

    metrics['TPR'] = round(float(tp/(tp + fn)), 8) #true positive rate or recall
    metrics['FPR'] = round(float(fp / (fp + tn)), 8) #false positive rate, proportion of negative examples incorrectly classified as positives
    metrics['FAR'] = div(fp, (fp + tp), 0) #False alarm rate, proportion of things classified as positives that are incorrect, also known as false discovery rate
    metrics['TNR'] = round(float(tn / (tn + fp)), 8) #true negative rate or specificity
    metrics['PPV'] = div(tp, (tp + fp), 0) #precision or positive predictive value
    metrics['NPV'] = div(tn, (tn + fn), 0) #negative predictive value
    metrics['FNR'] = round(float(fn / (tp + fn)), 8) #false negative rate
    metrics['ACC'] = round(float((tp + tn) / (tp + fp + fn + tn)), 8) #overall accuracy
    metrics['F1score'] = div(2*metrics['PPV']*metrics['TPR'], metrics['PPV']+metrics['TPR'], 0)#harmonic mean of precision and sensitivity
    metrics['Advantage'] = float(abs(metrics['TPR']-metrics['FPR']))
    #calculate AUC of model
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    metrics['AUC'] = round(roc_auc_score(y_test, y_pred_proba),8)

    # Calculate AUC p-val
    auc_null_var = expected_auc_var(0.5, y_test.sum(), len(y_test) - y_test.sum())
    # Probability of getting an AUC higher than that observed
    prob_higher_auc = 1 - norm.cdf(metrics['AUC'], loc=0.5, scale=np.sqrt(auc_null_var))
    metrics['P_HIGHER_AUC'] = prob_higher_auc

    fmax, fmin, fdif, pdif = min_max_disc(y_test, y_pred_proba)
    metrics['FMAX01'] = fmax
    metrics['FMIN01'] = fmin
    metrics['FDIF01'] = fdif
    metrics['PDIF01'] = -pdif # use -log(p) so answer is positive

    fmax, fmin, fdif, pdif = min_max_disc(y_test, y_pred_proba, x_prop=0.2)
    metrics['FMAX02'] = fmax
    metrics['FMIN02'] = fmin
    metrics['FDIF02'] = fdif
    metrics['PDIF02'] = -pdif # use -log(p) so answer is positive

    fmax, fmin, fdif, pdif = min_max_disc(y_test, y_pred_proba, x_prop=0.01)
    metrics['FMAX001'] = fmax
    metrics['FMIN001'] = fmin
    metrics['FDIF001'] = fdif
    metrics['PDIF001'] = -pdif # use -log(p) so answer is positive


    # Add some things useful for debugging / filtering
    metrics['pred_prob_var'] = y_pred_proba.var()

    # TPR at various FPR
    fpr_vals = [0.5, 0.2, 0.1, 0.01, 0.001, 0.00001] # taken from https://arxiv.org/pdf/2112.03570.pdf
    for fpr in fpr_vals:
        tpr = tpr_at_fpr(y_test, y_pred_proba, fpr=fpr)
        name = f'TPR@{fpr}'
        metrics[name] = tpr

    fpr, tpr, roc_thresh = roc_curve(y_test, y_pred_proba)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_thresh'] = roc_thresh

    return metrics
