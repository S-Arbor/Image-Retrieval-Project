## Evaluation methods for query results

import numpy as np

def evaluate(ranks, query_names, gallery_names, kappas = [1,5,10]):

    average_precisions = []
    precisions = {}
    for kappa in kappas:
        precisions[kappa] = []
    
    for (query_name, query_ranks) in zip(query_names, ranks):
        truth_vals = [query_name == gallery_names[index] for index in query_ranks]
        average_precisions.append(average_precision(truth_vals))
        for kappa in kappas:
            precisions[kappa].append(precision_at_k(truth_vals, kappa))

    precision = {kappa : np.mean(precisions[kappa]) for kappa in kappas}
    
    return(np.mean(average_precisions), precision)

def average_precision(truth_values):
    """Given a boolean input of whether returned query values are correct or false, return the average precision.
    e.g. average_precision([True, True, False, True]) ~ 0.85
    """
    precisions = []
    for (index, val) in enumerate(truth_values):
        if val: # == True
            precisions.append(truth_values[:index + 1].count(True) / (index + 1))      

    return(np.mean(precisions))

def precision_at_k(truth_values, k, warnings=True):
    """Return proportions of true values in the first k elements."""
    p_at_k = truth_values[:k].count(True) / k
    
    return(p_at_k)