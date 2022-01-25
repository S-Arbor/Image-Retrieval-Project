## Functions to carry out basic and expanded queries

import sklearn.metrics.pairwise
import numpy as np


def basic_query(query, query_target, metric_function = sklearn.metrics.pairwise.euclidean_distances):
    """Return the indexes of the query_target images, arranged in ascending euclidean distance as compared to the query image"""
    
    query = query.reshape((1, -1))
    D = metric_function(query_target, query).squeeze()
    index = np.argsort(D)

    return(index)