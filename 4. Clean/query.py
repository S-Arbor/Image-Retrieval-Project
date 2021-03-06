## Functions to carry out basic and expanded queries

import sklearn.metrics.pairwise
import numpy as np
import diffusion

def return_ranks(method, queries, gallery, **kwargs):

    if method == "basic":
        ranks = np.array([basic_query(query, gallery, **kwargs) for query in queries])
        return(ranks)
    elif method == "diffusion":
        ranks = diffusion.diffusion_ranks(queries, gallery, **kwargs)
        return(ranks)
    elif method == "expanded":
        ranks = np.array([qe_query(query, gallery, **kwargs) for query in queries])
        return(ranks)
    

def basic_query(query, query_target, metric_function = sklearn.metrics.pairwise.euclidean_distances):
    """Return the indexes of the query_target images, arranged in ascending euclidean distance as compared to the query image"""
    
    query = query.reshape((1, -1))
    D = metric_function(query_target, query).squeeze()
    index = np.argsort(D)

    return(index)

def qe_query(query, query_target, metric_function=sklearn.metrics.pairwise.euclidean_distances, type="qe baseline", n=5, alpha=1):
    """Run a query with query expansion, supported methods:
       - "qe baseline" : described in Total Recall (2007), new result is based on alpha proportion of requerying (e.g. alpha = 1,
                         then results after the top 5 will be completely determined by the top five"""
    
    original_results = basic_query(query, query_target, metric_function)

    if type == "qe baseline":
        # find top n results, combine top n into a new query, append results of new query to top n
        top_n_results = original_results[:n]
        second_query = np.average(query_target[top_n_results], axis=0)
        
        if alpha != 1:
            combined_queries = np.vstack([query, second_query])
            second_query = np.average([query, second_query], axis = 0, weights = [1-alpha, alpha])
            
        new_results = basic_query(second_query, query_target, metric_function)
        pruned_new_results = new_results[np.logical_not(np.isin(new_results, top_n_results))]
        results = np.concatenate([top_n_results, pruned_new_results])

        return(results)

    print("Something went wrong")
