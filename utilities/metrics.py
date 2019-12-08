import numpy as np


def precision_at_k(r, k):
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(ground_truth, prediction, k):
    r = get_relevance_list(ground_truth, prediction, k)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def get_relevance_list(ground_truth, prediction, k):
    return np.asarray([1 if item in ground_truth else 0 for item in prediction[:k]])


def hitrate(left_out, predicted, k):
    return 1 if left_out in predicted[:k] else 0


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')

    return 0.


def ndcg_at_k(ground_truth, predictions, k, method=0):
    r = get_relevance_list(ground_truth, predictions, k=k)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
