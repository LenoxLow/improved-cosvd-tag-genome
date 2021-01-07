"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mae
    fcp
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from six import iteritems

def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp

def dcg_score (y_true, y_score, k = 10, gains = "exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k: float
    """
    order = np.argsort (y_score) [::-1]
    y_true = np.take (y_true, order [: k])

    if gains == "exponential":
        gains = 2 ** y_true-1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError ("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2 (np.arange (len (y_true)) + 2)
    return np.sum (gains / discounts)

def ndcg_score (y_true, y_score, k = 10, gains = "exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true: array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score: array-like, shape = [n_samples]
        Predicted scores.
    k: int
        Rank.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k: float
    """
    best = dcg_score (y_true, y_true, k, gains)
    actual = dcg_score (y_true, y_score, k, gains)
    return actual / best

def ndcg_5(predictions, k=5, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    scores = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        true_score = list(true_r for (_, true_r) in user_ratings)
        est_score = list(est for (est, _) in user_ratings)

        scores[uid] = ndcg_score(true_score, est_score, k)

    score = sum(res for res in scores.values()) / len(scores)

    if verbose:
        print('nDCG@' + str(k) + ':  {0:1.4f}'.format(score))

    return score

def ndcg_10(predictions, k=10, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    scores = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        true_score = list(true_r for (_, true_r) in user_ratings)
        est_score = list(est for (est, _) in user_ratings)

        scores[uid] = ndcg_score(true_score, est_score, k)

    score = sum(res for res in scores.values()) / len(scores)

    if verbose:
        print('nDCG@' + str(k) + ':  {0:1.4f}'.format(score))

    return score

def ndcg_15(predictions, k=15, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    scores = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        true_score = list(true_r for (_, true_r) in user_ratings)
        est_score = list(est for (est, _) in user_ratings)

        scores[uid] = ndcg_score(true_score, est_score, k)

    score = sum(res for res in scores.values()) / len(scores)

    if verbose:
        print('nDCG@' + str(k) + ':  {0:1.4f}'.format(score))

    return score

def prec_5(predictions, trainset, k=5, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    if verbose:
        print('Precision@K:  {0:1.4f}'.format(precision))

    return precision

def rec_5(predictions, trainset, k=5, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    recall = (sum(rec for rec in recalls.values()) / len(recalls))
    if verbose:
        print('Recall@K:  {0:1.4f}'.format(recall))

    return recall

def prec_10(predictions, trainset, k=10, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    if verbose:
        print('Precision@K:  {0:1.4f}'.format(precision))

    return precision

def rec_10(predictions, trainset, k=10, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    recall = (sum(rec for rec in recalls.values()) / len(recalls))
    if verbose:
        print('Recall@K:  {0:1.4f}'.format(recall))

    return recall

def prec_15(predictions, trainset, k=15, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    if verbose:
        print('Precision@K:  {0:1.4f}'.format(precision))

    return precision

def rec_15(predictions, trainset, k=15, verbose=True):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        if trainset.ur.get(uid) is None:
            threshold = 3.5
        else:
            threshold = np.percentile(list(r for _, r in trainset.ur.get(uid)), 75)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    recall = (sum(rec for rec in recalls.values()) / len(recalls))
    if verbose:
        print('Recall@K:  {0:1.4f}'.format(recall))

    return recall
