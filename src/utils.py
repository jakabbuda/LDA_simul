import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


def markov_stationary(mtrx):
    """returns stationary distribution of a markov matrix"""
    if not np.isclose(mtrx.sum(axis=1), 1).all():
        raise ValueError(f"rowsums should be 1 but are {mtrx.sum(axis=1)}")
    evals, evecs = np.linalg.eig(mtrx.T)
    evec1 = evecs[:, np.isclose(evals, 1.0)].flatten().real  # eigenvector for eigenvalue == 1
    stat_dist = evec1 / evec1.sum()
    # resolve numerical precision issue
    stat_dist = np.clip(stat_dist, 0, None)
    return stat_dist / stat_dist.sum()


def match_topics(simulated_beta, recovered_beta):
    # similarity matrix (simulated x recovered)
    cost_matrix = 1 - cosine_similarity(simulated_beta, recovered_beta)
    # optimal pairing
    sim_ind, rec_ind = linear_sum_assignment(cost_matrix)
    # sim_ind[i] matches rec_ind[i]
    return list(zip(sim_ind, rec_ind))


def make_uncorrelated_markov(self_transition_vec):
    """
    makes an n*n uncorrelated markov mtrx based on a vector of n dimension
    representing self transition probab for each class
    """
    n = len(self_transition_vec)
    trm_v = (1 - self_transition_vec) / (n - 1)
    trm_mtr = np.vstack([trm_v] * n).T
    np.fill_diagonal(trm_mtr, self_transition_vec)

    return trm_mtr


def gini_coeff(lst):
    a = sorted(lst, reverse=True)
    return 2 * sum([(len(a) - i) * v for i, v in enumerate(a)]) / (sum(a) * len(a)) - (len(a) + 1) / len(a)
