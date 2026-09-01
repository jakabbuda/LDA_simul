import numpy as np
import os
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.metrics.pairwise import cosine_similarity
import time


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


def log_pipeline_event(output_base, phase, status, message):
    """writes unified execution trace to a shared log file"""
    log_path = os.path.join(output_base, "simulation_pipeline.log")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"{'-' * 40 + '\n' if status.lower() == 'start' else ''}[{timestamp}] [{phase.upper()}] [{status.upper()}] {message}\n{'-' * 40 + '\n' if status.lower() == 'finish' else ''}"
    os.makedirs(output_base, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(log_line)


def generate_proportions_with_gini_and_bounds(n, target_gini, min_val=0.01):
    """used to generate topic proportins with specific gini value"""
    if min_val * n > 1.0:
        raise ValueError(f"{min_val=} too large: {n * min_val=} > 1")

    def objective(w):
        return (gini_coeff(w) - target_gini) ** 2

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    bounds = [(min_val, 1.0) for _ in range(n)]

    initial_w = np.linspace(min_val, 1.0, n)
    initial_w = initial_w / np.sum(initial_w)

    # optimization
    result = minimize(
        objective,
        initial_w,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    return [float(np.round(i, 6)) for i in sorted(result.x, reverse=True)]
