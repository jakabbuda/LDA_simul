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
    idx = np.argmin(np.abs(evals - 1.0))
    evec1 = evecs[:, idx].flatten().real  # eigenvector for eigenvalue == 1
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


PREFIX_MAP = {
    "ndocs": "n_docs",
    "generationmode": "generation_mode",
    "stopwordratio": "stopword_ratio",
    "minwordfreq": "min_word_freq",
    "gini": "target_gini",
    "overlapratio": "overlap_ratio",
    "unstandardratio": "unstandard_ratio",
    "numtopics": "num_topics",
    "vocabsizepertopic": "vocab_size_per_topic",
    "preveffectsize": "prev_effect_size",
    "conteffectsize": "cont_effect_size",
    "topiccovar": "topic_covar",
    "doclen": "doc_len",
    "textlen": "doc_len",
    "ngroupsprev": "n_groups_prev",
    "ngroupscont": "n_groups_cont"
}


def parse_dir_tag(dir_tag):
    """
    Parses directory paths/tags to a dictionary of parameter names and typed values.
    """
    extracted = {}
    parts = str(dir_tag).replace("\\", "/").split("/")
    for part in parts:
        subparts = part.split("_")
        i = 0
        while i < len(subparts):
            sp = subparts[i]
            for prefix, col in sorted(PREFIX_MAP.items(), key=lambda x: -len(x[0])):
                clean_sp = sp.replace("_", "").lower()
                if clean_sp.startswith(prefix):
                    remainder = clean_sp[len(prefix):]
                    if len(remainder) > 0:
                        val_str = remainder
                    elif i + 1 < len(subparts):
                        val_str = subparts[i + 1]
                        i += 1
                    else:
                        continue
                    
                    val_str = val_str.replace("p", ".")
                    try:
                        if "." in val_str:
                            val = float(val_str)
                        else:
                            val = int(val_str)
                    except ValueError:
                        val = val_str
                    extracted[col] = val
                    break
            i += 1
    return extracted

