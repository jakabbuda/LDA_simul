import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

BASE_PATH = "simul_data/"
MODELS = ["lda", "ctm", "stm", "lsi"]
N_RUNS = 4
TOP_WORDS = 50


def calculate_rbo(list1, list2, p=0.9):
    """rank biased overlap - measures similarity of top word rankings - topic quality check"""
    s1, s2 = set(), set()
    score = 0.0
    max_depth = min(len(list1), len(list2))
    for d in range(1, max_depth + 1):
        if d <= len(list1):
            s1.add(list1[d - 1])
        if d <= len(list2):
            s2.add(list2[d - 1])
        intersection = len(s1.intersection(s2))
        score += (np.power(p, d - 1) * intersection) / d  # current position overlap inverse weighted by postition
    return score * (1 - p)


def evaluate_fit(true_beta, fitted_beta, true_theta, fitted_theta, vocab):
    """Hungarian alignment and metric calculation."""
    # Hungarian matching (cosine dist based)
    cost_matrix = cdist(true_beta, fitted_beta, metric='cosine')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Content metrics of matched topics
    cos_sims, rbo_scores = [], []
    vocab_arr = np.array(vocab)
    for t_idx, f_idx in zip(row_ind, col_ind):
        cos_sims.append(1 - cost_matrix[t_idx, f_idx])
        t_top = vocab_arr[np.argsort(true_beta[t_idx])[::-1][:TOP_WORDS]]
        f_top = vocab_arr[np.argsort(fitted_beta[f_idx])[::-1][:TOP_WORDS]]
        rbo_scores.append(calculate_rbo(t_top.tolist(), f_top.tolist()))

    # proportion metrics on theta
    rmse = np.nan
    if fitted_theta is not None and not fitted_theta.isnull().values.all():
        matched_true_th = true_theta.iloc[:, row_ind].values
        matched_fit_th = fitted_theta.iloc[:, col_ind].values
        rmse = np.sqrt(np.mean((matched_true_th - matched_fit_th) ** 2))

        # 'Orphaned' mass (fitted topics not mapped to true topics)
        all_fitted = set(range(fitted_beta.shape[0]))
        orphaned_idx = list(all_fitted - set(col_ind))
        orphaned_mass = fitted_theta.iloc[:, orphaned_idx].sum(axis=1).mean() if orphaned_idx else 0.0
    else:
        orphaned_mass = np.nan

    return {
        "match_count": len(row_ind),
        "mean_cosine": np.mean(cos_sims),
        "mean_RBO": np.mean(rbo_scores),
        "RMSE_theta": rmse,
        "orphaned_mass": orphaned_mass,
        "K_fitted": fitted_beta.shape[0],
        "K_true": true_beta.shape[0]
    }


# iterating data
master_data = []

for gini_folder in os.listdir(BASE_PATH):
    gini_path = os.path.join(BASE_PATH, gini_folder)
    if not os.path.isdir(gini_path):
        continue

    for sub in os.listdir(gini_path):
        sub_path = os.path.join(gini_path, sub)
        if not os.path.isdir(sub_path):
            continue

        print(f"Processing: {gini_folder} -> {sub}")

        # 1. Load Ground Truth
        try:
            config = json.load(open(os.path.join(sub_path, "_config.json")))
            dtm = pd.read_csv(os.path.join(sub_path, "_dtm.csv"))
            vocab = dtm.columns.tolist()

            keep_docs = dtm.sum(axis=1) > 0

            true_theta = pd.read_csv(os.path.join(sub_path, "_true_thetas.csv"))[keep_docs].reset_index(drop=True)
            true_beta_overall = pd.read_csv(os.path.join(sub_path, "_true_beta_overall.csv"))[vocab]

            meta = pd.read_csv(os.path.join(sub_path, "_meta.csv"))[keep_docs]
        except Exception as e:
            print(f"Skipping {sub}: Missing GT files ({e})")
            continue

        # 2. Load SearchK Logs
        fits_path = os.path.join(sub_path, "model_fits")
        try:
            search_log = pd.read_csv(os.path.join(fits_path, "simulation_master_log.csv")).iloc[0].to_dict()
        except:
            search_log = {}

        # 3. Evaluate Models
        for model in MODELS:
            is_lsi = (model == "lsi")
            runs_to_check = [None] if is_lsi else range(1, N_RUNS + 1)

            for run in runs_to_check:
                try:
                    current_path = os.path.join(fits_path, f"run_{run}") if run else fits_path

                    if model == "stm":
                        # STM requires weighted averaging of group betas
                        counts = meta['content_covar'].value_counts(normalize=True)
                        f_beta = None

                        # overall topic beta for stm  TODO: calculate separately
                        for group_id, weight in counts.items():
                            g_beta = pd.read_csv(os.path.join(current_path, f"stm_beta_group_{group_id}.csv"))[
                                vocab].values
                            if f_beta is None:
                                f_beta = np.zeros_like(g_beta)
                            f_beta += g_beta * weight

                        f_theta = pd.read_csv(os.path.join(current_path, "stm_theta.csv"))

                    else:
                        f_beta = pd.read_csv(os.path.join(current_path, f"{model}_beta_overall.csv"))[vocab].values

                        if is_lsi:
                            f_theta = None
                        else:
                            f_theta = pd.read_csv(os.path.join(current_path, f"{model}_theta.csv"))

                    # Perform Evaluation
                    eval_metrics = evaluate_fit(true_beta_overall.values, f_beta, true_theta, f_theta, vocab)

                    # Consolidate Row
                    row = {
                        "gini_folder": gini_folder,
                        "prop_gini_coeff": 0 if gini_folder == 'base_corp' else int(gini_folder.split('_')[-1]),
                        "generation_mode": "markov" if "markov" in sub else "stm",
                        "min_word_freq": 5 if "freq5" in sub else 1,
                        "model": model.upper(),
                        "run": run if run else 1,
                        **eval_metrics
                    }

                    row.update({k: v for k, v in search_log.items() if "Elbow" in k or "Final" in k})
                    row.update({k: v for k, v in config.items() if isinstance(v, (int, float, str))})

                    master_data.append(row)

                except Exception as e:
                    print(f"Error evaluating {model} in {sub} run {run}: {e}")

df_master = pd.DataFrame(master_data)
df_master.to_csv(f"{BASE_PATH}/prop_simulation_results.csv", index=False)
