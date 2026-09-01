import argparse
import json
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from utils import gini_coeff, log_pipeline_event


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate topic model fits.")
    parser.add_argument("--base_path", type=str, default="simul_data/", help="base path for simulation data.")
    parser.add_argument("--top_words", type=int, default=50, help="number of top words for RBO.")
    return parser.parse_args()


args = parse_args()
BASE_PATH = args.base_path
MODELS = ["lda", "ctm", "stm", "lsi"]
TOP_WORDS = args.top_words


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


# iterating data recursively
master_data = []

print(f"Scanning directory: {BASE_PATH} recursively for _corpus.json files...")
log_pipeline_event(BASE_PATH, "evaluation", "start", "Initiating evaluation pipeline across recursive subdirectories.")

for root, dirs, files in os.walk(BASE_PATH):
    if "_corpus.json" not in files:
        continue
        
    sub_path = root
    # Deduce a logical folder tag relative to BASE_PATH
    dir_tag = os.path.relpath(sub_path, BASE_PATH)
    print(f"Evaluating simulation: {dir_tag}")

    # load ground truth from JSON
    try:
        with open(os.path.join(sub_path, "_corpus.json"), 'r') as f:
            corpus_data = json.load(f)
        config = corpus_data["config"]
        vocab = corpus_data["vocab"]
        meta = pd.DataFrame(corpus_data["metadata"])
        
        # reconstruct keep_docs using document indices in JSON dtm
        keep_docs = [len(idx) > 0 for idx in corpus_data["dtm"]["indices"]]
        keep_docs = pd.Series(keep_docs)

        true_theta = pd.DataFrame(corpus_data["true_thetas"])[keep_docs].reset_index(drop=True)
        true_beta_overall = pd.DataFrame(corpus_data["true_betas"]["overall"])

        meta = meta[keep_docs].reset_index(drop=True)
        
        # calculate actual observed Gini of the corpus proportions dynamically
        observed_proportions = true_theta.mean(axis=0).tolist()
        observed_gini = gini_coeff(observed_proportions)
        
    except Exception as e:
        print(f"Skipping {dir_tag}: Missing or invalid unified corpus JSON ({e})")
        continue

    # load SearchK logs (global simulation-level log)
    fits_path = os.path.join(sub_path, "model_fits")
    try:
        global_search_log = pd.read_csv(os.path.join(fits_path, "simulation_master_log.csv")).iloc[0].to_dict()
    except:
        global_search_log = {}

    # determine number of model fits
    run_nums = [d.strip('run_') for d in os.listdir(fits_path) if (os.path.isdir(os.path.join(fits_path, d)) and d.startswith('run_'))]

    # evaluation
    for model in MODELS:
        is_deterministic = (model == "lsi") or (model == "stm" and os.path.exists(os.path.join(fits_path, "stm_theta.csv")))
        runs_to_check = [None] if is_deterministic else run_nums

        for run in runs_to_check:
            try:
                current_path = os.path.join(fits_path, f"run_{run}") if run else fits_path
                
                # run-specific SearchK metrics if they exist (stochastic search_k), otherwise fallback to global log
                run_metrics_path = os.path.join(current_path, "searchK_metrics.json")
                if os.path.exists(run_metrics_path):
                    with open(run_metrics_path, 'r') as f:
                        search_log = json.load(f)
                else:
                    search_log = global_search_log

                # Dynamic vocab alignment: identifying the active vocabulary that was actually fitted in the R model
                if model == "stm":
                    counts = meta['content_covar'].value_counts(normalize=True)
                    rep_group = list(counts.index)[0]
                    rep_file = os.path.join(current_path, f"stm_beta_group_{rep_group}.csv")
                    if not os.path.exists(rep_file):
                        rep_file = os.path.join(current_path, f"stm_beta_group_{float(rep_group)}.csv")
                else:
                    rep_file = os.path.join(current_path, f"{model}_beta_overall.csv")

                if not os.path.exists(rep_file):
                    raise FileNotFoundError(f"Representative beta file {rep_file} not found.")

                fitted_cols = pd.read_csv(rep_file, nrows=0).columns.tolist()
                active_vocab = [w for w in vocab if w in fitted_cols]
                vocab_indices = [vocab.index(w) for w in active_vocab]

                # filter ground truth parameters to active vocabulary space
                true_beta_subset = true_beta_overall.values[:, vocab_indices]

                if model == "stm":
                    # STM weighted averaging of group betas
                    counts = meta['content_covar'].value_counts(normalize=True)
                    f_beta = None

                    # overall topic beta for stm - weighted average of group-specific betas
                    for group_id, weight in counts.items():
                        file_name = f"stm_beta_group_{group_id}.csv"
                        file_path = os.path.join(current_path, file_name)
                        if not os.path.exists(file_path):
                            # float representation just in case
                            file_name = f"stm_beta_group_{float(group_id)}.csv"
                            file_path = os.path.join(current_path, file_name)
                            
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Fitted STM beta file not found for group {group_id} under {current_path}")
                            
                        g_beta = pd.read_csv(file_path)[active_vocab].values
                        if f_beta is None:
                            f_beta = np.zeros_like(g_beta)
                        f_beta += g_beta * weight

                    f_theta = pd.read_csv(os.path.join(current_path, "stm_theta.csv"))

                else:
                    f_beta = pd.read_csv(os.path.join(current_path, f"{model}_beta_overall.csv"))[active_vocab].values

                    if is_deterministic:
                        f_theta = None if model == "lsi" else pd.read_csv(os.path.join(current_path, f"{model}_theta.csv"))
                    else:
                        f_theta = pd.read_csv(os.path.join(current_path, f"{model}_theta.csv"))

                # calculaate evaluation metrics
                eval_metrics = evaluate_fit(true_beta_subset, f_beta, true_theta, f_theta, active_vocab)

                row = {
                    "dir_tag": dir_tag,
                    "prop_gini_coeff": round(observed_gini * 100),
                    "generation_mode": config.get("generation_mode", "stm"),
                    "min_word_freq": config.get("min_word_freq", 1),
                    "model": model.upper(),
                    "run": run if run else 1,
                    **eval_metrics
                }

                # adding searchK metrics
                row.update({k: v for k, v in search_log.items() if "Elbow" in k or "Final" in k})
                # config params
                row.update({k: v for k, v in config.items() if isinstance(v, (int, float, str))})

                master_data.append(row)

            except Exception as e:
                print(f"Error evaluating {model} in {dir_tag} run {run if run else 'deterministic'}: {e}")

if master_data:
    df_master = pd.DataFrame(master_data)
    output_csv = os.path.join(BASE_PATH, "prop_simulation_results.csv")
    df_master.to_csv(output_csv, index=False)
    msg = f"All evaluations complete! Saved master result table of length {len(df_master)} to {output_csv}"
    print(msg)
    log_pipeline_event(BASE_PATH, "evaluation", "finish", msg)
else:
    print("No evaluations completed successfully.")
    log_pipeline_event(BASE_PATH, "evaluation", "failure", "No evaluations completed successfully.")
