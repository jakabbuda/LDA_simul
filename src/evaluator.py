import argparse
import json
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
from utils import gini_coeff, log_pipeline_event, parse_dir_tag


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate topic model fits.")
    parser.add_argument("--base_path", type=str, default="simul_data/", help="base path for simulation data.")
    parser.add_argument("--top_words", type=int, default=50, help="number of top words for RBO.")
    return parser.parse_args()


MODELS = ["lda", "ctm", "stm", "lsi"]


def calculate_rbo(list1, list2, p=0.9):
    """rank biased overlap - measures similarity of top word rankings - topic quality check"""
    s1, s2 = set(), set()
    score = 0.0
    max_depth = min(len(list1), len(list2))
    if max_depth == 0:
        return 0.0
    for d in range(1, max_depth + 1):
        s1.add(list1[d - 1])
        s2.add(list2[d - 1])
        intersection = len(s1.intersection(s2))
        score += (np.power(p, d - 1) * intersection) / d  # current position overlap inverse weighted by position
    norm = 1.0 - np.power(p, max_depth)
    return (score * (1 - p)) / norm if norm > 0 else 0.0


def evaluate_fit(true_beta, fitted_beta, true_theta, fitted_theta, vocab, top_words=50):
    """Hungarian alignment and metric calculation."""
    # Hungarian matching (cosine dist based)
    cost_matrix = cdist(true_beta, fitted_beta, metric='cosine')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Content metrics of matched topics
    cos_sims, rbo_scores = [], []
    vocab_arr = np.array(vocab)
    for t_idx, f_idx in zip(row_ind, col_ind):
        cos_sims.append(1 - cost_matrix[t_idx, f_idx])
        t_top = vocab_arr[np.argsort(true_beta[t_idx])[::-1][:top_words]]
        f_top = vocab_arr[np.argsort(fitted_beta[f_idx])[::-1][:top_words]]
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


def run_evaluation(base_path="simul_data/", top_words=50):
    # iterating data recursively
    master_data = []

    print(f"Scanning directory: {base_path} recursively for _corpus.json files...")
    log_pipeline_event(base_path, "evaluation", "start", "Initiating evaluation pipeline across recursive subdirectories.")

    # Pass 1: collect all qualifying corpus directories
    all_corpus_dirs = []
    for root, dirs, files in os.walk(base_path):
        if "_corpus.json" not in files:
            continue
        if not os.path.exists(os.path.join(root, "model_fits")):
            continue
        all_corpus_dirs.append(root)

    total = len(all_corpus_dirs)
    print(f"Found {total} corpus director{'y' if total == 1 else 'ies'} to evaluate.")

    # Pass 2: evaluate with progress reporting
    start_time = time.time()
    for corpus_idx, sub_path in enumerate(all_corpus_dirs):
        elapsed = time.time() - start_time
        if corpus_idx > 0:
            est_remaining = (elapsed / corpus_idx) * (total - corpus_idx)
            eta_str = f"est. remaining: {int(est_remaining)}s"
        else:
            eta_str = "est. remaining: N/A"
        dir_tag = os.path.relpath(sub_path, base_path)
        fits_path = os.path.join(sub_path, "model_fits")
        print(f"[{corpus_idx + 1}/{total}] Evaluating: {dir_tag} | elapsed: {int(elapsed)}s | {eta_str}")

        # load ground truth from JSON
        try:
            with open(os.path.join(sub_path, "_corpus.json"), 'r') as f:
                corpus_data = json.load(f)
            config = corpus_data["config"]
            vocab = corpus_data["vocab"]
            vocab_lookup = {w: i for i, w in enumerate(vocab)}
            meta = pd.DataFrame(corpus_data["metadata"])
            
            # reconstruct keep_docs using document indices in JSON dtm
            keep_docs = [len(idx) > 0 for idx in corpus_data["dtm"]["indices"]]
            keep_docs = pd.Series(keep_docs)

            true_theta = pd.DataFrame(corpus_data["true_thetas"]).fillna(0.0)[keep_docs].reset_index(drop=True)
            true_beta_overall = pd.DataFrame(corpus_data["true_betas"]["overall"])

            meta = meta[keep_docs].reset_index(drop=True)
            
            # calculate actual observed Gini of the corpus proportions dynamically
            observed_proportions = true_theta.mean(axis=0).tolist()
            observed_gini = gini_coeff(observed_proportions)
            
            parsed_dir = parse_dir_tag(dir_tag)
            
            target_gini = None
            if "target_gini" in config:
                try:
                    target_gini = float(config["target_gini"])
                except Exception:
                    pass
            if target_gini is None and "gini" in config:
                try:
                    target_gini = float(config["gini"])
                except Exception:
                    pass
            if target_gini is None and isinstance(config.get("topic_imbalance"), dict):
                try:
                    target_gini = float(config["topic_imbalance"].get("gini"))
                except Exception:
                    pass
            if target_gini is None and isinstance(config.get("topic_imbalance"), list) and len(config["topic_imbalance"]) > 0:
                try:
                    target_gini = float(config["topic_imbalance"][0].get("gini"))
                except Exception:
                    pass
            if target_gini is None and "target_gini" in parsed_dir:
                try:
                    target_gini = float(parsed_dir["target_gini"])
                except Exception:
                    pass
            if target_gini is None:
                target_gini = 0.0
            
        except Exception as e:
            print(f"Skipping {dir_tag}: Missing or invalid unified corpus JSON ({e})")
            continue

        # load SearchK logs (global simulation-level log)
        try:
            global_search_log = pd.read_csv(os.path.join(fits_path, "simulation_master_log.csv")).iloc[0].to_dict()
        except:
            global_search_log = {}

        # determine number of model fits
        run_dirs = [d for d in os.listdir(fits_path) if (os.path.isdir(os.path.join(fits_path, d)) and d.startswith('run_'))]
        run_nums = sorted([d.removeprefix('run_') for d in run_dirs], key=lambda x: int(x) if x.isdigit() else x)

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
                    vocab_indices = [vocab_lookup[w] for w in active_vocab if w in vocab_lookup]

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
                    eval_metrics = evaluate_fit(true_beta_subset, f_beta, true_theta, f_theta, active_vocab, top_words=top_words)

                    # Determine n_docs scalar
                    n_docs_val = config.get("n_docs")
                    if isinstance(n_docs_val, list):
                        n_docs_val = n_docs_val[0] if len(n_docs_val) > 0 else len(true_theta)
                    elif n_docs_val is None:
                        n_docs_val = parsed_dir.get("n_docs", len(true_theta))

                    row = {
                        "dir_tag": dir_tag,
                        "n_docs": int(n_docs_val),
                        "target_gini": target_gini,
                        "prop_gini_coeff": round(observed_gini * 100),
                        "generation_mode": config.get("generation_mode", parsed_dir.get("generation_mode", "stm")),
                        "min_word_freq": config.get("min_word_freq", parsed_dir.get("min_word_freq", 1)),
                        "model": model.upper(),
                        "run": run if run else 1,
                        **eval_metrics
                    }

                    # adding searchK metrics
                    row.update({k: v for k, v in search_log.items() if "Elbow" in k or "Final" in k})
                    # config params - unwrap 1-element lists to scalars
                    for k, v in config.items():
                        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (int, float, str)):
                            row[k] = v[0]
                        elif isinstance(v, (int, float, str, bool)):
                            row[k] = v

                    # Determine doc_len
                    doc_len = None
                    if "doc_len" in config:
                        doc_len = config["doc_len"]
                    elif "text_len_params" in config and isinstance(config["text_len_params"], dict):
                        doc_len = config["text_len_params"].get("lam")
                    if doc_len is None and "doc_len" in parsed_dir:
                        doc_len = parsed_dir["doc_len"]
                    if doc_len is not None:
                        row["doc_len"] = doc_len
                    
                    # overlay parsed directory attributes
                    for k, v in parsed_dir.items():
                        row[k] = v
                    row["n_docs"] = int(n_docs_val)
                    row["target_gini"] = target_gini

                    master_data.append(row)

                except Exception as e:
                    print(f"Error evaluating {model} in {dir_tag} run {run if run else 'deterministic'}: {e}")

    if master_data:
        df_master = pd.DataFrame(master_data)
        output_csv = os.path.join(base_path, "prop_simulation_results.csv")
        df_master.to_csv(output_csv, index=False)
        msg = f"All evaluations complete! Saved master result table of length {len(df_master)} to {output_csv}"
        print(msg)
        log_pipeline_event(base_path, "evaluation", "finish", msg)
    else:
        print("No evaluations completed successfully.")
        log_pipeline_event(base_path, "evaluation", "failure", "No evaluations completed successfully.")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.base_path, args.top_words)
