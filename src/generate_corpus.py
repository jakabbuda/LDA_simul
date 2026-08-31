import argparse
from itertools import product
import json
import numpy as np
import os
import sys
import time

from corpus import SyntheticCorpus
from utils import make_uncorrelated_markov


def parse_args():
    parser = argparse.ArgumentParser(description="corpus generator")
    # find sibling config file path for default
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_config.json")
    
    parser.add_argument("--config", type=str, default=default_config,
                        help="Path to the JSON parameterization configuration file.")
    parser.add_argument("--output_base", type=str, default="simul_test01",
                        help="directory where corpora are saved")
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def save_reproduction_log(output_dir, args_dict, config_path, corpus_info=True, bash_command=True):
    os.makedirs(output_dir, exist_ok=True)
    if corpus_info:
        with open(os.path.join(output_dir, "reproduction_info.json"), "w") as f:
            json.dump({
                "argv": sys.argv,
                "config_file_used": config_path,
                "parameters": {k: str(v) if (callable(v) or isinstance(v, np.ndarray)) else v for k, v in args_dict.items()}
            }, f, indent=4)
        
    if bash_command:
        with open(os.path.join(output_dir, "reproduce_command.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run this to reproduce this exact run using python CLI\n")
            f.write("python3 " + " ".join(sys.argv) + "\n")
        os.chmod(os.path.join(output_dir, "reproduce_command.sh"), 0o755)

def run_config_simulation(config_data, output_base, config_path):
    base_params = config_data.get("base_parameters", {})
    grid_params = config_data.get("grid_parameters", {})
    
    # normalize grid_params lists
    normalized_grid = {}
    for k, v in grid_params.items():
        normalized_grid[k] = v if isinstance(v, list) else [v]
            
    varying_keys = [k for k, v in normalized_grid.items() if len(v) > 1]
    
    # single run fallback
    if not varying_keys:
        print("No varying parameters found in grid. Running a single synthesis...")
        single_params = {k: v[0] for k, v in normalized_grid.items()}
        merged_params = {**base_params, **single_params}
        
        # strip compound parameters if any
        if "topic_imbalance" in merged_params:
            imbal = merged_params["topic_imbalance"]
            if merged_params.get("generation_mode") == "stm":
                merged_params["topic_proportions"] = imbal.get("topic_proportions")
            else:
                t_probs = np.array(imbal.get("markov_transition_probs"))
                merged_params["markov_matrix"] = make_uncorrelated_markov(t_probs)
            del merged_params["topic_imbalance"]
            
        output_dir = os.path.join(output_base, "single_run")
        generate_single_run(merged_params, output_dir, config_path)
        return

    # group by largest varying parameter, subgroup by other combinations
    varying_keys.sort(key=lambda x: len(normalized_grid[x]), reverse=True)
    grouping_key = varying_keys[0]
    subgroup_keys = sorted([k for k in varying_keys if k != grouping_key])

    print(f"varying dimensions: {varying_keys}")
    print(f"grouping folders by '{grouping_key}', and subfolders by '{subgroup_keys}'")

    save_reproduction_log(output_base, {}, config_path, corpus_info=False)

    # cartesian product of parameters
    grid_keys = list(normalized_grid.keys())
    grid_value_combos = list(product(*(normalized_grid[k] for k in grid_keys)))
    
    total_runs = len(grid_value_combos)
    print(f"combinations to generate: {total_runs}")
    start_time = time.time()
    
    for idx, combo in enumerate(grid_value_combos):
        current_grid_params = dict(zip(grid_keys, combo))
        merged_params = {**base_params, **current_grid_params}
        
        # Handle conditional 'topic_imbalance' compound structure
        mrm = None
        tp = None
        gini_val = 0
        
        if "topic_imbalance" in current_grid_params:
            imbal = current_grid_params["topic_imbalance"]
            gini_val = imbal.get("gini", 0)
            
            if merged_params.get("generation_mode") == "stm":
                tp = imbal.get("topic_proportions")
            else:
                t_probs = np.array(imbal.get("markov_transition_probs"))
                mrm = make_uncorrelated_markov(t_probs)
                
            del merged_params["topic_imbalance"]
            
        merged_params["topic_proportions"] = tp
        merged_params["markov_matrix"] = mrm
        
        # folder name
        if grouping_key == "topic_imbalance":
            group_folder = f"gini_{gini_val}"
        else:
            val = str(current_grid_params[grouping_key]).replace('.', 'p')
            group_folder = f"{grouping_key.replace("_", "")}_{val}"
            
        subgroup_parts = []
        for sk in subgroup_keys:
            if sk == "topic_imbalance":
                subgroup_parts.append(f"gini_{gini_val}")
            else:
                val = current_grid_params[sk]
                subgroup_parts.append(f"{sk.replace("_", "")}_{val}")
                
        subgroup_folder = "_".join(subgroup_parts) if subgroup_parts else "run"
        output_dir = os.path.join(output_base, group_folder, subgroup_folder)

        generate_single_run(merged_params, output_dir, config_path)
        elapsed = int(np.round(time.time() - start_time))
        estim_rest = int(np.round(elapsed / (idx + 1) * (total_runs - idx - 1)))
        print(f"[{idx+1}/{total_runs}] Synthesizing to: {group_folder}/{subgroup_folder}...\ttime passed: {elapsed}, estimated remaining time: {estim_rest}")

def generate_single_run(params, output_dir, config_path):
    corpus = SyntheticCorpus(
        n_docs=params.get("n_docs"),
        num_topics=params.get("num_topics"),
        vocab_size_per_topic=params.get("vocab_size_per_topic"),
        text_len_dist=params.get("text_len_dist", np.random.poisson),
        text_len_params=params.get("text_len_params"),
        generation_mode=params.get("generation_mode", "stm"),
        n_groups_prev=params.get("n_groups_prev", 2),
        n_groups_cont=params.get("n_groups_cont", 2),
        prev_effect_size=params.get("prev_effect_size", 0.0),
        cont_effect_size=params.get("cont_effect_size", 0.0),
        topic_proportions=params.get("topic_proportions"),
        topic_covar=params.get("topic_covar", 0.0),
        stopword_ratio=params.get("stopword_ratio", 0.0),
        min_word_freq=params.get("min_word_freq", 1),
        topic_signal_boost=params.get("topic_signal_boost", 6.0),
        overlap_ratio=params.get("overlap_ratio", 0.0),
        unstandard_ratio=params.get("unstandard_ratio", 0.0),
        max_variants=params.get("max_variants", 8),
        zipf_s=params.get("zipf_s", 1.1),
        markov_matrix=params.get("markov_matrix"),
        store_documents=params.get("store_documents", False)
    )
    corpus.export_for_r(output_dir + "/")
    save_reproduction_log(output_dir, params, config_path, bash_command=False)

if __name__ == "__main__":
    args = parse_args()
    config_data = load_config(args.config)
    run_config_simulation(config_data, args.output_base, args.config)
    print("\n corpus generation done")
