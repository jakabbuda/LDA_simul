#!/usr/bin/env python3
"""
Unified pipeline orchestrator for bag-of-words topic model behavior simulations.
Allows running the entire end-to-end pipeline with a single command, or running individual steps.
underlying scripts: generate_corpus.py, topic_model.R, evaluator.py, searchK_eval.py, visu.py
"""

import argparse
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Unified pipeline for topic model simulation")
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "generate", "fit", "evaluate", "visualize"],
        default="all",
        help="Pipeline step to execute. Default is 'all' to run end-to-end.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration for corpus generation.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Base path where simulated corpora, models, and results are stored.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to use",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=4,
        help="Number of stochastic model fit runs (LDA, CTM) in topic_model.R",
    )
    parser.add_argument(
        "--searchk_range",
        type=int,
        default=5,
        help="SearchK evaluation range (+/- K_true) in topic_model.R.",
    )
    parser.add_argument(
        "--search_k_permodelfit",
        action="store_true",
        help="Run SearchK topic number selection per stochastic run rather than once per corpus.",
    )
    return parser.parse_args()


def run_command(cmd, step_name):
    print(f"\n{'=' * 60}")
    print(f"  STEP: {step_name}")
    print(f"  COMMAND: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    start_time = time.time()
    res = subprocess.run(cmd)
    elapsed = round(time.time() - start_time, 1)
    if res.returncode != 0:
        print(f"\n Step '{step_name}' failed with exit code {res.returncode} after {elapsed}s.")
        sys.exit(res.returncode)
    print(f"\n Step '{step_name}' completed successfully in {elapsed}s.")


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    # Corpus generation
    if args.step in ("all", "generate"):
        cmd = [
            python_bin,
            os.path.join(script_dir, "generate_corpus.py"),
            "--config", args.config,
            "--output_base", args.base_path,
            "--cores", str(args.cores),
        ]
        run_command(cmd, "1/4 Generate Synthetic Corpora")

    # Fitting topic models
    if args.step in ("all", "fit"):
        cmd = [
            "Rscript",
            os.path.join(script_dir, "topic_model.R"),
            "--base_path", args.base_path,
            "--n_runs", str(args.n_runs),
            "--cores", str(args.cores),
            "--searchk_range", str(args.searchk_range),
        ]
        if args.search_k_permodelfit:
            cmd.extend(["--search_k_permodelfit", "TRUE"])
        run_command(cmd, "2/4 Fit Topic Models (R)")

    # Evaluation
    if args.step in ("all", "evaluate"):
        cmd = [
            python_bin,
            os.path.join(script_dir, "evaluator.py"),
            "--base_path", args.base_path,
        ]
        run_command(cmd, "3/4 Evaluate Topic Fits")

    # Visualisation
    if args.step in ("all", "visualize"):
        cmd_searchk = [
            python_bin,
            os.path.join(script_dir, "searchK_eval.py"),
            "--base_path", args.base_path,
        ]
        run_command(cmd_searchk, "4a/4 Evaluate SearchK Metrics")

        cmd_visu = [
            python_bin,
            os.path.join(script_dir, "visu.py"),
            "--base_path", args.base_path,
        ]
        run_command(cmd_visu, "4b/4 Visualize Simulation Results")

    print("\n Pipeline execution finished successfully!")


if __name__ == "__main__":
    main()
