# Bag-of-Words topic model simulation framework to study model behaviour on simulated corpora

Simulation framework for evaluating the behavior, accuracy, and robustness of bag-of-words topic models
(LDA, CTM, STM, and LSI) across synthetic corpora with controlled noise, document lengths, topic imbalances, 
and covariate structures. Background word frequency follows Zipfian distribution for more realistic corpora.

Supports parallel corpus generation and model fitting on multiple CPU cores (`cores`)

---

## Key features

### Dual generative modes (`stm` & `markov`)
- **STM Mode (`generation_mode: "stm"`)**:
  - Simulates document topic proportions $\theta_d$ from a multivariate logistic-normal distribution $\text{softmax}(\eta_d)$, where $\eta_d \sim \mathcal{N}(\mu + \Gamma x_{\text{prev}, d}, \Sigma)$.
  - Supports custom pairwise topic covariance/correlation ($\Sigma$ via `topic_covar`).
  - Implements the generative model described in the article *Structural Topic Models for Open-Ended Survey Responses* (Roberts et al., 2014)
- **Markov Mode (`generation_mode: "markov"`)**:
  - Simulates documents as sequential Markov chains across topics using a transition matrix $M$.
  - Computes exact stationary distributions for initial topic assignment.
  - Supports group-specific transition dynamics under prevalence covariate shifts.

### 2. Noise and imbalance synthesis
- **Topic imbalance**: Custom Dirichlet priors or Markov self-transition vectors parameterized by target Gini coefficient (`topic_imbalance`).
- **Document length variation**: Poisson-distributed word counts sampled with mean parameter $\lambda$ (`text_len_params: {"lam": ...}` or `doc_len`).
- **Prevalence covariates**: Shifts document topic distributions across $G_{\text{prev}}$ groups via `prev_effect_size`, `n_groups_prev`, and optional group weight imbalance `prev_covar_imbal`.
- **Content covariates**: Shifts word emission probabilities $\beta_{k, g_c}$ across $G_{\text{cont}}$ groups via `cont_effect_size`, `n_groups_cont`, and optional `cont_covar_imbal`.
- **Cross-topic vocabulary overlap**: proportion of words shared across multiple topics (`overlap_ratio`).
- **Stemming Noise**: Simulates unstandardized lexical variants to mimic stemming/lemmatization failures (`unstandard_ratio`, `max_variants`).
- **Stopwords**: Ratio of stopwords (topic independent frequent tokens) (`stopword_ratio`, `n_stopwords`, `zipf_s`).
- **Infrequent word pruning**: Simulates vocabulary trimming during preprocessing (`min_word_freq`).

### Topic model fitting (`topic_model.R`)
- **Deterministic vs. stochastic separation**: deterministic methods (LSI, STM with spectral initialization) run once, while stochastic algorithms (LDA, CTM) iterate over `--n_runs`.
- **SearchK elbow & consensus Selection**: Determines optimal $K$ using held-out likelihood, semantic coherence, and residual variance analyzed by Kneedle elbow detection.
- **Resume fuction**: Skips already completed model fits and SearchK logs, allowing interrupted runs to resume instantly.

### Evaluation  and visualization
- **Hungarian algorithm based topic alignment**: Resolves topic label permutation via linear sum assignment on cosine distances between ground-truth and recovered word-topic distributions ($\beta$).
- **Topic model fit quality metrics**:
  - **Rank-biased overlap (RBO)**: Measures rank agreement of top-$N$ words per topic with exponential depth weighting.
  - **RMSE $\theta$**: Root mean square error between predicted document topic distributions and true $\theta$.
  - **STM content covariate handling**: For STM models with content covariates, group-specific $\beta$ matrices are frequency-weighted for alignment.
- **Dynamic grid visualization (`visu.py`)**: Automatically detects varied parameters from directory structures, assigning the parameter with the most levels to the x-axis, `generation_mode` to subplot rows, and the remaining dimension to subplot columns.

---

## Corpus Parameter Reference

Configurations are defined as JSON files with two top-level sections:
- `base_parameters`: Constant baseline settings across the experiment.
- `grid_parameters`: Arrays of values to vary (creating a full Cartesian product of corpora).

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `generation_mode` | `string` | `"stm"` | Generative mode: `"stm"` or `"markov"`. |
| `n_docs` | `int` or `list[int]` | `200` | Number of documents to synthesize. |
| `num_topics` | `int` or `list[int]` | `5` | Ground-truth number of topics $K$. |
| `vocab_size_per_topic` | `int` or `list[int]` | `500` | Base vocabulary size allocated per topic. |
| `text_len_params` / `doc_len` | `dict` or `int` | `{"lam": 100}` | Mean document length for Poisson sampling ($\lambda$). |
| `topic_imbalance` | `dict` | `None` | Imbalance structure containing `gini`, `topic_proportions` (for STM), and `markov_transition_probs` (for Markov). |
| `topic_covar` | `float` or `list[list]` | `0.0` | Pairwise topic covariance in logistic-normal distribution (STM mode). |
| `n_groups_prev` | `int` | `2` | Number of prevalence covariate categories. |
| `prev_effect_size` | `float` | `0.0` | Magnitude of prevalence covariate effect ($\sigma$ of perturbation). |
| `prev_covar_imbal` | `list[float]` | `None` | Class distribution for prevalence groups (defaults to uniform $1/G$). |
| `n_groups_cont` | `int` | `2` | Number of content covariate categories (must be $\ge 2$ for STM). |
| `cont_effect_size` | `float` | `0.0` | Magnitude of content covariate effect ($\sigma$ of lexical perturbation). |
| `cont_covar_imbal` | `list[float]` | `None` | Class distribution for content groups (defaults to uniform $1/G$). |
| `stopword_ratio` | `float` | `0.0` | Proportion of stopword tokens injected into documents. |
| `n_stopwords` | `int` | `50` | Total number of unique stopword types. |
| `overlap_ratio` | `float` | `0.0` | Ratio of topic words shared across multiple topics. |
| `unstandard_ratio` | `float` | `0.0` | Ratio of vocabulary words converted into multi-variant forms. |
| `max_variants` | `int` | `8` | Maximum number of variants per unstandardized word. |
| `min_word_freq` | `int` | `1` | Minimum corpus frequency threshold for vocabulary retention. |
| `topic_signal_boost` | `float` | `6.0` | Topical word emission signal boost ($\kappa_k$). |
| `zipf_s` | `float` | `1.1` | Zipf distribution exponent for background word frequencies. |
| `store_documents` | `bool` | `false` | Whether to write full text tokens to `_corpus.json`. |

---

## JSON configuration examples

### Topic proportion imbalance (Gini variation)

Varies target Gini imbalance (0 to 44) across both STM and Markov modes:

```json
{
    "base_parameters": {
        "num_topics": 5,
        "n_docs": 200,
        "vocab_size_per_topic": 80,
        "min_word_freq": 1,
        "stopword_ratio": 0.0
    },
    "grid_parameters": {
        "generation_mode": ["stm", "markov"],
        "min_word_freq": [1, 10],
        "topic_imbalance": [
            {
                "gini": 0,
                "topic_proportions": [0.2, 0.2, 0.2, 0.2, 0.2],
                "markov_transition_probs": [0.2, 0.2, 0.2, 0.2, 0.2]
            },
            {
                "gini": 19,
                "topic_proportions": [0.08, 0.13, 0.2, 0.26, 0.33],
                "markov_transition_probs": [0.1, 0.15, 0.2, 0.25, 0.3]
            },
            {
                "gini": 44,
                "topic_proportions": [0.03, 0.05, 0.12, 0.25, 0.55],
                "markov_transition_probs": [0.05, 0.08, 0.12, 0.25, 0.5]
            }
        ]
    }
}
```

### mean document length variation

```json
{
    "base_parameters": {
        "num_topics": 5,
        "n_docs": 200,
        "vocab_size_per_topic": 100,
        "generation_mode": "stm",
        "min_word_freq": 1,
        "stopword_ratio": 0.0
    },
    "grid_parameters": {
        "text_len_params": [
            {"lam": 10},
            {"lam": 50},
            {"lam": 250}
        ]
    }
}
```
*(Alternatively, use `"doc_len": [10, 50, 250]` directly).*

### Topic correlation variation

```json
{
    "base_parameters": {
        "num_topics": 5,
        "vocab_size_per_topic": 80,
        "generation_mode": "stm",
        "text_len_params": {"lam": 200},
        "min_word_freq": 1
    },
    "grid_parameters": {
        "n_docs": [50, 200, 800],
        "stopword_ratio": [0.0, 0.05, 0.2],
        "topic_covar": [0.0, 0.2, 0.4, 0.6, 0.8]
    }
}
```

### Prevalence and content covariate effect size variation

```json
{
    "base_parameters": {
        "num_topics": 5,
        "n_docs": 300,
        "vocab_size_per_topic": 100,
        "min_word_freq": 1,
        "stopword_ratio": 0.0
    },
    "grid_parameters": {
        "generation_mode": ["stm", "markov"],
        "prev_effect_size": [0.0, 0.5, 1.5],
        "cont_effect_size": [0.0, 0.5, 1.5],
        "n_groups_prev": [2, 3],
        "n_groups_cont": [2, 3]
    }
}
```

---

## Installation & Prerequisites

**Python 3.10+**; **R 4.3+**

### 1. Python Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

### 2. R Dependencies Setup
Launch R or run via bash:
```R
install.packages(c("stm", "topicmodels", "ldatuning", "lsa", "data.table", "jsonlite", "devtools"))
devtools::install_github("etam4260/kneedle")
```

---

## Step-by-Step Execution Guide

### Step A: Generate Synthetic Corpora
```bash
PYTHONPATH=src python3 src/generate_corpus.py --config path/to/config.json --output_base simul_data/ --cores 4
```
- Creates nested parameter directories (e.g. `simul_data/gini_19/generationmodestm_minwordfreq1/`).
- Writes `_corpus.json`, `reproduction_info.json`, and `reproduce_command.sh` inside each directory.

### Step B: Fit Topic Models (R)
```bash
Rscript src/topic_model.R --base_path simul_data/ --cores 4 --n_runs 4 --searchk_range 4
```
- **`--base_path`**: Base directory containing synthesized corpora.
- **`--cores`**: Parallel worker processes for multi-corpus fitting.
- **`--n_runs`**: Number of stochastic iterations for LDA and CTM.
- **`--searchk_range`**: Topic exploration range around $K_{\text{true}}$ ($[K_{\text{true}} - R, K_{\text{true}} + R]$).
- **`--search_k_permodelfit`**: Optional flag to execute SearchK per stochastic run instead of once globally.

### Step C: Evaluate Topic Fits
```bash
PYTHONPATH=src python3 src/evaluator.py --base_path simul_data/ --top_words 50
```
- Performs Hungarian alignment against true topic word distributions.
- Outputs `prop_simulation_results.csv` in `--base_path` containing RBO, RMSE, and SearchK selections.

### Step D: SearchK & Metric Visualizations
```bash
# 1. SearchK accuracy summary and deviation heatmap
PYTHONPATH=src python3 src/searchK_eval.py --base_path simul_data/

# 2. Performance curve facet plots (RBO and RMSE)
PYTHONPATH=src python3 src/visu.py --base_path simul_data/
```

#### Customizing Plots with `visu.py` Flags:
- `--x_var`: Force a specific parameter onto the x-axis (e.g. `--x_var target_gini` or `--x_var doc_len`).
- `--row_var`: Variable for subplot rows (e.g. `--row_var generation_mode`).
- `--col_var`: Variable for subplot columns (e.g. `--col_var min_word_freq`).
- `--filter`: Filter data subset before plotting (e.g. `--filter "unstandard_ratio=0.0,n_docs=200"`).
- `--no_group`: Disable multi-panel facets and render a single aggregated plot.

---

## Unified Pipeline Orchestrator (`run_pipeline.py`)

Run the complete four-step simulation with a single command:

```bash
python3 src/run_pipeline.py --step all --config path/to/config.json --base_path simul_data/ --cores 4 --n_runs 4
```

Or run individual steps selectively:
```bash
python3 src/run_pipeline.py --step generate  --config path/to/config.json --base_path simul_data/ --cores 4
python3 src/run_pipeline.py --step fit       --base_path simul_data/ --cores 4 --n_runs 4
python3 src/run_pipeline.py --step evaluate  --base_path simul_data/
python3 src/run_pipeline.py --step visualize --base_path simul_data/
```

---

## Multi-Corpus Simulation Pipeline (`run_pipeline_multiple_simul.py`)

To eliminate simulation noise and measure model performance across multiple independent corpus sampling for each grid point, use `run_pipeline_multiple_simul.py`:

```bash
python3 src/run_pipeline_multiple_simul.py \
    --step all \
    --config path/to/config.json \
    --base_path simul_multi/ \
    --n_parallel_corpus 5 \
    --n_runs 1 \
    --cores 4 \
    --keep_corpora false \
    --keep_models false
```

### Key Parameters:
- `--n_parallel_corpus`: Number of independent synthetic corpora generated for each grid point (default: `1`).
- `--n_runs`: Number of stochastic model fits (LDA, CTM) on each corpus (default: `1`).
- `--keep_corpora`: Whether to keep `_corpus.json` files after fitting (default: `True`). Set `--no_keep_corpora` or `--keep_corpora false` to delete the heavy corpus files and minimize disk usage (ground truth parameters are automatically preserved in `_ground_truth.json` for evaluations).
- `--keep_models`: Whether to keep fitted model parameter CSVs after evaluation (default: `True`). Set `--no_keep_models` or `--keep_models false` to delete the heavy matrix CSVs.
- **Resumption**: Automatically checks whether each iteration is already completed, resuming cleanly from interruptions.
- **Output**:
  - Merges evaluation metrics into `base_path/prop_simulation_results.csv`.
  - Produces `searchK_metric_accuracy_summary.csv` and `plot_3_SearchK_deviation_heatmap.png`.
  - Renders `prop_simulation_results_RBO_*.png` and `prop_simulation_results_RMSE_*.png` displaying the empirical mean lines and standard deviation shaded error bands across all parallel corpora.
  - Also outputs individual plots in each `sim_*/` directory for per-iteration inspection.
- If `--n_parallel_corpus 1` is run without multi-corpus settings, execution runs directly in `base_path` identically to `run_pipeline.py`.

---

## Funding:

> <img src="img/ekop-logo-rgb-horizontal_color%20angol.png" alt="EKOP-LOGO" width="150" align=right> SUPPORTED BY THE EKÖP-25 UNIVERSITY EXCELLENCE SCHOLARSHIP PROGRAM OF THE MINISTRY FOR CULTURE AND INNOVATION FROM THE SOURCE OF THE NATIONAL RESEARCH, DEVELOPMENT AND INNOVATION FUND.
