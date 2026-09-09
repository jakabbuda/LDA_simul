import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from utils import log_pipeline_event, PREFIX_MAP, parse_dir_tag

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SearchK metrics.")
    parser.add_argument("--base_path", type=str, default="simul_data/", help="Base path for simulation results.")
    return parser.parse_args()


args = parse_args()
BASE_PATH = args.base_path
log_pipeline_event(BASE_PATH, "searchK_eval", "start", "Initiating SearchK accuracy evaluation and heatmap rendering.")
df = pd.read_csv(f"{BASE_PATH}/prop_simulation_results.csv")

# Populate / synchronize columns from dir_tag if available
if "dir_tag" in df.columns:
    for idx, row in df.iterrows():
        dir_tag = row["dir_tag"]
        if pd.notna(dir_tag):
            parsed = parse_dir_tag(dir_tag)
            for col, val in parsed.items():
                df.loc[idx, col] = val

# Detect which parameters vary across simulation configurations to group them
potential_vars = [
    "target_gini", "min_word_freq", "generation_mode", 
    "overlap_ratio", "unstandard_ratio", "stopword_ratio", "n_docs",
    "num_topics", "vocab_size_per_topic", "prev_effect_size", "cont_effect_size",
    "topic_covar", "doc_len", "n_groups_prev", "n_groups_cont"
]
varying_vars = [v for v in potential_vars if v in df.columns and df[v].nunique() > 1]
if not varying_vars:
    # Fallback to default if nothing varies
    varying_vars = ["target_gini", "generation_mode", "min_word_freq"]

corpus_cols = list(set(varying_vars + ["K_true"]))
corpus_cols = [c for c in corpus_cols if c in df.columns]

k_cols = [c for c in df.columns if "Elbow_" in c or "Final_Consensus" in c]

# Extract distinct configurations based on varying parameters
df_k = df[corpus_cols + k_cols].drop_duplicates().reset_index(drop=True)

# y-axis of the heatmap ('Corpus_params') based on actual varying variables
label_series = []
for col in varying_vars:
    nice_name = col.replace("_", "").upper()
    label_series.append(f"{nice_name}: " + df_k[col].astype(str))

if label_series:
    corpus_params = label_series[0]
    for s in label_series[1:]:
        corpus_params = corpus_params + " | " + s
    df_k['Corpus_params'] = corpus_params
else:
    df_k['Corpus_params'] = "Standard"

# group similar corpora together
df_k = df_k.sort_values(by=varying_vars)

# metrics accuracy calculation relative to each row's actual true_k
summary_stats = []

for col in k_cols:
    estimates = df_k[col]
    actual_k = df_k['K_true']
    
    mae = np.nanmean(np.abs(estimates - actual_k))
    bias = np.nanmean(estimates - actual_k)
    exact_match = np.nanmean(estimates == actual_k) * 100
    failures = estimates.isna().sum()

    summary_stats.append({
        "metric": col.replace("Elbow_", "").replace("Final_Consensus_K_", "Consensus_"),
        "exact_match_%": round(exact_match, 1),
        "MAE": round(mae, 2),
        "bias (over/under)": round(bias, 2),
        "failed_detects": failures
    })

df_summary = pd.DataFrame(summary_stats).sort_values(by="MAE")
df_summary.to_csv(f"{BASE_PATH}/searchK_metric_accuracy_summary.csv", index=False)

print("\n--- SearchK parameter estimation accuracy summary---")
print(df_summary.to_string(index=False))

# Dynamic divergence heatmap (subtracting row-specific K_true)
heatmap_data = df_k.set_index("Corpus_params")[k_cols]
heatmap_data.columns = [c.replace("Elbow_", "").replace("Final_Consensus_K_", "Consensus_") for c in heatmap_data.columns]

# Subtract the true K corresponding to each row to get deviation
deviation_data = heatmap_data.sub(df_k['K_true'].values, axis=0)

plt.figure(figsize=(14, max(8, len(df_k) * 0.4)))
sns.set_theme(style="white")
# Diverging colormap:
# Centered at 0 (White) -> Perfect Match
# > 0 (Red) -> Overestimated K
# < 0 (Blue) -> Underestimated K
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap
ax = sns.heatmap(
    deviation_data,
    annot=heatmap_data.fillna("F").astype(str).replace(r'\.0$', '', regex=True),  # Show actual estimated values
    fmt="",  
    cmap=cmap,
    center=0,  
    vmin=-5, vmax=5,  
    linewidths=.5,
    cbar_kws={"label": "Deviation from true K"}
)

plt.title("SearchK Metric Estimation Behaviour\n(cell value: Estimated K; color: deviation from true K)",
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("topic selection metric", fontsize=12, labelpad=10)
plt.ylabel("corpus parameters", fontsize=12, labelpad=10)

# Rotate labels for readability
plt.xticks(rotation=45, ha='right')

# Highlight consensus cols
for i, col in enumerate(heatmap_data.columns):
    if "Consensus" in col:
        ax.axvline(i, color='black', lw=2)
        ax.axvline(i + 1, color='black', lw=2)

plt.tight_layout()
plt.savefig(f"{BASE_PATH}/plot_3_SearchK_deviation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
msg = f"Created generic SearchK deviation heatmap at: {BASE_PATH}/plot_3_SearchK_deviation_heatmap.png"
print(msg)
log_pipeline_event(BASE_PATH, "searchK_eval", "success", "Saved searchK_metric_accuracy_summary.csv and plot_3_SearchK_deviation_heatmap.png")
