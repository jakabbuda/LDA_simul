import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_PATH = "simul_data/"
df = pd.read_csv(f"{BASE_PATH}/prop_simulation_results.csv")

corpus_cols = ["prop_gini_coeff", "generation_mode", "min_word_freq", "K_true"]
k_cols = [c for c in df.columns if "Elbow_" in c or "Final_Consensus" in c]

df_k = df[corpus_cols + k_cols].drop_duplicates().reset_index(drop=True)

# y-axis of the heatmap
df_k['Corpus_params'] = (
        "Gini: " + df_k['prop_gini_coeff'].astype(int).astype(str).str.zfill(2) +
        " | " + df_k['generation_mode'].str.upper() +
        " | Freq: " + df_k['min_word_freq'].astype(str)
)

# group similar corpora together
df_k = df_k.sort_values(by=["prop_gini_coeff", "generation_mode", "min_word_freq"])

# metrics
summary_stats = []
true_k = df_k['K_true'].iloc[0]  # TODO: make it more fragile for variabe K as well

for col in k_cols:
    estimates = df_k[col]
    mae = np.nanmean(np.abs(estimates - true_k))
    bias = np.nanmean(estimates - true_k)

    exact_match = np.nanmean(estimates == true_k) * 100

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

print(df_summary.to_string(index=False))

# -divergence heatmap

# just the metric columns for the heatmap
heatmap_data = df_k.set_index("Corpus_params")[k_cols]
# clean column names
heatmap_data.columns = [c.replace("Elbow_", "").replace("Final_Consensus_K_", "Consensus_") for c in
                        heatmap_data.columns]

deviation_data = heatmap_data - true_k

plt.figure(figsize=(12, 10))
sns.set_theme(style="white")
# Diverging colormap:
# Centered at 0 (White/Light Gray) -> Perfect Match
# > 0 (Red) -> Overestimated K
# < 0 (Blue) -> Underestimated K
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap
ax = sns.heatmap(
    deviation_data,
    annot=heatmap_data.fillna("F").astype(str).replace(r'\.0$', '', regex=True),  # Show the actual K value in the cell
    fmt="",  # String formatting
    cmap=cmap,
    center=0,  # Forces 0 (True K) to be the neutral color
    vmin=-5, vmax=5,  # Cap the color scale at +/- 5 topics
    linewidths=.5,
    cbar_kws={"label": f"Deviation from true K ({int(true_k)})"}
)

plt.title(f"SearchK metric behavior accross corpora\n(cell value: Estimated K; color: distance from K={int(true_k)})",
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("topic selection metric", fontsize=12, labelpad=10)
plt.ylabel("corpus parameters", fontsize=12, labelpad=10)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Add a subtle highlight box around the Consensus columns
for i, col in enumerate(heatmap_data.columns):
    if "Consensus" in col:
        ax.axvline(i, color='black', lw=2)
        ax.axvline(i + 1, color='black', lw=2)

plt.tight_layout()
plt.savefig(f"{BASE_PATH}/plot_3_SearchK_deviation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
