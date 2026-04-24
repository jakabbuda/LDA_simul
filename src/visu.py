import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = "simul_data/"
df = pd.read_csv(f"{BASE_PATH}/prop_simulation_results.csv")
df['model'] = df['model'].str.upper()

model_palette = {
    "LDA": "#1f77b4",
    "CTM": "#ff7f0e",
    "STM": "#2ca02c",
    "LSI": "#d62728"
}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

######################
# RBO - topic quality
#####################
g_rbo = sns.relplot(
    data=df,
    x="prop_gini_coeff",
    y="mean_RBO",
    hue="model",
    style="model",
    col="min_word_freq",
    row="generation_mode",
    kind="line",
    palette=model_palette,
    markers=True,
    dashes=False,
    errorbar="sd",
    height=4,
    aspect=1.2
)

g_rbo.set_axis_labels("topic proportion imbalane (Gini * 100)", "mean rank biased overlap (RBO)")
g_rbo.set_titles(row_template="generation: {row_name}", col_template="min word freq: {col_name}")
g_rbo.fig.subplots_adjust(top=0.9)
g_rbo.fig.suptitle("topic quality", fontsize=16, fontweight='bold')

# Save Plot 1
plt.savefig(f"{BASE_PATH}/prop_simulation_results_RBO.png", dpi=300, bbox_inches='tight')
plt.close()

####################################
# Proportional accuracy (theta RMSE)
#
# LSI  automatically excluded because its RMSE is NaN
g_rmse = sns.relplot(
    data=df,
    x="prop_gini_coeff",
    y="RMSE_theta",
    hue="model",
    style="model",
    col="min_word_freq",
    row="generation_mode",
    kind="line",
    palette=model_palette,
    markers=True,
    dashes=False,
    errorbar="sd",
    height=4,
    aspect=1.2
)

g_rmse.set_axis_labels("topic proportion imbalane (Gini * 100)", "RMSE of document proportions")
g_rmse.set_titles(row_template="generation: {row_name}", col_template="min word freq: {col_name}")
g_rmse.fig.subplots_adjust(top=0.9)
g_rmse.fig.suptitle("proportional accuracy: topic distribution fit quality", fontsize=16, fontweight='bold')

plt.savefig(f"{BASE_PATH}/prop_simulation_results_RMSE.png", dpi=300, bbox_inches='tight')
plt.close()