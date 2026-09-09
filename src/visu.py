import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from utils import log_pipeline_event, PREFIX_MAP, parse_dir_tag


X_LABEL_MAPPING = {
    "target_gini": "Target topic proportion imbalance (Gini * 100)",
    "prop_gini_coeff": "Observed topic proportion imbalance (Gini * 100)",
    "min_word_freq": "minimum word frequency threshold",
    "generation_mode": "Corpus generation logic",
    "overlap_ratio": "Vocabulary sharing ration",
    "unstandard_ratio": "Unstandardized form ratio",
    "stopword_ratio": "stopwords frequency",
    "n_docs": "number of documents",
    "num_topics": "Number of topics",
    "vocab_size_per_topic": "Vocabulary size per topic",
    "prev_effect_size": "Prevalence effect size",
    "cont_effect_size": "Content effect size",
    "topic_covar": "Topic correlation",
    "doc_len": "Mean document length",
    "n_groups_prev": "Number of prevalence covariate categories",
    "n_groups_cont": "Number of content covariate categories"
}

def parse_args():
    parser = argparse.ArgumentParser(description="plot simulation evaluation results")
    parser.add_argument("--base_path", type=str, default="simul_test01", help="directory where simulation results are located")
    parser.add_argument("--x_var", type=str, default=None, help="Parameter for the x-axis. Defaults to the parameter with the most unique values.")
    parser.add_argument("--col_var", type=str, default=None, help="Parameter to group by columns. ('none' or 'None' to disable)")
    parser.add_argument("--row_var", type=str, default=None, help="Parameter to group by rows. ('none' or 'None' to disable)")
    parser.add_argument("--no_group", action="store_true", help="Disable automatic subplot grouping (force a single plot).")
    parser.add_argument("--filter", type=str, default=None, help="Comma-separated key=value filters to subset the corpus (e.g. 'unstandard_ratio=0.1,min_word_freq=5').")
    parser.add_argument("--file_suffix", type=str, default="", help="Suffix to append to each file name.")
    return parser.parse_args()

def normalize_var_name(v):
    if v is None:
        return None
    clean = str(v).replace("_", "").lower()
    return PREFIX_MAP.get(clean, v)

def get_column_from_folder_part(part):
    """
    Infers the DataFrame column name matching a given subdirectory string prefix.
    """
    clean = part.replace("_", "").lower()
    for prefix, col in sorted(PREFIX_MAP.items(), key=lambda x: -len(x[0])):
        if clean.startswith(prefix):
            return col
    return None

def detect_params_from_folders(df):
    """
    Examines df['dir_tag'] paths to identify the active independent variables (grid dimensions)
    represented in the nested folder directory hierarchy.
    """
    if "dir_tag" not in df.columns:
        return []
        
    sample_tags = df["dir_tag"].dropna().unique()
    if len(sample_tags) == 0:
        return []
        
    detected_vals = {}
    for tag in sample_tags:
        p = parse_dir_tag(tag)
        for k, v in p.items():
            detected_vals.setdefault(k, set()).add(v)
            
    # Parameters that actually vary across directories
    varying_in_folders = [k for k, vals in detected_vals.items() if len(vals) > 1]
    return varying_in_folders

def reconstruct_params_from_dir_tag(df):
    """
    Parses 'dir_tag' paths in the DataFrame and dynamically synchronizes
    independent variables to match directory hierarchy values.
    """
    if "dir_tag" not in df.columns:
        return df
        
    for idx, row in df.iterrows():
        dir_tag = row["dir_tag"]
        if pd.isna(dir_tag):
            continue
        parsed = parse_dir_tag(dir_tag)
        for col, val in parsed.items():
            df.loc[idx, col] = val
                            
    return df

def determine_plotting_vars(df, x_var=None, col_var=None, row_var=None, 
                            user_set_col=False, user_set_row=False, force_no_group=False):
    """
    Determines x_var, col_var, and row_var by combining folder directory structure
    and user-provided manual CLI overrides.
    """
    folder_vars = detect_params_from_folders(df)
    
    if not folder_vars:
        # Fallback to general varying columns in df if no folder structure is active
        potential_vars = [
            "target_gini", "min_word_freq", "generation_mode", 
            "overlap_ratio", "unstandard_ratio", "stopword_ratio", "n_docs",
            "num_topics", "vocab_size_per_topic", "prev_effect_size", "cont_effect_size",
            "topic_covar"
        ]
        folder_vars = [v for v in potential_vars if v in df.columns and df[v].nunique() > 1]
        
    print(f"Detected experimental grid parameters from directory structure: {folder_vars}")
    
    # Exclude variables explicitly specified by user from auto-detection candidacy
    chosen_vars = {x_var, col_var, row_var} - {None}
    available_folders = [v for v in folder_vars if v not in chosen_vars]
    
    # x_var: available parameter with the most unique values
    if x_var is None:
        if len(available_folders) > 0:
            sorted_avail = sorted(available_folders, key=lambda v: df[v].nunique() if v in df.columns else 0, reverse=True)
            x_var = sorted_avail[0]
            available_folders.remove(x_var)
        else:
            x_var = "run" if "run" in df.columns else df.columns[0]
            
    # col_var and row_var (for subplots)
    if force_no_group:
        col_var = None
        row_var = None
    else:
        # If generation_mode is available and row_var not explicitly set, prefer generation_mode for row_var
        if not user_set_row and row_var is None and "generation_mode" in available_folders:
            row_var = "generation_mode"
            available_folders.remove("generation_mode")
            
        if not user_set_col and col_var is None:
            if len(available_folders) > 0:
                col_var = available_folders[0]
                available_folders.remove(col_var)
        if not user_set_row and row_var is None:
            if len(available_folders) > 0:
                row_var = available_folders[0]
                available_folders.remove(row_var)
                
    # Identify extra varying parameters that are not represented in x/row/col axes
    extra_varying = [v for v in folder_vars if v not in (x_var, col_var, row_var) and v is not None]
    
    return x_var, col_var, row_var, extra_varying

def parse_filters(filter_str):
    filters = {}
    if not filter_str:
        return filters
    for item in filter_str.split(','):
        if '=' in item:
            k, v = item.split('=', 1)
            filters[k.strip()] = v.strip()
    return filters

def apply_filters(df, filters):
    for col, val in filters.items():
        if col not in df.columns:
            print(f"Warning: filter column '{col}' not found in the results. Ignoring.")
            continue
            
        col_dtype = df[col].dtype
        try:
            if pd.api.types.is_bool_dtype(col_dtype):
                if isinstance(val, str):
                    val = val.lower() in ('true', '1', 'yes')
                else:
                    val = bool(val)
            elif pd.api.types.is_numeric_dtype(col_dtype):
                if pd.api.types.is_float_dtype(col_dtype):
                    val = float(val)
                else:
                    val = int(val)
            else:
                val = str(val)
        except ValueError:
            pass
            
        print(f"Applying user filter: {col} == {val}")
        df = df[df[col] == val]
    return df

def generate_grid_plot(df, x_var, col_var, row_var, y_var, y_label, title, output_path, model_palette):
    # Keep only records containing valid evaluations for the target metric
    df_plot = df[df[y_var].notna()]
    if len(df_plot) == 0:
        print(f"Warning: No valid data found for metric {y_var}. Skipping plot.")
        return
        
    # Extract unique levels for row and column subplots
    row_vals = sorted(df_plot[row_var].dropna().unique()) if row_var else [None]
    col_vals = sorted(df_plot[col_var].dropna().unique()) if col_var else [None]
    
    nrows = len(row_vals)
    ncols = len(col_vals)
    
    print(f"Creating grid plot for {y_var}: {nrows} rows x {ncols} columns. Grouped by col={col_var}, row={row_var}")
    
    # Initialize multi-subplot figure
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    
    handles, labels = None, None
    
    for r, rv in enumerate(row_vals):
        for c, cv in enumerate(col_vals):
            ax = axes[r, c]
            
            # Filter subset corresponding to current grid cell
            sub_df = df_plot.copy()
            if row_var:
                sub_df = sub_df[sub_df[row_var] == rv]
            if col_var:
                sub_df = sub_df[sub_df[col_var] == cv]
                
            if len(sub_df) == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, color='gray')
                if r == nrows - 1:
                    ax.set_xlabel(X_LABEL_MAPPING.get(x_var, x_var.replace('_', ' ').title()), fontsize=11)
                if c == 0:
                    ax.set_ylabel(y_label, fontsize=11)
                continue
                
            # line curves and error bands for stochastic models
            show_legend = (handles is None)
            sns.lineplot(
                data=sub_df,
                x=x_var,
                y=y_var,
                hue="model",
                style="model",
                palette=model_palette,
                markers=True,
                dashes=False,
                errorbar="sd",
                legend=show_legend,
                ax=ax
            )
            
            # get standard labels for the global legend, remove subplot-local legends
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            
            # labels and subplot titles
            title_parts = []
            if row_var:
                title_parts.append(f"{row_var}={rv}")
            if col_var:
                title_parts.append(f"{col_var}={cv}")
            if title_parts:
                ax.set_title(", ".join(title_parts), fontsize=10, fontweight='semibold')
                
            # x-axis labels on bottom row only
            if r == nrows - 1:
                ax.set_xlabel(X_LABEL_MAPPING.get(x_var, x_var.replace('_', ' ').title()), fontsize=11)
            else:
                ax.set_xlabel("")
                
            # y-axis labels on left column only
            if c == 0:
                ax.set_ylabel(y_label, fontsize=11)
            else:
                ax.set_ylabel("")
                
            ax.grid(True, linestyle="--", alpha=0.6)
            
    # shared figure legend below subplots
    if handles and labels:
        ncol = min(4, len(labels))
        fig.legend(
            handles, labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.05 / nrows), 
            ncol=ncol,
            fontsize=11,
            frameon=True
        )
        
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98 if nrows == 1 else 0.95)
    plt.tight_layout()
    
    # Save image with tight crop to avoid cutting off titles or legends
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")

def main():
    args = parse_args()
    file_suffix = args.file_suffix
    BASE_PATH = args.base_path
    log_pipeline_event(BASE_PATH, "visualization", "start", "Initiating grid-based metric plot generation (RBO and RMSE).")
    
    csv_path = os.path.join(BASE_PATH, "prop_simulation_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Simulation results CSV not found at: {csv_path}")
        log_pipeline_event(BASE_PATH, "visualization", "failure", f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df['model'] = df['model'].str.upper()
    df = reconstruct_params_from_dir_tag(df)
    
    # apply filters
    filters = parse_filters(args.filter)
    if filters:
        df = apply_filters(df, filters)
        if len(df) == 0:
            print("Error: Filtering resulted in an empty DataFrame. No plots to generate.")
            log_pipeline_event(BASE_PATH, "visualization", "failure", "Filtering resulted in empty DataFrame.")
            return

    # normalize and resolve plotting variables
    x_var_arg = normalize_var_name(args.x_var)
    col_var_arg = normalize_var_name(args.col_var)
    row_var_arg = normalize_var_name(args.row_var)

    col_var = None if col_var_arg is None else (None if str(col_var_arg).lower() in ('none', 'null', 'disabled') else col_var_arg)
    row_var = None if row_var_arg is None else (None if str(row_var_arg).lower() in ('none', 'null', 'disabled') else row_var_arg)

    user_set_col = (col_var_arg is not None)
    user_set_row = (row_var_arg is not None)

    x_var, col_var, row_var, extra_varying = determine_plotting_vars(
        df, 
        x_var=x_var_arg, 
        col_var=col_var, 
        row_var=row_var,
        user_set_col=user_set_col,
        user_set_row=user_set_row,
        force_no_group=args.no_group
    )
    
    # handle extra varying parameters to avoid multi-point overlap (first unique value subset)
    if len(extra_varying) > 0:
        print(f"\nWarning: The following parameters are also varying in the data but not mapped to plot axes: {extra_varying}")
        print("To avoid mixing distinct experimental settings on the same curve, automatically subseted the data to their first unique value.")
        print("To control which subset is plotted, use the --filter parameter (e.g. --filter 'unstandard_ratio=0.1').\n")
        for extra_v in extra_varying:
            first_val = df[extra_v].dropna().unique()[0]
            print(f"Auto-subsetting: {extra_v} == {first_val}")
            df = df[df[extra_v] == first_val]
            
    print(f"\nFinal Plotting Grid Configuration:")
    print(f"  - X-axis (varied parameter): {x_var}")
    print(f"  - Grid Columns: {col_var}")
    print(f"  - Grid Rows: {row_var}")
    
    model_palette = {
        "LDA": "#1f77b4",
        "CTM": "#ff7f0e",
        "STM": "#2ca02c",
        "LSI": "#d62728"
    }
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # topic quality plot (RBO)
    rbo_path = os.path.join(BASE_PATH, f"prop_simulation_results_RBO_{file_suffix.replace(' ', '')}.png")
    generate_grid_plot(
        df=df,
        x_var=x_var,
        col_var=col_var,
        row_var=row_var,
        y_var="mean_RBO",
        y_label="Mean Rank Biased Overlap (RBO)",
        title="Topic Quality Evaluation (RBO)",
        output_path=rbo_path,
        model_palette=model_palette
    )
    
    # document topic proportions accuracy plot (RMSE)
    df_rmse = df[df['RMSE_theta'].notna()]
    rmse_path = os.path.join(BASE_PATH, f"prop_simulation_results_RMSE_{file_suffix.replace(' ', '')}.png")
    if len(df_rmse) > 0:
        generate_grid_plot(
            df=df_rmse,
            x_var=x_var,
            col_var=col_var,
            row_var=row_var,
            y_var="RMSE_theta",
            y_label="RMSE of Document Proportions",
            title="Proportional Accuracy: Topic Distribution Fit Quality",
            output_path=rmse_path,
            model_palette=model_palette
        )
        msg = f"Created generic evaluation plots (RBO and RMSE) in: {BASE_PATH}"
        print("\n" + msg)
        log_pipeline_event(BASE_PATH, "visualization", "success", f"Saved prop_simulation_results_RBO_{file_suffix.replace(' ', '')}.png and prop_simulation_results_RMSE_{file_suffix.replace(' ', '')}.png")
    else:
        msg = "Skipping RMSE plots: No valid RMSE_theta entries found (likely only LSI runs)."
        print("\n" + msg)
        log_pipeline_event(BASE_PATH, "visualization", "success", f"Saved prop_simulation_results_RBO_{file_suffix.replace(' ', '')}.png. " + msg)

if __name__ == "__main__":
    main()

