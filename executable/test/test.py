import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from math import ceil

# === CONFIG ===
golden_file = r"C:\Users\Home\Documents\REZEL\program_internship_paul_wurth\dynotears_results\weights_20250722_203402.csv"
anomaly_file = r"C:\Users\Home\Documents\REZEL\program_internship_paul_wurth\anomaly\dynotears_results_anomaly\weights_20250728_145417.csv"

# === LOAD FUNCTION ===
def load_weights(path):
    df = pd.read_csv(path)
    # edge id: "lag_i_j"
    df['edge_id'] = df.apply(lambda row: f"{int(row['lag'])}_{int(row['i'])}_{int(row['j'])}", axis=1)
    pivot = df.pivot(index='window_idx', columns='edge_id', values='weight').fillna(0.0)

    # also parse columns to a MultiIndex (lag, i, j) for easy grouping
    def parse_col(c):
        lag_s, i_s, j_s = c.split('_')
        return int(lag_s), int(i_s), int(j_s)

    tuples = [parse_col(c) for c in pivot.columns]
    mi = pd.MultiIndex.from_tuples(tuples, names=['lag', 'i', 'j'])
    pivot.columns = mi
    return pivot

# === LOAD DATA ===
golden_df = load_weights(golden_file)   # index: window_idx, cols: MultiIndex(lag,i,j)
anomaly_df = load_weights(anomaly_file)

# === ALIGN COLUMNS ===
# Align on the union of (lag,i,j) columns
all_cols = golden_df.columns.union(anomaly_df.columns)
golden_df = golden_df.reindex(columns=all_cols, fill_value=0.0)
anomaly_df = anomaly_df.reindex(columns=all_cols, fill_value=0.0)

# === GLOBAL SCALE + GLOBAL FROBENIUS (original behavior) ===
scaler_global = StandardScaler()
golden_scaled_global = scaler_global.fit_transform(golden_df.values)
anomaly_scaled_global = scaler_global.transform(anomaly_df.values)

golden_mean_global = np.mean(golden_scaled_global, axis=0)
frobenius_dist_global = np.linalg.norm(anomaly_scaled_global - golden_mean_global, axis=1)

plt.figure()
plt.plot(anomaly_df.index, frobenius_dist_global)
plt.title("Structural Anomaly Score (GLOBAL Frobenius Norm vs Golden Mean)")
plt.xlabel("Window Index")
plt.ylabel("Anomaly Score")
plt.grid(True)
plt.tight_layout()

# === HELPERS FOR GROUPED PLOTS ===

def _compute_group_score(golden_sub: pd.DataFrame, anomaly_sub: pd.DataFrame) -> np.ndarray:
    """
    Fit a scaler on the GOLDEN subset only, transform both,
    then compute Frobenius distance to golden mean per window (row).
    """
    if golden_sub.shape[1] == 0:
        return None
    scaler = StandardScaler()
    g_scaled = scaler.fit_transform(golden_sub.values)
    a_scaled = scaler.transform(anomaly_sub.values)
    g_mean = np.mean(g_scaled, axis=0)
    # distances per row
    d = np.linalg.norm(a_scaled - g_mean, axis=1)
    return d

def _grid(n, max_cols=3):
    cols = min(max_cols, n)
    rows = ceil(n / cols)
    return rows, cols

def _print_top5(name: str, x_index, scores: np.ndarray):
    # Top 5 values with their window_idx
    if scores is None or len(scores) == 0:
        print(f"[{name}] No data.")
        return
    order = np.argsort(scores)[::-1]
    topk = order[:5]
    print(f"\nTop 5 anomaly scores for {name}:")
    for rank, idx in enumerate(topk, 1):
        print(f"  {rank}. window_idx={x_index[idx]}  score={scores[idx]:.6f}")

def plot_by_lag(golden_df: pd.DataFrame, anomaly_df: pd.DataFrame, lags=None, max_plots=9, max_cols=3):
    """
    One subplot per lag, aggregating ALL (i,j) at that lag.
    """
    available_lags = sorted(set(golden_df.columns.get_level_values('lag')))
    lags = available_lags if lags is None else [l for l in lags if l in available_lags]
    if len(lags) == 0:
        print("No matching lags to plot.")
        return

    lags = lags[:max_plots]
    rows, cols = _grid(len(lags), max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.3*rows), squeeze=False)

    for k, lag in enumerate(lags):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        cols_sel = golden_df.columns[golden_df.columns.get_level_values('lag') == lag]
        g_sub = golden_df.loc[:, cols_sel]
        a_sub = anomaly_df.loc[:, cols_sel]
        scores = _compute_group_score(g_sub, a_sub)
        ax.plot(anomaly_df.index, scores)
        ax.set_title(f"lag={lag} (all i,j) | p={g_sub.shape[1]}")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Anomaly Score")
        ax.grid(True)
        _print_top5(f"lag={lag}", anomaly_df.index, scores)

    # hide unused subplots
    for k in range(len(lags), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    plt.tight_layout()

def plot_by_lag_i(golden_df: pd.DataFrame, anomaly_df: pd.DataFrame, lags=None, sources=None, max_plots=9, max_cols=3):
    """
    One subplot per (lag, i), aggregating over ALL j (the 'for all j' you asked for).
    """
    all_lags = sorted(set(golden_df.columns.get_level_values('lag')))
    all_i = sorted(set(golden_df.columns.get_level_values('i')))
    lags = all_lags if lags is None else [L for L in lags if L in all_lags]
    sources = all_i if sources is None else [I for I in sources if I in all_i]

    pairs = [(L, I) for L in lags for I in sources]
    if len(pairs) == 0:
        print("No (lag, i) pairs to plot.")
        return

    pairs = pairs[:max_plots]
    rows, cols = _grid(len(pairs), max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.3*rows), squeeze=False)

    for k, (L, I) in enumerate(pairs):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        mask = (golden_df.columns.get_level_values('lag') == L) & \
               (golden_df.columns.get_level_values('i') == I)
        cols_sel = golden_df.columns[mask]
        g_sub = golden_df.loc[:, cols_sel]
        a_sub = anomaly_df.loc[:, cols_sel]
        if g_sub.shape[1] == 0:
            ax.axis("off")
            print(f"[lag={L}, i={I}] No edges.")
            continue
        scores = _compute_group_score(g_sub, a_sub)
        ax.plot(anomaly_df.index, scores)
        ax.set_title(f"lag={L}, i={I} (all j) | p={g_sub.shape[1]}")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Anomaly Score")
        ax.grid(True)
        _print_top5(f"lag={L}, i={I}", anomaly_df.index, scores)

    # hide unused subplots
    for k in range(len(pairs), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    plt.tight_layout()

def plot_by_lag_j(golden_df: pd.DataFrame, anomaly_df: pd.DataFrame, lags=None, targets=None, max_plots=9, max_cols=3):
    """
    One subplot per (lag, j), aggregating over ALL i (symmetrical to plot_by_lag_i).
    """
    all_lags = sorted(set(golden_df.columns.get_level_values('lag')))
    all_j = sorted(set(golden_df.columns.get_level_values('j')))
    lags = all_lags if lags is None else [L for L in lags if L in all_lags]
    targets = all_j if targets is None else [J for J in targets if J in all_j]

    pairs = [(L, J) for L in lags for J in targets]
    if len(pairs) == 0:
        print("No (lag, j) pairs to plot.")
        return

    pairs = pairs[:max_plots]
    rows, cols = _grid(len(pairs), max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.3*rows), squeeze=False)

    for k, (L, J) in enumerate(pairs):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        mask = (golden_df.columns.get_level_values('lag') == L) & \
               (golden_df.columns.get_level_values('j') == J)
        cols_sel = golden_df.columns[mask]
        g_sub = golden_df.loc[:, cols_sel]
        a_sub = anomaly_df.loc[:, cols_sel]
        if g_sub.shape[1] == 0:
            ax.axis("off")
            print(f"[lag={L}, j={J}] No edges.")
            continue
        scores = _compute_group_score(g_sub, a_sub)
        ax.plot(anomaly_df.index, scores)
        ax.set_title(f"lag={L}, j={J} (all i) | p={g_sub.shape[1]}")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Anomaly Score")
        ax.grid(True)
        _print_top5(f"lag={L}, j={J}", anomaly_df.index, scores)

    for k in range(len(pairs), rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")
    plt.tight_layout()

# === EXAMPLES ===
# 1) Per-lag subplots for the first few lags (edit as needed)
plot_by_lag(golden_df, anomaly_df, lags=None, max_plots=6, max_cols=3)

# 2) Per-(lag,i) subplots (the "for all j" case). Example: first 6 (lag,i) pairs
#    You can pass specific lags=[1,2] and sources=[0,1,2] to control what’s shown.
plot_by_lag_i(golden_df, anomaly_df, lags=None, sources=None, max_plots=6, max_cols=3)

# 3) Per-(lag,j) subplots (symmetrical). Example: show first 6 pairs.
plot_by_lag_j(golden_df, anomaly_df, lags=None, targets=None, max_plots=6, max_cols=3)

# === SAVE GLOBAL RESULTS (unchanged) ===
results_df = pd.DataFrame({
    "window_idx": anomaly_df.index,
    "anomaly_score": frobenius_dist_global
})
results_df.to_csv("anomaly_scores.csv", index=False)

plt.show()
