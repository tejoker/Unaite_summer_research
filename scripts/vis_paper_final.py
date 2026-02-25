import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set style for "Tech/Dark" look if desired, but sticking to Paper-friendly (White) usually.
# However, user uploaded dark image. Let's do a dark background version for the RCA graph 
# to match the request "make it more look like one for a TS" (which often are dark).

output_dir = "paper_plots"
os.makedirs(output_dir, exist_ok=True)

def plot_rca_timeline():
    """
    Generates a Time-Series style Root Cause Analysis graph.
    Matches the user's uploaded style:
    - Horizontal lines for variables
    - Red arrows for Causal Propagation
    - Yellow lines for Anomaly Duration
    """
    print("Generating RCA Timeline Visualization...")
    
    # Config
    n_vars = 15
    duration = 100
    root_idx = 7  # Middle-ish
    child_1 = 4
    child_2 = 10
    
    # Anomaly Event
    t_start = 30
    t_propagate = 45 # Lagged effect
    t_end = 80
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Draw Horizontal "Series" Lines (White)
    for i in range(n_vars):
        ax.hlines(y=i, xmin=0, xmax=duration, color='white', linewidth=1, alpha=0.5)
        
    # 2. Add "Signal" noise to look like TS (Optional, keeping simple lines for now as per reference image)
    
    # 3. Draw Root Cause Anomaly (Yellow segment)
    # Variable 'root_idx' has anomaly from t_start to t_end
    ax.hlines(y=root_idx, xmin=t_start, xmax=t_end, color='yellow', linewidth=2, label='Anomaly Duration')
    
    # 4. Draw Propagated Anomalies (Yellow segment, lagged)
    ax.hlines(y=child_1, xmin=t_propagate, xmax=t_end, color='yellow', linewidth=2)
    ax.hlines(y=child_2, xmin=t_propagate, xmax=t_end, color='yellow', linewidth=2)
    
    # 5. Draw Causal Arrows (Red)
    # From (t_start, root) to (t_propagate, child)
    # Adjust arrow start/end to be visually clear
    
    # Arrow 1: Root -> Child 1
    ax.annotate('', xy=(t_propagate, child_1), xytext=(t_start, root_idx),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Arrow 2: Root -> Child 2
    ax.annotate('', xy=(t_propagate, child_2), xytext=(t_start, root_idx),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Labels and Grid
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels([f"Sensor_{i}" for i in range(n_vars)], fontsize=8, color='lightgrey')
    ax.set_xlabel("Time (steps)", color='white')
    ax.set_title("Time-Series Causal Propagation (RCA)", color='white', fontsize=14)
    
    # Grid
    ax.grid(True, which='both', color='gray', linestyle='dotted', alpha=0.3)
    
    # Legend manually
    l1 = patches.Patch(color='yellow', label='Anomaly Window')
    l2 = patches.Patch(color='red', label='Causal Propagation')
    ax.legend(handles=[l1, l2], loc='upper right', facecolor='black', edgecolor='white')
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rca_timeline_viz.png", dpi=300)
    print(f"Saved {output_dir}/rca_timeline_viz.png")
    plt.close()

def plot_smd_auc_pr():
    """
    Plots the operational points (Precision, Recall) for all SMD entities.
    Since we optimized for Best F1, these points represent the peak of the PR curve.
    """
    print("Generating SMD Precision-Recall Analysis...")
    
    try:
        df = pd.read_csv('academic_benchmark_results_v2.csv')
    except FileNotFoundError:
        print("Error: academic_benchmark_results_v2.csv not found.")
        return

    # Filter out NaNs
    df = df.dropna(subset=['Best_Precision_PA', 'Best_Recall_PA'])
    
    precision = df['Best_Precision_PA']
    recall = df['Best_Recall_PA']
    f1 = df['Best_F1_PA']
    stability = df['Causal_Stability']  # Color by stability
    
    plt.style.use('default') # Back to white for academic plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter Plot
    sc = ax.scatter(recall, precision, c=stability, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    
    # Add Iso-F1 Curves
    f_scores = np.linspace(0.2, 0.9, num=8)
    for f in f_scores:
        x = np.linspace(0.01, 1, 100)
        y = f * x / (2 * x - f)
        y[y < 0] = np.nan # basic filter
        ax.plot(x, y, color='gray', ls='--', alpha=0.3)
        ax.annotate(f'F1={f:.1f}', xy=(0.95, f * 0.95 / (2 * 0.95 - f)), fontsize=8, color='gray')

    # Highlight Mean Point
    mean_prec = precision.mean()
    mean_rec = recall.mean()
    ax.scatter(mean_rec, mean_prec, color='red', marker='*', s=300, label='SMD Mean Performance', edgecolors='black')
    
    # Labels
    ax.set_xlabel('Recall (Detection Rate)', fontsize=12)
    ax.set_ylabel('Precision (True Alarm Rate)', fontsize=12)
    ax.set_title('SMD Benchmark: Precision-Recall Operational Points', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Causal Stability Score (0-1)')
    
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/smd_pr_scatter.png", dpi=300)
    print(f"Saved {output_dir}/smd_pr_scatter.png")
    plt.close()

def plot_telemanom_placeholder():
    """
    Generates a placeholder/template for Telemanom since full results aren't ready.
    """
    print("Generating Telemanom Placeholder Plot...")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.text(0.5, 0.5, 'Telemanom Benchmark\nPending Execution (Phase 2)', 
            horizontalalignment='center', verticalalignment='center', fontsize=20, color='gray')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/telemanom_pr_placeholder.png", dpi=300)
    print(f"Saved {output_dir}/telemanom_pr_placeholder.png")
    plt.close()

if __name__ == "__main__":
    plot_rca_timeline()
    plot_smd_auc_pr()
    plot_telemanom_placeholder()
    print("All plots generated in 'paper_plots/'")
