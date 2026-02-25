import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "paper_plots"
os.makedirs(output_dir, exist_ok=True)

def generate_synthetic_ts():
    """
    Generates synthetic time series data to illustrate RCA.
    Root cause: Smooth drift + spike.
    Children: Lagged response.
    """
    length = 200
    t = np.arange(length)
    
    # Base signals (noise)
    np.random.seed(42)
    sigs = {f"Sensor_{i}": np.random.normal(0, 0.2, length) for i in range(1, 6)}
    
    # Root Cause (Sensor_1): Anomaly starts at t=50
    # Sigmoid ramp up
    sigs["Sensor_1"][50:150] += np.linspace(0, 3, 100) + np.random.normal(0, 0.2, 100)
    
    # Child 1 (Sensor_3): Responds to Sensor_1 at t=70 (Lag 20)
    # Effect is 0.8 * Root
    sigs["Sensor_3"][70:170] += np.linspace(0, 2.4, 100) + np.random.normal(0, 0.2, 100)
    
    # Child 2 (Sensor_5): Responds to Sensor_1 at t=60 (Lag 10)
    sigs["Sensor_5"][60:160] += np.linspace(0, 2.0, 100) + np.random.normal(0, 0.2, 100)
    
    return pd.DataFrame(sigs), t

def plot_rca_ts_style():
    print("Generating Time-Series RCA Visualization...")
    df, t = generate_synthetic_ts()
    
    # Plot setup: Stacked subplots or Offset lines?
    # User wanted "TS way". Stacked is standard for monitoring.
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 10), sharex=True)
    
    cols = sorted(df.columns)
    # Reorder to put Root (Sensor_1) at top or highlighted
    cols = ["Sensor_1", "Sensor_5", "Sensor_3", "Sensor_2", "Sensor_4"]
    
    for i, col in enumerate(cols):
        ax = axes[i]
        val = df[col]
        
        # Color logic
        color = 'cyan'
        if col == "Sensor_1": color = 'red' # Root
        elif col in ["Sensor_3", "Sensor_5"]: color = 'orange' # Symptoms
        
        ax.plot(t, val, color=color, linewidth=1.5)
        ax.set_ylabel(col, rotation=0, labelpad=30, fontsize=10, color='white')
        ax.grid(True, alpha=0.3, ls=':')
        
        # Highlight anomalous region
        if col == "Sensor_1":
            ax.axvspan(50, 150, color='red', alpha=0.1)
        elif col == "Sensor_5":
            ax.axvspan(60, 160, color='orange', alpha=0.1)
        elif col == "Sensor_3":
            ax.axvspan(70, 170, color='orange', alpha=0.1)

    # Draw Arrows spreading across subplots (Visual annotation)
    # We use fig coords or transform
    # Root(t=50) -> Child1(t=60)
    # Root(t=50) -> Child2(t=70)
    
    # We can't easily draw across axes with simple arrow commands without transforms
    # Instead, we just caption it.
    
    fig.suptitle("Causal RCA: Root Cause Propagation", fontsize=16, color='white')
    axes[-1].set_xlabel("Time", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rca_ts_style.png", dpi=300)
    print(f"Saved {output_dir}/rca_ts_style.png")
    plt.close()


def plot_smd_auc_curve():
    """
    Plots the Global AUC-PR Curve for SMD.
    """
    print("Generating SMD Global AUC-PR Curve...")
    
    CurveFile = "smd_global_curve.npz"
    if not os.path.exists(CurveFile):
        print(f"Error: {CurveFile} not found.")
        return

    data = np.load(CurveFile)
    
    # 1. Standard Curve
    precision = data['precision']
    recall = data['recall']
    auc_std = data['auc_std']
    
    # 2. Point-Adjusted Curve
    pa_precision = data['pa_precision']
    pa_recall = data['pa_recall']
    pa_auc = data['auc_pa']
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Standard
    ax.plot(recall, precision, color='blue', lw=2, linestyle='--', label=f'Standard (AUC = {auc_std:.4f})')
    # ax.fill_between(recall, precision, alpha=0.1, color='blue')
    
    # Plot PA
    ax.plot(pa_recall, pa_precision, color='green', lw=2, label=f'Point-Adjusted (AUC = {pa_auc:.4f})')
    ax.fill_between(pa_recall, pa_precision, alpha=0.1, color='green')

    ax.set_title("SMD Benchmark: Global Precision-Recall Curve", fontsize=14)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.legend(loc='lower left', fontsize=12) # Lower left usually empty for PR curves
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    
    output_path = f"{output_dir}/smd_auc_pr_curve_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")


def plot_smd_scatter_simple():
    """
    Plots the SMD Precision-Recall scatter points without Causal Stability coloring.
    """
    print("Generating SMD Scatter Plot (Simple)...")
    
    df = pd.read_csv('academic_benchmark_results_v2.csv')
    df = df.dropna(subset=['Best_Precision_PA', 'Best_Recall_PA'])
    
    prec = df['Best_Precision_PA']
    rec = df['Best_Recall_PA']
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points (Fixed color instead of Causal Stability)
    ax.scatter(rec, prec, alpha=0.7, s=100, c='steelblue', edgecolors='k', label='Individual Machine')
    
    # Mean
    ax.scatter(rec.mean(), prec.mean(), c='red', s=250, marker='*', edgecolors='k', label='Average Performance')
    
    # Iso-F1 curves
    f_scores = np.linspace(0.2, 0.9, 8)
    for f in f_scores:
        x = np.linspace(0.01, 1)
        y = f * x / (2 * x - f)
        ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, linestyle='--')
        ax.text(0.95, f / (2 - f), f'F1={f:.1f}', fontsize=8, color='gray')

    ax.set_title("SMD Benchmark: Operational Points (Precision vs Recall)", fontsize=14)
    ax.set_xlabel("Recall (Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision (True Alarm Rate)", fontsize=12)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    output_path = f"{output_dir}/smd_scatter_simple.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_rca_ts_style()
    plot_smd_auc_curve() 
    plot_smd_scatter_simple()

