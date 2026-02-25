
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def main():
    # Load data
    data_path = 'results/bagging_experiment_download/telemanom_curve.npz'
    try:
        data = np.load(data_path)
        precision = data['precision']
        recall = data['recall']
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please ensure you have downloaded the results.")
        return

    # Calculate AUC
    auc_val = auc(recall, precision)
    
    # Calculate Max F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    max_f1 = np.max(f1_scores)
    max_f1_idx = np.argmax(f1_scores)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_val:.4f})')
    
    # Mark Max F1
    plt.scatter(recall[max_f1_idx], precision[max_f1_idx], color='red', s=100, zorder=5, label=f'Best F1 = {max_f1:.4f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Telemanom Anomaly Detection')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = 'telemanom_pr_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
