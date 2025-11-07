#!/usr/bin/env python3
"""
Validate Linear SVAR Model Assumptions
======================================

This script tests whether the linear SVAR model used by DYNOTEARS
is appropriate for the time series data. It computes:

1. R² (coefficient of determination) - how much variance is explained
2. RMSE (root mean squared error) - prediction accuracy
3. Per-variable metrics - which sensors are well-modeled
4. Residual diagnostics - test for violations of assumptions

INTERPRETATION:
- R² > 0.85: Excellent - linear model is appropriate
- R² > 0.70: Good - linear approximation is reasonable
- R² > 0.50: Moderate - significant nonlinearity present
- R² < 0.50: Poor - linear model inappropriate

Usage:
    python validate_linear_svar.py --data data/Golden/golden_period_dataset.csv
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'final_pipeline'))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class LinearSVARValidator:
    """Validates linear SVAR model assumptions and fit quality"""

    def __init__(self, data_path: str, train_frac: float = 0.8, lag_order: int = 5):
        """
        Initialize validator.

        Args:
            data_path: Path to preprocessed CSV data (already differenced/standardized)
            train_frac: Fraction of data to use for training (rest is test)
            lag_order: Number of lags to use in SVAR model
        """
        self.data_path = Path(data_path)
        self.train_frac = train_frac
        self.lag_order = lag_order

        # Results storage
        self.df = None
        self.X_train = None
        self.X_test = None
        self.W = None
        self.A_list = None
        self.X_pred = None
        self.metrics = {}

        logger.info(f"Initialized validator for: {data_path}")
        logger.info(f"Train fraction: {train_frac}, Lag order: {lag_order}")

    def load_and_split_data(self):
        """Load preprocessed data and split into train/test"""
        logger.info("Loading data...")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path, index_col=0)
        logger.info(f"Loaded {len(self.df)} samples, {self.df.shape[1]} variables")
        logger.info(f"Variables: {self.df.columns.tolist()}")

        # Split into train/test
        n = len(self.df)
        n_train = int(n * self.train_frac)

        self.X_train = self.df.values[:n_train]
        self.X_test = self.df.values[n_train:]

        logger.info(f"Train: {len(self.X_train)} samples")
        logger.info(f"Test: {len(self.X_test)} samples")

        return self

    def learn_model(self, lambda_w: float = 0.1, lambda_a: float = 0.1, max_iter: int = 100):
        """
        Learn SVAR model from training data.

        Args:
            lambda_w: Regularization for contemporaneous weights W
            lambda_a: Regularization for lagged weights A
            max_iter: Maximum optimization iterations
        """
        logger.info("Learning SVAR model from training data...")

        try:
            from final_pipeline.dynotears import from_pandas_dynamic

            # Convert to DataFrame for dynotears
            df_train = pd.DataFrame(self.X_train, columns=self.df.columns)

            # Learn model
            logger.info(f"Running DYNOTEARS (λ_W={lambda_w}, λ_A={lambda_a}, p={self.lag_order}, max_iter={max_iter})")
            sm = from_pandas_dynamic(
                df_train,
                p=self.lag_order,
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                max_iter=max_iter,
                h_tol=1e-8,
                loss_tol=1e-6
            )

            # Extract weight matrices
            self.W = sm.get_weight_matrix(lag=0)  # Contemporaneous
            self.A_list = [sm.get_weight_matrix(lag=k) for k in range(1, self.lag_order + 1)]

            # Report sparsity
            w_edges = np.sum(np.abs(self.W) > 1e-8)
            logger.info(f"Learned W: {w_edges} edges, sparsity {1 - w_edges/(self.W.shape[0]**2):.3f}")

            for i, A in enumerate(self.A_list):
                a_edges = np.sum(np.abs(A) > 1e-8)
                logger.info(f"Learned A[{i+1}]: {a_edges} edges, sparsity {1 - a_edges/(A.shape[0]**2):.3f}")

        except Exception as e:
            logger.error(f"Failed to learn model: {e}")
            logger.error("Make sure dynotears.py is in final_pipeline/ directory")
            raise

        return self

    def predict_test_data(self):
        """Generate predictions on test data using learned model"""
        logger.info("Generating predictions on test data...")

        d = self.W.shape[0]
        p = self.lag_order
        n_test = len(self.X_test)

        # Initialize predictions
        self.X_pred = np.zeros((n_test, d))

        # Compute (I - W)^(-1) for solving contemporaneous layer
        S = np.eye(d) - self.W

        # Check if invertible
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("S = I - W is singular, using pseudo-inverse")
            S_inv = np.linalg.pinv(S)

        # For each test timestep, make one-step-ahead prediction
        for t in range(n_test):
            # Get history: need p previous values
            # For first few predictions, use end of training data
            if t < p:
                history_start = len(self.X_train) - (p - t)
                history = np.vstack([self.X_train[history_start:], self.X_pred[:t]])
            else:
                history = self.X_pred[t-p:t]

            # Compute lagged contribution: h(t) = sum_k A_k @ x(t-k)
            h = np.zeros(d)
            for k in range(p):
                if len(history) > k:
                    h += self.A_list[k] @ history[-(k+1)]

            # Solve contemporaneous layer: x(t) = S^(-1) @ h(t)
            self.X_pred[t] = S_inv @ h

        logger.info("Predictions generated")
        return self

    def compute_metrics(self):
        """Compute validation metrics"""
        logger.info("Computing validation metrics...")

        # Overall metrics
        r2_overall = r2_score(self.X_test, self.X_pred)
        rmse_overall = np.sqrt(mean_squared_error(self.X_test, self.X_pred))
        mae_overall = mean_absolute_error(self.X_test, self.X_pred)

        # Compute residuals
        residuals = self.X_test - self.X_pred

        # Per-variable metrics
        r2_per_var = []
        rmse_per_var = []
        mae_per_var = []

        for i, col in enumerate(self.df.columns):
            r2_i = r2_score(self.X_test[:, i], self.X_pred[:, i])
            rmse_i = np.sqrt(mean_squared_error(self.X_test[:, i], self.X_pred[:, i]))
            mae_i = mean_absolute_error(self.X_test[:, i], self.X_pred[:, i])

            r2_per_var.append(r2_i)
            rmse_per_var.append(rmse_i)
            mae_per_var.append(mae_i)

        # Store metrics
        self.metrics = {
            'r2_overall': r2_overall,
            'rmse_overall': rmse_overall,
            'mae_overall': mae_overall,
            'r2_per_var': r2_per_var,
            'rmse_per_var': rmse_per_var,
            'mae_per_var': mae_per_var,
            'residuals': residuals
        }

        logger.info(f"Overall R² = {r2_overall:.4f}")
        logger.info(f"Overall RMSE = {rmse_overall:.4f}")
        logger.info(f"Overall MAE = {mae_overall:.4f}")

        return self

    def test_residuals(self):
        """Test residual assumptions (normality, independence, homoscedasticity)"""
        logger.info("Testing residual assumptions...")

        residuals = self.metrics['residuals']

        # Flatten residuals for overall tests
        res_flat = residuals.flatten()

        # Test 1: Normality (Shapiro-Wilk test)
        # Sample if too many points (Shapiro-Wilk limited to 5000)
        if len(res_flat) > 5000:
            res_sample = np.random.choice(res_flat, 5000, replace=False)
        else:
            res_sample = res_flat

        shapiro_stat, shapiro_p = stats.shapiro(res_sample)

        # Test 2: Zero mean
        mean_test_stat, mean_test_p = stats.ttest_1samp(res_flat, 0)

        # Test 3: Autocorrelation (Ljung-Box approximation via Durbin-Watson)
        from scipy.stats import norm
        dw_stats = []
        for i in range(residuals.shape[1]):
            res_i = residuals[:, i]
            diff = np.diff(res_i)
            dw = np.sum(diff**2) / np.sum(res_i**2)
            dw_stats.append(dw)
        dw_mean = np.mean(dw_stats)

        self.metrics['residual_tests'] = {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'normal': shapiro_p > 0.05,
            'mean_test_p': mean_test_p,
            'zero_mean': abs(np.mean(res_flat)) < 0.1,
            'durbin_watson': dw_mean,
            'no_autocorr': 1.5 < dw_mean < 2.5
        }

        logger.info(f"Normality test: p = {shapiro_p:.4f} ({'PASS' if shapiro_p > 0.05 else 'FAIL'})")
        logger.info(f"Mean test: μ = {np.mean(res_flat):.4f} ({'PASS' if abs(np.mean(res_flat)) < 0.1 else 'FAIL'})")
        logger.info(f"Durbin-Watson: {dw_mean:.4f} ({'PASS' if 1.5 < dw_mean < 2.5 else 'FAIL'})")

        return self

    def generate_report(self, output_dir: str = 'validation_results'):
        """Generate comprehensive validation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        logger.info(f"Generating validation report in {output_path}")

        # Save metrics to JSON
        import json
        metrics_dict = {
            'r2_overall': float(self.metrics['r2_overall']),
            'rmse_overall': float(self.metrics['rmse_overall']),
            'mae_overall': float(self.metrics['mae_overall']),
            'r2_per_variable': {col: float(r2) for col, r2 in zip(self.df.columns, self.metrics['r2_per_var'])},
            'rmse_per_variable': {col: float(rmse) for col, rmse in zip(self.df.columns, self.metrics['rmse_per_var'])},
            'residual_tests': {k: float(v) if isinstance(v, (int, float, np.number)) else bool(v)
                             for k, v in self.metrics['residual_tests'].items()}
        }

        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        # Generate text report
        self._write_text_report(output_path / 'validation_report.txt')

        # Generate plots
        self._create_plots(output_path)

        logger.info(f"Report saved to {output_path}")

        return self

    def _write_text_report(self, filepath: Path):
        """Write detailed text report"""
        r2 = self.metrics['r2_overall']

        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LINEAR SVAR MODEL VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Data: {self.data_path}\n")
            f.write(f"Train samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Variables: {self.df.shape[1]}\n")
            f.write(f"Lag order: {self.lag_order}\n\n")

            f.write("-"*80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"R² (coefficient of determination): {r2:.4f}\n")
            f.write(f"RMSE (root mean squared error): {self.metrics['rmse_overall']:.4f}\n")
            f.write(f"MAE (mean absolute error): {self.metrics['mae_overall']:.4f}\n\n")

            f.write("-"*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-"*80 + "\n")

            if r2 > 0.85:
                verdict = "✓ EXCELLENT"
                explanation = "Linear SVAR model explains >85% of variance"
                recommendation = "Linear model is highly appropriate for this data"
                action = "No need for polynomial features or nonlinear methods"
            elif r2 > 0.70:
                verdict = "✓ GOOD"
                explanation = "Linear SVAR model explains >70% of variance"
                recommendation = "Linear approximation is reasonable"
                action = "Current theoretical guarantees likely hold with minor deviations"
            elif r2 > 0.50:
                verdict = "⚠ MODERATE"
                explanation = "Linear SVAR model explains 50-70% of variance"
                recommendation = "Significant nonlinearity is present"
                action = "Consider NOTEARS-MLP (neural network) instead of polynomial features"
            else:
                verdict = "✗ POOR"
                explanation = "Linear SVAR model explains <50% of variance"
                recommendation = "Linear SVAR is inappropriate for this data"
                action = "Need nonlinear methods: Neural ODEs, GP-VAR, or NOTEARS-MLP"

            f.write(f"Verdict: {verdict}\n")
            f.write(f"  {explanation}\n")
            f.write(f"  → {recommendation}\n")
            f.write(f"  → {action}\n\n")

            f.write("-"*80 + "\n")
            f.write("PER-VARIABLE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Variable':<40} {'R²':>8} {'RMSE':>8} {'MAE':>8}\n")
            f.write("-"*80 + "\n")

            for i, col in enumerate(self.df.columns):
                f.write(f"{col:<40} {self.metrics['r2_per_var'][i]:>8.4f} "
                       f"{self.metrics['rmse_per_var'][i]:>8.4f} "
                       f"{self.metrics['mae_per_var'][i]:>8.4f}\n")

            f.write("\n")
            f.write("-"*80 + "\n")
            f.write("RESIDUAL DIAGNOSTICS\n")
            f.write("-"*80 + "\n")

            rt = self.metrics['residual_tests']
            f.write(f"Normality (Shapiro-Wilk): p = {rt['shapiro_p']:.4f} ")
            f.write(f"[{'PASS' if rt['normal'] else 'FAIL'}]\n")
            f.write(f"  → {'Residuals are approximately normal' if rt['normal'] else 'Residuals are non-normal (may indicate model misspecification)'}\n\n")

            f.write(f"Zero mean: μ = {np.mean(self.metrics['residuals']):.4f} ")
            f.write(f"[{'PASS' if rt['zero_mean'] else 'FAIL'}]\n")
            f.write(f"  → {'No systematic bias' if rt['zero_mean'] else 'Systematic prediction bias detected'}\n\n")

            f.write(f"Independence (Durbin-Watson): {rt['durbin_watson']:.4f} ")
            f.write(f"[{'PASS' if rt['no_autocorr'] else 'FAIL'}]\n")
            f.write(f"  → {'Residuals are independent' if rt['no_autocorr'] else 'Autocorrelation in residuals (model may be underspecified)'}\n\n")

            f.write("="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n")

            if r2 > 0.70 and rt['normal'] and rt['zero_mean']:
                f.write("The linear SVAR model is appropriate for this data.\n")
                f.write("All assumptions are reasonably satisfied.\n")
                f.write("Proceed with current RW-CGE framework and theoretical guarantees.\n")
            elif r2 > 0.60:
                f.write("The linear SVAR model is acceptable but not ideal.\n")
                f.write("Some nonlinearity is present but model captures main effects.\n")
                f.write("Acknowledge approximation in paper, proceed with caution.\n")
            else:
                f.write("The linear SVAR model is NOT appropriate for this data.\n")
                f.write("Significant model misspecification detected.\n")
                f.write("MUST use nonlinear methods or revise theoretical claims.\n")

        logger.info(f"Text report written to {filepath}")

    def _create_plots(self, output_dir: Path):
        """Create validation plots"""
        logger.info("Creating validation plots...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Predicted vs Actual (scatter)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, col in enumerate(self.df.columns):
            if i >= len(axes):
                break

            ax = axes[i]
            ax.scatter(self.X_test[:, i], self.X_pred[:, i], alpha=0.5, s=10)

            # Add diagonal line (perfect prediction)
            min_val = min(self.X_test[:, i].min(), self.X_pred[:, i].min())
            max_val = max(self.X_test[:, i].max(), self.X_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')

            r2 = self.metrics['r2_per_var'][i]
            ax.set_title(f"{col}\nR² = {r2:.3f}", fontsize=10)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=150)
        plt.close()

        # Plot 2: Time series comparison
        fig, axes = plt.subplots(self.df.shape[1], 1, figsize=(15, 3*self.df.shape[1]))
        if self.df.shape[1] == 1:
            axes = [axes]

        # Only plot first 200 points for clarity
        n_plot = min(200, len(self.X_test))

        for i, col in enumerate(self.df.columns):
            axes[i].plot(range(n_plot), self.X_test[:n_plot, i], label='Actual', alpha=0.7, lw=2)
            axes[i].plot(range(n_plot), self.X_pred[:n_plot, i], label='Predicted', alpha=0.7, lw=2)
            axes[i].set_title(f"{col} - R² = {self.metrics['r2_per_var'][i]:.3f}")
            axes[i].set_xlabel('Time step')
            axes[i].set_ylabel('Value (standardized)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'time_series_comparison.png', dpi=150)
        plt.close()

        # Plot 3: Residual analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        residuals_flat = self.metrics['residuals'].flatten()

        # Histogram
        axes[0, 0].hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Residual Distribution')
        axes[0, 0].set_xlabel('Residual')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='r', linestyle='--', lw=2, label='Zero')
        axes[0, 0].legend()

        # Q-Q plot
        stats.probplot(residuals_flat, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Test)')
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals vs Predicted
        axes[1, 0].scatter(self.X_pred.flatten(), residuals_flat, alpha=0.3, s=5)
        axes[1, 0].axhline(0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_title('Residuals vs Predicted')
        axes[1, 0].set_xlabel('Predicted Value')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)

        # Residuals over time
        axes[1, 1].plot(residuals_flat[:1000], alpha=0.5)  # Plot first 1000
        axes[1, 1].axhline(0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_title('Residuals Over Time (first 1000)')
        axes[1, 1].set_xlabel('Time step')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'residual_analysis.png', dpi=150)
        plt.close()

        # Plot 4: R² bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        variables = [col[:30] for col in self.df.columns]  # Truncate long names
        r2_values = self.metrics['r2_per_var']

        bars = ax.barh(variables, r2_values, color=['green' if r2 > 0.8 else 'orange' if r2 > 0.6 else 'red' for r2 in r2_values])
        ax.axvline(0.7, color='black', linestyle='--', lw=2, label='Good threshold (0.7)')
        ax.axvline(0.85, color='green', linestyle='--', lw=2, label='Excellent threshold (0.85)')
        ax.set_xlabel('R² Score')
        ax.set_title('Model Fit Quality by Variable')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Add values on bars
        for i, (bar, r2) in enumerate(zip(bars, r2_values)):
            ax.text(r2 + 0.02, bar.get_y() + bar.get_height()/2, f'{r2:.3f}',
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'r2_by_variable.png', dpi=150)
        plt.close()

        logger.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Validate linear SVAR model assumptions')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed CSV data')
    parser.add_argument('--train-frac', type=float, default=0.8, help='Fraction for training (default: 0.8)')
    parser.add_argument('--lag-order', type=int, default=5, help='Lag order (default: 5)')
    parser.add_argument('--lambda-w', type=float, default=0.1, help='Regularization for W (default: 0.1)')
    parser.add_argument('--lambda-a', type=float, default=0.1, help='Regularization for A (default: 0.1)')
    parser.add_argument('--max-iter', type=int, default=100, help='Max iterations (default: 100)')
    parser.add_argument('--output', type=str, default='validation_results', help='Output directory')

    args = parser.parse_args()

    try:
        # Run validation pipeline
        validator = LinearSVARValidator(
            data_path=args.data,
            train_frac=args.train_frac,
            lag_order=args.lag_order
        )

        validator.load_and_split_data()
        validator.learn_model(lambda_w=args.lambda_w, lambda_a=args.lambda_a, max_iter=args.max_iter)
        validator.predict_test_data()
        validator.compute_metrics()
        validator.test_residuals()
        validator.generate_report(output_dir=args.output)

        # Print summary
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"Overall R² = {validator.metrics['r2_overall']:.4f}")
        print(f"Overall RMSE = {validator.metrics['rmse_overall']:.4f}")
        print(f"\nDetailed report: {args.output}/validation_report.txt")
        print(f"Plots: {args.output}/*.png")
        print("="*80)

        # Return exit code based on result
        if validator.metrics['r2_overall'] < 0.5:
            print("\n⚠ WARNING: Linear model is poor (R² < 0.5)")
            print("Consider using nonlinear methods instead")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
