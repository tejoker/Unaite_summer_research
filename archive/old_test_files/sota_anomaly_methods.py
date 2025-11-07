#!/usr/bin/env python3
"""
State-of-the-art anomaly detection methods for comparison with DYNOTEARS
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import anomaly detection libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.iforest import IForest
    from pyod.models.autoencoder import AutoEncoder
    from pyod.models.vae import VAE
    SKLEARN_AVAILABLE = True
    PYOD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some libraries not available: {e}")
    SKLEARN_AVAILABLE = False
    PYOD_AVAILABLE = False

class SOTAAnomalyDetector:
    def __init__(self):
        self.methods = {}
        self.scaler = StandardScaler()
        
    def prepare_data_from_matrices(self, baseline_data, anomaly_data):
        """Prepare data for anomaly detection from weight matrices"""
        # Flatten and concatenate data
        baseline_flat = np.array(baseline_data).flatten()
        anomaly_flat = np.array(anomaly_data).flatten()
        
        # Remove zeros and extreme values for better results
        baseline_clean = baseline_flat[baseline_flat != 0]
        anomaly_clean = anomaly_flat[anomaly_flat != 0]
        
        if len(baseline_clean) == 0 or len(anomaly_clean) == 0:
            return None, None, None
        
        # Combine for scaling
        combined_data = np.concatenate([baseline_clean, anomaly_clean])
        combined_data = combined_data.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(combined_data)
        
        # Split back
        baseline_scaled = scaled_data[:len(baseline_clean)]
        anomaly_scaled = scaled_data[len(baseline_clean):]
        
        return baseline_scaled, anomaly_scaled, combined_data
        
    def prepare_data_from_timeseries(self, baseline_df, anomaly_df):
        """Prepare data for anomaly detection from time series DataFrames"""
        # Extract statistical features from time series
        baseline_features = self._extract_statistical_features(baseline_df)
        anomaly_features = self._extract_statistical_features(anomaly_df)
        
        if baseline_features is None or anomaly_features is None:
            return None, None, None
            
        # Combine for scaling
        combined_features = np.vstack([baseline_features.reshape(1, -1), anomaly_features.reshape(1, -1)])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(combined_features)
        
        # Split back
        baseline_scaled = scaled_features[0].reshape(1, -1)
        anomaly_scaled = scaled_features[1].reshape(1, -1)
        
        return baseline_scaled, anomaly_scaled, combined_features
    
    def _extract_statistical_features(self, df):
        """Extract statistical features from time series DataFrame"""
        try:
            features = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    ts = df[col].dropna().values
                    if len(ts) > 0:
                        features.extend([
                            np.mean(ts), np.std(ts), np.var(ts),
                            np.min(ts), np.max(ts), np.median(ts),
                            np.percentile(ts, 25), np.percentile(ts, 75),
                            np.ptp(ts)  # peak-to-peak
                        ])
                        
                        # Add trend features
                        if len(ts) > 1:
                            diff = np.diff(ts)
                            features.extend([
                                np.mean(diff), np.std(diff),
                                np.sum(diff > 0) / len(diff)
                            ])
                        else:
                            features.extend([0, 0, 0])
            
            return np.array(features) if features else None
        except Exception:
            return None
    
    def load_timeseries(self, filepath):
        """Load time series data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Convert to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            return df.select_dtypes(include=[np.number]).dropna()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def prepare_data(self, baseline_data, anomaly_data):
        """Prepare data - handles both matrices and DataFrames"""
        if isinstance(baseline_data, pd.DataFrame) and isinstance(anomaly_data, pd.DataFrame):
            return self.prepare_data_from_timeseries(baseline_data, anomaly_data)
        else:
            return self.prepare_data_from_matrices(baseline_data, anomaly_data)
        
    def isolation_forest(self, baseline_data, anomaly_data, contamination=0.1):
        """Isolation Forest anomaly detection"""
        try:
            baseline_scaled, anomaly_scaled, combined_data = self.prepare_data(baseline_data, anomaly_data)
            if baseline_scaled is None:
                return {'error': 'Data preparation failed'}
            
            # Fit on baseline data
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_forest.fit(baseline_scaled)
            
            # Predict on anomaly data
            anomaly_predictions = iso_forest.predict(anomaly_scaled)
            anomaly_scores = iso_forest.decision_function(anomaly_scaled)
            baseline_scores = iso_forest.decision_function(baseline_scaled)
            
            results = {
                'method': 'Isolation Forest',
                'anomalies_detected': int(np.sum(anomaly_predictions == -1)),
                'total_samples': len(anomaly_predictions),
                'anomaly_ratio': float(np.sum(anomaly_predictions == -1) / len(anomaly_predictions)),
                'mean_anomaly_score': float(np.mean(anomaly_scores)),
                'mean_baseline_score': float(np.mean(baseline_scores)),
                'score_difference': float(np.mean(baseline_scores) - np.mean(anomaly_scores)),
                'min_anomaly_score': float(np.min(anomaly_scores)),
                'max_anomaly_score': float(np.max(anomaly_scores)),
                'contamination': contamination
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'Isolation Forest'}
    
    def one_class_svm(self, baseline_data, anomaly_data, nu=0.1):
        """One-Class SVM anomaly detection"""
        try:
            baseline_scaled, anomaly_scaled, _ = self.prepare_data(baseline_data, anomaly_data)
            if baseline_scaled is None:
                return {'error': 'Data preparation failed'}
            
            # Fit on baseline data
            oc_svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
            oc_svm.fit(baseline_scaled)
            
            # Predict on anomaly data
            anomaly_predictions = oc_svm.predict(anomaly_scaled)
            anomaly_scores = oc_svm.decision_function(anomaly_scaled)
            baseline_scores = oc_svm.decision_function(baseline_scaled)
            
            results = {
                'method': 'One-Class SVM',
                'anomalies_detected': int(np.sum(anomaly_predictions == -1)),
                'total_samples': len(anomaly_predictions),
                'anomaly_ratio': float(np.sum(anomaly_predictions == -1) / len(anomaly_predictions)),
                'mean_anomaly_score': float(np.mean(anomaly_scores)),
                'mean_baseline_score': float(np.mean(baseline_scores)),
                'score_difference': float(np.mean(baseline_scores) - np.mean(anomaly_scores)),
                'nu': nu
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'One-Class SVM'}
    
    def local_outlier_factor(self, baseline_data, anomaly_data, n_neighbors=20):
        """Local Outlier Factor anomaly detection"""
        try:
            baseline_scaled, anomaly_scaled, combined_data = self.prepare_data(baseline_data, anomaly_data)
            if baseline_scaled is None:
                return {'error': 'Data preparation failed'}
            
            # LOF needs to be fitted on the data it will predict on
            combined_scaled = np.vstack([baseline_scaled, anomaly_scaled])
            
            lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(combined_scaled)-1), 
                                   contamination=0.1, novelty=False)
            predictions = lof.fit_predict(combined_scaled)
            scores = lof.negative_outlier_factor_
            
            # Split predictions back
            baseline_predictions = predictions[:len(baseline_scaled)]
            anomaly_predictions = predictions[len(baseline_scaled):]
            baseline_scores = scores[:len(baseline_scaled)]
            anomaly_scores = scores[len(baseline_scaled):]
            
            results = {
                'method': 'Local Outlier Factor',
                'anomalies_detected': int(np.sum(anomaly_predictions == -1)),
                'total_samples': len(anomaly_predictions),
                'anomaly_ratio': float(np.sum(anomaly_predictions == -1) / len(anomaly_predictions)),
                'mean_anomaly_score': float(np.mean(anomaly_scores)),
                'mean_baseline_score': float(np.mean(baseline_scores)),
                'score_difference': float(np.mean(baseline_scores) - np.mean(anomaly_scores)),
                'n_neighbors': n_neighbors
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'Local Outlier Factor'}
    
    def elliptic_envelope(self, baseline_data, anomaly_data, contamination=0.1):
        """Elliptic Envelope (Robust Covariance) anomaly detection"""
        try:
            baseline_scaled, anomaly_scaled, _ = self.prepare_data(baseline_data, anomaly_data)
            if baseline_scaled is None:
                return {'error': 'Data preparation failed'}
            
            # Need at least 2D data for covariance
            if baseline_scaled.shape[1] < 2:
                # Create a simple 2D feature by adding lag
                baseline_2d = np.column_stack([baseline_scaled[:-1], baseline_scaled[1:]])
                anomaly_2d = np.column_stack([anomaly_scaled[:-1], anomaly_scaled[1:]])
            else:
                baseline_2d = baseline_scaled
                anomaly_2d = anomaly_scaled
            
            ee = EllipticEnvelope(contamination=contamination, random_state=42)
            ee.fit(baseline_2d)
            
            anomaly_predictions = ee.predict(anomaly_2d)
            anomaly_scores = ee.decision_function(anomaly_2d)
            baseline_scores = ee.decision_function(baseline_2d)
            
            results = {
                'method': 'Elliptic Envelope',
                'anomalies_detected': int(np.sum(anomaly_predictions == -1)),
                'total_samples': len(anomaly_predictions),
                'anomaly_ratio': float(np.sum(anomaly_predictions == -1) / len(anomaly_predictions)),
                'mean_anomaly_score': float(np.mean(anomaly_scores)),
                'mean_baseline_score': float(np.mean(baseline_scores)),
                'score_difference': float(np.mean(baseline_scores) - np.mean(anomaly_scores)),
                'contamination': contamination
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'Elliptic Envelope'}
    
    def dbscan_anomaly(self, baseline_data, anomaly_data, eps=0.5, min_samples=5):
        """DBSCAN clustering for anomaly detection"""
        try:
            baseline_scaled, anomaly_scaled, combined_data = self.prepare_data(baseline_data, anomaly_data)
            if baseline_scaled is None:
                return {'error': 'Data preparation failed'}
            
            combined_scaled = np.vstack([baseline_scaled, anomaly_scaled])
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(combined_scaled)
            
            # Outliers have label -1
            baseline_labels = labels[:len(baseline_scaled)]
            anomaly_labels = labels[len(baseline_scaled):]
            
            results = {
                'method': 'DBSCAN',
                'anomalies_detected': int(np.sum(anomaly_labels == -1)),
                'total_samples': len(anomaly_labels),
                'anomaly_ratio': float(np.sum(anomaly_labels == -1) / len(anomaly_labels)),
                'baseline_outliers': int(np.sum(baseline_labels == -1)),
                'total_clusters': int(len(set(labels)) - (1 if -1 in labels else 0)),
                'eps': eps,
                'min_samples': min_samples
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'DBSCAN'}
    
    def statistical_anomaly(self, baseline_data, anomaly_data, threshold=2.0):
        """Statistical anomaly detection using z-scores"""
        try:
            baseline_flat = np.array(baseline_data).flatten()
            anomaly_flat = np.array(anomaly_data).flatten()
            
            baseline_clean = baseline_flat[baseline_flat != 0]
            anomaly_clean = anomaly_flat[anomaly_flat != 0]
            
            if len(baseline_clean) == 0 or len(anomaly_clean) == 0:
                return {'error': 'No non-zero data'}
            
            # Calculate baseline statistics
            baseline_mean = np.mean(baseline_clean)
            baseline_std = np.std(baseline_clean)
            
            if baseline_std == 0:
                return {'error': 'Zero baseline standard deviation'}
            
            # Calculate z-scores for anomaly data
            z_scores = np.abs((anomaly_clean - baseline_mean) / baseline_std)
            anomalies = z_scores > threshold
            
            results = {
                'method': 'Statistical (Z-Score)',
                'anomalies_detected': int(np.sum(anomalies)),
                'total_samples': len(anomaly_clean),
                'anomaly_ratio': float(np.sum(anomalies) / len(anomaly_clean)),
                'mean_z_score': float(np.mean(z_scores)),
                'max_z_score': float(np.max(z_scores)),
                'threshold': threshold,
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(baseline_std)
            }
            return results
        except Exception as e:
            return {'error': str(e), 'method': 'Statistical'}

def load_dynotears_results(filepath):
    """Load DYNOTEARS results from CSV file (W_matrices format)"""
    import pandas as pd
    
    if filepath.endswith('.json'):
        # Legacy JSON format support
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['W_est'], data['A_est']
    elif filepath.endswith('.csv'):
        # New CSV format (W_matrices_*.csv)
        df = pd.read_csv(filepath)
        
        # Get unique window indices to determine matrix dimensions
        windows = df['window_idx'].unique()
        n_windows = len(windows)
        
        # Determine matrix size from coefficient indices
        n_vars = int(np.sqrt(df['coef_idx'].max() + 1))
        
        # Initialize matrices
        W_matrices = np.zeros((n_windows, n_vars, n_vars))
        
        # Fill matrices
        for _, row in df.iterrows():
            window_idx = int(row['window_idx'])
            coef_idx = int(row['coef_idx'])
            value = float(row['value'])
            
            # Convert linear index to 2D coordinates
            i = coef_idx // n_vars
            j = coef_idx % n_vars
            W_matrices[window_idx, i, j] = value
        
        # For compatibility with existing code, return first window and adjacency
        W_est = W_matrices[0]  # First window as baseline
        A_est = (np.abs(W_est) > 1e-6).astype(int)  # Binary adjacency matrix
        
        return W_est, A_est
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    

def main():
    import time
    
    parser = argparse.ArgumentParser(description='State-of-the-art anomaly detection methods')
    parser.add_argument('--baseline', required=True, help='Baseline data (CSV time series or JSON/CSV DYNOTEARS results)')
    parser.add_argument('--anomaly', required=True, help='Anomaly data (CSV time series or JSON/CSV DYNOTEARS results)')
    parser.add_argument('--output', required=True, help='Output file for SOTA methods results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['all', 'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope', 'dbscan', 'statistical'],
                       default=['all'], help='Methods to run')
    parser.add_argument('--mode', choices=['timeseries', 'matrices'], default='auto', 
                       help='Data mode: timeseries for raw CSV data, matrices for DYNOTEARS results')
    
    args = parser.parse_args()
    
    try:
        detector = SOTAAnomalyDetector()
        start_time = time.time()
        
        # Determine data mode
        mode = args.mode
        if mode == 'auto':
            # Auto-detect based on file extension and content
            if args.baseline.endswith('.csv') and args.anomaly.endswith('.csv'):
                try:
                    # Try to load as time series
                    test_df = pd.read_csv(args.baseline, nrows=5)
                    if len(test_df.columns) > 2:  # Likely time series with multiple variables
                        mode = 'timeseries'
                    else:
                        mode = 'matrices'
                except:
                    mode = 'matrices'
            else:
                mode = 'matrices'
        
        print(f"🚀 SOTA Anomaly Detection Benchmark")
        print(f"📊 Mode: {mode}")
        print(f"📁 Baseline: {args.baseline}")
        print(f"📁 Anomaly: {args.anomaly}")
        
        # Load data based on mode
        if mode == 'timeseries':
            print("📈 Loading time series data...")
            baseline_data = detector.load_timeseries(args.baseline)
            anomaly_data = detector.load_timeseries(args.anomaly)
            
            if baseline_data is None or anomaly_data is None:
                raise ValueError("Failed to load time series data")
                
            print(f"Baseline shape: {baseline_data.shape}")
            print(f"Anomaly shape: {anomaly_data.shape}")
            
            results = {
                'mode': 'timeseries',
                'baseline_file': args.baseline,
                'anomaly_file': args.anomaly,
                'baseline_shape': baseline_data.shape,
                'anomaly_shape': anomaly_data.shape,
                'methods_run': [],
                'results': {}
            }
        else:
            print("🔗 Loading DYNOTEARS matrix results...")
            W_baseline, A_baseline = load_dynotears_results(args.baseline)
            W_anomaly, A_anomaly = load_dynotears_results(args.anomaly)
            
            baseline_data = W_baseline
            anomaly_data = W_anomaly
            
            results = {
                'mode': 'matrices',
                'baseline_file': args.baseline,
                'anomaly_file': args.anomaly,
                'methods_run': [],
                'W_matrix_results': {},
                'A_matrix_results': {}
            }
        
        # Determine methods to run
        methods_to_run = args.methods
        if 'all' in methods_to_run:
            methods_to_run = ['isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope', 'dbscan', 'statistical']
        
        results['methods_run'] = methods_to_run
        
        # Run methods based on mode
        print("🔬 Running anomaly detection methods...")
        method_times = {}
        
        for method_name in methods_to_run:
            print(f"  ⚡ Running {method_name}...")
            method_start = time.time()
            
            try:
                if method_name == 'isolation_forest':
                    method_result = detector.isolation_forest(baseline_data, anomaly_data)
                elif method_name == 'one_class_svm':
                    method_result = detector.one_class_svm(baseline_data, anomaly_data)
                elif method_name == 'lof':
                    method_result = detector.local_outlier_factor(baseline_data, anomaly_data)
                elif method_name == 'elliptic_envelope':
                    method_result = detector.elliptic_envelope(baseline_data, anomaly_data)
                elif method_name == 'dbscan':
                    method_result = detector.dbscan_anomaly(baseline_data, anomaly_data)
                elif method_name == 'statistical':
                    method_result = detector.statistical_anomaly(baseline_data, anomaly_data)
                else:
                    method_result = {'error': f'Unknown method: {method_name}'}
                
                method_time = time.time() - method_start
                method_times[method_name] = method_time
                method_result['execution_time_seconds'] = method_time
                
                # Store results based on mode
                if mode == 'timeseries':
                    results['results'][method_name] = method_result
                else:
                    results['W_matrix_results'][method_name] = method_result
                    
                    # Also run on A matrices for matrix mode
                    if 'A_baseline' in locals() and 'A_anomaly' in locals():
                        if method_name == 'isolation_forest':
                            a_result = detector.isolation_forest(A_baseline, A_anomaly)
                        elif method_name == 'one_class_svm':
                            a_result = detector.one_class_svm(A_baseline, A_anomaly)
                        elif method_name == 'lof':
                            a_result = detector.local_outlier_factor(A_baseline, A_anomaly)
                        elif method_name == 'elliptic_envelope':
                            a_result = detector.elliptic_envelope(A_baseline, A_anomaly)
                        elif method_name == 'dbscan':
                            a_result = detector.dbscan_anomaly(A_baseline, A_anomaly)
                        elif method_name == 'statistical':
                            a_result = detector.statistical_anomaly(A_baseline, A_anomaly)
                        
                        results['A_matrix_results'][method_name] = a_result
                        
            except Exception as e:
                method_result = {'error': str(e), 'method': method_name}
                method_time = time.time() - method_start
                method_times[method_name] = method_time
                method_result['execution_time_seconds'] = method_time
                
                if mode == 'timeseries':
                    results['results'][method_name] = method_result
                else:
                    results['W_matrix_results'][method_name] = method_result
        
        # Calculate execution time and summary
        total_time = time.time() - start_time
        results['benchmark_info'] = {
            'total_execution_time_seconds': total_time,
            'method_execution_times': method_times,
            'mode': mode
        }
        
        # Calculate summary statistics
        results['summary'] = {'method_performance': {}}
        
        if mode == 'timeseries':
            for method_name in methods_to_run:
                method_result = results['results'].get(method_name, {})
                if 'error' not in method_result:
                    results['summary']['method_performance'][method_name] = {
                        'anomaly_detected': method_result.get('anomalies_detected', 0) > 0 or method_result.get('anomaly_ratio', 0) > 0.1,
                        'anomaly_ratio': method_result.get('anomaly_ratio', 0),
                        'execution_time': method_times.get(method_name, 0)
                    }
        else:
            for method_name in methods_to_run:
                w_result = results['W_matrix_results'].get(method_name, {})
                a_result = results['A_matrix_results'].get(method_name, {})
                
                if 'error' not in w_result and 'error' not in a_result:
                    avg_anomaly_ratio = (w_result.get('anomaly_ratio', 0) + a_result.get('anomaly_ratio', 0)) / 2
                    results['summary']['method_performance'][method_name] = {
                        'avg_anomaly_ratio': float(avg_anomaly_ratio),
                        'W_anomaly_ratio': w_result.get('anomaly_ratio', 0),
                        'A_anomaly_ratio': a_result.get('anomaly_ratio', 0),
                        'W_anomalies_detected': w_result.get('anomalies_detected', 0),
                        'A_anomalies_detected': a_result.get('anomalies_detected', 0),
                        'execution_time': method_times.get(method_name, 0)
                    }
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n✅ Results saved to: {args.output}")
        
        # Print benchmark summary
        print("\n" + "="*60)
        print("🎯 SOTA ANOMALY DETECTION BENCHMARK SUMMARY")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📊 Data mode: {mode}")
        print(f"🔬 Methods run: {len(method_times)}")
        
        if mode == 'timeseries':
            anomaly_detections = sum(1 for perf in results['summary']['method_performance'].values() 
                                   if perf.get('anomaly_detected', False))
            print(f"🚨 Methods detecting anomaly: {anomaly_detections}/{len(results['summary']['method_performance'])}")
            
            print("\nMethod Performance:")
            for method_name, performance in results['summary']['method_performance'].items():
                status = "✅ ANOMALY" if performance.get('anomaly_detected', False) else "❌ normal"
                print(f"  {method_name:20} | {status:12} | {performance['execution_time']:6.3f}s | ratio: {performance.get('anomaly_ratio', 0):.4f}")
        else:
            print("\nMethod Performance (Matrix Mode):")
            for method_name, performance in results['summary']['method_performance'].items():
                print(f"  {method_name.upper()}:")
                print(f"    Avg anomaly ratio: {performance['avg_anomaly_ratio']:.4f}")
                print(f"    Execution time: {performance['execution_time']:.3f}s")
                print(f"    W matrix anomalies: {performance['W_anomalies_detected']}")
                if 'A_anomalies_detected' in performance:
                    print(f"    A matrix anomalies: {performance['A_anomalies_detected']}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()