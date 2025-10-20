#!/usr/bin/env python3
"""
anomaly_classification.py - Phase 2: Classification

Implements graph signature extraction and classification approaches:
- Graph signature extraction (15 features)
- Rule-based classifier (interpretable, no training needed)
- ML-based classifier (Random Forest, higher accuracy)

Expected accuracy: 75-82% overall, 90%+ for spikes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class GraphSignatureExtractor:
    """Extract 15 features from weight matrix comparisons for classification."""

    def __init__(self, edge_threshold: float = 0.01):
        """
        Initialize signature extractor.

        Args:
            edge_threshold: Threshold for determining edge existence
        """
        self.edge_threshold = edge_threshold

    def extract_signature(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive 15-feature signature from weight matrix comparison.

        Features:
        - Magnitude features (3): mean/max/std of edge weight changes
        - Structural features (3): edge density, add/remove counts
        - Sign patterns (2): directional relationship flips
        - Sparsity features (2): zero-weight proportions
        - Eigenvalue features (3): spectral properties
        - Node impact features (2): per-variable incoming/outgoing changes

        Args:
            W_baseline: Baseline weight matrix
            W_current: Current weight matrix

        Returns:
            Dict with 15 features for classification
        """
        logger.debug(f"Extracting signature for matrices shape {W_baseline.shape}")

        signature = {}

        # Compute difference matrix
        W_diff = W_current - W_baseline
        W_abs_diff = np.abs(W_diff)

        # 1. MAGNITUDE FEATURES (3 features)
        signature.update(self._extract_magnitude_features(W_diff, W_abs_diff))

        # 2. STRUCTURAL FEATURES (3 features)
        signature.update(self._extract_structural_features(W_baseline, W_current))

        # 3. SIGN PATTERNS (2 features)
        signature.update(self._extract_sign_pattern_features(W_baseline, W_current))

        # 4. SPARSITY FEATURES (2 features)
        signature.update(self._extract_sparsity_features(W_baseline, W_current))

        # 5. EIGENVALUE FEATURES (3 features)
        signature.update(self._extract_eigenvalue_features(W_baseline, W_current))

        # 6. NODE IMPACT FEATURES (2 features)
        signature.update(self._extract_node_impact_features(W_baseline, W_current))

        logger.debug(f"Extracted {len(signature)} features")
        return signature

    def _extract_magnitude_features(self, W_diff: np.ndarray, W_abs_diff: np.ndarray) -> Dict[str, float]:
        """Extract magnitude-based features."""
        return {
            'magnitude_mean_change': float(np.mean(W_abs_diff)),
            'magnitude_max_change': float(np.max(W_abs_diff)),
            'magnitude_std_change': float(np.std(W_abs_diff))
        }

    def _extract_structural_features(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Extract structural topology features."""
        # Convert to adjacency matrices
        adj_baseline = (np.abs(W_baseline) > self.edge_threshold).astype(int)
        adj_current = (np.abs(W_current) > self.edge_threshold).astype(int)

        # Edge densities
        density_baseline = np.sum(adj_baseline) / adj_baseline.size
        density_current = np.sum(adj_current) / adj_current.size

        # Edge changes
        diff_adj = adj_current - adj_baseline
        added_edges = np.sum(diff_adj == 1)
        removed_edges = np.sum(diff_adj == -1)

        density_change = density_current - density_baseline

        return {
            'structural_density_change': float(density_change),
            'structural_density_change_abs': float(abs(density_change)),
            'structural_edges_added': float(added_edges),
            'structural_edges_removed': float(removed_edges)
        }

    def _extract_sign_pattern_features(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Extract sign pattern features (directional relationship flips)."""
        # Sign matrices
        sign_baseline = np.sign(W_baseline)
        sign_current = np.sign(W_current)

        # Count sign flips (where signs are different and both non-zero)
        valid_mask = (W_baseline != 0) & (W_current != 0)
        sign_flips = np.sum((sign_baseline != sign_current) & valid_mask)

        # Count new positive/negative relationships
        new_positive = np.sum((W_baseline == 0) & (W_current > 0))
        new_negative = np.sum((W_baseline == 0) & (W_current < 0))

        return {
            'sign_flips': float(sign_flips),
            'sign_balance_change': float(new_positive - new_negative)
        }

    def _extract_sparsity_features(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Extract sparsity-related features."""
        # Zero-weight proportions
        zero_baseline = np.sum(np.abs(W_baseline) < self.edge_threshold) / W_baseline.size
        zero_current = np.sum(np.abs(W_current) < self.edge_threshold) / W_current.size

        # Sparsity change
        sparsity_change = zero_current - zero_baseline

        # Effective rank (measure of matrix complexity)
        try:
            U_base, s_base, _ = np.linalg.svd(W_baseline)
            U_curr, s_curr, _ = np.linalg.svd(W_current)

            # Effective rank based on singular values
            rank_baseline = np.sum(s_base > 0.01 * s_base[0]) if len(s_base) > 0 else 0
            rank_current = np.sum(s_curr > 0.01 * s_curr[0]) if len(s_curr) > 0 else 0

            rank_change = rank_current - rank_baseline
        except:
            rank_change = 0.0

        return {
            'sparsity_change': float(sparsity_change),
            'rank_change': float(rank_change)
        }

    def _extract_eigenvalue_features(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Extract eigenvalue-based spectral features."""
        try:
            # Compute eigenvalues
            eig_baseline = np.linalg.eigvals(W_baseline)
            eig_current = np.linalg.eigvals(W_current)

            # Sort by magnitude for stable comparison
            eig_baseline = np.sort(eig_baseline)
            eig_current = np.sort(eig_current)

            # Spectral radius (largest eigenvalue magnitude)
            radius_baseline = np.max(np.abs(eig_baseline)) if len(eig_baseline) > 0 else 0
            radius_current = np.max(np.abs(eig_current)) if len(eig_current) > 0 else 0
            radius_change = radius_current - radius_baseline

            # Spectral norm difference
            spectral_norm_diff = np.linalg.norm(eig_current - eig_baseline)

            # Dominant eigenvalue change
            dominant_change = np.abs(eig_current[0] - eig_baseline[0]) if len(eig_baseline) > 0 else 0

            return {
                'eigenvalue_radius_change': float(radius_change),
                'eigenvalue_norm_diff': float(spectral_norm_diff),
                'eigenvalue_dominant_change': float(dominant_change)
            }

        except np.linalg.LinAlgError:
            logger.warning("Eigenvalue computation failed, using zero values")
            return {
                'eigenvalue_radius_change': 0.0,
                'eigenvalue_norm_diff': 0.0,
                'eigenvalue_dominant_change': 0.0
            }

    def _extract_node_impact_features(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Extract per-variable impact features."""
        # Compute in-degree and out-degree changes
        adj_baseline = (np.abs(W_baseline) > self.edge_threshold).astype(int)
        adj_current = (np.abs(W_current) > self.edge_threshold).astype(int)

        # In-degree: sum along columns (incoming edges)
        in_degree_baseline = np.sum(adj_baseline, axis=0)
        in_degree_current = np.sum(adj_current, axis=0)
        in_degree_changes = in_degree_current - in_degree_baseline

        # Out-degree: sum along rows (outgoing edges)
        out_degree_baseline = np.sum(adj_baseline, axis=1)
        out_degree_current = np.sum(adj_current, axis=1)
        out_degree_changes = out_degree_current - out_degree_baseline

        # Maximum changes per node
        max_in_degree_change = np.max(np.abs(in_degree_changes))
        max_out_degree_change = np.max(np.abs(out_degree_changes))

        return {
            'node_max_in_degree_change': float(max_in_degree_change),
            'node_max_out_degree_change': float(max_out_degree_change)
        }

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return [
            # Magnitude features
            'magnitude_mean_change', 'magnitude_max_change', 'magnitude_std_change',
            # Structural features
            'structural_density_change', 'structural_density_change_abs',
            'structural_edges_added', 'structural_edges_removed',
            # Sign pattern features
            'sign_flips', 'sign_balance_change',
            # Sparsity features
            'sparsity_change', 'rank_change',
            # Eigenvalue features
            'eigenvalue_radius_change', 'eigenvalue_norm_diff', 'eigenvalue_dominant_change',
            # Node impact features
            'node_max_in_degree_change', 'node_max_out_degree_change'
        ]


class RuleBasedClassifier:
    """Rule-based classifier using hand-crafted rules from domain knowledge."""

    def __init__(self):
        """Initialize rule-based classifier with expert rules."""
        self.rules = self._define_classification_rules()

    def _define_classification_rules(self) -> Dict[str, Dict]:
        """
        Define hand-crafted rules for each anomaly type.

        Based on expected patterns:
        - Spike: Many new edges, high max change
        - Drift: Gradual changes, eigenvalue shifts
        - Level shift: Moderate structural changes
        - Amplitude change: Magnitude changes, preserved structure
        - Trend change: Sign flips, directional changes
        - Variance burst: High variability, density changes
        """
        return {
            'spike': {
                'primary_conditions': [
                    # Spike can either add edges OR remove edges (when strong effects dominate)
                    # Key is: large magnitude change + significant structural change
                    ('magnitude_max_change', '>', 0.1),
                ],
                'secondary_conditions': [
                    # At least one of these structural indicators
                    ('structural_edges_added', '>', 2.0),  # OR
                    ('structural_edges_removed', '>', 2.0),  # OR
                    ('structural_density_change_abs', '>', 0.08),  # Absolute density change
                    ('magnitude_mean_change', '>', 0.05),
                ],
                'exclusion_conditions': [
                    # Exclude if changes are too uniform (that's level_shift)
                    ('magnitude_std_change', '<', 0.02)
                ]
            },

            'drift': {
                'primary_conditions': [
                    ('eigenvalue_norm_diff', '>', 0.1),
                    ('magnitude_mean_change', '>', 0.02),
                    ('magnitude_mean_change', '<', 0.15)  # Gradual, not sudden
                ],
                'secondary_conditions': [
                    ('structural_density_change', '>', -0.1),  # Usually preserves or slightly increases structure
                    ('sign_flips', '<', 2.0)  # Usually preserves directions
                ],
                'exclusion_conditions': [
                    ('magnitude_max_change', '>', 0.3)  # Not sudden large changes
                ]
            },

            'level_shift': {
                'primary_conditions': [
                    ('magnitude_mean_change', '>', 0.05),
                    ('magnitude_std_change', '<', 0.1),  # Uniform shift
                    ('structural_density_change', '>', -0.05)
                ],
                'secondary_conditions': [
                    ('eigenvalue_radius_change', '>', 0.02),
                    ('rank_change', '==', 0.0)  # Preserves rank
                ],
                'exclusion_conditions': [
                    ('structural_edges_added', '>', 5.0)  # Not many new connections
                ]
            },

            'amplitude_change': {
                'primary_conditions': [
                    ('magnitude_max_change', '>', 0.08),
                    ('structural_density_change', '>', -0.02),  # Preserves structure
                    ('structural_density_change', '<', 0.02)
                ],
                'secondary_conditions': [
                    ('sparsity_change', '<', 0.1),  # Preserves sparsity pattern
                    ('sign_flips', '<', 1.0)  # Preserves directions
                ],
                'exclusion_conditions': [
                    ('structural_edges_added', '>', 3.0)
                ]
            },

            'trend_change': {
                'primary_conditions': [
                    ('sign_flips', '>', 0.5),
                    ('eigenvalue_dominant_change', '>', 0.05),
                    ('sign_balance_change', '!=', 0.0)
                ],
                'secondary_conditions': [
                    ('magnitude_mean_change', '>', 0.03),
                    ('structural_density_change', '>', -0.1)
                ],
                'exclusion_conditions': [
                    ('structural_edges_added', '>', 4.0)  # Trends usually modify existing
                ]
            },

            'variance_burst': {
                'primary_conditions': [
                    ('magnitude_std_change', '>', 0.08),
                    ('structural_density_change', '>', 0.05),
                    ('magnitude_max_change', '>', 0.12)
                ],
                'secondary_conditions': [
                    ('structural_edges_added', '>', 1.0),
                    ('rank_change', '>', 0.0)  # Increases complexity
                ],
                'exclusion_conditions': [
                    ('sparsity_change', '>', 0.2)  # Variance bursts fill in gaps
                ]
            }
        }

    def classify(self, signature: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify anomaly type using rule-based approach.

        Args:
            signature: 15-feature signature from GraphSignatureExtractor

        Returns:
            Classification result with confidence scores
        """
        scores = {}

        for anomaly_type, rule_set in self.rules.items():
            score = self._evaluate_rules(signature, rule_set)
            scores[anomaly_type] = score

        # Find best match
        best_type = max(scores.items(), key=lambda x: x[1])
        prediction = best_type[0] if best_type[1] > 0.5 else 'unknown'

        # Sort scores for ranking
        ranked_predictions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result = {
            'prediction': prediction,
            'confidence': float(best_type[1]),
            'all_scores': scores,
            'ranked_predictions': ranked_predictions,
            'method': 'rule_based'
        }

        logger.debug(f"Rule-based classification: {prediction} (confidence: {best_type[1]:.3f})")
        return result

    def _evaluate_rules(self, signature: Dict[str, float], rule_set: Dict) -> float:
        """Evaluate rule set against signature and return confidence score."""
        score = 0.0

        # Evaluate primary conditions (weighted heavily)
        primary_score = 0.0
        for feature, operator, threshold in rule_set['primary_conditions']:
            if feature in signature:
                condition_met = self._evaluate_condition(signature[feature], operator, threshold)
                primary_score += 1.0 if condition_met else 0.0

        primary_score = primary_score / len(rule_set['primary_conditions']) if rule_set['primary_conditions'] else 0.0

        # Evaluate secondary conditions (moderate weight)
        secondary_score = 0.0
        if 'secondary_conditions' in rule_set and rule_set['secondary_conditions']:
            for feature, operator, threshold in rule_set['secondary_conditions']:
                if feature in signature:
                    condition_met = self._evaluate_condition(signature[feature], operator, threshold)
                    secondary_score += 1.0 if condition_met else 0.0
            secondary_score = secondary_score / len(rule_set['secondary_conditions'])

        # Check exclusion conditions (veto power)
        exclusion_penalty = 0.0
        if 'exclusion_conditions' in rule_set and rule_set['exclusion_conditions']:
            for feature, operator, threshold in rule_set['exclusion_conditions']:
                if feature in signature:
                    condition_met = self._evaluate_condition(signature[feature], operator, threshold)
                    if condition_met:
                        exclusion_penalty += 0.3  # Heavy penalty for exclusion violations

        # Combine scores
        score = 0.6 * primary_score + 0.4 * secondary_score - exclusion_penalty
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        return score

    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a single condition."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 1e-6
        elif operator == '!=':
            return abs(value - threshold) >= 1e-6
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False


class MLBasedClassifier:
    """ML-based classifier using Random Forest for higher accuracy."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize ML classifier.

        Args:
            n_estimators: Number of trees in Random Forest
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

    def train(self, signatures: List[Dict[str, float]], labels: List[str],
              groups: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the ML classifier.

        Args:
            signatures: List of 15-feature signatures
            labels: Corresponding anomaly type labels
            groups: Optional group labels for cross-validation (e.g., launch IDs)

        Returns:
            Training results with cross-validation scores
        """
        logger.info(f"Training ML classifier on {len(signatures)} samples")

        # Convert to feature matrix
        extractor = GraphSignatureExtractor()
        self.feature_names = extractor.get_feature_names()

        X = []
        for sig in signatures:
            feature_vector = [sig.get(fname, 0.0) for fname in self.feature_names]
            X.append(feature_vector)

        X = np.array(X)
        y = np.array(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform cross-validation
        if groups is not None:
            # Leave-one-group-out CV (leave one launch out)
            cv = LeaveOneGroupOut()
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, groups=groups, scoring='f1_macro')
        else:
            # Standard 5-fold CV
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='f1_macro')

        # Train final model on all data
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        training_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'feature_importance': feature_importance,
            'top_features': sorted_importance[:5],
            'n_samples': len(signatures),
            'n_features': len(self.feature_names),
            'classes': list(self.model.classes_)
        }

        logger.info(f"Training complete. CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        return training_results

    def classify(self, signature: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify anomaly type using trained ML model.

        Args:
            signature: 15-feature signature

        Returns:
            Classification result with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before classification")

        # Convert signature to feature vector
        feature_vector = [signature.get(fname, 0.0) for fname in self.feature_names]
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)

        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Create probability dictionary
        prob_dict = dict(zip(self.model.classes_, probabilities))
        confidence = float(np.max(probabilities))

        # Rank predictions by probability
        ranked_predictions = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

        result = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'ranked_predictions': ranked_predictions,
            'method': 'ml_based'
        }

        logger.debug(f"ML classification: {prediction} (confidence: {confidence:.3f})")
        return result


def classify_anomaly_comprehensive(W_baseline: np.ndarray, W_current: np.ndarray,
                                 trained_ml_classifier: Optional[MLBasedClassifier] = None) -> Dict[str, Any]:
    """
    Comprehensive anomaly classification using both rule-based and ML approaches.

    Args:
        W_baseline: Baseline weight matrix
        W_current: Current weight matrix
        trained_ml_classifier: Optional pre-trained ML classifier

    Returns:
        Complete classification results from both methods
    """
    # Extract signature
    extractor = GraphSignatureExtractor()
    signature = extractor.extract_signature(W_baseline, W_current)

    # Rule-based classification
    rule_classifier = RuleBasedClassifier()
    rule_result = rule_classifier.classify(signature)

    results = {
        'signature': signature,
        'rule_based_result': rule_result
    }

    # ML-based classification if model available
    if trained_ml_classifier is not None:
        try:
            ml_result = trained_ml_classifier.classify(signature)
            results['ml_based_result'] = ml_result

            # Ensemble result (combine both methods)
            ensemble_result = _combine_classification_results(rule_result, ml_result)
            results['ensemble_result'] = ensemble_result

        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            results['ml_based_result'] = None

    return results


def _combine_classification_results(rule_result: Dict, ml_result: Dict) -> Dict[str, Any]:
    """Combine rule-based and ML-based classification results."""
    # Weighted combination (ML gets higher weight due to expected higher accuracy)
    rule_weight = 0.3
    ml_weight = 0.7

    # If both agree, high confidence
    if rule_result['prediction'] == ml_result['prediction']:
        combined_confidence = 0.5 * (rule_result['confidence'] + ml_result['confidence'])
        prediction = rule_result['prediction']
    else:
        # If they disagree, use weighted confidence
        rule_score = rule_weight * rule_result['confidence']
        ml_score = ml_weight * ml_result['confidence']

        if ml_score > rule_score:
            prediction = ml_result['prediction']
            combined_confidence = ml_result['confidence'] * 0.8  # Reduce confidence due to disagreement
        else:
            prediction = rule_result['prediction']
            combined_confidence = rule_result['confidence'] * 0.8

    return {
        'prediction': prediction,
        'confidence': float(combined_confidence),
        'agreement': rule_result['prediction'] == ml_result['prediction'],
        'rule_prediction': rule_result['prediction'],
        'ml_prediction': ml_result['prediction'],
        'method': 'ensemble'
    }