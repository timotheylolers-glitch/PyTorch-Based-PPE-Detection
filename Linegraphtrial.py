"""
Validation Line Graph: Expected vs Actual Output with Error Criteria
Shows model performance with error thresholds for PPE detection
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json


class ValidationPlotter:
    """Create validation plots comparing expected vs actual outputs"""
    
    def __init__(self, error_threshold: float = 0.2):
        """
        Initialize the validation plotter.
        
        Args:
            error_threshold: Maximum acceptable error (default 0.2 = 20%)
        """
        self.error_threshold = error_threshold
        self.output_dir = Path('./output_analysis')
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_confidence_validation(
        self, 
        expected_confidences: List[float],
        actual_confidences: List[float],
        sample_ids: List[str] = None,
        title: str = "Model Confidence: Expected vs Actual Output"
    ) -> str:
        """
        Create a line graph comparing expected vs actual confidence scores.
        
        Args:
            expected_confidences: List of expected confidence values
            actual_confidences: List of actual confidence values  
            sample_ids: List of sample identifiers (optional)
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if len(expected_confidences) != len(actual_confidences):
            raise ValueError("Lengths of expected and actual values must match")
        
        n_samples = len(expected_confidences)
        if sample_ids is None:
            sample_ids = [f"Sample {i}" for i in range(n_samples)]
        
        # Calculate errors
        errors = np.abs(np.array(expected_confidences) - np.array(actual_confidences))
        within_threshold = errors <= self.error_threshold
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Expected vs Actual Line Graph
        x_pos = np.arange(n_samples)
        ax1.plot(x_pos, expected_confidences, 'o-', linewidth=2.5, markersize=8, 
                label='Expected', color='#2ecc71', alpha=0.8)
        ax1.plot(x_pos, actual_confidences, 's-', linewidth=2.5, markersize=8,
                label='Actual', color='#3498db', alpha=0.8)
        
        # Add error band
        ax1.fill_between(x_pos, expected_confidences, actual_confidences, 
                        alpha=0.2, color='#e74c3c', label='Error Zone')
        
        # Add error threshold line
        threshold_line = self.error_threshold
        ax1.axhline(y=threshold_line, color='red', linestyle='--', linewidth=2, 
                   label=f'Error Threshold ({self.error_threshold*100:.0f}%)', alpha=0.7)
        
        ax1.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"S{i}" for i in range(n_samples)], fontsize=9)
        
        # Plot 2: Error Analysis
        colors = ['#27ae60' if within else '#e74c3c' for within in within_threshold]
        bars = ax2.bar(x_pos, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axhline(y=self.error_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Error Threshold', alpha=0.7)
        
        ax2.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Error Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"S{i}" for i in range(n_samples)], fontsize=9)
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add statistics box
        accuracy = np.sum(within_threshold) / len(within_threshold) * 100
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        stats_text = f'Accuracy: {accuracy:.1f}%\nMean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}'
        fig.text(0.98, 0.02, stats_text, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                ha='right', va='bottom', family='monospace')
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        
        output_file = self.output_dir / 'validation_confidence.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def plot_class_prediction_validation(
        self,
        expected_classes: List[str],
        actual_classes: List[str],
        class_mapping: Dict[str, int] = None,
        title: str = "PPE Class Prediction: Expected vs Actual"
    ) -> str:
        """
        Create validation plot for class predictions.
        
        Args:
            expected_classes: List of expected class labels
            actual_classes: List of actual predicted classes
            class_mapping: Dict mapping class names to numeric values
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if len(expected_classes) != len(actual_classes):
            raise ValueError("Lengths of expected and actual classes must match")
        
        if class_mapping is None:
            unique_classes = sorted(set(expected_classes + actual_classes))
            class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        
        n_samples = len(expected_classes)
        x_pos = np.arange(n_samples)
        
        # Convert to numeric values
        expected_numeric = np.array([class_mapping[cls] for cls in expected_classes])
        actual_numeric = np.array([class_mapping[cls] for cls in actual_classes])
        
        # Calculate accuracy
        matches = expected_numeric == actual_numeric
        accuracy = np.sum(matches) / len(matches) * 100
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Expected vs Actual Classes
        ax1.plot(x_pos, expected_numeric, 'o-', linewidth=2.5, markersize=8,
                label='Expected', color='#2ecc71', alpha=0.8)
        ax1.plot(x_pos, actual_numeric, 's-', linewidth=2.5, markersize=8,
                label='Actual (Predicted)', color='#3498db', alpha=0.8)
        
        # Highlight mismatches
        mismatch_indices = np.where(~matches)[0]
        if len(mismatch_indices) > 0:
            ax1.scatter(mismatch_indices, actual_numeric[mismatch_indices], 
                       s=200, marker='x', color='red', linewidth=3, label='Mismatch', zorder=5)
        
        ax1.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Class Index', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_yticks(list(class_mapping.values()))
        ax1.set_yticklabels(list(class_mapping.keys()))
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"S{i}" for i in range(n_samples)], fontsize=9)
        ax1.legend(loc='best', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Prediction Accuracy per Sample
        colors = ['#27ae60' if match else '#e74c3c' for match in matches]
        ax2.bar(x_pos, matches.astype(int), color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Correct Prediction (1=Yes, 0=No)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Accuracy Distribution', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.1, 1.2)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"S{i}" for i in range(n_samples)], fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add statistics box
        stats_text = f'Overall Accuracy: {accuracy:.1f}%\nCorrect: {np.sum(matches)}/{n_samples}\nErrors: {len(mismatch_indices)}'
        fig.text(0.98, 0.02, stats_text, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                ha='right', va='bottom', family='monospace')
        
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        
        output_file = self.output_dir / 'validation_class_prediction.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def plot_multi_metric_validation(
        self,
        metrics_data: Dict[str, Dict[str, List[float]]],
        error_thresholds: Dict[str, float] = None,
        title: str = "Multi-Metric Validation: Expected vs Actual"
    ) -> str:
        """
        Create validation plot with multiple metrics.
        
        Args:
            metrics_data: Dict with structure {metric_name: {expected: [...], actual: [...]}}
            error_thresholds: Dict with error thresholds for each metric
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        n_metrics = len(metrics_data)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 5*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric_name, data) in enumerate(metrics_data.items()):
            expected = data.get('expected', [])
            actual = data.get('actual', [])
            
            if len(expected) != len(actual):
                print(f"Warning: {metric_name} has mismatched lengths")
                continue
            
            n_samples = len(expected)
            x_pos = np.arange(n_samples)
            threshold = error_thresholds.get(metric_name, self.error_threshold) if error_thresholds else self.error_threshold
            
            ax = axes[idx]
            
            # Plot lines
            ax.plot(x_pos, expected, 'o-', linewidth=2.5, markersize=7,
                   label='Expected', color='#2ecc71', alpha=0.8)
            ax.plot(x_pos, actual, 's-', linewidth=2.5, markersize=7,
                   label='Actual', color='#3498db', alpha=0.8)
            
            # Add error band
            ax.fill_between(x_pos, expected, actual, alpha=0.2, color='#e74c3c')
            
            # Add threshold if applicable
            if threshold is not None and threshold < max(expected + actual):
                ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Threshold: {threshold}')
            
            ax.set_xlabel('Sample ID', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"S{i}" for i in range(n_samples)], fontsize=8)
        
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / 'validation_multi_metric.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)


# Example usage and demo
if __name__ == "__main__":
    # Demo: Create sample data and generate validation plots
    plotter = ValidationPlotter(error_threshold=0.15)
    
    print("=" * 70)
    print("PPE Detection Validation Visualization")
    print("=" * 70)
    
    # Example 1: Confidence Score Validation
    print("\n📊 Generating Confidence Validation Plot...")
    expected_conf = [0.95, 0.87, 0.92, 0.78, 0.88, 0.81, 0.93, 0.85, 0.91, 0.76]
    actual_conf = [0.92, 0.85, 0.90, 0.82, 0.86, 0.79, 0.95, 0.83, 0.89, 0.75]
    
    conf_plot = plotter.plot_confidence_validation(
        expected_conf, actual_conf,
        title="PPE Detection Model: Confidence Score Validation"
    )
    print(f"✓ Saved to: {conf_plot}\n")
    
    # Example 2: Class Prediction Validation
    print("📊 Generating Class Prediction Validation Plot...")
    expected_classes = ['helmet', 'vest', 'helmet', 'no_ppe', 'vest', 'helmet', 'vest', 'no_ppe', 'helmet', 'vest']
    actual_classes = ['helmet', 'vest', 'helmet', 'no_ppe', 'vest', 'vest', 'vest', 'no_ppe', 'helmet', 'vest']
    
    class_plot = plotter.plot_class_prediction_validation(
        expected_classes, actual_classes,
        class_mapping={'helmet': 0, 'vest': 1, 'no_ppe': 2}
    )
    print(f"✓ Saved to: {class_plot}\n")
    
    # Example 3: Multi-Metric Validation
    print("📊 Generating Multi-Metric Validation Plot...")
    metrics = {
        'Precision': {
            'expected': [0.92, 0.88, 0.95, 0.81, 0.89],
            'actual': [0.90, 0.86, 0.94, 0.83, 0.87]
        },
        'Recall': {
            'expected': [0.88, 0.91, 0.87, 0.86, 0.92],
            'actual': [0.90, 0.89, 0.88, 0.84, 0.90]
        },
        'F1-Score': {
            'expected': [0.90, 0.89, 0.91, 0.83, 0.90],
            'actual': [0.90, 0.87, 0.91, 0.83, 0.88]
        }
    }
    
    multi_plot = plotter.plot_multi_metric_validation(
        metrics,
        error_thresholds={'Precision': 0.10, 'Recall': 0.10, 'F1-Score': 0.10}
    )
    print(f"✓ Saved to: {multi_plot}\n")
    
    print("=" * 70)
    print("✓ All validation plots generated successfully!")
    print("=" * 70)
