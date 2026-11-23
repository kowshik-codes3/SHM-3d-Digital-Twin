"""Evaluation engine for calculating performance metrics and saving diagnostic plots."""

from pathlib import Path
from typing import Dict
import numpy as np

# Set matplotlib backend to non-interactive Agg before import to prevent display errors on headless systems
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


class BaselineEvaluator:
    """Evaluates unsupervised metrics and generates diagnostic reports."""
    
    @staticmethod
    def evaluate_performance(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """
        Calculates ROC-AUC and Precision-Recall AUC indexes.
        
        Args:
            y_true: Binary ground truth labels (0/1).
            scores: Inverted anomaly likelihood outputs.
            
        Returns:
            Dict containing metric names and float scores.
        """
        roc_auc = roc_auc_score(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        return {"roc_auc": roc_auc, "pr_auc": pr_auc}

    @staticmethod
    def generate_report_plots(
        pca_latents: np.ndarray,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        output_dir: str = "reports/phase3"
    ) -> None:
        """
        Outputs 2D feature projection scatters and continuous anomaly timeline reports.
        
        Args:
            pca_latents: 2D project coordinates from PCA transform (M, 2).
            y_true: Ground truth damage mask (M,).
            scores: Isolation Forest anomaly scores (M,).
            threshold: Standard control threshold limit.
            output_dir: Folder path where files will be created.
        """
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        
        # 1. 2D PCA Latent Space Separation
        plt.figure(figsize=(8, 5))
        plt.scatter(
            pca_latents[y_true == 0, 0], 
            pca_latents[y_true == 0, 1], 
            c='seagreen', 
            alpha=0.5, 
            s=20, 
            label='Healthy Operation'
        )
        plt.scatter(
            pca_latents[y_true == 1, 0], 
            pca_latents[y_true == 1, 1], 
            c='crimson', 
            edgecolors='black', 
            s=45, 
            label='Injected Structural Defect'
        )
        plt.title("2D PCA Feature Space: Baseline vs Damaged Manifold")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(p / "pca_latent_clusters.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Continuous Anomaly Score Timeline
        plt.figure(figsize=(12, 4))
        plt.plot(scores, label="Isolation Forest Anomaly Score", color="#1f77b4", lw=1.2)
        plt.axhline(
            threshold, 
            color="crimson", 
            linestyle="--", 
            label=f"Threshold (μ + 3σ = {threshold:.3f})"
        )
        
        # Shade true defect region
        defect_indices = np.where(y_true == 1)[0]
        if len(defect_indices) > 0:
            plt.axvspan(
                defect_indices[0], 
                defect_indices[-1], 
                color="orange", 
                alpha=0.2, 
                label="Ground Truth Defect Window"
            )
            
        plt.title("Timeline: Anomaly Score vs Decision Threshold")
        plt.xlabel("Sliding Window Index")
        plt.ylabel("Anomaly Score")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.savefig(p / "anomaly_timeline.png", dpi=300, bbox_inches="tight")
        plt.close()
