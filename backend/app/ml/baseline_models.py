"""Encoder-based baseline anomaly detection algorithms (PCA, Isolation Forest, LOF)."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class BaselineStructuralDetector:
    """Encapsulates classical unsupervised anomaly detection models."""
    
    def __init__(self, pca_components: int = 2, contamination: float = 0.02):
        """
        Initializes models with specific parameters.
        
        Args:
            pca_components: Int representation of target dimensionality reduction (e.g. 2).
            contamination: Estimation of normal train outlier density. Default is 0.02.
        """
        self.pca = PCA(n_components=pca_components)
        self.iso_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.lof = LocalOutlierFactor(novelty=True, contamination=contamination, n_jobs=-1)
        self.threshold = 0.0

    def fit(self, X_train_feats: np.ndarray) -> 'BaselineStructuralDetector':
        """
        Fits unsupervised classical models on healthy baseline train split.
        
        Args:
            X_train_feats: Tabular feature matrix (M, D).
            
        Returns:
            Fitted BaselineStructuralDetector instance.
        """
        # Fit models on healthy baseline features
        self.pca.fit(X_train_feats)
        self.iso_forest.fit(X_train_feats)
        self.lof.fit(X_train_feats)
        
        # Calibrate baseline statistical control limit (mu + 3*sigma of inverted scores)
        train_scores = -self.iso_forest.score_samples(X_train_feats)
        self.threshold = np.mean(train_scores) + 3.0 * np.std(train_scores)
        return self

    def score(self, X_feats: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns anomaly scores and predictions for PCA and Isolation Forest.
        
        Args:
            X_feats: Tabular feature matrix containing data to project.
            
        Returns:
            Dict containing anomaly scores, binary threshold predictions,
            Q-residuals (SPE), and 2D/3D PCA projection latents.
        """
        # Isolation Forest score: inverted so higher score = higher anomaly likelihood
        iso_scores = -self.iso_forest.score_samples(X_feats)
        iso_preds = np.where(iso_scores > self.threshold, 1, 0)
        
        # PCA Reconstruction Error (SPE)
        X_projected = self.pca.inverse_transform(self.pca.transform(X_feats))
        spe = np.mean((X_feats - X_projected)**2, axis=1)
        
        return {
            "iso_scores": iso_scores,
            "iso_preds": iso_preds,
            "spe": spe,
            "pca_latents": self.pca.transform(X_feats)
        }

    def save(self, artifact_dir: Union[str, Path] = "artifacts") -> None:
        """
        Persists fitted modules to target location.
        
        Args:
            artifact_dir: Absolute or relative output folder path.
        """
        p = Path(artifact_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.iso_forest, p / "isolation_forest.joblib")
        joblib.dump(self.pca, p / "pca_baseline.joblib")
        joblib.dump(self.threshold, p / "iso_threshold.joblib")
