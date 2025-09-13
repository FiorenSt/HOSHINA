
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump, load
from pathlib import Path
import json, time

@dataclass
class ModelArtifacts:
    classes: List[str]
    clf_path: Path
    nn_path: Path
    iforest_path: Path
    lof_path: Path

class Trainer:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.model_dir = self.store_dir / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.clf: Optional[CalibratedClassifierCV] = None
        self.nn: Optional[NearestNeighbors] = None
        self.iforest: Optional[IsolationForest] = None
        self.lof: Optional[LocalOutlierFactor] = None
        self.classes_: Optional[List[str]] = None

    def train(self, X: np.ndarray, y: List[str]) -> ModelArtifacts:
        # Use a linear classifier + calibration for speed & usable probabilities
        base = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3)
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        clf.fit(X, y)
        self.clf = clf
        self.classes_ = list(clf.classes_)

        # k-NN index (cosine-ish via normalized vectors => Euclidean ~ cosine)
        nn = NearestNeighbors(n_neighbors=16, algorithm="auto", metric="euclidean")
        nn.fit(X)
        self.nn = nn

        # Outlier detectors
        iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
        iforest.fit(X)
        self.iforest = iforest

        # LOF (note: LOF is unsupervised; we fit on X)
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.02)
        lof.fit(X)
        self.lof = lof

        # Persist
        clf_path = self.model_dir / "clf.joblib"
        nn_path = self.model_dir / "nn.joblib"
        iforest_path = self.model_dir / "iforest.joblib"
        lof_path = self.model_dir / "lof.joblib"
        dump(clf, clf_path)
        dump(nn, nn_path)
        dump(iforest, iforest_path)
        dump(lof, lof_path)

        return ModelArtifacts(
            classes=self.classes_, 
            clf_path=clf_path, nn_path=nn_path,
            iforest_path=iforest_path, lof_path=lof_path
        )

    def load(self):
        clf_path = self.model_dir / "clf.joblib"
        if clf_path.exists():
            self.clf = load(clf_path)
            self.classes_ = list(self.clf.classes_)
        nn_path = self.model_dir / "nn.joblib"
        if nn_path.exists():
            self.nn = load(nn_path)
        iforest_path = self.model_dir / "iforest.joblib"
        if iforest_path.exists():
            self.iforest = load(iforest_path)
        lof_path = self.model_dir / "lof.joblib"
        if lof_path.exists():
            self.lof = load(lof_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            # uniform probabilities if not trained
            return np.ones((X.shape[0], 1)) / 1.0
        return self.clf.predict_proba(X)

    def knn(self, X: np.ndarray, k: int = 12) -> np.ndarray:
        if self.nn is None:
            nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
            nn.fit(X)
            return np.arange(min(k, len(X)))
        d, idx = self.nn.kneighbors(X, n_neighbors=min(k, self.nn.n_neighbors))
        return idx

    def outlier_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        scores = {}
        if self.iforest is not None:
            # lower score means more outlier in IsolationForest (decision_function: higher = inlier)
            scores["iforest"] = -self.iforest.score_samples(X)
        if self.lof is not None:
            scores["lof"] = -self.lof.score_samples(X)
        return scores
