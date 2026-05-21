from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    save_path: Path | None = None,
) -> RandomForestClassifier:
    rcfg = cfg["models"]["rf"]
    clf = RandomForestClassifier(
        n_estimators=rcfg["n_estimators"],
        max_depth=rcfg["max_depth"],
        min_samples_leaf=rcfg["min_samples_leaf"],
        n_jobs=rcfg["n_jobs"],
        class_weight=rcfg["class_weight"],
        random_state=cfg["seed"],
    )
    print("  Training Random Forest...")
    clf.fit(X_train, y_train)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
    return clf
