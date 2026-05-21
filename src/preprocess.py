from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features import LEAKAGE_COLUMNS, classify_columns

# Canonical CICIDS2017 Web Attack sub-labels (space-hyphen-space separator).
_CICIDS2017_WEB_ATTACK_LABELS = {
    "brute force": "Web Attack - Brute Force",
    "sql injection": "Web Attack - Sql Injection",
    "xss": "Web Attack - XSS",
}

# Known label variants from CSV encoding / dash differences.
_CICIDS2017_LABEL_ALIASES: dict[str, str] = {
    "Web Attack—Brute Force": "Web Attack - Brute Force",
    "Web Attack–Brute Force": "Web Attack - Brute Force",
    "Web Attack-Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack—Sql Injection": "Web Attack - Sql Injection",
    "Web Attack–Sql Injection": "Web Attack - Sql Injection",
    "Web Attack-Sql Injection": "Web Attack - Sql Injection",
    "Web Attack \ufffd Sql Injection": "Web Attack - Sql Injection",
    "Web Attack—XSS": "Web Attack - XSS",
    "Web Attack–XSS": "Web Attack - XSS",
    "Web Attack-XSS": "Web Attack - XSS",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
}

_WEB_ATTACK_LABEL_RE = re.compile(
    r"^Web Attack\s*[\u2013\u2014\ufffd\uFFFD\-–—]\s*(.+)$",
    re.IGNORECASE,
)


def _normalize_label_value(label: str) -> str:
    """Map a single label string to its canonical CICIDS2017 form."""
    label = label.strip()
    if label in _CICIDS2017_LABEL_ALIASES:
        return _CICIDS2017_LABEL_ALIASES[label]

    match = _WEB_ATTACK_LABEL_RE.match(label)
    if match:
        subtype = match.group(1).strip().lower()
        if subtype in _CICIDS2017_WEB_ATTACK_LABELS:
            return _CICIDS2017_WEB_ATTACK_LABELS[subtype]

    return label


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Strip and canonicalize CICIDS2017 label strings (fixes Web Attack mojibake)."""
    df = df.copy()
    labels = df[label_col].astype(str).str.strip()

    before = sorted(labels.unique())
    print(
        f"Unique labels before normalization ({len(before)}): "
        f"{[label.encode('unicode_escape').decode('ascii') for label in before]}"
    )

    df[label_col] = labels.map(_normalize_label_value)

    after = sorted(df[label_col].unique())
    print(
        f"Unique labels after normalization ({len(after)}): "
        f"{[label.encode('unicode_escape').decode('ascii') for label in after]}"
    )
    return df


def find_raw_csv_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        files = sorted(raw_dir.rglob("*.csv"))
    return files


def load_cicids2017(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load and concatenate CICIDS2017 MachineLearningCSV day files."""
    files = find_raw_csv_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}. Place CICIDS2017 MachineLearningCSV files there."
        )

    chunks: list[pd.DataFrame] = []
    rows_read = 0
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        # Normalize column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            df = df.head(remaining)
        chunks.append(df)
        rows_read += len(df)
        if max_rows is not None and rows_read >= max_rows:
            break

    data = pd.concat(chunks, ignore_index=True)
    return data


def clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Basic cleaning: infinities, missing labels, canonical label strings."""
    df = df.replace([np.inf, -np.inf], np.nan)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Columns: {list(df.columns)[:10]}...")

    df = normalize_labels(df, label_col)
    df = df.dropna(subset=[label_col])
    # Drop rows with NaN in feature columns after leakage drop
    return df


def prepare_features(
    df: pd.DataFrame,
    label_col: str,
    drop_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, dict[str, list[str]]]:
    drop_existing = [c for c in drop_columns if c in df.columns]
    feature_df = df.drop(columns=drop_existing + [label_col], errors="ignore")

    # Remove non-numeric columns except those we can encode (Protocol is numeric in CICIDS)
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df = feature_df.fillna(0)
    groups = classify_columns(feature_df.columns.tolist())

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    return feature_df, y, le, groups


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=seed
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_share), stratify=y_temp, random_state=seed
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def run_preprocess(cfg: dict[str, Any], pilot: bool = False) -> dict[str, Path]:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    max_rows = cfg["data"].get("pilot_max_rows") if pilot else cfg["data"].get("max_rows")

    print(f"Loading data from {raw_dir} (max_rows={max_rows})...")
    df = load_cicids2017(raw_dir, max_rows=max_rows)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    df = clean_dataframe(df, cfg["data"]["label_column"])
    feature_df, y, label_encoder, feature_groups = prepare_features(
        df,
        cfg["data"]["label_column"],
        cfg["data"]["drop_columns"],
    )

    X = feature_df.values.astype(np.float32)
    splits = stratified_split(
        X,
        y,
        cfg["split"]["train_ratio"],
        cfg["split"]["val_ratio"],
        cfg["split"]["test_ratio"],
        cfg["seed"],
    )

    scaler = StandardScaler()
    splits["X_train"] = scaler.fit_transform(splits["X_train"]).astype(np.float32)
    splits["X_val"] = scaler.transform(splits["X_val"]).astype(np.float32)
    splits["X_test"] = scaler.transform(splits["X_test"]).astype(np.float32)

    # Persist artifacts
    np.savez_compressed(
        out_dir / "splits.npz",
        X_train=splits["X_train"],
        X_val=splits["X_val"],
        X_test=splits["X_test"],
        y_train=splits["y_train"],
        y_val=splits["y_val"],
        y_test=splits["y_test"],
    )
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(label_encoder, out_dir / "label_encoder.joblib")

    meta = {
        "n_samples": int(len(y)),
        "n_features": int(feature_df.shape[1]),
        "feature_names": feature_df.columns.tolist(),
        "class_names": label_encoder.classes_.tolist(),
        "n_classes": int(len(label_encoder.classes_)),
        "pilot": pilot,
        "feature_groups": feature_groups,
        "model_input_space": "standardized",
        "constraint_projection_space": "raw_physical",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved processed data to {out_dir}")
    print(f"  Train: {len(splits['y_train']):,}  Val: {len(splits['y_val']):,}  Test: {len(splits['y_test']):,}")
    classes_preview = ", ".join(str(c) for c in meta["class_names"][:12])
    if meta["n_classes"] > 12:
        classes_preview += f", ... (+{meta['n_classes'] - 12} more)"
    print(f"  Classes ({meta['n_classes']}): {classes_preview}")
    return {"processed_dir": out_dir}
