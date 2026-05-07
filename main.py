"""
Cluster & Culture – FastAPI backend
PCA explorer + Custom-axis scatter + ML Classification for LLM benchmarks.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as sk_confmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# ── Load & enrich data ────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_DIR / "llm_benchmarks.csv")
df_raw["Company"]      = df_raw["Model"].str.split("/").str[0]
df_raw["Release Date"] = pd.to_datetime(df_raw["Release Date"], errors="coerce")
df_raw["Release Year"] = df_raw["Release Date"].dt.year.astype("Int64")

with open(DATA_DIR / "column_metadata.json") as f:
    col_meta = json.load(f)

BENCHMARK_COLS: list[str] = [
    col for col, meta in col_meta["columns"].items()
    if meta.get("is_benchmark") and col in df_raw.columns
]

PRIMARY_BENCHMARKS = [
    "oll_BBH", "oll_GPQA", "oll_IFEval", "oll_MATH_lvl5", "oll_MMLU_PRO", "oll_MUSR",
]

# Special non-benchmark axes available for scatter plots
SPECIAL_AXES = {"release_date": "Release Date"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_filters(df, origins, year_min, year_max, bench_cols, min_coverage):
    if origins:
        ol = [o.strip() for o in origins.split(",") if o.strip()]
        if ol:
            df = df[df["Origin"].isin(ol)]
    if year_min is not None:
        df = df[df["Release Year"] >= year_min]
    if year_max is not None:
        df = df[df["Release Year"] <= year_max]
    if bench_cols:
        coverage = df[bench_cols].notna().mean(axis=1)
        df = df[coverage >= min_coverage]
    return df

def _point_meta(row) -> dict:
    rd = row["Release Date"]
    return {
        "model":        row["Model"],
        "company":      row["Company"],
        "origin":       row["Origin"] if pd.notna(row["Origin"]) else "Unknown",
        "release_date": rd.strftime("%Y-%m-%d") if pd.notna(rd) else None,
        "release_year": int(row["Release Year"]) if pd.notna(row["Release Year"]) else None,
    }

def _axis_val(row, axis):
    """Read a scatter axis value; handles special axes like release_date."""
    if axis == "release_date":
        v = row["Release Date"]
        return v.strftime("%Y-%m-%d") if pd.notna(v) else None
    if axis in SPECIAL_AXES:
        v = row[SPECIAL_AXES[axis]]
        return float(int(v)) if pd.notna(v) else None
    v = row.get(axis)
    return round(float(v), 4) if pd.notna(v) else None

def _is_valid_axis(a):
    return a in BENCHMARK_COLS or a in SPECIAL_AXES

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cluster & Culture")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/filters")
def get_filters():
    origins = sorted(df_raw["Origin"].dropna().unique().tolist())
    years   = sorted(int(y) for y in df_raw["Release Year"].dropna().unique() if not pd.isna(y))
    benchmarks = []
    for col in BENCHMARK_COLS:
        series = df_raw[col].dropna()
        meta = col_meta["columns"].get(col, {})
        benchmarks.append({
            "col":              col,
            "description":      meta.get("description", ""),
            "test_areas":       meta.get("test_areas", []),
            "source":           meta.get("source", ""),
            "higher_is_better": meta.get("higher_is_better", True),
            "is_primary":       col in PRIMARY_BENCHMARKS,
            "null_pct":         round(df_raw[col].isna().mean() * 100, 1),
            "n":                int(len(series)),
            "mean":             round(float(series.mean()), 2) if len(series) else None,
            "median":           round(float(series.median()), 2) if len(series) else None,
            "min":              round(float(series.min()), 2) if len(series) else None,
            "max":              round(float(series.max()), 2) if len(series) else None,
        })
    return {"origins": origins, "years": years, "benchmarks": benchmarks,
            "primary_benchmarks": PRIMARY_BENCHMARKS}


@app.get("/api/pca")
def compute_pca(
    dims: int = Query(2, ge=2, le=3),
    benchmarks: str = Query(",".join(PRIMARY_BENCHMARKS)),
    origins: str = Query(""),
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_coverage: float = Query(0.5, ge=0.0, le=1.0),
):
    selected = [b for b in benchmarks.split(",") if b and b in df_raw.columns and b in BENCHMARK_COLS]
    if not selected:
        selected = PRIMARY_BENCHMARKS[:]

    df = _apply_filters(df_raw.copy(), origins, year_min, year_max, selected, min_coverage)
    if len(df) < 3:
        return {"error": "Not enough data (need ≥ 3 models).", "points": [],
                "variance_explained": [], "loadings": {}, "n_points": 0, "selected_benchmarks": selected}

    X = df[selected].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    n_comp = min(dims, X.shape[1], X.shape[0])
    pca    = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)
    while coords.shape[1] < dims:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    loadings = {
        f"pc{i+1}": sorted(
            [{"benchmark": b, "loading": round(float(pca.components_[i, j]), 4)}
             for j, b in enumerate(selected)],
            key=lambda x: abs(x["loading"]), reverse=True
        )
        for i in range(n_comp)
    }

    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        pt = _point_meta(row)
        pt["pc1"] = round(float(coords[i, 0]), 4)
        pt["pc2"] = round(float(coords[i, 1]), 4)
        if dims == 3:
            pt["pc3"] = round(float(coords[i, 2]), 4)
        pt["scores"] = {col: (round(float(row[col]), 2) if pd.notna(row.get(col)) else None)
                        for col in BENCHMARK_COLS}
        points.append(pt)

    return {
        "points":              points,
        "variance_explained":  [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "loadings":            loadings,
        "n_points":            len(points),
        "selected_benchmarks": selected,
    }


@app.get("/api/scatter")
def get_scatter(
    x_axis: str = Query("oll_MMLU_PRO"),
    y_axis: str = Query("oll_BBH"),
    z_axis: Optional[str] = None,
    origins: str = Query(""),
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_coverage: float = Query(0.0, ge=0.0, le=1.0),
):
    axes_requested = [x_axis, y_axis] + ([z_axis] if z_axis else [])
    valid_axes = [a for a in axes_requested if _is_valid_axis(a)]
    if len(valid_axes) < 2:
        return {"error": "Need at least 2 valid axes.", "points": [], "n_points": 0}

    bench_axes = [a for a in valid_axes if a in BENCHMARK_COLS]
    df = _apply_filters(df_raw.copy(), origins, year_min, year_max, bench_axes, min_coverage)
    if len(df) < 2:
        return {"error": "Not enough data after filtering (need ≥ 2 models).", "points": [], "n_points": 0}

    points = []
    for _, row in df.iterrows():
        xv = _axis_val(row, valid_axes[0])
        yv = _axis_val(row, valid_axes[1])
        if xv is None or yv is None:
            continue
        pt = _point_meta(row)
        pt["x"] = xv
        pt["y"] = yv
        if len(valid_axes) >= 3:
            pt["z"] = _axis_val(row, valid_axes[2])
        pt["scores"] = {col: (round(float(row[col]), 2) if pd.notna(row.get(col)) else None)
                        for col in BENCHMARK_COLS}
        points.append(pt)

    return {"points": points, "axes": valid_axes, "n_points": len(points)}


@app.get("/api/classify")
def classify_models(
    classifier: str = Query("svm"),
    target: str = Query("origin"),
    benchmarks: str = Query(",".join(PRIMARY_BENCHMARKS)),
    origins: str = Query(""),
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    min_coverage: float = Query(0.5, ge=0.0, le=1.0),
    top_n_classes: int = Query(8, ge=2, le=25),
    test_size: float = Query(0.25, ge=0.1, le=0.5),
):
    """Train a classifier to predict a categorical attribute from benchmark scores."""
    selected = [b for b in benchmarks.split(",") if b and b in df_raw.columns and b in BENCHMARK_COLS]
    if not selected:
        selected = PRIMARY_BENCHMARKS[:]

    df = _apply_filters(df_raw.copy(), origins, year_min, year_max, selected, min_coverage)
    if len(df) < 10:
        return {"error": "Not enough data after filtering (need ≥ 10 models).",
                "accuracy": None, "confusion_matrix": [], "labels": []}

    # Build target series
    target_map = {
        "origin":       ("Origin",       "Origin / Country"),
        "company":      ("Company",      "Company"),
        "release_year": ("Release Year", "Release Year"),
    }
    if target not in target_map:
        return {"error": f"Unknown target: {target}"}

    col_name, target_label = target_map[target]
    y_raw = df[col_name].fillna("Unknown").astype(str)

    # Keep only top N most-common classes; drop the rest (no "Other" bucket)
    top_classes = set(y_raw.value_counts().head(top_n_classes).index)
    mask        = y_raw.isin(top_classes)
    y_raw       = y_raw[mask]
    df          = df[mask]

    # Drop classes with fewer than 3 samples (can't stratify)
    class_counts = y_raw.value_counts()
    valid_classes = set(class_counts[class_counts >= 3].index)
    mask2 = y_raw.isin(valid_classes)
    y_raw = y_raw[mask2]
    df    = df[mask2]

    if len(df) < 8:
        return {"error": "Too few samples remain after class filtering (need ≥ 8).",
                "accuracy": None, "confusion_matrix": [], "labels": []}
    if y_raw.nunique() < 2:
        return {"error": "Need at least 2 classes to train a classifier.",
                "accuracy": None, "confusion_matrix": [], "labels": []}

    labels = sorted(y_raw.unique().tolist())
    y = y_raw.values

    X = df[selected].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    # Stratified split; fall back to random if stratification fails
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42)

    # Build classifier
    if classifier == "svm":
        clf      = SVC(kernel="rbf", C=1.0, random_state=42)
        clf_name = "SVM (RBF)"
    elif classifier == "dt":
        clf      = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=42)
        clf_name = "Decision Tree"
    elif classifier == "lr":
        clf      = LogisticRegression(max_iter=2000, random_state=42)
        clf_name = "Logistic Regression"
    else:
        return {"error": f"Unknown classifier: {classifier}"}

    clf.fit(X_tr, y_tr)
    y_pred   = clf.predict(X_te)
    accuracy = float(accuracy_score(y_te, y_pred))
    cm       = sk_confmat(y_te, y_pred, labels=labels).tolist()

    # Feature importances (DT and LR only)
    feature_importances = None
    if classifier == "dt":
        feature_importances = {b: round(float(v), 4)
                               for b, v in zip(selected, clf.feature_importances_)}
    elif classifier == "lr" and hasattr(clf, "coef_"):
        avg_abs = np.abs(clf.coef_).mean(axis=0)
        feature_importances = {b: round(float(v), 4) for b, v in zip(selected, avg_abs)}

    # Per-class metrics for the test set
    per_class = []
    for i, lbl in enumerate(labels):
        row_total = sum(cm[i])
        col_total = sum(cm[r][i] for r in range(len(labels)))
        tp = cm[i][i]
        precision = round(tp / col_total, 3) if col_total else 0.0
        recall    = round(tp / row_total,  3) if row_total else 0.0
        per_class.append({"label": lbl, "precision": precision, "recall": recall,
                           "n_test": row_total})

    return {
        "accuracy":             round(accuracy, 4),
        "n_train":              int(len(X_tr)),
        "n_test":               int(len(X_te)),
        "labels":               labels,
        "confusion_matrix":     cm,
        "feature_importances":  feature_importances,
        "per_class":            per_class,
        "classifier":           clf_name,
        "target":               target_label,
        "n_points":             int(len(df)),
        "warning":              "Small test set — accuracy may be noisy." if len(X_te) < 15 else None,
    }
