#!/usr/bin/env python
"""Pre-compute all visualization data for the Quarto OJS dashboard.

Usage:
    python precompute.py              # use existing CSVs
    python precompute.py --refresh    # download fresh HF data first

Outputs JSON files to precomputed/ that main.qmd loads client-side.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path("data")
OUT_DIR = Path("precomputed")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


def safe_val(val):
    """NaN/NaT → None for JSON."""
    if isinstance(val, float) and np.isnan(val):
        return None
    if pd.isna(val):
        return None
    return val


META_COLS = ["Origin", "Company", "Release Year", "Country", "Organization", "Access", "eci"]


def make_point(row, extra_cols=None):
    """Build a JSON-safe dict from a DataFrame row."""
    p = {"Model": row.get("Model", "")}
    for col in META_COLS:
        if col in row.index:
            p[col] = safe_val(row[col])
    for col in (extra_cols or []):
        if col in row.index:
            p[col] = safe_val(row[col])
    return p


# ── PCA ───────────────────────────────────────────────────────────────────────

def compute_pca(df, benchmarks, dims=2, min_coverage=0.5):
    selected = [b for b in benchmarks if b in df.columns]
    df = df.copy()
    coverage = df[selected].notna().mean(axis=1)
    df = df[coverage >= min_coverage].copy()

    X = df[selected].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    n_comp = min(dims, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)

    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        p = make_point(row)
        p["PC1"] = float(coords[i, 0])
        p["PC2"] = float(coords[i, 1])
        if dims >= 3 and n_comp >= 3:
            p["PC3"] = float(coords[i, 2])
        points.append(p)

    return {
        "variance": [float(v) for v in pca.explained_variance_ratio_],
        "points": points,
    }


def compute_loadings(df, benchmarks, min_coverage=0.5):
    selected = [b for b in benchmarks if b in df.columns]
    df = df.copy()
    coverage = df[selected].notna().mean(axis=1)
    df = df[coverage >= min_coverage]

    X = df[selected].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    n_comp = min(3, len(selected), X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)

    var = [float(v) for v in pca.explained_variance_ratio_]
    return {
        "benchmarks": selected,
        "components": [f"PC{i+1} ({var[i]*100:.1f}%)" for i in range(n_comp)],
        "matrix": [[float(v) for v in row] for row in pca.components_[:n_comp]],
    }


# ── UMAP ──────────────────────────────────────────────────────────────────────

def compute_umap(df, benchmarks, min_coverage=0.5, n_neighbors=10, min_dist=0.1):
    selected = [b for b in benchmarks if b in df.columns]
    df = df.copy()
    coverage = df[selected].notna().mean(axis=1)
    df = df[coverage >= min_coverage].copy()

    X = df[selected].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)

    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        p = make_point(row, extra_cols=selected)
        p["UMAP1"] = float(embedding[i, 0])
        p["UMAP2"] = float(embedding[i, 1])
        points.append(p)

    # Spearman correlations
    corr = {"benchmarks": selected, "UMAP1": [], "UMAP2": []}
    for bench in selected:
        vals = df[bench].values.astype(float)
        mask = ~np.isnan(vals)
        r1, _ = spearmanr(vals[mask], embedding[mask, 0])
        r2, _ = spearmanr(vals[mask], embedding[mask, 1])
        corr["UMAP1"].append(round(float(r1), 3))
        corr["UMAP2"].append(round(float(r2), 3))

    return {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "points": points,
    }, corr


# ── Scatter data ──────────────────────────────────────────────────────────────

def compute_scatter_data(df, benchmarks):
    selected = [b for b in benchmarks if b in df.columns]
    points = [make_point(row, extra_cols=selected) for _, row in df.iterrows()]
    return {"benchmarks": selected, "points": points}


# ── Classifier ────────────────────────────────────────────────────────────────

def compute_classifier(df, benchmarks, classifier, target_col, target_label,
                       origins, origin_label, top_n=8, min_coverage=0.5):
    df = df.copy()
    coverage = df[benchmarks].notna().mean(axis=1)
    df = df[coverage >= min_coverage]

    if origins is not None:
        origin_col = "Origin" if "Origin" in df.columns else "Country"
        df = df[df[origin_col].isin(origins)]

    y_raw = df[target_col].astype(object).fillna("Unknown").astype(str)
    top_classes = set(y_raw.value_counts().head(top_n).index)
    mask = y_raw.isin(top_classes)
    y_raw, df = y_raw[mask], df[mask]

    cc = y_raw.value_counts()
    valid = set(cc[cc >= 3].index)
    mask2 = y_raw.isin(valid)
    y_raw, df = y_raw[mask2], df[mask2]

    if len(y_raw) < 8 or y_raw.nunique() < 2:
        return None

    X = df[benchmarks].values.astype(float)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    y = y_raw.values
    labels = sorted(y_raw.unique().tolist())

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42)

    clf_map = {
        "svm": (SVC(kernel="rbf", C=1.0, random_state=42, probability=True), "SVM (RBF)"),
        "dt":  (DecisionTreeClassifier(max_depth=5, random_state=42), "Decision Tree"),
        "lr":  (LogisticRegression(max_iter=2000, random_state=42), "Logistic Regression"),
    }
    clf, clf_name = clf_map[classifier]
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    # ROC curves
    y_te_bin = label_binarize(y_te, classes=labels)
    roc = {}
    if len(labels) == 2:
        prob_col = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        fpr, tpr, _ = roc_curve(y_te_bin[:, 0], prob_col)
        roc_auc = float(auc(fpr, tpr))
        roc[labels[1]] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auc": roc_auc,
            "label": f"{labels[1]} vs {labels[0]}",
        }
    else:
        for i, lbl in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_te_bin[:, i], y_prob[:, i])
            roc_auc = float(auc(fpr, tpr))
            roc[lbl] = {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
                "auc": roc_auc,
                "label": lbl,
            }

    macro_auc = float(np.mean([v["auc"] for v in roc.values()]))

    importances = None
    if classifier == "dt":
        importances = {b: float(v) for b, v in zip(benchmarks, clf.feature_importances_)}
    elif classifier == "lr" and hasattr(clf, "coef_"):
        importances = {b: float(v) for b, v in zip(benchmarks, np.abs(clf.coef_).mean(axis=0))}

    return {
        "accuracy": acc,
        "clf_name": clf_name,
        "target_label": target_label,
        "origin_label": origin_label,
        "n_test": len(X_te),
        "labels": labels,
        "cm": cm.tolist(),
        "roc": roc,
        "macro_auc": macro_auc,
        "importances": importances,
    }


# ── Dataset builders ──────────────────────────────────────────────────────────

def precompute_hf():
    print("=== HuggingFace Open LLM Leaderboard ===")
    df = pd.read_csv(DATA_DIR / "llm_benchmarks.csv")
    df["Company"] = df["Model"].str.split("/").str[0]
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df["Release Year"] = df["Release Date"].dt.year.astype("Int64")

    benchmarks = [
        "oll_BBH", "oll_GPQA", "oll_IFEval", "oll_MATH_lvl5", "oll_MMLU_PRO", "oll_MUSR",
    ]

    print("  PCA...")
    pca_2d = compute_pca(df, benchmarks, dims=2)
    pca_3d = compute_pca(df, benchmarks, dims=3)
    loadings = compute_loadings(df, benchmarks)

    print("  UMAP...")
    umap_data, umap_corr = compute_umap(df, benchmarks)

    print("  Scatter data...")
    scatter = compute_scatter_data(df, benchmarks)

    print("  Classifiers...")
    targets = {
        "origin":       ("Origin",       "Origin / Country"),
        "company":      ("Company",      "Company"),
        "release_year": ("Release Year", "Release Year"),
    }
    origin_filters = {
        "us_china": (["United States", "China"], "United States, China"),
        "all":      (None, "All"),
    }

    clf_results = {}
    for clf_key in ["svm", "dt", "lr"]:
        for tgt_key, (tgt_col, tgt_label) in targets.items():
            for orig_key, (orig_list, orig_label) in origin_filters.items():
                key = f"{clf_key}|{tgt_key}|{orig_key}"
                print(f"    {key}...", end=" ")
                try:
                    result = compute_classifier(
                        df, benchmarks, clf_key, tgt_col, tgt_label,
                        orig_list, orig_label,
                    )
                    if result:
                        clf_results[key] = result
                        print(f"acc={result['accuracy']:.1%}")
                    else:
                        print("SKIPPED (insufficient data)")
                except Exception as e:
                    print(f"SKIPPED ({e})")

    return {
        "name": "Open LLM Leaderboard",
        "benchmarks": benchmarks,
        "color_fields": ["Origin", "Company", "Release Year"],
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "pca_loadings": loadings,
        "umap_2d": umap_data,
        "umap_correlations": umap_corr,
        "scatter_data": scatter,
        "classifier_targets": {k: {"col": v[0], "label": v[1]} for k, v in targets.items()},
        "classifier_origin_filters": {k: {"label": v[1]} for k, v in origin_filters.items()},
        "classifier_results": clf_results,
    }


def precompute_eci():
    print("\n=== ECI (Epoch Composite Index) ===")
    df_scores = pd.read_csv(DATA_DIR / "eci_scores.csv")
    df_raw = pd.read_csv(DATA_DIR / "processed_data_for_eci.csv")

    df_wide = df_raw.pivot_table(
        index="Model", columns="benchmark", values="performance", aggfunc="first",
    )

    TOP_K = 4
    coverage = df_wide.notna().sum().sort_values(ascending=False)
    candidates = coverage.head(15).index.tolist()

    best_n, best_combo = 0, None
    for combo in combinations(candidates, TOP_K):
        n = df_wide[list(combo)].dropna().shape[0]
        if n > best_n:
            best_n = n
            best_combo = combo

    benchmarks = list(best_combo)
    print(f"  Benchmarks: {benchmarks} ({best_n} models)")

    df = df_wide[benchmarks].copy()
    df = df.merge(
        df_scores[["Model", "eci", "date", "Organization",
                    "Country (of organization)", "Accessibility group"]],
        left_index=True, right_on="Model", how="left",
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Release Year"] = df["date"].dt.year.astype("Int64")
    df = df.rename(columns={
        "Country (of organization)": "Country",
        "Accessibility group": "Access",
    })
    df = df.dropna(subset=benchmarks).copy()

    print("  PCA...")
    pca_2d = compute_pca(df, benchmarks, dims=2, min_coverage=0)
    pca_3d = compute_pca(df, benchmarks, dims=3, min_coverage=0)
    loadings = compute_loadings(df, benchmarks, min_coverage=0)

    print("  UMAP...")
    umap_data, umap_corr = compute_umap(df, benchmarks, min_coverage=0)

    print("  Scatter data...")
    scatter = compute_scatter_data(df, benchmarks)

    print("  Classifiers...")
    targets = {
        "country":      ("Country",      "Country"),
        "organization": ("Organization", "Organization"),
    }
    origin_filters = {
        "us_china": (["United States of America", "China"],
                     "United States of America, China"),
        "all":      (None, "All"),
    }

    clf_results = {}
    for clf_key in ["svm", "dt", "lr"]:
        for tgt_key, (tgt_col, tgt_label) in targets.items():
            for orig_key, (orig_list, orig_label) in origin_filters.items():
                key = f"{clf_key}|{tgt_key}|{orig_key}"
                print(f"    {key}...", end=" ")
                try:
                    result = compute_classifier(
                        df, benchmarks, clf_key, tgt_col, tgt_label,
                        orig_list, orig_label, min_coverage=0,
                    )
                    if result:
                        clf_results[key] = result
                        print(f"acc={result['accuracy']:.1%}")
                    else:
                        print("SKIPPED (insufficient data)")
                except Exception as e:
                    print(f"SKIPPED ({e})")

    return {
        "name": "ECI (Epoch Composite Index)",
        "benchmarks": benchmarks,
        "color_fields": ["Country", "Organization", "Access", "Release Year"],
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "pca_loadings": loadings,
        "umap_2d": umap_data,
        "umap_correlations": umap_corr,
        "scatter_data": scatter,
        "classifier_targets": {k: {"col": v[0], "label": v[1]} for k, v in targets.items()},
        "classifier_origin_filters": {k: {"label": v[1]} for k, v in origin_filters.items()},
        "classifier_results": clf_results,
    }


def refresh_hf_data(arena: bool = False, enrich_dates: bool = False,
                    hf_token: str | None = None):
    """Download fresh HF benchmark data using data/scraper.py."""
    # Add data/ to path so we can import the scraper
    sys.path.insert(0, str(DATA_DIR.resolve()))
    import scraper as S

    print("=== Downloading fresh HuggingFace data ===")
    df = S.build_merged_table(
        include_arena=arena,
        enrich_dates=enrich_dates,
        hf_token=hf_token,
    )
    out_csv = DATA_DIR / "llm_benchmarks.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"  Wrote {len(df)} rows × {len(df.columns)} cols → {out_csv}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute visualization data.")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Download fresh HF leaderboard data before computing",
    )
    parser.add_argument(
        "--arena", action="store_true",
        help="Include Chatbot Arena Elo (only with --refresh)",
    )
    parser.add_argument(
        "--enrich-dates", action="store_true",
        help="Fill missing release dates via HF API (slow, only with --refresh)",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HF token for higher rate limits (only with --refresh)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    if args.refresh:
        refresh_hf_data(
            arena=args.arena,
            enrich_dates=args.enrich_dates,
            hf_token=args.hf_token,
        )

    hf = precompute_hf()
    out = OUT_DIR / "hf.json"
    print(f"\nWriting {out}...")
    with open(out, "w") as f:
        json.dump(hf, f, cls=NumpyEncoder)
    print(f"  {out.stat().st_size / 1024:.0f} KB")

    eci = precompute_eci()
    out = OUT_DIR / "eci.json"
    print(f"Writing {out}...")
    with open(out, "w") as f:
        json.dump(eci, f, cls=NumpyEncoder)
    print(f"  {out.stat().st_size / 1024:.0f} KB")

    print("\nDone! Now run: quarto render main.qmd")
