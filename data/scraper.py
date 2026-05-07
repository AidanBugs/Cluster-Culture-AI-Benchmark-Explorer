"""
Aggregate public LLM benchmark leaderboards into one CSV.

Data sources:
  - Hugging Face Open LLM Leaderboard (IFEval, BBH, MATH Lvl 5, GPQA, MUSR, MMLU-PRO, …)
  - OpenEvals multi-benchmark table (AIME, GPQA, GSM8K, MMLU-Pro, SWE-bench, …)
  - Optional: LMSYS Chatbot Arena (text) Elo via a public mirror API

Install: pip install pandas pyarrow huggingface_hub requests

Example:
  python scraper.py -o llm_benchmarks.csv
  python scraper.py -o out.csv --enrich-dates --arena
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OPENEVALS_PARQUET = (
    "hf://datasets/OpenEvals/leaderboard-data/data/train-00000-of-00001.parquet"
)
OLL_REPO = "open-llm-leaderboard/contents"
OLL_FILE = "data/train-00000-of-00001.parquet"

ARENA_API = "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard"

# HF org / namespace → country (heuristic; expand as needed)
NAMESPACE_ORIGIN: dict[str, str] = {
    "Llama": "United States",
    "gemma": "United States",
    "gemini": "United States",
    "mistralai": "France",
    "mistral-community": "France",
    "google": "United States",
    "deepseek": "China",
    "Qwen": "China",
    "01-ai": "China",
    "microsoft": "United States",
    "openai": "United States",
    "anthropic": "United States",
    "HuggingFaceTB": "United States",
    "tiiuae": "United Arab Emirates",
    "internlm": "China",
    "THUDM": "China",
    "stabilityai": "United Kingdom",
    "CohereForAI": "Canada",
    "amazon": "United States",
    "nvidia": "United States",
    "NVIDIA": "United States",
    "allenai": "United States",
    "EleutherAI": "United States",
    "bigscience": "Multinational",
    "togethercomputer": "United States",
    "moonshotai": "China",
    "MiniMaxAI": "China",
    "minimaxai": "China",
    "zai-org": "China",
    "openbmb": "China",
    "meituan-longcat": "China",
    "LiquidAI": "United States",
    "arcee-ai": "United States",
    "SWE-Lego": "United States",
    "GAIR": "China",
    "mlx-community": "United States",
    "SnurfyAI": "United States",
    "arcee": "United States",
    "upstage": "South Korea",
    "naver": "South Korea",
    "NousResearch": "United States",
    "HuggingFaceH4": "United States",
    "facebook": "United States",
    "ibm": "United States",
    "granite": "United States",
    "apple": "United States",
}

OLL_BENCHMARK_COLS = [
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
    "Average ⬆️",
    "CO₂ cost (kg)",
]

OE_SCORE_COLS = [
    "aime2026_score",
    "evasionBench_score",
    "gpqa_score",
    "gsm8k_score",
    "hle_score",
    "hmmt2026_score",
    "mmluPro_score",
    "olmOcr_score",
    "swePro_score",
    "sweVerified_score",
    "terminalBench_score",
    "aggregate_score",
    "coverage_count",
    "coverage_percent",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def infer_origin(hf_model_id: str) -> str:
    if not hf_model_id:
        return ""

    for key in NAMESPACE_ORIGIN.keys():
        if key.lower() in hf_model_id.lower():
            return NAMESPACE_ORIGIN[key]
    return ""


def parse_release_cell(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.replace("+00:00", ""), fmt).date().isoformat()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def load_open_llm_leaderboard() -> pd.DataFrame:
    path = hf_hub_download(repo_id=OLL_REPO, repo_type="dataset", filename=OLL_FILE)
    df = pd.read_parquet(path)
    if "Submission Date" in df.columns:
        df = df.assign(
            _sub=pd.to_datetime(df["Submission Date"], errors="coerce", utc=True)
        )
        df = df.sort_values("_sub", ascending=False)
    else:
        df = df.sort_values(
            "Average ⬆️", ascending=False, na_position="last"
        )
    df = df.drop_duplicates(subset=["fullname"], keep="first")
    return df.drop(columns=["_sub"], errors="ignore")


def load_openevals() -> pd.DataFrame:
    return pd.read_parquet(OPENEVALS_PARQUET)


def fetch_arena_text_elo() -> pd.DataFrame:
    r = requests.get(f"{ARENA_API}?name=text", timeout=60)
    r.raise_for_status()
    rows = r.json().get("models") or []
    return pd.DataFrame(
        [
            {
                "arena_model_slug": row.get("model"),
                "arena_vendor": row.get("vendor"),
                "arena_text_elo": row.get("score"),
                "arena_ci": row.get("ci"),
                "arena_votes": row.get("votes"),
            }
            for row in rows
        ]
    )


def attach_arena_elo(df: pd.DataFrame, arena: pd.DataFrame) -> pd.DataFrame:
    """Best-effort match: Arena slugs rarely equal HF ids; use slug containment."""

    if arena.empty:
        df["arena_text_elo"] = pd.NA
        return df

    slugs = []
    for _, r in arena.iterrows():
        m = r.get("arena_model_slug")
        if not m:
            continue
        slugs.append((_slug(str(m)), float(r["arena_text_elo"])))

    elos = []
    for mid in df["Model"]:
        if not isinstance(mid, str) or "/" not in mid:
            elos.append(pd.NA)
            continue
        repo = mid.split("/", 1)[1]
        cand = _slug(repo)
        best = None
        for am_slug, elo in slugs:
            if am_slug and (am_slug in cand or cand in am_slug):
                best = elo
                break
        elos.append(best if best is not None else pd.NA)

    out = df.copy()
    out["arena_text_elo"] = elos
    return out


def enrich_missing_dates(
    df: pd.DataFrame,
    token: str | None,
    delay_s: float = 0.15,
) -> pd.DataFrame:
    api = HfApi(token=token)
    out = df.copy()
    mask = out["Release Date"].isna() | (out["Release Date"].astype(str) == "")
    to_fix = out.loc[mask, "Model"].dropna().unique()
    cache: dict[str, str | None] = {}

    for mid in to_fix:
        if not isinstance(mid, str) or "/" not in mid:
            cache[mid] = None
            continue
        try:
            info = api.model_info(mid)
            ts = getattr(info, "created_at", None) or getattr(info, "last_modified", None)
            if ts is not None:
                if hasattr(ts, "date"):
                    cache[mid] = ts.date().isoformat()
                else:
                    cache[mid] = parse_release_cell(ts)
            else:
                cache[mid] = None
        except Exception:
            cache[mid] = None
        time.sleep(delay_s)

    def fill(row):
        if row["Release Date"] and str(row["Release Date"]).strip():
            return row["Release Date"]
        return cache.get(row["Model"])

    out["Release Date"] = out.apply(fill, axis=1)
    return out


def build_merged_table(
    include_arena: bool,
    enrich_dates: bool,
    hf_token: str | None,
) -> pd.DataFrame:
    oll = load_open_llm_leaderboard()
    oe = load_openevals()

    oll_keep = ["fullname", "Upload To Hub Date"] + [
        c for c in OLL_BENCHMARK_COLS if c in oll.columns
    ]
    oll_sub = oll[[c for c in oll_keep if c in oll.columns]].copy()

    oe_cols = ["model_name", "model_id"] + [c for c in OE_SCORE_COLS if c in oe.columns]
    oe_sub = oe[[c for c in oe_cols if c in oe.columns]].copy()

    merged = oll_sub.merge(
        oe_sub,
        left_on="fullname",
        right_on="model_name",
        how="outer",
        indicator=True,
    )

    merged["Model"] = merged["fullname"].fillna(merged["model_name"])
    merged = merged.drop(columns=["fullname", "model_name"], errors="ignore")

    rel_oll = merged["Upload To Hub Date"].apply(parse_release_cell)
    merged["Release Date"] = rel_oll
    merged = merged.drop(columns=["Upload To Hub Date"], errors="ignore")

    merged["Origin"] = merged["Model"].apply(infer_origin)

    oll_rename = {
        "IFEval": "oll_IFEval",
        "BBH": "oll_BBH",
        "MATH Lvl 5": "oll_MATH_lvl5",
        "GPQA": "oll_GPQA",
        "MUSR": "oll_MUSR",
        "MMLU-PRO": "oll_MMLU_PRO",
        "Average ⬆️": "oll_average",
        "CO₂ cost (kg)": "oll_CO2_kg",
    }
    merged = merged.rename(columns={k: v for k, v in oll_rename.items() if k in merged.columns})

    oe_rename = {
        "aime2026_score": "oe_aime2026",
        "evasionBench_score": "oe_evasionBench",
        "gpqa_score": "oe_gpqa",
        "gsm8k_score": "oe_gsm8k",
        "hle_score": "oe_hle",
        "hmmt2026_score": "oe_hmmt2026",
        "mmluPro_score": "oe_mmlu_pro",
        "olmOcr_score": "oe_olmOcr",
        "swePro_score": "oe_swePro",
        "sweVerified_score": "oe_sweVerified",
        "terminalBench_score": "oe_terminalBench",
        "aggregate_score": "oe_aggregate",
        "coverage_count": "oe_coverage_count",
        "coverage_percent": "oe_coverage_percent",
    }
    merged = merged.rename(columns={k: v for k, v in oe_rename.items() if k in merged.columns})

    drop_cols = [c for c in ("_merge", "model_id") if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    front = ["Model", "Release Date", "Origin"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + sorted(rest)]

    if include_arena:
        arena = fetch_arena_text_elo()
        merged = attach_arena_elo(merged, arena)

    if enrich_dates:
        merged = enrich_missing_dates(merged, token=hf_token)

    return merged.sort_values("Model", na_position="last").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build combined LLM benchmark CSV.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_benchmarks.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--arena",
        action="store_true",
        help="Attach Chatbot Arena (text) Elo where a slug heuristic matches",
    )
    p.add_argument(
        "--enrich-dates",
        action="store_true",
        help="Fill missing release dates via Hugging Face model_info (slower)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        help="HF token for higher rate limits / private models (optional)",
    )
    args = p.parse_args()

    df = build_merged_table(
        include_arena=args.arena,
        enrich_dates=args.enrich_dates,
        hf_token=args.hf_token,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
