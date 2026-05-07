"""
OpenEvals-first benchmark table: load OpenEvals, then left-join Open LLM Leaderboard.

Every output row is an OpenEvals model; OLL benchmark and date columns are filled only
when that Hugging Face id appears on the Open LLM Leaderboard.

Reuses loaders and helpers from scraper.py (same column names as llm_benchmarks.csv).

  python scrape_openevals.py -o llm_benchmarks_openevals.csv
  python scrape_openevals.py --enrich-dates --arena
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import scraper as S


def build_openevals_left_join(
    include_arena: bool,
    enrich_dates: bool,
    hf_token: str | None,
) -> pd.DataFrame:
    oll = S.load_open_llm_leaderboard()
    oe = S.load_openevals()

    oll_keep = ["fullname", "Upload To Hub Date"] + [
        c for c in S.OLL_BENCHMARK_COLS if c in oll.columns
    ]
    oll_sub = oll[[c for c in oll_keep if c in oll.columns]].copy()

    oe_cols = ["model_name", "model_id"] + [
        c for c in S.OE_SCORE_COLS if c in oe.columns
    ]
    oe_sub = oe[[c for c in oe_cols if c in oe.columns]].copy()

    merged = oe_sub.merge(
        oll_sub,
        left_on="model_name",
        right_on="fullname",
        how="left",
    )

    merged["Model"] = merged["model_name"]
    merged = merged.drop(columns=["fullname", "model_name"], errors="ignore")

    merged["Release Date"] = merged["Upload To Hub Date"].apply(S.parse_release_cell)
    merged = merged.drop(columns=["Upload To Hub Date"], errors="ignore")

    merged["Origin"] = merged["Model"].apply(S.infer_origin)

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
    merged = merged.rename(
        columns={k: v for k, v in oll_rename.items() if k in merged.columns}
    )

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
    merged = merged.rename(
        columns={k: v for k, v in oe_rename.items() if k in merged.columns}
    )

    merged = merged.drop(columns=["model_id"], errors="ignore")

    front = ["Model", "Release Date", "Origin"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + sorted(rest)]

    if include_arena:
        arena = S.fetch_arena_text_elo()
        merged = S.attach_arena_elo(merged, arena)

    if enrich_dates:
        merged = S.enrich_missing_dates(merged, token=hf_token)

    return merged.sort_values("Model", na_position="last").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenEvals rows with Open LLM Leaderboard columns left-joined."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_benchmarks_openevals.csv",
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

    df = build_openevals_left_join(
        include_arena=args.arena,
        enrich_dates=args.enrich_dates,
        hf_token=args.hf_token,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
