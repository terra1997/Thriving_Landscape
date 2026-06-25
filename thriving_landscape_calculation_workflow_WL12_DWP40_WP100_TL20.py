from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Run definition: DWP40 + WL12 + TL20 + WorldPop 100 m
# ---------------------------------------------------------------------------

INPUT_CSV = ROOT / "data" / "GEE-export" / "DW_Feq_Thrd" / (
    "TL_2024_DW_pastureV1_treecrop_Thrd_40_merged.csv"
)
OUTPUT_DIR = ROOT / "results" / "WL12_DWP40_WP100_TL20" / "calculation_check_outputs"

JOIN_KEY = "seqnum16"
POPULATION_DENSITY_COL = "population_density_worldpop100"

COUNTRY_COL = "country"
REGION_COL = "region"
BIOME_COL = "potveg"
ANTHROME_COL = "anthrome"
ANTHROME_LEVEL_COL = "anthrome_level"

N_TOTAL_COL = "n_total"
N_WATER_COL = "n_water"
N_SNOW_COL = "n_snow"
N_NONHABITAT_COL = "n_anthro"

HEX_AREA_KM2 = 1.18491
SAMPLE_FRACTION = 0.01
COUNTRY_MIN_ESTIMATED_FULL_AREA_KM2 = 10_000

HEADLINE_WL_MODEL = "WL12"
HEADLINE_TL_THRESHOLD = 0.20

TL_THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.50]

WORKING_MODELS = [
    {"id": "WL1", "nonhabitat_threshold": 0.000, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
    {"id": "WL2", "nonhabitat_threshold": 0.001, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
    {"id": "WL3", "nonhabitat_threshold": 0.005, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
    {"id": "WL4", "nonhabitat_threshold": 0.010, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
    {"id": "WL5", "nonhabitat_threshold": 0.000, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
    {"id": "WL6", "nonhabitat_threshold": 0.001, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
    {"id": "WL7", "nonhabitat_threshold": 0.005, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
    {"id": "WL8", "nonhabitat_threshold": 0.010, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
    {"id": "WL9", "nonhabitat_threshold": 0.000, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
    {"id": "WL10", "nonhabitat_threshold": 0.001, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
    {"id": "WL11", "nonhabitat_threshold": 0.005, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
    {"id": "WL12", "nonhabitat_threshold": 0.010, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
    {"id": "WL13", "nonhabitat_threshold": 0.000, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
    {"id": "WL14", "nonhabitat_threshold": 0.001, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
    {"id": "WL15", "nonhabitat_threshold": 0.005, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
    {"id": "WL16", "nonhabitat_threshold": 0.010, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
    {"id": "WL17", "nonhabitat_threshold": 0.000, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
    {"id": "WL18", "nonhabitat_threshold": 0.001, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
    {"id": "WL19", "nonhabitat_threshold": 0.005, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
    {"id": "WL20", "nonhabitat_threshold": 0.010, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
]


def model_lookup() -> dict[str, dict]:
    return {model["id"]: model for model in WORKING_MODELS}


def population_rule(model: dict) -> str:
    if model["population_operator"] is None:
        return "N/A"
    return f"{model['population_operator']}{model['population_threshold']:g}"


def load_hex_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the merged hex table and calculate land/habitat areas and fractions."""
    required_cols = [
        JOIN_KEY,
        COUNTRY_COL,
        REGION_COL,
        BIOME_COL,
        ANTHROME_COL,
        ANTHROME_LEVEL_COL,
        POPULATION_DENSITY_COL,
        N_TOTAL_COL,
        N_WATER_COL,
        N_SNOW_COL,
        N_NONHABITAT_COL,
    ]
    df = pd.read_csv(path, usecols=required_cols)

    numeric_cols = [POPULATION_DENSITY_COL, N_TOTAL_COL, N_WATER_COL, N_SNOW_COL, N_NONHABITAT_COL]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_total = df[N_TOTAL_COL].fillna(0)
    n_land_raw = df[N_TOTAL_COL] - df[N_WATER_COL] - df[N_SNOW_COL]
    n_land = n_land_raw.clip(lower=0).fillna(0)
    n_nonhabitat = df[N_NONHABITAT_COL].clip(lower=0).fillna(0)
    n_habitat = (n_land - n_nonhabitat).clip(lower=0)
    valid_total = n_total > 0

    df["land_km2"] = np.where(valid_total, n_land / n_total * HEX_AREA_KM2, 0.0)
    df["habitat_km2"] = np.where(valid_total, n_habitat / n_total * HEX_AREA_KM2, 0.0)
    df["frac_habitat_land"] = np.where(n_land > 0, n_habitat / n_land, np.nan)
    df["frac_nonhabitat_land"] = np.where(n_land > 0, n_nonhabitat / n_land, np.nan)

    qa = pd.DataFrame(
        [
            ("hex_rows", len(df)),
            ("unique_join_keys", df[JOIN_KEY].nunique()),
            ("missing_population_density", df[POPULATION_DENSITY_COL].isna().sum()),
            (
                "missing_population_density_positive_land",
                (df[POPULATION_DENSITY_COL].isna() & (df["land_km2"] > 0)).sum(),
            ),
            ("negative_population_density", (df[POPULATION_DENSITY_COL] < 0).sum()),
            ("missing_country", df[COUNTRY_COL].isna().sum()),
            ("missing_region", df[REGION_COL].isna().sum()),
            ("missing_biome", df[BIOME_COL].isna().sum()),
            ("missing_anthrome", df[ANTHROME_COL].isna().sum()),
            ("negative_land_pixel_count_rows", (n_land_raw < 0).sum()),
            ("nonhabitat_pixel_count_above_land_rows", (n_nonhabitat > n_land).sum()),
        ],
        columns=["check", "value"],
    )
    return df, qa


def make_working_mask(df: pd.DataFrame, model: dict) -> pd.Series:
    """Classify each hex as working or non-working under one WL definition."""
    land = df["land_km2"].fillna(0) > 0
    relation = model["relation"].upper()

    nonhabitat = pd.Series(False, index=df.index)
    if model["nonhabitat_threshold"] is not None:
        nonhabitat = df["frac_nonhabitat_land"].fillna(-np.inf) > model["nonhabitat_threshold"]

    if model["population_operator"] is None:
        return land & nonhabitat

    population = df[POPULATION_DENSITY_COL]
    if model["population_operator"] == ">":
        population_condition = population > model["population_threshold"]
    elif model["population_operator"] == ">=":
        population_condition = population >= model["population_threshold"]
    else:
        raise ValueError(f"Unsupported population operator: {model['population_operator']}")
    population_condition = population_condition.fillna(False)

    if relation == "AND":
        return land & nonhabitat & population_condition
    if relation == "OR":
        return land & (nonhabitat | population_condition)
    if relation == "POPULATION":
        return land & population_condition
    raise ValueError(f"Unsupported relation for {model['id']}: {relation}")


def summarize_global(df: pd.DataFrame, working: pd.Series, tl_threshold: float) -> pd.DataFrame:
    thriving = working & (df["frac_habitat_land"] >= tl_threshold)
    supporting = (df["land_km2"] > 0) & ~working

    total_land = df["land_km2"].sum()
    working_land = df.loc[working, "land_km2"].sum()
    supporting_land = df.loc[supporting, "land_km2"].sum()
    total_habitat = df["habitat_km2"].sum()
    habitat_in_working = df.loc[working, "habitat_km2"].sum()
    habitat_elsewhere = df.loc[supporting, "habitat_km2"].sum()
    thriving_land = df.loc[thriving, "land_km2"].sum()

    return pd.DataFrame(
        [
            ("total_land_area_in_sample", total_land, 1.0, np.nan),
            ("working_landscapes", working_land, working_land / total_land, np.nan),
            ("supporting_elsewhere_landscapes", supporting_land, supporting_land / total_land, np.nan),
            ("total_habitat_area_in_sample", total_habitat, total_habitat / total_land, np.nan),
            ("habitat_in_working_landscapes", habitat_in_working, habitat_in_working / total_habitat, np.nan),
            ("habitat_in_supporting_elsewhere", habitat_elsewhere, habitat_elsewhere / total_habitat, np.nan),
            (
                f"thriving_land_working_and_TL{round(tl_threshold * 100)}",
                thriving_land,
                thriving_land / total_land,
                thriving_land / working_land,
            ),
        ],
        columns=["metric", "area_km2", "share_of_total_land_or_habitat", "share_of_working_land"],
    )


def summarize_by_group(
    df: pd.DataFrame,
    group_col: str,
    working: pd.Series,
    tl_threshold: float,
    drop_unassigned: bool = True,
) -> pd.DataFrame:
    """Calculate the same area and share metrics by country, region, biome, or anthrome."""
    thriving = working & (df["frac_habitat_land"] >= tl_threshold)
    temp = pd.DataFrame(
        {
            "group": df[group_col],
            "land_km2": df["land_km2"],
            "habitat_km2": df["habitat_km2"],
            "working_land_km2": np.where(working, df["land_km2"], 0.0),
            "working_habitat_km2": np.where(working, df["habitat_km2"], 0.0),
            "thriving_land_km2": np.where(thriving, df["land_km2"], 0.0),
            "n_hex": 1,
        }
    )
    if drop_unassigned:
        temp = temp[temp["group"].notna() & (temp["group"] != "Unassigned")]

    out = temp.groupby("group", as_index=False).sum(numeric_only=True)
    out["supporting_land_km2"] = out["land_km2"] - out["working_land_km2"]
    out["habitat_elsewhere_km2"] = out["habitat_km2"] - out["working_habitat_km2"]
    out["share_land_working"] = out["working_land_km2"] / out["land_km2"]
    out["share_land_supporting"] = out["supporting_land_km2"] / out["land_km2"]
    out["share_habitat_in_working"] = out["working_habitat_km2"] / out["habitat_km2"]
    out["share_habitat_elsewhere"] = out["habitat_elsewhere_km2"] / out["habitat_km2"]
    out["share_thriving_of_land"] = out["thriving_land_km2"] / out["land_km2"]
    out["share_thriving_within_working"] = out["thriving_land_km2"] / out["working_land_km2"]
    return out.sort_values("land_km2", ascending=False)


def summarize_country(df: pd.DataFrame, working: pd.Series, tl_threshold: float) -> pd.DataFrame:
    country = summarize_by_group(df, COUNTRY_COL, working, tl_threshold)
    country["estimated_full_land_km2"] = country["land_km2"] / SAMPLE_FRACTION
    country = country[country["estimated_full_land_km2"] >= COUNTRY_MIN_ESTIMATED_FULL_AREA_KM2].copy()
    return country.sort_values("estimated_full_land_km2", ascending=False)


def run_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all WL models crossed with TL thresholds and return global and country results."""
    total_land = df["land_km2"].sum()
    total_habitat = df["habitat_km2"].sum()
    global_rows = []
    country_share_by_scenario = {}
    country_rank_by_scenario = {}

    for model in WORKING_MODELS:
        working = make_working_mask(df, model)
        working_land = df.loc[working, "land_km2"].sum()
        supporting_land = total_land - working_land
        habitat_in_working = df.loc[working, "habitat_km2"].sum()
        habitat_elsewhere = total_habitat - habitat_in_working

        country_working = summarize_by_group(
            df,
            COUNTRY_COL,
            working,
            tl_threshold=0.20,
        )[["group", "land_km2", "working_land_km2"]].rename(columns={"group": "country"})
        country_working["estimated_full_land_km2"] = country_working["land_km2"] / SAMPLE_FRACTION
        country_working = country_working[
            country_working["estimated_full_land_km2"] >= COUNTRY_MIN_ESTIMATED_FULL_AREA_KM2
        ].copy()

        for tl_threshold in TL_THRESHOLDS:
            thriving = working & (df["frac_habitat_land"] >= tl_threshold)
            thriving_land = df.loc[thriving, "land_km2"].sum()
            scenario = f"{model['id']}_TL{round(tl_threshold * 100)}"

            global_rows.append(
                {
                    "scenario": scenario,
                    "wl_model": model["id"],
                    "wl_relation": model["relation"],
                    "wl_nonhabitat_threshold": model["nonhabitat_threshold"],
                    "wl_population_rule": population_rule(model),
                    "tl_habitat_threshold": tl_threshold,
                    "total_land_km2": total_land,
                    "working_land_km2": working_land,
                    "supporting_land_km2": supporting_land,
                    "total_habitat_km2": total_habitat,
                    "habitat_in_working_km2": habitat_in_working,
                    "habitat_elsewhere_km2": habitat_elsewhere,
                    "thriving_land_km2": thriving_land,
                    "share_land_working": working_land / total_land,
                    "share_habitat_in_working": habitat_in_working / total_habitat,
                    "share_thriving_of_total_land": thriving_land / total_land,
                    "share_thriving_within_working": thriving_land / working_land,
                }
            )

            country_thriving = summarize_by_group(
                df,
                COUNTRY_COL,
                working,
                tl_threshold,
            )[["group", "thriving_land_km2"]].rename(columns={"group": "country"})
            country = country_working.merge(country_thriving, on="country", how="outer")
            country["share_thriving_within_working"] = np.divide(
                country["thriving_land_km2"],
                country["working_land_km2"],
                out=np.full(len(country), np.nan),
                where=country["working_land_km2"].to_numpy(float) > 0,
            )
            shares = country.set_index("country")["share_thriving_within_working"]
            country_share_by_scenario[scenario] = shares
            country_rank_by_scenario[scenario] = shares.rank(ascending=False, method="average")

    global_sensitivity = pd.DataFrame(global_rows)
    country_shares = pd.DataFrame(country_share_by_scenario)
    country_ranks = pd.DataFrame(country_rank_by_scenario)
    return global_sensitivity, country_shares, country_ranks


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models = model_lookup()
    headline_working = make_working_mask(df, models[HEADLINE_WL_MODEL])

    summarize_global(df, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "01_global_summary_WL12_TL20.csv",
        index=False,
    )
    summarize_country(df, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "02_by_country_WL12_TL20.csv",
        index=False,
    )
    summarize_by_group(df, REGION_COL, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "03_by_region_WL12_TL20.csv",
        index=False,
    )
    summarize_by_group(df, BIOME_COL, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "04_by_biome_WL12_TL20.csv",
        index=False,
    )
    summarize_by_group(df, ANTHROME_COL, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "05_by_anthrome_WL12_TL20.csv",
        index=False,
    )
    summarize_by_group(df, ANTHROME_LEVEL_COL, headline_working, HEADLINE_TL_THRESHOLD).to_csv(
        output_dir / "06_by_anthrome_level_WL12_TL20.csv",
        index=False,
    )

    global_sensitivity, country_shares, country_ranks = run_sensitivity(df)
    global_sensitivity.to_csv(output_dir / "07_sensitivity_global_all_WL_TL.csv", index=False)
    country_shares.to_csv(output_dir / "08_sensitivity_country_thriving_share_within_working.csv")
    country_ranks.to_csv(output_dir / "09_sensitivity_country_rank_within_working.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculation-only Thriving Landscape workflow for DWP40 + WL12 + TL20 + WorldPop 100 m."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    df, qa = load_hex_data(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa.to_csv(args.output_dir / "00_input_data_QA.csv", index=False)
    write_outputs(df, args.output_dir)

    print(f"Input: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Headline model: {HEADLINE_WL_MODEL} + TL{round(HEADLINE_TL_THRESHOLD * 100)}")
    print(f"Sensitivity scenarios: {len(WORKING_MODELS) * len(TL_THRESHOLDS)}")


if __name__ == "__main__":
    main()
