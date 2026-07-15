"""Compute baseline Shared Landscapes summaries from a DGG hexagon CSV.

The script reads large CSV files in chunks, applies the configured baseline
definition, and writes global, country, region, biome, and model-agreement
tables. It intentionally contains no report or figure generation code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_HEX_AREA_KM2 = 1.18491


def resolve(path: str | Path) -> Path:
    """Resolve a relative configuration path from this code package."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PACKAGE_ROOT / candidate


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write reproducible outputs with the project-standard four decimals."""
    output = frame.copy()
    float_columns = output.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_columns):
        output[float_columns] = output[float_columns].round(4)
    output.to_csv(path, index=False)


def wl_models() -> list[dict]:
    """Return the fixed WL1-WL32 Shared Landscapes definition set."""
    return [
        {"id": "WL1", "nonhabitat_threshold": 0.0, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL2", "nonhabitat_threshold": 0.001, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL3", "nonhabitat_threshold": 0.005, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL4", "nonhabitat_threshold": 0.01, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL5", "nonhabitat_threshold": 0.0, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
        {"id": "WL6", "nonhabitat_threshold": 0.001, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
        {"id": "WL7", "nonhabitat_threshold": 0.005, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
        {"id": "WL8", "nonhabitat_threshold": 0.01, "population_operator": ">", "population_threshold": 0.0, "relation": "AND"},
        {"id": "WL9", "nonhabitat_threshold": 0.0, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL10", "nonhabitat_threshold": 0.001, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL11", "nonhabitat_threshold": 0.005, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL12", "nonhabitat_threshold": 0.01, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL13", "nonhabitat_threshold": 0.0, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
        {"id": "WL14", "nonhabitat_threshold": 0.001, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
        {"id": "WL15", "nonhabitat_threshold": 0.005, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
        {"id": "WL16", "nonhabitat_threshold": 0.01, "population_operator": ">", "population_threshold": 0.0, "relation": "OR"},
        {"id": "WL17", "nonhabitat_threshold": 0.0, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
        {"id": "WL18", "nonhabitat_threshold": 0.001, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
        {"id": "WL19", "nonhabitat_threshold": 0.005, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
        {"id": "WL20", "nonhabitat_threshold": 0.01, "population_operator": ">=", "population_threshold": 1.0, "relation": "OR"},
        {"id": "WL21", "nonhabitat_threshold": None, "population_operator": ">", "population_threshold": 0.0, "relation": "POPULATION"},
        {"id": "WL22", "nonhabitat_threshold": None, "population_operator": ">=", "population_threshold": 1.0, "relation": "POPULATION"},
        {"id": "WL23", "nonhabitat_threshold": 0.02, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL24", "nonhabitat_threshold": 0.05, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL25", "nonhabitat_threshold": 0.10, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL26", "nonhabitat_threshold": 0.25, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL27", "nonhabitat_threshold": 0.50, "population_operator": ">=", "population_threshold": 1.0, "relation": "AND"},
        {"id": "WL28", "nonhabitat_threshold": 0.02, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL29", "nonhabitat_threshold": 0.05, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL30", "nonhabitat_threshold": 0.10, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL31", "nonhabitat_threshold": 0.25, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
        {"id": "WL32", "nonhabitat_threshold": 0.50, "population_operator": None, "population_threshold": None, "relation": "NONHABITAT"},
    ]


def nonhabitat_rule(model: dict) -> str:
    threshold = model["nonhabitat_threshold"]
    if threshold is None:
        return "N/A"
    return ">0" if threshold == 0 else f">{threshold:.1%}"


def population_rule(model: dict) -> str:
    if model["population_operator"] is None:
        return "N/A"
    return f"{model['population_operator']}{model['population_threshold']:g}"


def working_mask(model: dict, land_positive: np.ndarray, nonhabitat_fraction: np.ndarray, population: np.ndarray) -> np.ndarray:
    """Classify land-containing hexagons under one WL rule."""
    if model["nonhabitat_threshold"] is None:
        nonhabitat_condition = np.zeros_like(land_positive, dtype=bool)
    else:
        nonhabitat_condition = nonhabitat_fraction > model["nonhabitat_threshold"]

    if model["population_operator"] is None:
        return land_positive & nonhabitat_condition

    if model["population_operator"] == ">":
        population_condition = population > model["population_threshold"]
    elif model["population_operator"] == ">=":
        population_condition = population >= model["population_threshold"]
    else:
        raise ValueError(f"Unsupported population operator: {model['population_operator']}")
    population_condition = np.nan_to_num(population_condition, nan=False).astype(bool)

    if model["relation"] == "AND":
        return land_positive & nonhabitat_condition & population_condition
    if model["relation"] == "OR":
        return land_positive & (nonhabitat_condition | population_condition)
    if model["relation"] == "POPULATION":
        return land_positive & population_condition
    raise ValueError(f"Unsupported WL relation: {model['relation']}")


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=float), np.nan),
        where=np.asarray(denominator, dtype=float) != 0,
    )


@dataclass
class GroupAccumulator:
    """Accumulate baseline and sensitivity areas for categorical CSV fields."""

    keep_sensitivity: bool
    n_models: int
    n_scenarios: int
    mapping: dict[str, int]
    land: np.ndarray
    habitat: np.ndarray
    working_land: np.ndarray
    working_habitat: np.ndarray
    shl_land: np.ndarray
    shl_habitat: np.ndarray
    n_hex: np.ndarray
    model_working_land: np.ndarray | None
    scenario_shl_land: np.ndarray | None

    @classmethod
    def create(cls, n_models: int, n_scenarios: int, keep_sensitivity: bool) -> "GroupAccumulator":
        return cls(
            keep_sensitivity=keep_sensitivity,
            n_models=n_models,
            n_scenarios=n_scenarios,
            mapping={},
            land=np.zeros(0, dtype=float),
            habitat=np.zeros(0, dtype=float),
            working_land=np.zeros(0, dtype=float),
            working_habitat=np.zeros(0, dtype=float),
            shl_land=np.zeros(0, dtype=float),
            shl_habitat=np.zeros(0, dtype=float),
            n_hex=np.zeros(0, dtype=float),
            model_working_land=np.zeros((0, n_models), dtype=float) if keep_sensitivity else None,
            scenario_shl_land=np.zeros((0, n_scenarios), dtype=float) if keep_sensitivity else None,
        )

    def ensure_size(self, n_groups: int) -> None:
        old_size = len(self.land)
        if n_groups <= old_size:
            return
        pad = n_groups - old_size
        for attribute in ["land", "habitat", "working_land", "working_habitat", "shl_land", "shl_habitat", "n_hex"]:
            setattr(self, attribute, np.pad(getattr(self, attribute), (0, pad)))
        if self.model_working_land is not None:
            self.model_working_land = np.pad(self.model_working_land, ((0, pad), (0, 0)))
        if self.scenario_shl_land is not None:
            self.scenario_shl_land = np.pad(self.scenario_shl_land, ((0, pad), (0, 0)))

    def encode(self, values: pd.Series) -> np.ndarray:
        cleaned = values.astype("string").str.strip()
        valid = cleaned.notna() & (cleaned != "") & (cleaned.str.lower() != "unassigned")
        for value in pd.unique(cleaned[valid]):
            if value not in self.mapping:
                self.mapping[value] = len(self.mapping)
        self.ensure_size(len(self.mapping))
        return cleaned.map(self.mapping).fillna(-1).to_numpy(dtype=int)

    def add_baseline(self, codes: np.ndarray, land: np.ndarray, habitat: np.ndarray, working: np.ndarray, shl: np.ndarray) -> None:
        valid = codes >= 0
        if not np.any(valid):
            return
        n_groups = len(self.mapping)
        self.land += np.bincount(codes[valid], weights=land[valid], minlength=n_groups)[:n_groups]
        self.habitat += np.bincount(codes[valid], weights=habitat[valid], minlength=n_groups)[:n_groups]
        self.n_hex += np.bincount(codes[valid], minlength=n_groups)[:n_groups]
        self.working_land += np.bincount(codes[valid & working], weights=land[valid & working], minlength=n_groups)[:n_groups]
        self.working_habitat += np.bincount(codes[valid & working], weights=habitat[valid & working], minlength=n_groups)[:n_groups]
        self.shl_land += np.bincount(codes[valid & shl], weights=land[valid & shl], minlength=n_groups)[:n_groups]
        self.shl_habitat += np.bincount(codes[valid & shl], weights=habitat[valid & shl], minlength=n_groups)[:n_groups]

    def add_model_working(self, codes: np.ndarray, model_index: int, working: np.ndarray, land: np.ndarray) -> None:
        if self.model_working_land is None:
            return
        valid = (codes >= 0) & working
        if np.any(valid):
            self.model_working_land[:, model_index] += np.bincount(codes[valid], weights=land[valid], minlength=len(self.mapping))[:len(self.mapping)]

    def add_scenario_shl(self, codes: np.ndarray, scenario_index: int, shl: np.ndarray, land: np.ndarray) -> None:
        if self.scenario_shl_land is None:
            return
        valid = (codes >= 0) & shl
        if np.any(valid):
            self.scenario_shl_land[:, scenario_index] += np.bincount(codes[valid], weights=land[valid], minlength=len(self.mapping))[:len(self.mapping)]

    def summary(self, sample_fraction: float, minimum_full_land_km2: float = 0.0) -> pd.DataFrame:
        labels = np.empty(len(self.mapping), dtype=object)
        for label, index in self.mapping.items():
            labels[index] = label
        result = pd.DataFrame({
            "group": labels,
            "land_area_km2": self.land,
            "estimated_full_land_km2": self.land / sample_fraction,
            "working_land_area_km2": self.working_land,
            "supporting_land_area_km2": self.land - self.working_land,
            "total_habitat_area_km2": self.habitat,
            "habitat_in_working_area_km2": self.working_habitat,
            "habitat_in_supporting_area_km2": self.habitat - self.working_habitat,
            "shl_land_area_km2": self.shl_land,
            "sil_land_area_km2": self.working_land - self.shl_land,
            "shl_habitat_area_km2": self.shl_habitat,
            "sil_habitat_area_km2": self.working_habitat - self.shl_habitat,
            "n_hex": self.n_hex.astype(int),
        })
        result["share_land_working"] = safe_divide(result["working_land_area_km2"], result["land_area_km2"])
        result["share_land_supporting"] = safe_divide(result["supporting_land_area_km2"], result["land_area_km2"])
        result["share_habitat_in_working"] = safe_divide(result["habitat_in_working_area_km2"], result["total_habitat_area_km2"])
        result["share_habitat_in_supporting"] = safe_divide(result["habitat_in_supporting_area_km2"], result["total_habitat_area_km2"])
        result["share_shl_land"] = safe_divide(result["shl_land_area_km2"], result["land_area_km2"])
        result["share_shl_within_working"] = safe_divide(result["shl_land_area_km2"], result["working_land_area_km2"])
        result["share_sil_within_working"] = safe_divide(result["sil_land_area_km2"], result["working_land_area_km2"])
        result = result[result["estimated_full_land_km2"] >= minimum_full_land_km2].copy()
        return result.sort_values("land_area_km2", ascending=False)


def load_country_lookup(config: dict) -> dict[str, str]:
    mapping_path = config.get("country_name_mapping_csv")
    if not mapping_path:
        return {}
    mapping = pd.read_csv(resolve(mapping_path), dtype={"ISO3": "string", "country": "string"})
    mapping["ISO3"] = mapping["ISO3"].astype("string").str.upper().str.strip()
    mapping["country"] = mapping["country"].astype("string").str.strip()
    mapping = mapping.dropna(subset=["ISO3", "country"]).drop_duplicates(subset=["ISO3"], keep="first")
    return dict(zip(mapping["ISO3"], mapping["country"]))


def country_key(country: pd.Series, iso3: pd.Series | None, country_lookup: dict[str, str]) -> pd.Series:
    cleaned_country = country.astype("string").str.strip()
    if iso3 is None:
        return cleaned_country.where(cleaned_country.notna() & (cleaned_country != ""), pd.NA)
    cleaned_iso3 = iso3.astype("string").str.upper().str.strip().fillna("")
    mapped_country = cleaned_iso3.map(country_lookup).astype("string") if country_lookup else pd.Series(pd.NA, index=country.index, dtype="string")
    cleaned_country = mapped_country.where(mapped_country.notna() & (mapped_country != ""), cleaned_country)
    return (cleaned_country + "||" + cleaned_iso3).where(cleaned_country.notna() & (cleaned_country != ""), pd.NA)


def split_country_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parts = result["group"].astype("string").str.split(r"\|\|", n=1, expand=True)
    if parts.shape[1] == 2:
        result["group"] = parts[0]
        result.insert(1, "iso3", parts[1].replace("", pd.NA))
    return result


def model_agreement(country_accumulator: GroupAccumulator, models: list[dict], thresholds: list[float], baseline_threshold: float, country_minimum: float, sample_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate country SHL/working shares and Spearman agreement at baseline SHL."""
    names = np.empty(len(country_accumulator.mapping), dtype=object)
    for name, index in country_accumulator.mapping.items():
        names[index] = name
    eligible = country_accumulator.land / sample_fraction >= country_minimum
    rows = np.where(eligible)[0]
    scenario_ids = [f"{model['id']}_SHL{round(threshold * 100)}" for model in models for threshold in thresholds]
    shares_data: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(models):
        denominator = country_accumulator.model_working_land[rows, model_index]
        for threshold_index, _ in enumerate(thresholds):
            scenario_index = model_index * len(thresholds) + threshold_index
            numerator = country_accumulator.scenario_shl_land[rows, scenario_index]
            shares_data[scenario_ids[scenario_index]] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    shares = pd.DataFrame(shares_data, index=names[rows])
    shares.index.name = "country"
    baseline_columns = [f"{model['id']}_SHL{round(baseline_threshold * 100)}" for model in models]
    agreement = shares[baseline_columns].corr(method="spearman")
    agreement.index.name = "model"
    ranks = shares.rank(axis=0, ascending=False, method="average")
    ranks.index.name = "country"
    return shares, ranks, agreement


def validate_columns(input_csv: Path, columns: dict) -> None:
    available = set(pd.read_csv(input_csv, nrows=0).columns)
    required_keys = ["total", "land", "habitat", "nonhabitat", "population", "country", "region"]
    required = [columns[key] for key in required_keys]
    for optional_key in ["iso3", "biome", "hex_id"]:
        if columns.get(optional_key):
            required.append(columns[optional_key])
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"Input CSV is missing required configured columns: {', '.join(missing)}")


def run(config: dict) -> None:
    input_csv = resolve(config["input_csv"])
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    columns = config["columns"]
    validate_columns(input_csv, columns)

    models = wl_models()
    thresholds = config["shl_thresholds"]
    baseline_model = next(model for model in models if model["id"] == config["baseline"]["working_model"])
    baseline_threshold = float(config["baseline"]["shl_habitat_threshold"])
    if baseline_threshold not in thresholds:
        raise ValueError("The baseline SHL threshold must be one of shl_thresholds.")
    hex_area_km2 = float(config.get("hex_area_km2", DEFAULT_HEX_AREA_KM2))
    sample_fraction = float(config.get("sample_fraction", 1.0))
    country_minimum = float(config.get("country_min_land_area_km2", 0.0))
    country_lookup = load_country_lookup(config)

    scenario_count = len(models) * len(thresholds)
    country_accumulator = GroupAccumulator.create(len(models), scenario_count, keep_sensitivity=True)
    region_accumulator = GroupAccumulator.create(len(models), scenario_count, keep_sensitivity=False)
    biome_accumulator = GroupAccumulator.create(len(models), scenario_count, keep_sensitivity=False)
    total_land = 0.0
    total_habitat = 0.0
    global_working_land = np.zeros(len(models), dtype=float)
    global_working_habitat = np.zeros(len(models), dtype=float)
    global_shl_land = np.zeros((len(models), len(thresholds)), dtype=float)
    global_shl_habitat = np.zeros((len(models), len(thresholds)), dtype=float)
    diagnostics = {"rows": 0, "missing_population": 0, "missing_country": 0, "missing_region": 0, "missing_iso3": 0, "missing_biome": 0, "nonhabitat_above_land_rows": 0}

    use_columns = [columns[key] for key in ["total", "land", "habitat", "nonhabitat", "population", "country", "region"]]
    for optional_key in ["iso3", "biome"]:
        if columns.get(optional_key):
            use_columns.append(columns[optional_key])
    dtype = {columns["country"]: "string", columns["region"]: "string"}
    if columns.get("iso3"):
        dtype[columns["iso3"]] = "string"
    if columns.get("biome"):
        dtype[columns["biome"]] = "string"

    for chunk_number, chunk in enumerate(pd.read_csv(input_csv, usecols=use_columns, chunksize=int(config.get("chunksize", 500_000)), dtype=dtype), start=1):
        diagnostics["rows"] += len(chunk)
        diagnostics["missing_population"] += int(chunk[columns["population"]].isna().sum())
        diagnostics["missing_country"] += int(chunk[columns["country"]].isna().sum())
        diagnostics["missing_region"] += int(chunk[columns["region"]].isna().sum())
        if columns.get("iso3"):
            diagnostics["missing_iso3"] += int(chunk[columns["iso3"]].isna().sum())
        if columns.get("biome"):
            diagnostics["missing_biome"] += int(chunk[columns["biome"]].isna().sum())

        n_total = pd.to_numeric(chunk[columns["total"]], errors="coerce").fillna(0).to_numpy(float)
        n_land = pd.to_numeric(chunk[columns["land"]], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
        n_habitat = pd.to_numeric(chunk[columns["habitat"]], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
        n_nonhabitat = pd.to_numeric(chunk[columns["nonhabitat"]], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
        population = pd.to_numeric(chunk[columns["population"]], errors="coerce").to_numpy(float)
        diagnostics["nonhabitat_above_land_rows"] += int(np.sum(n_nonhabitat > n_land))

        land_area = np.divide(n_land, n_total, out=np.zeros_like(n_land), where=n_total > 0) * hex_area_km2
        habitat_area = np.divide(n_habitat, n_total, out=np.zeros_like(n_habitat), where=n_total > 0) * hex_area_km2
        habitat_fraction = np.divide(n_habitat, n_land, out=np.full_like(n_land, np.nan), where=n_land > 0)
        nonhabitat_fraction = np.divide(n_nonhabitat, n_land, out=np.full_like(n_land, np.nan), where=n_land > 0)
        land_positive = land_area > 0
        total_land += np.nansum(land_area)
        total_habitat += np.nansum(habitat_area)

        country_codes = country_accumulator.encode(country_key(chunk[columns["country"]], chunk[columns["iso3"]] if columns.get("iso3") else None, country_lookup))
        region_codes = region_accumulator.encode(chunk[columns["region"]])
        biome_codes = biome_accumulator.encode(chunk[columns["biome"]]) if columns.get("biome") else None
        baseline_working = working_mask(baseline_model, land_positive, nonhabitat_fraction, population)
        baseline_shl = baseline_working & (habitat_fraction >= baseline_threshold)
        country_accumulator.add_baseline(country_codes, land_area, habitat_area, baseline_working, baseline_shl)
        region_accumulator.add_baseline(region_codes, land_area, habitat_area, baseline_working, baseline_shl)
        if biome_codes is not None:
            biome_accumulator.add_baseline(biome_codes, land_area, habitat_area, baseline_working, baseline_shl)

        for model_index, model in enumerate(models):
            working = working_mask(model, land_positive, nonhabitat_fraction, population)
            global_working_land[model_index] += np.nansum(land_area[working])
            global_working_habitat[model_index] += np.nansum(habitat_area[working])
            country_accumulator.add_model_working(country_codes, model_index, working, land_area)
            for threshold_index, threshold in enumerate(thresholds):
                shl = working & (habitat_fraction >= threshold)
                global_shl_land[model_index, threshold_index] += np.nansum(land_area[shl])
                global_shl_habitat[model_index, threshold_index] += np.nansum(habitat_area[shl])
                country_accumulator.add_scenario_shl(country_codes, model_index * len(thresholds) + threshold_index, shl, land_area)

        if chunk_number % int(config.get("progress_every_chunks", 20)) == 0:
            print(f"Processed {diagnostics['rows']:,} rows", flush=True)

    scenario_rows = []
    for model_index, model in enumerate(models):
        for threshold_index, threshold in enumerate(thresholds):
            working_land = global_working_land[model_index]
            working_habitat = global_working_habitat[model_index]
            shl_land = global_shl_land[model_index, threshold_index]
            shl_habitat = global_shl_habitat[model_index, threshold_index]
            scenario_rows.append({
                "wl_model": model["id"], "relation": model["relation"], "nonhabitat_cover_rule": nonhabitat_rule(model), "population_rule": population_rule(model),
                "shl_model": f"SHL{round(threshold * 100)}", "shl_habitat_threshold": threshold,
                "total_land_area_km2": total_land, "working_land_area_km2": working_land, "supporting_land_area_km2": total_land - working_land,
                "total_habitat_area_km2": total_habitat, "habitat_in_working_area_km2": working_habitat, "habitat_in_supporting_area_km2": total_habitat - working_habitat,
                "shl_land_area_km2": shl_land, "sil_land_area_km2": working_land - shl_land, "shl_habitat_area_km2": shl_habitat, "sil_habitat_area_km2": working_habitat - shl_habitat,
                "share_land_working": working_land / total_land, "share_land_supporting": (total_land - working_land) / total_land,
                "share_habitat_in_working": working_habitat / total_habitat, "share_habitat_in_supporting": (total_habitat - working_habitat) / total_habitat,
                "share_shl_land": shl_land / total_land, "share_shl_within_working": shl_land / working_land if working_land else np.nan,
                "share_sil_within_working": (working_land - shl_land) / working_land if working_land else np.nan,
            })
    scenario_results = pd.DataFrame(scenario_rows)
    baseline = scenario_results[(scenario_results["wl_model"] == baseline_model["id"]) & np.isclose(scenario_results["shl_habitat_threshold"], baseline_threshold)].iloc[0]

    country_summary = split_country_columns(country_accumulator.summary(sample_fraction, country_minimum))
    region_summary = region_accumulator.summary(sample_fraction)
    biome_summary = biome_accumulator.summary(sample_fraction) if columns.get("biome") else pd.DataFrame()
    country_shares, country_ranks, agreement = model_agreement(country_accumulator, models, thresholds, baseline_threshold, country_minimum, sample_fraction)
    country_shares = split_country_columns(country_shares.rename_axis("group").reset_index()).rename(columns={"group": "country"})
    country_ranks = split_country_columns(country_ranks.rename_axis("group").reset_index()).rename(columns={"group": "country"})

    global_summary = pd.DataFrame([
        ("Total land area", baseline["total_land_area_km2"], 1.0),
        ("Working Landscapes", baseline["working_land_area_km2"], baseline["share_land_working"]),
        ("Supporting Landscapes (SUL)", baseline["supporting_land_area_km2"], baseline["share_land_supporting"]),
        ("Total habitat area", baseline["total_habitat_area_km2"], baseline["total_habitat_area_km2"] / baseline["total_land_area_km2"]),
        ("Habitat in Working Landscapes", baseline["habitat_in_working_area_km2"], baseline["share_habitat_in_working"]),
        ("Habitat in SUL", baseline["habitat_in_supporting_area_km2"], baseline["share_habitat_in_supporting"]),
        ("Shared Landscapes (SHL)", baseline["shl_land_area_km2"], baseline["share_shl_land"]),
        ("SHL within Working Landscapes", baseline["shl_land_area_km2"], baseline["share_shl_within_working"]),
        ("Simplified Landscapes (SIL)", baseline["sil_land_area_km2"], baseline["share_sil_within_working"]),
    ], columns=["metric", "area_km2", "share"])

    write_csv(global_summary, output_dir / "global_summary.csv")
    write_csv(country_summary, output_dir / "country_summary.csv")
    write_csv(region_summary, output_dir / "region_summary.csv")
    write_csv(biome_summary, output_dir / "biome_summary.csv")
    write_csv(agreement.reset_index(), output_dir / "model_agreement.csv")
    write_csv(scenario_results, output_dir / "scenario_results.csv")
    write_csv(country_shares, output_dir / "country_shl_share_within_working_by_scenario.csv")
    write_csv(country_ranks, output_dir / "country_shl_rank_within_working_by_scenario.csv")
    write_csv(pd.DataFrame(list(diagnostics.items()), columns=["check", "value"]), output_dir / "qa_diagnostics.csv")
    write_csv(pd.DataFrame(models), output_dir / "working_landscape_definitions.csv")
    print(f"Completed. Outputs written to: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Shared Landscapes baseline summary tables from a hexagon CSV.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON configuration file.")
    args = parser.parse_args()
    run(read_config(args.config))


if __name__ == "__main__":
    main()
