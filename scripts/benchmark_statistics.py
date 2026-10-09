import math
import statistics
from typing import TypedDict

class Distribution(TypedDict):
    sample_count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    p95: float
    standard_deviation: float
    coefficient_of_variation: float

def calculate_distribution(samples: list[float],) -> Distribution:
    if not samples or any(((not math.isfinite(sample,)) or (sample < 0) for sample in samples),):
        raise ValueError("Failed to summarize empty, negative, or nonfinite benchmark samples.",)

    ordered_samples = sorted(samples,)
    percentile_index = (len(samples,) - 1) * 0.95
    lower_index = math.floor(percentile_index,)
    upper_index = math.ceil(percentile_index,)
    fraction = percentile_index - lower_index
    p95 = ordered_samples[lower_index] + fraction * (ordered_samples[upper_index] - ordered_samples[lower_index])
    mean = statistics.mean(samples,)
    standard_deviation = statistics.stdev(samples,) if len(samples,) > 1 else 0.0

    return {
        "sample_count": len(samples,),
        "minimum": ordered_samples[0],
        "maximum": ordered_samples[-1],
        "mean": mean,
        "median": statistics.median(samples,),
        "p95": p95,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean if mean else 0.0,
    }

class Sample(TypedDict, total=False):
    suite: str
    name: str
    model: str
    phase: str
    sample_index: int
    operation_count: int
    elapsed_nanoseconds: int
    node_count: int
    record_count: int
    epoch_count: int
    cpu_seconds: float | None
    peak_rss_bytes: int | None
    checksum: int
    best_move: str
    completed_depth_plies: int
    score_centipawns: int
    hash_mib: int
    network_fingerprint: int
    overshoot_nanoseconds: int

class Measurement(TypedDict):
    suite: str
    name: str
    model: str
    metric: str
    distribution: Distribution

def summarize_samples(samples: list[Sample],) -> list[Measurement]:
    groups: dict[
        tuple[
            str,
            str,
            str,
            str,
        ],
        list[float],
    ] = {}

    for sample in samples:
        if sample.get("phase",) != "measured":
            continue

        elapsed_nanoseconds = sample["elapsed_nanoseconds"]
        operation_count = sample["operation_count"]

        if (elapsed_nanoseconds <= 0) or (operation_count <= 0):
            raise ValueError("Failed to summarize a benchmark with nonpositive duration or operations.",)

        metrics = {"nanoseconds_per_operation": elapsed_nanoseconds / operation_count,}

        if "node_count" in sample:
            metrics["nodes_per_second"] = sample["node_count"] * 1_000_000_000 / elapsed_nanoseconds

        if "record_count" in sample:
            metrics["records_per_second"] = sample["record_count"] * sample["epoch_count"] * 1_000_000_000 / elapsed_nanoseconds

        for metric in ["cpu_seconds", "peak_rss_bytes", "overshoot_nanoseconds",]:
            if sample.get(metric,) is not None:
                metrics[metric] = float(sample[metric],)

        for metric, measurement in metrics.items():
            key = (sample["suite"], sample["name"], sample["model"], metric,)
            groups.setdefault(
                key,
                [],
            ).append(measurement,)

    return [
        {
            "suite": key[0],
            "name": key[1],
            "model": key[2],
            "metric": key[3],
            "distribution": calculate_distribution(measurements,),
        }
        for key, measurements in sorted(groups.items(),)
    ]

class Comparison(TypedDict):
    suite: str
    name: str
    model: str
    baseline_median_nanoseconds: float
    candidate_median_nanoseconds: float
    change_percent: float
    has_regressed: bool

def compare_measurements(
    candidate: list[Measurement],
    baseline: list[Measurement],
    maximum_regression_percent: float,
) -> list[Comparison]:
    if (not math.isfinite(maximum_regression_percent,)) or (maximum_regression_percent < 0):
        raise ValueError("Failed to compare benchmarks with an invalid regression threshold.",)

    candidate_latencies = {
        (measurement["suite"], measurement["name"], measurement["model"],): measurement["distribution"]["median"]
        for measurement in candidate if measurement["metric"] == "nanoseconds_per_operation"
    }
    baseline_latencies = {
        (measurement["suite"], measurement["name"], measurement["model"],): measurement["distribution"]["median"]
        for measurement in baseline if measurement["metric"] == "nanoseconds_per_operation"
    }

    if (not candidate_latencies) or (candidate_latencies.keys() != baseline_latencies.keys()):
        raise ValueError("Failed to compare benchmarks with mismatched workloads.",)

    comparisons = []

    for key, candidate_median in sorted(candidate_latencies.items(),):
        baseline_median = baseline_latencies[key]

        if (baseline_median <= 0) or (not math.isfinite(baseline_median,)) or (not math.isfinite(candidate_median,)):
            raise ValueError("Failed to compare benchmarks with invalid median durations.",)

        change_percent = (candidate_median / baseline_median - 1) * 100
        comparisons.append({
            "suite": key[0],
            "name": key[1],
            "model": key[2],
            "baseline_median_nanoseconds": baseline_median,
            "candidate_median_nanoseconds": candidate_median,
            "change_percent": change_percent,
            "has_regressed": change_percent > maximum_regression_percent,
        },)

    return comparisons
