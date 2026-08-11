"""Layer-wise profiling 결과 검증, 집계 및 저장."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ProfileConfig
from .layer_benchmark import LayerProfileRow
from .model_runtime import ModelArtifacts, RuntimeInfo
from .utils import print_section

RAW_COLUMNS = [
    "device",
    "phase",
    "prompt_tokens",
    "decode_step",
    "cache_tokens_before",
    "cache_tokens_after",
    "layer",
    "repeat",
    "latency_ms",
    "parameter_bytes",
    "parameter_mib",
    "activation_bytes",
    "activation_mib",
    "kv_cache_bytes",
    "kv_cache_mib",
]


@dataclass(frozen=True)
class LayerResultPaths:
    """Layer profiling으로 생성된 결과 파일 경로."""

    raw_csv: Path
    summary_csv: Path
    environment_json: Path


def _validate_raw_results(raw_results: pd.DataFrame, num_layers: int) -> None:
    """Layer 누락, latency, KV cache와 parameter 불변 조건을 검사한다."""

    if raw_results.empty:
        raise RuntimeError("Layer profiling 결과가 비어 있습니다.")
    if set(raw_results["phase"].unique()) != {"prefill", "decode"}:
        raise RuntimeError("phase에는 prefill과 decode 결과가 모두 있어야 합니다.")

    expected_layers = set(range(num_layers))
    measured_layers = set(raw_results["layer"].unique())
    if measured_layers != expected_layers:
        raise RuntimeError(
            f"측정 layer {sorted(measured_layers)}가 예상 layer {sorted(expected_layers)}와 다릅니다."
        )

    layer_counts = raw_results.groupby(
        ["phase", "prompt_tokens", "repeat", "decode_step"],
        dropna=False,
    )["layer"].nunique()
    if not (layer_counts == num_layers).all():
        raise RuntimeError("일부 forward에서 Transformer layer 측정치가 누락되었습니다.")

    if not raw_results["latency_ms"].map(math.isfinite).all():
        raise RuntimeError("Layer latency에 NaN 또는 infinity가 포함되었습니다.")
    if (raw_results["latency_ms"] < 0).any():
        raise RuntimeError("Layer latency에 음수가 포함되었습니다.")
    if (raw_results[["activation_bytes", "kv_cache_bytes"]] <= 0).any().any():
        raise RuntimeError("Activation 또는 KV-cache 크기가 0 이하인 row가 있습니다.")

    parameter_variants = raw_results.groupby("layer")["parameter_bytes"].nunique()
    if (parameter_variants != 1).any():
        raise RuntimeError("Layer parameter memory가 repeat에 따라 변했습니다.")

    cache_progress = (
        raw_results[
            [
                "prompt_tokens",
                "repeat",
                "layer",
                "cache_tokens_after",
                "kv_cache_bytes",
            ]
        ]
        .drop_duplicates()
        .sort_values(["prompt_tokens", "repeat", "layer", "cache_tokens_after"])
    )
    cache_growth = cache_progress.groupby(
        ["prompt_tokens", "repeat", "layer"]
    )["kv_cache_bytes"].diff()
    if (cache_growth.dropna() < 0).any():
        raise RuntimeError("Context가 증가하는 동안 layer KV-cache 크기가 감소했습니다.")


def summarize_layer_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Prefill은 prompt/layer, decode는 context/layer 단위로 집계한다."""

    prefill = raw_results[raw_results["phase"] == "prefill"]
    prefill_summary = (
        prefill.groupby(["phase", "prompt_tokens", "layer"], as_index=False)
        .agg(
            cache_tokens_before=("cache_tokens_before", "first"),
            cache_tokens_after=("cache_tokens_after", "first"),
            latency_ms_mean=("latency_ms", "mean"),
            latency_ms_std=("latency_ms", "std"),
            latency_ms_median=("latency_ms", "median"),
            parameter_bytes=("parameter_bytes", "first"),
            parameter_mib=("parameter_mib", "first"),
            activation_bytes=("activation_bytes", "mean"),
            activation_mib=("activation_mib", "mean"),
            kv_cache_bytes=("kv_cache_bytes", "mean"),
            kv_cache_mib=("kv_cache_mib", "mean"),
        )
    )
    prefill_summary["decode_step"] = pd.NA

    decode = raw_results[raw_results["phase"] == "decode"]
    decode_summary = (
        decode.groupby(
            [
                "phase",
                "prompt_tokens",
                "decode_step",
                "cache_tokens_before",
                "cache_tokens_after",
                "layer",
            ],
            as_index=False,
        )
        .agg(
            latency_ms_mean=("latency_ms", "mean"),
            latency_ms_std=("latency_ms", "std"),
            latency_ms_median=("latency_ms", "median"),
            parameter_bytes=("parameter_bytes", "first"),
            parameter_mib=("parameter_mib", "first"),
            activation_bytes=("activation_bytes", "mean"),
            activation_mib=("activation_mib", "mean"),
            kv_cache_bytes=("kv_cache_bytes", "mean"),
            kv_cache_mib=("kv_cache_mib", "mean"),
        )
    )

    summary_columns = [
        "phase",
        "prompt_tokens",
        "decode_step",
        "cache_tokens_before",
        "cache_tokens_after",
        "layer",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_median",
        "parameter_bytes",
        "parameter_mib",
        "activation_bytes",
        "activation_mib",
        "kv_cache_bytes",
        "kv_cache_mib",
    ]
    return (
        pd.concat([prefill_summary, decode_summary], ignore_index=True)[summary_columns]
        .sort_values(
            ["phase", "prompt_tokens", "decode_step", "layer"],
            na_position="first",
        )
        .reset_index(drop=True)
    )


def save_layer_results(
    rows: list[LayerProfileRow],
    config: ProfileConfig,
    runtime: RuntimeInfo,
    artifacts: ModelArtifacts,
    *,
    output_dir: Path,
    decode_steps: int,
    layer_profile_repeats: int,
) -> tuple[pd.DataFrame, LayerResultPaths]:
    """Layer raw/summary CSV와 실행 환경 JSON을 별도 directory에 저장한다."""

    print_section("Layer profiling 결과 저장")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = LayerResultPaths(
        raw_csv=output_dir / "qwen3_4b_layer_profile_raw.csv",
        summary_csv=output_dir / "qwen3_4b_layer_profile_summary.csv",
        environment_json=output_dir / "environment.json",
    )

    num_layers = len(artifacts.model.model.layers)
    raw_results = pd.DataFrame(rows, columns=RAW_COLUMNS)
    _validate_raw_results(raw_results, num_layers)
    summary = summarize_layer_results(raw_results)

    raw_results.to_csv(paths.raw_csv, index=False)
    summary.to_csv(paths.summary_csv, index=False)

    environment = {
        "model": config.model_name,
        "gpu": runtime.gpu_name,
        "precision": config.precision_label,
        "batch_size": 1,
        "num_layers": num_layers,
        "hidden_size": artifacts.hidden_size,
        "prompt_lengths": list(config.prompt_lengths),
        "decode_steps": decode_steps,
        "layer_profile_repeats": layer_profile_repeats,
        "kv_cache": True,
        "logits_to_keep": config.logits_to_keep,
        "pytorch": runtime.pytorch_version,
        "cuda": runtime.cuda_version,
        "transformers": runtime.transformers_version,
    }
    with paths.environment_json.open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=4, ensure_ascii=False)

    return summary, paths


def print_layer_report(summary: pd.DataFrame, paths: LayerResultPaths) -> None:
    """저장된 layer profiling 결과를 간단히 출력한다."""

    print_section("Layer-wise Profiling 완료")
    print("Summary rows:", len(summary))
    print("RAW         :", paths.raw_csv)
    print("SUMMARY     :", paths.summary_csv)
    print("ENV         :", paths.environment_json)
