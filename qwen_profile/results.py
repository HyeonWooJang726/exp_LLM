"""프로파일링 결과 집계, 저장과 최종 출력."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import ProfileConfig
from .utils import bytes_to_gib, print_section

if TYPE_CHECKING:
    from gpu_preflight import SystemGpuStatus

    from .benchmark import ProfileResult
    from .model_runtime import ModelArtifacts, RuntimeInfo


@dataclass(frozen=True)
class SavedResultPaths:
    """한 번의 실행에서 생성된 결과 파일 경로."""

    raw_csv: Path
    summary_csv: Path
    environment_json: Path


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """실험 조건별 평균과 표준편차를 계산한다."""

    return (
        raw_results.groupby(["prompt_tokens", "output_tokens"])
        .agg(
            ttft_ms_mean=("ttft_ms", "mean"),
            ttft_ms_std=("ttft_ms", "std"),
            tpot_ms_mean=("tpot_ms", "mean"),
            tpot_ms_std=("tpot_ms", "std"),
            tokens_per_second_mean=("tokens_per_second", "mean"),
            tokens_per_second_std=("tokens_per_second", "std"),
            total_latency_ms_mean=("total_latency_ms", "mean"),
            total_latency_ms_std=("total_latency_ms", "std"),
            peak_vram_gib_mean=("peak_vram_gib", "mean"),
            peak_vram_gib_std=("peak_vram_gib", "std"),
        )
        .reset_index()
    )


def save_results(
    results: list[ProfileResult],
    config: ProfileConfig,
    runtime: RuntimeInfo,
    artifacts: ModelArtifacts,
    *,
    initial_system_gpu_status: SystemGpuStatus | None = None,
) -> tuple[pd.DataFrame, SavedResultPaths]:
    """raw/summary CSV와 실행 환경 JSON을 저장한다."""

    print_section("STEP 9. 결과 저장")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_results = pd.DataFrame(results)
    summary = summarize_results(raw_results)
    paths = SavedResultPaths(
        raw_csv=config.output_dir / "qwen3_4b_profile_raw.csv",
        summary_csv=config.output_dir / "qwen3_4b_profile_summary.csv",
        environment_json=config.output_dir / "environment.json",
    )

    raw_results.to_csv(paths.raw_csv, index=False)
    summary.to_csv(paths.summary_csv, index=False)

    # nvidia-smi baseline은 Windows/GUI/다른 프로세스를 포함한 system-wide 값이다.
    baseline = initial_system_gpu_status
    environment = {
        "model": config.model_name,
        "precision": config.precision_label,
        "batch_size": 1,
        "kv_cache": True,
        "logits_to_keep": config.logits_to_keep,
        "decoding": "greedy",
        "gpu": runtime.gpu_name,
        "gpu_vram_gib": runtime.total_vram_gib,
        "system_gpu_memory_used_mib_before_model_load": (
            baseline.memory_used_mib if baseline is not None else None
        ),
        "system_gpu_memory_total_mib": (
            baseline.memory_total_mib if baseline is not None else None
        ),
        "system_gpu_utilization_percent_before_model_load": (
            baseline.utilization_gpu_percent if baseline is not None else None
        ),
        "system_gpu_temperature_c_before_model_load": (
            baseline.temperature_gpu_c if baseline is not None else None
        ),
        "system_gpu_power_draw_w_before_model_load": (
            baseline.power_draw_w if baseline is not None else None
        ),
        "system_gpu_pstate_before_model_load": (
            baseline.pstate if baseline is not None else None
        ),
        "system_gpu_baseline_timestamp": (
            baseline.timestamp if baseline is not None else None
        ),
        "python": runtime.python_version,
        "pytorch": runtime.pytorch_version,
        "cuda": runtime.cuda_version,
        "transformers": runtime.transformers_version,
        "model_parameters": artifacts.parameter_count,
        "model_parameter_memory_gib": bytes_to_gib(artifacts.parameter_bytes),
        # 아래 값은 system-wide baseline이 아닌 현재 PyTorch 프로세스의 할당량이다.
        "model_loaded_vram_gib": bytes_to_gib(artifacts.model_vram_bytes),
        "tokenizer_load_time_sec": artifacts.tokenizer_load_time_sec,
        "model_load_time_sec": artifacts.model_load_time_sec,
        "prompt_lengths": list(config.prompt_lengths),
        "output_lengths": list(config.output_lengths),
        "warmup_runs": config.warmup_runs,
        "repeats": config.repeats,
    }
    with paths.environment_json.open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=4, ensure_ascii=False)

    return summary, paths


def print_final_report(summary: pd.DataFrame, paths: SavedResultPaths) -> None:
    """집계 표, 생성 파일과 핵심 지표 설명을 출력한다."""

    print_section("PROFILING 완료")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))

    print_section("저장된 파일")
    print("RAW     :", paths.raw_csv)
    print("SUMMARY :", paths.summary_csv)
    print("ENV     :", paths.environment_json)

    print_section("핵심적으로 볼 값")
    print("1. ttft_ms_mean              -> 첫 Token까지 걸린 시간")
    print("2. tpot_ms_mean              -> 이후 Token 하나 생성 시간")
    print("3. tokens_per_second_mean    -> 초당 Decode Token 수")
    print("4. total_latency_ms_mean     -> 요청 전체 처리시간")
    print("5. peak_vram_gib_mean        -> 최대 GPU Memory")
    print()
    print("모든 Profiling이 완료되었습니다.")
