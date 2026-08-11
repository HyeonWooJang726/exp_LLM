"""전체 프로파일링 단계를 순서대로 조율하는 실행 모듈."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .benchmark import run_benchmarks
from .config import DEFAULT_CONFIG, ProfileConfig
from .model_runtime import (
    build_token_pool,
    inspect_environment,
    load_model_artifacts,
    print_profiling_config,
    run_smoke_test,
    warm_up_gpu,
)
from .results import print_final_report, save_results
from .utils import clear_gpu_memory, print_section

if TYPE_CHECKING:
    from gpu_preflight import SystemGpuStatus


def run(
    config: ProfileConfig = DEFAULT_CONFIG,
    *,
    initial_system_gpu_status: SystemGpuStatus | None = None,
) -> None:
    """환경 확인부터 결과 저장까지 전체 프로파일링을 실행한다."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = inspect_environment()
    clear_gpu_memory(reset_peak_stats=True)
    artifacts = load_model_artifacts(config)

    print_section("STEP 5. Profiling 입력 생성")
    token_pool = build_token_pool(artifacts.tokenizer, config)
    print_profiling_config(config)

    run_smoke_test(artifacts.model, token_pool, config)
    warm_up_gpu(artifacts.model, token_pool, config)
    raw_results = run_benchmarks(
        artifacts.model,
        token_pool,
        config,
        artifacts.model_vram_bytes,
    )
    summary, paths = save_results(
        raw_results,
        config,
        runtime,
        artifacts,
        initial_system_gpu_status=initial_system_gpu_status,
    )
    print_final_report(summary, paths)
