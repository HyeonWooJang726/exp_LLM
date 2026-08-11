"""Qwen3-4B layer-wise profiler 실행 진입점."""

from pathlib import Path

from gpu_preflight import (
    measure_initial_system_gpu_status,
    print_initial_system_gpu_status,
)

LAYER_PROFILE_REPEATS = 10
LAYER_RESULT_DIR = Path("qwen_layer_profile_results")


def main() -> None:
    """Model 준비 후 layer-wise prefill/decode profiling을 실행한다."""

    # ============================================================
    # PyTorch / Transformers import 전 system-wide GPU baseline
    # ============================================================
    initial_gpu_status = measure_initial_system_gpu_status()
    print_initial_system_gpu_status(initial_gpu_status)

    # ============================================================
    # Baseline 측정이 끝난 후 profiling 모듈 import 및 실행
    # ============================================================
    from qwen_profile.config import DEFAULT_CONFIG
    from qwen_profile.layer_benchmark import run_layer_benchmarks
    from qwen_profile.layer_results import print_layer_report, save_layer_results
    from qwen_profile.model_runtime import (
        build_token_pool,
        inspect_environment,
        load_model_artifacts,
        run_smoke_test,
        warm_up_gpu,
    )
    from qwen_profile.utils import clear_gpu_memory, print_section

    runtime = inspect_environment()
    clear_gpu_memory(reset_peak_stats=True)
    artifacts = load_model_artifacts(DEFAULT_CONFIG)

    print_section("Layer profiling 입력 생성")
    token_pool = build_token_pool(artifacts.tokenizer, DEFAULT_CONFIG)
    run_smoke_test(artifacts.model, token_pool, DEFAULT_CONFIG)
    warm_up_gpu(artifacts.model, token_pool, DEFAULT_CONFIG)

    decode_steps = max(DEFAULT_CONFIG.output_lengths) - 1
    rows = run_layer_benchmarks(
        artifacts.model,
        token_pool,
        DEFAULT_CONFIG,
        repeats=LAYER_PROFILE_REPEATS,
        decode_steps=decode_steps,
        device_label=runtime.gpu_name,
    )
    summary, paths = save_layer_results(
        rows,
        DEFAULT_CONFIG,
        runtime,
        artifacts,
        output_dir=LAYER_RESULT_DIR,
        decode_steps=decode_steps,
        layer_profile_repeats=LAYER_PROFILE_REPEATS,
        initial_system_gpu_status=initial_gpu_status,
    )
    print_layer_report(summary, paths)


if __name__ == "__main__":
    main()
