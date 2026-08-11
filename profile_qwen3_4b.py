"""Qwen 프로파일러 실행 진입점.

System-wide GPU baseline을 가장 먼저 측정하기 위해 이 모듈의 초기 import는
Python 표준 라이브러리만 사용하는 ``gpu_preflight``로 제한한다.
"""

from gpu_preflight import (
    measure_initial_system_gpu_status,
    print_initial_system_gpu_status,
)


def main() -> None:
    """GPU baseline 측정 후 무거운 profiling 모듈을 불러 실행한다."""

    # ============================================================
    # PyTorch / Transformers import 전 system-wide GPU baseline
    # ============================================================
    initial_gpu_status = measure_initial_system_gpu_status()
    print_initial_system_gpu_status(initial_gpu_status)

    # ============================================================
    # Baseline 측정이 끝난 뒤 profiling 모듈 import 및 실행
    # ============================================================
    from qwen_profile.runner import run

    run(initial_system_gpu_status=initial_gpu_status)


if __name__ == "__main__":
    main()
