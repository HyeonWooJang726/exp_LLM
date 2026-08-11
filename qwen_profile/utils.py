"""여러 프로파일링 단계가 함께 사용하는 작은 유틸리티."""

import gc

import torch


def cuda_sync() -> None:
    """정확한 시간 측정을 위해 대기 중인 CUDA 연산을 완료한다."""

    torch.cuda.synchronize()


def bytes_to_gib(byte_value: int) -> float:
    """바이트를 GiB로 변환한다."""

    return byte_value / (1024**3)


def print_section(title: str) -> None:
    """콘솔 출력에 단계 구분선을 표시한다."""

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def clear_gpu_memory(*, reset_peak_stats: bool = False) -> None:
    """사용하지 않는 Python 객체와 CUDA 캐시를 정리한다."""

    gc.collect()
    torch.cuda.empty_cache()
    if reset_peak_stats:
        torch.cuda.reset_peak_memory_stats()
