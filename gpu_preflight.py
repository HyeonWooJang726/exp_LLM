"""PyTorch import 전에 시스템 전체 GPU 상태를 측정한다.

이 모듈은 의도적으로 Python 표준 라이브러리만 사용한다. 측정값은
``nvidia-smi`` 기준이므로 Windows, GUI, 다른 프로세스의 VRAM 사용량을
모두 포함하며 PyTorch allocator 통계와는 별개의 값이다.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime


# ============================================================
# Initial system GPU baseline 데이터
# ============================================================


@dataclass(frozen=True)
class SystemGpuStatus:
    """GPU 0의 system-wide 상태. 조회 실패 항목은 ``None``이다."""

    timestamp: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    utilization_gpu_percent: int | None
    temperature_gpu_c: int | None
    power_draw_w: float | None
    pstate: str | None
    error_message: str | None = None


# ============================================================
# nvidia-smi 조회 및 값 변환
# ============================================================


def _parse_int(value: str) -> int | None:
    """nvidia-smi의 정수형 문자열을 변환한다."""

    normalized = value.strip()
    if (
        not normalized
        or "N/A" in normalized.upper()
        or "NOT SUPPORTED" in normalized.upper()
    ):
        return None
    return int(float(normalized))


def _parse_float(value: str) -> float | None:
    """nvidia-smi의 실수형 문자열을 변환한다."""

    normalized = value.strip()
    if (
        not normalized
        or "N/A" in normalized.upper()
        or "NOT SUPPORTED" in normalized.upper()
    ):
        return None
    return float(normalized)


def _parse_text(value: str) -> str | None:
    """nvidia-smi의 텍스트 값을 변환한다."""

    normalized = value.strip()
    if (
        not normalized
        or "N/A" in normalized.upper()
        or "NOT SUPPORTED" in normalized.upper()
    ):
        return None
    return normalized


def measure_initial_system_gpu_status() -> SystemGpuStatus:
    """GPU 0의 baseline을 조회하며, 실패해도 예외를 밖으로 전파하지 않는다."""

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    command = [
        "nvidia-smi",
        "--id=0",
        (
            "--query-gpu=memory.used,memory.total,utilization.gpu,"
            "temperature.gpu,power.draw,pstate"
        ),
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_row = next(
            line for line in completed.stdout.splitlines() if line.strip()
        )
        values = [value.strip() for value in first_row.split(",")]
        if len(values) != 6:
            raise ValueError(
                f"nvidia-smi가 예상과 다른 열 수를 반환했습니다: {len(values)}"
            )

        return SystemGpuStatus(
            timestamp=timestamp,
            memory_used_mib=_parse_int(values[0]),
            memory_total_mib=_parse_int(values[1]),
            utilization_gpu_percent=_parse_int(values[2]),
            temperature_gpu_c=_parse_int(values[3]),
            power_draw_w=_parse_float(values[4]),
            pstate=_parse_text(values[5]),
        )
    except Exception as error:
        return SystemGpuStatus(
            timestamp=timestamp,
            memory_used_mib=None,
            memory_total_mib=None,
            utilization_gpu_percent=None,
            temperature_gpu_c=None,
            power_draw_w=None,
            pstate=None,
            error_message=f"{type(error).__name__}: {error}",
        )


# ============================================================
# Initial system GPU baseline 출력
# ============================================================


def _format_metric(value: int | float | str | None, unit: str = "") -> str:
    """조회되지 않은 값은 N/A로 표시한다."""

    if value is None:
        return "N/A"
    if isinstance(value, float):
        rendered = f"{value:g}"
    else:
        rendered = str(value)
    return f"{rendered} {unit}".rstrip()


def print_initial_system_gpu_status(status: SystemGpuStatus) -> None:
    """프로파일링 모듈 import 전에 baseline을 콘솔에 출력한다."""

    print("=" * 72)
    print("INITIAL SYSTEM GPU BASELINE")
    print("=" * 72)
    print("GPU Memory Used :", _format_metric(status.memory_used_mib, "MiB"))
    print("GPU Memory Total:", _format_metric(status.memory_total_mib, "MiB"))
    print("GPU Utilization :", _format_metric(status.utilization_gpu_percent, "%"))
    print("Temperature     :", _format_metric(status.temperature_gpu_c, "C"))
    print("Power Draw      :", _format_metric(status.power_draw_w, "W"))
    print("P-State         :", _format_metric(status.pstate))
    print("Timestamp       :", status.timestamp)
    if status.error_message is not None:
        print("Baseline Status : unavailable -", status.error_message)
