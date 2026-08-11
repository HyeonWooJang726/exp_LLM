"""Full-model과 layer-wise profiling 및 figure 생성을 순차 실행한다."""

import subprocess
import sys


SCRIPTS = [
    "profile_qwen3_4b.py",
    "profile_qwen3_4b_layers.py",
    "Qwen3-4B Profiling Result Plotter.py",
    "plot_qwen3_4b_layers.py",
]


def main() -> None:
    """실행 전 주의사항을 출력하고 각 단계를 별도 process에서 실행한다."""

    print("=" * 72)
    print("Qwen3-4B Profiling")
    print("=" * 72)
    print()
    print("Before profiling:")
    print("- Close ChatGPT / browsers")
    print("- Close GitHub Desktop")
    print("- Close unnecessary GPU applications")
    print("- Stop other CUDA / Python workloads")
    print("- Keep the system idle during profiling")
    print()
    print("Sequence:")
    print("1. Full-model profiling")
    print("2. Layer-wise profiling")
    print("3. Full-model figures")
    print("4. Layer-wise figures")
    print()
    print("=" * 72)
    sys.stdout.flush()

    for script_name in SCRIPTS:
        subprocess.run(
            [sys.executable, script_name],
            check=True,
        )


if __name__ == "__main__":
    main()
