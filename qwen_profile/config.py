"""프로파일링 실험 설정.

실험 조건을 바꾸고 싶을 때는 이 파일의 ``DEFAULT_CONFIG``만 수정하면 됩니다.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ProfileConfig:
    """한 번의 프로파일링 실행에 필요한 설정."""

    model_name: str
    prompt_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]
    warmup_runs: int
    repeats: int
    output_dir: Path
    dtype: torch.dtype = torch.bfloat16
    precision_label: str = "BF16"
    device: str = "cuda"
    # logits_to_keep=1은 Transformer prefill이나 KV cache를 줄이는 옵션이 아니다.
    # Prompt 전체의 Transformer 계산과 KV cache 생성은 그대로 수행한다.
    # 마지막 hidden state에만 LM Head를 적용해 next-token logits를 생성하며,
    # full-sequence logits 생성을 제거하기 위한 설정이다.
    logits_to_keep: int = 1
    smoke_test_prompt_length: int = 32
    warmup_prompt_length: int = 128

    def __post_init__(self) -> None:
        if not self.prompt_lengths or any(length <= 0 for length in self.prompt_lengths):
            raise ValueError("prompt_lengths에는 양의 정수가 하나 이상 필요합니다.")
        if not self.output_lengths or any(length <= 0 for length in self.output_lengths):
            raise ValueError("output_lengths에는 양의 정수가 하나 이상 필요합니다.")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs는 0 이상이어야 합니다.")
        if self.repeats <= 0:
            raise ValueError("repeats는 1 이상이어야 합니다.")


DEFAULT_CONFIG = ProfileConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    prompt_lengths=(128, 512, 1024, 2048),
    output_lengths=(32, 128),
    warmup_runs=2,
    repeats=3,
    output_dir=Path("qwen_profile_results"),
)
