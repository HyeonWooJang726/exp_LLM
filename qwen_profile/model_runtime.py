"""CUDA 환경 확인, 모델 준비, 입력 생성과 워밍업."""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ProfileConfig
from .utils import bytes_to_gib, clear_gpu_memory, cuda_sync, print_section


@dataclass(frozen=True)
class RuntimeInfo:
    """실행 환경과 GPU 정보."""

    gpu_name: str
    total_vram_gib: float
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    transformers_version: str


@dataclass(frozen=True)
class ModelArtifacts:
    """로드된 모델과 결과 저장에 필요한 모델 메타데이터."""

    tokenizer: Any
    model: Any
    tokenizer_load_time_sec: float
    model_load_time_sec: float
    model_vram_bytes: int
    parameter_count: int
    parameter_bytes: int
    num_layers: int
    hidden_size: int


def inspect_environment() -> RuntimeInfo:
    """CUDA 사용 가능 여부를 확인하고 실행 환경을 출력한다."""

    print_section("STEP 1. 실행 환경 확인")
    print("Python        :", sys.version.split()[0])
    print("OS            :", platform.platform())
    print("PyTorch       :", torch.__version__)
    print("Transformers  :", transformers.__version__)
    print("Torch CUDA    :", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다.\n"
            "PyTorch CUDA 환경을 확인하세요."
        )

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gib = bytes_to_gib(torch.cuda.get_device_properties(0).total_memory)
    print("GPU           :", gpu_name)
    print("GPU VRAM      :", f"{total_vram_gib:.2f} GiB")

    return RuntimeInfo(
        gpu_name=gpu_name,
        total_vram_gib=total_vram_gib,
        python_version=sys.version,
        pytorch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        transformers_version=transformers.__version__,
    )


def load_model_artifacts(config: ProfileConfig) -> ModelArtifacts:
    """토크나이저와 모델을 로드하고 모델 메타데이터를 계산한다."""

    print_section("STEP 2. Tokenizer Load")
    print("Model:", config.model_name)
    print("첫 실행이면 Hugging Face에서 Tokenizer와 모델 파일을 다운로드합니다.")

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer_load_time_sec = time.perf_counter() - tokenizer_start
    print(f"Tokenizer load time: {tokenizer_load_time_sec:.2f} sec")

    print_section("STEP 3. Qwen3-4B 모델 Load")
    print("Precision :", config.precision_label)
    print("Device    :", config.device.upper())

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=config.dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(config.device)
    model.eval()
    cuda_sync()
    model_load_time_sec = time.perf_counter() - load_start

    # 현재 PyTorch 프로세스가 모델 로드 후 할당한 VRAM이다.
    # nvidia-smi로 측정한 system-wide GPU baseline과는 다른 값이다.
    model_vram_bytes = torch.cuda.memory_allocated()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )

    print("Model loaded successfully.")
    print("Model load time :", f"{model_load_time_sec:.2f} sec")
    print("Model VRAM      :", f"{bytes_to_gib(model_vram_bytes):.2f} GiB")

    artifacts = ModelArtifacts(
        tokenizer=tokenizer,
        model=model,
        tokenizer_load_time_sec=tokenizer_load_time_sec,
        model_load_time_sec=model_load_time_sec,
        model_vram_bytes=model_vram_bytes,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        num_layers=model.config.num_hidden_layers,
        hidden_size=model.config.hidden_size,
    )
    print_model_info(artifacts)
    return artifacts


def print_model_info(artifacts: ModelArtifacts) -> None:
    """프로파일링에 사용되는 모델 구조를 출력한다."""

    print_section("STEP 4. 모델 정보")
    print("Transformer Layers :", artifacts.num_layers)
    print("Hidden Size        :", artifacts.hidden_size)
    print("Parameters         :", f"{artifacts.parameter_count / 1e9:.3f} B")
    print("Parameter Memory   :", f"{bytes_to_gib(artifacts.parameter_bytes):.2f} GiB")


def build_token_pool(tokenizer: Any, config: ProfileConfig) -> torch.Tensor:
    """요청한 모든 prompt 길이를 만들 수 있는 합성 token pool을 생성한다."""

    base_text = (
        "Large language models use transformer architectures "
        "to perform autoregressive text generation. "
        "Artificial intelligence applications require computation, "
        "memory, and efficient inference systems. "
        "Edge computing provides computational resources "
        "close to users and data sources. "
    )
    token_pool = tokenizer(
        base_text * 10000,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    if len(token_pool) < max(config.prompt_lengths):
        raise RuntimeError("Synthetic prompt token 수가 충분하지 않습니다.")
    return token_pool


def make_input(
    token_pool: torch.Tensor,
    prompt_length: int,
    device: str,
) -> torch.Tensor:
    """token pool 앞부분으로 정확히 N token 길이의 배치 입력을 만든다."""

    return token_pool[:prompt_length].clone().unsqueeze(0).to(device)


def print_profiling_config(config: ProfileConfig) -> None:
    """실행할 실험 조합을 출력한다."""

    total_cases = len(config.prompt_lengths) * len(config.output_lengths) * config.repeats
    print("Prompt lengths :", list(config.prompt_lengths))
    print("Output lengths :", list(config.output_lengths))
    print("Warm-up runs   :", config.warmup_runs)
    print("Repeats        :", config.repeats)
    print("Total profiling runs:", total_cases)


def run_smoke_test(model: Any, token_pool: torch.Tensor, config: ProfileConfig) -> None:
    """짧은 입력으로 모델 forward가 정상 동작하는지 확인한다."""

    print_section("STEP 6. Smoke Test")
    smoke_input = make_input(
        token_pool,
        config.smoke_test_prompt_length,
        config.device,
    )
    with torch.inference_mode():
        smoke_output = model(
            input_ids=smoke_input,
            use_cache=True,
            logits_to_keep=config.logits_to_keep,
        )
    cuda_sync()
    print("Forward pass 성공.")
    print("Logits shape:", tuple(smoke_output.logits.shape))

    del smoke_input, smoke_output
    clear_gpu_memory()


def warm_up_gpu(model: Any, token_pool: torch.Tensor, config: ProfileConfig) -> None:
    """초기 CUDA kernel 실행 오버헤드를 본 측정 전에 제거한다."""

    print_section("STEP 7. GPU Warm-up")
    for warmup_index in range(config.warmup_runs):
        warm_input = make_input(token_pool, config.warmup_prompt_length, config.device)
        with torch.inference_mode():
            warm_output = model(
                input_ids=warm_input,
                use_cache=True,
                logits_to_keep=config.logits_to_keep,
            )
        cuda_sync()
        print(f"Warm-up {warmup_index + 1}/{config.warmup_runs}")

        del warm_input, warm_output
        clear_gpu_memory()
    print("Warm-up 완료.")
