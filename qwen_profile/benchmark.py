"""TTFT, TPOT, 처리량, 지연 시간과 GPU 메모리 측정."""

from __future__ import annotations

import time
from typing import Any

import torch

from .config import ProfileConfig
from .model_runtime import make_input
from .utils import bytes_to_gib, clear_gpu_memory, cuda_sync, print_section

ProfileResult = dict[str, int | float]


def profile_once(
    model: Any,
    token_pool: torch.Tensor,
    *,
    prompt_length: int,
    output_length: int,
    repeat: int,
    model_vram_bytes: int,
    device: str,
    logits_to_keep: int,
) -> ProfileResult:
    """하나의 prompt/output 조합을 한 번 측정한다."""

    clear_gpu_memory(reset_peak_stats=True)
    input_ids = make_input(token_pool, prompt_length, device)
    memory_before_prefill = torch.cuda.memory_allocated()

    cuda_sync()
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        first_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
    cuda_sync()
    ttft_sec = time.perf_counter() - prefill_start

    past_key_values = outputs.past_key_values
    memory_after_prefill = torch.cuda.memory_allocated()
    next_token = first_token
    decode_times: list[float] = []

    with torch.inference_mode():
        for _ in range(output_length - 1):
            cuda_sync()
            decode_start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
            past_key_values = outputs.past_key_values
            cuda_sync()
            decode_times.append(time.perf_counter() - decode_start)

    decode_total_sec = sum(decode_times)
    if decode_times:
        tpot_sec = decode_total_sec / len(decode_times)
        tokens_per_second = 1.0 / tpot_sec
    else:
        tpot_sec = 0.0
        tokens_per_second = 0.0

    total_latency_sec = ttft_sec + decode_total_sec

    # 현재 PyTorch profiling 프로세스 allocator의 peak 값이다.
    # Windows 및 다른 프로세스까지 포함하는 nvidia-smi baseline과는 다르다.
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    prefill_memory_delta = memory_after_prefill - memory_before_prefill

    result: ProfileResult = {
        "prompt_tokens": prompt_length,
        "output_tokens": output_length,
        "repeat": repeat,
        "ttft_ms": ttft_sec * 1000,
        "decode_total_ms": decode_total_sec * 1000,
        "tpot_ms": tpot_sec * 1000,
        "tokens_per_second": tokens_per_second,
        "total_latency_ms": total_latency_sec * 1000,
        "model_vram_gib": bytes_to_gib(model_vram_bytes),
        "memory_before_prefill_gib": bytes_to_gib(memory_before_prefill),
        "memory_after_prefill_gib": bytes_to_gib(memory_after_prefill),
        "prefill_memory_delta_gib": bytes_to_gib(prefill_memory_delta),
        "peak_vram_gib": bytes_to_gib(peak_vram_bytes),
    }

    print(f"TTFT        : {ttft_sec * 1000:.2f} ms")
    print(f"TPOT        : {tpot_sec * 1000:.2f} ms/token")
    print(f"Throughput  : {tokens_per_second:.2f} token/s")
    print(f"Total       : {total_latency_sec:.3f} sec")
    print(f"Peak VRAM   : {bytes_to_gib(peak_vram_bytes):.2f} GiB")

    del input_ids, outputs, first_token, next_token, past_key_values, decode_times
    clear_gpu_memory()
    return result


def run_benchmarks(
    model: Any,
    token_pool: torch.Tensor,
    config: ProfileConfig,
    model_vram_bytes: int,
) -> list[ProfileResult]:
    """설정에 포함된 모든 prompt/output/repeat 조합을 측정한다."""

    print_section("STEP 8. Profiling 시작")
    results: list[ProfileResult] = []
    total_cases = len(config.prompt_lengths) * len(config.output_lengths) * config.repeats
    current_case = 0

    for prompt_length in config.prompt_lengths:
        for output_length in config.output_lengths:
            for repeat in range(1, config.repeats + 1):
                current_case += 1
                print()
                print("-" * 72)
                print(
                    f"[{current_case}/{total_cases}] "
                    f"Prompt={prompt_length} | Output={output_length} | Run={repeat}"
                )
                print("-" * 72)
                results.append(
                    profile_once(
                        model,
                        token_pool,
                        prompt_length=prompt_length,
                        output_length=output_length,
                        repeat=repeat,
                        model_vram_bytes=model_vram_bytes,
                        device=config.device,
                        logits_to_keep=config.logits_to_keep,
                    )
                )
    return results
