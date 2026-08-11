"""Hugging Face Qwen forward hook을 이용한 layer-wise 측정."""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import ProfileConfig
from .model_runtime import make_input
from .utils import clear_gpu_memory, cuda_sync, print_section

LayerProfileRow = dict[str, str | int | float | None]


def _tensor_bytes(tensor: torch.Tensor) -> int:
    """실제 tensor의 크기를 byte 단위로 계산한다."""

    return tensor.numel() * tensor.element_size()


def _main_hidden_state(layer_output: Any, layer_index: int) -> torch.Tensor:
    """Layer output에서 다음 layer로 전달되는 hidden state를 가져온다."""

    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output:
        hidden_state = layer_output[0]
        if isinstance(hidden_state, torch.Tensor):
            return hidden_state
    raise RuntimeError(
        f"Layer {layer_index}의 main hidden-state tensor를 hook output에서 읽을 수 없습니다."
    )


def _transformer_layers(model: Any) -> Any:
    """Hard-coding 없이 실제 Qwen decoder layer 목록을 반환한다."""

    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("model.model.layers에서 Transformer layer를 확인할 수 없습니다.")
    return layers


def _parameter_sizes(layers: Any) -> list[tuple[int, float]]:
    """Layer별 parameter tensor에서 weight memory를 직접 계산한다."""

    sizes: list[tuple[int, float]] = []
    for layer in layers:
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in layer.parameters()
        )
        sizes.append((parameter_bytes, parameter_bytes / (1024**2)))
    return sizes


def _cache_seq_length(past_key_values: Any) -> int:
    """Transformers Cache object에서 현재 context token 수를 읽는다."""

    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if not callable(get_seq_length):
        raise RuntimeError("past_key_values에서 cache sequence length를 읽을 수 없습니다.")
    return int(get_seq_length())


def _kv_cache_sizes(
    past_key_values: Any,
    num_layers: int,
) -> list[tuple[int, float]]:
    """Transformers Cache layer의 실제 Key/Value tensor 크기를 계산한다."""

    cache_layers = getattr(past_key_values, "layers", None)
    if cache_layers is None:
        raise RuntimeError(
            "past_key_values.layers를 읽을 수 없어 실제 layer별 K/V tensor 크기를 "
            "계산할 수 없습니다."
        )
    if len(cache_layers) != num_layers:
        raise RuntimeError(
            f"Transformer layer 수({num_layers})와 cache layer 수({len(cache_layers)})가 다릅니다."
        )

    sizes: list[tuple[int, float]] = []
    for layer_index, cache_layer in enumerate(cache_layers):
        key_tensor = getattr(cache_layer, "keys", None)
        value_tensor = getattr(cache_layer, "values", None)
        if not isinstance(key_tensor, torch.Tensor) or not isinstance(value_tensor, torch.Tensor):
            raise RuntimeError(
                f"Cache layer {layer_index}의 실제 Key/Value tensor를 읽을 수 없습니다."
            )

        kv_cache_bytes = _tensor_bytes(key_tensor) + _tensor_bytes(value_tensor)
        if kv_cache_bytes <= 0:
            raise RuntimeError(f"Cache layer {layer_index}의 K/V tensor가 비어 있습니다.")
        sizes.append((kv_cache_bytes, kv_cache_bytes / (1024**2)))
    return sizes


def _measure_layer_forward(
    model: Any,
    layers: Any,
    **model_kwargs: Any,
) -> tuple[Any, list[float], list[tuple[int, float]]]:
    """Forward 한 번의 layer latency와 boundary activation 크기를 측정한다."""

    num_layers = len(layers)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
    activation_sizes: list[tuple[int, float] | None] = [None] * num_layers
    hook_handles = []

    for layer_index, layer in enumerate(layers):

        def record_start(_module: Any, _inputs: Any, index: int = layer_index) -> None:
            start_events[index].record()

        def record_end(
            _module: Any,
            _inputs: Any,
            output: Any,
            index: int = layer_index,
        ) -> None:
            end_events[index].record()
            hidden_state = _main_hidden_state(output, index)
            activation_bytes = _tensor_bytes(hidden_state)
            activation_sizes[index] = (
                activation_bytes,
                activation_bytes / (1024**2),
            )

        hook_handles.append(layer.register_forward_pre_hook(record_start))
        hook_handles.append(layer.register_forward_hook(record_end))

    try:
        with torch.inference_mode():
            outputs = model(**model_kwargs)

        # Layer별 synchronize는 하지 않고, 전체 forward 종료 후 한 번만 동기화한다.
        cuda_sync()
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

    latencies_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(start_events, end_events)
    ]
    if any(not math.isfinite(latency) or latency < 0 for latency in latencies_ms):
        raise RuntimeError("Layer latency에 NaN, infinity 또는 음수가 포함되었습니다.")
    if any(size is None for size in activation_sizes):
        raise RuntimeError("모든 Transformer layer의 boundary activation을 측정하지 못했습니다.")

    return outputs, latencies_ms, [size for size in activation_sizes if size is not None]


def _append_layer_rows(
    rows: list[LayerProfileRow],
    *,
    device: str,
    phase: str,
    prompt_tokens: int,
    decode_step: int | None,
    cache_tokens_before: int,
    cache_tokens_after: int,
    repeat: int,
    latencies_ms: list[float],
    parameter_sizes: list[tuple[int, float]],
    activation_sizes: list[tuple[int, float]],
    kv_cache_sizes: list[tuple[int, float]],
) -> None:
    """Forward 한 번의 layer별 측정치를 raw row로 추가한다."""

    for layer_index, latency_ms in enumerate(latencies_ms):
        parameter_bytes, parameter_mib = parameter_sizes[layer_index]
        activation_bytes, activation_mib = activation_sizes[layer_index]
        kv_cache_bytes, kv_cache_mib = kv_cache_sizes[layer_index]
        rows.append(
            {
                "device": device,
                "phase": phase,
                "prompt_tokens": prompt_tokens,
                "decode_step": decode_step,
                "cache_tokens_before": cache_tokens_before,
                "cache_tokens_after": cache_tokens_after,
                "layer": layer_index,
                "repeat": repeat,
                "latency_ms": latency_ms,
                "parameter_bytes": parameter_bytes,
                "parameter_mib": parameter_mib,
                "activation_bytes": activation_bytes,
                "activation_mib": activation_mib,
                "kv_cache_bytes": kv_cache_bytes,
                "kv_cache_mib": kv_cache_mib,
            }
        )


def run_layer_benchmarks(
    model: Any,
    token_pool: torch.Tensor,
    config: ProfileConfig,
    *,
    repeats: int,
    decode_steps: int,
) -> list[LayerProfileRow]:
    """Prompt별 prefill과 autoregressive decode의 layer-wise cost를 측정한다."""

    if repeats <= 0:
        raise ValueError("Layer profiling repeats는 1 이상이어야 합니다.")
    if decode_steps < 0:
        raise ValueError("Decode steps는 0 이상이어야 합니다.")

    layers = _transformer_layers(model)
    num_layers = len(layers)
    parameter_sizes = _parameter_sizes(layers)
    rows: list[LayerProfileRow] = []

    print_section("Layer-wise Profiling 시작")
    print("Transformer layers :", num_layers)
    print("Prompt lengths     :", list(config.prompt_lengths))
    print("Decode steps       :", decode_steps)
    print("Layer repeats      :", repeats)

    for prompt_tokens in config.prompt_lengths:
        for repeat in range(1, repeats + 1):
            print(
                f"Prompt={prompt_tokens} | Repeat={repeat}/{repeats} | "
                f"Decode steps={decode_steps}"
            )
            clear_gpu_memory()
            input_ids = make_input(token_pool, prompt_tokens, config.device)

            outputs, latencies_ms, activation_sizes = _measure_layer_forward(
                model,
                layers,
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=config.logits_to_keep,
            )
            past_key_values = outputs.past_key_values
            cache_tokens_after = _cache_seq_length(past_key_values)
            if cache_tokens_after != prompt_tokens:
                raise RuntimeError(
                    f"Prefill cache token 수({cache_tokens_after})가 prompt token 수({prompt_tokens})와 "
                    "다릅니다."
                )
            kv_cache_sizes = _kv_cache_sizes(past_key_values, num_layers)
            _append_layer_rows(
                rows,
                device=config.device,
                phase="prefill",
                prompt_tokens=prompt_tokens,
                decode_step=None,
                cache_tokens_before=0,
                cache_tokens_after=cache_tokens_after,
                repeat=repeat,
                latencies_ms=latencies_ms,
                parameter_sizes=parameter_sizes,
                activation_sizes=activation_sizes,
                kv_cache_sizes=kv_cache_sizes,
            )

            next_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
            del input_ids, outputs

            for decode_step in range(1, decode_steps + 1):
                cache_tokens_before = _cache_seq_length(past_key_values)
                outputs, latencies_ms, activation_sizes = _measure_layer_forward(
                    model,
                    layers,
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=config.logits_to_keep,
                )
                past_key_values = outputs.past_key_values
                cache_tokens_after = _cache_seq_length(past_key_values)
                if cache_tokens_after < cache_tokens_before:
                    raise RuntimeError("Decode 중 cache token 수가 감소했습니다.")

                kv_cache_sizes = _kv_cache_sizes(past_key_values, num_layers)
                _append_layer_rows(
                    rows,
                    device=config.device,
                    phase="decode",
                    prompt_tokens=prompt_tokens,
                    decode_step=decode_step,
                    cache_tokens_before=cache_tokens_before,
                    cache_tokens_after=cache_tokens_after,
                    repeat=repeat,
                    latencies_ms=latencies_ms,
                    parameter_sizes=parameter_sizes,
                    activation_sizes=activation_sizes,
                    kv_cache_sizes=kv_cache_sizes,
                )

                next_token = torch.argmax(
                    outputs.logits[:, -1, :],
                    dim=-1,
                    keepdim=True,
                )
                del outputs

            del next_token, past_key_values
            clear_gpu_memory()

    return rows
