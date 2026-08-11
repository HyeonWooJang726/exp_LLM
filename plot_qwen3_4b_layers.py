"""Qwen3-4B layer-wise profiling 결과를 독립된 PNG 그림으로 생성한다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = Path("qwen_layer_profile_results")
RAW_CSV = RESULT_DIR / "qwen3_4b_layer_profile_raw.csv"
FIGURE_DIR = RESULT_DIR / "figures"


def _validate_input(raw: pd.DataFrame) -> list[int]:
    """그래프에 필요한 column과 0부터 시작하는 layer index를 확인한다."""

    required_columns = {
        "phase",
        "prompt_tokens",
        "decode_step",
        "layer",
        "latency_ms",
        "parameter_mib",
        "activation_mib",
        "kv_cache_mib",
    }
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        raise ValueError(f"RAW CSV에 필수 column이 없습니다: {sorted(missing_columns)}")

    layers = sorted(int(layer) for layer in raw["layer"].unique())
    if layers != list(range(len(layers))):
        raise ValueError("Layer index는 0부터 연속적이어야 합니다.")
    return layers


def _plot_condition_lines(
    data: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    y_label: str,
    output_name: str,
    layers: list[int],
) -> None:
    """Starting prompt length별 line을 하나의 독립된 figure에 그린다."""

    plt.figure(figsize=(12, 6))
    for prompt_tokens in sorted(data["prompt_tokens"].unique()):
        prompt_data = data[data["prompt_tokens"] == prompt_tokens].sort_values("layer")
        plt.plot(
            prompt_data["layer"],
            prompt_data[value_column],
            marker="o",
            markersize=3,
            label=f"Prompt {int(prompt_tokens)}",
        )
    plt.title(title)
    plt.xlabel("Transformer Layer")
    plt.ylabel(y_label)
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    output_path = FIGURE_DIR / output_name
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved:", output_path)


def main() -> None:
    """RAW CSV에서 필수 layer-wise 그래프 7개를 생성한다."""

    raw = pd.read_csv(RAW_CSV)
    layers = _validate_input(raw)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    prefill = raw[raw["phase"] == "prefill"]
    decode = raw[raw["phase"] == "decode"]
    if prefill.empty or decode.empty:
        raise ValueError("RAW CSV에 prefill과 decode row가 모두 필요합니다.")

    prefill_latency = (
        prefill.groupby(["prompt_tokens", "layer"], as_index=False)["latency_ms"].mean()
    )
    _plot_condition_lines(
        prefill_latency,
        value_column="latency_ms",
        title="Qwen3-4B Prefill Latency by Transformer Layer",
        y_label="Mean Prefill Latency (ms)",
        output_name="prefill_layer_latency.png",
        layers=layers,
    )

    decode_latency = (
        decode.groupby(["prompt_tokens", "layer"], as_index=False)["latency_ms"].mean()
    )
    _plot_condition_lines(
        decode_latency,
        value_column="latency_ms",
        title="Qwen3-4B Decode Latency by Transformer Layer",
        y_label="Mean Decode Latency (ms)",
        output_name="decode_layer_latency.png",
        layers=layers,
    )

    parameter_memory = (
        raw.groupby("layer", as_index=False)["parameter_mib"].first().sort_values("layer")
    )
    plt.figure(figsize=(12, 6))
    plt.bar(parameter_memory["layer"], parameter_memory["parameter_mib"])
    plt.title("Qwen3-4B Parameter Memory by Transformer Layer")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Parameter Memory (MiB)")
    plt.xticks(layers)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    parameter_path = FIGURE_DIR / "layer_parameter_memory.png"
    plt.savefig(parameter_path, dpi=300)
    plt.close()
    print("Saved:", parameter_path)

    prefill_activation = (
        prefill.groupby(["prompt_tokens", "layer"], as_index=False)["activation_mib"].mean()
    )
    _plot_condition_lines(
        prefill_activation,
        value_column="activation_mib",
        title="Qwen3-4B Prefill Boundary Activation Size",
        y_label="Boundary Activation Size (MiB)",
        output_name="prefill_activation_size.png",
        layers=layers,
    )

    decode_activation = (
        decode.groupby(["prompt_tokens", "layer"], as_index=False)["activation_mib"].mean()
    )
    _plot_condition_lines(
        decode_activation,
        value_column="activation_mib",
        title="Qwen3-4B Decode Boundary Activation Size",
        y_label="Boundary Activation Size (MiB)",
        output_name="decode_activation_size.png",
        layers=layers,
    )

    prefill_kv_cache = (
        prefill.groupby(["prompt_tokens", "layer"], as_index=False)["kv_cache_mib"].mean()
    )
    _plot_condition_lines(
        prefill_kv_cache,
        value_column="kv_cache_mib",
        title="Qwen3-4B Prefill KV-Cache Size",
        y_label="KV Cache Size (MiB)",
        output_name="prefill_kv_cache_size.png",
        layers=layers,
    )

    last_decode_steps = decode.groupby(["prompt_tokens", "repeat"])[
        "decode_step"
    ].transform("max")
    final_decode = decode[decode["decode_step"] == last_decode_steps]
    final_decode_kv_cache = (
        final_decode.groupby(["prompt_tokens", "layer"], as_index=False)["kv_cache_mib"].mean()
    )
    _plot_condition_lines(
        final_decode_kv_cache,
        value_column="kv_cache_mib",
        title="Qwen3-4B Final Decode-Step KV-Cache Size",
        y_label="KV Cache Size (MiB)",
        output_name="decode_kv_cache_size.png",
        layers=layers,
    )


if __name__ == "__main__":
    main()
