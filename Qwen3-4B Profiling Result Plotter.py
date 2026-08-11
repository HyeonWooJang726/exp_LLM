# ============================================================
# Qwen3-4B Profiling Result Plotter
# ============================================================
#
# 입력 파일:
#
# qwen_profile_results/
# ├── qwen3_4b_profile_raw.csv
# └── qwen3_4b_profile_summary.csv
#
#
# 생성 Figure:
#
# qwen_profile_results/figures/
#
# ├── fig1_ttft.png
# ├── fig2_tpot.png
# ├── fig3_throughput.png
# ├── fig4_total_latency.png
# └── fig5_peak_vram.png
#
#
# 실행:
#
# python plot_qwen_profile.py
#
# ============================================================


# ============================================================
# 1. Library Import
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 2. 파일 경로 설정
# ============================================================

RESULT_DIR = Path("qwen_profile_results")

RAW_CSV = (
    RESULT_DIR
    / "qwen3_4b_profile_raw.csv"
)

SUMMARY_CSV = (
    RESULT_DIR
    / "qwen3_4b_profile_summary.csv"
)

FIGURE_DIR = (
    RESULT_DIR
    / "figures"
)


# ============================================================
# 3. Figure 저장 폴더 생성 및 기존 그림 정리
# ============================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# 실행할 때마다 이전에 생성된 그림이 남지 않도록 먼저 삭제함.
# 현재 생성하는 PNG와 과거에 생성됐을 수 있는 PDF만 정리함.
for figure_pattern in (
    "*.png",
    "*.pdf",
):
    for existing_figure in FIGURE_DIR.glob(
        figure_pattern
    ):
        existing_figure.unlink()


# ============================================================
# 4. Profiling CSV Load
# ============================================================

raw = pd.read_csv(
    RAW_CSV
)

summary = pd.read_csv(
    SUMMARY_CSV
)


print("=" * 72)
print("Profiling data loaded")
print("=" * 72)

print()
print("RAW:")
print(raw.head())

print()
print("SUMMARY:")
print(summary)


# ============================================================
# 5. 기본 Plot 설정
# ============================================================
#
# 논문에 넣기 편하도록:
#
# - font size 일정
# - grid 사용
# - 300 DPI PNG
# - PDF 생성은 비활성화하고 PNG만 저장
#
# ============================================================

plt.rcParams.update({

    "font.size": 11,

    "axes.labelsize": 12,

    "axes.titlesize": 13,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

})


# ============================================================
# 6. Figure 저장 함수
# ============================================================

def save_figure(
    figure,
    filename,
):

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    png_path = (
        FIGURE_DIR
        / f"{filename}.png"
    )

    figure.savefig(

        png_path,

        dpi=300,

        bbox_inches="tight",
    )


    # --------------------------------------------------------
    # PDF 생성 비활성화
    # --------------------------------------------------------
    # PDF 파일이 생성되지 않도록 아래 저장 코드를 주석 처리함.
    # 필요한 경우 아래 코드의 주석을 해제하면 PDF 저장을 다시 활성화할 수 있음.
    #
    # pdf_path = (
    #     FIGURE_DIR
    #     / f"{filename}.pdf"
    # )
    #
    # figure.savefig(
    #
    #     pdf_path,
    #
    #     bbox_inches="tight",
    # )


    print(
        f"Saved: {png_path}"
    )

    # PDF를 저장하지 않으므로 PDF 저장 완료 출력도 주석 처리함.
    # print(
    #     f"Saved: {pdf_path}"
    # )


# ============================================================
# 7. Figure 1
# Prompt Length vs TTFT
# ============================================================
#
# TTFT는 Output length가 결정되기 전에 측정되는 값임.
#
# 따라서 output=32와 output=128을 분리하지 않고
# 같은 Prompt Length의 모든 반복 데이터를 합쳐서
# 평균 + 표준편차를 계산.
#
# ============================================================

ttft = (

    raw.groupby(
        "prompt_tokens"
    )

    .agg(

        mean=(
            "ttft_ms",
            "mean",
        ),

        std=(
            "ttft_ms",
            "std",
        ),
    )

    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(6.4, 4.5)
)


ax.errorbar(

    ttft["prompt_tokens"],

    ttft["mean"],

    yerr=ttft["std"],

    marker="o",

    linewidth=2,

    capsize=4,
)


ax.set_xlabel(
    "Prompt Length (tokens)"
)

ax.set_ylabel(
    "TTFT (ms)"
)

ax.set_title(
    "Time to First Token vs. Prompt Length"
)

ax.set_xticks(
    ttft["prompt_tokens"]
)

ax.grid(
    True,
    alpha=0.3,
)


save_figure(
    fig,
    "fig1_ttft",
)

plt.close(fig)


# ============================================================
# 8. Figure 2
# Prompt Length vs TPOT
# ============================================================
#
# TPOT = Time Per Output Token
#
# Output 32 tokens
# Output 128 tokens
#
# 두 조건을 각각 Line으로 표시
#
# error bar = 3회 반복의 표준편차
#
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.5)
)


for output_tokens in sorted(
    summary["output_tokens"].unique()
):

    data = summary[
        summary["output_tokens"]
        == output_tokens
    ]


    ax.errorbar(

        data["prompt_tokens"],

        data["tpot_ms_mean"],

        yerr=data["tpot_ms_std"],

        marker="o",

        linewidth=2,

        capsize=4,

        label=(
            f"Output = "
            f"{output_tokens} tokens"
        ),
    )


ax.set_xlabel(
    "Prompt Length (tokens)"
)

ax.set_ylabel(
    "TPOT (ms/token)"
)

ax.set_title(
    "Time Per Output Token vs. Prompt Length"
)

ax.set_xticks(
    sorted(
        summary[
            "prompt_tokens"
        ].unique()
    )
)

ax.legend()

ax.grid(
    True,
    alpha=0.3,
)


save_figure(
    fig,
    "fig2_tpot",
)

plt.close(fig)


# ============================================================
# 9. Figure 3
# Prompt Length vs Decode Throughput
# ============================================================
#
# Throughput:
#
# tokens / second
#
# 높을수록 빠른 것
#
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.5)
)


for output_tokens in sorted(
    summary["output_tokens"].unique()
):

    data = summary[
        summary["output_tokens"]
        == output_tokens
    ]


    ax.errorbar(

        data["prompt_tokens"],

        data[
            "tokens_per_second_mean"
        ],

        yerr=data[
            "tokens_per_second_std"
        ],

        marker="o",

        linewidth=2,

        capsize=4,

        label=(
            f"Output = "
            f"{output_tokens} tokens"
        ),
    )


ax.set_xlabel(
    "Prompt Length (tokens)"
)

ax.set_ylabel(
    "Decode Throughput (tokens/s)"
)

ax.set_title(
    "Decode Throughput vs. Prompt Length"
)

ax.set_xticks(
    sorted(
        summary[
            "prompt_tokens"
        ].unique()
    )
)

ax.legend()

ax.grid(
    True,
    alpha=0.3,
)


save_figure(
    fig,
    "fig3_throughput",
)

plt.close(fig)


# ============================================================
# 10. Figure 4
# Prompt Length vs Total Latency
# ============================================================
#
# Total Latency =
#
# TTFT
# +
# 모든 Decode token 생성 시간
#
# Output Length의 영향이 매우 크므로
# 32 / 128 token을 반드시 분리해서 표시
#
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.4, 4.5)
)


for output_tokens in sorted(
    summary["output_tokens"].unique()
):

    data = summary[
        summary["output_tokens"]
        == output_tokens
    ]


    # --------------------------------------------------------
    # ms -> sec 변환
    # --------------------------------------------------------

    mean_seconds = (

        data[
            "total_latency_ms_mean"
        ]

        / 1000
    )


    std_seconds = (

        data[
            "total_latency_ms_std"
        ]

        / 1000
    )


    ax.errorbar(

        data["prompt_tokens"],

        mean_seconds,

        yerr=std_seconds,

        marker="o",

        linewidth=2,

        capsize=4,

        label=(
            f"Output = "
            f"{output_tokens} tokens"
        ),
    )


ax.set_xlabel(
    "Prompt Length (tokens)"
)

ax.set_ylabel(
    "Total Latency (s)"
)

ax.set_title(
    "Total Model Inference Latency"
)

ax.set_xticks(
    sorted(
        summary[
            "prompt_tokens"
        ].unique()
    )
)

ax.legend()

ax.grid(
    True,
    alpha=0.3,
)


save_figure(
    fig,
    "fig4_total_latency",
)

plt.close(fig)


# ============================================================
# 11. Figure 5
# Prompt Length vs Peak VRAM
# ============================================================
#
# 현재 측정 결과에서는 동일 Prompt Length에서
# Output 32 / 128의 Peak VRAM이 같음.
#
# 따라서 Output Length별로 중복해서 그리지 않고
# Prompt Length별 하나의 값으로 표현.
#
# ============================================================

vram = (

    raw.groupby(
        "prompt_tokens"
    )

    .agg(

        mean=(
            "peak_vram_gib",
            "mean",
        ),

        std=(
            "peak_vram_gib",
            "std",
        ),
    )

    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(6.4, 4.5)
)


ax.errorbar(

    vram["prompt_tokens"],

    vram["mean"],

    yerr=vram["std"],

    marker="o",

    linewidth=2,

    capsize=4,
)


ax.set_xlabel(
    "Prompt Length (tokens)"
)

ax.set_ylabel(
    "Peak GPU Memory (GiB)"
)

ax.set_title(
    "Peak PyTorch GPU Memory by Prompt Length"
)

ax.set_xticks(
    vram[
        "prompt_tokens"
    ]
)

ax.grid(
    True,
    alpha=0.3,
)


save_figure(
    fig,
    "fig5_peak_vram",
)

plt.close(fig)


# ============================================================
# 12. 종료 메시지
# ============================================================

print()

print("=" * 72)

print(
    "All figures generated successfully."
)

print("=" * 72)

print(
    f"Figure directory: "
    f"{FIGURE_DIR.resolve()}"
)
