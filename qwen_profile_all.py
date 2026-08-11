# ============================================================
# Qwen3-4B-Instruct-2507 GPU Inference Profiler
# ============================================================
#
# 목적:
#   Qwen3-4B-Instruct-2507을 RTX GPU에서 BF16으로 실행하고
#   입력/출력 token 길이에 따른 inference 성능을 측정
#
# 측정 항목:
#   1. 모델 Loading Time
#   2. 모델 Parameter 수
#   3. 모델 VRAM
#   4. TTFT (Time To First Token)
#   5. TPOT (Time Per Output Token)
#   6. Tokens/s
#   7. Total Latency
#   8. Peak VRAM
#
# 실험 조건:
#   Model      : Qwen3-4B-Instruct-2507
#   Precision  : BF16
#   Batch Size : 1
#   KV Cache   : ON
#   Decoding   : Greedy
#
# 실행:
#   python qwen_profile_all.py
#
# ============================================================


# ============================================================
# 1. 라이브러리 Import
# ============================================================

import gc
import json
import os
import platform
import sys
import time

import pandas as pd
import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# 2. 실험 설정
# ============================================================

# 사용할 Hugging Face 모델
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# ------------------------------------------------------------
# 입력 Prompt 길이
# ------------------------------------------------------------

PROMPT_LENGTHS = [
    128,
    512,
    1024,
    2048,
]


# ------------------------------------------------------------
# 생성할 Output 길이
# ------------------------------------------------------------

OUTPUT_LENGTHS = [
    32,
    128,
]


# ------------------------------------------------------------
# 본 측정 전에 수행할 Warm-up 횟수
# ------------------------------------------------------------

WARMUP_RUNS = 2


# ------------------------------------------------------------
# 각 실험 조건 반복 횟수
# ------------------------------------------------------------

REPEATS = 3


# ------------------------------------------------------------
# 결과 저장 폴더
# ------------------------------------------------------------

OUTPUT_DIR = "qwen_profile_results"


# ------------------------------------------------------------
# 모델 Precision
# ------------------------------------------------------------

DTYPE = torch.bfloat16


# ============================================================
# 3. Utility 함수
# ============================================================

def cuda_sync():
    """
    GPU 연산이 모두 끝날 때까지 대기.

    CUDA는 기본적으로 asynchronous하게 실행되므로
    정확한 latency 측정을 위해 반드시 synchronization 필요.
    """
    torch.cuda.synchronize()


def bytes_to_gib(byte_value):
    """
    Byte -> GiB 변환
    """
    return byte_value / (1024 ** 3)


def print_section(title):
    """
    터미널 출력 구분용
    """
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# 4. 결과 저장 폴더 생성
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 5. Python / PyTorch / GPU 환경 확인
# ============================================================

print_section("STEP 1. 실행 환경 확인")


print("Python        :", sys.version.split()[0])
print("OS            :", platform.platform())

print("PyTorch       :", torch.__version__)
print("Transformers  :", transformers.__version__)

print("Torch CUDA    :", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())


# ------------------------------------------------------------
# CUDA 사용 가능 여부 검사
# ------------------------------------------------------------

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU를 사용할 수 없습니다.\n"
        "PyTorch CUDA 환경을 확인하세요."
    )


# ------------------------------------------------------------
# GPU 정보
# ------------------------------------------------------------

gpu_name = torch.cuda.get_device_name(0)

gpu_properties = torch.cuda.get_device_properties(0)

total_vram = bytes_to_gib(
    gpu_properties.total_memory
)


print("GPU           :", gpu_name)
print("GPU VRAM      :", f"{total_vram:.2f} GiB")


# ============================================================
# 6. GPU Memory 초기화
# ============================================================

gc.collect()

torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()


# ============================================================
# 7. Tokenizer 다운로드 / Load
# ============================================================

print_section("STEP 2. Tokenizer Load")


print("Model:", MODEL_NAME)

print(
    "첫 실행이면 Hugging Face에서 "
    "Tokenizer와 모델 파일을 자동 다운로드합니다."
)


tokenizer_start = time.perf_counter()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


tokenizer_time = (
    time.perf_counter()
    - tokenizer_start
)


print(
    f"Tokenizer load time: "
    f"{tokenizer_time:.2f} sec"
)


# ============================================================
# 8. Qwen3-4B 모델 다운로드 / GPU Load
# ============================================================

print_section("STEP 3. Qwen3-4B 모델 Load")


print("Precision : BF16")
print("Device    : CUDA")


# ------------------------------------------------------------
# 모델 Loading 시작
# ------------------------------------------------------------

load_start = time.perf_counter()


model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    # BF16
    dtype=DTYPE,

    # CPU RAM 사용량 절약
    low_cpu_mem_usage=True,
)


# ------------------------------------------------------------
# 전체 모델을 GPU로 이동
# ------------------------------------------------------------

model = model.to("cuda")


# ------------------------------------------------------------
# Inference mode
# ------------------------------------------------------------

model.eval()


# ------------------------------------------------------------
# GPU 작업 완료 대기
# ------------------------------------------------------------

cuda_sync()


# ------------------------------------------------------------
# Loading 시간 계산
# ------------------------------------------------------------

load_time = (
    time.perf_counter()
    - load_start
)


# ------------------------------------------------------------
# 모델 Loading 후 GPU Memory
# ------------------------------------------------------------

model_vram = torch.cuda.memory_allocated()


print("Model loaded successfully.")

print(
    "Model load time :",
    f"{load_time:.2f} sec"
)

print(
    "Model VRAM      :",
    f"{bytes_to_gib(model_vram):.2f} GiB"
)


# ============================================================
# 9. 모델 Architecture 정보 확인
# ============================================================

print_section("STEP 4. 모델 정보")


# ------------------------------------------------------------
# Transformer Layer 수
# ------------------------------------------------------------

num_layers = model.config.num_hidden_layers


# ------------------------------------------------------------
# Hidden dimension
# ------------------------------------------------------------

hidden_size = model.config.hidden_size


# ------------------------------------------------------------
# Parameter 수
# ------------------------------------------------------------

parameter_count = sum(

    parameter.numel()

    for parameter in model.parameters()
)


# ------------------------------------------------------------
# Parameter가 차지하는 실제 Memory
# ------------------------------------------------------------

parameter_bytes = sum(

    parameter.numel()
    * parameter.element_size()

    for parameter in model.parameters()
)


print(
    "Transformer Layers :",
    num_layers
)

print(
    "Hidden Size        :",
    hidden_size
)

print(
    "Parameters         :",
    f"{parameter_count / 1e9:.3f} B"
)

print(
    "Parameter Memory   :",
    f"{bytes_to_gib(parameter_bytes):.2f} GiB"
)


# ============================================================
# 10. Synthetic Prompt 생성
# ============================================================

print_section("STEP 5. Profiling 입력 생성")


# ------------------------------------------------------------
# 실제 의미는 중요하지 않음.
#
# 이번 실험에서는 "정확히 N token 입력"을 만드는 것이 목적.
# ------------------------------------------------------------

base_text = (

    "Large language models use transformer architectures "
    "to perform autoregressive text generation. "

    "Artificial intelligence applications require computation, "
    "memory, and efficient inference systems. "

    "Edge computing provides computational resources "
    "close to users and data sources. "
)


# ------------------------------------------------------------
# 충분히 긴 문자열 생성
# ------------------------------------------------------------

large_text = base_text * 10000


# ------------------------------------------------------------
# 전체 문자열 Tokenization
#
# CPU Memory에 저장
# ------------------------------------------------------------

token_pool = tokenizer(

    large_text,

    return_tensors="pt",

    add_special_tokens=False,

)["input_ids"][0]


# ------------------------------------------------------------
# 필요한 최대 Prompt보다 충분히 긴지 확인
# ------------------------------------------------------------

if len(token_pool) < max(PROMPT_LENGTHS):

    raise RuntimeError(
        "Synthetic prompt token이 충분하지 않습니다."
    )


# ============================================================
# 11. 정확히 N개의 Token을 GPU Input으로 만드는 함수
# ============================================================

def make_input(prompt_length):

    input_ids = (

        token_pool[:prompt_length]

        .clone()

        .unsqueeze(0)

        .to("cuda")
    )

    return input_ids


print(
    "Prompt lengths :",
    PROMPT_LENGTHS
)

print(
    "Output lengths :",
    OUTPUT_LENGTHS
)

print(
    "Warm-up runs   :",
    WARMUP_RUNS
)

print(
    "Repeats        :",
    REPEATS
)


total_cases = (

    len(PROMPT_LENGTHS)
    * len(OUTPUT_LENGTHS)
    * REPEATS
)


print(
    "Total profiling runs:",
    total_cases
)


# ============================================================
# 12. Smoke Test
# ============================================================

print_section("STEP 6. Smoke Test")


# ------------------------------------------------------------
# 아주 짧은 입력으로 Model Forward가 정상인지 검사
# ------------------------------------------------------------

smoke_input = make_input(32)


with torch.inference_mode():

    smoke_output = model(

        input_ids=smoke_input,

        use_cache=True,
    )


cuda_sync()


print("Forward pass 성공.")

print(
    "Logits shape:",
    tuple(smoke_output.logits.shape)
)


# ------------------------------------------------------------
# Smoke Test tensor 제거
# ------------------------------------------------------------

del smoke_input
del smoke_output


gc.collect()

torch.cuda.empty_cache()


# ============================================================
# 13. GPU Warm-up
# ============================================================

print_section("STEP 7. GPU Warm-up")


# ------------------------------------------------------------
# 첫 CUDA kernel 실행 등에서 발생하는
# 초기화 overhead를 본 실험에서 제외하기 위함
# ------------------------------------------------------------

for warmup_index in range(WARMUP_RUNS):

    warm_input = make_input(128)

    with torch.inference_mode():

        warm_output = model(

            input_ids=warm_input,

            use_cache=True,
        )

    cuda_sync()


    print(
        f"Warm-up "
        f"{warmup_index + 1}/"
        f"{WARMUP_RUNS}"
    )


    # --------------------------------------------------------
    # 중요:
    # 마지막 Warm-up output까지 완전히 제거
    # --------------------------------------------------------

    del warm_input
    del warm_output


    gc.collect()

    torch.cuda.empty_cache()


print("Warm-up 완료.")


# ============================================================
# 14. 본 Profiling 시작
# ============================================================

print_section("STEP 8. Profiling 시작")


results = []

current_case = 0


# ============================================================
# 15. Prompt Length 반복
# ============================================================

for prompt_length in PROMPT_LENGTHS:


    # ========================================================
    # 16. Output Length 반복
    # ========================================================

    for output_length in OUTPUT_LENGTHS:


        # ====================================================
        # 17. 동일 조건 반복 측정
        # ====================================================

        for repeat in range(
            1,
            REPEATS + 1
        ):


            current_case += 1


            print()
            print("-" * 72)

            print(
                f"[{current_case}/{total_cases}] "
                f"Prompt={prompt_length} | "
                f"Output={output_length} | "
                f"Run={repeat}"
            )

            print("-" * 72)


            # =================================================
            # 18. 이전 실험 GPU Memory 정리
            # =================================================

            gc.collect()

            torch.cuda.empty_cache()


            # -------------------------------------------------
            # Peak memory 측정 초기화
            # -------------------------------------------------

            torch.cuda.reset_peak_memory_stats()


            # =================================================
            # 19. Input 생성
            # =================================================

            input_ids = make_input(
                prompt_length
            )


            # -------------------------------------------------
            # Prefill 직전 GPU memory
            # -------------------------------------------------

            memory_before_prefill = (
                torch.cuda.memory_allocated()
            )


            # =================================================
            # 20. PREFILL 측정
            # =================================================
            #
            # Prompt 전체를 처리하고
            # 첫 번째 Output Token을 계산
            #
            # 여기서 측정한 시간을 TTFT로 사용
            # =================================================

            cuda_sync()


            prefill_start = (
                time.perf_counter()
            )


            with torch.inference_mode():

                outputs = model(

                    input_ids=input_ids,

                    use_cache=True,
                )


                # ---------------------------------------------
                # 마지막 position의 logits에서
                # 가장 확률 높은 token 선택
                #
                # Greedy decoding
                # ---------------------------------------------

                first_token = torch.argmax(

                    outputs.logits[:, -1, :],

                    dim=-1,

                    keepdim=True,
                )


            cuda_sync()


            prefill_end = (
                time.perf_counter()
            )


            # -------------------------------------------------
            # Time To First Token
            # -------------------------------------------------

            ttft_sec = (

                prefill_end
                - prefill_start
            )


            # -------------------------------------------------
            # Prefill에서 생성된 KV Cache
            # -------------------------------------------------

            past_key_values = (
                outputs.past_key_values
            )


            # -------------------------------------------------
            # Prefill 이후 GPU memory
            # -------------------------------------------------

            memory_after_prefill = (
                torch.cuda.memory_allocated()
            )


            # =================================================
            # 21. AUTOREGRESSIVE DECODE 측정
            # =================================================
            #
            # Prefill에서 첫 번째 token은 이미 생성됨.
            #
            # 따라서 나머지:
            #
            # output_length - 1
            #
            # 개 token 생성
            # =================================================

            next_token = first_token


            decode_times = []


            with torch.inference_mode():


                for _ in range(
                    output_length - 1
                ):


                    # -----------------------------------------
                    # Decode 시작 전 Synchronization
                    # -----------------------------------------

                    cuda_sync()


                    decode_start = (
                        time.perf_counter()
                    )


                    # -----------------------------------------
                    # 이전 KV Cache를 재사용하여
                    # 다음 Token 한 개 계산
                    # -----------------------------------------

                    outputs = model(

                        input_ids=next_token,

                        past_key_values=
                            past_key_values,

                        use_cache=True,
                    )


                    # -----------------------------------------
                    # Greedy Token 선택
                    # -----------------------------------------

                    next_token = torch.argmax(

                        outputs.logits[:, -1, :],

                        dim=-1,

                        keepdim=True,
                    )


                    # -----------------------------------------
                    # 증가된 KV Cache 저장
                    # -----------------------------------------

                    past_key_values = (
                        outputs.past_key_values
                    )


                    # -----------------------------------------
                    # Decode GPU 연산 완료 대기
                    # -----------------------------------------

                    cuda_sync()


                    decode_end = (
                        time.perf_counter()
                    )


                    # -----------------------------------------
                    # Token 하나 생성 시간 저장
                    # -----------------------------------------

                    decode_times.append(

                        decode_end
                        - decode_start
                    )


            # =================================================
            # 22. Decode 결과 계산
            # =================================================

            decode_total_sec = sum(
                decode_times
            )


            # -------------------------------------------------
            # TPOT
            #
            # Time Per Output Token
            # -------------------------------------------------

            if len(decode_times) > 0:

                tpot_sec = (

                    decode_total_sec
                    / len(decode_times)
                )


                # ---------------------------------------------
                # Tokens per Second
                # ---------------------------------------------

                tokens_per_second = (

                    1.0
                    / tpot_sec
                )


            else:

                tpot_sec = 0.0

                tokens_per_second = 0.0


            # =================================================
            # 23. Total Latency 계산
            # =================================================

            total_latency_sec = (

                ttft_sec
                + decode_total_sec
            )


            # =================================================
            # 24. Peak GPU Memory
            # =================================================

            peak_vram = (
                torch.cuda.max_memory_allocated()
            )


            # =================================================
            # 25. Prefill 이후 Memory 증가량
            # =================================================
            #
            # 주의:
            #
            # 이것을 "순수 KV Cache 크기"라고 부르면 안 됨.
            #
            # KV Cache 이외의 tensor도 포함될 수 있음.
            # =================================================

            prefill_memory_delta = (

                memory_after_prefill
                - memory_before_prefill
            )


            # =================================================
            # 26. 측정 결과 Record
            # =================================================

            result = {


                "prompt_tokens":
                    prompt_length,


                "output_tokens":
                    output_length,


                "repeat":
                    repeat,


                # ---------------------------------------------
                # First Token latency
                # ---------------------------------------------

                "ttft_ms":
                    ttft_sec * 1000,


                # ---------------------------------------------
                # Decode 전체 시간
                # ---------------------------------------------

                "decode_total_ms":
                    decode_total_sec * 1000,


                # ---------------------------------------------
                # 평균 Decode Token latency
                # ---------------------------------------------

                "tpot_ms":
                    tpot_sec * 1000,


                # ---------------------------------------------
                # Decode throughput
                # ---------------------------------------------

                "tokens_per_second":
                    tokens_per_second,


                # ---------------------------------------------
                # 전체 inference latency
                # ---------------------------------------------

                "total_latency_ms":
                    total_latency_sec * 1000,


                # ---------------------------------------------
                # 모델 Loading 후 VRAM
                # ---------------------------------------------

                "model_vram_gib":
                    bytes_to_gib(
                        model_vram
                    ),


                # ---------------------------------------------
                # Prefill 직전 VRAM
                # ---------------------------------------------

                "memory_before_prefill_gib":
                    bytes_to_gib(
                        memory_before_prefill
                    ),


                # ---------------------------------------------
                # Prefill 직후 VRAM
                # ---------------------------------------------

                "memory_after_prefill_gib":
                    bytes_to_gib(
                        memory_after_prefill
                    ),


                # ---------------------------------------------
                # Prefill Memory 증가량
                # ---------------------------------------------

                "prefill_memory_delta_gib":
                    bytes_to_gib(
                        prefill_memory_delta
                    ),


                # ---------------------------------------------
                # 실행 중 Peak VRAM
                # ---------------------------------------------

                "peak_vram_gib":
                    bytes_to_gib(
                        peak_vram
                    ),
            }


            results.append(
                result
            )


            # =================================================
            # 27. 현재 실험 결과 화면 출력
            # =================================================

            print(
                f"TTFT        : "
                f"{ttft_sec * 1000:.2f} ms"
            )


            print(
                f"TPOT        : "
                f"{tpot_sec * 1000:.2f} ms/token"
            )


            print(
                f"Throughput  : "
                f"{tokens_per_second:.2f} token/s"
            )


            print(
                f"Total       : "
                f"{total_latency_sec:.3f} sec"
            )


            print(
                f"Peak VRAM   : "
                f"{bytes_to_gib(peak_vram):.2f} GiB"
            )


            # =================================================
            # 28. 현재 실험 Tensor 제거
            # =================================================

            del input_ids
            del outputs
            del first_token
            del next_token
            del past_key_values
            del decode_times


            gc.collect()

            torch.cuda.empty_cache()


# ============================================================
# 29. Raw Profiling 결과 DataFrame 생성
# ============================================================

print_section("STEP 9. 결과 저장")


df = pd.DataFrame(
    results
)


# ============================================================
# 30. Raw CSV 저장
# ============================================================

raw_csv_path = os.path.join(

    OUTPUT_DIR,

    "qwen3_4b_profile_raw.csv",
)


df.to_csv(

    raw_csv_path,

    index=False,
)


# ============================================================
# 31. 평균 / 표준편차 계산
# ============================================================

summary = (

    df.groupby(

        [
            "prompt_tokens",
            "output_tokens",
        ]
    )

    .agg(


        # ----------------------------------------------------
        # TTFT
        # ----------------------------------------------------

        ttft_ms_mean=(

            "ttft_ms",

            "mean",
        ),


        ttft_ms_std=(

            "ttft_ms",

            "std",
        ),


        # ----------------------------------------------------
        # TPOT
        # ----------------------------------------------------

        tpot_ms_mean=(

            "tpot_ms",

            "mean",
        ),


        tpot_ms_std=(

            "tpot_ms",

            "std",
        ),


        # ----------------------------------------------------
        # Throughput
        # ----------------------------------------------------

        tokens_per_second_mean=(

            "tokens_per_second",

            "mean",
        ),


        tokens_per_second_std=(

            "tokens_per_second",

            "std",
        ),


        # ----------------------------------------------------
        # Total latency
        # ----------------------------------------------------

        total_latency_ms_mean=(

            "total_latency_ms",

            "mean",
        ),


        total_latency_ms_std=(

            "total_latency_ms",

            "std",
        ),


        # ----------------------------------------------------
        # Peak VRAM
        # ----------------------------------------------------

        peak_vram_gib_mean=(

            "peak_vram_gib",

            "mean",
        ),


        peak_vram_gib_std=(

            "peak_vram_gib",

            "std",
        ),

    )

    .reset_index()
)


# ============================================================
# 32. Summary CSV 저장
# ============================================================

summary_csv_path = os.path.join(

    OUTPUT_DIR,

    "qwen3_4b_profile_summary.csv",
)


summary.to_csv(

    summary_csv_path,

    index=False,
)


# ============================================================
# 33. 실험 환경 정보 저장
# ============================================================

environment = {


    "model":
        MODEL_NAME,


    "precision":
        "BF16",


    "batch_size":
        1,


    "kv_cache":
        True,


    "decoding":
        "greedy",


    "gpu":
        gpu_name,


    "gpu_vram_gib":
        total_vram,


    "python":
        sys.version,


    "pytorch":
        torch.__version__,


    "cuda":
        torch.version.cuda,


    "transformers":
        transformers.__version__,


    "model_parameters":
        parameter_count,


    "model_parameter_memory_gib":
        bytes_to_gib(
            parameter_bytes
        ),


    "model_loaded_vram_gib":
        bytes_to_gib(
            model_vram
        ),


    "tokenizer_load_time_sec":
        tokenizer_time,


    "model_load_time_sec":
        load_time,


    "prompt_lengths":
        PROMPT_LENGTHS,


    "output_lengths":
        OUTPUT_LENGTHS,


    "warmup_runs":
        WARMUP_RUNS,


    "repeats":
        REPEATS,
}


# ============================================================
# 34. Environment JSON 저장
# ============================================================

environment_path = os.path.join(

    OUTPUT_DIR,

    "environment.json",
)


with open(

    environment_path,

    "w",

    encoding="utf-8",

) as file:


    json.dump(

        environment,

        file,

        indent=4,

        ensure_ascii=False,
    )


# ============================================================
# 35. 최종 결과 출력
# ============================================================

print_section("PROFILING 완료")


pd.set_option(

    "display.max_columns",

    None,
)


pd.set_option(

    "display.width",

    220,
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 36. 생성된 파일 출력
# ============================================================

print()

print("=" * 72)

print("저장된 파일")

print("=" * 72)


print(
    "RAW     :",
    raw_csv_path
)

print(
    "SUMMARY :",
    summary_csv_path
)

print(
    "ENV     :",
    environment_path
)


# ============================================================
# 37. 핵심 Profiling 지표 안내
# ============================================================

print()

print("=" * 72)

print("핵심적으로 볼 값")

print("=" * 72)


print(
    "1. ttft_ms_mean"
    "             -> 첫 Token까지 걸린 시간"
)

print(
    "2. tpot_ms_mean"
    "             -> 이후 Token 하나 생성 시간"
)

print(
    "3. tokens_per_second_mean"
    " -> 초당 Decode Token 수"
)

print(
    "4. total_latency_ms_mean"
    "    -> 요청 전체 처리시간"
)

print(
    "5. peak_vram_gib_mean"
    "       -> 최대 GPU Memory"
)


print()

print("모든 Profiling이 완료되었습니다.")