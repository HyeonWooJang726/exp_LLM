# Qwen GPU Inference Profiler

Qwen3-4B-Instruct-2507 모델의 TTFT, TPOT, 처리량, 전체 지연 시간과 GPU 메모리를 측정합니다.

## 실행

```powershell
python profile_qwen3_4b.py
```

프로그램은 PyTorch를 불러오기 전에 `nvidia-smi`로 GPU 0의 system-wide baseline을 측정한 뒤 Qwen profiling을 시작합니다.

## 파일 구조

- `gpu_preflight.py`: 모델 로드 전 system-wide GPU baseline 측정
- `qwen_profile/config.py`: 모델명, prompt/output 길이, 반복 횟수 등 실험 설정
- `qwen_profile/model_runtime.py`: CUDA 환경 확인, 모델 로드, 입력 생성, 워밍업
- `qwen_profile/benchmark.py`: prefill/decode 측정과 개별 결과 생성
- `qwen_profile/results.py`: 평균·표준편차 집계 및 CSV/JSON 저장
- `qwen_profile/runner.py`: 위 단계를 순서대로 실행
- `qwen_profile/utils.py`: CUDA 동기화, 메모리 정리 등 공용 도구

측정 조건만 바꿀 때는 `qwen_profile/config.py`의 `DEFAULT_CONFIG`를 수정하면 됩니다.

## 모델과 결과 파일

- Qwen model weights는 Hugging Face의 기본 user cache에 저장됩니다.
- model weights 및 Hugging Face cache는 Git repository에 commit하지 않습니다.
- `venv1`을 포함한 Python 가상환경은 Git에서 제외됩니다.
- `qwen_profile_results`의 CSV와 JSON은 연구 결과이므로 Git에 저장할 수 있습니다.

VRAM 값은 측정 범위가 서로 다릅니다. `system_gpu_memory_used_mib_before_model_load`는 Windows와 다른 프로그램을 포함한 `nvidia-smi` system-wide 사용량이고, `model_loaded_vram_gib`와 `peak_vram_gib`는 현재 PyTorch 프로세스의 allocator 기준입니다.

## Layer profiling

```powershell
python profile_qwen3_4b_layers.py
```

Plot:

```powershell
python plot_qwen3_4b_layers.py
```

결과는 `qwen_layer_profile_results/`에 저장됩니다. 각 Transformer layer의 prefill latency, decode latency, parameter memory, boundary activation size, KV-cache size를 측정합니다.
