# Qwen GPU Inference Profiler

Qwen3-4B-Instruct-2507 모델의 TTFT, TPOT, 처리량, 전체 지연 시간과 GPU 메모리를 측정합니다.

## 실행

```powershell
python qwen_profile_all.py
```

기존 실행 파일은 그대로 유지하고, 실제 코드는 수정 목적에 따라 나누었습니다.

- `qwen_profile/config.py`: 모델명, prompt/output 길이, 반복 횟수 등 실험 설정
- `qwen_profile/model_runtime.py`: CUDA 환경 확인, 모델 로드, 입력 생성, 워밍업
- `qwen_profile/benchmark.py`: prefill/decode 측정과 개별 결과 생성
- `qwen_profile/results.py`: 평균·표준편차 집계 및 CSV/JSON 저장
- `qwen_profile/runner.py`: 위 단계를 순서대로 실행
- `qwen_profile/utils.py`: CUDA 동기화, 메모리 정리 등 공용 도구

측정 조건만 바꿀 때는 `qwen_profile/config.py`의 `DEFAULT_CONFIG`를 수정하면 됩니다.
