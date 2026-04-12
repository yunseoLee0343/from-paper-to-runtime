# From Paper to Runtime

AI 논문을 단순 요약으로 끝내지 않고, 실제 코드와 runtime 관점까지 연결해 해석하는 스터디 저장소입니다.

이 저장소의 핵심 관심사는 다음 세 가지입니다.

- 논문의 아이디어가 코드에서 어떻게 구현되는가
- 구현된 코드가 runtime에서 어떤 실행 경로를 가지는가
- 실제 병목이 연산량이 아니라 scheduler, memory access, kernel launch 정책에서 어떻게 발생하는가

## Current Status

현재는 `AlexNet`을 중심으로 첫 번째 분석 묶음을 정리해 두었습니다.

- 논문 개념 정리
- PyTorch 기반 구현 및 실험 코드
- GPU kernel/runtime 관점 분석
- NPU-style pipeline/scheduler 관점 실험

즉, 이 저장소는 아직 전체 커리큘럼이 완성된 상태가 아니라, `AlexNet`을 시작점으로 런타임 관점의 분석 틀을 만들어가는 과정에 있습니다.

## Why This Repo Exists

논문을 읽고 모델 구조를 이해하는 것만으로는 실제 시스템 병목을 설명하기 어려운 경우가 많습니다.

예를 들어 inference 경로에서:

- scheduler loop는 반복되는데
- 실제 GPU kernel launch는 발생하지 않고
- 결과적으로 GPU가 idle 상태로 남는 구간

같은 현상은, 단순히 "모델이 무겁다"로 설명되지 않습니다.

이 저장소는 이런 문제를 논문, 코드, runtime execution path를 함께 보면서 해석하기 위해 만들어졌습니다.

## What Is In The Repo

### `논문_개념/`

논문 자체의 구조, 핵심 아이디어, 역사적 의미를 정리합니다.

현재 포함:

- `AlexNet`

### `Python_논문_구현/`

논문 내용을 PyTorch 레벨에서 직접 구현하고 실험합니다.

현재 포함:

- AlexNet 2012 구조
- split-GPU 버전
- 학습 파이프라인
- 메모리 프로파일링
- benchmark / ONNX export 유틸리티

### `GPU_커널_런타임/`

GPU runtime 관점에서 AlexNet의 일부 연산을 커스텀 kernel, SASS 분석, TensorRT bucket runtime 형태로 확장해 봅니다.

현재 포함:

- fused CUDA kernel
- PyTorch extension
- SASS dump 스크립트
- runtime bucket/stream 실험
- phase별 테스트와 벤치마크

### `NPU_커널_런타임/`

NPU 스타일의 pipeline, tiling, scheduler, synchronization 관점에서 실행 모델을 실험합니다.

현재 포함:

- scheduler / pipeline 시뮬레이터
- phase별 C++ 테스트
- hardware contract 및 최적화 문서

## Study Flow

이 저장소는 대체로 아래 흐름을 따릅니다.

1. 논문 구조 이해
2. 코드로 직접 재구현
3. kernel / scheduler / memory 흐름 분석
4. 실제 runtime 병목을 설명 가능한 형태로 정리

## Repository Map

```text
.
├─ 논문_개념/
│  └─ Alexnet/
├─ Python_논문_구현/
│  ├─ alexnet.md
│  └─ alexnet.py
├─ GPU_커널_런타임/
└─ NPU_커널_런타임/
```

## Current Focus

지금 저장소에서 가장 먼저 봐야 할 것은 아래 네 곳입니다.

- `논문_개념/Alexnet/alexnet.md`
- `Python_논문_구현/alexnet.md`
- `Python_논문_구현/alexnet.py`
- `GPU_커널_런타임/README.md`

## Roadmap

이후에는 AlexNet 이후의 대표 논문들을 같은 방식으로 추가할 예정입니다.

예정 방향:

- Transformer 계열
- BERT / GPT 계열
- LoRA / RLHF / RAG
- attention vs state-space model의 runtime 차이
- scheduler / memory / kernel launch 병목 분석 축 확장

## One-line Summary

논문을 읽는 데서 멈추지 않고, 코드와 runtime까지 연결해서 실제 병목을 설명하는 스터디 저장소.
