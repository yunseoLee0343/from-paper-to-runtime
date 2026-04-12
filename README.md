# From Paper to Runtime

AI 논문을 읽는 데서 멈추지 않고, 코드와 runtime execution까지 연결해서 해석하는 스터디 저장소입니다.

이 저장소는 특히 아래 질문에 집중합니다.

- 논문의 핵심 아이디어가 실제 코드에서 어떻게 구현되는가
- 구현된 모델이 runtime에서 어떤 execution path를 가지는가
- 실제 병목이 compute가 아니라 scheduler, memory access, kernel launch 정책에서 어떻게 생기는가

## Repository Structure

```text
.
├─ README.md
├─ docs/
│  ├─ runtime-methodology.md
│  └─ study-roadmap.md
└─ studies/
   └─ alexnet/
      ├─ README.md
      ├─ paper/
      ├─ pytorch/
      ├─ gpu_runtime/
      └─ npu_runtime/
```

## How To Read This Repo

처음 보는 경우에는 아래 순서가 가장 좋습니다.

1. [루트 README](/d:/ai_study/README.md)
2. [공통 방법론](/d:/ai_study/docs/runtime-methodology.md)
3. [AlexNet 스터디 개요](/d:/ai_study/studies/alexnet/README.md)

## Current Status

현재는 `AlexNet` 스터디가 정리되어 있습니다.

- 논문 개념 정리
- PyTorch 재구현
- GPU kernel/runtime 실험
- NPU-style scheduler/pipeline 실험

이후에는 같은 구조로 다른 논문들을 추가할 예정입니다.

## Study Philosophy

이 저장소의 기본 흐름은 다음과 같습니다.

1. 논문 구조 이해
2. 코드 매핑
3. runtime execution 분석
4. 병목 원인 설명 가능 상태까지 정리

한 줄로 요약하면, 논문을 "읽는 저장소"가 아니라 "실행 관점으로 해석하는 저장소"입니다.
