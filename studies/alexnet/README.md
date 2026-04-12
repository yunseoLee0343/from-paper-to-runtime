# AlexNet Study

AlexNet은 이 저장소에서 첫 번째로 정리한 스터디입니다.

이 스터디의 목적은 단순히 "초기 CNN 구조를 이해하는 것"이 아니라, AlexNet이 당시의 하드웨어 제약과 어떻게 맞물려 있었는지, 그리고 그 구조가 코드와 runtime 관점에서 어떻게 해석되는지를 연결해 보는 것입니다.

## What Is Included

### `paper/`

논문 개념과 역사적 맥락을 정리합니다.

- [paper/notes.md](/d:/ai_study/studies/alexnet/paper/notes.md)

### `pytorch/`

PyTorch 레벨에서 AlexNet을 재구현하고, 학습/프로파일링/벤치마크 실험 코드를 정리합니다.

- [pytorch/notes.md](/d:/ai_study/studies/alexnet/pytorch/notes.md)
- [pytorch/alexnet.py](/d:/ai_study/studies/alexnet/pytorch/alexnet.py)

포함 내용:

- AlexNet 2012 구조
- split-GPU 버전
- 학습 파이프라인
- 메모리 프로파일링
- benchmark / ONNX export

### `gpu_runtime/`

GPU kernel/runtime 관점에서 AlexNet 일부 연산을 실험합니다.

- [gpu_runtime/README.md](/d:/ai_study/studies/alexnet/gpu_runtime/README.md)

포함 내용:

- fused CUDA kernel
- PyTorch extension
- SASS dump
- bucket/stream runtime 실험
- phase별 테스트

### `npu_runtime/`

NPU-style scheduler, pipeline, tiling, synchronization 관점의 실행 모델을 실험합니다.

- [npu_runtime/README.md](/d:/ai_study/studies/alexnet/npu_runtime/README.md)

포함 내용:

- scheduler / pipeline 코드
- hardware contract 문서
- phase별 C++ 테스트
- raw AlexNet NPU-style 실험 코드

## Recommended Reading Order

1. [paper/notes.md](/d:/ai_study/studies/alexnet/paper/notes.md)
2. [pytorch/notes.md](/d:/ai_study/studies/alexnet/pytorch/notes.md)
3. [pytorch/alexnet.py](/d:/ai_study/studies/alexnet/pytorch/alexnet.py)
4. [gpu_runtime/README.md](/d:/ai_study/studies/alexnet/gpu_runtime/README.md)
5. [npu_runtime/README.md](/d:/ai_study/studies/alexnet/npu_runtime/README.md)

## Why AlexNet First

AlexNet은 구조적으로는 오래된 모델이지만, 실행 관점에서는 여전히 좋은 출발점입니다.

- convolution 기반 구조를 명확하게 볼 수 있고
- split-GPU 같은 당시의 하드웨어 제약 대응을 확인할 수 있으며
- 이후 모델과 비교할 기준점을 만들기 좋기 때문입니다.

이 스터디는 AlexNet을 "초기 CNN"으로만 보지 않고, model-system co-design의 초기 사례로 읽는 것을 목표로 합니다.
