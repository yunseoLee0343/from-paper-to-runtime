# GPU Kernel Runtime Study

이 프로젝트는 AlexNet의 첫 번째 블록을 대상으로 CUDA 커스텀 커널, SASS 분석, 그리고 TensorRT 스타일 런타임 배포 흐름을 한 저장소 안에서 실험하는 예제입니다.

핵심 목표는 아래 3가지입니다.

- PyTorch C++/CUDA extension으로 `Conv1 + Bias + ReLU + MaxPool` 융합 연산을 구현한다.
- `cuobjdump`, `nvdisasm`으로 컴파일 결과를 분석하고 저수준 커널 최적화를 실험한다.
- 입력 크기별 버킷과 멀티 스트림을 사용하는 TensorRT 스타일 서빙 런타임을 구성한다.

## 구성

주요 파일은 아래와 같습니다.

- [csrc/ops.cpp](/d:/ai_study/GPU_커널_런타임/csrc/ops.cpp): `torch.library` 기반 custom op 등록
- [csrc/kernel.cu](/d:/ai_study/GPU_커널_런타임/csrc/kernel.cu): fused conv1 CUDA 커널 구현
- [setup.py](/d:/ai_study/GPU_커널_런타임/setup.py): PyTorch CUDA extension 빌드 스크립트
- [tools/dump_sass.sh](/d:/ai_study/GPU_커널_런타임/tools/dump_sass.sh): SASS 덤프 및 FFMA 비율 확인 스크립트
- [model.py](/d:/ai_study/GPU_커널_런타임/model.py): AlexNet 기반 image-to-image 모델
- [plugin_stub.py](/d:/ai_study/GPU_커널_런타임/plugin_stub.py): 커스텀 커널 삽입 지점용 plugin stub
- [compile_trt.py](/d:/ai_study/GPU_커널_런타임/compile_trt.py): 224/256/512 버킷별 엔진 컴파일 진입점
- [runtime.py](/d:/ai_study/GPU_커널_런타임/runtime.py): RequestTable 기반 버킷 배칭 및 멀티 스트림 런타임
- [tests/test_phase1.py](/d:/ai_study/GPU_커널_런타임/tests/test_phase1.py): fused op 정확성 테스트
- [tests/bench_phase2.py](/d:/ai_study/GPU_커널_런타임/tests/bench_phase2.py): PyTorch 대비 커널 벤치마크
- [tests/test_phase3.py](/d:/ai_study/GPU_커널_런타임/tests/test_phase3.py): 버킷 선택과 멀티 스트림 통합 테스트
- [tests/bench_phase3.py](/d:/ai_study/GPU_커널_런타임/tests/bench_phase3.py): end-to-end 런타임 측정

## 요구 환경

권장 환경은 Dockerfile 기준입니다.

- Python 3.x
- PyTorch with CUDA
- NVIDIA CUDA toolkit
- `cuobjdump`, `nvdisasm` 사용 가능한 CUDA 설치
- 선택 사항: `torch_tensorrt`

Docker를 쓴다면 먼저 이미지를 빌드합니다.

```bash
docker build -t gpu-kernel-runtime:dev .
```

실행 예시는 아래와 같습니다.

```bash
docker run --gpus all -it --rm -v ${PWD}:/workspace gpu-kernel-runtime:dev
```

## 빠른 시작

### 1. CUDA extension 빌드

```bash
python setup.py build_ext --inplace
```

이 단계가 성공하면 `myimg_ext` 확장이 생성되고 `torch.ops.myimg.alex_conv1_fused`를 사용할 수 있습니다.

### 2. Phase 1 정확성 테스트

```bash
pytest tests/test_phase1.py
```

테스트 내용:

- 입력 shape: `8 x 3 x 224 x 224`
- 기준 연산: `torch.nn.functional.conv2d + relu + max_pool2d`
- 비교 방식: `torch.allclose(..., rtol=1e-4, atol=1e-4)`

### 3. Phase 2 SASS 분석

```bash
bash tools/dump_sass.sh
```

생성 결과:

- `sass_dump/cuobjdump_sass.txt`
- `sass_dump/nvdisasm.txt`

이 스크립트는 아래를 확인하기 위한 출발점입니다.

- `FFMA` 명령 비율
- `ld.global.nc` 로드가 반영되었는지 여부
- `ptxas -v` 기반 레지스터 사용량

SASS 리포트 예시는 [tools/sass_report_example.md](/d:/ai_study/GPU_커널_런타임/tools/sass_report_example.md)에 정리되어 있습니다.

### 4. Phase 2 성능 벤치마크

```bash
python tests/bench_phase2.py
```

벤치마크는 두 프로파일을 시뮬레이션합니다.

- `EDGE_LOW_MEM`: 작은 배치 중심
- `A6000_HIGH_THROUGHPUT`: 큰 배치 중심

출력 예시는 아래 형태입니다.

```text
EDGE_LOW_MEM: pytorch=... ms custom=... ms speedup=...x shape=(...)
  correctness=True
A6000_HIGH_THROUGHPUT: pytorch=... ms custom=... ms speedup=...x shape=(...)
  correctness=True
```

### 5. TensorRT 스타일 엔진 준비

```bash
python compile_trt.py
```

동작 방식:

- 224, 256, 512 입력 버킷용 profile을 준비합니다.
- `torch_tensorrt`가 있으면 TensorRT compile 경로를 사용합니다.
- 없으면 eager fallback으로 동작하며 버킷 manifest만 기록합니다.

### 6. Phase 3 통합 테스트

```bash
pytest tests/test_phase3.py
```

검증 내용:

- 서로 다른 입력 크기가 올바른 버킷으로 배정되는지 확인
- 버킷별 CUDA stream이 분리되는지 확인
- 결과 tensor shape이 입력 크기와 맞는지 확인

### 7. End-to-End 런타임 측정

```bash
python tests/bench_phase3.py
```

출력 예시는 아래 형태입니다.

```text
avg_batch_ms=...
bucket_counts={'224': 2, '256': 1, '512': 1}
req=0 bucket=224 stream=stream_224 out_shape=(1, 3, 224, 224)
req=1 bucket=224 stream=stream_224 out_shape=(1, 3, 224, 224)
req=2 bucket=256 stream=stream_256 out_shape=(1, 3, 256, 256)
req=3 bucket=512 stream=stream_512 out_shape=(1, 3, 512, 512)
```

## 실행 순서 요약

처음부터 끝까지 한 번에 따라가려면 아래 순서를 권장합니다.

```bash
python setup.py build_ext --inplace
pytest tests/test_phase1.py
bash tools/dump_sass.sh
python tests/bench_phase2.py
python compile_trt.py
pytest tests/test_phase3.py
python tests/bench_phase3.py
```

## 구현 포인트

현재 커널과 런타임의 핵심 포인트는 아래와 같습니다.

- conv1 fused 커널은 `11x11 / stride 4 / padding 2`에 맞춘 direct convolution baseline입니다.
- bias, ReLU, max-pool이 커널 경로에서 함께 처리됩니다.
- 읽기 전용 데이터 로드 최적화를 위해 `ld.global.nc.f32` 인라인 PTX를 사용합니다.
- `__launch_bounds__`와 `-Xptxas=-v`로 register pressure를 관찰할 수 있게 해두었습니다.
- 런타임은 요청을 크기별로 버킷팅하고, 각 버킷을 별도 stream에 스케줄링합니다.

## 주의사항

- 현재 셸 환경에 PyTorch가 없으면 빌드와 테스트가 실패합니다.
- TensorRT 실사용 경로는 `torch_tensorrt` 설치 여부에 따라 eager fallback으로 대체될 수 있습니다.
- 이 저장소의 conv1 커널은 정확성 검증과 SASS 분석용 baseline 성격이 강합니다. 대형 GPU에서 최고 성능을 내려면 추가 타일링, shared memory, vectorized load, fp16 경로가 더 필요합니다.
