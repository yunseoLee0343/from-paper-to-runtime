# GPU AlexNet Optimization Report

## 문서 개요

본 문서는 GPU 기반 AlexNet 가속 파이프라인의 Phase 1~3 결과를 하나의 기술 백서 형태로 정리한 문서다. 대상 범위는 다음 세 축으로 구성된다.

- PyTorch ATen/CUDA extension 기반 `alex_conv1_fused` 구현
- SASS 레벨 분석 및 저수준 커널 최적화
- vLLM 스타일 GPU 런타임과 TensorRT 배포 전략

이 보고서는 현재 저장소의 구현 코드와 참고 설계 파일을 함께 반영한다.

- 구현 코드: `csrc/ops.cpp`, `csrc/kernel.cu`, `setup.py`, `plugin_stub.py`, `compile_trt.py`, `runtime.py`
- 참고 설계: `alexnet_raw_cuda_dual_profile.cu`, `alexnet_raw_cuda_dual_profile_runtime.cu`, `detailed.txt`, `detailed2.txt`

---

## 1. 고성능 CUDA 커널 설계 (Kernel Fusion & PTX)

### 1.1 Fused Kernel 설계 목표

Phase 1의 핵심 목표는 AlexNet의 첫 번째 블록인

- Convolution
- Bias Add
- ReLU
- MaxPool

을 하나의 커스텀 경로로 묶어 프레임워크 경계와 중간 메모리 왕복을 줄이는 데 있다. 저장소 구현에서 이 경로는 `torch.ops.myimg.alex_conv1_fused`로 노출되며, 등록 지점은 `csrc/ops.cpp`의 `TORCH_LIBRARY(myimg, ...)`와 `TORCH_LIBRARY_IMPL(myimg, CUDA, ...)`이다.

실제 CUDA 실행 경로는 `csrc/kernel.cu`의 아래 컴포넌트로 이어진다.

- `alex_conv1_fused_cuda`
- `conv1_bias_relu_kernel`
- `maxpool3x3s2_kernel`
- `ldg_nc_f32`

### 1.2 Convolution 수식과 메모리 오프셋

입력 텐서를 X ∈ ℝ^{N × C × H × W},  
필터를 W_f ∈ ℝ^{K × C × R × S},  
바이어스를 b ∈ ℝ^{K}라고 두면, Conv1 출력은 다음과 같다.

Y*conv[n, k, o_h, o_w] =
b[k] +
Σ*{c=0}^{C-1} Σ*{r=0}^{R-1} Σ*{s=0}^{S-1}
X[n, c, i_h, i_w] · W_f[k, c, r, s]

여기서

i_h = o_h · stride_h - pad_h + r  
i_w = o_w · stride_w - pad_w + s

출력 공간 크기는 다음과 같이 계산된다.

outH = floor((H + 2·pad_h - R) / stride_h) + 1  
outW = floor((W + 2·pad_w - S) / stride_w) + 1

현재 커널 구현은 NCHW contiguous layout을 가정하며, 선형 오프셋은 아래와 같이 계산한다.

x_idx = ((n · C + c) · H + i_h) · W + i_w  
w_idx = ((k · C + c) · R + r) · S + s  
y_idx = ((n · K + k) · outH + o_h) · outW + o_w

---

### 1.3 Bias, ReLU, MaxPool 융합 전략

현재 구현은 convolution 결과에 즉시 bias를 누적하고, 같은 스레드 문맥에서 ReLU를 적용한 뒤, 별도의 풀링 커널로 넘어간다.

즉:

- Conv + Bias + ReLU → 하나의 커널
- MaxPool → 별도의 커널

이 설계의 이유는 다음과 같다.

- Conv inner loop의 레지스터 사용량 증가 방지
- Pool까지 완전 융합 시 address recompute 증가
- baseline 단계에서는 정확성 검증과 해석이 우선

ReLU는 다음과 같다.

Y_relu[n,k,o_h,o_w] = max(0, Y_conv[n,k,o_h,o_w])

이후 MaxPool은 3×3, stride 2로 계산된다.

Y*pool[n,k,p_h,p_w] =
max*{0 ≤ r < 3, 0 ≤ s < 3}
Y_relu[n,k, 2p_h + r, 2p_w + s]

출력 크기:

poolH = floor((outH - 3) / 2) + 1  
poolW = floor((outW - 3) / 2) + 1

### 1.3 Bias, ReLU, MaxPool 융합 전략

현재 저장소 구현은 convolution 결과에 즉시 bias를 누적하고, 같은 스레드 문맥에서 ReLU를 적용한 뒤, 별도의 풀링 커널로 넘어간다. 엄밀히 말하면 Conv/Bias/ReLU는 하나의 커널에 묶여 있고, MaxPool은 두 번째 커널로 분리되어 있다. 그 이유는 다음과 같다.

- Conv inner loop의 레지스터 사용량을 급격히 늘리지 않는다.
- Pool까지 완전 융합할 경우 address recompute와 window bookkeeping이 증가한다.
- baseline 단계에서는 정확성 검증과 SASS 해석이 우선이다.

ReLU 단계는 다음 식으로 표현된다.

\[
Y*{relu}[n,k,o_h,o_w] = \max(0, Y*{conv}[n,k,o_h,o_w])
\]

이후 MaxPool은 \(3 \times 3\), stride 2로 계산된다.

\[
Y*{pool}[n,k,p_h,p_w] =
\max*{\substack{0 \le r < 3 \\ 0 \le s < 3}}
Y\_{relu}[n,k,2p_h + r, 2p_w + s]
\]

출력 풀링 크기는

\[
poolH = \left\lfloor \frac{outH - 3}{2} \right\rfloor + 1,\quad
poolW = \left\lfloor \frac{outW - 3}{2} \right\rfloor + 1
\]

이다.

### 1.4 인라인 PTX와 읽기 전용 로드 최적화

`csrc/kernel.cu`에는 다음 device helper가 들어 있다.

```cpp
__device__ __forceinline__ float ldg_nc_f32(const float* ptr) {
  float out;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(out) : "l"(ptr));
  return out;
}
```

이 함수는 `x`, `w`, `b` 로드 경로에 적용되어 generic global load 대신 non-coherent read-only 성격의 로드를 명시한다. 이 방식의 의도는 다음과 같다.

- 입력/가중치/바이어스가 커널 내부에서 읽기 전용이라는 점을 활용
- SASS 상에서 load path를 더 예측 가능하게 관찰
- L1/L2와 instruction mix를 해석하기 좋은 baseline 확보

특히 Conv1처럼 필터가 크고 입력 spatial reuse가 높은 경우, 이러한 load 힌트는 direct convolution baseline에서 유의미한 해석 지점을 만든다.

---

## 2. SASS 레벨 성능 분석 (Instruction & Register Analysis)

### 2.1 분석 도구와 워크플로

SASS 분석 경로는 `tools/dump_sass.sh`에 정리되어 있다. 이 스크립트는 다음 순서로 동작한다.

1. `python setup.py build_ext --inplace`
2. 확장 모듈 `.so` 탐색
3. `cuobjdump --dump-sass`
4. `nvdisasm`
5. `FFMA` 카운트 및 간단한 비율 출력

핵심 산출물은 다음 두 파일이다.

- `sass_dump/cuobjdump_sass.txt`
- `sass_dump/nvdisasm.txt`

### 2.2 Instruction Mix와 FFMA 비중 해석

Conv 커널의 성능은 결국 다음 instruction mix의 비중으로 귀결된다.

- arithmetic: `FFMA`, `FADD`, `FMUL`
- memory: global load/store
- address generation: integer add, shift, multiply
- control: predicate, branch

이상적인 fused Conv1 direct kernel은 inner loop에서 `FFMA` 비중이 높아야 한다. 이유는 convolution 본체가 본질적으로 multiply-accumulate 연산이기 때문이다. `tools/dump_sass.sh`는 전체 파일 기준 `FFMA` 카운트를 세지만, 실제 해석은 다음 질문에 초점을 맞춰야 한다.

- `conv1_bias_relu_kernel` 본문에 `FFMA`가 충분히 등장하는가
- 주소 계산용 integer instruction이 산술 instruction을 과도하게 잠식하는가
- load-to-use latency가 길어 보이는 구간이 있는가

본 저장소의 baseline은 direct convolution이므로, 완전한 GEMM-style Tensor Core 경로에 비해 memory/address instruction 비중이 높게 나타나는 것이 자연스럽다.

### 2.3 Register Pressure와 Spill 방지 전략

`setup.py`의 NVCC 옵션에는 `-Xptxas=-v`가 포함되어 있다. 이 옵션은 다음을 확인하기 위한 장치다.

- kernel당 register 사용량
- spill/store 발생 여부
- stack frame 크기

커널 코드에는 다음 launch bounds가 명시되어 있다.

```cpp
__global__ __launch_bounds__(128, 2) void conv1_bias_relu_kernel(...)
__global__ __launch_bounds__(128, 4) void maxpool3x3s2_kernel(...)
```

이 설계의 목적은 다음과 같다.

- block 크기를 고정해 occupancy 예측을 단순화
- register allocator가 과도한 register expansion을 하지 않도록 제약
- Conv와 Pool의 서로 다른 resource profile을 분리

현재 구현에서 spill 방지를 위해 취한 구조적 선택은 아래와 같다.

- thread당 출력 원소 1개 매핑
- shared memory staging 미사용
- Conv와 Pool 분리
- accumulator를 단일 `float`로 유지
- vectorized load를 아직 도입하지 않음

이는 최고 성능보다 분석 가능성과 제어 가능성을 우선한 설계다.

### 2.4 권장 해석 순서

SASS와 `ptxas` 로그를 볼 때는 아래 순서를 권장한다.

1. `register count`와 spill 존재 여부를 먼저 본다.
2. `conv1_bias_relu_kernel`에서 `FFMA`가 inner loop 중심으로 나타나는지 확인한다.
3. `ld.global.nc` 인라인 PTX가 실제 SASS load sequence에 어떤 영향을 주는지 본다.
4. memory bottleneck이 지배적이면 다음 단계로 shared memory 또는 vectorized load를 고려한다.

---

## 3. vLLM 기반 GPU 런타임 및 배치 관리 (Memory & Batching)

### 3.1 Raw CUDA 런타임 설계 배경

참고 파일 `alexnet_raw_cuda_dual_profile_runtime.cu`는 framework-free GPU runtime의 설계 스케치를 제공한다. 여기서 중요한 개념은 다음 세 가지다.

- `ArenaSlot`: 중간 activation을 담는 메모리 슬롯
- `RequestTable`: 들어온 요청의 메타데이터 테이블
- `BatchTable`: shape-compatible 요청들을 묶은 실행 단위

이 설계는 Python 프레임워크 오버헤드를 줄이고, GPU에 가까운 수준에서 배칭과 메모리 수명을 관리하려는 방향과 맞닿아 있다.

### 3.2 ArenaSlot 기반 메모리 관리

`ArenaSlot`은 중간 텐서를 위한 명시적 메모리 슬롯이다. 핵심 아이디어는 다음과 같다.

- activation lifecycle을 프레임워크 그래프가 아니라 런타임이 직접 관리
- 필요한 shape만큼만 slot을 예약
- 단계별 operator 실행이 끝나면 slot 재사용 가능

이는 일반적인 eager graph보다 다음 측면에서 유리하다.

- 중간 tensor allocation/free 오버헤드 감소
- 메모리 사용량 예측 가능
- edge 환경에서 workspace 예산 통제 용이

실무적으로는 phase 확장 시 다음 전략으로 이어질 수 있다.

- ping-pong buffer
- shape bucket별 arena pre-allocation
- activation aliasing

### 3.3 RequestTable과 BatchTable

참고 런타임의 `RequestTable`과 `BatchTable`은 가변 입력 요청을 shape-compatible batch로 묶기 위한 기본 자료구조다. 저장소의 Python 런타임 `runtime.py`는 같은 아이디어를 더 간단한 형태로 구현한다.

실제 구현 컴포넌트는 다음과 같다.

- `ImageRequest`
- `RequestTable`
- `BatchAssignment`
- `ShapeBucketSelector`
- `TensorRTRuntime`

`TensorRTRuntime.build_assignments()`는 요청을 버킷별로 모은 뒤, 각 버킷의 최대 batch 크기로 다시 chunking한다. 이 구조는 raw CUDA runtime의 `RequestTable -> BatchTable` 흐름과 개념적으로 동일하다.

### 3.4 런타임 데이터 흐름

아래 ASCII 도식은 Host 입력부터 결과 writeback까지의 흐름을 나타낸다.

```text
+------------------+
| Host Requests    |
| ImageRequest[]   |
+---------+--------+
          |
          v
+------------------+
| RequestTable     |
| req_id, tensor   |
+---------+--------+
          |
          v
+---------------------------+
| ShapeBucketSelector       |
| 224 / 256 / 512 bucket    |
+---------+-----------------+
          |
          v
+---------------------------+
| BatchAssignment Builder   |
| group + chunk by max N    |
+---------+-----------------+
          |
          v
+---------------------------+
| Host -> Device Transfer   |
| torch.cat + .to(cuda)     |
+---------+-----------------+
          |
          v
+---------------------------+
| CUDA Stream Dispatch      |
| stream_224 / 256 / 512    |
+---------+-----------------+
          |
          v
+---------------------------+
| Kernel / Engine Execute   |
| custom op or TRT engine   |
+---------+-----------------+
          |
          v
+---------------------------+
| Split Outputs             |
| out[idx:idx+1] per req    |
+---------+-----------------+
          |
          v
+---------------------------+
| Writeback / Return        |
| InferenceResult[]         |
+---------------------------+
```

### 3.5 프레임워크 오버헤드 최소화 관점

현재 Python 런타임은 아직 raw CUDA allocator 수준까지 내려가지는 않지만, 다음 두 지점에서 오버헤드 절감 방향을 이미 반영한다.

- per-bucket stream 분리
- shape-aware batch formation

즉, 완전한 custom runtime 이전 단계로서 충분히 유의미한 실험 기반을 제공한다.

---

## 4. TensorRT 기반 배포 전략 (Dynamic Shape & Engines)

### 4.1 ShapeBucket과 Optimization Profile

`compile_trt.py`에는 `ShapeBucket` dataclass가 정의되어 있고, 세 개의 대표 해상도 bucket이 등록되어 있다.

- `224`: `(1,3,224,224)` ~ `(8,3,224,224)`
- `256`: `(1,3,256,256)` ~ `(8,3,256,256)`
- `512`: `(1,3,512,512)` ~ `(4,3,512,512)`

이 설계는 TensorRT dynamic shape profile의 일반적인 실무 패턴과 일치한다. 하나의 거대한 범용 profile로 모든 shape를 감당하기보다, 대표 해상도별로 profile을 분리해 아래 이점을 얻는다.

- tactic selection의 안정화
- workspace 사용량 예측 가능
- latency variance 감소
- throughput target에 맞춘 profile별 max batch 제어

### 4.2 ShapeBucketSelector의 역할

런타임에서 실제 요청은 `runtime.py`의 `ShapeBucketSelector`가 선택한다. 핵심 판단 기준은 다음과 같다.

\[
(n, c, h, w) \leq (maxN, maxC, maxH, maxW)
\]

즉, 요청 shape가 bucket의 최대 허용 shape 이내이면 해당 bucket으로 보낸다. 이는 TensorRT execution context가 profile 범위 내 입력을 처리한다는 개념과 정확히 대응한다.

### 4.3 엔진 컴파일과 eager fallback

`compile_trt.py`는 `torch_tensorrt.compile(...)` 경로를 우선 시도한다. 다만 실제 현장에서는 다음 이유로 compile failure가 발생할 수 있다.

- custom op가 TensorRT converter에 없음
- plugin 미구현
- 특정 op decomposition 부재
- 환경 mismatch

이 문제를 다루기 위해 저장소는 eager fallback 경로를 함께 둔다.

- `EagerFallbackModule`
- bucket별 `manifest.json`

이 설계의 장점은 다음과 같다.

- TensorRT가 준비되지 않아도 runtime 구조는 유지된다.
- 테스트와 문서화가 환경 의존성에 덜 묶인다.
- unsupported op 구간을 점진적으로 plugin/converter로 대체할 수 있다.

### 4.4 Plugin/Converter 폴백 경로

`plugin_stub.py`는 `myimg::alex_conv1_plugin_stub`를 정의한다. 이 op는 TensorRT 쪽에서 plugin insertion point 역할을 하며, eager 모드에서는 아래 둘 중 하나로 처리된다.

- CUDA extension이 로드되면 `torch.ops.myimg.alex_conv1_fused`
- 아니면 PyTorch 기준 연산 fallback

이 구조가 중요한 이유는 다음과 같다.

- graph 상에서 hotspot을 명시적 노드로 분리 가능
- 나중에 TensorRT plugin이나 converter를 연결하기 쉬움
- unsupported 구간만 선택적으로 고성능 경로로 바꿀 수 있음

즉, `plugin_stub.py`는 단순 placeholder가 아니라, deployment-ready graph segmentation의 경계점 역할을 한다.

---

## 5. 하드웨어 프로파일 및 성능 결론 (Hardware Profiles)

### 5.1 EDGE_LOW_MEM vs A6000_HIGH_THROUGHPUT

참고 코드와 벤치 스크립트는 두 가지 하드웨어 운용 철학을 전제로 한다.

#### EDGE_LOW_MEM

- 작은 batch
- workspace 제한 우선
- direct convolution baseline 유지
- activation memory 안정성 중시

권장 설정:

- batch size 1 중심
- bucket은 224, 256 위주
- eager 또는 작은 TensorRT workspace
- allocator churn을 줄이는 arena/pool 전략

#### A6000_HIGH_THROUGHPUT

- 큰 batch
- 높은 occupancy와 launch amortization 중시
- 다중 stream/context 활용
- 큰 workspace와 tactic 탐색 허용

권장 설정:

- batch size 8 이상
- 224/256/512 전체 bucket 활성화
- TensorRT profile별 stream 병렬화
- 이후 단계에서 fp16, vectorized load, shared memory 도입

### 5.2 현재 벤치마크 지표 해석 방법

저장소에는 두 종류의 측정 코드가 있다.

- `tests/bench_phase2.py`
- `tests/bench_phase3.py`

Phase 2는 custom fused kernel과 PyTorch 기준 연산의 평균 latency를 비교한다. 여기서 봐야 할 핵심 지표는 다음과 같다.

- `custom ms`
- `pytorch ms`
- `speedup = pytorch / custom`
- correctness 유지 여부

Phase 3는 mixed-size request batch를 대상으로 bucket routing과 stream dispatch 결과를 함께 출력한다. 여기서 중요한 해석 포인트는 다음과 같다.

- `avg_batch_ms`: end-to-end 배치 처리 시간
- `bucket_counts`: shape admission policy가 실제로 어떻게 작동했는지
- `stream_*`: 멀티 스트림 dispatch가 분리되었는지

### 5.3 현재 구현의 성능적 의미

현재 구현은 최고점 성능 제품이 아니라, 다음 세 가지를 동시에 만족하는 연구용 baseline으로 보는 것이 맞다.

- correctness 검증 가능
- SASS 관찰 가능
- deployment path로 확장 가능

즉, 이 프로젝트는 "최종 최적화가 끝난 커널"보다 "최적화 루프를 반복할 수 있는 구조"에 더 큰 가치를 둔다.

### 5.4 향후 고도화 방향

다음 단계의 우선순위는 아래와 같다.

1. Conv1 direct kernel에 output-channel tiling 도입
2. shared memory staging과 vectorized load 추가
3. fp16 input/weight + fp32 accumulate 경로 추가
4. TensorRT plugin 또는 converter 정식 구현
5. Python runtime의 batch builder를 raw CUDA arena 기반으로 치환
6. bucket별 execution context 재사용과 timing cache 고도화

---

## 결론

이 프로젝트는 AlexNet의 Conv1 hotspot을 출발점으로 하여, 커널 구현, 저수준 분석, 런타임 배칭, TensorRT 배포까지 하나의 수직 통합 스택으로 연결했다는 점에서 의미가 있다.

핵심 실체는 다음 컴포넌트에 응축되어 있다.

- `alex_conv1_fused`
- `ldg_nc_f32`
- `conv1_bias_relu_kernel`
- `ShapeBucket`
- `ShapeBucketSelector`
- `TensorRTRuntime`
- `myimg::alex_conv1_plugin_stub`

정리하면, 본 저장소는 NVIDIA GPU 환경에서 다음 질문에 답하기 위한 실험 기반이다.

- 어떤 연산을 커널 수준에서 직접 융합할 것인가
- 그 결과를 SASS 수준에서 어떻게 검증할 것인가
- 그 커널을 TensorRT 및 버킷 기반 런타임에 어떻게 실전 배포할 것인가

이 세 질문에 대한 현재 답은 baseline이지만, 구조는 이미 확장 가능하게 설계되어 있다. 따라서 본 보고서는 단순 구현 설명을 넘어, 이후 shared-memory tiling, Tensor Core 경로, plugin 정식화, runtime 메모리 관리자 확장으로 이어지는 로드맵의 출발점으로 볼 수 있다.
