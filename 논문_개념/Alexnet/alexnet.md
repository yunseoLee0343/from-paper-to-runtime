# AlexNet → YOLOv8: <br>Architecture, Memory, and Hardware Co-Design

## 1. Executive Abstract

**Problem.**  
과거 컴퓨터 비전은 수작업 특징 추출의 한계와 연산 하드웨어의 극심한 제약, 특히 단일 GPU 3GB VRAM 수준의 메모리 한계 때문에 대규모 고해상도 이미지의 표현 공간을 깊게 학습하는 데 근본적인 병목이 있었다.

**Method.**  
AlexNet은 이미지의 **locality**와 **stationarity**를 강한 귀납적 편향으로 활용하는 합성곱 구조를 채택하고, **ReLU**, **dropout**, **LRN**, **overlapping pooling**, 그리고 **2-GPU 분할 학습**을 결합해 모델을 실제 하드웨어 위에서 학습 가능한 형태로 재구성했다.

**Result.**  
그 결과 약 **60M parameters**, **650K neurons**, **5 Conv + 3 FC** 구조의 대형 CNN을 두 장의 GTX 580 3GB GPU 위에서 학습시켜 ImageNet 분류 성능을 크게 끌어올렸고, 이후 딥러닝 모델은 정확도뿐 아니라 **메모리 흐름, 연산 배치, 데이터 이동**까지 포함한 시스템 설계 문제로 발전했다.

---

## 2. Problem Formulation: 왜 기존 방식이 깨졌는가

### 2.1 전통적 CV 파이프라인의 한계

전통적 컴퓨터 비전은 **SIFT**, **HOG**, **sparse coding**, **Fisher Vector**처럼 사람이 설계한 feature extractor를 먼저 만들고, 그 위에 얕은 분류기를 올리는 구조였다. 이 방식은 **픽셀 → feature → classifier**라는 분리된 파이프라인을 가지며, 무엇을 볼지는 이미 사람이 정의한다. 따라서 데이터가 커져도 모델이 새로운 feature hierarchy를 스스로 학습하는 데에는 한계가 있다.

YOLOv8이 이미지를 넣으면 box와 class를 한 번에 예측하는 **end-to-end 시스템**인 것과 달리, 당시 전통적 CV는 각 단계를 사람이 나눠 설계해야 했고, 그 결과 표현 자체가 고정되어 있었다.

### 2.2 Dense MLP의 붕괴

문제는 단순히 feature engineering의 한계가 아니라, 고해상도 입력을 그대로 dense하게 다루는 방식이 물리적으로 무너진다는 점이다. AlexNet 입력 기준으로 이미지 크기는 $224 \times 224 \times 3 = 150{,}528$ 이다. 이 입력을 첫 번째 은닉층의 96개 뉴런에 Dense layer로 연결하면 필요한 파라미터 수는 $150{,}528 \times 96 = 14{,}450{,}688$ 이다.

즉, 단 하나의 레이어에서만 약 **1,440만 개의 독립 파라미터**가 필요하다. 이는 세 가지 문제를 동시에 만든다.

- **메모리 문제**: 가중치 행렬이 VRAM을 빠르게 잠식한다.
- **연산 문제**: 모든 입력과 출력을 전부 곱해야 하므로 비용이 급증한다.
- **일반화 문제**: 파라미터 수가 지나치게 많아 과적합 위험이 커진다.

### 2.3 시스템 병목 정의

ImageNet 규모 문제에서는 병목이 하나가 아니다. 최소한 다음 세 가지가 동시에 존재한다.

- **Parameter footprint**: 모델 가중치 자체가 차지하는 메모리
- **Activation footprint**: 배치 단위 중간 feature map이 차지하는 메모리
- **Data movement cost**: CPU↔GPU, GPU↔GPU 간 텐서 이동 비용

결국 “좋은 모델”이 아니라 **실제로 GPU 메모리 위에서 돌아가는 모델**이 필요해진다.

---

## 3. Core Breakthrough: AlexNet

### 3.1 구조적 전환: Dense → Convolution

AlexNet의 핵심은 파라미터를 단순히 줄이는 것이 아니라, **구조적으로 폭발하지 못하게 만드는 것**이다. Convolution은 이미지에 대해 두 가지 강한 가정을 도입한다.

- **Locality**: 가까운 픽셀끼리 더 강한 상관을 가진다.
- **Stationarity**: 유용한 패턴은 위치가 바뀌어도 반복된다.

```Python
# conceptual conv
for oc in range(out_channels):
    for ic in range(in_channels):
        for kh in range(K_h):
            for kw in range(K_w):
                out[oc] += weight[oc, ic, kh, kw] * x[ic, h+kh, w+kw]
```

이 두 가정은 곧 **inductive bias**이며, Dense MLP가 모든 연결을 허용하는 대신 Conv는 **작은 receptive field + weight sharing**으로 연결 구조를 제한한다.

입력이 $224 \times 224 \times 3$ 이고 첫 번째 Conv layer가 `nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)` 라면, 파라미터 수는 $(11 \times 11 \times 3) \times 96 = 34{,}848$ 이다. 이는 Dense 연결 대비 약 **0.24%** 수준이다.

```python
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
```

이 한 줄은 다음을 동시에 의미한다.

- 전체 이미지를 한 번에 보지 않는다 → **locality**
- 필터를 공간 전체에 이동시키며 적용한다 → **sliding window**
- 같은 필터를 반복 사용한다 → **weight sharing**

즉, Conv는 파라미터를 “줄이는 트릭”이 아니라, **이미지 구조를 모델에 강제로 주입하는 설계**다.

### 3.2 하드웨어 설계: Split-GPU

```Python
x = x.to(self.dev0, non_blocking=True)
x = self.features_0(x)

x = x.to(self.dev1, non_blocking=True) # 핵심: GPU 간 이동

x = self.features_1(x)
```

Conv를 도입해도 끝이 아니다. AlexNet은 전체적으로 약 **60M parameters**를 가지며, 여기에 미니배치 activation map까지 더하면 당시 단일 GPU의 3GB VRAM 안에 안전하게 담기 어렵다. 그래서 AlexNet은 모델을 두 GPU에 분산한다.

- **GPU 0**: 초기 Conv 레이어 담당 — 해상도가 커서 activation footprint가 크다.
- **GPU 1**: 후반 Conv + FC 담당 — 해상도는 작지만 파라미터가 집중된다.

이는 단순한 모델 분할이 아니라, **연산과 메모리 흐름을 분리하는 설계**다. `nvmatrix (1).cu`에서는 GPU별로 독립적인 cuBLAS handle과 CUDA stream을 유지한다.

```cpp
std::map<int, cublasHandle_t> NVMatrix::_cublasHandles;
std::map<int, cudaStream_t> NVMatrix::_defaultStreams;
```

이 구조는 디바이스별 실행 context를 분리하고, 메모리 트랜잭션이 다른 GPU 영역을 침범하지 않도록 한다. `convnet.cu`에서는 replica 간 연결을 구성해 필요한 시점에만 activation을 교환한다.

```cpp
it->second[r]->addPrev(*_layerMap[inputName][rp], ridx);
_layerMap[inputName][rp]->addNext(*it->second[r]);
```

핵심은 **전체 통신을 허용하지 않고, 필수 구간에서만 D2D transfer를 일으키는 것**이다. 이는 단순한 멀티-GPU가 아니라, **통신 latency를 숨기고 compute throughput을 최대화하는 HPC 파이프라이닝**에 가깝다.

### 3.3 학습 안정성: ReLU, LRN, 그리고 BatchNorm

구조와 메모리를 해결해도 학습이 수렴하지 않으면 전체 시스템은 실패한다. 기존 활성화 함수인 $\tanh(x)$ 는 도함수 $1 - \tanh^2(x)$ 를 가지므로 입력 절댓값이 커질수록 기울기가 0으로 수렴한다. 깊은 네트워크에서는 이러한 작은 gradient가 연쇄적으로 곱해져 하위 레이어로 갈수록 신호가 사라진다.

AlexNet은 이를 해결하기 위해 **ReLU**를 도입한다. ReLU는 $f(x)=\max(0,x)$ 이고, $x>0$ 구간에서는 도함수가 1이므로 gradient를 더 잘 전달한다. 결과적으로 학습 속도가 크게 향상된다.

그러나 ReLU는 출력 상한이 없어 activation 값이 과도하게 커질 수 있다. AlexNet은 이를 억제하기 위해 **LRN (Local Response Normalization)** 을 사용한다.

```python
nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
```

LRN은 같은 공간 위치에서 인접 채널 간 경쟁을 유도해 큰 activation을 주변 값으로 나눠 억제한다. 다만 이는 국소적 보정에 가깝고, 미니배치 전체 분포를 통제하지는 못한다. 현대 모델에서는 대체로 **BatchNorm** 이 이를 대신한다.

```python
nn.BatchNorm2d(num_features=96)
```

BatchNorm은 미니배치 단위 평균과 분산을 정규화해 입력 분포를 안정화하므로, 학습 속도와 일반화 모두에서 더 강력한 효과를 낸다. 즉, AlexNet 시점의 LRN은 **정규화의 초기 형태**이고, 현대 표준은 BN 또는 LayerNorm 같은 더 강한 분포 제어 기법이다.

---

## 4. Memory-Centric Interpretation

### 4.1 Activation vs Parameter 분해

AlexNet을 메모리 관점에서 보면 레이어별 병목의 성격이 다르다.

- **앞단 레이어**: 공간 해상도가 크므로 activation footprint가 지배적이다.
- **뒷단 레이어**: 공간 해상도는 작아지지만 FC 파라미터가 몰려 있으므로 parameter footprint가 지배적이다.

예를 들어 raw split 구조 기준에서 GPU 0의 Conv1 activation은 약 $128 \times 48 \times 27 \times 27 \times 4 \text{ bytes} \approx 17.9 \text{ MB}$ 수준이고, GPU 1의 FC1~FC2 파라미터만 약 **142.5 MB**를 차지한다. 즉, 앞단은 **activation-bound**, 뒷단은 **parameter-bound**에 가깝다.

### 4.2 Tensor Flow on GPU

AlexNet의 텐서 흐름은 다음과 같이 요약할 수 있다.

- 원본 입력은 GPU 0으로 이동한다.
- Conv1~Conv2에서 고해상도 feature를 추출한다.
- 중간 activation을 GPU 1로 복사한다.
- Conv3~Conv5와 FC에서 의미론적 특징을 완성한다.

이 과정에서 가장 민감한 병목은 **GPU 0 → GPU 1 전송**이다. 당시에는 PCIe 대역폭이 제한적이었기 때문에, 이 지점은 단순한 함수 호출이 아니라 전체 시스템 지연의 핵심이었다. 그래서 non-blocking transfer와 stream overlap이 중요했다.

### 4.3 Execution Shape

AlexNet을 “모델 구조”로만 보면 Conv, Pooling, FC의 조합이지만, 실행 관점에서는 다음과 같이 다시 읽힌다.

- **Dense → Conv**: parameter memory를 compute reuse로 전환
- **High-resolution → Downsample**: activation footprint를 점진적으로 축소
- **Single GPU → Split-GPU**: memory capacity를 구조적으로 확장
- **Blocking copy → Overlap**: 통신 지연을 계산으로 은닉

즉, AlexNet은 단순한 네트워크가 아니라 **GPU 메모리 위에서 흐르도록 재구성된 execution shape**다.

---

## 5. Source Code Forensics

### 5.1 raw_code.py 분석

`raw_code.py`는 논문 구조를 현대 PyTorch로 재구성하면서도, 2012년 시스템 제약을 드러내는 요소들을 남겨 둔다. 주요 구성은 다음과 같다.

- `AlexNet2012`: 논문형 단일 디바이스 AlexNet
- `SplitAlexNet`: 2개의 CUDA device에 레이어를 분산하는 구조
- `AlexNetLighting`: PCA lighting augmentation
- `PrefetchLoader`: CPU→GPU 전송 overlap
- `MemoryProfiler`: layer-wise activation memory profiling

특히 `SplitAlexNet.forward()` 는 메모리 흐름 자체를 코드로 보여준다.

```python
x = x.to(self.dev0, non_blocking=True)
x = self.features_0(x)
x = x.to(self.dev1, non_blocking=True)
x = self.features_1(x)
x = torch.flatten(x, 1)
x = self.classifier(x)
```

이 흐름은 단순한 forward가 아니라, **GPU 0에서 activation을 만들고 GPU 1로 넘긴 뒤 분류하는 파이프라인**이다.

### 5.2 CUDA 구현 분석

`nvmatrix.cu` 는 GPU별 메모리 및 stream 자원을 분리해 관리하며, `convnet.cu` 는 레이어 replica 간 연결 그래프를 만든다. 이 두 파일은 현대 프레임워크가 자동화하는 장치를 수작업으로 구현하던 시기의 흔적이다.

요약하면:

- `nvmatrix` → **메모리 isolation / stream 관리**
- `convnet` → **replica graph / selective communication**

즉, AlexNet의 멀티-GPU는 단순히 `cuda:0`, `cuda:1` 를 나눈 것이 아니라 **실행 그래프 전체를 디바이스 단위로 쪼개는 수동 런타임**이었다.

### 5.3 torchvision과 비교

현대 `torchvision.models.alexnet` 은 구조적으로 훨씬 단순하다.

- **Split-GPU 없음** → 단일 GPU 메모리로 충분
- **LRN 없음** → 성능 대비 효용 낮음
- **AdaptiveAvgPool2d((6, 6))** 도입 → 다양한 입력 크기 지원
- **제한된 채널 연결 없음** → 단일 타워 full connectivity 채택

즉, 현대 구현에서 살아남은 것은 **Conv + ReLU + Dropout 같은 본질적 요소**이고, 사라진 것은 **하드웨어 제약에 묶여 있던 요소**다.

---

## 6. Evolution: AlexNet → Modern CNN

### 6.1 VGG

VGG는 AlexNet보다 더 단순한 원리로 간다. 큰 커널을 줄이고, **3×3 kernel stacking** 으로 receptive field를 쌓는다. 이 방식은 파라미터 효율과 구조 단순성을 동시에 얻는다.

### 6.2 ResNet

ResNet은 깊이가 커질수록 생기는 학습 불안정을 **skip connection** 으로 해결한다. AlexNet이 ReLU로 gradient 전달 문제를 완화했다면, ResNet은 아예 identity path를 넣어 깊은 네트워크를 안정화한다.

### 6.3 Fully Convolutional Transition

AlexNet의 가장 큰 약점은 FC에 파라미터가 몰린다는 점이었다. 이후 모델들은 **Global Average Pooling** 혹은 convolutional head로 이동하면서 FC 의존성을 줄였다. 이 흐름이 detection 모델에서는 거의 완전한 **Fully Convolutional Network (FCN)** 형태로 이어진다.

---

## 7. Modern System: YOLOv8

### 7.1 Architecture

YOLOv8은 backbone–neck–head 구조를 가지며, `yolov8.yaml` 에서 **Conv**, **C2f**, **SPPF**, **Concat**, **Upsample**, **Detect** 블록이 조합된다. AlexNet이 분류를 위한 단일 표현 파이프라인이었다면, YOLOv8은 **multi-scale feature fusion**을 통해 box와 class를 동시에 예측한다.

즉, AlexNet이 “깊은 CNN이 된다”를 증명했다면, YOLOv8은 그 CNN을 **실시간 detection에 맞게 재배치**한 구조다.

### 7.2 C2f & CSP 구조

YOLOv8의 C2f는 입력 채널 전체를 무겁게 처리하지 않고, 일부는 깊은 연산 경로로 보내고 일부는 shortcut으로 우회시킨 뒤 마지막에 concat한다. 이 구조는 메모리 관점에서 매우 중요하다.

- 전체 채널을 매번 무겁게 처리하지 않음
- global memory read/write 감소
- working set 축소
- cache/shared memory 적재 용이
- arithmetic intensity 개선

즉, C2f는 단순한 모델 블록이 아니라, **메모리 대역폭 병목을 줄이기 위한 구조적 선택**이다.

### 7.3 Training System

AlexNet의 split-GPU는 **모델을 나누는 방식**이었다. 현대 대규모 학습은 보통 **모델은 복제하고, 데이터를 나눠 학습한 뒤 gradient를 동기화**한다. 대표가 DDP와 AMP다.

- **DDP**: GPU마다 서로 다른 mini-batch shard 할당
- **AMP**: FP16/FP32 혼합으로 메모리 절감 + Tensor Core 활용

\*DDP의 본질은 각 GPU에서 계산된 gradient를 합쳐 동일한 모델을 유지하는 것이다.

```Python
# DDP 핵심 (개념적)
for param in model.parameters():
    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    param.grad /= world_size
```

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

def train_yolov8_ddp(rank, world_size, model, dataloader, optimizer, epochs=10):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        for images, targets in dataloader:
            images = images.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    dist.destroy_process_group()
```

즉, AlexNet의 split-GPU가 “모델을 쪼개서 올리는” 방식이었다면, 현대 DDP는 “모델은 복제하고 데이터를 나눠 처리하는” 방식이다.

---

## 8. Hardware-Level Optimization

### 8.1 Memory Coalescing

CUDA 최적화의 핵심은 전역 메모리 접근을 흩어지지 않게 만드는 것이다. warp 내 스레드들이 연속된 주소를 읽으면 메모리 트랜잭션이 병합되고, 그렇지 않으면 버스 요청 수가 증가해 ALU가 대기하게 된다.

### 8.2 Arithmetic Intensity

모델이 빠르다는 것은 FLOPs가 적다는 뜻만이 아니다. 같은 메모리 이동 대비 얼마나 많은 유효 계산을 수행하는지가 중요하다. C2f와 CSP류 구조는 불필요한 채널 연산을 줄이고 useful compute 비율을 높인다.

### 8.3 Kernel vs Runtime

최적화는 커널 수준과 런타임 수준으로 나뉜다.

- **Kernel optimization**: coalescing, tiling, shared memory reuse
- **Runtime optimization**: stream overlap, data prefetch, inter-device scheduling

AlexNet의 split-GPU는 런타임 설계에 가깝고, 현대 CUDA 최적화는 여기에 커널 친화적 tensor layout까지 더한다.

---

## 9. Quantization & Inference

### 9.1 FP32 → FP16 → INT8

현대 inference 최적화의 핵심은 FP32를 FP16 혹은 INT8로 줄이는 것이다. 기본적인 양자화 식은 $x_q = \text{round}(x_{fp32}/s) + z$ 형태다.

- **FP16**: 메모리 절감, Tensor Core 활용
- **INT8**: 더 큰 메모리 절감, 전력 효율 증가

### 9.2 AlexNet vs YOLOv8

AlexNet은 INT8 양자화에 상대적으로 취약하다. 이유는 파라미터 대부분이 FC layer에 몰려 있고, 이 가중치 분포의 dynamic range가 넓기 때문이다. 반면 YOLOv8은 Conv 중심 구조이며, BN folding 등을 통해 분포 제어가 쉬워 양자화에 유리하다.

즉,

- **AlexNet**: FC-heavy → quantization 취약
- **YOLOv8**: Conv-heavy + BN-friendly → quantization 우호적

---

## 10. Synthesis & Critical Insight

### 10.1 공통 철학

AlexNet과 YOLOv8은 시대와 목적이 다르지만, 같은 질문을 공유한다.

- 어떻게 파라미터를 줄일 것인가
- 어떻게 activation footprint를 통제할 것인가
- 어떻게 메모리 이동을 최소화할 것인가
- 어떻게 실제 디바이스 위에서 빠르게 돌릴 것인가

### 10.2 핵심 진화

AlexNet은 “딥러닝이 GPU 위에서 돌아갈 수 있다”를 증명한 사건이고, YOLOv8은 이를 “실제 서비스 속도와 규모에 맞게 운용할 수 있다”로 확장한 구조다. 즉, 발전의 본질은 단순한 정확도 경쟁이 아니라 **model → system → hardware co-design** 으로의 이동이다.

### 10.3 Counter-Argument

2012년의 LRN과 split-GPU가 오늘날 H100에서도 그대로 유효한가? 대답은 아니오에 가깝다.

- **LRN**: 성능 대비 이득이 작고, BN/LayerNorm에 밀려 사실상 퇴장
- **Split-GPU**: AlexNet 크기 모델은 현대 GPU 단일 메모리에도 충분히 적재 가능

다만 개념적 유산은 남는다. 현대 초거대 모델 학습에서 tensor parallelism, pipeline parallelism, interconnect-aware scheduling은 모두 AlexNet 시기의 split 사고방식을 더 큰 규모에서 다시 구현한 것이다.

---

## 11. Conclusion

AlexNet의 본질은 단순히 더 깊은 CNN을 만든 것이 아니다. 그것은 **표현 학습, 메모리 배치, 데이터 이동, 학습 안정성**을 한꺼번에 설계한 첫 사례였다. 이후의 YOLOv8, DDP, AMP, C2f, quantization-friendly architecture는 모두 이 문제를 더 큰 하드웨어와 더 복잡한 작업에 맞게 다시 푼 결과다.

> Deep learning architecture is not just model design,  
> but execution design on hardware.
