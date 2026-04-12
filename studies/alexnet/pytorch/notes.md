# AlexNet: The Archetype of Modern AI & Hardware Co-Design

딥러닝의 폭발적 성장을 이끈 **AlexNet**의 본질은 단순히 '깊은 신경망'이 아닙니다. 이는 당시 하드웨어의 한계(VRAM 3GB)를 수학적 구조로 돌파한 **최초의 고성능 컴퓨팅(HPC) 기반 인공지능 설계**입니다.

---

## 1. 모델 비유: 감각에서 인지까지의 파이프라인

### 단일 GPU 버전: "지능형 다단계 필터링 체(Sieve)"

거대한 원석(이미지 데이터)을 갈아 넣어 다이아몬드(클래스 분류)를 찾아내는 공정과 같습니다.

- **초반부**: 아주 큰 돋보기(11x11 필터)로 거친 윤곽과 색상을 훑습니다.
- **중반부**: 점차 정밀한 필터(3x3)로 질감과 무늬를 층층이 걸러냅니다.
- **후반부**: 걸러진 미세한 조각들을 모아 최종 판정관들이 결론을 내립니다.

### Split-GPU 버전: "이기종 분산 인지 엔진 (Heterogeneous Distributed Perception Engine)"

두 명의 전문 분석가가 초고속 통신망(PCIe)으로 연결된 고도의 분업 상태입니다.

- **GPU 0 (전방 감각 처리기)**: 방대한 데이터를 받아들여 기초 시각 특징을 처리합니다. 처리 속도가 중요한 **감각(Sensory) 단계**입니다.
- **GPU 1 (후방 의미 해석기)**: 압축된 특징을 조합해 고차원적인 판단을 내립니다. 지식 저장이 핵심인 **추론(Cognitive) 단계**입니다.

---

## 2. 입출력 및 시스템 사양

| 항목                 | 데이터 형태 / 값       | 설명                                                  |
| :------------------- | :--------------------- | :---------------------------------------------------- |
| **입력값 (Input)**   | `(Batch, 3, 224, 224)` | RGB 3채널의 고해상도 이미지                           |
| **출력값 (Output)**  | `(Batch, 1000)`        | 1,000개 클래스에 대한 확률 점수 (Logits)              |
| **Kernel Size**      | `11x11, 5x5, 3x3`      | 수용 영역(Receptive Field)의 크기 결정                |
| **Stride / Padding** | `s4, s2 / p2, p1`      | 특징 맵의 해상도 축소 및 테두리 정보 보존             |
| **Dropout**          | `0.5`                  | 뉴런을 무작위로 0으로 만들어 과적합(Overfitting) 방지 |
| **LRN**              | `Local Response Norm`  | 강한 신호가 주변 신호를 억제하는 측면 억제 효과 모사  |

---

## 3. 핵심 아키텍처 구현 (Full Code & Shape Analysis)

### 3.1 단일 GPU 아키텍처 (Standard AlexNet)

```python
import torch
import torch.nn as nn

class AlexNet2012(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # [입력 데이터] (Batch, 3, 224, 224)

        self.features = nn.Sequential(
            # (B, 3, 224, 224) → [Conv1: 11x11, s4, p2] → (B, 96, 55, 55)
            # 이유: 큰 커널로 광범위한 공간 정보(선, 색상)를 굵직하게 추출합니다.
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),

            # (B, 96, 55, 55) → [Pool1: 3x3, s2] → (B, 96, 27, 27)
            # 이유: 중복된 정보는 덜어내고 가장 강한 신호만 남깁니다.
            nn.MaxPool2d(kernel_size=3, stride=2),

            # (B, 96, 27, 27) → [Conv2: 5x5, p2] → (B, 256, 27, 27)
            # 이유: 좀 더 세밀한 특징(패턴, 복잡한 모양)을 학습합니다.
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),

            # (B, 256, 27, 27) → [Pool2: 3x3, s2] → (B, 256, 13, 13)
            # 이유: 연산 효율을 위해 해상도를 한 번 더 압축합니다.
            nn.MaxPool2d(kernel_size=3, stride=2),

            # (B, 256, 13, 13) → [Conv3-4-5: 3x3, p1] → (B, 256, 13, 13)
            # 이유: 3x3의 작은 커널을 깊게 쌓아 고차원적인 특징 조합을 완성합니다.
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # (B, 256, 13, 13) → [Pool3: 3x3, s2] → (B, 256, 6, 6)
            # 이유: 분류기(FC)에 최적화된 고밀도 특징 맵을 만듭니다.
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            # (B, 256*6*6=9216) → [FC1: 4096] → (B, 4096)
            # 이유: 추출된 공간적 정보를 완전히 융합하여 의미론적 분석을 시작합니다.
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            # (B, 4096) → [FC2: 4096] → (B, 4096)
            # 이유: 거대한 파라미터를 통해 학습된 시각적 지식을 추론에 활용합니다.
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # (B, 4096) → [FC3: 1000] → (B, 1000)
            # 이유: 최종적으로 1000개 클래스 중 가장 유사한 항목을 결정합니다.
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1) # 1차원 벡터로 변환
        x = self.classifier(x)
        return x
```

---

### 3.2 분할 GPU 아키텍처 (Split-GPU AlexNet)

```python
class SplitAlexNet(nn.Module):
    """
    2012년 당시 3GB VRAM 한계를 극복하기 위해 설계된 병렬 분할 모델입니다.
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.dev0 = torch.device("cuda:0") # Activation 중심 (고해상도 처리)
        self.dev1 = torch.device("cuda:1") # Parameter 중심 (대용량 메모리)

        # GPU 0: 고해상도 전처리에 집중 (채널 수를 96→48로 쪼개어 담음)
        self.features_0 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev0)

        # GPU 1: 대용량 FC 레이어와 깊은 특징 추출 담당
        self.features_1 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ).to(self.dev1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPU 0 연산: 감각 처리
        x = x.to(self.dev0, non_blocking=True)
        x = self.features_0(x)

        # 핵심 병목: GPU 간 데이터 전송 (D2D Transfer)
        x = x.to(self.dev1, non_blocking=True)

        # GPU 1 연산: 고도 인지 및 분류
        x = self.features_1(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---
