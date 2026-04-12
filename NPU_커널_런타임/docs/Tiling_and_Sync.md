```md
# Tile Scheduling & Synchronization Manual

## Overview

이 문서는 Phase 2에서 구현한 Intel NPU 타일 스케줄러 로직을 설명한다. 중심 구성요소는 `GraphTileScheduler::schedule_two_conv_chain()`, `build_tile_reuse_dependencies()`, `BarrierState`, 그리고 `ConvTileContract`이다.

설계 목표는 다음과 같다.

- NCHW 텐서를 Intel NPU의 `max_tile_h`, `max_tile_w`, `max_tile_c` 제약에 맞게 분할한다.
- `Conv0 -> Conv1` 연쇄에서 producer/consumer 관계를 하드웨어 배리어로 변환한다.
- 컴파일러가 생성한 타일 계약을 런타임이 그대로 해석할 수 있도록 구조화한다.
- 가능하면 `Conv0` 결과가 SRAM에 머무는 동안 `Conv1`이 바로 재사용하도록 `Tile Reuse`를 허용한다.

## 1. Tiling Strategy

### 1.1 Output-Centric Tile Partitioning

스케줄러는 각 Conv를 출력 텐서 기준으로 먼저 분할한다. 하드웨어가 허용하는 최대 타일 크기는 `HardwareCaps`에 의해 제한된다.

\[
T_h = \min(\text{max\_tile\_h}, O_H)
\]

\[
T_w = \min(\text{max\_tile\_w}, O_W)
\]

\[
T_c = \min(\text{max\_tile\_c}, O_C)
\]

여기서:

- \(O_H\): 출력 feature map 높이
- \(O_W\): 출력 feature map 너비
- \(O_C\): 출력 채널 수

실제 타일 생성 루프는 다음 구조를 가진다.

```text
for oc0 in [0, OC) step T_c:
  for oh0 in [0, OH) step T_h:
    for ow0 in [0, OW) step T_w:
      create output tile
```

각 타일의 출력 범위는 다음과 같이 정의된다.

\[
oc_1 = \min(oc_0 + T_c, O_C)
\]

\[
oh_1 = \min(oh_0 + T_h, O_H)
\]

\[
ow_1 = \min(ow_0 + T_w, O_W)
\]

따라서 하나의 출력 타일은 다음 구간을 나타낸다.

\[
\text{output\_region} = [n_0, n_1) \times [oc_0, oc_1) \times [oh_0, oh_1) \times [ow_0, ow_1)
\]

### 1.2 Input Footprint Back-Projection

출력 타일이 정해지면, 해당 타일을 계산하는 데 필요한 입력 footprint를 stride, padding, kernel 크기에서 역산한다.

\[
in\_h_0 = \max(0, oh_0 \cdot s_h - p_h)
\]

\[
in\_w_0 = \max(0, ow_0 \cdot s_w - p_w)
\]

\[
in\_h_1 = \min(I_H, (oh_1 - 1)\cdot s_h - p_h + k_h)
\]

\[
in\_w_1 = \min(I_W, (ow_1 - 1)\cdot s_w - p_w + k_w)
\]

즉 출력 타일 하나는 실제로 더 넓은 입력 영역을 필요로 할 수 있으며, 이 입력 범위가 SRAM에 적재되어야 한다.

### 1.3 ASCII Art: Spatial Tiling

다음은 `OH x OW = 32 x 32`, `T_h = T_w = 16`인 경우의 공간 타일링 예시다.

```text
Output Feature Map (single OC slice)

          W ->
      +----------------+----------------+
  H   | Tile (0,0)     | Tile (0,1)     |
  |   | h:[0,16)       | h:[0,16)       |
  v   | w:[0,16)       | w:[16,32)      |
      +----------------+----------------+
      | Tile (1,0)     | Tile (1,1)     |
      | h:[16,32)      | h:[16,32)      |
      | w:[0,16)       | w:[16,32)      |
      +----------------+----------------+
```

채널 차원은 별도의 축으로 분할된다.

```text
OC axis:

[0...............................OC)
[ Tile OC Block 0 ][ Tile OC Block 1 ][ ... ]
```

즉 전체 타일링은 "출력 채널 블록"과 "공간 타일"의 곱으로 구성된다.

## 2. Dependency Graph

### 2.1 Conv0 -> Conv1 Producer/Consumer Mapping

Phase 2에서 `schedule_two_conv_chain()`는 두 개의 Conv stage를 각각 타일링한 뒤, `build_tile_reuse_dependencies()`에서 producer-consumer 관계를 만든다.

개념적으로는 다음과 같다.

```text
Conv0 output tiles
    |
    | produce activation fragments
    v
Conv1 input tiles
```

어떤 `Conv1` 타일이 실행되려면, 자신이 필요한 입력 범위를 생성한 `Conv0` 타일들이 먼저 완료되어야 한다. 이 의존성은 공간 중첩과 채널 중첩으로 판정된다.

### 2.2 Spatial Overlap Formula

두 타일 \(A\), \(B\)의 높이 방향 overlap은 다음과 같다.

\[
\text{overlap}_h(A, B) =
\neg (A_{h1} \le B_{h0} \lor B_{h1} \le A_{h0})
\]

너비 방향 overlap은:

\[
\text{overlap}_w(A, B) =
\neg (A_{w1} \le B_{w0} \lor B_{w1} \le A_{w0})
\]

전체 spatial overlap은:

\[
\text{spatial\_overlap}(A, B) = \text{overlap}_h(A, B) \land \text{overlap}_w(A, B)
\]

코드에서는 이 로직이 `overlaps_spatial()`로 구현된다.

### 2.3 Channel Overlap Formula

채널 overlap은 다음 조건으로 정의된다.

\[
\text{channel\_overlap}(A, B) =
\neg (A_{c1} \le B_{c0} \lor B_{c1} \le A_{c0})
\]

여기서:

- \(A\): `Conv0.output_region`
- \(B\): `Conv1.input_region`

즉 공간과 채널이 모두 겹쳐야 `Conv0` 타일이 `Conv1` 타일의 producer가 된다.

### 2.4 Barrier Conversion

producer-consumer 관계가 생기면, 스케줄러는 이를 하드웨어 배리어로 변환한다.

변환 규칙은 다음과 같다.

1. 각 producer tile은 자기 완료를 알리는 `signal_barriers`를 가진다.
2. consumer tile은 자신이 의존하는 producer들의 barrier id를 `wait_barriers`에 저장한다.
3. 런타임은 `wait_barriers`가 모두 signaled 상태일 때만 해당 consumer tile을 실행한다.

개념도는 아래와 같다.

```text
Conv0.Tile[k]
   |
   | signal barrier Bk
   v
BarrierState[Bk] = signaled
   |
   | wait barrier Bk
   v
Conv1.Tile[m] may start
```

여러 producer가 하나의 consumer에 기여하는 경우:

```text
Conv0.Tile[1] ---- signal B1001 ---+
                                   |
Conv0.Tile[2] ---- signal B1002 ---+--> Conv1.Tile[7] waits on [B1001, B1002]
                                   |
Conv0.Tile[3] ---- signal B1003 ---+
```

이 구조 덕분에 DPU가 잘못된 순서로 consumer tile을 실행하는 일이 없다.

## 3. Tile Reuse Strategy

### 3.1 Reusing SRAM-Resident Producer Output

Phase 2 구현의 핵심 확장은 `Conv0` 결과를 DDR에 내렸다가 다시 읽지 않고, 가능한 경우 SRAM에 남겨둔 채 `Conv1`이 그대로 읽도록 하는 `Tile Reuse` 전략이다.

reuse 판단 조건은 개념적으로 다음과 같다.

- producer와 consumer가 공간적으로 겹친다.
- producer와 consumer가 채널 방향으로도 겹친다.
- consumer 입력이 producer output bank를 그대로 참조할 수 있다.

reuse가 성립하면 consumer는:

- `mem_plan.reuses_producer_sram = true`
- `mem_plan.reused_from_tile_id = producer_tile_id`
- `mem_plan.input_tile = producer.mem_plan.output_tile`

를 갖게 된다.

즉 `Conv1` 입력은 별도의 DMA load 없이 producer의 SRAM-resident output allocation을 그대로 참조한다.

### 3.2 ASCII Art: Reuse Path

```text
Without reuse:

Conv0 compute
   -> SRAM output
   -> DMA store to DDR
   -> DMA load back to SRAM
   -> Conv1 compute

With reuse:

Conv0 compute
   -> SRAM output
   -> Conv1 reads same SRAM tile directly
```

이 전략은 DMA 트래픽을 줄이고, DPU idle 구간을 줄이는 데 중요하다.

## 4. Tile Contract

### 4.1 Role of the Contract

`ConvTileContract`는 컴파일러가 런타임에 넘겨주는 타일 단위 실행 계약이다. 이 구조체는 "무엇을 계산할지", "어디에 놓을지", "언제 시작할 수 있는지"를 동시에 담는다.

### 4.2 Contract Fields

`ConvTileContract`는 다음 필드를 가진다.

```text
ConvTileContract
├─ task
│  ├─ global_tile_id
│  ├─ op_id
│  ├─ stage_id
│  ├─ seq_no
│  └─ pipeline_phase
├─ input_tensor_id
├─ weight_tensor_id
├─ output_tensor_id
├─ params
├─ input_region
├─ weight_region
├─ output_region
├─ mem_plan
├─ wait_barriers
├─ signal_barriers
├─ producer_tile_ids
└─ consumer_tile_ids
```

### 4.3 `task` Metadata

`task`는 실행 순서를 위한 최소 메타데이터를 가진다.

- `global_tile_id`: 전체 그래프에서 유일한 타일 id
- `op_id`: 원본 Conv op id
- `stage_id`: `Conv0`, `Conv1` 같은 stage 구분
- `seq_no`: stage 내부 순서
- `pipeline_phase`: bank A/B 핑퐁을 위한 parity 정보

`pipeline_phase`는 다음처럼 사용된다.

\[
\text{pipeline\_phase} = \text{seq\_no} \bmod \text{sram\_bank\_count}
\]

즉 2-bank 구조라면:

```text
seq_no:          0    1    2    3    4
pipeline_phase:  0    1    0    1    0
bank:            A    B    A    B    A
```

### 4.4 `mem_plan`

`TileMemoryPlan`은 타일이 사용할 SRAM bank와 local allocation을 정의한다.

주요 필드는 다음과 같다.

- `bank_id`
- `input_tile`
- `weight_tile`
- `output_tile`
- `reuses_producer_sram`
- `reused_from_tile_id`

이 구조 덕분에 런타임은 추가 추론 없이도 "이 타일이 어느 bank에서 실행되는지", "입력을 새로 DMA로 받아야 하는지", "producer SRAM을 재사용하는지"를 즉시 알 수 있다.

## 5. End-to-End Scheduling Flow

전체 스케줄링 흐름은 다음과 같다.

```text
GraphDesc (Conv0 -> Conv1)
        |
        v
make_problem()
        |
        v
schedule_conv_tiles() for Conv0
schedule_conv_tiles() for Conv1
        |
        v
build_tile_reuse_dependencies()
        |
        v
ConvTileContract list
        |
        v
Runtime consumes wait/signal + mem_plan
```

컴파일러 관점:

```text
NCHW tensor
  -> output tile partition
  -> input footprint back-projection
  -> SRAM bank assignment
  -> producer/consumer graph
  -> wait/signal barrier contract
```

런타임 관점:

```text
ConvTileContract
  -> wait on producer barriers
  -> load / reuse SRAM input
  -> compute on assigned bank
  -> signal completion barrier
```

## 6. Why This Matters for Intel NPU

Intel NPU는 제한된 LOCAL_SRAM과 고정된 bank 구조를 가진다. 따라서 scheduler는 단순히 연산 순서만 정하는 것이 아니라:

- 어떤 타일이 어느 bank에 올라갈지
- 어떤 타일이 SRAM 결과를 재사용할지
- 어떤 barrier를 기다려야 할지

를 명시적으로 결정해야 한다.

이 방식은 다음 장점을 준다.

- DPU가 producer 미완료 데이터를 읽는 일을 방지
- DDR round-trip을 줄여 bandwidth pressure 완화
- 이후 Phase 3의 DMA/Compute overlap lowering에 그대로 연결 가능

## Conclusion

Phase 2의 스케줄러는 단순한 타일 분할기가 아니라, Intel NPU 하드웨어 실행 모델을 반영한 contract generator다. 핵심은 다음 세 가지다.

1. 출력 중심의 NCHW 타일 분할
2. spatial/channel overlap 기반 producer-consumer 의존성 계산
3. wait/signal barrier와 SRAM reuse 정보를 포함한 `ConvTileContract` 생성

이 구조 덕분에 런타임은 배리어 위반 없이 안전하게 타일을 실행할 수 있고, 가능한 경우 `Conv0`의 SRAM-resident 결과를 `Conv1`이 즉시 재사용할 수 있다.
```
