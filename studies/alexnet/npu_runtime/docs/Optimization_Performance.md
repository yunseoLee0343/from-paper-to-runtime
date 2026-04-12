# NPU Pipeline Optimization & Benchmarking Report

## Overview

이 문서는 Phase 3에서 구현한 Intel NPU 파이프라인 최적화 로직을 설명한다. 기준 구현은 `intel_npu_pipeline.cpp`의 `OverlapLowerer::lower_stage_stream()`, `PipelineRuntime::run()`, 그리고 `tests/intel_npu_test_p3.cpp`다.

Phase 3의 목표는 세 가지다.

- Bank A/B를 교차 사용하는 double buffering으로 SRAM 유휴 시간을 줄인다.
- `Load(Next) / Compute(Curr) / Store(Prev)`를 비동기적으로 겹쳐서 DMA와 DPU를 동시에 바쁘게 만든다.
- 실행 trace에서 overlap triplet, DMA 바이트 수, overlap 비율을 계산해 정량적으로 해석할 수 있게 한다.

## 1. Double Buffering

### 1.1 Ping-Pong Buffering Principle

Intel NPU의 LOCAL_SRAM은 banked scratchpad로 모델링되며, Phase 3에서는 두 개의 bank를 ping-pong 방식으로 사용한다.

기본 개념은 다음과 같다.

- Bank A: 현재 타일의 compute working set
- Bank B: 다음 타일의 DMA prefetch 대상
- 다음 cycle group에서 역할 교대

즉 한 bank가 compute에 묶여 있는 동안, 다른 bank는 다음 tile input/weight를 미리 적재한다.

### 1.2 Phase Parity Control

Bank 선택은 `pipeline_phase`에 의해 결정된다. Phase 2 contract에서 이미 다음 식으로 parity가 만들어져 있다.

\[
\text{pipeline\_phase} = \text{seq\_no} \bmod \text{sram\_bank\_count}
\]

bank 수가 2개일 때:

```text
seq_no:          0    1    2    3    4
pipeline_phase:  0    1    0    1    0
bank_id:         A    B    A    B    A
```

이 값은 `ConvTileContract.mem_plan.bank_id`에 반영되고, `OverlapLowerer::lower_stage_stream()`는 그 bank id를 그대로 사용해 DMA와 compute command를 발행한다.

### 1.3 Why It Matters

double buffering이 없으면 다음 순서가 된다.

```text
load tile 0
compute tile 0
store tile 0
load tile 1
compute tile 1
store tile 1
```

이 경우 DMA와 compute는 번갈아 쉬게 된다.

반면 ping-pong을 쓰면:

```text
bank A: compute tile 0
bank B: load tile 1

then

bank B: compute tile 1
bank A: load tile 2 / store tile 0
```

처럼 engine overlap이 가능해진다.

## 2. DMA/Compute Overlap

### 2.1 Asynchronous Level Zero Style Lowering

`OverlapLowerer::lower_stage_stream()`는 각 tile에 대해 세 시점을 겹치도록 command를 만든다.

- 현재 tile의 dependency barrier wait
- 다음 tile의 prefetch DMA
- 현재 tile의 compute
- 이전 tile의 output store DMA
- 현재 tile의 completion signal

개념적으로는 다음 파이프라인을 만든다.

```text
Phase 0:  Load(T0)
Phase 1:  Load(T1) + Compute(T0)
Phase 2:  Load(T2) + Compute(T1) + Store(T0)
Phase 3:  Load(T3) + Compute(T2) + Store(T1)
...
Drain:                            Store(Tlast)
```

이 구조는 Level Zero의 비동기 command submission처럼 DMA queue와 compute queue가 독립적으로 진행되는 모델에 가깝다.

### 2.2 Tile Reuse Interaction

Phase 2에서 consumer tile이 producer SRAM output을 재사용하는 경우, `mem_plan.reuses_producer_sram == true`가 되고 input DMA load를 생략할 수 있다.

그 결과 overlap 구조는 더 좋아질 수 있다.

- input reload 제거
- DDR traffic 감소
- DMA queue 혼잡 완화
- compute start 시점 단축 가능

즉 Phase 2의 tile reuse는 Phase 3 overlap 효율을 올리는 전처리 최적화 역할을 한다.

## 3. Pipeline Timeline

### 3.1 High-Level Gantt View

아래 표는 `Load(Next) / Compute(Curr) / Store(Prev)` 구조를 시간축으로 나타낸 것이다.

```text
Time -----> 

DMA Engine      | Load T0 | Load T1 | Load T2 | Store T0 | Load T3 | Store T1 | ... |
Compute Engine  |         | Compute T0      | Compute T1      | Compute T2      | ... |
Control/Barrier | wait/signal around each tile dependency boundary                          |
SRAM Bank A     | T0 compute data | T2 staging/compute | T4 staging/compute | ... |
SRAM Bank B     | T1 staging      | T1 compute data    | T3 staging/compute | ... |
```

좀 더 타일 중심으로 보면:

```text
Step         DMA Activity                  Compute Activity
-----------  ----------------------------  -------------------------
Warm-up      Load(T0)                      idle
Steady-1     Load(T1)                      Compute(T0)
Steady-2     Load(T2) + Store(T0)          Compute(T1)
Steady-3     Load(T3) + Store(T1)          Compute(T2)
Drain        Store(Tlast)                  idle
```

핵심은 steady-state에 들어가면 DMA와 compute가 동시에 바쁘게 유지된다는 점이다.

### 3.2 Barrier-Aware Timing

`PipelineRuntime::run()`은 command마다 아래 readiness를 계산한다.

- DMA engine ready cycle
- Compute engine ready cycle
- Control engine ready cycle
- Wait barrier가 signaled 되는 cycle

즉 command의 실제 시작 시점은 단순 FIFO가 아니라:

\[
\text{start\_cycle} = \max(\text{engine\_ready}, \text{dependency\_ready})
\]

로 결정된다.

이 때문에 barrier가 늦게 풀리면 overlap 구조가 있어도 compute가 늦게 시작될 수 있다.

## 4. Runtime Metrics

### 4.1 Overlap Triplets

`PipelineRuntime::run()`은 command stream에서 다음 패턴을 발견할 때 `overlapped_triplets`를 증가시킨다.

```text
DMA_LOAD -> COMPUTE_CONV -> DMA_STORE
```

이 값은 steady-state overlap이 실제로 형성되었는지 보는 가장 직관적인 지표다.

- 값이 0에 가까우면 파이프라인이 직렬화되었을 가능성이 높다.
- 값이 증가할수록 load/compute/store 세 단계가 겹친 구간이 많다는 뜻이다.

### 4.2 DMA Throughput Proxies

Phase 3 구현은 실제 GB/s 대신 다음 counter를 제공한다.

- `dma_load_bytes`
- `dma_store_bytes`
- `dma_busy_cycles`

이 값들로 평균 DMA 처리 효율을 상대 비교할 수 있다.

\[
\text{effective load bytes per cycle} =
\frac{\text{dma\_load\_bytes}}{\text{dma\_busy\_cycles}}
\]

\[
\text{effective store bytes per cycle} =
\frac{\text{dma\_store\_bytes}}{\text{dma\_busy\_cycles}}
\]

이 문서의 구현에서는 `dma_cycles()`가 64B alignment 단위로 DMA 시간을 근사하므로, alignment 위반이나 작은 fragmented transfer가 많아질수록 효율이 떨어진다.

### 4.3 DMA/Compute Overlap Percent

가장 중요한 지표는 `overlap_percent`다.

\[
\text{overlap\_percent} =
100 \times
\frac{\text{dma\_compute\_overlap\_cycles}}{\text{total\_cycles}}
\]

이 값은 전체 실행 시간 중 DMA와 compute가 동시에 active였던 비율을 의미한다.

해석은 보통 다음처럼 할 수 있다.

- `0%` 근처: 거의 직렬 실행
- `30% ~ 60%`: 일부 overlap 형성
- `60%+`: steady-state pipeline이 비교적 잘 형성됨
- `80%+`: 매우 공격적으로 겹치지만 barrier, bank pressure, store drain 비용도 함께 점검 필요

## 5. Benchmark Result Summary

현재 Phase 3 테스트(`tests/intel_npu_test_p3.cpp`) 실행 결과는 다음과 같다.

```text
Intel NPU Phase 3 test passed
  Total cycles      : 18432
  Overlap triplets  : 2
  DMA/Compute overlap % : 73.65
```

이 결과의 해석은 다음과 같다.

- `Total cycles = 18432`
  전체 그래프(`Conv0 -> Conv1`)가 완료되는 데 걸린 시뮬레이션 시간
- `Overlap triplets = 2`
  load/compute/store의 3중 overlap 패턴이 적어도 두 번 형성됨
- `Overlap % = 73.65`
  총 실행 시간의 약 74%에서 DMA와 compute가 동시에 active였음

즉 현재 파이프라인은 warm-up/drain 구간을 제외한 steady-state에서 상당히 높은 수준의 overlap을 형성하고 있다고 볼 수 있다.

## 6. Troubleshooting Checklist

성능이 기대보다 낮거나 overlap 비율이 떨어질 때는 아래 항목을 순서대로 확인하는 것이 좋다.

### Scheduling and Contract

- `pipeline_phase`가 실제로 `0/1/0/1` 식으로 교차하는가
- `mem_plan.bank_id`가 parity와 일치하는가
- `reuses_producer_sram`가 기대한 tile에서 활성화되는가
- `wait_barriers`가 과도하게 많아 consumer가 너무 오래 막히지 않는가

### DMA Path

- DMA transfer 크기가 64B alignment를 만족하는가
- 너무 작은 load/store가 많이 쪼개져 있지 않은가
- 불필요한 input reload가 발생하고 있지 않은가
- tile reuse가 가능한데도 DDR round-trip을 하고 있지 않은가

### Compute Path

- `max_tile_h/w/c` 설정이 너무 작아 compute granularity가 지나치게 잘리지 않는가
- compute workload가 DMA보다 너무 작아서 overlap 구간이 짧아지지 않는가
- output tile shape가 DPU utilization에 비해 비효율적으로 설정되지 않았는가

### Barrier and Dependency Graph

- producer가 signal하기 전에 consumer가 wait하고 있지 않은가
- spatial overlap 판정이 너무 보수적이라 불필요한 dependency가 늘어나지 않았는가
- deadlock은 없더라도 long wait chain이 생기고 있지 않은가

### Runtime Metrics

- `overlapped_triplets`가 0 또는 지나치게 낮지 않은가
- `dma_busy_cycles`만 크고 `compute_busy_cycles`가 작지 않은가
- `overlap_percent`가 warm-up/drain 비용 때문에 떨어지는지, steady-state 자체가 나쁜 것인지 구분했는가

## 7. Recommendations

현재 Phase 3 구조를 더 개선하려면 다음 방향이 유효하다.

- stage-local lowering을 graph-global lowering으로 확장해 stage 간 경계에서도 overlap 유지
- tile reuse 조건을 더 정교하게 만들어 input DMA를 더 많이 제거
- DMA queue와 compute queue를 분리된 command list 모델로 세분화
- 실하드웨어 포팅 시 Level Zero event/signal object와 barrier id를 직접 매핑

## Conclusion

Phase 3의 핵심은 단순히 명령을 나열하는 것이 아니라, `phase parity -> bank assignment -> asynchronous load/compute/store overlap -> runtime metric accounting`으로 이어지는 전체 파이프라인을 일관되게 만드는 것이다.

현재 구현은 다음을 달성한다.

- Bank A/B 기반 ping-pong double buffering
- `Load(Next) / Compute(Curr) / Store(Prev)` 구조의 overlap lowering
- `Overlap Triplets`, DMA bytes, overlap percent를 통한 정량 평가

즉 Intel NPU 스타일의 타일 기반 실행을 성능 관점에서 검증할 수 있는 최소한의 benchmarkable runtime 모델이 완성된 상태다.
