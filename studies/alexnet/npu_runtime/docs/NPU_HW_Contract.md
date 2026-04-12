# Intel NPU Hardware Interface Specification

## Overview

이 문서는 Phase 1에서 구현한 `intel_npu_defs.hpp`, `intel_npu_simulator.cpp`, `tests/intel_npu_test_p1.cpp`를 기준으로 Intel NPU 스타일 하드웨어 인터페이스 계약을 정리한다. 초점은 다음 세 가지다.

- `HardwareCaps`가 정의하는 SRAM 및 DMA 하드웨어 한계
- DDR과 `MemorySpace::LOCAL_SRAM` 사이의 데이터 이동 규칙
- FP16/BF16 중심 데이터 타입과 메모리 점유 계산 규칙

이 계약은 이후의 타일 스케줄러와 overlap 파이프라인이 의존하는 가장 낮은 수준의 메모리/전송 규약이다.

## 1. Hardware Capabilities

### 1.1 SRAM Capacity and Bank Topology

Phase 1 구현에서 Intel NPU의 on-chip scratchpad는 `HardwareCaps::local_sram_bytes`와 `HardwareCaps::sram_bank_count`로 모델링된다.

기본값은 다음과 같다.

- Total SRAM: `4 * 1024 * 1024` bytes
- Bank count: `2`
- DMA alignment: `64` bytes

즉 기본 bank 크기는 아래와 같이 계산된다.

\[
\text{bank\_size} = \frac{\text{local\_sram\_bytes}}{\text{sram\_bank\_count}}
\]

기본 설정에서는

\[
\text{bank\_size} = \frac{4\,\text{MiB}}{2} = 2\,\text{MiB}
\]

이다.

`IntelNPUSimulator`는 각 bank를 독립적인 byte array로 유지하고, `LocalSramAllocator`는 bank-local offset을 기준으로 allocation을 관리한다. 이 설계는 Intel NPU의 타일 기반 실행에서 한 bank가 현재 연산용 working set을 담고, 다른 bank가 다음 타일 prefetch를 받는 ping-pong 패턴과 자연스럽게 맞는다.

### 1.2 DMA Alignment Rule

DMA 정렬 규칙은 `HardwareCaps::dma_alignment = 64`로 표현된다. `IntelNPUSimulator::validate_dma_alignment()`는 아래 세 조건을 강제한다.

- `desc.bytes % 64 == 0`
- `desc.bank_offset % 64 == 0`
- `reinterpret_cast<uintptr_t>(desc.host_ptr) % 64 == 0`

즉 DMA payload, SRAM bank offset, host staging buffer 시작 주소가 모두 64바이트 배수여야 한다.

### 1.3 Why 64-Byte Alignment Matters

Intel NPU 관점에서 64바이트 정렬이 중요한 이유는 다음과 같다.

1. DMA burst 효율
   Intel 계열 가속기는 고정 burst 또는 cacheline 유사 단위로 메모리를 읽고 쓰는 경향이 있다. 64B 정렬은 burst split을 줄여 descriptor 하나당 실제 유효 payload 비율을 높인다.

2. SRAM bank access 단순화
   bank-local offset이 64B 경계에 맞으면 SRAM 내부 주소 decode와 bank arbitration이 단순해지고, 부분 line 접근으로 인한 read-modify-write 비용을 줄이기 쉽다.

3. 하드웨어/펌웨어 descriptor 규격 일관성
   Level Zero 스타일 비동기 제출 경로에서는 command descriptor와 transfer payload가 predictable한 alignment를 가질수록 firmware 검증, queue packing, completion tracking이 단순해진다.

4. 타일 단위 재사용 안정성
   이후 phase에서 output tile을 다음 stage input으로 SRAM reuse할 때도 동일 alignment 규칙을 유지하면 bank A/B 교차 재사용 시 주소 해석이 흔들리지 않는다.

정리하면 64B alignment는 단순한 구현 편의가 아니라, DMA 엔진 처리량, bank conflict 완화, descriptor 검증 단순화까지 묶는 핵심 하드웨어 계약이다.

## 2. Memory Layout

### 2.1 Address Domains

Phase 1은 세 가지 메모리 공간을 정의한다.

- `MemorySpace::HOST`
  CPU가 직접 접근하는 테스트 및 staging 메모리
- `MemorySpace::DDR`
  외부 DRAM/DDR에 대응하는 off-chip 저장 공간
- `MemorySpace::LOCAL_SRAM`
  Intel NPU on-chip shared SRAM

실제 Phase 1 테스트는 host aligned buffer를 사용하지만, 의미적으로는 `HOST`가 `DDR` staging을 포함한 상위 계층이고 `LOCAL_SRAM`이 NPU 내부 scratchpad 계층이다.

### 2.2 DDR to SRAM Mapping

DMA descriptor는 다음 정보를 통해 DDR/host와 SRAM 사이 매핑을 정의한다.

- `direction`
- `host_ptr`
- `bank_id`
- `bank_offset`
- `bytes`
- `alignment`

`HostToLocal`인 경우:

```text
host_ptr / DDR staging
    -> DMA
    -> LOCAL_SRAM[bank_id][bank_offset : bank_offset + bytes)
```

`LocalToHost`인 경우:

```text
LOCAL_SRAM[bank_id][bank_offset : bank_offset + bytes)
    -> DMA
    -> host_ptr / DDR staging
```

Phase 1 구현은 bank-local offset만 다루므로 global SRAM 주소를 평탄화하지 않는다. 이 방식은 bank conflict 검사와 bounds validation을 더 직관적으로 만든다.

### 2.3 LocalAlloc Strategy

`LocalSramAllocator::allocate()`는 다음 순서로 `LocalAlloc`을 만든다.

1. 요청 크기와 alignment 유효성 검사
2. `preferred_bank`가 있으면 해당 bank 우선 사용
3. 없으면 cursor가 가장 작은 bank부터 시도
4. 각 bank에서 `align_up(cursor, alignment)`로 시작 offset 계산
5. bank capacity 초과 여부 확인
6. 기존 reservation과 겹치는지 확인
7. 성공 시 reservation 등록 후 cursor 갱신

즉 allocation은 아래 제약을 만족해야 한다.

\[
\text{offset} \equiv 0 \pmod{\text{alignment}}
\]

\[
\text{offset} + \text{bytes} \le \text{bank\_size}
\]

두 allocation \(A\), \(B\)가 같은 bank에 있을 때 bank conflict가 없으려면

\[
A_{\text{end}} \le B_{\text{offset}}
\quad \text{or} \quad
B_{\text{end}} \le A_{\text{offset}}
\]

여기서

\[
A_{\text{end}} = A_{\text{offset}} + A_{\text{bytes}}
\]

이다.

### 2.4 Bank Conflict Avoidance

Phase 1 설계는 bank 내부 충돌을 "사전 방지"하는 방식이다. 즉 이미 배치된 모든 `LocalAlloc`과 candidate를 비교해 조금이라도 겹치면 allocation을 거부한다.

이 접근의 장점은 다음과 같다.

- DMA와 compute가 같은 bank 내 동일 영역을 우발적으로 덮어쓰지 않음
- 후속 phase에서 타일별 bank residency 추론이 쉬움
- 디버깅 시 bank별 reservation dump만으로 오류 위치를 좁히기 쉬움

## 3. Data Types and Footprint Calculation

### 3.1 Supported Data Types

Phase 1 헤더에는 이후 단계와 호환되도록 다음 `DataType`을 명시했다.

- `FP32`
- `FP16`
- `BF16`
- `INT8`

각 element size는 `dtype_size()`로 계산된다.

| DataType | Bytes per element |
| --- | ---: |
| `FP32` | 4 |
| `FP16` | 2 |
| `BF16` | 2 |
| `INT8` | 1 |

### 3.2 Tensor Footprint Formula

총 메모리 점유는 element 수와 data type byte width의 곱으로 계산한다.

\[
\text{tensor\_bytes} = \text{elements} \times \text{dtype\_size}
\]

헤더 유틸은 이를 `tensor_bytes(elements, dtype)`로 제공한다.

예를 들어 FP16 activation tile이 4096 element면:

\[
4096 \times 2 = 8192\ \text{bytes}
\]

BF16도 element width가 2바이트이므로 footprint는 FP16과 동일하다. 반면 FP32 accumulator는 같은 element 수라도 FP16 대비 2배 공간을 사용한다.

### 3.3 Practical Guidance

Intel NPU에서 타입 선택은 단순 precision 문제가 아니라 SRAM residency와 직결된다.

- FP16/BF16
  activation과 weight tile에 적합하며 SRAM occupancy를 낮출 수 있다.
- FP32
  partial sum이나 accumulation에 유리하지만 SRAM pressure를 키운다.
- INT8
  가장 작은 footprint를 제공하지만 quantization contract가 추가로 필요하다.

따라서 tile contract를 설계할 때는 precision 목표와 bank 크기를 동시에 고려해야 한다.

## 4. Interface Summary

Phase 1 기준 핵심 인터페이스는 아래와 같다.

### `HardwareCaps`

- SRAM 총량
- SRAM bank 개수
- DMA alignment
- 향후 phase에서 재사용되는 타일 크기 상한

### `LocalAlloc`

- allocation 이름
- 메모리 공간
- bank id
- bank-local offset
- byte size
- alignment

### `DmaDescriptor`

- 전송 방향
- host pointer
- target/source bank id
- bank-local offset
- transfer bytes
- alignment tag

### `IntelNPUSimulator`

- `allocate_local()`
  bank-aware LOCAL_SRAM allocation
- `exec_dma()`
  alignment + bounds 검증 후 실제 copy 수행
- `read_local()`, `write_local()`
  테스트 및 상태 검증용 SRAM access helper

## 5. Validation in Phase 1 Test

`tests/intel_npu_test_p1.cpp`는 다음 하드웨어 계약을 검증한다.

1. bank 0과 bank 1에 독립 allocation 가능
2. host -> SRAM load가 정상 동작
3. SRAM -> host store가 정상 동작
4. 48-byte misaligned DMA가 예외로 거부됨
5. DMA load/store byte counter가 기대값과 일치함

즉 Phase 1은 "bank-aware SRAM allocator + 64B aligned DMA contract"가 실제 실행 경로에서 강제된다는 점을 테스트로 닫아준다.

## Conclusion

Intel NPU Hardware Interface Specification의 핵심은 세 줄로 요약할 수 있다.

- on-chip SRAM은 banked scratchpad로 다뤄야 하며 allocation은 bank-local conflict를 허용하지 않는다.
- DMA는 64바이트 정렬을 필수 계약으로 삼아야 한다.
- 데이터 타입 선택은 precision뿐 아니라 SRAM 점유와 파이프라인 지속성까지 좌우한다.

이 Phase 1 계약이 안정적으로 잡혀 있기 때문에, 상위 단계에서 타일 스케줄링, barrier graph, DMA/compute overlap 같은 최적화가 안전하게 쌓일 수 있다.
