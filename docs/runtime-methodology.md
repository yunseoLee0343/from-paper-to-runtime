# Runtime Methodology

이 저장소는 논문을 단순 요약하지 않고, 아래 3단계를 연결해서 봅니다.

1. `paper`
2. `code`
3. `runtime`

## 1. Paper

먼저 논문이 제안하는 구조와 가정을 정리합니다.

- 어떤 연산 구조를 도입했는가
- 어떤 병목을 해결하려 했는가
- 당시 하드웨어 제약과 어떤 관계가 있었는가

## 2. Code

다음으로 논문의 아이디어가 실제 코드에서 어떤 모듈, 함수, 텐서 형태로 나타나는지 확인합니다.

- PyTorch 구현
- CUDA / C++ 구현
- scheduler / pipeline 로직

## 3. Runtime

마지막으로 실행 시점의 동작을 해석합니다.

- 어떤 kernel이 launch되는가
- 어떤 시점에 scheduler가 forward progress를 만드는가
- memory access와 data movement가 병목이 되는가
- idle / stall 구간이 어디서 생기는가

## What We Care About

이 저장소는 특히 다음 문제를 중요하게 봅니다.

- scheduler는 반복되는데 실제 launch가 없는 no-progress 상태
- memory bandwidth와 tensor movement가 latency를 만드는 상황
- 논문 구조만으로 설명되지 않는 runtime behavior

## Expected Output

각 스터디는 가능하면 아래 산출물을 갖도록 정리합니다.

- 논문 요약 또는 개념 노트
- 코드 구현 또는 코드 매핑 자료
- runtime 분석 메모
- 테스트 또는 재현 코드

핵심은 "논문이 맞다"가 아니라, "왜 실제 실행이 이렇게 보이는지 설명할 수 있다"입니다.
