# AlexNet NPU Runtime

이 디렉터리는 AlexNet을 NPU-style execution model 관점에서 실험한 코드와 문서를 담고 있습니다.

핵심 관심사는 아래와 같습니다.

- scheduler가 어떤 단위로 work를 쪼개는가
- pipeline과 synchronization이 execution에 어떤 제약을 주는가
- tiling과 hardware contract가 실제 실행 경로를 어떻게 제한하는가

## Main Files

- [intel_npu_scheduler.cpp](/d:/ai_study/studies/alexnet/npu_runtime/intel_npu_scheduler.cpp)
- [intel_npu_pipeline.cpp](/d:/ai_study/studies/alexnet/npu_runtime/intel_npu_pipeline.cpp)
- [raw/alexnet_npu.cpp](/d:/ai_study/studies/alexnet/npu_runtime/raw/alexnet_npu.cpp)

## Documents

- [docs/NPU_HW_Contract.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/NPU_HW_Contract.md)
- [docs/Tiling_and_Sync.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/Tiling_and_Sync.md)
- [docs/Optimization_Performance.md](/d:/ai_study/studies/alexnet/npu_runtime/docs/Optimization_Performance.md)

## Tests

`tests/` 아래의 phase별 C++ 테스트로 pipeline/scheduler 동작을 확인할 수 있습니다.
