# Study Roadmap

현재 저장소는 `AlexNet`을 시작점으로 정리되어 있습니다.

이후에는 같은 분석 프레임으로 다음 축을 확장할 예정입니다.

## Planned Topics

- Transformer
- BERT
- GPT 계열
- LoRA
- RLHF / InstructGPT
- RAG
- attention vs state-space model runtime 비교

## Common Template

가능하면 각 논문 또는 주제는 아래 구조를 따릅니다.

```text
studies/<topic>/
├─ README.md
├─ paper/
├─ pytorch/
├─ gpu_runtime/
└─ npu_runtime/
```

모든 주제가 정확히 같은 하위 폴더를 가지는 것은 아니지만, 최소한 아래 흐름은 유지하려고 합니다.

1. 논문 이해
2. 코드 확인 또는 재구현
3. runtime path 분석
4. 병목 정리
