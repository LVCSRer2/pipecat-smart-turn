# Smart Turn v3.2 실시간 마이크 데모

Pipecat(Daily.co)의 [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn) 모델을 활용한 실시간 발화 완료(End-of-Turn) 감지 데모입니다.

## 개요

마이크 음성 입력을 실시간으로 처리하여 사용자의 발화가 완료되었는지(EOT) 아직 말하는 중인지(CONT) 판별합니다.

```
마이크 (16kHz mono) → Silero VAD (음성 구간 감지) → Smart Turn v3.2 (EOT/CONT 판별)
```

| 항목 | 값 |
|------|---|
| 베이스 모델 | Whisper Tiny 인코더 + Linear 분류 헤드 |
| 모델 크기 | 8MB (CPU, int8 QAT) |
| 추론 속도 | ~12ms (CPU) |
| 입력 | 오디오 (16kHz mono PCM, 최대 8초) |
| 출력 | 확률 (0~1), ≥0.5 → EOT |
| 지원 언어 | 23개 (한국어 96.85%, 영어 94.31% 등) |

## 파일 구조

```
.
├── demo.py              # 메인 데모 (실시간 마이크 입력)
├── inference.py         # Smart Turn ONNX 추론 엔진
├── audio_utils.py       # 오디오 전처리 (8초 자르기/패딩)
├── requirements.txt     # 의존성 패키지
└── smart_turn_survey.md # Smart Turn 모델 서베이 문서
```

## 설치 및 실행

```bash
pip install -r requirements.txt
python demo.py
```

첫 실행 시 아래 모델을 자동 다운로드합니다:
- **Silero VAD** ONNX (~2.3MB) - 음성 구간 감지용
- **Smart Turn v3.2** ONNX (~8.7MB) - 발화 완료 판별용 ([HuggingFace](https://huggingface.co/pipecat-ai/smart-turn-v3))

## 동작 방식

1. **Silero VAD**가 마이크 입력에서 음성 시작/종료를 실시간 감지
2. 침묵이 1초 지속되면 해당 발화 구간을 **Smart Turn v3.2**에 전달
3. 확률 ≥ 0.5 → **EOT** (발화 완료, AI 응답 타이밍) / < 0.5 → **CONT** (아직 말하는 중)

### 출력 예시

```
  음성 감지... 분석 중...

  ┌─ 발화 #1 (1.28초) ──────────────────
  │ 판정: EOT (발화 완료)
  │ 확률: [███████████████████░] 0.9575
  │ 추론: 338.6ms
  └──────────────────────────────────────
```

## 설정

`demo.py` 상단에서 주요 파라미터를 조정할 수 있습니다:

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `VAD_THRESHOLD` | 0.5 | Silero VAD 음성 감지 임계값 |
| `STOP_MS` | 1000 | 침묵 지속 시 발화 종료 판정 (ms) |
| `MAX_DURATION_SECONDS` | 8 | 최대 발화 길이 (초) |
| `DEBUG_SAVE_WAV` | False | True 시 각 발화를 WAV 파일로 저장 |

## 참고 자료

- [Smart Turn GitHub](https://github.com/pipecat-ai/smart-turn)
- [Smart Turn v3 HuggingFace](https://huggingface.co/pipecat-ai/smart-turn-v3)
- [Smart Turn v3 블로그](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)
