# Smart Turn 서베이

Pipecat(Daily.co)의 오디오 기반 Semantic Turn Detection 모델 조사 결과.

---

## 1. 기본 정보

| 항목 | 값 |
|------|---|
| **개발** | Pipecat (Daily.co) |
| **최신 버전** | v3.2 |
| **베이스 모델** | Whisper Tiny 인코더 (디코더 미사용) |
| **파라미터 수** | ~8M (Whisper Tiny 39M 중 인코더 + Linear 분류 헤드) |
| **모델 크기** | 8MB (CPU, int8 QAT) / 32MB (GPU, fp32) |
| **추론 속도** | 12ms (CPU) / 3.3ms (GPU) |
| **입력** | 오디오 (16kHz mono PCM, 최대 8초) |
| **출력** | 확률 (0~1), ≥0.5 → EOT |
| **지원 언어** | 23개 (한국어 포함) |
| **라이선스** | BSD 2-Clause (가중치 + 학습 데이터 + 학습 코드 모두 공개) |
| **GitHub** | [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) |
| **HuggingFace** | [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) |

---

## 2. 버전 변천

| 버전 | 베이스 모델 | 크기 | 특징 |
|------|-----------|------|------|
| **v1** | wav2vec2-BERT | - | 영어 전용, 초기 버전 |
| **v2** | wav2vec2 + Linear | ~400MB | 14개 언어로 확장, 필러 워드 학습 |
| **v3** | Whisper Tiny 인코더 + Linear | **8MB** | 23개 언어, 50배 경량화, 12ms 추론 |
| **v3.1** | Whisper Tiny 인코더 + Linear | 8MB | 영어/스페인어 정확도 개선 |
| **v3.2** | Whisper Tiny 인코더 + Linear | 8MB | **전체 학습 데이터셋 공개 및 품질 고도화** |

---

## 3. 모델 아키텍처

### 구조

```
Whisper Tiny (39M params)
├── 인코더 (사용) ← Smart Turn이 활용하는 부분
│   ├── Conv1d (오디오 → 특징 추출)
│   └── Transformer 인코더 (4 layers × 6 heads)
│       → 음향 + 의미적 잠재 표현 출력
│
└── 디코더 (사용하지 않음, 제거)

Smart Turn 분류 헤드
└── Linear Layer → 확률 (0~1)
```

### 아키텍처 선택 과정

ablation study를 통해 여러 구조를 비교한 결과:

| 후보 | 결과 |
|------|------|
| wav2vec2-BERT | v2에서 사용, 크기 큼 |
| wav2vec2 | v2 대안 |
| LSTM | 성능 부족 |
| 추가 Transformer 분류 레이어 | 오버킬 |
| **Whisper Tiny + Linear** | **최종 채택 (v3)** |

---

## 4. 내부 동작 파이프라인

```
사용자 음성
    │
    ▼
┌─────────────────────────────────┐
│  1. Silero VAD (음성 구간 감지)   │
│  → 음성이 끝나고 침묵 시작 감지    │
│  → 해당 발화 오디오 구간 추출      │
└─────────────────────────────────┘
    │
    ▼  (16kHz mono PCM, 최대 8초)
┌─────────────────────────────────┐
│  2. 오디오 전처리 (~3ms)          │
│  · 8초 미만: 앞에 zero-padding   │
│  · 8초 초과: 앞부분 절삭 (끝 유지) │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. Whisper Tiny 인코더           │
│                                  │
│  오디오 → Mel Spectrogram →       │
│  Conv1d → Transformer (4 layers) │
│  → 음향 특징 벡터 추출             │
│                                  │
│  명시적으로 추출되는 정보:          │
│  · 억양/톤 변화 (문장 끝 하강 등)  │
│  · 말 속도/리듬 (prosody)         │
│  · 필러 워드 ("음...", "그...")    │
│  · 침묵 패턴                      │
│                                  │
│  암묵적으로 포함된 정보:            │
│  · 발화 내용의 의미적 표현          │
│  · 문법적 완성도 (간접적)          │
│  · 언어 종류 인식                  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. Linear 분류 헤드               │
│  음향 특징 → Linear → 확률 (0~1)  │
│  · ≥ 0.5 → EOT (발화 완료)       │
│  · < 0.5 → CONT (발화 중)        │
└─────────────────────────────────┘
    │
    ▼
  EOT or CONT
```

### 의미 정보 사용 여부

Smart Turn은 "오디오만 보고 의미는 안 본다"가 아니다. Whisper 인코더는 ASR(음성→텍스트) 학습 과정에서 **의미적 표현을 내재화**했기 때문에, 분류 헤드는 음향 + 의미 정보를 모두 활용할 수 있다.

| Whisper 인코더 층 | 포함된 정보 |
|-----------------|-----------|
| 초기 층 | 음향 특징 (톤, 피치, 에너지) |
| 중간 층 | 화자 특성, 운율(prosody) |
| 최종 층 | 언어적/의미적 내용 (linguistic content) |

그래서 Pipecat 공식 문서에서도 Smart Turn을 **"semantic VAD"** 라고 부른다.

다만 NAMO/LiveKit처럼 텍스트를 직접 분석하는 것보다는 의미 해석의 정밀도가 낮고, 대신 톤, 리듬, 필러 워드 같은 비언어적 신호를 추가로 활용할 수 있다.

---

## 5. 학습 데이터 생성 방식

학습 데이터는 합성(synthetic) 방식으로 생성된다.

```
1. 텍스트 큐레이션
   Gemini 2.5 Flash로 자연스러운 구어체 문장 필터링
   → 50~80% 부적합 샘플 제거

2. 필러 워드 주입
   LLM이 문장 끝 근처를 분리하고 언어별 필러 워드 추가
   예: "My phone number is, um..."
   예: "전화번호가, 음..."

3. TTS 합성
   Google Chirp3 모델로 음성 생성
   · 필러 워드를 자연스럽게 발음
   · 문장 경계에서 적절한 억양 표현

4. 사람 검수
   인간 평가를 통해 품질 95% → 99%로 향상
```

### 공개 데이터셋

- 학습: `pipecat-ai/smart-turn-data-v3.2-train` (HuggingFace)
- 테스트: `pipecat-ai/smart-turn-data-v3.2-test` (HuggingFace)
- 커뮤니티 기여: annotation 플랫폼 및 training game 인터페이스 제공

---

## 6. 지원 언어 및 정확도

23개 언어 지원 (v3 기준):

| 언어 | 코드 | 정확도 |
|------|------|--------|
| Turkish | tr | 97.31% |
| Korean | ko | **96.85%** |
| Japanese | ja | 94.36% |
| English | en | 94.31% (v3.2: 94.7%) |
| German | de | 94.25% |
| Hindi | hi | 93.48% |
| Chinese | zh | - |
| French | fr | - |
| Spanish | es | - (v3.2: 90.1%) |
| 기타 14개 | - | - |

전체 언어: Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Marathi, Norwegian, Polish, Portuguese, Russian, Spanish, Turkish, Ukrainian, Vietnamese

---

## 7. 텍스트 기반 모델과의 비교

### 접근 방식 차이

| | NAMO / LiveKit | Smart Turn |
|--|----------------|------------|
| **입력** | STT가 변환한 텍스트 | 오디오 파형 (16kHz PCM) |
| **STT 필요?** | ✅ 반드시 필요 | ❌ 불필요 |
| **파이프라인 위치** | STT → **Turn Detector** → LLM | VAD → **Turn Detector** → STT → LLM |
| **의미 정보** | 명시적 (텍스트 직접 분석) | 암묵적 (Whisper 인코더 잠재 표현) |
| **음향 정보** | ❌ 없음 | ✅ 톤, 리듬, 필러 워드 |
| **의미 해석 정밀도** | 높음 | 상대적으로 낮음 |
| **음향 해석 정밀도** | 없음 | 높음 |

### 각 방식이 잡는 것 / 못 잡는 것

**텍스트 기반 (NAMO/LiveKit)이 잘 잡는 것:**
- "I think the next logical step is to" → 문법적으로 미완성 (전치사 to로 끝남)
- "그건 좀 생각을 해봐야 할 것 같아." → 의미적으로 완성 (LiveKit이 정확 판별)

**텍스트 기반이 못 잡는 것:**
- 톤 하강/상승 (같은 텍스트라도 억양에 따라 의미가 다름)
- "음...", "그..." 같은 필러 워드 (STT가 무시)
- 말 속도 변화

**오디오 기반 (Smart Turn)이 잘 잡는 것:**
- 톤 하강 → 발화 완료 신호
- 톤 상승 → 질문 또는 발화 중 신호
- 필러 워드 감지 → 사용자가 생각 중
- 말 속도 느려짐 → 문장 끝 패턴

**오디오 기반이 못 잡는 것:**
- 문법적 완성도에 대한 정밀한 분석
- 대화 맥락 (이전 턴 참조 불가)
- 텍스트 수준의 의미 이해

### 보완적 조합

두 방식은 서로 보완적이며, 실제 프로덕션에서는 조합하면 가장 높은 정확도를 얻을 수 있다:

```
오디오 → Silero VAD → Smart Turn (오디오 기반 판별)
                  └→ STT → NAMO/LiveKit (텍스트 기반 판별)
                              └→ 두 결과 종합 → 최종 EOT 판정
```

---

## 8. 크기 및 성능 비교

| 모델 | 입력 | 크기 | 추론 속도 | 한국어 | 라이선스 |
|------|------|------|----------|--------|---------|
| **Smart Turn v3** | 오디오 | **8MB** | **12ms** | ✅ 96.85% | BSD 2-Clause |
| NAMO (단일) | 텍스트 | 135MB | <19ms | ✅ 96.85% | Apache 2.0 |
| NAMO (다국어) | 텍스트 | 295MB | <29ms | ✅ (다국어 평균 90.25%) | Apache 2.0 |
| LiveKit (다국어) | 텍스트 | 281MB | 50-160ms | ✅ TNR 94.5% | LiveKit Model License |

Smart Turn v3는 8MB로 현존하는 turn detection 모델 중 가장 경량이며, 추론 속도도 가장 빠르다.

---

## 9. 프로덕션 통합

### Pipecat 프레임워크 통합

```python
from pipecat.audio.turn import LocalSmartTurnAnalyzerV3

analyzer = LocalSmartTurnAnalyzerV3()
# Pipecat v0.0.85+ 에서 사용 가능
```

### 독립 실행 (standalone)

리포지토리의 `predict.py` 및 `record_and_predict.py`로 독립 추론 가능.

```python
from model import SmartTurnModel
from inference import predict_endpoint

# 오디오 로드 (16kHz mono PCM)
audio_samples = load_audio("input.wav")
result = predict_endpoint(audio_samples)
# result >= 0.5 → 발화 완료
```

### 커스텀 학습

```bash
# 로컬 학습
python train.py

# Modal을 통한 분산 학습
modal run train.py
```

Weights & Biases 연동으로 학습 메트릭 로깅 지원.

---

## 참고 자료

- [Smart Turn v3 블로그](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)
- [Smart Turn v3.2 블로그](https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/)
- [Smart Turn v2 블로그](https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/)
- [Smart Turn GitHub](https://github.com/pipecat-ai/smart-turn)
- [Smart Turn v3 HuggingFace](https://huggingface.co/pipecat-ai/smart-turn-v3)
- [Whisper Encoder 표현 연구](https://www.emergentmind.com/topics/whisper-based-encoder)
- [WhiSPA: Whisper의 의미적 정렬 연구](https://arxiv.org/html/2501.16344v1)
