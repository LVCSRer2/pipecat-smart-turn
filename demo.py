"""
Smart Turn v3.1 실시간 마이크 데모
=================================
마이크로 음성을 입력받아 Silero VAD로 발화 구간을 감지하고,
Smart Turn 모델로 발화 완료(EOT) 여부를 실시간으로 판별합니다.

사용법:
    pip install -r requirements.txt
    python demo.py

첫 실행 시 Silero VAD ONNX 모델과 Smart Turn v3.1 ONNX 모델을
자동으로 다운로드합니다.
"""

import os
import sys
import time
import math
import urllib.request
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import onnxruntime as ort

# ── 설정 ──────────────────────────────────────────────────────────

RATE = 16000            # 샘플링 레이트 (16kHz)
CHUNK = 512             # VAD 청크 크기
CHANNELS = 1

VAD_THRESHOLD = 0.5     # Silero VAD 임계값
PRE_SPEECH_MS = 200     # 음성 시작 전 버퍼 (ms)
STOP_MS = 1000          # 침묵 지속 시 발화 종료 판정 (ms)
MAX_DURATION_SECONDS = 8  # 최대 발화 길이 (초)

DEBUG_SAVE_WAV = False  # True로 설정하면 각 발화를 WAV로 저장
TEMP_OUTPUT_DIR = "debug_wavs"

# 모델 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILERO_VAD_ONNX = os.path.join(SCRIPT_DIR, "silero_vad.onnx")
SMART_TURN_ONNX = os.path.join(SCRIPT_DIR, "smart-turn-v3.1-cpu.onnx")

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
MODEL_RESET_STATES_TIME = 5.0


# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    """Silero VAD ONNX 래퍼 (16kHz mono, chunk=512)."""

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.context_size = 64
        self._state = None
        self._context = None
        self._last_reset_time = time.time()
        self._init_states()

    def _init_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def maybe_reset(self):
        if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != CHUNK:
            raise ValueError(f"Expected {CHUNK} samples, got {x.shape[1]}")
        x = np.concatenate((self._context, x), axis=1)

        ort_inputs = {
            "input": x.astype(np.float32),
            "state": self._state,
            "sr": np.array(16000, dtype=np.int64),
        }
        out, self._state = self.session.run(None, ort_inputs)

        self._context = x[:, -self.context_size :]
        self.maybe_reset()
        return float(out[0][0])


# ── 모델 다운로드 ─────────────────────────────────────────────────

def ensure_silero_vad():
    """Silero VAD ONNX 모델이 없으면 다운로드."""
    if not os.path.exists(SILERO_VAD_ONNX):
        print("[다운로드] Silero VAD ONNX 모델...")
        urllib.request.urlretrieve(SILERO_VAD_URL, SILERO_VAD_ONNX)
        print("[완료] Silero VAD 다운로드 완료")
    return SILERO_VAD_ONNX


def ensure_smart_turn():
    """Smart Turn v3.1 ONNX 모델이 없으면 HuggingFace에서 다운로드."""
    if not os.path.exists(SMART_TURN_ONNX):
        print("[다운로드] Smart Turn v3.1 ONNX 모델 (HuggingFace)...")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="pipecat-ai/smart-turn-v3",
                filename="smart-turn-v3.1-cpu.onnx",
                local_dir=SCRIPT_DIR,
            )
            print("[완료] Smart Turn v3.1 다운로드 완료")
        except Exception as e:
            print(f"[오류] Smart Turn 모델 다운로드 실패: {e}")
            print("수동으로 다운로드하세요:")
            print("  https://huggingface.co/pipecat-ai/smart-turn-v3")
            sys.exit(1)
    return SMART_TURN_ONNX


# ── 발화 처리 ─────────────────────────────────────────────────────

_segment_count = 0


def process_segment(segment_audio_f32, predict_fn):
    """발화 구간을 Smart Turn으로 분석."""
    global _segment_count

    if segment_audio_f32.size == 0:
        return

    _segment_count += 1
    dur_sec = segment_audio_f32.size / RATE

    # 디버그 WAV 저장
    if DEBUG_SAVE_WAV:
        os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
        wav_path = os.path.join(TEMP_OUTPUT_DIR, f"segment_{_segment_count:03d}.wav")
        wavfile.write(wav_path, RATE, (segment_audio_f32 * 32767.0).astype(np.int16))

    # Smart Turn 추론
    t0 = time.perf_counter()
    result = predict_fn(segment_audio_f32)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    prediction = result["prediction"]
    probability = result["probability"]

    # 결과 출력
    label = "EOT (발화 완료)" if prediction == 1 else "CONT (발화 중)"
    bar = "█" * int(probability * 20) + "░" * (20 - int(probability * 20))

    print()
    print(f"  ┌─ 발화 #{_segment_count} ({dur_sec:.2f}초) ──────────────────")
    print(f"  │ 판정: {label}")
    print(f"  │ 확률: [{bar}] {probability:.4f}")
    print(f"  │ 추론: {dt_ms:.1f}ms")
    print(f"  └──────────────────────────────────────")
    print()


# ── 메인 루프 ─────────────────────────────────────────────────────

def main():
    print()
    print("=" * 50)
    print("  Smart Turn v3.1 실시간 마이크 데모")
    print("=" * 50)
    print()

    # 모델 준비
    print("[초기화] 모델 로딩 중...")
    ensure_silero_vad()
    ensure_smart_turn()

    from inference import predict_endpoint

    # VAD 초기화
    vad = SileroVAD(SILERO_VAD_ONNX)

    # 오디오 스트림 파라미터
    chunk_ms = (CHUNK / RATE) * 1000.0
    pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
    stop_chunks = math.ceil(STOP_MS / chunk_ms)
    max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

    # 상태 변수
    pre_buffer = deque(maxlen=pre_chunks)
    segment = []
    speech_active = False
    trailing_silence = 0
    since_trigger_chunks = 0

    # 마이크 열기 (sounddevice blocking stream)
    stream = sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK,
    )
    stream.start()

    print("[준비 완료] 마이크 입력 대기 중... (Ctrl+C로 종료)")
    print()
    print("  파이프라인: 마이크 → Silero VAD → Smart Turn v3.1 → EOT/CONT")
    print("  - 말을 하면 VAD가 음성을 감지합니다")
    print("  - 침묵이 1초 지속되면 Smart Turn이 발화 완료 여부를 판별합니다")
    print("  - EOT: 발화 완료 (AI가 응답할 타이밍)")
    print("  - CONT: 발화 중 (사용자가 아직 말하는 중)")
    print()

    try:
        while True:
            data, overflowed = stream.read(CHUNK)
            int16 = data[:, 0]  # mono
            f32 = int16.astype(np.float32) / 32768.0

            is_speech = vad.prob(f32) > VAD_THRESHOLD

            if not speech_active:
                pre_buffer.append(f32)
                if is_speech:
                    # 음성 시작 감지
                    segment = list(pre_buffer)
                    segment.append(f32)
                    speech_active = True
                    trailing_silence = 0
                    since_trigger_chunks = 1
                    print("  음성 감지...", end="", flush=True)
            else:
                segment.append(f32)
                since_trigger_chunks += 1

                if is_speech:
                    trailing_silence = 0
                else:
                    trailing_silence += 1

                # 침묵 지속 또는 최대 길이 도달 → 발화 종료 처리
                if trailing_silence >= stop_chunks or since_trigger_chunks >= max_chunks:
                    print(" 분석 중...")
                    stream.stop()

                    audio = np.concatenate(segment, dtype=np.float32)
                    process_segment(audio, predict_endpoint)

                    # 상태 초기화
                    segment.clear()
                    speech_active = False
                    trailing_silence = 0
                    since_trigger_chunks = 0
                    pre_buffer.clear()

                    stream.start()
                    print("  마이크 입력 대기 중...")

    except KeyboardInterrupt:
        print("\n\n[종료] 데모를 종료합니다.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
