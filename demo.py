"""
Smart Turn v3.2 실시간 마이크 데모 (분석 모드 옵션 추가)
======================================================
--mode [single|retain] 옵션을 통해 분석 방식을 선택할 수 있습니다.
- single: 각 발화 구간을 독립적으로 분석 (기존 방식)
- retain: CONT 판정 시 오디오를 누적하여 다음 발화와 함께 분석 (링 버퍼 방식)
"""

import os
import sys
import time
import math
import argparse
import urllib.request
from collections import deque
import queue

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 설정 ──────────────────────────────────────────────────────────

RATE = 16000
CHUNK = 512
CHANNELS = 1

VAD_THRESHOLD = 0.5
PRE_SPEECH_MS = 200
STOP_MS = 1000
MAX_DURATION_SECONDS = 8
MAX_TOTAL_BUFFER_SECONDS = 15 # retain 모드에서 누적할 최대 시간

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILERO_VAD_ONNX = os.path.join(SCRIPT_DIR, "silero_vad.onnx")
SMART_TURN_ONNX = os.path.join(SCRIPT_DIR, "smart-turn-v3.2-cpu.onnx")
SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# ── Silero VAD ────────────────────────────────────────────────────

class SileroVAD:
    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.context_size = 64
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)

    def prob(self, chunk_f32: np.ndarray) -> float:
        x = np.reshape(chunk_f32, (1, -1))
        x = np.concatenate((self._context, x), axis=1)
        ort_inputs = {"input": x.astype(np.float32), "state": self._state, "sr": np.array(16000, dtype=np.int64)}
        out, self._state = self.session.run(None, ort_inputs)
        self._context = x[:, -self.context_size :]
        return float(out[0][0])

# ── 모델 다운로드 ─────────────────────────────────────────────────

def ensure_models():
    if not os.path.exists(SILERO_VAD_ONNX):
        print("[다운로드] Silero VAD ONNX...")
        urllib.request.urlretrieve(SILERO_VAD_URL, SILERO_VAD_ONNX)
    if not os.path.exists(SMART_TURN_ONNX):
        print("[다운로드] Smart Turn v3.2 ONNX...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="pipecat-ai/smart-turn-v3", filename="smart-turn-v3.2-cpu.onnx", local_dir=SCRIPT_DIR)

# ── 시각화 클래스 ──────────────────────────────────────────────────

class PNGPlotter:
    def __init__(self, mode_name, max_samples=200):
        self.max_samples = max_samples
        self.mode_name = mode_name
        self.vad_probs = deque([0.0] * max_samples, maxlen=max_samples)
        self.smart_turn_results = []
        self.output_path = "realtime_monitor.png"

    def update_vad(self, prob):
        self.vad_probs.append(prob)
        new_results = []
        for res in self.smart_turn_results:
            res[0] -= 1
            if res[0] >= 0: new_results.append(res)
        self.smart_turn_results = new_results

    def add_smart_turn(self, prob, is_eot):
        self.smart_turn_results.append([self.max_samples - 1, prob, is_eot])

    def clear_results(self):
        self.smart_turn_results = []

    def save(self, status_text="", info_text=""):
        plt.figure(figsize=(10, 5))
        plt.plot(self.vad_probs, label="VAD Probability", color='blue', alpha=0.6, linewidth=2)
        
        eot_x, eot_y = [], []
        cont_x, cont_y = [], []
        for x, p, eot in self.smart_turn_results:
            if eot: eot_x.append(x); eot_y.append(p)
            else: cont_x.append(x); cont_y.append(p)
        
        if eot_x: plt.scatter(eot_x, eot_y, color='red', marker='x', s=150, linewidth=3, label="EOT (End)", zorder=5)
        if cont_x: plt.scatter(cont_x, cont_y, color='green', marker='o', s=80, label="CONT (Continue)", zorder=5)

        plt.ylim(-0.05, 1.05)
        plt.xlim(0, self.max_samples)
        plt.axhline(y=VAD_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
        
        title_main = f"Smart Turn v3.2 Monitor [{self.mode_name.upper()}]"
        title_sub = f" | {status_text}" if status_text else ""
        plt.title(f"{title_main}{title_sub} ({time.strftime('%H:%M:%S')})")
        
        if info_text:
            plt.text(self.max_samples - 5, 0.05, info_text, ha='right', fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()

# ── 메인 로직 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart Turn v3.2 Demo with Retain Mode Option")
    parser.add_argument("--mode", type=str, choices=["single", "retain"], default="retain",
                        help="Analysis mode: 'single' (no history) or 'retain' (keep history if CONT)")
    args = parser.parse_args()

    ensure_models()
    from inference import predict_endpoint
    vad = SileroVAD(SILERO_VAD_ONNX)
    
    audio_queue = queue.Queue()
    plotter = PNGPlotter(mode_name=args.mode)

    def audio_callback(indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    chunk_ms = (CHUNK / RATE) * 1000.0
    pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
    stop_chunks = math.ceil(STOP_MS / chunk_ms)
    max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

    pre_buffer = deque(maxlen=pre_chunks)
    current_segment = []
    speech_active = False
    trailing_silence = 0
    since_trigger_chunks = 0
    
    continued_audio_accum = np.array([], dtype=np.float32)
    
    frame_count = 0
    status_msg = "Ready"
    info_msg = ""

    print(f"\n[설정] 분석 모드: {args.mode.upper()}")
    print("마이크 입력을 시작합니다... (Ctrl+C로 종료)")

    stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK, callback=audio_callback)
    
    with stream:
        try:
            while True:
                while not audio_queue.empty():
                    data = audio_queue.get()
                    int16 = data[:, 0]
                    f32 = int16.astype(np.float32) / 32768.0
                    
                    v_prob = vad.prob(f32)
                    plotter.update_vad(v_prob)
                    is_speech = v_prob > VAD_THRESHOLD

                    if not speech_active:
                        pre_buffer.append(f32)
                        if is_speech:
                            plotter.clear_results()
                            current_segment = list(pre_buffer)
                            current_segment.append(f32)
                            speech_active = True
                            trailing_silence = 0
                            since_trigger_chunks = 1
                            status_msg = "Speech Detected"
                            print(f"\n[{status_msg}]", end="", flush=True)
                    else:
                        current_segment.append(f32)
                        since_trigger_chunks += 1
                        if is_speech: trailing_silence = 0
                        else: trailing_silence += 1

                        if trailing_silence >= stop_chunks or since_trigger_chunks >= max_chunks:
                            print(" Analyzing...", end="")
                            this_audio = np.concatenate(current_segment)
                            
                            # 오디오 병합 로직 (retain 모드일 때만 적용)
                            if args.mode == "retain" and continued_audio_accum.size > 0:
                                input_audio = np.concatenate([continued_audio_accum, this_audio])
                            else:
                                input_audio = this_audio
                            
                            info_msg = f"Audio: {input_audio.size/RATE:.1f}s"
                            
                            result = predict_endpoint(input_audio)
                            prob = result["probability"]
                            is_eot = result["prediction"] == 1
                            
                            plotter.add_smart_turn(prob, is_eot)
                            
                            if is_eot:
                                label = "EOT (End)"
                                continued_audio_accum = np.array([], dtype=np.float32)
                                status_msg = "Finished"
                            else:
                                label = "CONT (Continued)"
                                if args.mode == "retain":
                                    continued_audio_accum = input_audio
                                    max_samples = MAX_TOTAL_BUFFER_SECONDS * RATE
                                    if continued_audio_accum.size > max_samples:
                                        continued_audio_accum = continued_audio_accum[-max_samples:]
                                status_msg = "Continuing..."

                            print(f" Result: {label} ({prob:.2f}) | {info_msg}")
                            plotter.save(status_msg, info_msg)
                            
                            current_segment = []
                            speech_active = False
                            trailing_silence = 0
                            since_trigger_chunks = 0
                            pre_buffer.clear()
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        plotter.save(status_msg, info_msg)
                
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n데모를 종료합니다.")

if __name__ == "__main__":
    main()
