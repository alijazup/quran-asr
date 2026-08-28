import os

# Guarantee CUDA 12 and cuDNN dynamic libraries are loaded into LD_LIBRARY_PATH
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    cublas_dir = os.path.dirname(nvidia.cublas.lib.__file__)
    cudnn_dir = os.path.dirname(nvidia.cudnn.lib.__file__)
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{cublas_dir}:{cudnn_dir}:{current_ld}"
    import ctypes
    for f in os.listdir(cublas_dir):
        if f.endswith(".so.12") or f.endswith(".so"):
            try:
                ctypes.CDLL(os.path.join(cublas_dir, f))
            except Exception:
                pass
    for f in os.listdir(cudnn_dir):
        if f.endswith(".so.9") or f.endswith(".so.8") or f.endswith(".so"):
            try:
                ctypes.CDLL(os.path.join(cudnn_dir, f))
            except Exception:
                pass
except Exception:
    pass

from cog import BasePredictor, Input, Path
import subprocess
import json
import torch
from faster_whisper import WhisperModel

class Predictor(BasePredictor):
    def setup(self):
        """Load pre-baked faster-whisper large-v3-turbo model from local disk with zero network requests"""
        print("Loading faster-whisper large-v3-turbo from local disk on GPU...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model_path = "/src/weights/model"
        if not os.path.exists(model_path):
            model_path = "weights/model"
        if not os.path.exists(model_path):
            model_path = "large-v3-turbo"

        is_local = os.path.exists(model_path) and os.path.isdir(model_path)
        print(f"Loading WhisperModel from {'LOCAL DISK ' + model_path if is_local else model_path} on {device} ({compute_type})...", flush=True)

        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            local_files_only=is_local
        )
        print(f"Faster-Whisper Large-V3-Turbo loaded successfully and ready on {device}.", flush=True)

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4, etc.)"),
        min_silence_gap: float = Input(
            description="Minimum pause in seconds to split into a new subtitle segment",
            default=0.45
        ),
        max_words_per_segment: int = Input(
            description="Maximum words per subtitle segment",
            default=6
        )
    ) -> str:
        # Step 1: Resample input audio to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe with Faster-Whisper Large-V3-Turbo using Silero VAD and word-level timestamps
        segments_gen, _ = self.model.transcribe(
            wav_path,
            language="ar",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=250,
                speech_pad_ms=200
            ),
            temperature=0.0,
            initial_prompt="بسم الله الرحمن الرحيم. سورة من القرآن الكريم."
        )

        words = []
        raw_segments = []
        full_text_list = []

        for seg in segments_gen:
            seg_text = seg.text.strip()
            if seg_text:
                full_text_list.append(seg_text)
            if seg.words:
                for w in seg.words:
                    clean_w = w.word.strip()
                    if clean_w:
                        words.append({
                            "word": clean_w,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3)
                        })
            raw_segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "arabic_snippet": seg_text
            })

        print(f"Transcribed {len(words)} words across {len(raw_segments)} raw segments.", flush=True)

        # Step 3: Re-segment words into subtitle-friendly chunks
        final_segments = []
        if words:
            current_words = []
            for i, w in enumerate(words):
                current_words.append(w)
                gap_to_next = (words[i + 1]["start"] - w["end"]) if (i + 1 < len(words)) else 999.0
                word_count = len(current_words)
                should_split = (
                    (gap_to_next >= min_silence_gap and word_count >= 2) or
                    (word_count >= max_words_per_segment) or
                    (i == len(words) - 1)
                )
                if should_split and current_words:
                    final_segments.append({
                        "start": round(current_words[0]["start"], 3),
                        "end": round(current_words[-1]["end"], 3),
                        "arabic_snippet": " ".join([item["word"] for item in current_words])
                    })
                    current_words = []
        else:
            final_segments = raw_segments

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        full_raw_text = " ".join(full_text_list)
        return json.dumps({
            "raw_text": full_raw_text,
            "words": words,
            "segments": final_segments,
            "status": "success"
        }, ensure_ascii=False)
