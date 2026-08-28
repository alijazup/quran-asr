from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
from faster_whisper import WhisperModel

class Predictor(BasePredictor):
    def setup(self):
        """Load Tarteel AI Quran Whisper model with CTranslate2 GPU acceleration and pre-baked weights"""
        print("Loading Tarteel AI Quran Whisper model on GPU...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        weights_path = "/src/weights"
        if not os.path.exists(weights_path):
            weights_path = "weights"

        self.model = WhisperModel(
            "tarteel-ai/whisper-base-ar-quran",
            device=device,
            compute_type=compute_type,
            download_root=weights_path if os.path.exists(weights_path) else None
        )
        print(f"Tarteel AI Quran Whisper model ready on {device} ({compute_type}).", flush=True)

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

        # Step 2: Transcribe with Tarteel Quran Whisper model using Silero VAD and word-level timestamps
        segments_gen, _ = self.model.transcribe(
            wav_path,
            language="ar",
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=250,
                speech_pad_ms=200
            ),
            temperature=0.0
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
