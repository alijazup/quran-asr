from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
from faster_whisper import WhisperModel

class Predictor(BasePredictor):
    def setup(self):
        """Load Tarteel AI Quran Whisper model with CTranslate2 acceleration"""
        print("Loading Tarteel AI Quran Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel("tarteel-ai/whisper-base-ar-quran", device=device, compute_type=compute_type)
        print(f"Tarteel AI Quran Whisper model ready on {device} ({compute_type}).")

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
        # Convert to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Transcribe with Tarteel Quran model using Silero VAD and word-level timestamps
        segments_gen, info = self.model.transcribe(
            wav_path,
            language="ar",
            word_timestamps=True,
            vad_filter=True
        )

        words = []
        raw_segments = []

        for seg in segments_gen:
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
                "arabic_snippet": seg.text.strip()
            })

        print(f"Transcribed {len(words)} words across {len(raw_segments)} segments.")

        # Re-segment words by user preferences if words exist
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
            os.remove(wav_path)

        return json.dumps({"words": words, "segments": final_segments}, ensure_ascii=False)
