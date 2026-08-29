import os
import subprocess
import json
from faster_whisper import WhisperModel
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load official Tarteel AI Quran Whisper model with native CTranslate2 GPU acceleration & Silero VAD"""
        print("Loading official Tarteel AI model with faster-whisper on GPU...", flush=True)
        model_path = "/src/weights/model"
        if not os.path.exists(model_path):
            model_path = "weights/model"
        if not os.path.exists(model_path):
            model_path = "tarteel-ai/whisper-base-ar-quran"

        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16"
        )
        print("Official Tarteel AI faster-whisper model loaded successfully.", flush=True)

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4, MOV, etc.)"),
        min_silence_gap: float = Input(
            description="Minimum pause in seconds to split into a new subtitle segment",
            default=0.35
        ),
        max_words_per_segment: int = Input(
            description="Maximum words per subtitle segment",
            default=6
        )
    ) -> str:
        # Step 1: Resample input audio to clean 16kHz mono WAV using ffmpeg
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe using faster-whisper without VAD clipping
        segments_iter, info = self.model.transcribe(
            wav_path,
            language="ar",
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        final_segments = []
        all_words = []
        raw_text_parts = []

        for seg in segments_iter:
            seg_text = (seg.text or "").strip()
            if seg_text:
                raw_text_parts.append(seg_text)
            
            seg_words = []
            for w in (seg.words or []):
                w_clean = (w.word or "").strip()
                if w_clean:
                    w_dict = {
                        "word": w_clean,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    }
                    all_words.append(w_dict)

        # Build natural subtitle phrases from transcribed words based on acoustic silence gaps
        current_words = []
        for i, w in enumerate(all_words):
            current_words.append(w)
            gap_to_next = (all_words[i + 1]["start"] - w["end"]) if (i + 1 < len(all_words)) else 999.0
            count = len(current_words)
            
            # Split when:
            # 1) Natural breath pause (gap >= min_silence_gap) and we have at least 2 words (or single word before long pause >= 0.8s)
            # 2) Or phrase reached max_words_per_segment (6 words)
            # 3) Or last word of the recitation
            should_split = (
                (gap_to_next >= min_silence_gap and (count >= 2 or gap_to_next >= 0.8)) or
                (count >= max_words_per_segment) or
                (i == len(all_words) - 1)
            )
            if should_split and current_words:
                final_segments.append({
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "arabic_snippet": " ".join([x["word"] for x in current_words])
                })
                current_words = []

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        raw_text = " ".join(raw_text_parts)
        print(f"Faster-Whisper transcribed {len(all_words)} words in {len(final_segments)} VAD segments. Text: {raw_text[:60]}...", flush=True)

        return json.dumps({
            "raw_text": raw_text,
            "words": all_words,
            "segments": final_segments,
            "status": "success"
        }, ensure_ascii=False)
