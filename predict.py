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
        # Step 1: Resample input audio to 16kHz mono WAV using ffmpeg with acoustic bandpass filter
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1",
            "-af", "highpass=f=75,lowpass=f=7500",
            "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe using faster-whisper with Silero VAD filtering
        vad_params = {
            "min_silence_duration_ms": 200,
            "threshold": 0.35,
            "speech_pad_ms": 30,
        }

        segments_iter, info = self.model.transcribe(
            wav_path,
            language="ar",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            vad_parameters=vad_params,
            word_timestamps=True,
            initial_prompt="سورة آية بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
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
                    seg_words.append(w_dict)
            
            # Each VAD speech segment is a natural breath phrase.
            # If a single VAD segment exceeds max_words_per_segment, split it gracefully:
            if seg_words:
                if len(seg_words) <= max_words_per_segment:
                    final_segments.append({
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                        "arabic_snippet": " ".join([x["word"] for x in seg_words])
                    })
                else:
                    chunk = []
                    for w in seg_words:
                        chunk.append(w)
                        if len(chunk) >= max_words_per_segment:
                            final_segments.append({
                                "start": chunk[0]["start"],
                                "end": chunk[-1]["end"],
                                "arabic_snippet": " ".join([x["word"] for x in chunk])
                            })
                            chunk = []
                    if chunk:
                        final_segments.append({
                            "start": chunk[0]["start"],
                            "end": chunk[-1]["end"],
                            "arabic_snippet": " ".join([x["word"] for x in chunk])
                        })

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
