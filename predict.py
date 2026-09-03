import os
import subprocess
import json
import time
import soundfile as sf
from faster_whisper import WhisperModel
from cog import BasePredictor, Input, Path
from gap_repair import (
    build_repair_reel,
    find_voiced_word_gaps,
    map_reel_words_to_gaps,
    merge_words,
)


MODEL_REPOSITORY = "deepdml/faster-whisper-large-v3-turbo-ct2"

class Predictor(BasePredictor):
    def setup(self):
        """Load the pinned CTranslate2 Whisper weights used by this deployment."""
        print(f"Loading {MODEL_REPOSITORY} on GPU...", flush=True)
        model_path = "/src/weights/model"
        if not os.path.exists(model_path):
            model_path = "weights/model"
        if not os.path.exists(model_path):
            model_path = MODEL_REPOSITORY

        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16"
        )
        print(f"{MODEL_REPOSITORY} loaded successfully.", flush=True)

    def _transcribe(self, audio_source):
        segments_iter, _ = self.model.transcribe(
            audio_source,
            language="ar",
            task="transcribe",
            beam_size=5,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        words = []
        text_parts = []
        for segment in segments_iter:
            text = (segment.text or "").strip()
            if text:
                text_parts.append(text)
            for word in (segment.words or []):
                clean = (word.word or "").strip()
                if clean:
                    words.append(
                        {
                            "word": clean,
                            "start": round(float(word.start), 3),
                            "end": round(float(word.end), 3),
                        }
                    )
        return words, " ".join(text_parts)

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

        # Step 2: Run the normal full transcription once.
        initial_decode_started = time.perf_counter()
        initial_words, initial_raw_text = self._transcribe(wav_path)
        initial_decode_seconds = time.perf_counter() - initial_decode_started

        # Step 3: While the same GPU prediction is still alive, repair only long
        # timestamp gaps that contain sustained audio energy. This avoids a second
        # Replicate request/queue and does no extra decoding for clean inputs.
        detection_started = time.perf_counter()
        audio_samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        gap_candidates = find_voiced_word_gaps(audio_samples, sample_rate, initial_words)
        gap_detection_seconds = time.perf_counter() - detection_started
        recovered_words = []
        repaired_gaps = []
        repair_audio_seconds = 0.0
        repair_decoder_passes = 0
        repair_decode_seconds = 0.0

        if gap_candidates:
            repair_reel, mappings = build_repair_reel(
                audio_samples, sample_rate, gap_candidates
            )
            repair_audio_seconds = len(repair_reel) / float(sample_rate)
            repair_decoder_passes = 1
            repair_decode_started = time.perf_counter()
            reel_words, _ = self._transcribe(repair_reel)
            repair_decode_seconds = time.perf_counter() - repair_decode_started
            recovered_by_gap = map_reel_words_to_gaps(reel_words, mappings)

            for index, candidate in enumerate(gap_candidates):
                kept = recovered_by_gap.get(index, [])
                recovered_words.extend(kept)
                repaired_gaps.append(
                    {
                        **candidate,
                        "recovered_word_count": len(kept),
                        "recovered_text": " ".join(word["word"] for word in kept),
                    }
                )

        all_words = merge_words(initial_words, recovered_words)

        # Step 4: Build natural subtitle phrases from the complete word stream.
        final_segments = []

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

        raw_text = " ".join(word["word"] for word in all_words)
        print(
            f"Faster-Whisper transcribed {len(initial_words)} initial words, "
            f"recovered {len(recovered_words)} words from {len(gap_candidates)} voiced gaps, "
            f"and built {len(final_segments)} subtitle segments. Text: {raw_text[:60]}...",
            flush=True,
        )

        return json.dumps({
            "raw_text": raw_text,
            "words": all_words,
            "segments": final_segments,
            "repair": {
                "strategy": "same-prediction-voiced-gap-redecode",
                "model": MODEL_REPOSITORY,
                "initial_word_count": len(initial_words),
                "initial_raw_text": initial_raw_text,
                "candidate_gap_count": len(gap_candidates),
                "recovered_word_count": len(recovered_words),
                "repair_audio_seconds": round(repair_audio_seconds, 3),
                "repair_decoder_passes": repair_decoder_passes,
                "initial_decode_seconds": round(initial_decode_seconds, 3),
                "gap_detection_seconds": round(gap_detection_seconds, 3),
                "repair_decode_seconds": round(repair_decode_seconds, 3),
                "gaps": repaired_gaps,
            },
            "status": "success"
        }, ensure_ascii=False)
