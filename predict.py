from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        """Load FastConformer model via native ASRModel.from_pretrained with TDT patch"""
        print("Applying NeMo TDT compatibility patch...")
        try:
            import nemo.collections.asr.parts.utils.asr_confidence_utils as asr_confidence_utils
            orig_init = asr_confidence_utils.ConfidenceConfig.__init__
            def safe_init(self, *args, **kwargs):
                kwargs.pop('tdt_include_duration', None)
                orig_init(self, *args, **kwargs)
            asr_confidence_utils.ConfidenceConfig.__init__ = safe_init
            print("ConfidenceConfig patched successfully.")
        except Exception as e:
            print("Warning: could not patch ConfidenceConfig:", e)

        import nemo.collections.asr as nemo_asr
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading NightPrince/stt-ar-fastconformer-quran-minshawi on {device}...")

        weights_path = "/src/weights/quran_minshawi_final.nemo"
        if os.path.exists(weights_path):
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=weights_path,
                map_location=device
            )
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                "NightPrince/stt-ar-fastconformer-quran-minshawi",
                map_location=device
            )
        self.model.eval()
        print("FastConformer model ready.")

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4,钩 etc.)"),
        min_silence_gap: float = Input(
            description="Minimum pause in seconds to split into a new subtitle segment",
            default=0.45
        ),
        max_words_per_segment: int = Input(
            description="Maximum words per subtitle segment",
            default=6
        )
    ) -> str:
        import soundfile as sf

        # Convert to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        data, samplerate = sf.read(wav_path)
        total_duration = len(data) / float(samplerate)

        # Chunk audio into <= 25-second windows to respect FastConformer positional window
        chunk_duration = 25.0
        chunk_samples = int(chunk_duration * samplerate)
        all_words = []
        num_chunks = max(1, int(np.ceil(len(data) / chunk_samples)))

        for c_idx in range(num_chunks):
            c_start_sample = c_idx * chunk_samples
            c_end_sample = min((c_idx + 1) * chunk_samples, len(data))
            chunk_data = data[c_start_sample:c_end_sample]
            if len(chunk_data) == 0:
                continue

            chunk_wav_path = f"/tmp/chunk_{c_idx}.wav"
            sf.write(chunk_wav_path, chunk_data, samplerate)

            chunk_start_time = c_start_sample / float(samplerate)
            chunk_dur = len(chunk_data) / float(samplerate)

            raw_res = self.model.transcribe(paths2audio_files=[chunk_wav_path])

            chunk_text = ""
            if isinstance(raw_res, (list, tuple)) and len(raw_res) > 0:
                item = raw_res[0]
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    chunk_text = str(getattr(item[0], 'text', item[0]))
                elif hasattr(item, 'text'):
                    chunk_text = str(item.text)
                else:
                    chunk_text = str(item)
            else:
                chunk_text = str(raw_res)

            print(f"Raw chunk {c_idx+1}/{num_chunks} text: {chunk_text}")

            chunk_words = [w for w in chunk_text.split() if w and not w.startswith("[") and not w.startswith("Hypothesis") and w != "⁇"]

            if chunk_words:
                step = chunk_dur / max(1, len(chunk_words))
                for w_i, w in enumerate(chunk_words):
                    all_words.append({
                        "word": w,
                        "start": round(chunk_start_time + w_i * step, 3),
                        "end": round(chunk_start_time + (w_i + 1) * step, 3)
                    })

            if os.path.exists(chunk_wav_path):
                os.remove(chunk_wav_path)

        # Build subtitle segments
        segments = []
        current_words = []
        for i, w in enumerate(all_words):
            current_words.append(w)
            gap_to_next = (all_words[i + 1]["start"] - w["end"]) if (i + 1 < len(all_words)) else 999.0
            word_count = len(current_words)
            should_split = (
                (gap_to_next >= min_silence_gap and word_count >= 2) or
                (word_count >= max_words_per_segment) or
                (i == len(all_words) - 1)
            )
            if should_split and current_words:
                segments.append({
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "arabic_snippet": " ".join([item["word"] for item in current_words])
                })
                current_words = []

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return json.dumps({"words": all_words, "segments": segments}, ensure_ascii=False)
