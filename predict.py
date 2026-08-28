from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch

MODEL_PATH = "/root/model/quran_minshawi_final.nemo"

class Predictor(BasePredictor):
    def setup(self):
        """Load pre-baked FastConformer model instantly from local disk"""
        print("Loading FastConformer Quran ASR model from local image...")
        import nemo.collections.asr as nemo_asr
        
        path = MODEL_PATH if os.path.exists(MODEL_PATH) else "quran_minshawi_final.nemo"
        if not os.path.exists(path):
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="NightPrince/stt-ar-fastconformer-quran-minshawi",
                filename="quran_minshawi_final.nemo"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading checkpoint on {device} from {path}...")

        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=path,
            map_location=device
        )
        self.model.eval()
        print("FastConformer model ready.")

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4, M4A, etc.)"),
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

        # Transcribe using in-memory model
        hypotheses = self.model.transcribe(paths2audio_files=[wav_path], return_hypotheses=True)
        hyp = hypotheses[0]
        text = hyp.text if hasattr(hyp, "text") else str(hyp)
        raw_words = text.split()

        # Duration & timestamps
        data, samplerate = sf.read(wav_path)
        duration = len(data) / float(samplerate)

        words = []
        if raw_words:
            step = duration / max(1, len(raw_words))
            for i, w in enumerate(raw_words):
                words.append({
                    "word": w,
                    "start": round(i * step, 3),
                    "end": round((i + 1) * step, 3)
                })

        segments = []
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
                segments.append({
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "arabic_snippet": " ".join([item["word"] for item in current_words])
                })
                current_words = []

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return json.dumps({"words": words, "segments": segments}, ensure_ascii=False)
