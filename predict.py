from cog import BasePredictor, Input, Path
import subprocess
import os

class Predictor(BasePredictor):
    def setup(self):
        """Warm up environment and pre-download model checkpoint"""
        print("Pre-downloading FastConformer Quran checkpoint...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="NightPrince/stt-ar-fastconformer-quran-minshawi",
            filename="quran_minshawi_final.nemo"
        )
        print("FastConformer environment ready.")

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
        cmd = [
            "python3", "worker.py",
            str(audio),
            str(min_silence_gap),
            str(max_words_per_segment)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("WORKER ERROR:\n", res.stderr)
            raise RuntimeError(f"FastConformer worker failed: {res.stderr}")
        return res.stdout.strip()
