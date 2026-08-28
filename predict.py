from cog import BasePredictor, Input, Path
import subprocess
import os

class Predictor(BasePredictor):
    def setup(self):
        """Warm up environment"""
        print("FastConformer Quran ASR worker environment ready.")

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
