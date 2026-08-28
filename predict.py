from cog import BasePredictor, Input, Path
import nemo.collections.asr as nemo_asr
import torch
import json
import subprocess
import os
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self):
        """Download and load FastConformer Quran ASR model once during startup"""
        print("Downloading Quran FastConformer checkpoint from HuggingFace...")
        model_path = hf_hub_download(
            repo_id="mohammed/fastconformer-quran-ar",
            filename="phase3_full_finetune/phase3_full_finetune_wer0.0852.nemo"
        )
        print("Restoring NeMo FastConformer model...")
        self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        print("FastConformer Quran ASR ready.")

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
        """Transcribe recitation audio and return word timestamps and formatted subtitle segments"""
        wav_path = "/tmp/recitation_16k.wav"
        
        # 1. Convert to 16kHz mono 16-bit PCM WAV using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Transcribe with word-level timestamps
        with torch.no_grad():
            hypotheses = self.model.transcribe([wav_path], return_hypotheses=True)

        if not hypotheses:
            return json.dumps({"words": [], "segments": []})

        hyp = hypotheses[0]
        words = []
        if hasattr(hyp, "timestep") and hyp.timestep:
            for token in hyp.timestep:
                token_word = getattr(token, "word", str(token)).strip()
                if token_word:
                    words.append({
                        "word": token_word,
                        "start": round(float(token.start), 3),
                        "end": round(float(token.end), 3)
                    })

        # 3. Group words into video subtitle segments based on natural breath pauses & length limits
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
                snippet = " ".join([item["word"] for item in current_words])
                segments.append({
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "arabic_snippet": snippet
                })
                current_words = []

        # Cleanup temporary audio file
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return json.dumps({
            "words": words,
            "segments": segments
        }, ensure_ascii=False)
