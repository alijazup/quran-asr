from cog import BasePredictor, Input, Path
import sherpa_onnx
import soundfile as sf
import json
import subprocess
import os
import zipfile
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self):
        """Download and load FastConformer Quran CTC ONNX model once during startup"""
        print("Downloading FastConformer Quran CTC ONNX from HuggingFace...")
        zip_path = hf_hub_download(
            repo_id="Na3Na33/fastconformer-quran-coreml",
            filename="FastConformerQuranCTC.onnx.zip"
        )
        extract_dir = "/tmp/fastconformer_ctc"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        model_path = os.path.join(extract_dir, "FastConformerQuranCTC.onnx")
        tokens_path = hf_hub_download(
            repo_id="mohammed/fastconformer-quran-ar-onnx-int8",
            filename="tokens.txt"
        )

        print("Initializing sherpa-onnx NeMo FastConformer CTC Recognizer on CPU...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model_path,
            tokens=tokens_path,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu"
        )
        print("FastConformer Quran CTC ASR ready.")

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
        wav_path = "/tmp/recitation_16k.wav"
        
        # 1. Convert to 16kHz mono 16-bit PCM WAV using ffmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Transcribe with sherpa-onnx OfflineRecognizer
        samples, sample_rate = sf.read(wav_path, dtype="float32")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=samples)
        self.recognizer.decode_stream(stream)
        result = stream.result

        # Extract words & timestamps
        words = []
        if hasattr(result, "tokens") and hasattr(result, "timestamps") and len(result.tokens) > 0 and len(result.timestamps) > 0:
            for word, start_time in zip(result.tokens, result.timestamps):
                w_clean = str(word).replace(" ", " ").strip()
                if w_clean:
                    words.append({
                        "word": w_clean,
                        "start": round(float(start_time), 3),
                        "end": round(float(start_time) + 0.35, 3)
                    })
        elif getattr(result, "text", None):
            raw_words = result.text.split()
            duration = len(samples) / 16000.0
            step = duration / max(1, len(raw_words))
            for i, w in enumerate(raw_words):
                words.append({
                    "word": w,
                    "start": round(i * step, 3),
                    "end": round((i + 1) * step, 3)
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

        if os.path.exists(wav_path):
            os.remove(wav_path)

        return json.dumps({
            "words": words,
            "segments": segments
        }, ensure_ascii=False)
