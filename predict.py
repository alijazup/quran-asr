from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

class Predictor(BasePredictor):
    def setup(self):
        """Load Tarteel AI Quran Whisper model via standard HuggingFace pipeline"""
        print("Loading Tarteel AI Quran Whisper model with PyTorch GPU...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "tarteel-ai/whisper-base-ar-quran"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        print(f"Tarteel AI Quran model successfully loaded on {device}.")

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
        # Step 1: Convert input audio to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe with Tarteel Quran model
        result = self.pipe(
            wav_path,
            generate_kwargs={
                "language": "arabic",
                "task": "transcribe"
            }
        )

        raw_chunks = result.get("chunks", [])
        raw_text = (result.get("text") or "").strip()

        words = []
        segments = []

        for chunk in raw_chunks:
            text = (chunk.get("text") or "").strip()
            ts = chunk.get("timestamp")
            if not text or not ts or len(ts) < 2 or ts[0] is None or ts[1] is None:
                continue

            c_start = float(ts[0])
            c_end = float(ts[1])
            chunk_words = text.split()
            if not chunk_words:
                continue

            step = (c_end - c_start) / len(chunk_words)
            for idx, w in enumerate(chunk_words):
                w_start = round(c_start + idx * step, 3)
                w_end = round(c_start + (idx + 1) * step, 3)
                words.append({
                    "word": w,
                    "start": w_start,
                    "end": w_end
                })

            for w_idx in range(0, len(chunk_words), max_words_per_segment):
                sub_slice = chunk_words[w_idx:w_idx + max_words_per_segment]
                sub_start = round(c_start + w_idx * step, 3)
                sub_end = round(c_start + (w_idx + len(sub_slice)) * step, 3)
                segments.append({
                    "start": sub_start,
                    "end": sub_end,
                    "arabic_snippet": " ".join(sub_slice)
                })

        if not segments and raw_text:
            segments.append({
                "start": 0.0,
                "end": 5.0,
                "arabic_snippet": raw_text
            })

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        return json.dumps({
            "raw_text": raw_text,
            "words": words,
            "segments": segments,
            "status": "success"
        }, ensure_ascii=False)
