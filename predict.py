import os
import subprocess
import json
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load official Tarteel AI Quran Whisper model with native PyTorch GPU acceleration"""
        print("Loading official Tarteel AI model (tarteel-ai/whisper-base-ar-quran)...", flush=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_path = "/src/weights/model"
        if not os.path.exists(model_path):
            model_path = "weights/model"
        if not os.path.exists(model_path):
            model_path = "tarteel-ai/whisper-base-ar-quran"

        print(f"Loading model & processor from {model_path} on {device} ({torch_dtype})...", flush=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=os.path.exists(model_path)
        )
        try:
            gen_cfg_path = os.path.join(model_path, "generation_config.json")
            if os.path.exists(gen_cfg_path):
                model.generation_config = GenerationConfig.from_pretrained(model_path)
            else:
                model.generation_config = GenerationConfig.from_pretrained("openai/whisper-base")
        except Exception as e:
            print(f"GenerationConfig fallback warning: {e}", flush=True)

        model.to(device)

        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=os.path.exists(model_path)
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            stride_length_s=(4, 2),
            batch_size=1,
            return_timestamps="word",
            torch_dtype=torch_dtype,
            device=device,
        )
        print(f"Official Tarteel AI model loaded successfully and ready on {device}.", flush=True)

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4, MOV, etc.)"),
        min_silence_gap: float = Input(
            description="Minimum pause in seconds to split into a new subtitle segment",
            default=0.40
        ),
        max_words_per_segment: int = Input(
            description="Maximum words per subtitle segment",
            default=6
        )
    ) -> str:
        # Step 1: Resample input audio to 16kHz mono WAV with acoustic bandpass normalization
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-af", "highpass=f=75,lowpass=f=7500",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe with official Tarteel AI pipeline with beam search and Quranic initial prompt
        result = self.pipe(
            wav_path,
            return_timestamps="word",
            generate_kwargs={
                "language": "arabic",
                "task": "transcribe",
                "num_beams": 5,
                "do_sample": False,
                "initial_prompt": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
            }
        )

        raw_text = (result.get("text", "") or "").strip()
        chunks = result.get("chunks", []) or []

        words = []
        for chunk in chunks:
            w_text = (chunk.get("text", "") or "").strip()
            ts = chunk.get("timestamp", (0.0, 0.0))
            if w_text and ts and len(ts) == 2:
                start_val = ts[0] if ts[0] is not None else 0.0
                end_val = ts[1] if ts[1] is not None else start_val + 0.5
                words.append({
                    "word": w_text,
                    "start": round(start_val, 3),
                    "end": round(end_val, 3)
                })

        print(f"Tarteel AI transcribed {len(words)} words. Raw text: {raw_text[:60]}...", flush=True)

        # Step 3: Group words into subtitle-friendly chunks
        final_segments = []
        if words:
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
                    final_segments.append({
                        "start": round(current_words[0]["start"], 3),
                        "end": round(current_words[-1]["end"], 3),
                        "arabic_snippet": " ".join([item["word"] for item in current_words])
                    })
                    current_words = []
        elif raw_text:
            final_segments.append({
                "start": 0.0,
                "end": 30.0,
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
            "segments": final_segments,
            "status": "success"
        }, ensure_ascii=False)
