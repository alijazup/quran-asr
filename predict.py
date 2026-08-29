import os
import subprocess
import json
import time
import torch
import numpy as np
import soundfile as sf
from transformers import (
    AutoModelForAudioFrameClassification,
    AutoFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from recitations_segmenter import segment_recitations, clean_speech_intervals
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load QuranCaption engine: obadx/recitation-segmenter-v2 + tarteel-ai/whisper-base-ar-quran"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. Load Quranic VAD Segmenter (obadx/recitation-segmenter-v2)
        segmenter_path = "/src/weights/segmenter"
        if not os.path.exists(segmenter_path):
            segmenter_path = "weights/segmenter"
        if not os.path.exists(segmenter_path):
            segmenter_path = "obadx/recitation-segmenter-v2"

        print(f"Loading Quran VAD Segmenter from {segmenter_path} on {self.device}...", flush=True)
        self.segmenter_model = AutoModelForAudioFrameClassification.from_pretrained(
            segmenter_path,
            local_files_only=os.path.exists(segmenter_path)
        ).to(self.device, dtype=self.dtype)
        self.segmenter_model.eval()

        self.segmenter_processor = AutoFeatureExtractor.from_pretrained(
            segmenter_path,
            local_files_only=os.path.exists(segmenter_path)
        )

        # 2. Load Tarteel AI Whisper Model (tarteel-ai/whisper-base-ar-quran)
        model_path = "/src/weights/model"
        if not os.path.exists(model_path):
            model_path = "weights/model"
        if not os.path.exists(model_path):
            model_path = "tarteel-ai/whisper-base-ar-quran"

        print(f"Loading Tarteel AI Whisper from {model_path} on {self.device}...", flush=True)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=os.path.exists(model_path)
        ).to(self.device)
        self.whisper_model.eval()

        self.whisper_processor = WhisperProcessor.from_pretrained(
            model_path,
            local_files_only=os.path.exists(model_path)
        )

        # Configure Whisper generation for Quranic Arabic
        forced_decoder_ids = None
        for language in ("arabic", "ar"):
            try:
                forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe"
                )
                if forced_decoder_ids:
                    break
            except Exception:
                continue

        self.gen_config = self.whisper_model.generation_config
        if forced_decoder_ids:
            self.gen_config.forced_decoder_ids = forced_decoder_ids
        if hasattr(self.gen_config, "language"):
            self.gen_config.language = "ar"
        if hasattr(self.gen_config, "task"):
            self.gen_config.task = "transcribe"

        print(f"QuranCaption VAD + Tarteel Whisper engine loaded successfully and ready on {self.device}.", flush=True)

    def predict(
        self,
        audio: Path = Input(description="Input audio file (WAV, MP3, MP4, MOV, etc.)"),
        min_silence_ms: int = Input(
            description="Minimum silence duration in milliseconds to split segments",
            default=200
        ),
        min_speech_ms: int = Input(
            description="Minimum speech duration in milliseconds",
            default=1000
        ),
        pad_ms: int = Input(
            description="Padding in milliseconds before and after each segment",
            default=50
        )
    ) -> str:
        # Step 1: Resample input audio to 16kHz mono WAV using ffmpeg
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read 16kHz audio with soundfile
        audio_data, sample_rate = sf.read(wav_path)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        total_duration = len(audio_data) / 16000.0
        print(f"Loaded audio: {total_duration:.2f}s", flush=True)

        # Step 2: Detect Quranic speech intervals via obadx/recitation-segmenter-v2
        t0 = time.time()
        audio_tensor = torch.from_numpy(audio_data).float()
        
        try:
            outputs = segment_recitations(
                [audio_tensor],
                self.segmenter_model,
                self.segmenter_processor,
                device=self.device,
                dtype=self.dtype,
                batch_size=1
            )
            clean_out = clean_speech_intervals(
                outputs[0].speech_intervals,
                outputs[0].is_complete,
                min_silence_duration_ms=min_silence_ms,
                min_speech_duration_ms=min_speech_ms,
                pad_duration_ms=pad_ms,
                return_seconds=True
            )
            intervals = clean_out.clean_speech_intervals.tolist()
        except Exception as e:
            print(f"VAD segmentation fallback warning: {e}", flush=True)
            intervals = [[0.0, total_duration]]

        print(f"VAD detected {len(intervals)} Quran recitation intervals in {time.time() - t0:.2f}s", flush=True)

        # Step 3: Transcribe each discrete phrase with Tarteel AI Whisper
        final_segments = []
        all_transcribed_texts = []

        for idx, (start_s, end_s) in enumerate(intervals):
            start_s = max(0.0, round(float(start_s), 3))
            end_s = min(total_duration, round(float(end_s), 3))
            if end_s <= start_s:
                continue

            start_idx = int(start_s * 16000)
            end_idx = int(end_s * 16000)
            chunk = audio_data[start_idx:end_idx]
            if len(chunk) < 1600:  # < 0.1s
                continue

            feats = self.whisper_processor(
                audio=chunk,
                sampling_rate=16000,
                return_tensors="pt"
            )["input_features"].to(device=self.device, dtype=self.dtype)

            with torch.no_grad():
                out_ids = self.whisper_model.generate(
                    feats,
                    generation_config=self.gen_config,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1
                )

            phrase_text = self.whisper_processor.batch_decode(
                out_ids,
                skip_special_tokens=True
            )[0].strip()

            if phrase_text:
                all_transcribed_texts.append(phrase_text)
                final_segments.append({
                    "start": start_s,
                    "end": end_s,
                    "arabic_snippet": phrase_text
                })

        raw_text = " ".join(all_transcribed_texts)
        print(f"Transcription complete: {len(final_segments)} segments generated. Text: {raw_text[:70]}...", flush=True)

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        return json.dumps({
            "raw_text": raw_text,
            "segments": final_segments,
            "status": "success"
        }, ensure_ascii=False)
