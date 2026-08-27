from cog import BasePredictor, Input, Path
import onnxruntime as ort
import sentencepiece as spm
import numpy as np
import soundfile as sf
import torch
import torchaudio
import json
import subprocess
import os
from huggingface_hub import hf_hub_download

class Predictor(BasePredictor):
    def setup(self):
        """Download and load ONNX FastConformer model and tokenizer once during startup"""
        print("Downloading ONNX model and tokenizer from HuggingFace...")
        self.model_path = hf_hub_download(
            repo_id="Muno459/fastconformer-quran",
            filename="onnx/model.onnx"
        )
        self.sp_path = hf_hub_download(
            repo_id="Muno459/fastconformer-quran",
            filename="tokenizer.model"
        )
        self.sp = spm.SentencePieceProcessor(model_file=self.sp_path)
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        print("ONNX FastConformer model ready.")

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

        # 2. Extract 80-channel log-mel spectrogram features
        waveform, sample_rate = torchaudio.load(wav_path)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80
        )
        mel = mel_transform(waveform)
        log_mel = torch.log(mel + 1e-5).numpy() # (1, 80, T)
        length = np.array([log_mel.shape[2]], dtype=np.int64)

        # 3. Run ONNX inference
        outputs = self.session.run(None, {
            "audio_signal": log_mel,
            "length": length
        })
        logprobs = outputs[0][0] # (T_out, 1025)

        # 4. CTC greedy argmax & frame timing mapping
        best_tokens = np.argmax(logprobs, axis=-1)
        frame_duration = 0.080 # FastConformer 8x downsampling on 10ms hop
        blank_id = 1024

        token_timestamps = []
        prev_token = blank_id
        
        for t_idx, token_id in enumerate(best_tokens):
            if token_id != blank_id and token_id != prev_token:
                time_sec = t_idx * frame_duration
                word = self.sp.decode([int(token_id)]).strip()
                if word:
                    token_timestamps.append({
                        "word": word,
                        "start": round(time_sec, 3),
                        "end": round(time_sec + frame_duration, 3)
                    })
            prev_token = token_id

        # 5. Group words into video subtitle segments based on natural breath pauses & length limits
        segments = []
        current_words = []

        for i, w in enumerate(token_timestamps):
            current_words.append(w)
            gap_to_next = (token_timestamps[i + 1]["start"] - w["end"]) if (i + 1 < len(token_timestamps)) else 999.0
            word_count = len(current_words)

            should_split = (
                (gap_to_next >= min_silence_gap and word_count >= 2) or
                (word_count >= max_words_per_segment) or
                (i == len(token_timestamps) - 1)
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
            "words": token_timestamps,
            "segments": segments
        }, ensure_ascii=False)
