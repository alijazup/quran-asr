from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
import nemo.collections.asr as nemo_asr

class Predictor(BasePredictor):
    def setup(self):
        """Load pre-baked NVIDIA FastConformer-Quran ASR model (WER 0.0038)"""
        print("Starting FastConformer-Quran setup...", flush=True)
        model_path = "/src/weights/fastconformer-quran.nemo"
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            print("Downloading model from Hugging Face...", flush=True)
            model_path = hf_hub_download(
                repo_id="mohammed/fastconformer-quran-ar",
                filename="phase1_top3/phase1_top3_wer0.0038.nemo"
            )
        print(f"Loading NeMo model from {model_path}...", flush=True)
        
        try:
            self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=model_path)
            self.model.change_decoding_strategy(decoder_type="ctc")
        except Exception as e:
            print(f"Hybrid restore fallback: {e}", flush=True)
            self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=model_path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on CUDA GPU.", flush=True)
        else:
            print("Model loaded on CPU.", flush=True)
            
        self.model.eval()
        print("FastConformer-Quran model setup complete and ready.", flush=True)

    def predict(
        self,
        audio: Path = Input(description="Input Quran recitation audio file"),
        min_silence_gap: float = Input(
            description="Minimum pause in seconds to split into a new subtitle segment",
            default=0.35
        ),
        max_words_per_segment: int = Input(
            description="Maximum words per subtitle segment",
            default=6
        )
    ) -> str:
        print(f"Received audio file: {audio}", flush=True)
        
        # Step 1: Strictly convert audio to 16kHz mono WAV (fixes ??? bug)
        clean_wav = "/tmp/quran_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", clean_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 2: Transcribe with FastConformer-Quran
        with torch.no_grad():
            hypotheses = self.model.transcribe([clean_wav], batch_size=1)

        if os.path.exists(clean_wav):
            try:
                os.remove(clean_wav)
            except Exception:
                pass

        if not hypotheses or len(hypotheses) == 0:
            return json.dumps({"raw_text": "", "segments": []}, ensure_ascii=False)

        raw_hypothesis = hypotheses[0]
        if hasattr(raw_hypothesis, 'text'):
            arabic_text = raw_hypothesis.text
        else:
            arabic_text = str(raw_hypothesis)

        arabic_text = (arabic_text or "").strip()
        words_list = arabic_text.split()
        print(f"Transcribed: {arabic_text[:80]}... ({len(words_list)} words)", flush=True)

        # Step 3: Format subtitle segments
        segments = []
        for i in range(0, len(words_list), max_words_per_segment):
            chunk = words_list[i:i + max_words_per_segment]
            segments.append({
                "arabic_snippet": " ".join(chunk)
            })

        return json.dumps({
            "raw_text": arabic_text,
            "segments": segments,
            "status": "success"
        }, ensure_ascii=False)
