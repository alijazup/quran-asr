from cog import BasePredictor, Input, Path
import subprocess
import os
import json
import torch
import numpy as np
import tarfile
import sentencepiece as spm

class Predictor(BasePredictor):
    def setup(self):
        """Load FastConformer model and initialize direct PyTorch CTC decoder"""
        print("Applying NeMo TDT compatibility patch...", flush=True)
        try:
            import nemo.collections.asr.parts.utils.asr_confidence_utils as asr_confidence_utils
            orig_init = asr_confidence_utils.ConfidenceConfig.__init__
            def safe_init(self, *args, **kwargs):
                kwargs.pop('tdt_include_duration', None)
                orig_init(self, *args, **kwargs)
            asr_confidence_utils.ConfidenceConfig.__init__ = safe_init
            print("ConfidenceConfig patched successfully.", flush=True)
        except Exception as e:
            print("Warning on ConfidenceConfig:", e, flush=True)

        import nemo.collections.asr as nemo_asr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = "/src/weights/quran_minshawi_final.nemo"
        if not os.path.exists(weights_path):
            weights_path = "weights/quran_minshawi_final.nemo"

        # 1. Extract SentencePiece tokenizer model directly from archive
        self.tok_path = "/src/weights/extracted_tok.model"
        if os.path.exists(weights_path) and not os.path.exists(self.tok_path):
            try:
                with tarfile.open(weights_path, "r:*") as tar:
                    for m in tar.getmembers():
                        if m.name.endswith(".model") or "tokenizer" in m.name:
                            f = tar.extractfile(m)
                            if f is not None:
                                with open(self.tok_path, "wb") as out_f:
                                    out_f.write(f.read())
                                print(f"Extracted SentencePiece tokenizer model: {m.name}", flush=True)
                                break
            except Exception as e:
                print("Tar extraction warning:", e, flush=True)

        # 2. Load standalone SentencePiece processor
        self.sp_proc = spm.SentencePieceProcessor()
        self.sp_proc.Load(self.tok_path)
        print(f"SentencePieceProcessor loaded successfully with {len(self.sp_proc)} tokens.", flush=True)

        # 3. Restore NeMo model on device
        print(f"Restoring FastConformer model on {self.device}...", flush=True)
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            restore_path=weights_path,
            map_location=self.device
        )
        self.model.eval()
        print("FastConformer Quran ASR model and direct CTC head ready on GPU.", flush=True)

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
        import soundfile as sf

        # Convert to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        data, samplerate = sf.read(wav_path)
        total_duration = len(data) / float(samplerate)

        # Chunk audio into <= 25-second windows for optimal acoustic encoding
        chunk_duration = 25.0
        chunk_samples = int(chunk_duration * samplerate)
        all_words = []
        full_text_chunks = []
        num_chunks = max(1, int(np.ceil(len(data) / chunk_samples)))

        blank_id = len(self.sp_proc)

        for c_idx in range(num_chunks):
            c_start_sample = c_idx * chunk_samples
            c_end_sample = min((c_idx + 1) * chunk_samples, len(data))
            chunk_data = data[c_start_sample:c_end_sample]
            if len(chunk_data) == 0:
                continue

            chunk_start_time = c_start_sample / float(samplerate)
            chunk_dur = len(chunk_data) / float(samplerate)

            # Direct PyTorch Forward Pass through Preprocessor -> Conformer Encoder -> CTC Head
            audio_tensor = torch.tensor(chunk_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            audio_len = torch.tensor([len(chunk_data)], dtype=torch.long).to(self.device)

            with torch.no_grad():
                processed_signal, processed_signal_length = self.model.preprocessor(
                    input_signal=audio_tensor, length=audio_len
                )
                encoded, encoded_len = self.model.encoder(
                    audio_signal=processed_signal, length=processed_signal_length
                )
                log_probs = self.model.ctc_decoder(encoder_output=encoded)
                preds = torch.argmax(log_probs, dim=-1)[0].cpu().numpy()

            # CTC Greedy Collapse & Decode
            collapsed_ids = []
            prev = None
            for p in preds:
                p_int = int(p)
                if p_int != prev:
                    if p_int != blank_id and p_int != 0 and p_int < len(self.sp_proc):
                        collapsed_ids.append(p_int)
                    prev = p_int

            chunk_text = self.sp_proc.decode(collapsed_ids).strip()
            chunk_text = chunk_text.replace("⁇", "").replace("?", "").strip()
            print(f"Decoded Chunk {c_idx+1}/{num_chunks}: {chunk_text}", flush=True)

            if chunk_text:
                full_text_chunks.append(chunk_text)

            chunk_words = [w for w in chunk_text.split() if w]

            if chunk_words:
                step = chunk_dur / max(1, len(chunk_words))
                for w_i, w in enumerate(chunk_words):
                    all_words.append({
                        "word": w,
                        "start": round(chunk_start_time + w_i * step, 3),
                        "end": round(chunk_start_time + (w_i + 1) * step, 3)
                    })

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        # Build subtitle segments
        segments = []
        current_words = []
        for i, w in enumerate(all_words):
            current_words.append(w)
            gap_to_next = (all_words[i + 1]["start"] - w["end"]) if (i + 1 < len(all_words)) else 999.0
            word_count = len(current_words)
            should_split = (
                (gap_to_next >= min_silence_gap and word_count >= 2) or
                (word_count >= max_words_per_segment) or
                (i == len(all_words) - 1)
            )
            if should_split and current_words:
                segments.append({
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "arabic_snippet": " ".join([item["word"] for item in current_words])
                })
                current_words = []

        full_raw_text = " ".join(full_text_chunks)
        return json.dumps({
            "raw_text": full_raw_text,
            "words": all_words,
            "segments": segments,
            "status": "success"
        }, ensure_ascii=False)
