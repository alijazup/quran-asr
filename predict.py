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
        """Load FastConformer Quran model with TDT compatibility patch and SentencePiece decoding"""
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = "/src/weights/quran_minshawi_final.nemo"
        if not os.path.exists(weights_path):
            weights_path = "weights/quran_minshawi_final.nemo"

        # Extract tokenizer directly from archive
        self.tok_path = "/src/weights/extracted_tok.model"
        if os.path.exists(weights_path) and not os.path.exists(self.tok_path):
            try:
                with tarfile.open(weights_path, "r:*") as tar:
                    for m in tar.getmembers():
                        if m.name.endswith(".model") or "tokenizer.model" in m.name:
                            f = tar.extractfile(m)
                            if f is not None:
                                with open(self.tok_path, "wb") as out_f:
                                    out_f.write(f.read())
                                print(f"Extracted SentencePiece tokenizer model: {m.name}", flush=True)
                                break
            except Exception as e:
                print("Tar extraction warning:", e, flush=True)

        self.sp_proc = None
        if os.path.exists(self.tok_path):
            try:
                self.sp_proc = spm.SentencePieceProcessor()
                self.sp_proc.Load(self.tok_path)
                print(f"SentencePieceProcessor loaded successfully with {len(self.sp_proc)} tokens.", flush=True)
            except Exception as e:
                print("SentencePiece load warning:", e, flush=True)

        print(f"Restoring FastConformer model on {device}...", flush=True)
        try:
            self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                restore_path=weights_path,
                map_location=device
            )
            self.model.change_decoding_strategy(decoder_type="ctc")
        except Exception:
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=weights_path,
                map_location=device
            )

        if self.sp_proc is not None and hasattr(self.model, "tokenizer"):
            try:
                self.model.tokenizer.tokenizer = self.sp_proc
            except Exception:
                pass

        self.model.eval()
        print("FastConformer Quran ASR model fully initialized and ready on GPU.", flush=True)

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

        # Step 1: Strictly convert input audio to 16kHz mono WAV
        wav_path = "/tmp/audio_16k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        data, samplerate = sf.read(wav_path)
        duration = len(data) / float(samplerate)

        # Step 2: Transcribe with FastConformer with return_hypotheses=True
        raw_res = self.model.transcribe(paths2audio_files=[wav_path], return_hypotheses=True)
        
        text = ""
        # Extract text from Hybrid tuple (RNNT, CTC) or List
        candidates = []
        if isinstance(raw_res, tuple):
            for part in raw_res:
                if isinstance(part, list):
                    candidates.extend(part)
                else:
                    candidates.append(part)
        elif isinstance(raw_res, list):
            candidates = raw_res
        else:
            candidates = [raw_res]

        for item in reversed(candidates):  # CTC is typically second
            if hasattr(item, "text") and item.text and str(item.text).strip():
                t = str(item.text).strip()
                if t and t not in ["['  ']", "['']", "''", ""]:
                    text = t
                    break
            elif isinstance(item, str) and item.strip():
                t = item.strip()
                if t and t not in ["['  ']", "['']", "''", ""]:
                    text = t
                    break
            elif hasattr(item, "y_sequence") and self.sp_proc is not None:
                ids = item.y_sequence
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                clean_ids = [int(x) for x in ids if int(x) > 0 and int(x) < len(self.sp_proc)]
                if clean_ids:
                    text = self.sp_proc.decode(clean_ids).strip()
                    if text:
                        break

        # Clean text
        text = text.replace("⁇", "").replace("?", "").replace("['", "").replace("']", "").strip()
        print(f"Decoded Arabic text: {text}", flush=True)

        raw_words = [w for w in text.split() if w and not w.startswith("[") and not w.startswith("Hypothesis")]

        words = []
        if raw_words:
            step = duration / max(1, len(raw_words))
            for i, w in enumerate(raw_words):
                words.append({
                    "word": w,
                    "start": round(i * step, 3),
                    "end": round((i + 1) * step, 3)
                })

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
                segments.append({
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                    "arabic_snippet": " ".join([item["word"] for item in current_words])
                })
                current_words = []

        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        return json.dumps({
            "raw_text": text,
            "words": words,
            "segments": segments,
            "status": "success"
        }, ensure_ascii=False)
