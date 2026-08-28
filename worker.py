import sys
import json
import os
import subprocess

try:
    audio_path = sys.argv[1]
    min_silence_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    max_words_per_segment = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    wav_path = "/tmp/audio_16k.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    import soundfile as sf
    
    nemo_model_path = hf_hub_download(
        repo_id="NightPrince/stt-ar-fastconformer-quran-minshawi",
        filename="quran_minshawi_final.nemo"
    )

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=nemo_model_path, map_location="cpu")
    model.eval()

    hypotheses = model.transcribe(paths2audio_files=[wav_path], return_hypotheses=True)
    hyp = hypotheses[0]
    
    text = hyp.text if hasattr(hyp, "text") else str(hyp)
    raw_words = text.split()

    data, samplerate = sf.read(wav_path)
    duration = len(data) / float(samplerate)

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
        os.remove(wav_path)

    print(json.dumps({
        "words": words,
        "segments": segments
    }, ensure_ascii=False))

except Exception as e:
    import traceback
    sys.stderr.write(f"ERROR: {str(e)}\n{traceback.format_exc()}\n")
    sys.exit(1)
