"""Pure helpers for detecting and merging speech omitted by the first ASR pass."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_seconds: float = 0.02):
    samples_per_frame = max(1, int(round(sample_rate * frame_seconds)))
    usable = (len(audio) // samples_per_frame) * samples_per_frame
    if usable == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), frame_seconds

    frames = audio[:usable].reshape(-1, samples_per_frame).astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    centers = (np.arange(len(rms), dtype=np.float32) + 0.5) * frame_seconds
    return rms, centers, frame_seconds


def _longest_true_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for active in mask:
        if active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def find_voiced_word_gaps(
    audio: np.ndarray,
    sample_rate: int,
    words: list[dict],
    *,
    min_gap_seconds: float = 1.00,
    min_voiced_seconds: float = 0.35,
) -> list[dict]:
    """Return internal word-timestamp gaps that contain sustained voice-like energy.

    The energy threshold is learned from regions the ASR already recognized as speech.
    This keeps the detector independent of a specific reciter, recording level, surah,
    or test file. It only nominates gaps; the same ASR model decides their text.
    """
    if len(words) < 2 or sample_rate <= 0 or len(audio) == 0:
        return []

    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)

    rms, centers, frame_seconds = _frame_rms(mono, sample_rate)
    if len(rms) == 0:
        return []

    known_speech = np.zeros(len(rms), dtype=bool)
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if end > start:
            known_speech |= (centers >= start) & (centers <= end)

    reference = rms[known_speech]
    reference = reference[reference > 1e-7]
    if len(reference) == 0:
        return []

    speech_rms = float(np.median(reference))
    global_floor = float(np.percentile(rms, 20))
    active_threshold = max(3e-4, speech_rms * 0.12, global_floor * 3.0)
    strong_threshold = max(active_threshold * 2.0, speech_rms * 0.25)

    candidates = []
    for previous, following in zip(words, words[1:]):
        gap_start = float(previous["end"])
        gap_end = float(following["start"])
        gap_duration = gap_end - gap_start
        if gap_duration < min_gap_seconds:
            continue

        in_gap = (centers >= gap_start) & (centers <= gap_end)
        gap_rms = rms[in_gap]
        if len(gap_rms) == 0:
            continue

        active = gap_rms >= active_threshold
        strong = gap_rms >= strong_threshold
        active_seconds = float(np.count_nonzero(active) * frame_seconds)
        strong_seconds = float(np.count_nonzero(strong) * frame_seconds)
        longest_active_seconds = float(_longest_true_run(active) * frame_seconds)
        active_ratio = float(np.mean(active))

        # Sustained energy plus at least a brief strong core filters hum and clicks.
        if (
            active_seconds >= min_voiced_seconds
            and strong_seconds >= 0.10
            and longest_active_seconds >= 0.12
            and active_ratio >= 0.08
        ):
            candidates.append(
                {
                    "start": round(gap_start, 3),
                    "end": round(gap_end, 3),
                    "duration": round(gap_duration, 3),
                    "active_seconds": round(active_seconds, 3),
                    "active_ratio": round(active_ratio, 3),
                }
            )

    return candidates


def keep_words_inside_gap(
    words: Iterable[dict], gap_start: float, gap_end: float, edge_guard: float = 0.10
) -> list[dict]:
    """Keep decoded words centered inside a gap, excluding context duplicates."""
    inner_start = gap_start + edge_guard
    inner_end = gap_end - edge_guard
    kept = []
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        if inner_start <= midpoint <= inner_end:
            kept.append(word)
    return kept


def build_repair_reel(
    audio: np.ndarray,
    sample_rate: int,
    gaps: Iterable[dict],
    *,
    context_seconds: float = 0.75,
    separator_seconds: float = 0.50,
):
    """Join all nominated gaps into one compact ASR input and retain time mappings."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)

    duration = len(mono) / float(sample_rate)
    separator = np.zeros(int(round(separator_seconds * sample_rate)), dtype=np.float32)
    pieces = []
    mappings = []
    reel_cursor = 0.0

    for index, gap in enumerate(gaps):
        if index:
            pieces.append(separator)
            reel_cursor += separator_seconds

        gap_start = float(gap["start"])
        gap_end = float(gap["end"])
        source_start = max(0.0, gap_start - context_seconds)
        source_end = min(duration, gap_end + context_seconds)
        first_sample = int(round(source_start * sample_rate))
        last_sample = int(round(source_end * sample_rate))
        clip = mono[first_sample:last_sample]
        clip_duration = len(clip) / float(sample_rate)
        pieces.append(clip)
        mappings.append(
            {
                "gap_index": index,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "source_start": source_start,
                "reel_start": reel_cursor,
                "reel_end": reel_cursor + clip_duration,
            }
        )
        reel_cursor += clip_duration

    reel = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
    return reel, mappings


def map_reel_words_to_gaps(words: Iterable[dict], mappings: Iterable[dict]):
    """Map one repair-reel decode back to original timestamps and gap indexes."""
    recovered_by_gap = {}
    mapping_list = list(mappings)
    for word in words:
        midpoint = (float(word["start"]) + float(word["end"])) / 2.0
        mapping = next(
            (
                item
                for item in mapping_list
                if float(item["reel_start"]) <= midpoint <= float(item["reel_end"])
            ),
            None,
        )
        if mapping is None:
            continue

        offset = float(mapping["source_start"]) - float(mapping["reel_start"])
        global_word = {
            "word": word["word"],
            "start": round(float(word["start"]) + offset, 3),
            "end": round(float(word["end"]) + offset, 3),
        }
        kept = keep_words_inside_gap(
            [global_word], float(mapping["gap_start"]), float(mapping["gap_end"])
        )
        if kept:
            recovered_by_gap.setdefault(int(mapping["gap_index"]), []).extend(kept)
    return recovered_by_gap


def merge_words(original: Iterable[dict], recovered: Iterable[dict]) -> list[dict]:
    """Merge timestamped words in stable chronological order."""
    combined = [dict(word) for word in original]
    combined.extend(dict(word) for word in recovered)
    combined.sort(key=lambda word: (float(word["start"]), float(word["end"])))
    return combined
