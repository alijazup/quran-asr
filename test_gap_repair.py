import unittest

import numpy as np

from gap_repair import (
    build_repair_reel,
    find_voiced_word_gaps,
    keep_words_inside_gap,
    map_reel_words_to_gaps,
    merge_words,
)


SAMPLE_RATE = 16_000


def tone(audio, start, end, amplitude=0.2, frequency=220.0):
    first = int(start * SAMPLE_RATE)
    last = int(end * SAMPLE_RATE)
    times = np.arange(last - first, dtype=np.float32) / SAMPLE_RATE
    audio[first:last] = amplitude * np.sin(2 * np.pi * frequency * times)


class GapRepairTests(unittest.TestCase):
    def setUp(self):
        self.words = [
            {"word": "a", "start": 0.2, "end": 1.0},
            {"word": "b", "start": 4.0, "end": 4.8},
        ]

    def test_detects_sustained_voice_inside_long_word_gap(self):
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        tone(audio, 0.2, 1.0)
        tone(audio, 1.7, 3.3)
        tone(audio, 4.0, 4.8)

        gaps = find_voiced_word_gaps(audio, SAMPLE_RATE, self.words)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["start"], 1.0)
        self.assertEqual(gaps[0]["end"], 4.0)

    def test_ignores_true_silence_inside_long_word_gap(self):
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        tone(audio, 0.2, 1.0)
        tone(audio, 4.0, 4.8)

        self.assertEqual(find_voiced_word_gaps(audio, SAMPLE_RATE, self.words), [])

    def test_ignores_short_click_in_long_gap(self):
        audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
        tone(audio, 0.2, 1.0)
        tone(audio, 2.4, 2.46)
        tone(audio, 4.0, 4.8)

        self.assertEqual(find_voiced_word_gaps(audio, SAMPLE_RATE, self.words), [])

    def test_ignores_short_gap_even_when_it_contains_energy(self):
        words = [
            {"word": "a", "start": 0.2, "end": 1.0},
            {"word": "b", "start": 1.8, "end": 2.5},
        ]
        audio = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
        tone(audio, 0.2, 2.5)

        self.assertEqual(find_voiced_word_gaps(audio, SAMPLE_RATE, words), [])

    def test_context_words_are_not_kept_as_gap_recovery(self):
        decoded = [
            {"word": "previous", "start": 0.85, "end": 1.10},
            {"word": "missing", "start": 2.0, "end": 2.5},
            {"word": "following", "start": 3.90, "end": 4.15},
        ]

        kept = keep_words_inside_gap(decoded, 1.0, 4.0)

        self.assertEqual([word["word"] for word in kept], ["missing"])

    def test_merge_preserves_repeated_text_at_distinct_times(self):
        original = [
            {"word": "same", "start": 1.0, "end": 1.4},
            {"word": "next", "start": 4.0, "end": 4.4},
        ]
        recovered = [{"word": "same", "start": 2.0, "end": 2.4}]

        merged = merge_words(original, recovered)

        self.assertEqual([word["word"] for word in merged], ["same", "same", "next"])

    def test_all_gaps_are_joined_into_one_repair_reel_and_mapped_back(self):
        audio = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
        gaps = [
            {"start": 2.0, "end": 3.0},
            {"start": 7.0, "end": 8.0},
        ]

        reel, mappings = build_repair_reel(
            audio,
            SAMPLE_RATE,
            gaps,
            context_seconds=0.5,
            separator_seconds=0.5,
        )
        reel_words = [
            {"word": "first", "start": 1.0, "end": 1.2},
            {"word": "second", "start": 3.5, "end": 3.7},
            {"word": "separator", "start": 2.1, "end": 2.3},
        ]

        recovered = map_reel_words_to_gaps(reel_words, mappings)

        self.assertAlmostEqual(len(reel) / SAMPLE_RATE, 4.5)
        self.assertEqual(recovered[0][0]["word"], "first")
        self.assertAlmostEqual(recovered[0][0]["start"], 2.5)
        self.assertEqual(recovered[1][0]["word"], "second")
        self.assertAlmostEqual(recovered[1][0]["start"], 7.5)
        self.assertNotIn("separator", [w["word"] for ws in recovered.values() for w in ws])


if __name__ == "__main__":
    unittest.main()
