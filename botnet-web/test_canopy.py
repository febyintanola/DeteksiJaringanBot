import unittest
from unittest.mock import patch

import numpy as np

from app.pipeline.canopy import _canopy_from_similarity_matrix, build_canopies


class _FailingANNIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def is_available(self) -> bool:
        return True

    def init(self, total: int, m: int = 32, ef_construction: int = 200) -> None:
        return None

    def add(self, vectors) -> None:
        return None

    def query(self, vector, k: int):
        raise RuntimeError("Cannot return the results in a contiguous 2D array")


class CanopyFallbackTests(unittest.TestCase):
    def test_build_canopies_falls_back_when_ann_query_fails(self):
        user_texts = {
            f"user_{idx:02d}": f"spam promo akun {idx % 3} diskon cepat"
            for idx in range(12)
        }

        with patch("app.pipeline.canopy._ANNIndex", _FailingANNIndex):
            canopy = build_canopies(user_texts, t1=0.6, t2=0.8, use_ann=True)

        self.assertTrue(canopy.assignments)
        self.assertTrue(canopy.canopies)
        self.assertEqual(set(canopy.assignments), set(canopy.embeddings))
        self.assertTrue(canopy.timing.used_ann)
        self.assertTrue(canopy.timing.ann_fallback)
        self.assertGreaterEqual(canopy.timing.total_sec, 0.0)

    def test_outer_similarity_members_are_not_removed_as_center_candidates(self):
        user_ids = ["center", "outer_match", "unrelated"]
        similarities = np.array(
            [
                [1.0, 0.7, 0.2],
                [0.7, 1.0, 0.2],
                [0.2, 0.2, 1.0],
            ]
        )

        _assignments, canopies = _canopy_from_similarity_matrix(user_ids, similarities, t1=0.6, t2=0.8)

        self.assertEqual(len(canopies), 3)
        self.assertTrue(any({"center", "outer_match"}.issubset(set(members)) for members in canopies.values()))

    def test_legacy_threshold_order_is_reported_as_outer_then_inner(self):
        user_texts = {
            "user_a": "promo diskon cepat",
            "user_b": "promo diskon hemat",
            "user_c": "komentar biasa tentang video",
        }

        canopy = build_canopies(user_texts, t1=0.8, t2=0.6, use_ann=False, auto_tune=False)

        self.assertEqual(canopy.threshold_t1, 0.6)
        self.assertEqual(canopy.threshold_t2, 0.8)


if __name__ == "__main__":
    unittest.main()
