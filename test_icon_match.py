"""아이콘 dHash + 색상 매칭 테스트."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from icon_match import (
    IconBook,
    dhash,
    hamming,
    hue_distance,
    identify_cells,
    mean_hue,
)


def tile(hue_deg: float, seed: int = 0, size: int = 24, sat: int = 200) -> np.ndarray:
    """지정 색상의 합성 아이콘. seed가 같으면 모양도 같다."""
    import cv2

    rng = np.random.default_rng(seed)
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[:, :, 0] = int(hue_deg / 2) % 180
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = rng.integers(40, 255, (size, size), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TestDHash(unittest.TestCase):
    def test_identical(self):
        t = tile(0, seed=1)
        self.assertEqual(hamming(dhash(t), dhash(t)), 0)

    def test_brightness_invariant(self):
        t = tile(0, seed=1)
        dark = np.clip(t * 0.6, 0, 255).astype(np.uint8)
        self.assertLessEqual(hamming(dhash(t), dhash(dark)), 4)

    def test_different_shapes_differ(self):
        a, b = tile(0, seed=1), tile(0, seed=99)
        self.assertGreater(hamming(dhash(a), dhash(b)), 10)

    def test_hash_fits_64_bits(self):
        self.assertLess(dhash(tile(120, seed=3)), 1 << 64)


class TestHue(unittest.TestCase):
    def test_measures_hue(self):
        self.assertAlmostEqual(mean_hue(tile(200)), 200, delta=6)

    def test_circular_mean_near_zero(self):
        # 350도와 10도의 평균은 180도가 아니라 0도여야 한다
        import cv2

        a = tile(350, size=12)
        b = tile(10, size=12)
        merged = np.hstack([a, b])
        h = mean_hue(merged)
        self.assertLess(min(h, 360 - h), 20)

    def test_greyscale_returns_none(self):
        grey = np.full((24, 24, 3), 128, np.uint8)
        self.assertIsNone(mean_hue(grey))

    def test_distance_wraps(self):
        self.assertAlmostEqual(hue_distance(350, 10), 20, delta=0.1)
        self.assertAlmostEqual(hue_distance(10, 350), 20, delta=0.1)

    def test_distance_with_none_is_zero(self):
        # 무채색이면 색으로 판별할 수 없으므로 통과시킨다
        self.assertEqual(hue_distance(None, 100), 0.0)
        self.assertEqual(hue_distance(100, None), 0.0)


class TestIconBook(unittest.TestCase):
    def setUp(self):
        self.book = IconBook()
        # 모양은 같고 색만 다른 4종. 실제 서폿 버프가 이 형태다.
        self.tiles = {
            "red": tile(5, seed=7),
            "gold": tile(50, seed=7),
            "blue": tile(205, seed=7),
            "purple": tile(290, seed=7),
        }
        for k, v in self.tiles.items():
            self.book.add(k, v)

    def test_exact_match(self):
        for k, v in self.tiles.items():
            r = self.book.match(v)
            self.assertTrue(r.ok)
            self.assertEqual(r.buff_id, k)
            self.assertEqual(r.distance, 0)

    def test_same_shape_different_color_distinguished(self):
        # dHash만으로는 갈리지 않는 조합. Hue가 결정해야 한다.
        self.assertEqual(
            hamming(dhash(self.tiles["red"]), dhash(self.tiles["blue"])), 0
        )
        self.assertEqual(self.book.match(self.tiles["blue"]).buff_id, "blue")

    def test_unknown_icon_rejected(self):
        self.assertFalse(self.book.match(tile(120, seed=42)).ok)

    def test_empty_book(self):
        self.assertFalse(IconBook().match(tile(0)).ok)

    def test_brightness_change_still_matches(self):
        dim = np.clip(self.tiles["gold"] * 0.7, 0, 255).astype(np.uint8)
        self.assertEqual(self.book.match(dim).buff_id, "gold")

    def test_multiple_samples_per_icon(self):
        self.book.add("red", np.clip(self.tiles["red"] * 0.8, 0, 255).astype(np.uint8))
        self.assertEqual(len(self.book.entries["red"].hashes), 2)
        self.assertEqual(self.book.match(self.tiles["red"]).buff_id, "red")

    def test_ignored_entry_not_ok(self):
        self.book.add("potion", tile(100, seed=3), ignore=True)
        r = self.book.match(tile(100, seed=3))
        self.assertEqual(r.buff_id, "potion")
        self.assertTrue(r.ignored)
        self.assertFalse(r.ok)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "icons.json"
            self.book.save(p)
            loaded = IconBook.load(p)
        self.assertEqual(set(loaded.entries), set(self.tiles))
        self.assertEqual(loaded.match(self.tiles["purple"]).buff_id, "purple")

    def test_load_missing_file(self):
        self.assertEqual(IconBook.load(Path("/nonexistent/icons.json")).entries, {})


class TestShiftTolerance(unittest.TestCase):
    """1px 어긋남이 오인의 주된 원인이므로 별도로 검증한다."""

    def setUp(self):
        import cv2

        rng = np.random.default_rng(11)
        big = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
        self.big = cv2.GaussianBlur(big, (5, 5), 0)
        self.book = IconBook()
        self.book.add("x", self.big[8:32, 8:32])

    def test_shift_breaks_naive_hash(self):
        # 시프트 보정이 없으면 거리가 크게 벌어진다는 사실을 남긴다
        base = dhash(self.big[8:32, 8:32])
        moved = dhash(self.big[8:32, 9:33])
        self.assertGreater(hamming(base, moved), 6)

    def test_match_survives_one_px_shift(self):
        for dx in (-1, 1):
            r = self.book.match(self.big[8:32, 8 + dx:32 + dx])
            self.assertEqual(r.buff_id, "x", f"dx={dx}")

    def test_shift_disabled_is_stricter(self):
        strict = self.book.match(self.big[8:32, 9:33], shift=0)
        loose = self.book.match(self.big[8:32, 9:33], shift=1)
        self.assertLessEqual(loose.distance, strict.distance)


class TestIdentifyCells(unittest.TestCase):
    def test_maps_cells_and_skips_unknown(self):
        known = tile(200, seed=5)
        other = tile(30, seed=61)
        row = np.hstack([known, np.zeros((24, 3, 3), np.uint8), other])
        book = IconBook()
        book.add("known", known[2:22, 2:22])

        got = identify_cells(row, [(0, 24), (27, 51)], book, inset=2)
        self.assertEqual(list(got), [0])
        self.assertEqual(got[0].buff_id, "known")

    def test_out_of_range_bounds_stop(self):
        book = IconBook()
        book.add("a", tile(10, seed=2))
        row = tile(10, seed=2)
        got = identify_cells(row, [(0, 24), (100, 124)], book, inset=0)
        self.assertNotIn(1, got)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestHueRelaxation(unittest.TestCase):
    """압축으로 모양이 흐려진 소스 대응.

    dHash는 밝기 미세구조를 보므로 영상 압축에 흔들리지만, 평균 Hue는
    거의 그대로 남는다. 색이 아주 가까울 때만 해밍을 더 봐주는 규칙이
    실제로 그렇게 동작하는지 고정한다.

    해밍 거리를 정확히 통제하려고 하위 k비트를 뒤집고 shift=0으로 둔다.
    (시프트를 켜면 변형본 중 최소 거리를 취해 값이 흔들린다)
    """

    def setUp(self):
        from icon_match import IconEntry

        self.IconEntry = IconEntry
        self.t = tile(200, seed=5)
        self.h = dhash(self.t)
        self.hue = mean_hue(self.t)

    def _book(self, flip_bits: int, hue_shift: float = 0.0):
        mask = (1 << flip_bits) - 1
        return IconBook({
            "x": self.IconEntry("x", [self.h ^ mask], [self.hue + hue_shift])
        })

    def test_degraded_shape_matches_when_hue_is_near(self):
        # 해밍 13 (기본 임계 10 초과) + 색은 거의 동일 -> 완화로 통과
        res = self._book(13, hue_shift=3.0).match(self.t, shift=0)
        self.assertEqual(res.buff_id, "x")
        self.assertEqual(res.distance, 13)

    def test_rejected_when_shape_too_far_even_if_hue_matches(self):
        # 색이 같아도 모양이 RELAXED_HAM(16)을 넘으면 거부
        res = self._book(20, hue_shift=1.0).match(self.t, shift=0)
        self.assertIsNone(res.buff_id)

    def test_rejected_when_hue_not_close_enough(self):
        # 해밍은 완화 범위 안이지만 색이 TIGHT_HUE_DEG(8도)를 넘으면 거부
        res = self._book(13, hue_shift=15.0).match(self.t, shift=0)
        self.assertIsNone(res.buff_id)

    def test_relaxation_can_be_disabled(self):
        res = self._book(13, hue_shift=1.0).match(
            self.t, shift=0, relax_on_hue=False
        )
        self.assertIsNone(res.buff_id)

    def test_strict_match_still_works(self):
        # 완화와 무관하게 원래 경로는 그대로여야 한다
        res = self._book(2, hue_shift=1.0).match(self.t, shift=0)
        self.assertEqual(res.buff_id, "x")
        self.assertEqual(res.distance, 2)
