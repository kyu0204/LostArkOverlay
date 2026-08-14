"""그리드 자동 검출 테스트. 합성 버프바로 검증한다."""

import unittest

import numpy as np

from grid_detect import Grid, _pitch_from_lefts, _runs, detect_grid


def make_bar(cell=26, pitch=29.0, n=8, h=28, left=2, border=(80, 220, 80),
             noisy=True, skip=()):
    """가짜 버프바. 테두리 사각형 + 내부 무늬."""
    w = int(round(left + (n - 1) * pitch)) + cell + 4
    img = np.full((h, w, 3), 24, np.uint8)
    rng = np.random.default_rng(0)
    for i in range(n):
        if i in skip:
            continue
        x = int(round(left + i * pitch))
        if noisy:
            # 내부 무늬. 세로로 짧게 끊기는 밝은 픽셀들
            img[3:h - 3, x + 2:x + cell - 2] = rng.integers(
                30, 200, (h - 6, cell - 4, 3), dtype=np.uint8
            )
        img[0, x:x + cell] = border
        img[h - 1, x:x + cell] = border
        img[0:h, x] = border
        img[0:h, x + cell - 1] = border
    return img


class TestRuns(unittest.TestCase):
    def test_basic(self):
        f = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], bool)
        self.assertEqual(_runs(f), [(1, 3), (6, 7)])

    def test_min_len(self):
        f = np.array([1, 0, 1, 1, 1, 0], bool)
        self.assertEqual(_runs(f, min_len=2), [(2, 4)])

    def test_edges(self):
        f = np.array([1, 1, 0, 1], bool)
        self.assertEqual(_runs(f), [(0, 1), (3, 3)])

    def test_empty(self):
        self.assertEqual(_runs(np.zeros(5, bool)), [])


class TestPitchFromLefts(unittest.TestCase):
    def test_even_spacing(self):
        pitch, score = _pitch_from_lefts([0, 29, 58, 87])
        self.assertAlmostEqual(pitch, 29.0, places=1)
        self.assertEqual(score, 3)

    def test_missing_icon_in_middle(self):
        # 58이 검출되지 않아 간격이 58(=29*2)로 벌어진 경우
        pitch, score = _pitch_from_lefts([0, 29, 87, 116])
        self.assertAlmostEqual(pitch, 29.0, places=1)

    def test_fractional_pitch_recovered(self):
        # 실측 배율: 피치 23.33 -> 정수 좌표는 0,23,47,70
        pitch, _ = _pitch_from_lefts([0, 23, 47, 70])
        self.assertAlmostEqual(pitch, 23.33, places=1)

    def test_noise_gap_not_chosen(self):
        # 9는 노이즈 간격. 23이 주기여야 한다
        pitch, _ = _pitch_from_lefts([0, 9, 23, 46, 69])
        self.assertGreater(pitch, 20)

    def test_too_few(self):
        self.assertEqual(_pitch_from_lefts([5]), (None, 0))


class TestDetectGrid(unittest.TestCase):
    def test_integer_pitch(self):
        g = detect_grid(make_bar(cell=26, pitch=29.0, n=8))
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.pitch, 29.0, places=0)
        self.assertEqual(g.cell, 26)
        self.assertEqual(g.count, 8)

    def test_fractional_pitch(self):
        g = detect_grid(make_bar(cell=21, pitch=23.33, n=8, h=23))
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.pitch, 23.33, delta=0.4)
        self.assertAlmostEqual(g.cell, 21, delta=1)

    def test_survives_missing_icons(self):
        # 가운데 두 칸이 다른 색이라 검출되지 않은 상황
        g = detect_grid(make_bar(n=10, skip=(3, 4)))
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.pitch, 29.0, delta=0.6)

    def test_orange_border(self):
        g = detect_grid(make_bar(border=(40, 150, 240), n=6))
        self.assertIsNotNone(g)
        self.assertAlmostEqual(g.pitch, 29.0, delta=0.6)

    def test_rejects_blank(self):
        self.assertIsNone(detect_grid(np.full((28, 300, 3), 20, np.uint8)))

    def test_rejects_too_short_roi(self):
        self.assertIsNone(detect_grid(np.zeros((2, 300, 3), np.uint8)))

    def test_rejects_single_icon(self):
        self.assertIsNone(detect_grid(make_bar(n=1)))


class TestGridGeometry(unittest.TestCase):
    def test_bounds_no_drift(self):
        g = Grid(left=0, cell=21, pitch=23.33, count=8)
        for i, (l, r) in enumerate(g.bounds()):
            self.assertEqual(r - l, 21)
            self.assertLessEqual(abs(l - i * 23.33), 0.5)

    def test_gap(self):
        self.assertAlmostEqual(Grid(0, 26, 29.0, 4).gap, 3.0)

    def test_spec_roundtrip(self):
        from capture import parse_grid

        g = Grid(left=5, cell=21, pitch=23.33, count=4)
        cell, gap, off = parse_grid(g.spec())
        self.assertEqual(cell, 21)
        self.assertAlmostEqual(gap, 2.33, places=2)
        self.assertEqual(off, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
