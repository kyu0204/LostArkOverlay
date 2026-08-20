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

    def test_red_border(self):
        # sup_red 아이콘 테두리 색. border_mask의 reddish 분기 검증.
        g = detect_grid(make_bar(border=(30, 30, 200), n=6))
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


class TestAutocorrFallback(unittest.TestCase):
    """색 임계값이 무너져도 주기로 그리드를 잡는다.

    border_mask는 고정 색 임계값을 쓰므로 밝은 배경에서 경계에 아슬아슬
    하게 걸친다. 실측: 같은 화면을 파일로 저장한 것과 화면에서 다시
    캡처한 것의 픽셀 평균차가 6뿐인데도 피치가 28.8 vs 84.6으로 갈렸다.
    아이콘 좌우 테두리가 만드는 '주기'는 색과 무관하므로 이를 쓴다.
    """

    def test_autocorr_finds_pitch(self):
        from grid_detect import pitch_by_autocorr

        pitch, score = pitch_by_autocorr(make_bar(cell=26, pitch=29.0, n=8))
        self.assertAlmostEqual(pitch, 29.0, delta=1.0)
        self.assertGreater(score, 0.25)

    def test_grid_found_when_border_color_fails(self):
        # 테두리를 무채색으로 바꿔 border_mask가 아무것도 못 잡게 만든다.
        # (내부 무늬도 끈다. 무늬의 랜덤 색이 마스크를 통과해 버린다)
        bar = make_bar(
            cell=26, pitch=29.0, n=8, border=(150, 150, 150), noisy=False
        )
        from grid_detect import border_mask

        self.assertEqual(border_mask(bar).sum(), 0, "이 테스트는 마스크가 비어야 유효")
        g = detect_grid(bar)
        self.assertIsNotNone(g, "주기 폴백이 동작하지 않았다")
        self.assertAlmostEqual(g.pitch, 29.0, delta=1.5)

    def test_rejects_flat_image(self):
        # 아무 구조도 없으면 주기도 없다
        self.assertIsNone(detect_grid(np.full((28, 300, 3), 20, np.uint8)))


class TestLocateBuffBar(unittest.TestCase):
    """화면 전체에서 버프바를 찾는다 (드래그로 ROI를 잡지 않게 하려는 것).

    해상도별 좌표표를 두지 않는 이유: 같은 1920x1080에서도 UI 배율에
    따라 아이콘이 21px과 26px로 갈린다(README '실측값'). 해상도는
    좌표를 결정하지 못하므로, 화면에서 직접 찾는 편이 맞다.
    """

    def _screen(self, bar_x=400, bar_y=900, W=1280, H=1000, **kw):
        """하단에 버프바를 얹은 가짜 화면."""
        scr = np.full((H, W, 3), 18, np.uint8)
        bar = make_bar(**kw)
        h, w = bar.shape[:2]
        scr[bar_y:bar_y + h, bar_x:bar_x + w] = bar
        return scr, bar_x, bar_y, w

    def test_finds_bar_position(self):
        from grid_detect import locate_buff_bar

        scr, bx, by, bw = self._screen(n=10)
        roi = locate_buff_bar(scr)
        self.assertIsNotNone(roi, "버프바를 찾지 못했다")
        # ROI는 바보다 조금 위에서 시작해야 한다.
        # 세로 정렬 보정이 아래로만 훑으므로 위쪽 여유가 필요하다.
        self.assertLessEqual(roi["y"], by)
        self.assertGreaterEqual(roi["y"], by - 10)
        self.assertAlmostEqual(roi["pitch"], 29.0, delta=2.0)

    def test_roi_covers_the_bar(self):
        from grid_detect import locate_buff_bar

        scr, bx, by, bw = self._screen(n=10)
        roi = locate_buff_bar(scr)
        self.assertLessEqual(roi["x"], bx + 8)
        self.assertGreaterEqual(roi["x"] + roi["w"], bx + bw - 30)

    def test_slots_anchors_width_from_right(self):
        # 버프바는 오른쪽 끝이 고정이고 왼쪽으로 늘어난다.
        # 칸수를 주면 그만큼만 잡아야 한다.
        from grid_detect import locate_buff_bar

        scr, bx, by, bw = self._screen(n=10)
        wide = locate_buff_bar(scr)
        narrow = locate_buff_bar(scr, slots=4)
        self.assertLess(narrow["w"], wide["w"])
        # 오른쪽 끝은 그대로여야 한다
        self.assertAlmostEqual(
            narrow["x"] + narrow["w"], wide["x"] + wide["w"], delta=4
        )

    def test_returns_none_on_empty_screen(self):
        from grid_detect import locate_buff_bar

        self.assertIsNone(locate_buff_bar(np.full((1000, 1280, 3), 18, np.uint8)))
