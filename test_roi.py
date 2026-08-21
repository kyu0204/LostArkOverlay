"""ROI 기하 + 그리드 분할 테스트. 화면 없이 전부 검증 가능."""

import unittest

import numpy as np

from capture import cell_bounds, draw_grid, parse_grid, slice_cells
from roi_picker import clamp, is_usable, normalize, nudge


class TestNormalize(unittest.TestCase):
    def test_left_to_right(self):
        self.assertEqual(normalize(10, 20, 110, 60),
                         {"x": 10, "y": 20, "w": 100, "h": 40})

    def test_right_to_left_drag(self):
        # 역방향 드래그도 같은 결과가 나와야 한다
        self.assertEqual(normalize(110, 60, 10, 20),
                         {"x": 10, "y": 20, "w": 100, "h": 40})

    def test_zero_size_click(self):
        self.assertEqual(normalize(5, 5, 5, 5), {"x": 5, "y": 5, "w": 0, "h": 0})


class TestClamp(unittest.TestCase):
    bounds = (0, 0, 1920, 1080)

    def test_inside_unchanged(self):
        r = {"x": 100, "y": 100, "w": 200, "h": 50}
        self.assertEqual(clamp(r, self.bounds), r)

    def test_overflow_right_is_trimmed(self):
        r = clamp({"x": 1800, "y": 10, "w": 400, "h": 50}, self.bounds)
        self.assertEqual(r["x"] + r["w"], 1920)

    def test_negative_origin_pulled_in(self):
        r = clamp({"x": -50, "y": -20, "w": 100, "h": 40}, self.bounds)
        self.assertEqual((r["x"], r["y"]), (0, 0))


class TestNudge(unittest.TestCase):
    base = {"x": 100, "y": 100, "w": 200, "h": 40}

    def test_move(self):
        self.assertEqual(nudge(self.base, 1, 0)["x"], 101)
        self.assertEqual(nudge(self.base, 0, -1)["y"], 99)

    def test_resize(self):
        r = nudge(self.base, 1, -1, resize=True)
        self.assertEqual((r["w"], r["h"]), (201, 39))
        self.assertEqual((r["x"], r["y"]), (100, 100))  # 원점은 안 움직인다

    def test_resize_floor(self):
        r = nudge({"x": 0, "y": 0, "w": 1, "h": 1}, -5, -5, resize=True)
        self.assertEqual((r["w"], r["h"]), (1, 1))


class TestUsable(unittest.TestCase):
    def test_click_without_drag_rejected(self):
        self.assertFalse(is_usable({"x": 5, "y": 5, "w": 0, "h": 0}))

    def test_tiny_rejected(self):
        self.assertFalse(is_usable({"x": 0, "y": 0, "w": 3, "h": 40}))

    def test_normal_accepted(self):
        self.assertTrue(is_usable({"x": 0, "y": 0, "w": 300, "h": 40}))

    def test_none_rejected(self):
        self.assertFalse(is_usable(None))


class TestGridSpec(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(parse_grid("32x4"), (32, 4.0, 0))

    def test_with_offset(self):
        self.assertEqual(parse_grid("30x2+5"), (30, 2.0, 5))

    def test_zero_gap(self):
        self.assertEqual(parse_grid("28x0"), (28, 0.0, 0))

    def test_fractional_gap(self):
        # 게임 UI 배율 때문에 간격이 소수로 나온다
        self.assertEqual(parse_grid("21x2.33"), (21, 2.33, 0))
        self.assertEqual(parse_grid("21x2.33+1"), (21, 2.33, 1))

    def test_invalid(self):
        for bad in ("32", "32-4", "x4", "", "32x4+", "21x2.33.4"):
            with self.assertRaises(ValueError):
                parse_grid(bad)


class TestCellBounds(unittest.TestCase):
    def test_exact_fit(self):
        # 셀 30 + 간격 2. 세 번째 셀은 64+30=94 > 92 라 제외된다
        self.assertEqual(cell_bounds(92, 30, 2), [(0, 30), (32, 62)])

    def test_partial_cell_excluded(self):
        # 마지막에 20px만 남으면 셀을 만들지 않는다
        b = cell_bounds(100, 40, 0)
        self.assertEqual(b, [(0, 40), (40, 80)])

    def test_offset_shifts_all(self):
        b = cell_bounds(100, 40, 0, offset=5)
        self.assertEqual(b, [(5, 45), (45, 85)])

    def test_no_gap(self):
        self.assertEqual(cell_bounds(120, 40, 0), [(0, 40), (40, 80), (80, 120)])

    def test_too_narrow(self):
        self.assertEqual(cell_bounds(10, 40, 0), [])

    def test_fractional_pitch_no_drift(self):
        # 로아 실측: 셀 21, 피치 23.33. 8칸까지 1px 이내여야 한다.
        b = cell_bounds(200, 21, 2.33)
        lefts = [l for l, _ in b]
        for i, l in enumerate(lefts):
            self.assertLessEqual(abs(l - i * 23.33), 0.5, f"{i}번 셀이 밀림")

    def test_integer_pitch_would_drift(self):
        # 정수 피치로 하면 4번째부터 어긋난다는 것을 명시적으로 남긴다
        int_lefts = [l for l, _ in cell_bounds(200, 21, 2)]
        self.assertEqual(int_lefts[3], 69)      # 정수: 0,23,46,69
        flt_lefts = [l for l, _ in cell_bounds(200, 21, 2.33)]
        self.assertEqual(flt_lefts[3], 70)      # 실수: 0,23,47,70
        self.assertNotEqual(int_lefts[7], flt_lefts[7])


class TestSliceCells(unittest.TestCase):
    def test_shapes(self):
        img = np.zeros((36, 160, 3), dtype=np.uint8)
        cells = slice_cells(img, 32, 4)
        self.assertEqual(len(cells), 4)
        for c in cells:
            self.assertEqual(c.shape, (36, 32, 3))

    def test_content_alignment(self):
        # 각 셀에 서로 다른 값을 심어두고 잘린 위치가 맞는지 본다
        img = np.zeros((10, 100, 3), dtype=np.uint8)
        img[:, 0:40] = 10
        img[:, 40:80] = 20
        cells = slice_cells(img, 40, 0)
        self.assertEqual(cells[0][0, 0, 0], 10)
        self.assertEqual(cells[1][0, 0, 0], 20)


class TestDrawGrid(unittest.TestCase):
    def test_does_not_mutate_input(self):
        img = np.zeros((36, 160, 3), dtype=np.uint8)
        before = img.copy()
        out, n = draw_grid(img, 32, 4)
        self.assertEqual(n, 4)
        self.assertTrue(np.array_equal(img, before))
        self.assertFalse(np.array_equal(out, before))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProfileKey(unittest.TestCase):
    """기하 프로필 이름 (config.profile_key)."""

    def test_combines_both_scale_knobs(self):
        # 게임에는 배율 손잡이가 둘 있다. 하나만 이름에 쓰면 서로 다른
        # 기하가 같은 이름 아래 덮어써진다.
        import config
        a = config.profile_key({"hud_scale": "100", "ui_scale": "100"})
        b = config.profile_key({"hud_scale": "110", "ui_scale": "100"})
        c = config.profile_key({"hud_scale": "100", "ui_scale": "120"})
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_missing_knob_becomes_auto(self):
        import config
        self.assertEqual(config.profile_key({}), "hudauto-buffauto")
        self.assertEqual(config.profile_key({"ui_scale": "100"}),
                         "hudauto-buff100")

    def test_scalar_keeps_old_behaviour(self):
        # 설정 탭이 없던 시절 저장된 값과 섞이면 안 된다.
        import config
        self.assertEqual(config.profile_key("100"), "100")
        self.assertEqual(config.profile_key(None), "auto")
        self.assertEqual(config.profile_key(""), "auto")

    def test_same_settings_give_same_name(self):
        import config
        s = {"hud_scale": "100", "ui_scale": "100", "slots": 14}
        self.assertEqual(config.profile_key(s), config.profile_key(dict(s)))
