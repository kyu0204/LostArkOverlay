"""파이프라인 테스트. 합성 프레임으로 전 구간을 검증한다."""

import unittest

import numpy as np

from buff_state import BuffTracker
from icon_match import IconBook
from pipeline import OVERFLOW_ID, Recognizer, consistency_gap
from text_parse import GlyphBook


def icon(hue_deg: float, seed: int, size: int = 26) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    hsv = np.zeros((size, size, 3), np.uint8)
    hsv[:, :, 0] = int(hue_deg / 2) % 180
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = rng.integers(40, 255, (size, size), dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 검출 가능한 테두리를 두른다
    bgr[0, :] = bgr[-1, :] = (60, 220, 60)
    bgr[:, 0] = bgr[:, -1] = (60, 220, 60)
    return bgr


def build_frame(tiles, cell=26, gap=3, text_h=18):
    """아이콘 행 + 빈 텍스트 행으로 구성된 프레임."""
    n = len(tiles)
    w = n * cell + (n - 1) * gap
    icon_row = np.zeros((cell, w, 3), np.uint8)
    for i, t in enumerate(tiles):
        x = i * (cell + gap)
        icon_row[:, x:x + cell] = t
    text_row = np.full((text_h, w, 3), 20, np.uint8)
    return np.vstack([icon_row, text_row])


class TestRecognizer(unittest.TestCase):
    def setUp(self):
        self.tiles = {
            "a": icon(10, 3),
            "b": icon(120, 3),
            "c": icon(250, 3),
        }
        self.icons = IconBook()
        for k, v in self.tiles.items():
            self.icons.add(k, v[2:24, 2:24])
        self.rec = Recognizer(icons=self.icons, glyphs=GlyphBook(), icon_row_h=26)

    def test_identifies_all_cells(self):
        frame = build_frame(list(self.tiles.values()))
        res = self.rec.read_frame(frame)
        self.assertIsNotNone(res.grid)
        self.assertEqual(
            sorted(o.buff_id for o in res.observations), ["a", "b", "c"]
        )

    def test_remaining_is_none_without_glyphs(self):
        # 글자 템플릿이 비어 있으면 시간은 못 읽지만 아이콘은 인식된다
        frame = build_frame(list(self.tiles.values()))
        res = self.rec.read_frame(frame)
        self.assertTrue(all(o.remaining is None for o in res.observations))
        self.assertEqual(res.visible, 3)

    def test_unknown_icons_skipped(self):
        frame = build_frame([self.tiles["a"], icon(60, 77)])
        res = self.rec.read_frame(frame)
        self.assertEqual([o.buff_id for o in res.observations], ["a"])
        self.assertEqual(res.visible, 1)

    def test_grid_is_cached(self):
        frame = build_frame(list(self.tiles.values()))
        self.rec.read_frame(frame)
        first = self.rec._grid
        self.rec.read_frame(frame)
        self.assertIs(self.rec._grid, first)

    def test_single_icon_works_once_grid_cached(self):
        # 버프가 1개만 남아도 인식되어야 한다. 1개로는 피치를 추정할 수
        # 없으므로, 캐시된 그리드를 계속 쓰는 동작이 필수다.
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        res = self.rec.read_frame(build_frame([self.tiles["a"]]))
        self.assertEqual([o.buff_id for o in res.observations], ["a"])

    def test_reset_grid(self):
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        self.assertIsNotNone(self.rec._grid)
        self.rec.reset_grid()
        self.assertIsNone(self.rec._grid)

    def test_empty_frame_returns_nothing(self):
        blank = np.zeros((44, 200, 3), np.uint8)
        res = self.rec.read_frame(blank)
        self.assertEqual(res.observations, [])
        self.assertIsNone(res.grid)


class TestOverflowCell(unittest.TestCase):
    def setUp(self):
        self.counter = icon(100, 5)
        self.buff = icon(250, 5)
        self.icons = IconBook()
        self.icons.add(OVERFLOW_ID, self.counter[2:24, 2:24])
        self.icons.add("x", self.buff[2:24, 2:24])
        self.rec = Recognizer(icons=self.icons, glyphs=GlyphBook(), icon_row_h=26)

    def test_counter_not_reported_as_buff(self):
        frame = build_frame([self.counter, self.buff])
        res = self.rec.read_frame(frame)
        self.assertEqual([o.buff_id for o in res.observations], ["x"])
        self.assertEqual(res.visible, 1)

    def test_counter_absent_means_no_overflow(self):
        frame = build_frame([self.buff])
        self.assertIsNone(self.rec.read_frame(frame).overflow)


class TestConsistency(unittest.TestCase):
    def test_no_counter_is_unknown(self):
        from pipeline import FrameResult

        self.assertIsNone(consistency_gap(FrameResult(visible=3), tracked=3))

    def test_consistent(self):
        from pipeline import FrameResult

        r = FrameResult(visible=9, overflow=3)
        self.assertEqual(consistency_gap(r, tracked=12), 0)

    def test_missed_buff_is_negative(self):
        from pipeline import FrameResult

        r = FrameResult(visible=9, overflow=3)
        self.assertLess(consistency_gap(r, tracked=10), 0)

    def test_stale_timer_is_positive(self):
        from pipeline import FrameResult

        r = FrameResult(visible=9, overflow=3)
        self.assertGreater(consistency_gap(r, tracked=15), 0)


class TestEndToEnd(unittest.TestCase):
    """인식 -> 상태 머신 -> 오버레이 행까지."""

    def setUp(self):
        self.tiles = {"a": icon(10, 9), "b": icon(200, 9)}
        icons = IconBook()
        for k, v in self.tiles.items():
            icons.add(k, v[2:24, 2:24])
        self.rec = Recognizer(icons=icons, glyphs=GlyphBook(), icon_row_h=26)
        self.tracker = BuffTracker({
            "a": {"name": "A", "duration": 10.0, "priority": 1},
            "b": {"name": "B", "duration": 6.0, "priority": 2},
        })
        self.frame = build_frame(list(self.tiles.values()))

    def test_catalog_fallback_fills_duration(self):
        res = self.rec.read_frame(self.frame)
        self.tracker.update(res.observations, now=0.0)
        rows = {r["id"]: r for r in self.tracker.snapshot(0.0)}
        self.assertAlmostEqual(rows["a"]["remaining"], 10.0)
        self.assertAlmostEqual(rows["b"]["remaining"], 6.0)

    def test_pushed_out_buff_keeps_running(self):
        # 첫 프레임에 둘 다 보이고, 이후 한 칸만 남는다.
        # 그리드는 캐시되므로 남은 칸이 1개여도 계속 인식된다.
        self.tracker.update(self.rec.read_frame(self.frame).observations, now=0.0)
        one = build_frame([self.tiles["a"]])
        self.tracker.update(self.rec.read_frame(one).observations, now=3.0)

        rows = {r["id"]: r for r in self.tracker.snapshot(3.0)}
        self.assertIn("b", rows)                       # 안 보여도 살아 있어야 한다
        self.assertEqual(rows["b"]["confidence"], "predicted")
        self.assertEqual(rows["a"]["confidence"], "observed")
        self.assertAlmostEqual(rows["b"]["remaining"], 3.0)

    def test_expires_by_timer_while_hidden(self):
        self.tracker.update(self.rec.read_frame(self.frame).observations, now=0.0)
        self.tracker.update([], now=6.0)
        self.assertNotIn("b", self.tracker.active_ids)  # 6초 버프는 만료
        self.assertIn("a", self.tracker.active_ids)     # 10초 버프는 생존


if __name__ == "__main__":
    unittest.main(verbosity=2)
