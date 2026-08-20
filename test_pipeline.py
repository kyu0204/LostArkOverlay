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

    def test_grid_geometry_is_stable_across_frames(self):
        # 매 프레임 다시 잡지만, 화면이 같으면 결과도 같아야 한다.
        # (객체 동일성이 아니라 기하가 같은지를 본다)
        frame = build_frame(list(self.tiles.values()))
        self.rec.read_frame(frame)
        first = self.rec._grid
        self.rec.read_frame(frame)
        now = self.rec._grid
        self.assertEqual(
            (first.left, first.cell, first.pitch, first.count),
            (now.left, now.cell, now.pitch, now.count),
        )

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

    def test_grid_kept_while_bar_is_visible_even_if_icons_unknown(self):
        # 등록되지 않은 아이콘만 보여도 버프바 자체는 검출되므로
        # 그리드를 버릴 이유가 없다. (배율이 바뀌었다면 새로 잡힌 값이
        # 반영되므로 굳이 None으로 되돌릴 필요도 없다)
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        unknown = build_frame([icon(60, 77), icon(60, 78), icon(60, 79)])
        for _ in range(40):
            self.rec.read_frame(unknown)
        self.assertIsNotNone(self.rec._grid)

    def test_stale_grid_dropped_when_bar_disappears(self):
        # 바가 아예 사라지면(검출도 실패, 인식도 0) 오래된 좌표를
        # 붙들고 있을 이유가 없다. 30프레임 뒤 버린다.
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        self.assertIsNotNone(self.rec._grid)

        blank = np.zeros((44, 200, 3), np.uint8)
        for i in range(29):
            self.rec.read_frame(blank)
            self.assertIsNotNone(
                self.rec._grid, f"{i + 1}번째: 30번째 전에는 유지해야 한다"
            )
        self.rec.read_frame(blank)
        self.assertIsNone(self.rec._grid)


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


class TestGridFollowsMovement(unittest.TestCase):
    """버프바가 가로로 밀려도 따라가야 한다.

    그리드를 한 번 잡고 계속 쓰면 좌표가 굳는다. dHash가 견디는 것은
    ±1px뿐이라 그보다 밀리면 매칭이 통째로 무너진다. 게다가 재검출
    조건이 '연속 미인식'이면, 일부만 맞는 상태에서는 연속이 끊겨
    영영 복구되지 않는다. 그래서 매 프레임 다시 잡는다.
    """

    def setUp(self):
        self.tiles = {"a": icon(10, 3), "b": icon(120, 3), "c": icon(250, 3)}
        self.icons = IconBook()
        for k, v in self.tiles.items():
            self.icons.add(k, v[2:24, 2:24])
        self.rec = Recognizer(icons=self.icons, glyphs=GlyphBook(), icon_row_h=26)

    def _shifted(self, dx):
        """왼쪽에 dx만큼 여백을 넣어 바가 오른쪽으로 밀린 프레임."""
        frame = build_frame(list(self.tiles.values()))
        pad = np.zeros((frame.shape[0], dx, 3), np.uint8)
        return np.hstack([pad, frame])

    def test_phase_is_locked_while_bar_stays_put(self):
        # 버프바는 오른쪽 끝이 고정이고 왼쪽으로만 늘어난다. 즉 셀 경계의
        # 위상은 세션 내내 상수다. 버프 수가 변해도 인식이 흔들리면 안 된다.
        full = build_frame(list(self.tiles.values()))
        for _ in range(3):
            self.rec.read_frame(full)
        two = build_frame([self.tiles["a"], self.tiles["b"]])
        res = self.rec.read_frame(two)
        self.assertEqual(sorted(o.buff_id for o in res.observations), ["a", "b"])
        res = self.rec.read_frame(full)
        self.assertEqual(
            sorted(o.buff_id for o in res.observations), ["a", "b", "c"]
        )

    def test_recovers_if_bar_actually_moves(self):
        # 해상도 변경 등으로 ROI 자체가 어긋나면 위상이 틀어진다.
        # 인식이 끊기면 다시 맞춰 스스로 복구해야 한다.
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        moved = self._shifted(13)
        for _ in range(15):
            res = self.rec.read_frame(moved)
        self.assertEqual(
            sorted(o.buff_id for o in res.observations), ["a", "b", "c"],
            "바가 실제로 밀렸을 때 재보정으로 복구되지 않았다",
        )

    def test_keeps_grid_when_detection_fails(self):
        # 버프가 1개만 남으면 피치를 추정할 수 없다. 이때는 이전 그리드를
        # 유지해야 인식이 끊기지 않는다 (캐시가 있는 원래 이유).
        self.rec.read_frame(build_frame(list(self.tiles.values())))
        before = self.rec._grid
        self.assertIsNotNone(before)

        full = build_frame(list(self.tiles.values()))
        one = np.zeros_like(full)
        one[:, :26] = full[:, :26]          # 첫 칸만 남긴다
        res = self.rec.read_frame(one)

        self.assertIsNotNone(self.rec._grid)
        self.assertEqual(self.rec._grid.pitch, before.pitch)
        self.assertEqual([o.buff_id for o in res.observations], ["a"])


class TestVerticalAlignment(unittest.TestCase):
    """ROI 세로 위치가 몇 px 어긋나도 스스로 맞춘다.

    ROI는 사람이 잡으므로 정확할 수 없는데, dHash는 ±1px까지만 견딘다.
    실측에서 수련장 캡처가 4px 어긋나 인식 0개였고, 전투 캡처는 4px
    어긋나 14개 중 7개만 맞았다. 일부만 맞는 상태는 '실패'로 보이지
    않아 미인식 복구 경로에도 걸리지 않으므로, 시작할 때 한 번 맞춘다.
    """

    def setUp(self):
        self.tiles = {"a": icon(10, 3), "b": icon(120, 3), "c": icon(250, 3)}
        self.icons = IconBook()
        for k, v in self.tiles.items():
            self.icons.add(k, v[2:24, 2:24])

    def _frame_with_margin(self, top_pad):
        """아이콘 행 위에 top_pad만큼 배경을 덧붙인 프레임."""
        frame = build_frame(list(self.tiles.values()))
        pad = np.zeros((top_pad, frame.shape[1], 3), np.uint8)
        return np.vstack([pad, frame])

    def test_recovers_from_vertical_offset(self):
        for pad in (0, 2, 4, 6):
            rec = Recognizer(icons=self.icons, glyphs=GlyphBook())
            frame = self._frame_with_margin(pad)
            for _ in range(3):
                res = rec.read_frame(frame)
            self.assertEqual(
                sorted(o.buff_id for o in res.observations), ["a", "b", "c"],
                f"{pad}px 아래로 밀렸을 때 인식이 회복되지 않았다",
            )

    def test_calibration_runs_only_once(self):
        rec = Recognizer(icons=self.icons, glyphs=GlyphBook())
        frame = self._frame_with_margin(4)
        rec.read_frame(frame)
        self.assertTrue(rec._calibrated)
        top = rec._icon_top
        for _ in range(5):
            rec.read_frame(frame)
        self.assertEqual(rec._icon_top, top)

    def test_no_calibration_without_icon_db(self):
        # DB가 비어 있으면 무엇이 맞는지 판단할 근거가 없다
        rec = Recognizer(icons=IconBook(), glyphs=GlyphBook())
        rec.read_frame(self._frame_with_margin(4))
        self.assertEqual(rec._icon_top, 0)


class TestSavedGeometry(unittest.TestCase):
    """측정한 기하를 저장해 두고 재사용한다.

    버프가 적을 때 켜면 기하 자체를 잘못 잡는다. 실측(전투 캡처):

        버프 15개 -> pitch 28.9   버프 3개 -> pitch 24.0
        버프  6개 -> pitch 29.0   버프 2개 -> pitch 14.0

    전투 전에 앱을 켜는 것은 흔한 일이고, 한 번 잘못 잡으면 위상이
    고정돼 그 세션 내내 인식이 죽는다. 잘 측정된 값을 저장해 두면
    버프가 없어도 정확하게 시작한다.
    """

    def setUp(self):
        self.tiles = {"a": icon(10, 3), "b": icon(120, 3), "c": icon(250, 3)}
        self.icons = IconBook()
        for k, v in self.tiles.items():
            self.icons.add(k, v[2:24, 2:24])

    def _rec(self, **kw):
        return Recognizer(icons=self.icons, glyphs=GlyphBook(), **kw)

    def test_roundtrip(self):
        a = self._rec()
        frame = build_frame(list(self.tiles.values()))
        for _ in range(3):
            a.read_frame(frame)
        geom = a.geometry()
        self.assertIsNotNone(geom)

        b = self._rec(geometry=geom)
        res = b.read_frame(frame)
        self.assertEqual(
            sorted(o.buff_id for o in res.observations), ["a", "b", "c"]
        )
        self.assertAlmostEqual(b._grid.pitch, geom["pitch"], places=2)

    def test_saved_pitch_survives_a_bad_frame(self):
        # 버프가 거의 없는 프레임이 들어와도 저장된 피치를 지켜야 한다
        a = self._rec()
        frame = build_frame(list(self.tiles.values()))
        for _ in range(3):
            a.read_frame(frame)
        geom = a.geometry()

        b = self._rec(geometry=geom)
        one = build_frame([self.tiles["a"]])
        for _ in range(3):
            b.read_frame(one)
        self.assertAlmostEqual(b._grid.pitch, geom["pitch"], places=2)

    def test_geometry_is_none_before_any_frame(self):
        self.assertIsNone(self._rec().geometry())

    def test_bad_geometry_is_ignored(self):
        # 값이 비었으면 무시하고 평소대로 측정한다
        b = self._rec(geometry={"pitch": None, "cell": None})
        res = b.read_frame(build_frame(list(self.tiles.values())))
        self.assertEqual(
            sorted(o.buff_id for o in res.observations), ["a", "b", "c"]
        )


class TestRecoveryUsesBuffMatches(unittest.TestCase):
    """복구 판단은 '버프를 맞혔는가'로 한다.

    matches에는 오버플로 카운터도 들어오는데, 그 `+` 아이콘은 정렬이
    어긋난 자리에서도 곧잘 매칭된다. matches가 비었는지로만 보면
    카운터 하나 때문에 연속 카운터가 쌓이지 않아 복구가 안 된다
    (실측: 잘못된 위상이 60프레임 내내 고정됐다).
    """

    def test_stale_phase_recovers_even_if_counter_matches(self):
        tiles = {"a": icon(10, 3), "b": icon(120, 3), "c": icon(250, 3)}
        icons = IconBook()
        for k, v in tiles.items():
            icons.add(k, v[2:24, 2:24])
        counter = icon(100, 5)
        icons.add(OVERFLOW_ID, counter[2:24, 2:24])

        rec = Recognizer(icons=icons, glyphs=GlyphBook())
        frame = build_frame([counter] + list(tiles.values()))
        rec.read_frame(frame)
        # 위상을 일부러 어긋나게 만든다
        rec._phase = (rec._phase or 0.0) + rec._grid.pitch * 0.5
        rec._grid = rec._apply_phase(rec._grid)

        for _ in range(20):
            res = rec.read_frame(frame)
        self.assertGreaterEqual(
            res.visible, 3, "카운터가 맞는 동안 복구가 막히면 안 된다"
        )
