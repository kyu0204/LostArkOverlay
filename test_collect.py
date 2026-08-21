"""collect.py 순수 함수 테스트 (화면 없이 돈다)."""

import unittest

import numpy as np

from collect import (
    DONE,
    RECORD,
    WAIT,
    Session,
    band_region,
    frame_diff,
    is_blank,
    is_duplicate,
    parse_region,
)


class TestBandRegion(unittest.TestCase):
    def test_covers_buff_bar_at_1080p(self):
        # 지금 가진 캡처 두 장에서 버프창은 y 900~965에 있다.
        r = band_region(1920, 1080)
        self.assertLessEqual(r["y"], 900)
        self.assertGreaterEqual(r["y"] + r["h"], 965)

    def test_full_width(self):
        # 버프가 늘면 창이 좌우로 자라므로 폭은 자르지 않는다.
        r = band_region(2560, 1440)
        self.assertEqual(r["x"], 0)
        self.assertEqual(r["w"], 2560)

    def test_scales_with_resolution(self):
        small = band_region(1280, 720)
        big = band_region(2560, 1440)
        self.assertAlmostEqual(small["y"] / 720, big["y"] / 1440, places=2)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            band_region(0, 1080)
        with self.assertRaises(ValueError):
            band_region(1920, 1080, top=0.9, bottom=0.5)


class TestParseRegion(unittest.TestCase):
    def test_parses(self):
        self.assertEqual(parse_region("470,900,460,70"),
                         {"x": 470, "y": 900, "w": 460, "h": 70})

    def test_allows_spaces(self):
        self.assertEqual(parse_region(" 1, 2 ,3, 4 ")["w"], 3)

    def test_rejects_bad(self):
        for bad in ("1,2,3", "a,b,c,d", "1,2,0,4", "1,2,3,-1", ""):
            with self.assertRaises(ValueError, msg=bad):
                parse_region(bad)


class TestDuplicateDetection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.base = rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)

    def test_first_frame_always_saved(self):
        self.assertFalse(is_duplicate(None, self.base))
        self.assertEqual(frame_diff(None, self.base), float("inf"))

    def test_identical_is_duplicate(self):
        self.assertTrue(is_duplicate(self.base, self.base.copy()))

    def test_changed_digits_are_not_duplicate(self):
        # 몇 초 간격이면 초 표기가 바뀐다. 그 변화는 남겨야 한다.
        cur = self.base.copy()
        cur[10:20, 30:45] = 255
        self.assertFalse(is_duplicate(self.base, cur))

    def test_shape_change_is_not_duplicate(self):
        self.assertFalse(is_duplicate(self.base, self.base[:, :60]))


class TestBlank(unittest.TestCase):
    def test_loading_screen_is_blank(self):
        self.assertTrue(is_blank(np.zeros((40, 120, 3), np.uint8)))

    def test_game_frame_is_not_blank(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)
        self.assertFalse(is_blank(frame))


class TestSession(unittest.TestCase):
    def test_waits_until_detected(self):
        s = Session(minutes=3, warmup=300)
        s.begin(0.0)
        self.assertEqual(s.step(1.0, False), "건너뜀")
        self.assertEqual(s.step(2.0, False), "건너뜀")
        self.assertIs(s.state, WAIT)

    def test_starts_on_detection(self):
        s = Session(minutes=3, warmup=300)
        s.begin(0.0)
        s.step(1.0, False)
        self.assertEqual(s.step(2.0, True), "저장")
        self.assertIs(s.state, RECORD)
        self.assertEqual(s.trigger, "버프창 인식")

    def test_records_for_requested_duration(self):
        s = Session(minutes=3, warmup=300)
        s.begin(0.0)
        s.step(10.0, True)                    # 10초에 시작
        self.assertEqual(s.step(100.0, True), "저장")
        self.assertEqual(s.step(189.0, True), "저장")   # 179초 경과
        self.assertEqual(s.step(190.0, True), "종료")   # 180초 경과
        self.assertIs(s.state, DONE)

    def test_keeps_recording_after_buffs_disappear(self):
        # 버프가 잠깐 사라져도 촬영은 이어져야 한다. 버프 없는 배경도
        # 자료가 되고, 무엇보다 인식이 끊긴 장면이 제일 필요하다.
        s = Session(minutes=3, warmup=300)
        s.begin(0.0)
        s.step(1.0, True)
        self.assertEqual(s.step(20.0, False), "저장")

    def test_starts_anyway_after_warmup(self):
        # 인식이 안 되는 환경을 찍으려고 도는 도구다. 탐지 실패가
        # 곧 촬영 실패가 되면 목적을 잃는다.
        s = Session(minutes=3, warmup=60)
        s.begin(0.0)
        self.assertEqual(s.step(59.0, False), "건너뜀")
        self.assertEqual(s.step(60.0, False), "저장")
        self.assertIn("대기 시간 초과", s.trigger)

    def test_now_skips_waiting(self):
        s = Session(minutes=3, warmup=300, start_now=True)
        s.begin(0.0)
        self.assertIs(s.state, RECORD)
        self.assertEqual(s.step(0.0, False), "저장")

    def test_done_stays_done(self):
        s = Session(minutes=1, warmup=10, start_now=True)
        s.begin(0.0)
        self.assertEqual(s.step(61.0, True), "종료")
        self.assertEqual(s.step(62.0, True), "종료")


if __name__ == "__main__":
    unittest.main()
