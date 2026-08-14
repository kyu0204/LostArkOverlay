"""상태 머신 테스트. 게임 접속 없이 전부 검증 가능."""

import unittest

from buff_state import (
    CONF_OBSERVED,
    CONF_PREDICTED,
    BuffTracker,
    Observation,
)


CATALOG = {
    "emperor": {"name": "황제", "duration": 12.0, "priority": 1, "color": "#ffb300"},
    "judgment": {"name": "심판", "type": "instant"},
    "potion": {"name": "물약", "duration": 30.0, "priority": 5},
    "synergy": {"name": "시너지", "duration": 8.0, "priority": 2},
}


class TestBasics(unittest.TestCase):
    def setUp(self):
        self.t = BuffTracker(CATALOG)

    def test_observed_buff_is_tracked(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        snap = self.t.snapshot(0.0)
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0]["id"], "emperor")
        self.assertEqual(snap[0]["name"], "황제")
        self.assertAlmostEqual(snap[0]["remaining"], 12.0)

    def test_countdown(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.assertAlmostEqual(self.t.snapshot(5.0)[0]["remaining"], 7.0)

    def test_falls_back_to_catalog_duration(self):
        # 아이콘은 인식했으나 남은 시간 파싱 실패
        self.t.update([Observation("emperor", None)], now=0.0)
        self.assertAlmostEqual(self.t.snapshot(0.0)[0]["remaining"], 12.0)

    def test_instant_type_is_not_timed(self):
        self.t.update([Observation("judgment", None)], now=0.0)
        self.assertEqual(self.t.snapshot(0.0), [])

    def test_unknown_buff_still_tracked_with_observed_time(self):
        self.t.update([Observation("mystery", 5.0)], now=0.0)
        snap = self.t.snapshot(0.0)
        self.assertEqual(snap[0]["id"], "mystery")
        self.assertAlmostEqual(snap[0]["remaining"], 5.0)

    def test_unknown_buff_without_time_is_ignored(self):
        self.t.update([Observation("mystery", None)], now=0.0)
        self.assertEqual(self.t.snapshot(0.0), [])


class TestPushedOut(unittest.TestCase):
    """핵심 요구사항: 버프바에서 밀려 사라져도 만료 전까지 유지."""

    def setUp(self):
        self.t = BuffTracker(CATALOG)

    def test_survives_disappearing_from_observation(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        # 3초 시점에 밀려서 ROI 밖으로 나감 -> 관측 없음
        self.t.update([], now=3.0)
        self.assertIn("emperor", self.t.active_ids)
        # 10초에도 여전히 살아있어야 한다
        self.t.update([], now=10.0)
        self.assertIn("emperor", self.t.active_ids)
        self.assertAlmostEqual(self.t.snapshot(10.0)[0]["remaining"], 2.0)

    def test_expires_only_by_timer(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.update([], now=11.9)
        self.assertIn("emperor", self.t.active_ids)
        self.t.update([], now=12.0)
        self.assertNotIn("emperor", self.t.active_ids)

    def test_many_empty_frames_do_not_expire(self):
        self.t.update([Observation("potion", 30.0)], now=0.0)
        for i in range(1, 200):  # 10Hz로 20초간 계속 미관측
            self.t.update([], now=i * 0.1)
        self.assertIn("potion", self.t.active_ids)


class TestConfidence(unittest.TestCase):
    def setUp(self):
        self.t = BuffTracker(CATALOG, predict_after=0.5)

    def test_starts_observed(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.assertEqual(self.t.snapshot(0.0)[0]["confidence"], CONF_OBSERVED)

    def test_brief_miss_stays_observed(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.update([], now=0.3)  # 한 프레임 놓친 정도
        self.assertEqual(self.t.snapshot(0.3)[0]["confidence"], CONF_OBSERVED)

    def test_long_miss_becomes_predicted(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.update([], now=2.0)
        self.assertEqual(self.t.snapshot(2.0)[0]["confidence"], CONF_PREDICTED)

    def test_reobservation_restores_observed(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.update([], now=2.0)
        self.t.update([Observation("emperor", 9.0)], now=3.0)
        self.assertEqual(self.t.snapshot(3.0)[0]["confidence"], CONF_OBSERVED)


class TestRefresh(unittest.TestCase):
    """갱신 시 앞으로 당겨지므로 재관측으로 자연히 처리된다."""

    def setUp(self):
        self.t = BuffTracker(CATALOG)

    def test_refresh_extends_timer(self):
        self.t.update([Observation("synergy", 8.0)], now=0.0)
        self.t.update([], now=5.0)  # 밀려있는 동안
        # 6초에 재사용 -> 앞으로 당겨져 다시 관측됨
        self.t.update([Observation("synergy", 8.0)], now=6.0)
        self.assertAlmostEqual(self.t.snapshot(6.0)[0]["remaining"], 8.0)
        self.assertIn("synergy", self.t.active_ids)
        # 원래대로면 8초에 만료됐어야 하지만 갱신됐으므로 살아있다
        self.t.update([], now=9.0)
        self.assertIn("synergy", self.t.active_ids)

    def test_observed_value_overrides_drifted_estimate(self):
        # 추정 타이머가 어긋나 있어도 재관측되면 실측으로 교정
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.update([Observation("emperor", 3.0)], now=1.0)
        self.assertAlmostEqual(self.t.snapshot(1.0)[0]["remaining"], 3.0)


class TestOrderingAndFlush(unittest.TestCase):
    def setUp(self):
        self.t = BuffTracker(CATALOG)

    def test_sorted_by_priority_then_remaining(self):
        self.t.update(
            [
                Observation("potion", 30.0),   # priority 5
                Observation("synergy", 8.0),   # priority 2
                Observation("emperor", 12.0),  # priority 1
            ],
            now=0.0,
        )
        ids = [r["id"] for r in self.t.snapshot(0.0)]
        self.assertEqual(ids, ["emperor", "synergy", "potion"])

    def test_same_priority_sorted_by_remaining(self):
        t = BuffTracker({
            "a": {"name": "A", "duration": 10.0, "priority": 3},
            "b": {"name": "B", "duration": 4.0, "priority": 3},
        })
        t.update([Observation("a", 10.0), Observation("b", 4.0)], now=0.0)
        self.assertEqual([r["id"] for r in t.snapshot(0.0)], ["b", "a"])

    def test_flush_clears_everything(self):
        self.t.update([Observation("emperor", 12.0)], now=0.0)
        self.t.flush()
        self.assertEqual(self.t.active_ids, [])

    def test_event_shape_is_serializable(self):
        import json

        self.t.update([Observation("emperor", 12.0)], now=0.0)
        ev = self.t.event(0.0)
        self.assertEqual(ev["type"], "buff_update")
        json.dumps(ev)  # 예외 없이 직렬화되어야 한다


if __name__ == "__main__":
    unittest.main(verbosity=2)
