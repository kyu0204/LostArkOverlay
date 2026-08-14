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

    def test_buff_without_readable_time_is_still_listed(self):
        # 시간을 못 읽어도 '걸려 있다'는 사실은 알려줘야 한다.
        # 대신 남은 시간은 None으로 내보내 UI가 숫자를 지어내지 않게 한다.
        self.t.update([Observation("mystery", None)], now=0.0)
        snap = self.t.snapshot(0.0)
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0]["id"], "mystery")
        self.assertIsNone(snap[0]["remaining"])
        self.assertFalse(snap[0]["duration_known"])
        self.assertEqual(snap[0]["progress"], 0.0)

    def test_unknown_time_buff_sorts_below_known_ones(self):
        self.t.update(
            [Observation("mystery", None), Observation("emperor", 9.0)], now=0.0
        )
        # 같은 priority가 아니어도, 시간을 아는 쪽이 위에 와야 한다
        t2 = BuffTracker({})
        t2.update(
            [Observation("nope", None), Observation("yes", 5.0)], now=0.0
        )
        self.assertEqual([r["id"] for r in t2.snapshot(0.0)], ["yes", "nope"])


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


class TestFallbackDoesNotRefresh(unittest.TestCase):
    """폴백값은 갱신 근거가 아니다.

    글자 템플릿이 없으면 모든 관측이 remaining=None으로 들어온다.
    이때 카탈로그 duration으로 매 프레임 expire_at을 다시 잡으면
    계속 보이는 버프의 남은 시간이 영원히 줄지 않는다.
    """

    def setUp(self):
        self.t = BuffTracker(CATALOG)

    def test_continuous_observation_still_counts_down(self):
        self.t.update([Observation("emperor", None)], now=0.0)
        # 10Hz로 계속 보이는 상황
        for i in range(1, 51):
            self.t.update([Observation("emperor", None)], now=i * 0.1)
        # 5초 지났으면 12 - 5 = 7초여야 한다
        self.assertAlmostEqual(self.t.snapshot(5.0)[0]["remaining"], 7.0, places=3)

    def test_timer_actually_reaches_expiry_while_visible(self):
        # 계속 보이더라도 카탈로그 지속시간이 다하면 만료돼야 한다.
        # (얼어붙어 있으면 이 시점에 살아남는다)
        self.t.update([Observation("synergy", None)], now=0.0)   # 8초짜리
        for i in range(1, 80):
            self.t.update([Observation("synergy", None)], now=i * 0.1)
        self.assertAlmostEqual(self.t.snapshot(7.9)[0]["remaining"], 0.1, places=3)
        self.t.update([Observation("synergy", None)], now=8.0)
        self.assertNotIn("synergy", self.t.active_ids)

    def test_still_visible_after_expiry_is_retracked(self):
        # 만료됐는데도 화면에 남아 있다면 카탈로그 값이 틀렸거나 갱신된 것.
        # 아이콘이 보이는 이상 '없다'고 하는 편이 더 틀리므로 다시 잡는다.
        self.t.update([Observation("synergy", None)], now=0.0)
        self.t.update([Observation("synergy", None)], now=8.0)   # 만료
        self.assertNotIn("synergy", self.t.active_ids)
        self.t.update([Observation("synergy", None)], now=8.1)   # 여전히 보임
        self.assertIn("synergy", self.t.active_ids)

    def test_stays_observed_while_visible(self):
        # 타이머는 흐르되 신뢰도는 observed를 유지해야 한다
        self.t.update([Observation("emperor", None)], now=0.0)
        self.t.update([Observation("emperor", None)], now=3.0)
        snap = self.t.snapshot(3.0)[0]
        self.assertEqual(snap["confidence"], CONF_OBSERVED)
        self.assertAlmostEqual(snap["remaining"], 9.0)

    def test_measured_value_still_refreshes(self):
        # 실측이 들어오면 폴백과 달리 타이머를 덮어써야 한다
        self.t.update([Observation("emperor", None)], now=0.0)
        self.t.update([Observation("emperor", 12.0)], now=5.0)
        self.assertAlmostEqual(self.t.snapshot(5.0)[0]["remaining"], 12.0)


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


class TestUnknownDuration(unittest.TestCase):
    """남은 시간을 못 읽은 버프의 취급 (요구사항 4)."""

    def setUp(self):
        self.t = BuffTracker({})

    def test_stays_while_visible(self):
        now = 0.0
        for i in range(30):
            now = i * 0.1
            self.t.update([Observation("x", None)], now=now)
        self.assertIn("x", self.t.active_ids)
        self.assertIsNone(self.t.snapshot(now)[0]["remaining"])

    def test_dropped_as_soon_as_unobserved(self):
        # 타이머가 없으니 안 보이는 순간 띄울 근거가 사라진다
        self.t.update([Observation("x", None)], now=0.0)
        self.assertIn("x", self.t.active_ids)
        self.t.update([], now=0.1)
        self.assertNotIn("x", self.t.active_ids)

    def test_gets_timer_once_time_becomes_readable(self):
        # 처음엔 못 읽다가 나중에 읽히면 그때부터 타이머가 붙는다
        self.t.update([Observation("x", None)], now=0.0)
        self.assertIsNone(self.t.snapshot(0.0)[0]["remaining"])
        self.t.update([Observation("x", 8.0)], now=1.0)
        row = self.t.snapshot(1.0)[0]
        self.assertTrue(row["duration_known"])
        self.assertAlmostEqual(row["remaining"], 8.0)

    def test_known_timer_is_not_dropped_by_grace(self):
        # 실측 타이머가 있으면 안 보여도 만료 전까지 살아 있어야 한다
        # (미관측은 소멸 근거가 아니라는 원칙)
        self.t.update([Observation("x", 30.0)], now=0.0)
        self.t.update([], now=10.0)
        self.assertIn("x", self.t.active_ids)


class TestMeasurementSmoothing(unittest.TestCase):
    """읽은 값은 눈금이 굵은 관측이다 (요구사항 3).

    게임은 1초 이상을 정수로 표시하므로 같은 값이 1초 내내 읽힌다.
    매번 expire_at을 다시 잡으면 그동안 남은 시간이 줄지 않는다.
    """

    def setUp(self):
        self.t = BuffTracker({})

    def test_holds_at_lower_bound_while_display_holds(self):
        # 표기가 '4초'인 동안 실제 남은 시간은 4.0 이상이다.
        # 그 아래로 내려가면 화면이 말해주는 사실과 어긋난다.
        self.t.update([Observation("x", 4.0)], now=0.0)
        for i in range(1, 8):
            self.t.update([Observation("x", 4.0)], now=i * 0.1)
        rem = self.t.snapshot(0.7)[0]["remaining"]
        self.assertGreaterEqual(rem, 4.0)
        self.assertLess(rem, 5.0)

    def test_transition_pins_to_top_of_band(self):
        # '4초' -> '3초'로 바뀌는 순간을 봤다면 방금 4초 구간의 아래끝을
        # 지난 것이므로 남은 시간은 3.9다.
        self.t.update([Observation("x", 4.0)], now=0.0)
        self.t.update([Observation("x", 3.0)], now=0.5)
        self.assertAlmostEqual(self.t.snapshot(0.5)[0]["remaining"], 3.9, places=2)

    def test_free_runs_between_transitions(self):
        # 전환을 본 뒤에는 다음 전환까지 그대로 흘러야 한다
        self.t.update([Observation("x", 9.0)], now=0.0)
        self.t.update([Observation("x", 8.0)], now=1.0)   # 전환 관측 -> 8.9
        self.t.update([Observation("x", 8.0)], now=1.4)   # 같은 표기 유지
        self.assertAlmostEqual(self.t.snapshot(1.4)[0]["remaining"], 8.5, places=2)

    def test_new_reading_outside_tolerance_snaps(self):
        # 갱신되어 시간이 늘면 실측으로 맞춘다
        self.t.update([Observation("x", 4.0)], now=0.0)
        self.t.update([Observation("x", 12.0)], now=1.0)
        self.assertAlmostEqual(self.t.snapshot(1.0)[0]["remaining"], 12.0)

    def test_drifted_estimate_is_corrected(self):
        self.t.update([Observation("x", 10.0)], now=0.0)
        self.t.update([Observation("x", 3.0)], now=1.0)   # 예측 9.0과 크게 다름
        self.assertAlmostEqual(self.t.snapshot(1.0)[0]["remaining"], 3.0)

    def test_sub_second_uses_finer_tolerance(self):
        # 1초 미만은 0.1초 단위로 표시되므로 허용 오차도 작아야 한다
        self.t.update([Observation("x", 0.8)], now=0.0)
        self.t.update([Observation("x", 0.5)], now=0.1)   # 예측 0.7, 차이 0.2
        self.assertAlmostEqual(self.t.snapshot(0.1)[0]["remaining"], 0.5, places=2)
