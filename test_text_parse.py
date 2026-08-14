"""지속시간 텍스트 파싱 테스트."""

import unittest

import numpy as np

from text_parse import (
    UNIT_MIN,
    UNIT_SEC,
    Glyph,
    GlyphBook,
    assign_to_cells,
    binarize,
    group_glyphs,
    normalize_glyph,
    read_group,
    segment_glyphs,
    to_seconds,
)


def box(x, w, h=8, fill=1):
    return Glyph(x, 0, w, h, np.full((h, w), fill, np.uint8))


class TestToSeconds(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(to_seconds(f"19{UNIT_SEC}"), 19.0)

    def test_minutes_rejected_by_default(self):
        # 추적 대상 버프는 모두 초 단위. '분'을 받아들이면 '초'와의
        # 오인이 그대로 잘못된 타이머가 된다.
        self.assertIsNone(to_seconds(f"36{UNIT_MIN}"))

    def test_minutes_when_explicitly_allowed(self):
        self.assertEqual(to_seconds(f"3{UNIT_MIN}", allow_minutes=True), 180.0)

    def test_absurd_value_rejected(self):
        # 숫자를 잘못 읽어 초 단위에 나올 수 없는 크기가 된 경우
        self.assertIsNone(to_seconds(f"9999{UNIT_SEC}"))

    def test_upper_bound_is_inclusive(self):
        from text_parse import MAX_SECONDS

        self.assertEqual(to_seconds(f"{int(MAX_SECONDS)}{UNIT_SEC}"), MAX_SECONDS)

    def test_decimal(self):
        self.assertAlmostEqual(to_seconds(f"0.3{UNIT_SEC}"), 0.3)

    def test_bare_number_treated_as_seconds(self):
        self.assertEqual(to_seconds("12"), 12.0)

    def test_garbage(self):
        for bad in ("", UNIT_SEC, "..", "1.2.3", "abc"):
            self.assertIsNone(to_seconds(bad), bad)

    def test_negative_rejected(self):
        self.assertIsNone(to_seconds(f"-5{UNIT_SEC}"))


class TestGrouping(unittest.TestCase):
    def test_splits_on_wide_gap(self):
        gs = [box(0, 6), box(7, 6), box(30, 6)]
        groups = group_glyphs(gs, max_gap=4)
        self.assertEqual([len(g) for g in groups], [2, 1])

    def test_single_group(self):
        gs = [box(0, 6), box(7, 6), box(14, 8)]
        self.assertEqual(len(group_glyphs(gs, max_gap=4)), 1)

    def test_empty(self):
        self.assertEqual(group_glyphs([]), [])


class TestSegmentation(unittest.TestCase):
    def test_finds_separated_marks(self):
        bw = np.zeros((10, 40), np.uint8)
        bw[2:9, 3:9] = 1
        bw[2:9, 15:21] = 1
        self.assertEqual(len(segment_glyphs(bw)), 2)

    def test_rejects_flooded_row(self):
        # 밝은 배경을 글자로 오인한 상황. 빈 목록이어야 한다.
        bw = np.ones((10, 40), np.uint8)
        self.assertEqual(segment_glyphs(bw), [])

    def test_merges_broken_hangul_strokes(self):
        # '초'처럼 획이 세로로 끊긴 글자. x가 겹치므로 하나로 합쳐진다.
        bw = np.zeros((12, 20), np.uint8)
        bw[1:3, 4:11] = 1     # 윗획
        bw[5:11, 4:11] = 1    # 아랫부분
        gs = segment_glyphs(bw)
        self.assertEqual(len(gs), 1)
        self.assertGreaterEqual(gs[0].h, 9)

    def test_keeps_decimal_point_separate(self):
        # 소수점은 인접 숫자와 x가 겹치지 않아 따로 남아야 한다
        bw = np.zeros((12, 30), np.uint8)
        bw[2:10, 2:8] = 1     # 숫자
        bw[8:10, 10:12] = 1   # 소수점
        bw[2:10, 14:20] = 1   # 숫자
        self.assertEqual(len(segment_glyphs(bw)), 3)

    def test_splits_touching_glyphs(self):
        # 두 글자가 1px로 닿아 하나의 성분이 된 경우
        bw = np.zeros((12, 24), np.uint8)
        bw[2:10, 2:8] = 1
        bw[6:7, 8:9] = 1      # 얇게 연결
        bw[2:10, 9:17] = 1
        gs = segment_glyphs(bw)
        self.assertEqual(len(gs), 2)


class TestAssign(unittest.TestCase):
    def setUp(self):
        # 셀 26, 피치 29 -> 중심 13, 42, 71, 100
        self.centers = [13.0, 42.0, 71.0, 100.0]

    def test_exact_centers(self):
        groups = [[box(8, 10)], [box(37, 10)]]
        got = assign_to_cells(groups, self.centers, max_dist=26.0)
        self.assertEqual(sorted(got), [0, 1])

    def test_wide_text_overflowing_cell(self):
        # '53분'처럼 아이콘보다 넓은 텍스트. 중심이 셀 안이면 배정된다.
        groups = [[box(56, 30)]]   # 중심 71 -> 셀 2
        got = assign_to_cells(groups, self.centers, max_dist=26.0)
        self.assertEqual(list(got), [2])

    def test_one_group_per_cell(self):
        # 두 그룹이 한 셀에 몰리면 가까운 쪽만 배정된다
        groups = [[box(10, 6)], [box(14, 6)]]
        got = assign_to_cells(groups, self.centers, max_dist=26.0)
        self.assertEqual(len(got), 2)
        self.assertIn(0, got)

    def test_far_group_dropped(self):
        groups = [[box(400, 10)]]
        self.assertEqual(assign_to_cells(groups, self.centers, max_dist=26.0), {})


class TestGlyphBook(unittest.TestCase):
    def setUp(self):
        self.book = GlyphBook()
        a = np.zeros((12, 8), np.uint8); a[2:10, 1:7] = 1
        b = np.zeros((12, 8), np.uint8); b[2:10, 1:4] = 1
        self.a, self.b = a, b
        self.book.add("0", a)
        self.book.add("1", b)

    def test_exact_match(self):
        label, score = self.book.match(self.a)
        self.assertEqual(label, "0")
        self.assertAlmostEqual(score, 1.0)

    def test_distinguishes(self):
        self.assertEqual(self.book.match(self.b)[0], "1")

    def test_empty_book(self):
        self.assertEqual(GlyphBook().match(self.a), (None, 0.0))

    def test_scale_invariant(self):
        import cv2

        big = cv2.resize(self.a * 255, (16, 24), interpolation=cv2.INTER_NEAREST)
        label, score = self.book.match((big > 127).astype(np.uint8))
        self.assertEqual(label, "0")
        self.assertGreater(score, 0.9)

    def test_roundtrip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "g.json"
            self.book.save(p)
            loaded = GlyphBook.load(p)
        self.assertEqual(set(loaded.templates), {"0", "1"})
        self.assertEqual(loaded.match(self.a)[0], "0")

    def test_load_missing_file(self):
        from pathlib import Path

        self.assertEqual(GlyphBook.load(Path("/nonexistent/g.json")).templates, {})


class TestReadGroup(unittest.TestCase):
    def setUp(self):
        self.book = GlyphBook()
        self.digits = {}
        for i, ch in enumerate("0123456789"):
            img = np.zeros((12, 8), np.uint8)
            # 숫자마다 확실히 다른 패턴을 심는다
            img[2:10, 1:7] = 1
            img[3 + (i % 5), 1:7] = 0
            img[2:10, 1 + (i % 3)] = 0
            self.digits[ch] = img
            self.book.add(ch, img)
        sec = np.zeros((12, 8), np.uint8); sec[1:11, 0:8] = 1
        self.book.add(UNIT_SEC, sec)
        self.sec = sec

    def test_reads_sequence(self):
        grp = [
            Glyph(0, 0, 8, 12, self.digits["1"]),
            Glyph(9, 0, 8, 12, self.digits["9"]),
            Glyph(18, 0, 8, 12, self.sec),
        ]
        text, score = read_group(grp, self.book, min_score=0.9)
        self.assertEqual(text, f"19{UNIT_SEC}")
        self.assertAlmostEqual(score, 1.0)

    def test_rejects_low_confidence(self):
        noise = np.zeros((12, 8), np.uint8)
        noise[::2, ::2] = 1
        text, _ = read_group([Glyph(0, 0, 8, 12, noise)], self.book, min_score=0.95)
        self.assertIsNone(text)


class TestAdaptiveBinarize(unittest.TestCase):
    def test_dark_background(self):
        from text_parse import adaptive_binarize

        g = np.full((16, 200), 30, np.uint8)
        g[4:12, 10:16] = 220          # 밝은 글자
        g[4:12, 30:36] = 220
        bw, thr = adaptive_binarize(g)
        self.assertIsNotNone(thr)
        self.assertGreater(bw.sum(), 0)
        self.assertLess(bw.mean(), 0.35)

    def test_bright_background(self):
        from text_parse import adaptive_binarize

        # 배경이 밝고 글자가 더 밝은 경우에도 비율로 임계값을 찾는다
        g = np.full((16, 200), 150, np.uint8)
        g[4:12, 10:16] = 245
        g[4:12, 30:36] = 245
        bw, thr = adaptive_binarize(g)
        self.assertIsNotNone(thr)
        self.assertLess(bw.mean(), 0.35)

    def test_no_text_returns_none(self):
        from text_parse import adaptive_binarize

        g = np.full((16, 200), 128, np.uint8)
        bw, thr = adaptive_binarize(g)
        self.assertIsNone(thr)
        self.assertEqual(bw.sum(), 0)


class TestBinarize(unittest.TestCase):
    def test_fixed_threshold(self):
        g = np.array([[10, 200], [150, 50]], np.uint8)
        self.assertTrue((binarize(g, 100) == np.array([[0, 1], [1, 0]])).all())

    def test_otsu_runs(self):
        g = (np.random.default_rng(0).integers(0, 255, (16, 64))).astype(np.uint8)
        self.assertEqual(binarize(g).shape, g.shape)


class TestNormalize(unittest.TestCase):
    def test_fixed_shape(self):
        from text_parse import GLYPH_H, GLYPH_W

        out = normalize_glyph(np.ones((7, 3), np.uint8))
        self.assertEqual(out.shape, (GLYPH_H, GLYPH_W))
        self.assertTrue(set(np.unique(out)) <= {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
