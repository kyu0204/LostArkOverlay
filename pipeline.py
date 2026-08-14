"""
인식 파이프라인

캡처 한 프레임을 받아 Observation 목록으로 바꾼다.

    프레임
      ├─ 아이콘 행 → 그리드 → dHash+Hue 매칭 → buff_id
      └─ 텍스트 행 → 이진화 → 글자 분할 → 템플릿 매칭 → 남은 시간
                                    ↓
                          중심 x로 셀에 배정
                                    ↓
                      Observation(buff_id, remaining)
                                    ↓
                          BuffTracker.update()

설계 메모
---------
텍스트를 읽지 못한 셀은 remaining=None으로 넘긴다. BuffTracker가
카탈로그 duration으로 폴백하므로, 글자 템플릿이 비어 있어도
아이콘 인식만으로 동작한다.

그리드는 매 프레임 재검출하지 않는다. UI 배율은 세션 중에 바뀌지
않으므로 한 번 잡아 캐시하고, 인식이 급격히 나빠질 때만 다시 잡는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from buff_state import Observation
from grid_detect import Grid, detect_grid
from icon_match import IconBook, identify_cells
from text_parse import (
    GlyphBook,
    adaptive_binarize,
    assign_to_cells,
    group_glyphs,
    read_group,
    segment_glyphs,
    to_seconds,
)

# 아이콘 행 아래 어디까지가 지속시간 텍스트인지. 실측(배율 B) 기준
# 아이콘 26px, 텍스트 행이 그 아래 약 20px에 걸쳐 있다.
TEXT_ROW_RATIO = 0.8

# 오버플로 카운터의 buff_id. 이 칸은 버프가 아니라 '몇 개가 밀렸는지'다.
OVERFLOW_ID = "overflow_counter"


@dataclass
class FrameResult:
    observations: List[Observation] = field(default_factory=list)
    overflow: Optional[int] = None      # 밀려서 안 보이는 버프 수
    visible: int = 0                    # 화면에 보이는 버프 수
    grid: Optional[Grid] = None
    text_threshold: Optional[int] = None

    @property
    def consistent(self) -> Optional[bool]:
        """카운터가 있을 때만 의미가 있다.

        추적 수와 화면 수의 일관성 검사는 호출부에서
        BuffTracker의 활성 수와 함께 판단한다.
        """
        if self.overflow is None:
            return None
        return self.overflow >= 0


class Recognizer:
    def __init__(
        self,
        icons: Optional[IconBook] = None,
        glyphs: Optional[GlyphBook] = None,
        icon_row_h: Optional[int] = None,
        min_glyph_score: float = 0.86,
    ):
        self.icons = icons if icons is not None else IconBook.load()
        self.glyphs = glyphs if glyphs is not None else GlyphBook.load()
        self.icon_row_h = icon_row_h
        self.min_glyph_score = min_glyph_score
        self._grid: Optional[Grid] = None
        self._miss_streak = 0

    # ------------------------------------------------------------------
    # 그리드 캐시
    # ------------------------------------------------------------------

    def _split_rows(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """프레임을 아이콘 행과 텍스트 행으로 나눈다."""
        h = frame.shape[0]
        if self.icon_row_h is not None:
            cut = min(h, self.icon_row_h)
        elif self._grid is not None:
            cut = min(h, self._grid.cell + 2)
        else:
            cut = max(4, int(h * 0.55))
        return frame[:cut], frame[cut:]

    def ensure_grid(self, icon_row: np.ndarray, force: bool = False) -> Optional[Grid]:
        if self._grid is not None and not force:
            return self._grid
        g = detect_grid(icon_row)
        if g is not None:
            self._grid = g
            self._miss_streak = 0
        return self._grid

    def reset_grid(self) -> None:
        self._grid = None
        self._miss_streak = 0

    # ------------------------------------------------------------------
    # 인식
    # ------------------------------------------------------------------

    def read_frame(self, frame_bgr: np.ndarray, inset: int = 2) -> FrameResult:
        icon_row, text_row = self._split_rows(frame_bgr)

        grid = self.ensure_grid(icon_row)
        if grid is None:
            return FrameResult()

        matches = identify_cells(icon_row, grid.bounds(), self.icons, inset=inset)

        # 인식이 계속 비면 배율이 바뀌었을 수 있다. 그리드를 다시 잡는다.
        if not matches:
            self._miss_streak += 1
            if self._miss_streak >= 30:
                self.reset_grid()
        else:
            self._miss_streak = 0

        durations, thr = self._read_durations(text_row, grid)

        obs, overflow, visible = [], None, 0
        for ci, m in matches.items():
            if m.buff_id == OVERFLOW_ID:
                # 이 칸의 숫자는 남은 시간이 아니라 밀린 버프 수다
                raw = durations.get(ci)
                overflow = int(raw) if raw is not None else None
                continue
            visible += 1
            obs.append(Observation(m.buff_id, durations.get(ci)))

        return FrameResult(
            observations=obs,
            overflow=overflow,
            visible=visible,
            grid=grid,
            text_threshold=thr,
        )

    def _read_durations(
        self, text_row: np.ndarray, grid: Grid
    ) -> Tuple[Dict[int, float], Optional[int]]:
        """셀 인덱스 -> 남은 시간(초). 읽지 못한 셀은 빠진다."""
        import cv2

        if text_row.size == 0 or not self.glyphs.templates:
            return {}, None

        gray = cv2.cvtColor(text_row, cv2.COLOR_BGR2GRAY)
        bw, thr = adaptive_binarize(gray)
        if thr is None:
            return {}, None

        glyphs = segment_glyphs(bw)
        groups = group_glyphs(glyphs, max_gap=3)
        centers = [(l + r) / 2.0 for l, r in grid.bounds()]
        assigned = assign_to_cells(groups, centers, max_dist=grid.pitch * 0.9)

        out: Dict[int, float] = {}
        for ci, grp in assigned.items():
            text, _ = read_group(grp, self.glyphs, self.min_glyph_score)
            if text is None:
                continue
            sec = to_seconds(text)
            if sec is not None:
                out[ci] = sec
        return out, thr


# ----------------------------------------------------------------------
# 일관성 검사
# ----------------------------------------------------------------------

def consistency_gap(result: FrameResult, tracked: int) -> Optional[int]:
    """추적 중인 수와 화면 상태의 차이.

    0이면 일관됨. 음수면 놓친 버프가 있다는 뜻이고, 양수면 타이머가
    실제보다 오래 살아 있다는 뜻이다. 카운터가 없으면 판단할 수 없다.

    카운터 숫자가 '초과분'인지 '총합'인지는 아직 미확인이므로,
    초과분으로 가정한다. 확인되면 여기만 고치면 된다.
    """
    if result.overflow is None:
        return None
    return tracked - (result.visible + result.overflow)
