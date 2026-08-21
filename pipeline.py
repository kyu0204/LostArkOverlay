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
    binarize,
    group_glyphs,
    read_group,
    segment_glyphs,
    text_color_mask,
    text_tophat_mask,
    to_seconds,
)

# 아이콘 행 아래 어디까지를 지속시간 텍스트로 볼지. 셀 크기 대비 비율.
# 실측: 아이콘 26px에 글자 높이 8px(약 0.31배). 여유를 둬 0.6으로 잡되,
# 무한정 아래까지 보지는 않는다. ROI를 넉넉히 잡으면 HP 바가 딸려 들어와
# 글자로 오인되기 때문이다.
TEXT_ROW_RATIO = 0.6

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
        geometry: Optional[dict] = None,
    ):
        self.icons = icons if icons is not None else IconBook.load()
        self.glyphs = glyphs if glyphs is not None else GlyphBook.load()
        self.icon_row_h = icon_row_h
        self.min_glyph_score = min_glyph_score
        self._grid: Optional[Grid] = None
        self._miss_streak = 0
        # 아이콘 행이 ROI 안에서 몇 px 아래부터 시작하는지.
        # 사람이 잡은 ROI는 정확할 수 없으므로 인식 결과로 맞춘다.
        self._icon_top = 0
        self._align_cooldown = 0
        self._calibrated = False
        # 셀 경계의 가로 위상(left를 pitch로 나눈 나머지).
        # 피치는 안정적으로 잡히지만 어디서 시작하는지는 ROI 왼쪽에 있는
        # 다른 UI에 낚일 수 있다. 한 번 맞춰두고 이후 검출에 강제한다.
        self._phase: Optional[float] = None
        self._geom_locked = False
        if geometry:
            self.apply_geometry(geometry)

    # ------------------------------------------------------------------
    # 기하 저장/복원
    # ------------------------------------------------------------------

    def apply_geometry(self, geom: dict) -> None:
        """저장해 둔 기하를 그대로 쓴다. 측정 단계를 건너뛴다.

        버프가 적을 때 켜면 기하를 잘못 잡는다(실측: 버프 2개에서
        pitch 14). 전투 전에 켜는 것은 흔한 일이므로, 잘 측정된 값을
        저장해 두고 그대로 쓰는 편이 훨씬 안전하다.
        """
        pitch, cell = geom.get("pitch"), geom.get("cell")
        if not pitch or not cell:
            return
        self._icon_top = int(geom.get("icon_top", 0))
        self._phase = geom.get("phase")
        count = int(geom.get("count", 0)) or 2
        self._grid = Grid(
            left=int(geom.get("left", 0)), cell=int(cell),
            pitch=float(pitch), count=count,
        )
        self._calibrated = True     # 저장값이 있으면 시작 보정을 하지 않는다
        self._geom_locked = True    # 매 프레임 재검출이 덮어쓰지 못하게 한다

    def geometry(self) -> Optional[dict]:
        """지금 쓰고 있는 기하. 저장해 두면 다음 실행에 재사용한다."""
        if self._grid is None:
            return None
        return {
            "pitch": round(float(self._grid.pitch), 3),
            "cell": int(self._grid.cell),
            "left": int(self._grid.left),
            "count": int(self._grid.count),
            "icon_top": int(self._icon_top),
            "phase": None if self._phase is None else round(float(self._phase), 3),
        }

    # ------------------------------------------------------------------
    # 그리드 캐시
    # ------------------------------------------------------------------

    def _split_rows(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """프레임을 아이콘 행과 텍스트 행으로 나눈다.

        시작 위치(_icon_top)는 ROI 맨 위가 아니라 검출된 값이다.
        ROI를 사람이 잡으므로 몇 px 어긋나기 마련인데, dHash는 ±1px까지만
        견딘다. 3~4px만 밀려도 인식이 통째로 죽는다(실측: 수련장 캡처가
        4px 어긋나 0개였다). 어긋남은 recalibrate()가 바로잡는다.

        높이 기준은 검출된 셀 크기다. ROI 높이 비율로 자르면 ROI를
        넉넉히 잡을수록 자르는 위치가 밀린다. 아이콘이 잘리면 해시가
        흔들리고, 텍스트 행이 길어지면 HP 바가 글자로 들어온다.
        """
        h = frame.shape[0]
        top = max(0, min(self._icon_top, h - 8))

        if self.icon_row_h is not None:
            ih = self.icon_row_h
        elif self._grid is not None:
            ih = self._grid.cell + 2
        else:
            # 셀 크기를 아직 모르는 첫 프레임. 아이콘을 자르지 않는 것이
            # 우선이므로 넉넉히 두고, 그리드를 잡은 뒤 다시 나눈다.
            ih = max(4, int((h - top) * 0.55))
        ih = max(4, min(ih, h - top))

        text_start = top + ih
        text_h = h - text_start
        if self._grid is not None:
            text_h = min(text_h, max(4, int(self._grid.cell * TEXT_ROW_RATIO)))
        return frame[top:text_start], frame[text_start:text_start + max(0, text_h)]

    def ensure_grid(self, icon_row: np.ndarray, force: bool = False) -> Optional[Grid]:
        if self._grid is not None and not force:
            return self._grid
        g = detect_grid(icon_row)
        if g is not None:
            self._grid = self._apply_phase(g)
        # _miss_streak은 '아이콘을 못 맞힌 연속 프레임'이다. 그리드 검출
        # 성공과는 별개이므로 여기서 건드리지 않는다. 매 프레임 재검출로
        # 바꾼 뒤 여기서 초기화하면 카운터가 영영 쌓이지 않는다.
        return self._grid

    def reset_grid(self) -> None:
        self._grid = None
        self._miss_streak = 0
        self._geom_locked = False

    @staticmethod
    def _fit_count(grid: Grid, width: int) -> Grid:
        """ROI 폭 안에 들어가는 칸 수로 맞춘다. 나머지 기하는 그대로."""
        count = 0
        while int(round(grid.left + count * grid.pitch)) + grid.cell <= width:
            count += 1
        return Grid(left=grid.left, cell=grid.cell, pitch=grid.pitch,
                    count=max(1, count))

    # ------------------------------------------------------------------
    # 세로 정렬
    # ------------------------------------------------------------------

    def _apply_phase(self, grid: Grid) -> Grid:
        """검출된 그리드의 셀 경계를 보정된 위상에 맞춘다.

        피치는 안정적으로 잡히지만 '어디서 시작하는지'는 ROI 왼쪽에 있는
        다른 UI나 밝은 배경에 낚인다(실측: 수련장 캡처에서 첫 셀이 42px
        밀려 인식 0개). 버프바는 오른쪽 끝이 고정이고 왼쪽으로 늘어나므로
        위상은 세션 내내 상수다. 한 번 맞춰두고 이후 검출에 강제한다.
        """
        if self._phase is None:
            return grid
        k = round((grid.left - self._phase) / grid.pitch)
        left = self._phase + k * grid.pitch
        while left < -0.5:
            left += grid.pitch
        return Grid(
            left=int(round(left)), cell=grid.cell, pitch=grid.pitch, count=grid.count
        )

    def _sweep(self, frame_bgr, saved_top: int, max_off: int, shift: int):
        """세로 위치 x 가로 위상을 훑어 가장 많이 맞는 조합을 찾는다.

        가로 위상은 성기게(3px) 본다. 정밀한 값은 호출부가 한 번 더
        좁혀 찾는다. 전 조합을 촘촘히 보면 1초 가까이 걸린다.
        """
        best_top, best_dx, best_n = saved_top, 0.0, -1
        for off in range(0, max_off + 1):
            self._icon_top, self._grid = off, None
            icon_row, _ = self._split_rows(frame_bgr)
            if self.ensure_grid(icon_row, force=True) is None:
                continue
            icon_row, _ = self._split_rows(frame_bgr)
            g = self._grid
            for dx in range(0, max(1, int(round(g.pitch))), 3):
                shifted = Grid(
                    left=g.left - dx, cell=g.cell, pitch=g.pitch, count=g.count
                )
                if shifted.left < -g.pitch:
                    break
                n = len(identify_cells(
                    icon_row, shifted.bounds(), self.icons, inset=2, shift=shift
                ))
                if n > best_n:
                    best_top, best_dx, best_n = off, float(dx), n
        return best_top, best_dx, best_n

    def recalibrate(self, frame_bgr: np.ndarray, max_off: int = 10) -> int:
        """아이콘 행의 시작 위치를 인식 결과로 찾는다.

        경계(수평 그래디언트)만 보고 고르면 테두리가 2px 두께라 ±2px
        모호함이 남는데, dHash는 그만큼도 못 견딘다. 그래서 후보마다
        실제로 매칭해 보고 가장 많이 맞는 위치를 고른다.

        아이콘 DB가 비어 있으면 판단 근거가 없으므로 건드리지 않는다.
        """
        if not self.icons.entries:
            return self._icon_top

        saved_top, saved_grid, saved_phase = self._icon_top, self._grid, self._phase
        self._phase = None          # 보정 중에는 이전 위상을 강제하지 않는다

        best_top, best_dx, best_n = self._sweep(frame_bgr, saved_top, max_off, 0)
        if best_n <= 0:
            # 시프트 없이 못 찾았다. ±1px 여유를 켜고 다시 본다.
            # 등록 당시와 크롭이 미세하게 달라지면 그 여유가 있어야
            # 매칭된다. 비용이 9배라 빠른 경로를 먼저 쓴다.
            best_top, best_dx, best_n = self._sweep(frame_bgr, saved_top, max_off, 1)
            fine_shift = 1
        else:
            fine_shift = 0

        if best_n <= 0:
            # 어느 위치에서도 못 맞혔다. 근거가 없으니 원래대로 되돌린다.
            self._icon_top, self._grid, self._phase = saved_top, saved_grid, saved_phase
            return self._icon_top

        # 고른 세로 위치에서 가로 위상만 1px 단위로 다시 훑는다.
        self._icon_top, self._grid = best_top, None
        icon_row, _ = self._split_rows(frame_bgr)
        g0 = self.ensure_grid(icon_row, force=True)
        if g0 is not None:
            icon_row, _ = self._split_rows(frame_bgr)
            lo, hi = int(best_dx) - 3, int(best_dx) + 3
            for dx in range(max(0, lo), hi + 1):
                shifted = Grid(
                    left=g0.left - dx, cell=g0.cell, pitch=g0.pitch, count=g0.count
                )
                if shifted.left < -g0.pitch:
                    break
                n = len(identify_cells(
                    icon_row, shifted.bounds(), self.icons, inset=2, shift=fine_shift
                ))
                if n > best_n:
                    best_dx, best_n = float(dx), n

        self._grid = None
        icon_row, _ = self._split_rows(frame_bgr)
        g = self.ensure_grid(icon_row, force=True)
        if g is None:
            self._icon_top, self._grid, self._phase = saved_top, saved_grid, saved_phase
            return self._icon_top

        # 찾은 위상을 고정한다. 우측이 고정이므로 세션 내내 유효하다.
        self._phase = (g.left - best_dx) % g.pitch
        self._grid = self._apply_phase(g)
        return self._icon_top

    # ------------------------------------------------------------------
    # 인식
    # ------------------------------------------------------------------

    def read_frame(self, frame_bgr: np.ndarray, inset: int = 2) -> FrameResult:
        icon_row, text_row = self._split_rows(frame_bgr)

        # 매 프레임 다시 잡되, 실패하면 이전 값을 유지한다(ensure_grid가
        # 성공했을 때만 덮어쓴다). 캐시의 원래 목적 - 버프가 1개만 남으면
        # 피치를 추정할 수 없어 인식이 끊긴다 - 은 그대로 지키면서,
        # 버프바가 가로로 밀렸을 때 좌표가 굳어버리는 문제를 없앤다.
        #
        # 좌표가 굳으면 ±1px 밖에서는 매칭이 무너진다(실측: 2px 밀림에
        # 14개 -> 2개, 3px 밀림에 0개). 게다가 재검출 조건이 '연속 미인식'
        # 이라, 일부만 맞는 상태에서는 연속이 끊겨 영영 복구되지 않았다.
        # 비용은 프레임당 0.23ms로 전체의 1.3% 수준이라 캐시를 고집할
        # 이유가 없다.
        if self._geom_locked and self._grid is not None:
            # 저장된 기하를 쓴다. ROI 폭에 맞춰 칸 수만 다시 센다.
            grid = self._fit_count(self._grid, frame_bgr.shape[1])
            self._grid = grid
        else:
            grid = self.ensure_grid(icon_row, force=True)
            if grid is None:
                return FrameResult()

        # 셀 크기를 알았으니 그 기준으로 다시 나눈다. 첫 프레임의
        # 임시 분할과 이후 프레임의 분할이 어긋나면, 등록해 둔 아이콘/
        # 글자 템플릿과 크롭이 달라져 매칭이 통째로 실패한다.
        icon_row, text_row = self._split_rows(frame_bgr)

        # 첫 프레임에 한 번 세로 정렬을 맞춘다. ROI는 사람이 잡으므로
        # 몇 px 어긋나 있기 마련이고, 그 어긋남은 세션 내내 그대로다.
        # 일부만 맞는 상태(예: 14개 중 7개)는 '실패'로 안 보여서 아래
        # 미인식 경로가 영영 안 걸리므로, 시작할 때 한 번은 훑어야 한다.
        if not self._calibrated and self.icons.entries:
            self._calibrated = True
            self.recalibrate(frame_bgr)
            grid = self._grid
            if grid is None:
                return FrameResult()
            icon_row, text_row = self._split_rows(frame_bgr)

        matches = identify_cells(icon_row, grid.bounds(), self.icons, inset=inset)

        # 복구 판단은 '버프를 맞혔는가'로 한다. matches에는 오버플로
        # 카운터도 들어오는데, 그 `+` 아이콘은 정렬이 어긋난 자리에서도
        # 곧잘 매칭된다. matches가 비었는지로만 보면 카운터 하나 때문에
        # 연속 카운터가 영영 쌓이지 않아 복구가 안 된다(실측: 잘못된
        # 위상이 60프레임 내내 고정됐다).
        buff_hits = sum(1 for m in matches.values() if m.buff_id != OVERFLOW_ID)

        # 하나도 못 맞히면 세로 정렬이 어긋났을 수 있다. 몇 프레임 지켜본 뒤
        # 한 번만 훑는다(매 프레임 훑기엔 비싸다). 맞는 게 하나라도 있으면
        # 정렬은 옳다는 뜻이므로 건드리지 않는다.
        if not buff_hits:
            self._miss_streak += 1
            if self._align_cooldown > 0:
                self._align_cooldown -= 1
            elif self._miss_streak >= 10 and self.icons.entries:
                self._align_cooldown = 30
                self.recalibrate(frame_bgr)
                # 보정 뒤에는 세로 오프셋이 그대로여도 가로 위상만 바뀌었을
                # 수 있다. '오프셋이 변했는지'로 판단하면 위상만 어긋난
                # 경우를 놓쳐 영영 복구되지 않는다. 항상 다시 계산한다.
                if self._grid is not None:
                    grid = self._grid
                    icon_row, text_row = self._split_rows(frame_bgr)
                    matches = identify_cells(
                        icon_row, grid.bounds(), self.icons, inset=inset
                    )
                    buff_hits = sum(
                        1 for m in matches.values() if m.buff_id != OVERFLOW_ID
                    )
                    if buff_hits:
                        self._miss_streak = 0
            if not buff_hits and self._miss_streak >= 30:
                # 저장된 기하가 실제와 어긋났을 수 있다(UI 배율 변경 등).
                # 잠금을 풀어 다시 측정하게 한다.
                self._calibrated = False
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

    def cell_glyph_candidates(
        self, text_row: np.ndarray, grid: Grid
    ) -> Dict[int, List[Tuple[list, int]]]:
        """셀 인덱스 -> [(글자 그룹, 임계값), ...] 우선순위 순 후보.

        **템플릿 수집 도구도 반드시 이 메서드를 거쳐야 한다.**
        수집 때와 인식 때의 크롭·이진화가 조금이라도 다르면 글자 모양이
        미묘하게 달라져 매칭이 통째로 실패한다. 아이콘 등록에서 같은
        문제를 이미 두 번 겪었다.

        후보를 두 가지 방식으로 만드는 이유:

        1. 행 전체 한 장 - 이웃 글자와의 경계가 정확하다. 배경이 고른
           장면(일반 레이드/던전)에서 가장 잘 맞는다.
        2. 칸별 국소 임계값 - 폭발 이펙트처럼 화면 일부만 밝을 때
           1번은 그 구간이 통째로 뭉개진다. 한 칸 주변만 보면 밝기
           범위가 좁아 임계값이 제대로 잡힌다. 대신 창에 이웃 글자가
           걸쳐 들어올 수 있어 경계는 1번보다 부정확하다.

        어느 쪽이 맞는지는 칸마다 다르므로 둘 다 내고, 호출부가 먼저
        읽히는 것을 쓴다.
        """
        import cv2

        out: Dict[int, List[Tuple[list, int]]] = {}
        if text_row.size == 0:
            return out

        gray = cv2.cvtColor(text_row, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1]
        bounds = grid.bounds()
        centers = [(l + r) / 2.0 for l, r in bounds]

        # (0) 글자 색 마스크. 밝기만 보는 이진화와 달리 밝은 배경에
        # 흔들리지 않아 가장 정확하다. 배경색이 글자색과 겹치면
        # 마스크가 가득 차는데, 그때는 segment_glyphs가 빈 결과를
        # 돌려주므로 아래 밝기 기반 후보로 자연히 넘어간다.
        color_bw = text_color_mask(text_row)
        groups = group_glyphs(segment_glyphs(color_bw), max_gap=3)
        if groups:
            assigned = assign_to_cells(groups, centers, max_dist=grid.pitch * 0.9)
            for ci, grp in assigned.items():
                out.setdefault(ci, []).append((grp, -2))   # -2 = 색 마스크

        # (0-b) 국소 대비(top-hat). 배경이 밝아 색 마스크가 무너지는
        # 장면(수련장 금색 바닥)에서도 '옆보다 밝은 획'은 남는다.
        hat_bw = text_tophat_mask(text_row)
        groups = group_glyphs(segment_glyphs(hat_bw), max_gap=3)
        if groups:
            assigned = assign_to_cells(groups, centers, max_dist=grid.pitch * 0.9)
            for ci, grp in assigned.items():
                out.setdefault(ci, []).append((grp, -3))   # -3 = top-hat

        # (1) 행 전체
        bw, thr = adaptive_binarize(gray)
        if thr is not None:
            groups = group_glyphs(segment_glyphs(bw), max_gap=3)
            assigned = assign_to_cells(groups, centers, max_dist=grid.pitch * 0.9)
            for ci, grp in assigned.items():
                out.setdefault(ci, []).append((grp, thr))

        # (2) 칸별 국소 임계값.
        # 표기가 셀보다 넓을 수 있어(예: '53분'=30px > 26px) 창을 넉넉히 연다.
        half = grid.pitch * 0.75
        for ci, (l, r) in enumerate(bounds):
            cx = centers[ci]
            x0, x1 = max(0, int(cx - half)), min(width, int(cx + half))
            if x1 - x0 < 4:
                continue
            bwc, thrc = adaptive_binarize(gray[:, x0:x1])
            if thrc is None:
                continue
            groups = group_glyphs(segment_glyphs(bwc), max_gap=3)
            if not groups:
                continue
            # 창에 이웃 표기가 함께 들어올 수 있다. 중심에 가장 가까운 것만.
            grp = min(groups, key=lambda g: abs(x0 + (g[0].x + g[-1].right) / 2.0 - cx))
            if abs(x0 + (grp[0].x + grp[-1].right) / 2.0 - cx) > grid.pitch * 0.5:
                continue
            out.setdefault(ci, []).append((grp, thrc))
        return out

    def _read_durations(
        self, text_row: np.ndarray, grid: Grid
    ) -> Tuple[Dict[int, float], Optional[int]]:
        """셀 인덱스 -> 남은 시간(초). 읽지 못한 셀은 빠진다.

        칸마다 후보가 여럿이면 먼저 읽히는 것을 쓴다. 어느 이진화가
        맞는지는 그 칸의 배경 밝기에 달려 있어 미리 알 수 없다.
        """
        if not self.glyphs.templates:
            return {}, None

        out: Dict[int, float] = {}
        last_thr: Optional[int] = None
        for ci, cands in self.cell_glyph_candidates(text_row, grid).items():
            for grp, thr in cands:
                text, _ = read_group(grp, self.glyphs, self.min_glyph_score)
                if text is None:
                    continue
                # 단위가 없으면 뒷글자를 놓친 부분 판독이다.
                # 받아들이면 '15분'이 15초가 되는 식의 조용한 오독이 된다.
                sec = to_seconds(text, require_unit=True)
                if sec is not None:
                    out[ci] = sec
                    last_thr = thr
                    break
        return out, last_thr


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
