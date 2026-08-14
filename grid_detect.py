"""
버프바 그리드 자동 검출

왜 필요한가
-----------
같은 1920x1080이라도 게임 UI 배율 설정에 따라 아이콘 크기가 달라진다.
실측 두 케이스:

    배율 A: 셀 21px, 피치 23.33
    배율 B: 셀 26px, 피치 29.0     (A의 약 1.24배)

테스터마다 배율이 다르므로 값을 하드코딩할 수 없다.
ROI만 잡으면 셀 크기와 피치를 자동으로 찾아낸다.

원리
----
버프 아이콘에는 뚜렷한 채색 테두리(녹색/주황/빨강)가 있다.
테두리 픽셀의 세로 방향 합을 x축 프로파일로 만들면 아이콘마다
봉우리가 서고, 봉우리 간 간격이 곧 피치다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Grid:
    left: int      # 첫 셀의 x
    cell: int      # 셀 한 변
    pitch: float   # 셀 + 간격
    count: int     # 검출된 셀 개수

    @property
    def gap(self) -> float:
        return self.pitch - self.cell

    def bounds(self) -> List[Tuple[int, int]]:
        return [
            (int(round(self.left + i * self.pitch)),
             int(round(self.left + i * self.pitch)) + self.cell)
            for i in range(self.count)
        ]

    def spec(self) -> str:
        """capture.py --grid 에 넣을 문자열."""
        return f"{self.cell}x{self.gap:.2f}+{self.left}"

    def __str__(self) -> str:
        return (f"cell={self.cell} pitch={self.pitch:.2f} "
                f"gap={self.gap:.2f} count={self.count} left={self.left}")


def border_mask(bgr: np.ndarray) -> np.ndarray:
    """아이콘 테두리로 보이는 픽셀. 녹색/주황/빨강 계열의 선명한 색.

    게임 배경은 대체로 어둡고 저채도라 밝기+채도 조건으로 상당히 걸러진다.
    """
    b = bgr[:, :, 0].astype(int)
    g = bgr[:, :, 1].astype(int)
    r = bgr[:, :, 2].astype(int)

    greenish = (g > 80) & (g > b + 30) & (g > r + 20)
    orangish = (r > 130) & (g > 70) & (b < 110) & (r > b + 60)
    reddish = (r > 110) & (r > g + 50) & (r > b + 50)
    return greenish | orangish | reddish


def _runs(flags, min_len: int = 1):
    """True가 연속된 구간 목록 [(start, end_inclusive), ...]"""
    out = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def _border_rows(mask, top_frac: float = 0.35):
    """아이콘의 상단/하단 테두리에 해당하는 행 인덱스.

    테두리 행은 아이콘 전체 폭에 걸쳐 마스크가 켜지므로 행 합이 크게 튄다.
    아이콘 내부 무늬는 이만큼 넓게 이어지지 않는다.
    """
    rowsum = mask.sum(axis=1).astype(float)
    if rowsum.max() <= 0:
        return []
    thr = rowsum.max() * top_frac
    cand = [i for i in range(len(rowsum)) if rowsum[i] >= thr]
    if not cand:
        return []
    # 상단 근처와 하단 근처에서 각각 가장 강한 행 하나씩
    h = len(rowsum)
    upper = [i for i in cand if i < h * 0.5]
    lower = [i for i in cand if i >= h * 0.5]
    picked = []
    if upper:
        picked.append(max(upper, key=lambda i: rowsum[i]))
    if lower:
        picked.append(max(lower, key=lambda i: rowsum[i]))
    return picked


def _pitch_from_lefts(lefts, tol: float = 0.15):
    """아이콘 좌표 목록에서 피치를 추정한다.

    반환: (pitch, 설명된 간격 수). 설명 수는 어느 행이 더 믿을 만한지
    고를 때 쓴다. 단순히 구간이 많은 행을 고르면 노이즈가 많은 행이
    뽑히므로, '주기로 설명되는 정도'를 봐야 한다.

    중간에 검출되지 않은 아이콘이 있어도(테두리 색이 달라 놓친 경우)
    간격이 피치의 정수배로 나타나므로 복원할 수 있다.
    """
    if len(lefts) < 2:
        return None, 0
    diffs = [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    diffs = [d for d in diffs if d >= 8]
    if not diffs:
        return None, 0

    best, best_score = None, -1
    for cand in sorted(set(diffs)):
        score = sum(
            1 for d in diffs
            if round(d / cand) >= 1 and abs(d - round(d / cand) * cand) <= cand * tol
        )
        # 같은 점수면 큰 후보를 택한다. 작은 값은 노이즈 간격일 때가 많다.
        if score > best_score or (score == best_score and best and cand > best):
            best, best_score = cand, score
    if best is None:
        return None, 0

    total, weight = 0.0, 0
    for d in diffs:
        k = round(d / best)
        if k >= 1 and abs(d - k * best) <= best * tol:
            total += d
            weight += k
    pitch = total / weight if weight else float(best)
    return pitch, best_score


def detect_grid(roi_bgr, min_cells: int = 2):
    """ROI에서 셀 크기와 피치를 추정한다.

    min_cells가 2인 이유: 전투 시작 직후처럼 버프가 1~2개뿐인 순간이
    실제로 있다. 3을 요구하면 그동안 아무것도 인식하지 못한다.
    아이콘 테두리는 신호가 뚜렷해 2개로도 피치를 잡을 수 있다.

    roi_bgr: 아이콘 행만 크롭한 이미지. 텍스트 행은 빼는 편이 정확하다.

    상단/하단 테두리 행을 먼저 찾고, 그 행에서 연속 구간을 세는 방식이다.
    아이콘 내부 무늬는 그만큼 넓게 이어지지 않으므로 자연히 걸러진다.
    """
    if roi_bgr.ndim != 3 or roi_bgr.shape[0] < 4:
        return None

    mask = border_mask(roi_bgr)
    rows = _border_rows(mask)
    if not rows:
        return None

    best = None
    for ry in rows:
        segs = _runs(mask[ry], min_len=6)
        if len(segs) < 2:
            continue

        # 부분적으로만 검출된 짧은 구간은 좌표를 오염시킨다.
        # 가장 긴 구간의 절반에 못 미치는 것은 버린다.
        longest = max(b - a + 1 for a, b in segs)
        segs = [(a, b) for a, b in segs if (b - a + 1) >= longest * 0.5]
        if len(segs) < 2:
            continue

        lefts = [a for a, _ in segs]
        pitch, explained = _pitch_from_lefts(lefts)
        if pitch is None or pitch < 8:
            continue

        widths = [b - a + 1 for a, b in segs if (b - a + 1) <= pitch * 1.05]
        if not widths:
            continue
        cell = int(round(float(np.median(widths))))

        cand = (explained, lefts[0], cell, pitch)
        if best is None or explained > best[0]:
            best = cand

    if best is None:
        return None

    _, left, cell, pitch = best
    cell = max(4, min(cell, int(round(pitch))))

    # 왼쪽으로 더 뻗을 수 있는지 (첫 아이콘을 놓쳤을 수 있다)
    while left - pitch >= -0.5:
        left = left - pitch
    left = int(round(max(0.0, left)))

    w = roi_bgr.shape[1]
    count = 0
    while int(round(left + count * pitch)) + cell <= w:
        count += 1
    if count < min_cells:
        return None

    return Grid(left=left, cell=int(cell), pitch=float(pitch), count=count)


def annotate(roi_bgr: np.ndarray, grid: Grid) -> np.ndarray:
    """검출 결과를 눈으로 확인하기 위한 오버레이."""
    import cv2

    out = roi_bgr.copy()
    h = out.shape[0]
    for i, (l, r) in enumerate(grid.bounds()):
        c = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.rectangle(out, (l, 0), (r - 1, h - 1), c, 1)
    return out
