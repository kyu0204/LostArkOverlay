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

import cv2
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


def pitch_by_autocorr(roi_bgr, lo: int = 12, hi: int = 60):
    """세로 경계의 주기를 자기상관으로 찾는다. 반환: (피치, 신뢰도).

    `border_mask`는 색 임계값을 쓰므로 밝은 배경에서 경계에 아슬아슬하게
    걸친다. 실측: 같은 화면인데 파일로 저장한 것과 화면에서 다시 캡처한
    것의 픽셀 평균차가 6밖에 안 되는데도, 한쪽은 피치 28.8을, 다른 쪽은
    84.6을 내놨다.

    아이콘 좌우 테두리는 일정 간격으로 세로 경계를 만든다. 그 간격은
    색이 아니라 '주기'라서 밝기가 변해도 남는다.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    prof = prof - prof.mean()
    if prof.std() < 1e-6:
        return None, 0.0
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return None, 0.0
    k = int(np.argmax(ac[lo:hi])) + lo
    return float(k), float(ac[k])


def _edge_profile(band_bgr: np.ndarray) -> np.ndarray:
    """열별 세로 경계 세기. 아이콘 좌우 테두리에서 봉우리가 선다."""
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)


def _span_at_pitch(prof: np.ndarray, pitch: float, min_cells: int = 3):
    """주기가 실제로 이어지는 가로 구간을 찾는다. 반환: (x_left, x_right).

    버프바는 오른쪽 끝이 고정이고 왼쪽으로 늘어난다. 그래서 가장 강한
    봉우리에서 출발해 한 칸씩 좌우로 뻗으며, 그 자리에 봉우리가 있는지
    확인한다. 화면에는 스킬바나 파티창처럼 다른 주기 구조도 있으므로
    '주기가 끊기지 않는 구간'만 취해야 버프바만 남는다.
    """
    if pitch < 4 or len(prof) < pitch * min_cells:
        return None

    smooth = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 1.0).ravel()
    thr = float(np.percentile(smooth, 75))
    tol = max(2, int(round(pitch * 0.25)))

    def supported(x: float) -> bool:
        a, b = int(round(x)) - tol, int(round(x)) + tol + 1
        a, b = max(0, a), min(len(smooth), b)
        return a < b and float(smooth[a:b].max()) >= thr

    start = float(np.argmax(smooth))
    left = right = start
    while supported(left - pitch):
        left -= pitch
    while supported(right + pitch):
        right += pitch
    if (right - left) < pitch * (min_cells - 1):
        return None
    return int(round(left)), int(round(right))


def locate_buff_bar(screen_bgr: np.ndarray, slots: Optional[int] = None,
                    band_top: float = 0.70, min_pitch: int = 18,
                    max_pitch: int = 36):
    """화면 전체에서 버프바를 찾아 ROI를 돌려준다. 실패하면 None.

    ROI를 사람이 드래그하지 않아도 되게 하는 것이 목적이다. 좌표를
    해상도별 표로 들고 있는 방법은 쓰지 않는다. 같은 1920x1080에서도
    UI 배율에 따라 아이콘이 21px과 26px로 갈리므로 해상도는 키가 될 수
    없기 때문이다(README '실측값' 참고).

    버프바는 화면 하단에서 가장 강한 주기 구조다. 실측 두 캡처 모두
    상위 후보가 실제 위치 ±6px 안에 몰렸고, 스킬바나 파티창은 앞서지
    않았다.

    slots: 게임의 버프 표시 칸수를 알고 있으면 넘긴다. 오른쪽 끝이
           고정이므로 거기서 slots*pitch만큼 왼쪽으로 잡으면 폭이
           정확해진다. 모르면 주기가 이어지는 구간으로 추정한다.
    """
    if screen_bgr.ndim != 3:
        return None
    h, w = screen_bgr.shape[:2]
    icon_h = 30

    def scan(ys):
        """(순위값, y, 피치). 순위값 = 주기 점수 x 신호량.

        주기 점수만 보면 아이콘 행이 아니라 위쪽 테두리 몇 줄만 걸친
        창이 이긴다. 내부 무늬가 없어 주기가 더 깨끗하기 때문이다
        (실측: 바가 y=900인데 y=872가 0.897로 1등). 창에 실제로 아이콘이
        얼마나 들어왔는지를 함께 봐야 제자리를 고른다.
        """
        out = []
        for y in ys:
            if y < 0 or y + icon_h > h:
                continue
            win = screen_bgr[y:y + icon_h, :]
            p, s = pitch_by_autocorr(win)
            if p is None or not (min_pitch <= p <= max_pitch):
                continue
            energy = float(_edge_profile(win).sum())
            out.append((s * energy, y, p))
        return out

    # 성기게 훑고, 가장 좋은 자리 근처만 촘촘히 다시 본다
    coarse = scan(range(int(h * band_top), h - icon_h, 4))
    if not coarse:
        return None
    top = max(coarse)[1]
    cands = scan(range(top - 6, top + 7)) or coarse
    cands.sort(reverse=True)

    # 주기 점수가 높아도 그 자리에서 구간이 안 잡힐 수 있다(스택 표시 행에
    # 걸치는 등). 점수 순으로 훑어 구간까지 잡히는 자리를 고른다.
    picked = None
    for rank_val, y, pitch in cands[:8]:
        span = _span_at_pitch(_edge_profile(screen_bgr[y:y + icon_h, :]), pitch)
        if span is not None:
            picked = (rank_val, y, pitch, span)
            break
    if picked is None:
        return None
    rank_val, y, pitch, (x_left, x_right) = picked

    cell = int(round(pitch * 0.9))
    if slots:
        # 오른쪽 끝 기준으로 칸수만큼 왼쪽으로
        x_left = int(round(x_right - (slots - 1) * pitch))

    # 왼쪽에 한 칸 더 붙인다. 버프 표기 한도를 넘으면 맨 앞에 오버플로
    # `+` 칸이 생기는데, 그 칸은 다른 아이콘보다 신호가 약해 구간 추정에서
    # 잘려나간다(실측: 잘린 ROI에서 오버플로가 None이 됐다).
    # 없을 때 빈 칸 하나가 더 들어오는 것은 미인식으로 남아 무해하다.
    x_left = max(0, int(round(x_left - pitch)) - 2)
    x_w = min(w - x_left, int(round(x_right + cell + 2 - x_left)))

    # 찾은 자리보다 조금 위에서 시작한다.
    # 세로 정렬 보정(Recognizer.recalibrate)은 아래로만 훑으므로, ROI가
    # 아이콘보다 1px이라도 아래에서 시작하면 잘린 첫 줄을 되찾을 수 없다.
    # 실측: y를 1px 낮게 잡았더니 14개 중 12개로 떨어졌다.
    margin = 4
    y_top = max(0, y - margin)
    roi_h = min(h - y_top, int(round(cell * 1.85)) + margin)
    return {"x": x_left, "y": y_top, "w": x_w, "h": roi_h,
            "pitch": round(float(pitch), 2)}


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

    # 색 임계값에 기대지 않는 주기 추정으로 교차 검증한다.
    ac_pitch, ac_score = pitch_by_autocorr(roi_bgr)
    trusted_ac = ac_pitch is not None and ac_score >= 0.25

    if best is None:
        # 테두리를 하나도 못 찾았어도 주기가 보이면 그걸로 간다.
        # 시작 위치는 알 수 없으므로 0으로 두고, 위상은 인식 결과로
        # 맞춘다(Recognizer.recalibrate).
        if not trusted_ac:
            return None
        pitch = ac_pitch
        left, cell = 0.0, int(round(pitch * 0.9))
    else:
        _, left, cell, pitch = best
        if trusted_ac and abs(pitch - ac_pitch) > ac_pitch * 0.15:
            # 테두리 기반 추정이 주기와 크게 어긋난다. 배경을 테두리로
            # 오인한 경우이므로 주기 쪽을 믿는다.
            pitch = ac_pitch
            left, cell = 0.0, int(round(pitch * 0.9))

    cell = max(4, min(cell, int(round(pitch))))

    # 테두리가 뭉개지면(영상 압축, 색 프로파일 차이) 연속 구간이 조각나
    # 폭이 실제보다 짧게 잡힌다. 반면 피치는 봉우리 '시작 위치'의
    # 주기에서 나오므로 같은 상황에서도 잘 버틴다.
    # 실측 두 배율 모두 cell/pitch ≈ 0.90 (26/29.0, 21/23.33)이므로,
    # 그보다 크게 벗어나면 폭 추정이 무너진 것으로 보고 피치에서 되돌린다.
    if cell < pitch * 0.82:
        cell = int(round(pitch * 0.9))

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
