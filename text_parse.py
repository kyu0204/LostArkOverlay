"""
버프 지속시간 텍스트 파싱

로아는 아이콘 아래에 남은 시간을 텍스트로 표시한다.

    7초   19초   0.3초   36분   53분

1초 미만이면 소수점 한 자리까지 나온다.

왜 범용 OCR을 쓰지 않는가
-------------------------
Tesseract 같은 엔진은 10Hz로 16개 영역을 돌리기엔 느리고, 13px 높이의
작은 글자에서 정확도도 낮다. 폰트가 고정이므로 글자 단위 템플릿
매칭이 훨씬 빠르고 정확하다. 필요한 템플릿은 12개뿐이다.

    0 1 2 3 4 5 6 7 8 9 . 초 분

배경 문제
---------
게임 배경이 밝으면(예: 수련장 금색 바닥) 글자와 배경의 밝기/색상이
겹쳐 분리가 어렵다. 이때는 **잘못 읽느니 못 읽는 편이 낫다**.
매칭 점수가 낮으면 None을 반환하고, BuffTracker가 카탈로그
duration으로 폴백한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TEMPLATE_PATH = Path(__file__).with_name("glyphs.json")

# 템플릿 정규화 크기. 글자가 13px 안팎이라 이보다 크게 잡을 이유가 없다.
GLYPH_W, GLYPH_H = 10, 14

UNIT_SEC = "초"
UNIT_MIN = "분"

# 버프창에 찍힐 수 있는 수의 상한. 60초가 되면 '1분'으로 바뀌므로
# '83초' 같은 표기는 존재할 수 없다. 단위와 무관하게 화면에 보이는
# 수 자체가 59를 넘지 않는다. 넘으면 숫자를 잘못 읽은 것으로 본다.
MAX_DISPLAY = 59.0


# ----------------------------------------------------------------------
# 분할
# ----------------------------------------------------------------------

@dataclass
class Glyph:
    """분할된 글자 하나."""

    x: int          # ROI 기준 왼쪽
    y: int
    w: int
    h: int
    img: np.ndarray  # 이진 이미지 (0/1)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def right(self) -> int:
        return self.x + self.w


def binarize(gray: np.ndarray, thr: Optional[int] = None) -> np.ndarray:
    """텍스트 행 이진화.

    thr을 주면 그 값으로 자른다. 생략하면 Otsu.

    배경이 밝은 장면(수련장 금색 바닥 등)에서는 Otsu가 배경을 글자로
    잡아 화면 절반이 켜진다. 그래서 켜진 비율이 과한지 호출부에서
    검사한다. adaptive_binarize를 쓰면 그 상황까지 대응한다.
    """
    import cv2

    if thr is not None:
        return (gray > thr).astype(np.uint8)
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.astype(np.uint8)


def adaptive_binarize(
    gray: np.ndarray, lo: float = 0.03, hi: float = 0.35
) -> Tuple[np.ndarray, Optional[int]]:
    """켜진 비율이 글자다운 범위에 들어오는 임계값을 고른다.

    배경 밝기가 장면마다 크게 달라(어두운 레이드 vs 밝은 수련장)
    고정 임계값이나 Otsu 하나로는 둘 다 잡지 못한다. 텍스트가
    차지하는 면적 비율은 어느 장면에서든 비슷하다는 점을 이용한다.

    반환: (이진 이미지, 사용한 임계값). 적절한 값이 없으면 (빈 마스크, None).
    """
    best, best_thr, best_gap = None, None, 1e9
    target = (lo + hi) / 2

    otsu = binarize(gray, None)
    if lo <= otsu.mean() <= hi:
        best, best_thr, best_gap = otsu, -1, abs(otsu.mean() - target)

    for t in range(60, 231, 10):
        bw = (gray > t).astype(np.uint8)
        f = bw.mean()
        if not (lo <= f <= hi):
            continue
        gap = abs(f - target)
        if gap < best_gap:
            best, best_thr, best_gap = bw, t, gap

    if best is None:
        return np.zeros_like(gray, np.uint8), None
    return best, best_thr


# 지속시간 글자 색. OpenCV의 H는 0~179라 실제 각도의 절반이므로,
# 아래 값은 2배한 실제 각도 기준이다. (README의 'H≈49'는 원시값이다)
# 실측 분포: 글자 Hue 60~110, 채도 34~132, 명도 119~227.
TEXT_HUE_GREEN = 100.0
TEXT_HUE_ORANGE = 50.0
TEXT_HUE_WIN = 26.0
TEXT_SAT_MIN = 30
TEXT_VAL_MIN = 110


def text_color_mask(
    bgr: np.ndarray,
    hue_win: float = TEXT_HUE_WIN,
    sat_min: int = TEXT_SAT_MIN,
    val_min: int = TEXT_VAL_MIN,
) -> np.ndarray:
    """지속시간 글자 색만 남긴 이진 마스크.

    밝기만 보는 이진화는 배경이 밝아지면 무너진다. 글자 색은 녹색과
    주황 두 가지로 고정돼 있으므로, 색을 함께 보면 폭발 이펙트처럼
    밝지만 색이 다른 배경을 훨씬 잘 걸러낸다.

    다만 수련장 금색 바닥처럼 배경 색이 주황 글자와 겹치면 마스크가
    통째로 켜진다. 그 경우는 segment_glyphs의 max_fill이 걸러내므로
    호출부가 밝기 기반 결과로 넘어가면 된다.
    """
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(int) * 2
    s = hsv[:, :, 1].astype(int)
    v = hsv[:, :, 2].astype(int)
    greenish = np.abs(h - TEXT_HUE_GREEN) < hue_win
    orangish = np.abs(h - TEXT_HUE_ORANGE) < hue_win * 0.7
    return ((greenish | orangish) & (s > sat_min) & (v > val_min)).astype(np.uint8)


def text_tophat_mask(bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    """주변보다 밝은 얇은 구조만 남긴다 (모폴로지 top-hat).

    글자는 '배경보다 밝다'가 아니라 '바로 옆보다 밝다'는 성질이 있다.
    전역 임계값은 배경 밝기가 바뀌면 무너지지만 이 성질은 유지되므로,
    수련장 금색 바닥처럼 배경이 밝은 장면에서도 글자가 남는다.

    ksize는 글자 획보다 크고 글자 전체보다 작아야 한다. 실측 글자
    높이가 8px이라 5가 잘 맞았다(7, 9는 배경을 더 끌어온다).
    임계값은 Otsu로 잡는다. 장면마다 대비가 달라 고정값을 쓰기 어렵다.
    """
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, bw = cv2.threshold(hat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw.astype(np.uint8)


def _merge_overlapping(boxes, overlap: float = 0.8, max_w: int = 12):
    """x 구간이 크게 겹치는 성분들을 병합한다.

    한글 '초'와 '분'은 여러 획으로 끊겨 별개 성분이 되는데, 그 획들은
    x가 거의 완전히 겹친다. 반면 이웃한 숫자끼리는 겹치지 않는다.
    소수점도 인접 숫자와 겹치지 않으므로 따로 남는다.

    max_w: 병합 결과가 이보다 넓어지면 서로 다른 글자를 붙인 것이므로
    병합하지 않는다. 실측 글자 폭은 숫자 6, 한글 8 안팎이다.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for x, y, w, h in boxes[1:]:
        mx, my, mw, mh = merged[-1]
        inter = min(mx + mw, x + w) - max(mx, x)
        nx = min(mx, x)
        nw = max(mx + mw, x + w) - nx
        if inter > 0 and inter >= overlap * min(mw, w) and nw <= max_w:
            ny = min(my, y)
            merged[-1] = [nx, ny, nw, max(my + mh, y + h) - ny]
        else:
            merged.append([x, y, w, h])
    return [tuple(b) for b in merged]


def _split_wide(cell: np.ndarray, x0: int, max_w: int = 10,
                depth_ratio: float = 0.45):
    """붙어버린 글자를 수직 투영의 골짜기에서 나눈다.

    작은 폰트에서는 이웃 글자의 획이 1px로 닿아 하나의 연결 성분이
    되곤 한다('7초'가 폭 16의 덩어리로 나오는 식). 열 합이 주변보다
    뚜렷하게 낮은 지점만 자르고, 얕은 골짜기는 한 글자 안의 획
    간격이므로 건드리지 않는다.

    반환: [(절대 x, 폭), ...]
    """
    w = cell.shape[1]
    if w <= max_w:
        return [(x0, w)]

    col = cell.sum(axis=0).astype(float)
    margin = 3
    if w <= margin * 2:
        return [(x0, w)]

    inner = col[margin:w - margin]
    if len(inner) == 0:
        return [(x0, w)]

    k = int(np.argmin(inner)) + margin
    around = float(col.mean())
    if around <= 0 or col[k] > around * depth_ratio:
        # 자를 만한 골짜기가 없다. 이 폭 그대로 둔다.
        return [(x0, w)]

    left = _split_wide(cell[:, :k], x0, max_w, depth_ratio)
    right = _split_wide(cell[:, k:], x0 + k, max_w, depth_ratio)
    return left + right


def segment_glyphs(
    bw: np.ndarray, min_h: int = 2, min_area: int = 4, max_fill: float = 0.45
) -> List[Glyph]:
    """이진 이미지에서 글자 후보를 뽑는다.

    min_h가 2로 낮은 이유: 소수점이 1~2px 높이라 이보다 높게 잡으면
    '0.3초'가 '03초'로 읽힌다. 낮춘 대신 들어오는 노이즈는 템플릿
    매칭 점수에서 걸러진다.

    max_fill: 켜진 픽셀 비율이 이보다 크면 배경을 글자로 오인한
    상황이므로 빈 목록을 반환한다. 밝은 배경에서의 오독 방지.
    """
    import cv2

    if bw.mean() > max_fill:
        return []

    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    raw = []
    for i in range(1, n):
        x, y, w, h, area = (
            st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
            st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT],
            st[i, cv2.CC_STAT_AREA],
        )
        if area < min_area:
            continue
        if w > h * 4:  # 가로로 지나치게 긴 것은 UI 선
            continue
        raw.append((x, y, w, h))

    out = []
    for x, y, w, h in _merge_overlapping(raw):
        if h < min_h:
            continue
        for gx, gw in _split_wide(bw[y:y + h, x:x + w], x):
            sub = bw[y:y + h, gx:gx + gw]
            rows = np.where(sub.any(axis=1))[0]
            if len(rows) == 0:
                continue
            ty, th = y + int(rows[0]), int(rows[-1] - rows[0] + 1)
            if th < min_h:
                continue
            out.append(
                Glyph(gx, ty, gw, th, (bw[ty:ty + th, gx:gx + gw] > 0).astype(np.uint8))
            )
    out.sort(key=lambda g: g.x)
    return _drop_noise(out, min_h)


def _drop_noise(glyphs: List[Glyph], min_h: int) -> List[Glyph]:
    """글자 높이에 못 미치는 조각을 버린다.

    같은 줄의 글자는 높이가 고르다(실측 8px). 배경 이펙트가 남긴
    조각은 3~4px로 눈에 띄게 낮은데, min_h를 소수점(1~2px) 때문에
    낮춰 놓아 그대로 통과한다. 그 조각이 글자 옆에 붙으면 같은 그룹에
    묶여 '8초'가 4조각이 되고, 조각 수가 안 맞아 그룹 전체가 버려진다.

    소수점만은 살려야 하므로 '아주 낮고(<=2px) 밑선이 글자와 맞는' 것은
    예외로 둔다. 소수점은 baseline에 붙어 있고 노이즈는 대개 그렇지 않다.
    """
    if len(glyphs) < 2:
        return glyphs

    heights = sorted(g.h for g in glyphs)
    median_h = heights[len(heights) // 2]
    if median_h <= min_h:
        return glyphs   # 전부 작다면 기준으로 삼을 게 없다

    baseline = max(g.y + g.h for g in glyphs)
    kept = []
    for g in glyphs:
        if g.h >= median_h * 0.6:
            kept.append(g)
        elif g.h <= 2 and abs((g.y + g.h) - baseline) <= 1:
            kept.append(g)   # 소수점
    return kept or glyphs


def group_glyphs(glyphs: List[Glyph], max_gap: int = 4) -> List[List[Glyph]]:
    """가까운 글자끼리 묶는다. 한 그룹 = 버프 하나의 남은 시간 표기.

    소수점은 폭이 작고 아래쪽에 붙지만, 인접 글자와의 간격이 좁아
    같은 그룹으로 들어온다.
    """
    if not glyphs:
        return []
    groups = [[glyphs[0]]]
    for g in glyphs[1:]:
        if g.x - groups[-1][-1].right <= max_gap:
            groups[-1].append(g)
        else:
            groups.append([g])
    return groups


# ----------------------------------------------------------------------
# 템플릿 매칭
# ----------------------------------------------------------------------

def normalize_glyph(img: np.ndarray) -> np.ndarray:
    """템플릿 비교용 고정 크기 이진 벡터."""
    import cv2

    r = cv2.resize(
        img.astype(np.uint8) * 255, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA
    )
    return (r > 127).astype(np.uint8)


class GlyphBook:
    """글자 템플릿 모음. JSON으로 저장/로드한다."""

    def __init__(self, templates: Optional[Dict[str, List[np.ndarray]]] = None):
        self.templates: Dict[str, List[np.ndarray]] = templates or {}

    def add(self, label: str, glyph_img: np.ndarray) -> None:
        self.templates.setdefault(label, []).append(normalize_glyph(glyph_img))

    def match(self, glyph_img: np.ndarray) -> Tuple[Optional[str], float]:
        """가장 잘 맞는 라벨과 일치율(0~1).

        아이콘 매칭과 달리 ±1px 시프트를 시도하지 않는다. 시도해 봤더니
        숫자끼리 혼동이 늘어 오히려 나빠졌다(실측: 수련장 캡처가 7칸 ->
        4칸으로 줄고, '0.3초'를 '3.0'으로 읽는 오독까지 생겼다).
        글자는 10x14로 이미 정규화돼 있어 아이콘만큼 정렬에 예민하지 않고,
        후보가 12종뿐이라 서로 가까워 여유를 주면 바로 섞인다.
        """
        if not self.templates:
            return None, 0.0
        v = normalize_glyph(glyph_img)
        best, best_score = None, -1.0
        for label, samples in self.templates.items():
            for t in samples:
                score = float((v == t).mean())
                if score > best_score:
                    best, best_score = label, score
        return best, best_score

    # -- 직렬화 --------------------------------------------------------

    def save(self, path: Path = TEMPLATE_PATH) -> None:
        data = {
            label: ["".join(map(str, t.flatten().tolist())) for t in samples]
            for label, samples in self.templates.items()
        }
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path = TEMPLATE_PATH) -> "GlyphBook":
        if not path.exists():
            return GlyphBook()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            tpl = {
                label: [
                    np.array([int(c) for c in s], np.uint8).reshape(GLYPH_H, GLYPH_W)
                    for s in samples
                ]
                for label, samples in raw.items()
            }
            return GlyphBook(tpl)
        except Exception:
            # 템플릿이 깨졌다고 프로그램이 죽으면 안 된다. 빈 템플릿으로 간다.
            return GlyphBook()


# ----------------------------------------------------------------------
# 해석
# ----------------------------------------------------------------------

def to_seconds(
    text: str, allow_minutes: bool = False, max_display: float = MAX_DISPLAY,
    require_unit: bool = False,
) -> Optional[float]:
    """'19초' -> 19.0,  '0.3초' -> 0.3

    allow_minutes=False (기본): '분' 표기를 거부한다.

    추적 대상 버프는 모두 초 단위이므로 분 표기가 나올 이유가 없다.
    오히려 '분'과 '초'는 둘 다 8px 폭에 획이 끊기는 한글이라 서로
    오인될 수 있는데, '53분'을 '53초'로 잘못 읽으면 3180초짜리 버프가
    53초로 표시되어 오히려 방해가 된다. 아예 후보에서 빼는 편이 안전하다.

    max_display: **화면에 찍힌 수**의 상한. 환산한 초가 아니다.
    60초는 '1분'으로 바뀌므로 어느 단위든 표기되는 수는 59를 넘지
    않는다. 그래서 '83초'는 존재할 수 없는 표기이고, 나왔다면 숫자를
    잘못 읽은 것이다. ('3분'=180초처럼 환산값이 큰 것은 정상이므로
    환산 후에 재면 멀쩡한 표기를 버리게 된다)

    require_unit: 단위 글자가 없으면 거부한다. 인식 경로는 반드시
    켜야 한다. 게임은 단위 없이 숫자만 찍지 않으므로, 단위가 빠진
    문자열은 뒷글자를 놓친 **부분 판독**이라는 뜻이다. 부분 판독을
    받아들이면 조용한 오독이 된다:

        '15분'에서 '분'을 놓침 -> '15'  ->  15초   (실제 900초)
        '15초'에서 '5초'를 놓침 -> '1'   ->   1초   (실제 15초)

    실측(실전 캡처 31장): 판독 92건 중 17건(18%)이 단위 없는 부분
    판독이었다. 기본값을 False로 둔 것은 단위 없는 문자열을 직접
    넘겨 쓰는 유틸 호출을 깨지 않기 위해서다.
    """
    if not text:
        return None

    if text.endswith(UNIT_SEC):
        unit, body = 1.0, text[:-1]
    elif text.endswith(UNIT_MIN):
        if not allow_minutes:
            return None
        unit, body = 60.0, text[:-1]
    else:
        if require_unit:
            return None
        unit, body = 1.0, text

    if not body:
        return None

    # 앞자리 0 뒤에 숫자가 오는 표기는 게임에 없다('03초'는 '3초'로 뜬다).
    # 이런 문자열이 나왔다면 소수점을 놓친 것이다.
    #
    # 버프창이 표기할 수 있는 값은 1~59초와 0.1~0.9초뿐이다. 어휘가
    # 닫혀 있으므로 앞자리 0은 소수점 표기라는 뜻 외에 다른 해석이
    # 없다. 그래서 버리지 않고 되살린다 - '.' 템플릿이 없어도 값을
    # 복원할 수 있다.
    #
    # 세 자리 이상은 되살리지 않는다. 그런 표기는 어휘에 없으니
    # 소수점을 놓친 것이 아니라 숫자를 잘못 읽은 것이다.
    if len(body) >= 2 and body[0] == "0" and body[1].isdigit():
        if len(body) != 2:
            return None
        body = "0." + body[1]

    try:
        value = float(body)
    except ValueError:
        return None
    if value < 0:
        return None

    if value > max_display:
        return None
    return value * unit


def read_group(
    group: List[Glyph], book: GlyphBook, min_score: float = 0.86
) -> Tuple[Optional[str], float]:
    """글자 그룹을 문자열로. 한 글자라도 확신이 없으면 실패로 본다.

    소수점만은 템플릿으로 맞히지 않고 기하로 집는다. 3x2px짜리라
    모양 정보가 사실상 없어서, 10x14로 늘려 대조하면 그냥 채워진
    사각형이 되어 아무 글자에나 어중간하게 붙는다. 대신 '아주 낮고,
    밑선에 붙어 있고, 글자 사이에 있는' 조각은 소수점으로 본다 -
    표기 어휘상 그 자리에 올 수 있는 것이 그것뿐이다.
    """
    chars, worst = [], 1.0
    baseline = max(g.y + g.h for g in group)
    last = len(group) - 1
    for i, g in enumerate(group):
        if 0 < i < last and g.h <= 2 and abs((g.y + g.h) - baseline) <= 1:
            chars.append(".")
            continue
        label, score = book.match(g.img)
        if label is None or score < min_score:
            return None, score
        chars.append(label)
        worst = min(worst, score)
    return "".join(chars), worst


# ----------------------------------------------------------------------
# 아이콘 배정
# ----------------------------------------------------------------------

def assign_to_cells(
    groups: List[List[Glyph]], cell_centers: List[float], max_dist: float
) -> Dict[int, List[Glyph]]:
    """텍스트 그룹을 가장 가까운 아이콘에 배정한다.

    셀 경계로 자르지 않는 이유: 텍스트가 아이콘보다 넓을 수 있다.
    실측에서 '53분'은 30px로 아이콘(26px)보다 넓어 옆 칸을 침범했다.
    그룹의 중심 x를 쓰면 침범해도 올바른 아이콘에 붙는다.
    """
    out: Dict[int, List[Glyph]] = {}
    used = set()
    # 가까운 쌍부터 확정해야 한 아이콘에 두 그룹이 몰리지 않는다
    pairs = []
    for gi, grp in enumerate(groups):
        cx = (grp[0].x + grp[-1].right) / 2.0
        for ci, cc in enumerate(cell_centers):
            d = abs(cx - cc)
            if d <= max_dist:
                pairs.append((d, gi, ci))
    pairs.sort()
    taken_groups = set()
    for d, gi, ci in pairs:
        if gi in taken_groups or ci in used:
            continue
        out[ci] = groups[gi]
        taken_groups.add(gi)
        used.add(ci)
    return out


def parse_durations(
    text_bgr: np.ndarray,
    cell_centers: List[float],
    book: GlyphBook,
    pitch: float,
    thr: Optional[int] = None,
    min_score: float = 0.86,
    allow_minutes: bool = False,
) -> Dict[int, float]:
    """텍스트 행에서 셀 인덱스 -> 남은 시간(초).

    읽지 못한 셀은 결과에 없다. 호출부는 그 셀을 remaining=None으로
    넘겨 카탈로그 폴백을 태우면 된다.
    """
    import cv2

    gray = cv2.cvtColor(text_bgr, cv2.COLOR_BGR2GRAY)
    bw = binarize(gray, thr)
    glyphs = segment_glyphs(bw)
    groups = group_glyphs(glyphs, max_gap=3)  # Recognizer._read_durations와 동일 값
    assigned = assign_to_cells(groups, cell_centers, max_dist=pitch * 0.9)

    out: Dict[int, float] = {}
    for ci, grp in assigned.items():
        text, _ = read_group(grp, book, min_score)
        if text is None:
            continue
        sec = to_seconds(text, allow_minutes=allow_minutes)
        if sec is not None:
            out[ci] = sec
    return out
