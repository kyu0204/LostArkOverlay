"""
버프 아이콘 인식

dHash(모양) + 평균 Hue(색상)를 함께 쓴다.

왜 둘 다 필요한가
-----------------
서폿 버프 4종(빨강/금색/파랑/보라)은 실루엣과 이펙트 배치가 사실상
동일하고 색만 다르다. dHash는 흑백 기반이라 이런 조합에서 마진이 좁다.
실측 해밍 거리:

    red-purple  8      <- 가장 가까운 쌍
    red-gold   12
    blue-gold  12

반면 평균 Hue는 최소 46도 떨어져 있어 아주 잘 갈린다:

    red 9도, gold 55도, blue 205도, purple 295도

그래서 dHash로 후보를 좁히고 Hue로 확정한다.

1px 시프트 문제
---------------
dHash는 밝기 변화, JPEG 압축, 반투명 이펙트에는 매우 강하다(거리 0~5).
그런데 크롭이 1px만 어긋나면 거리가 11~15까지 튄다. 이는 서로 다른
아이콘 간 거리(8~14)보다 크다. 즉 **정렬 오차가 오인의 주된 원인**이다.

따라서 매칭할 때 ±1px 시프트를 모두 시도해 최소 거리를 쓴다.
비용은 9배지만 XOR 몇 번이라 여전히 무시할 수준이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ICON_DB_PATH = Path(__file__).with_name("icons.json")

HASH_SIZE = 8          # 8x8 = 64비트
SAT_MIN = 60           # 색상 계산에 쓸 최소 채도 (어두운 실루엣 제외)
# 임계값은 실측으로 잡았다. 서폿 4종 + 같은 프레임의 다른 아이콘 12개 기준:
#   정탐(1px 시프트 포함): 해밍 0~8,  Hue 0~2도
#   오탐(가장 가까운 남):   해밍 13,   Hue 35도
# 그 사이를 넉넉히 가르되 정탐 쪽에 여유를 더 뒀다.
MAX_HAM = 10           # 이보다 멀면 다른 아이콘
MAX_HUE_DEG = 20.0     # 이보다 색이 다르면 다른 아이콘


# ----------------------------------------------------------------------
# 특징 추출
# ----------------------------------------------------------------------

def dhash(bgr: np.ndarray, size: int = HASH_SIZE) -> int:
    """가로 방향 밝기 대소관계를 64비트로.

    절대 밝기가 아니라 이웃 간 대소만 보므로 배경이 비쳐 전체가
    밝아지거나 어두워져도 값이 유지된다.
    """
    import cv2

    small = cv2.resize(bgr, (size + 1, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    bits = gray[:, 1:] > gray[:, :-1]
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), "big")


def mean_hue(bgr: np.ndarray, sat_min: int = SAT_MIN) -> Optional[float]:
    """채도 있는 픽셀의 평균 색상(0~360도).

    Hue는 원형이라 산술평균을 쓰면 안 된다(350도와 10도의 평균은
    180도가 아니라 0도다). 단위벡터로 바꿔 평균낸 뒤 각도로 되돌린다.
    """
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 1] > sat_min
    if mask.sum() < max(4, mask.size * 0.05):
        return None  # 무채색 아이콘. 색으로 판별할 수 없다.
    ang = np.deg2rad(hsv[:, :, 0][mask].astype(float) * 2.0)
    x, y = np.cos(ang).mean(), np.sin(ang).mean()
    if x * x + y * y < 1e-6:
        return None
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def hue_distance(a: Optional[float], b: Optional[float]) -> float:
    """원형 각도 거리. 한쪽이라도 무채색이면 0(판별 불가 → 통과)."""
    if a is None or b is None:
        return 0.0
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ----------------------------------------------------------------------
# 아이콘 DB
# ----------------------------------------------------------------------

@dataclass
class IconEntry:
    buff_id: str
    hashes: List[int]
    hues: List[Optional[float]]
    ignore: bool = False   # 물약 퀵슬롯처럼 버프가 아닌 칸

    def to_dict(self) -> dict:
        return {
            "hashes": [f"{h:016x}" for h in self.hashes],
            "hues": self.hues,
            "ignore": self.ignore,
        }

    @staticmethod
    def from_dict(buff_id: str, raw: dict) -> "IconEntry":
        return IconEntry(
            buff_id=buff_id,
            hashes=[int(h, 16) for h in raw.get("hashes", [])],
            hues=raw.get("hues", []),
            ignore=bool(raw.get("ignore", False)),
        )


@dataclass
class MatchResult:
    buff_id: Optional[str]
    distance: int
    hue_gap: float
    ignored: bool = False

    @property
    def ok(self) -> bool:
        return self.buff_id is not None and not self.ignored


class IconBook:
    def __init__(self, entries: Optional[Dict[str, IconEntry]] = None):
        self.entries: Dict[str, IconEntry] = entries or {}

    # -- 등록 ----------------------------------------------------------

    def add(self, buff_id: str, tile_bgr: np.ndarray, ignore: bool = False) -> None:
        e = self.entries.get(buff_id)
        if e is None:
            e = IconEntry(buff_id, [], [], ignore)
            self.entries[buff_id] = e
        e.hashes.append(dhash(tile_bgr))
        e.hues.append(mean_hue(tile_bgr))
        e.ignore = e.ignore or ignore

    # -- 조회 ----------------------------------------------------------

    def match(
        self,
        tile_bgr: np.ndarray,
        max_ham: int = MAX_HAM,
        max_hue: float = MAX_HUE_DEG,
        shift: int = 1,
    ) -> MatchResult:
        """가장 가까운 아이콘. 없으면 buff_id=None.

        shift: 이만큼의 픽셀 어긋남까지 감안해 최소 거리를 취한다.
        정렬 오차가 오인의 주된 원인이므로 0으로 두지 말 것.
        """
        if not self.entries:
            return MatchResult(None, 64, 0.0)

        variants = _shifted_variants(tile_bgr, shift)
        cand_hashes = [dhash(v) for v in variants]
        hue = mean_hue(tile_bgr)

        best: Optional[Tuple[int, float, IconEntry]] = None
        for entry in self.entries.values():
            for th, thue in zip(entry.hashes, entry.hues):
                d = min(hamming(h, th) for h in cand_hashes)
                hg = hue_distance(hue, thue)
                if d > max_ham or hg > max_hue:
                    continue
                # 색이 확실히 다르면 모양이 비슷해도 배제된다.
                # 동점일 때는 색이 가까운 쪽을 택한다.
                if best is None or (d, hg) < (best[0], best[1]):
                    best = (d, hg, entry)

        if best is None:
            return MatchResult(None, 64, 0.0)
        d, hg, entry = best
        return MatchResult(entry.buff_id, d, hg, entry.ignore)

    # -- 직렬화 --------------------------------------------------------

    def save(self, path: Path = ICON_DB_PATH) -> None:
        data = {bid: e.to_dict() for bid, e in self.entries.items()}
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path = ICON_DB_PATH) -> "IconBook":
        if not path.exists():
            return IconBook()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return IconBook({bid: IconEntry.from_dict(bid, v) for bid, v in raw.items()})


def _shifted_variants(tile: np.ndarray, shift: int) -> List[np.ndarray]:
    """±shift 픽셀만큼 밀어본 사본들. 원본이 항상 첫 번째."""
    if shift <= 0:
        return [tile]
    h, w = tile.shape[:2]
    out = [tile]
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            if dx == 0 and dy == 0:
                continue
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            if y1 - y0 < h * 0.7 or x1 - x0 < w * 0.7:
                continue
            out.append(tile[y0:y1, x0:x1])
    return out


# ----------------------------------------------------------------------
# 프레임 단위 인식
# ----------------------------------------------------------------------

def identify_cells(
    icon_row_bgr: np.ndarray,
    bounds: List[Tuple[int, int]],
    book: IconBook,
    inset: int = 1,
    **kw,
) -> Dict[int, MatchResult]:
    """셀별 인식 결과. 무시 대상과 미확인은 결과에서 빠진다.

    inset: 테두리를 몇 px 잘라낼지. 테두리 색은 버프 종류가 아니라
    상태(버프/디버프 등)에 따라 바뀌므로 특징에서 빼는 편이 안정적이다.
    """
    out: Dict[int, MatchResult] = {}
    h = icon_row_bgr.shape[0]
    for i, (l, r) in enumerate(bounds):
        if r > icon_row_bgr.shape[1]:
            break
        tile = icon_row_bgr[inset:h - inset, l + inset:r - inset]
        if tile.size == 0:
            continue
        res = book.match(tile, **kw)
        if res.ok:
            out[i] = res
    return out
