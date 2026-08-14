"""
ROI 캡처 + 그리드 정렬 확인

게임 접속 후 처음 할 일이 이것. 저장된 ROI 프레임을 떠서
아이콘 크기/간격, 지속시간 표현 방식, 신규 버프 삽입 위치를 눈으로 확인한다.

실행
----
    python capture.py                          # 1장 캡처
    python capture.py --frames 30 --interval 1 # 30초간 1초 간격
    python capture.py --grid 21x2.33           # 셀 21px, 간격 2.33px 격자
    python capture.py --grid 21x2.33+1         # 시작 오프셋 1px
    python capture.py --annotate captures/x.png --grid 21x2.33

그리드는 dHash 매칭 전에 셀 경계가 아이콘과 맞는지 확인하는 용도다.
경계가 어긋나면 해시가 전부 흔들려 매칭이 안 된다.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

OUT_DIR = Path(__file__).with_name("captures")


# ----------------------------------------------------------------------
# 순수 함수 (화면 없이 테스트 가능)
# ----------------------------------------------------------------------

GRID_RE = re.compile(r"^(\d+)x(\d+(?:\.\d+)?)(?:\+(\d+))?$")


def parse_grid(spec: str) -> Tuple[int, float, int]:
    """'21x2.33+0' -> (cell=21, gap=2.33, offset=0)

    간격이 소수인 이유는 게임 UI가 배율 스케일링을 쓰기 때문이다.
    정수로 반올림하면 아이콘이 늘어날수록 오차가 누적된다.
    """
    m = GRID_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"그리드 형식이 잘못됐습니다: {spec!r} (예: 21x2 또는 21x2.33+1)"
        )
    cell, gap, off = int(m.group(1)), float(m.group(2)), int(m.group(3) or 0)
    if cell <= 0:
        raise ValueError("셀 크기는 1 이상이어야 합니다")
    if gap < 0:
        raise ValueError("간격은 0 이상이어야 합니다")
    return cell, gap, off


def cell_bounds(
    width: int, cell: int, gap: float, offset: int = 0
) -> List[Tuple[int, int]]:
    """가로 방향 셀 경계 목록. [(left, right), ...]

    피치(cell + gap)를 실수로 누적한 뒤 각 셀에서 반올림한다.
    정수 피치로 더해나가면 아이콘 8개쯤에서 몇 px씩 밀려
    해시가 전부 흔들린다.

    마지막 셀이 폭을 넘으면 잘라내지 않고 제외한다.
    부분적으로 걸친 셀을 해싱하면 오탐의 원인이 된다.
    """
    out = []
    pitch = cell + gap
    i = 0
    while True:
        left = int(round(offset + i * pitch))
        right = left + cell
        if right > width:
            break
        out.append((left, right))
        i += 1
    return out


def draw_grid(
    img: np.ndarray, cell: int, gap: float, offset: int = 0
) -> Tuple[np.ndarray, int]:
    """격자를 덧그린 사본과 셀 개수를 반환."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    bounds = cell_bounds(w, cell, gap, offset)
    for i, (l, r) in enumerate(bounds):
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
        cv2.rectangle(out, (l, 0), (r - 1, h - 1), color, 1)
        cv2.putText(out, str(i), (l + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1, cv2.LINE_AA)
    return out, len(bounds)


def slice_cells(
    img: np.ndarray, cell: int, gap: float, offset: int = 0
) -> List[np.ndarray]:
    """아이콘 후보 셀들을 잘라낸다. 다음 단계(dHash)의 입력."""
    h = img.shape[0]
    return [img[0:h, l:r] for l, r in cell_bounds(img.shape[1], cell, gap, offset)]


# ----------------------------------------------------------------------
# 캡처
# ----------------------------------------------------------------------

def grab(region: dict) -> np.ndarray:
    """ROI 한 프레임을 BGR ndarray로."""
    import mss

    with mss.mss() as sct:
        raw = sct.grab(
            {"left": region["x"], "top": region["y"],
             "width": region["w"], "height": region["h"]}
        )
        return np.array(raw)[:, :, :3]  # BGRA -> BGR


def resolve_roi(name: str) -> Optional[dict]:
    from PySide6.QtWidgets import QApplication

    import config

    app = QApplication.instance() or QApplication(sys.argv)
    key = config.screen_key(app)
    roi = config.get_roi(key, name)
    if roi is None:
        print(f"[{key}] '{name}' ROI가 없습니다. 먼저 실행하세요:")
        print(f"    python roi_picker.py --name {name}")
    return roi


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="buffbar")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--grid", help="셀x간격[+오프셋], 예: 21x2.33")
    ap.add_argument("--annotate", help="캡처 대신 기존 PNG에 격자만 덧그림")
    ap.add_argument("--auto", action="store_true",
                    help="셀/피치를 자동 검출 (UI 배율이 달라도 동작)")
    args = ap.parse_args()

    grid = parse_grid(args.grid) if args.grid else None
    OUT_DIR.mkdir(exist_ok=True)

    # 기존 파일에 격자만 덧그리는 모드
    if args.annotate:
        src = Path(args.annotate)
        img = cv2.imread(str(src))
        if img is None:
            print(f"읽을 수 없습니다: {src}")
            return 1
        if args.auto:
            from grid_detect import detect_grid
            g = detect_grid(img)
            if g is None:
                print("자동 검출 실패. 아이콘 행만 크롭한 이미지인지 확인하세요.")
                return 1
            print(f"검출: {g}   --grid {g.spec()}")
            grid = (g.cell, g.pitch - g.cell, g.left)
        if not grid:
            print("--annotate 는 --grid 또는 --auto 와 함께 써야 합니다.")
            return 1
        out, n = draw_grid(img, *grid)
        dst = src.with_name(src.stem + "_grid.png")
        cv2.imwrite(str(dst), out)
        print(f"{dst}  (셀 {n}개)")
        return 0

    roi = resolve_roi(args.name)
    if roi is None:
        return 1

    stamp = time.strftime("%m%d_%H%M%S")
    for i in range(args.frames):
        img = grab(roi)
        if args.auto and grid is None:
            from grid_detect import detect_grid
            g = detect_grid(img)
            if g is not None:
                grid = (g.cell, g.pitch - g.cell, g.left)
                print(f"자동 검출: {g}   --grid {g.spec()}")
        path = OUT_DIR / f"{args.name}_{stamp}_{i:03d}.png"
        cv2.imwrite(str(path), img)
        msg = f"{path.name}  {img.shape[1]}x{img.shape[0]}"
        if grid:
            g, n = draw_grid(img, *grid)
            gpath = path.with_name(path.stem + "_grid.png")
            cv2.imwrite(str(gpath), g)
            msg += f"  -> 격자 {n}셀"
        print(msg)
        if i < args.frames - 1:
            time.sleep(args.interval)

    print(f"\n{OUT_DIR} 확인. 볼 것:")
    print("  1. 아이콘 한 변 크기와 간격 (--grid 로 맞춰보기)")
    print("  2. 남은 시간 표현: 숫자 / 부채꼴 / 표시 없음")
    print("  3. 신규 버프가 앞에 들어오는지 뒤에 붙는지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
