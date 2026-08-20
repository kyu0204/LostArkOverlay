"""
ROI 지정 UI

화면을 한 장 캡처해 어둡게 깔고, 드래그로 영역을 잡는다.
잡은 영역은 해상도/DPI별 프리셋으로 저장된다.

실행
----
    python roi_picker.py                  # 기본 이름 'buffbar'
    python roi_picker.py --name slots     # 다른 영역 지정
    python roi_picker.py --list           # 저장된 프리셋 확인

조작
----
    드래그        영역 선택
    방향키        선택 영역 1px 이동 (Shift: 크기 조절)
    Enter         저장 후 종료
    Esc           취소
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Tuple

import config

Rect = Dict[str, int]


# ----------------------------------------------------------------------
# 순수 기하 함수 (GUI 없이 테스트 가능)
# ----------------------------------------------------------------------

def normalize(x0: int, y0: int, x1: int, y1: int) -> Rect:
    """드래그 방향과 무관하게 좌상단 기준 사각형으로 정규화."""
    return {
        "x": min(x0, x1),
        "y": min(y0, y1),
        "w": abs(x1 - x0),
        "h": abs(y1 - y0),
    }


def clamp(rect: Rect, bounds: Tuple[int, int, int, int]) -> Rect:
    """화면 밖으로 나가지 않게 자른다. bounds = (x, y, w, h)."""
    bx, by, bw, bh = bounds
    x = max(bx, min(rect["x"], bx + bw))
    y = max(by, min(rect["y"], by + bh))
    w = max(0, min(rect["w"], bx + bw - x))
    h = max(0, min(rect["h"], by + bh - y))
    return {"x": x, "y": y, "w": w, "h": h}


def nudge(rect: Rect, dx: int, dy: int, resize: bool = False) -> Rect:
    """방향키 미세조정. resize=True면 우/하단 모서리만 움직인다."""
    r = dict(rect)
    if resize:
        r["w"] = max(1, r["w"] + dx)
        r["h"] = max(1, r["h"] + dy)
    else:
        r["x"] += dx
        r["y"] += dy
    return r


def is_usable(rect: Optional[Rect], min_side: int = 8) -> bool:
    """실수로 클릭만 하고 놓은 경우를 걸러낸다."""
    if not rect:
        return False
    return rect["w"] >= min_side and rect["h"] >= min_side


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------

def run_picker(name: str) -> int:
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
    from PySide6.QtWidgets import QApplication, QWidget

    LOUPE = 120      # 돋보기 한 변 (px)
    LOUPE_ZOOM = 6   # 확대 배율

    class Picker(QWidget):
        def __init__(self, shot, geo, existing: Optional[Rect]):
            super().__init__()
            self.shot = shot
            self.geo = geo
            self.rect_sel: Optional[Rect] = existing
            self.dragging = False
            self.start = (0, 0)
            self.cursor = (0, 0)
            self.result: Optional[Rect] = None

            self.setWindowFlags(
                Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
            )
            self.setGeometry(geo)
            self.setCursor(Qt.CrossCursor)
            self.setMouseTracking(True)
            self.font = QFont("Malgun Gothic", 10, QFont.DemiBold)

        # -- 입력 ---------------------------------------------------

        def mousePressEvent(self, ev):
            if ev.button() == Qt.LeftButton:
                self.dragging = True
                p = ev.position().toPoint()
                self.start = (p.x(), p.y())
                self.rect_sel = normalize(p.x(), p.y(), p.x(), p.y())
                self.update()

        def mouseMoveEvent(self, ev):
            p = ev.position().toPoint()
            self.cursor = (p.x(), p.y())
            if self.dragging:
                self.rect_sel = normalize(*self.start, p.x(), p.y())
            self.update()

        def mouseReleaseEvent(self, _ev):
            self.dragging = False
            self.update()

        def keyPressEvent(self, ev):
            k = ev.key()
            if k == Qt.Key_Escape:
                self.result = None
                self.close()
                return
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self.result = self.rect_sel if is_usable(self.rect_sel) else None
                self.close()
                return
            if not self.rect_sel:
                return
            resize = bool(ev.modifiers() & Qt.ShiftModifier)
            delta = {
                Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0),
                Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1),
            }.get(k)
            if delta:
                b = (0, 0, self.width(), self.height())
                self.rect_sel = clamp(nudge(self.rect_sel, *delta, resize), b)
                self.update()

        # -- 렌더 ---------------------------------------------------

        def paintEvent(self, _ev):
            p = QPainter(self)
            p.drawPixmap(0, 0, self.shot)
            p.fillRect(self.rect(), QColor(0, 0, 0, 130))  # 전체 딤

            r = self.rect_sel
            if r and r["w"] > 0 and r["h"] > 0:
                qr = QRect(r["x"], r["y"], r["w"], r["h"])
                # 선택 영역만 원본 밝기로
                p.drawPixmap(qr, self.shot, qr)
                p.setPen(QPen(QColor("#4fc3f7"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(qr)
                self._draw_label(p, qr, r)

            self._draw_loupe(p)
            self._draw_help(p)
            p.end()

        def _draw_label(self, p, qr, r):
            text = f'{r["w"]} x {r["h"]}   ({r["x"]}, {r["y"]})'
            p.setFont(self.font)
            fm = p.fontMetrics()
            w, h = fm.horizontalAdvance(text) + 12, fm.height() + 8
            ly = qr.top() - h - 4
            if ly < 0:
                ly = qr.bottom() + 4
            box = QRect(qr.left(), ly, w, h)
            p.fillRect(box, QColor(12, 15, 20, 230))
            p.setPen(QColor("#e6eef8"))
            p.drawText(box, Qt.AlignCenter, text)

        def _draw_loupe(self, p):
            """픽셀 단위로 경계를 맞추기 위한 돋보기."""
            cx, cy = self.cursor
            src = LOUPE // LOUPE_ZOOM
            sx, sy = cx - src // 2, cy - src // 2
            lx = cx + 24 if cx + 24 + LOUPE < self.width() else cx - 24 - LOUPE
            ly = cy + 24 if cy + 24 + LOUPE < self.height() else cy - 24 - LOUPE

            dst = QRect(lx, ly, LOUPE, LOUPE)
            p.drawPixmap(dst, self.shot, QRect(sx, sy, src, src))
            p.setPen(QPen(QColor(255, 255, 255, 60), 1))
            p.drawLine(dst.left(), dst.center().y(), dst.right(), dst.center().y())
            p.drawLine(dst.center().x(), dst.top(), dst.center().x(), dst.bottom())
            p.setPen(QPen(QColor("#4fc3f7"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(dst)

        def _draw_help(self, p):
            lines = [
                f"[{name}] 드래그로 영역 선택",
                "방향키 이동 / Shift+방향키 크기조절",
                "Enter 저장   Esc 취소",
            ]
            p.setFont(self.font)
            fm = p.fontMetrics()
            w = max(fm.horizontalAdvance(t) for t in lines) + 20
            h = fm.height() * len(lines) + 16
            box = QRect(20, 20, w, h)
            p.fillRect(box, QColor(12, 15, 20, 220))
            p.setPen(QColor("#9fb4c8"))
            y = 28
            for t in lines:
                p.drawText(30, y + fm.ascent(), t)
                y += fm.height()

    app = QApplication(sys.argv)
    key = config.screen_key(app)
    scr = QGuiApplication.primaryScreen()
    geo = scr.geometry()
    shot = scr.grabWindow(0)

    w = Picker(shot, geo, config.get_roi(key, name))
    w.show()
    w.activateWindow()
    w.raise_()
    app.exec()

    if w.result is None:
        print("취소됨. 저장하지 않았습니다.")
        return 1

    config.put_roi(key, name, w.result)
    r = w.result
    print(f'저장 완료  [{key}] {name} = x={r["x"]} y={r["y"]} w={r["w"]} h={r["h"]}')
    return 0


def list_presets() -> int:
    data = config.load_all()
    if not data:
        print("저장된 프리셋이 없습니다.")
        return 0
    for key, section in data.items():
        rois = section.get("roi", {})
        if not rois:
            continue
        print(f"[{key}]")
        for name, r in rois.items():
            print(f'  {name:12s} x={r["x"]:5d} y={r["y"]:5d} '
                  f'w={r["w"]:4d} h={r["h"]:4d}')
    return 0


def run_auto(name: str, slots: Optional[int] = None) -> int:
    """버프바를 자동으로 찾아 미리보기를 남기고, 확인 후 저장한다.

    드래그 단계를 없애는 것이 목적이다. 해상도별 좌표표를 두는 방법은
    쓰지 않는다. 같은 1920x1080에서도 UI 배율에 따라 아이콘이 21px과
    26px로 갈리므로 해상도만으로는 좌표가 정해지지 않기 때문이다.

    자동 탐색이 헛짚을 수 있으므로 반드시 눈으로 확인하게 한다.
    """
    import cv2
    import mss
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from grid_detect import annotate, detect_grid, locate_buff_bar

    app = QApplication.instance() or QApplication(sys.argv)
    key = config.screen_key(app)

    with mss.mss() as sct:
        shot = np.array(sct.grab(sct.monitors[1]))[:, :, :3]

    roi = locate_buff_bar(shot, slots=slots)
    if roi is None:
        print("버프바를 찾지 못했습니다. 게임 화면이 보이는 상태인지 확인하고,")
        print("안 되면 드래그로 잡으세요:  python roi_picker.py")
        return 1

    rect = {k: roi[k] for k in ("x", "y", "w", "h")}
    print(f"[{key}] 찾음: x={rect['x']} y={rect['y']} "
          f"w={rect['w']} h={rect['h']}  (피치 {roi['pitch']}, 신뢰도 {roi['score']})")

    crop = shot[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"]]
    g = detect_grid(crop[: max(4, int(roi["pitch"] * 0.9) + 2)])
    preview = annotate(crop.copy(), g) if g else crop.copy()
    out = "roi_auto_preview.png"
    cv2.imwrite(out, cv2.resize(preview, None, fx=3, fy=3,
                                interpolation=cv2.INTER_NEAREST))
    print(f"미리보기: {out}  — 격자가 아이콘과 맞는지 확인하세요.")

    config.put_roi(key, name, rect)
    print(f"'{name}'으로 저장했습니다. 어긋났다면 다시 잡으세요:")
    print("    python roi_picker.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="buffbar", help="영역 이름 (기본: buffbar)")
    ap.add_argument("--list", action="store_true", help="저장된 프리셋 출력")
    ap.add_argument("--auto", action="store_true",
                    help="드래그 없이 버프바를 자동으로 찾아 저장")
    ap.add_argument("--slots", type=int,
                    help="버프 표시 칸수. 알면 폭이 정확해진다 "
                         "(오른쪽 끝이 고정이라 거기서 칸수만큼 왼쪽으로 잡는다)")
    args = ap.parse_args()

    if args.list:
        return list_presets()
    if args.auto:
        return run_auto(args.name, args.slots)
    return run_picker(args.name)


if __name__ == "__main__":
    raise SystemExit(main())
