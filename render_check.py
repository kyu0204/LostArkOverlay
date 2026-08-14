"""헤드리스 렌더 확인용. 오버레이를 PNG로 떠서 눈으로 검증한다."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QSize
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from overlay import ROW_GAP, ROW_H, PAD, WIDTH, BuffOverlay


ROWS = [
    {"name": "황제", "remaining": 9.4, "progress": 0.78,
     "confidence": "observed", "color": "#ffb300"},
    {"name": "치명타 시너지", "remaining": 3.1, "progress": 0.39,
     "confidence": "observed", "color": "#4fc3f7"},
    {"name": "각성 물약", "remaining": 21.6, "progress": 0.72,
     "confidence": "predicted", "color": "#81c784"},
    {"name": "공격력 증가 (아주 긴 이름 말줄임 확인)", "remaining": 14.0,
     "progress": 0.70, "confidence": "predicted", "color": "#ba9cf5"},
]


def main() -> int:
    app = QApplication(sys.argv)
    w = BuffOverlay(setup_mode=False)
    w.set_rows(ROWS)
    h = len(ROWS) * (ROW_H + ROW_GAP) + PAD * 2

    # 게임 화면 위에 얹힌 상황을 흉내내기 위해 어두운 배경을 깐다
    img = QImage(QSize(WIDTH, h), QImage.Format_ARGB32)
    img.fill(QColor("#232733"))
    p = QPainter(img)
    w.render(p, QPoint())
    p.end()

    out = "render_check.png"
    img.save(out)
    print(f"saved {out} ({WIDTH}x{h}, rows={len(ROWS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
