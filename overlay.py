"""
로스트아크 버프 오버레이 - 렌더링 셸

인식 파이프라인 없이 단독 실행 가능하다. 더미 피드로 렌더링,
정렬, 클릭 통과, z-order, 위치 저장을 먼저 검증하기 위한 것.

실행
----
    python overlay.py           # 일반 모드. 클릭 통과, 항상 위
    python overlay.py --setup   # 배치 모드. 드래그로 위치 조정 후 저장

중요
----
게임이 전체화면 모드면 이 오버레이는 보이지 않는다.
반드시 '테두리 없는 창 모드'로 실행할 것.
"""

from __future__ import annotations

import argparse
import sys
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication, QWidget

import config
from buff_state import CONF_PREDICTED, BuffTracker, Observation
from icon_match import ICON_IMG_DIR

# 레이아웃 상수
ROW_H = 40
ROW_GAP = 4
ICON = 30
PAD = 8
WIDTH = 230

# 상태 표시. 버프가 하나도 없을 때 화면이 비어 있으면 앱이 켜져 있는지
# 알 수 없다. 무엇까지 되고 있는지도 함께 보여준다.
ST_OK = "ok"                # 버프바를 찾았고 인식 중
ST_SEARCHING = "searching"  # 캡처는 되는데 버프바를 못 찾음
ST_ERROR = "error"          # 캡처 자체가 실패

STATUS_TEXT = {
    ST_OK: "인식 중",
    ST_SEARCHING: "버프바 찾는 중",
    ST_ERROR: "캡처 실패",
}
STATUS_COLOR = {
    ST_OK: QColor(129, 199, 132),
    ST_SEARCHING: QColor(255, 179, 0),
    ST_ERROR: QColor(255, 82, 82),
}
FONT_STACK = ["Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "sans-serif"]


# ----------------------------------------------------------------------
# 위치 설정 (config.py의 해상도별 프리셋을 공유)
# ----------------------------------------------------------------------

def load_pos(key: str):
    p = config.get(key, "overlay", "pos")
    return (p["x"], p["y"]) if p else None


def save_pos(key: str, x: int, y: int) -> None:
    config.put(key, "overlay", "pos", {"x": int(x), "y": int(y)})


# ----------------------------------------------------------------------
# Windows 전용: 포커스 탈취 방지 + 작업표시줄 숨김
# ----------------------------------------------------------------------

def apply_win_flags(widget: QWidget, click_through: bool) -> None:
    """WS_EX_NOACTIVATE로 게임 포커스를 뺏지 않게 하고,
    WS_EX_TOOLWINDOW로 알트탭 목록에서 숨긴다."""
    if sys.platform != "win32":
        return
    import ctypes

    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOOLWINDOW = 0x00000080
    WS_EX_LAYERED = 0x00080000
    WS_EX_NOACTIVATE = 0x08000000

    hwnd = int(widget.winId())
    user32 = ctypes.windll.user32
    get_fn = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_fn = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)

    style = get_fn(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    if click_through:
        style |= WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    set_fn(hwnd, GWL_EXSTYLE, style)


# ----------------------------------------------------------------------
# 오버레이 위젯
# ----------------------------------------------------------------------

class BuffOverlay(QWidget):
    def __init__(self, setup_mode: bool = False):
        super().__init__()
        self.setup_mode = setup_mode
        self.rows: list[dict] = []
        self._drag_from: QPointF | None = None
        self._icon_cache: dict[str, QPixmap | None] = {}
        self._status = (ST_SEARCHING, "")
        self._beat = 0

        flags = (
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.NoDropShadowWindowHint
        )
        if not setup_mode:
            flags |= Qt.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        if not setup_mode:
            self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.setFixedWidth(WIDTH)
        self.resize(WIDTH, 240)

        self.font_name = QFont(FONT_STACK[0], 10, QFont.DemiBold)
        self.font_time = QFont(FONT_STACK[0], 11, QFont.Bold)

    # -- 아이콘 이미지 ---------------------------------------------------

    def _icon_pixmap(self, buff_id):
        """등록된 아이콘 원본. 없으면 None.

        10Hz로 다시 그리므로 한 번 읽은 것은 캐시한다. 없는 것도
        None으로 캐시해 매 프레임 디스크를 두드리지 않게 한다.
        """
        if not buff_id:
            return None
        if buff_id in self._icon_cache:
            return self._icon_cache[buff_id]

        path = ICON_IMG_DIR / f"{buff_id}.png"
        pix = None
        if path.exists():
            loaded = QPixmap(str(path))
            if not loaded.isNull():
                pix = loaded
        self._icon_cache[buff_id] = pix
        return pix

    # -- 데이터 주입 -----------------------------------------------------

    def set_rows(self, rows: list[dict]) -> None:
        self.rows = rows
        h = max(1, len(rows)) * (ROW_H + ROW_GAP) + PAD * 2
        if h != self.height():
            self.resize(WIDTH, h)
        self._beat += 1          # 살아 있다는 표시(점이 깜빡인다)
        self.update()

    def set_status(self, state: str, detail: str = "") -> None:
        self._status = (state, detail)

    # -- 렌더링 ----------------------------------------------------------

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.setup_mode:
            p.setPen(QPen(QColor(120, 200, 255, 220), 2, Qt.DashLine))
            p.setBrush(QColor(20, 24, 32, 90))
            p.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 8, 8)

        y = PAD
        for row in self.rows:
            self._paint_row(p, y, row)
            y += ROW_H + ROW_GAP

        if self.setup_mode and not self.rows:
            p.setPen(QColor(200, 220, 255))
            p.setFont(self.font_name)
            p.drawText(self.rect(), Qt.AlignCenter, "드래그해서 위치 조정")
        elif not self.rows:
            # 버프가 없을 때 화면이 완전히 비면 앱이 죽은 것과 구분되지
            # 않는다. 작게라도 살아 있음을 남긴다.
            self._paint_status(p, PAD)

        p.end()

    def _paint_status(self, p: QPainter, y: int) -> None:
        state, detail = self._status
        color = STATUS_COLOR.get(state, STATUS_COLOR[ST_SEARCHING])
        text = STATUS_TEXT.get(state, state)
        if detail:
            text += f"  {detail}"

        p.setFont(self.font_name)
        fm = QFontMetrics(self.font_name)
        w = fm.horizontalAdvance(text) + 34
        pill = QRectF(PAD, y + 4, min(w, self.width() - PAD * 2), 22)

        path = QPainterPath()
        path.addRoundedRect(pill, 11, 11)
        p.fillPath(path, QColor(12, 15, 20, 190))

        # 점은 프레임마다 밝기가 바뀐다. 멈춰 있으면 깜빡임도 멈춘다.
        dot = QColor(color)
        dot.setAlpha(255 if self._beat % 2 == 0 else 120)
        p.setBrush(QBrush(dot))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(pill.left() + 8, pill.center().y() - 4, 8, 8))

        p.setPen(QColor(220, 230, 240, 220))
        p.drawText(
            QRectF(pill.left() + 22, pill.top(), pill.width() - 26, pill.height()),
            Qt.AlignLeft | Qt.AlignVCenter,
            text,
        )

    def _paint_row(self, p: QPainter, y: int, row: dict) -> None:
        predicted = row.get("confidence") == CONF_PREDICTED
        alpha = 150 if predicted else 235
        accent = QColor(row.get("color") or "#7ec8ff")

        body = QRectF(PAD, y, self.width() - PAD * 2, ROW_H)

        # 배경
        path = QPainterPath()
        path.addRoundedRect(body, 6, 6)
        p.fillPath(path, QColor(12, 15, 20, alpha))

        # 남은시간 게이지 (배경 위에 은은하게 깔린다)
        prog = float(row.get("progress", 0.0))
        if prog > 0:
            gauge = QRectF(body)
            gauge.setWidth(body.width() * prog)
            gp = QPainterPath()
            gp.addRoundedRect(body, 6, 6)
            p.save()
            p.setClipPath(gp)
            g = QColor(accent)
            g.setAlpha(45 if predicted else 70)
            p.fillRect(gauge, g)
            p.restore()

        # 테두리: 추정치는 점선으로 구분
        pen = QPen(accent, 1.4)
        if predicted:
            pen.setStyle(Qt.DashLine)
            pen.setColor(QColor(accent.red(), accent.green(), accent.blue(), 140))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(body, 6, 6)

        # 아이콘. 등록된 원본이 있으면 그걸 그리고, 없으면 색 사각형으로 둔다.
        icon = QRectF(body.left() + 5, y + (ROW_H - ICON) / 2, ICON, ICON)
        pix = self._icon_pixmap(row.get("id"))
        if pix is not None:
            p.save()
            clip = QPainterPath()
            clip.addRoundedRect(icon, 4, 4)
            p.setClipPath(clip)
            if predicted:
                p.setOpacity(0.55)   # 추정치는 아이콘도 흐리게
            p.drawPixmap(icon.toRect(), pix)
            p.restore()
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawRoundedRect(icon, 4, 4)
        else:
            ib = QColor(accent)
            ib.setAlpha(90 if predicted else 160)
            p.setBrush(QBrush(ib))
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawRoundedRect(icon, 4, 4)

        # 남은시간 (우측 정렬, 먼저 폭을 확보)
        # 화면에서 시간을 읽지 못하면 숫자 대신 '?'를 쓴다. 0이나 추정치를
        # 보여주면 유저가 그걸 딜사이클에 쓰게 되어 없느니만 못하다.
        remaining = row.get("remaining")
        known = remaining is not None
        t_text = f"{float(remaining):.1f}" if known else "?"
        p.setFont(self.font_time)
        t_w = QFontMetrics(self.font_time).horizontalAdvance(t_text) + 6
        t_rect = QRectF(body.right() - t_w - 8, y, t_w, ROW_H)
        if known:
            p.setPen(QColor(255, 255, 255, 200 if predicted else 255))
        else:
            p.setPen(QColor(255, 255, 255, 110))   # 흐리게 = 근거 없음
        p.drawText(t_rect, Qt.AlignRight | Qt.AlignVCenter, t_text)

        # 이름 (남는 공간에 맞춰 말줄임)
        n_left = icon.right() + 8
        n_rect = QRectF(n_left, y, t_rect.left() - n_left - 6, ROW_H)
        p.setFont(self.font_name)
        name = QFontMetrics(self.font_name).elidedText(
            str(row.get("name", "")), Qt.ElideRight, int(n_rect.width())
        )
        p.setPen(QColor(230, 238, 248, 190 if predicted else 255))
        p.drawText(n_rect, Qt.AlignLeft | Qt.AlignVCenter, name)

    # -- 배치 모드 드래그 -------------------------------------------------

    def mousePressEvent(self, ev):
        if self.setup_mode and ev.button() == Qt.LeftButton:
            self._drag_from = ev.globalPosition() - QPointF(self.pos())

    def mouseMoveEvent(self, ev):
        if self._drag_from is not None:
            self.move((ev.globalPosition() - self._drag_from).toPoint())

    def mouseReleaseEvent(self, _ev):
        self._drag_from = None

    def keyPressEvent(self, ev):
        if self.setup_mode and ev.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter):
            self.close()


# ----------------------------------------------------------------------
# 더미 피드: 실제 상태 머신을 그대로 구동한다
# ----------------------------------------------------------------------

DEMO_CATALOG = {
    "emperor": {"name": "황제", "duration": 12.0, "priority": 0, "color": "#ffb300"},
    "synergy": {"name": "치명타 시너지", "duration": 8.0, "priority": 1, "color": "#4fc3f7"},
    "awakening": {"name": "각성 물약", "duration": 30.0, "priority": 2, "color": "#81c784"},
    "atk": {"name": "공격력 증가", "duration": 20.0, "priority": 3, "color": "#ba9cf5"},
}

# (시각, 관측된 버프들) — 밀림 시나리오 포함
DEMO_SCRIPT = [
    (0.0, ["awakening", "atk"]),
    (1.0, ["awakening", "atk", "emperor"]),
    (2.0, ["awakening", "atk", "emperor", "synergy"]),
    # 3초부터 awakening / atk 가 밀려서 관측에서 빠진다.
    # 그래도 타이머로 계속 표시되어야 한다.
    (3.0, ["emperor", "synergy"]),
    (7.0, ["emperor"]),
    (10.0, ["synergy"]),   # 시너지 갱신 -> 앞으로 당겨져 재관측
    (11.0, []),
]


class DemoFeed:
    def __init__(self):
        self.tracker = BuffTracker(DEMO_CATALOG)
        self.t0 = time.perf_counter()
        self.idx = 0
        self.cycle = 34.0

    def step(self) -> list[dict]:
        now = time.perf_counter() - self.t0
        if now > self.cycle:  # 루프
            self.tracker.flush()
            self.t0 = time.perf_counter()
            self.idx = 0
            now = 0.0

        obs = []
        while self.idx < len(DEMO_SCRIPT) and DEMO_SCRIPT[self.idx][0] <= now:
            obs = [Observation(b) for b in DEMO_SCRIPT[self.idx][1]]
            self.idx += 1
        self.tracker.update(obs, now)
        return self.tracker.snapshot(now)

    def status(self):
        return (ST_OK, "데모")


# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="위치 조정 모드")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    key = config.screen_key(app)

    w = BuffOverlay(setup_mode=args.setup)
    pos = load_pos(key)
    if pos:
        w.move(*pos)
    else:
        geo = app.primaryScreen().geometry()
        w.move(geo.width() - WIDTH - 40, 120)

    w.show()
    apply_win_flags(w, click_through=not args.setup)

    feed = DemoFeed()
    timer = QTimer()
    def tick():
        w.set_rows(feed.step())
        w.set_status(*feed.status())

    timer.timeout.connect(tick)
    timer.start(100)  # 10Hz

    if args.setup:
        app.aboutToQuit.connect(lambda: save_pos(key, w.x(), w.y()))
        print(f"[setup] 드래그로 위치 조정 후 Esc. 프리셋 키: {key}")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
