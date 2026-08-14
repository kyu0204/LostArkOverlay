"""
로스트아크 버프 오버레이 - 실행 진입점

    python main.py                # 오버레이 실행
    python main.py --setup        # 오버레이 위치 조정
    python main.py --debug        # 인식 결과를 콘솔에 출력
    python main.py --no-capture   # 더미 피드 (게임 없이 UI만 확인)

전제
----
게임을 '테두리 없는 창 모드'로 실행할 것.
전체화면에서는 외부 오버레이가 표시되지 않는다.

먼저 ROI를 잡아두어야 한다:
    python roi_picker.py --name buffbar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import config
from buff_state import BuffTracker
from icon_match import IconBook
from pipeline import Recognizer, consistency_gap
from text_parse import GlyphBook

CATALOG_PATH = Path(__file__).with_name("buffs.json")
FPS = 10


def load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        return {}
    try:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[경고] buffs.json을 읽지 못했습니다: {e}")
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


class CaptureFeed:
    """ROI를 실제로 캡처해 인식한다."""

    def __init__(self, roi: dict, tracker: BuffTracker, debug: bool = False):
        import mss

        self.roi = roi
        self.tracker = tracker
        self.debug = debug
        self.rec = Recognizer()
        self._sct = mss.mss()
        self.t0 = time.perf_counter()
        self._warned_empty = False

        if not self.rec.icons.entries:
            print("[경고] icons.json이 비어 있습니다. "
                  "icon_tool.py로 아이콘을 먼저 등록하세요.")
        if not self.rec.glyphs.templates:
            print("[안내] glyphs.json이 비어 있어 남은 시간은 "
                  "buffs.json의 duration으로 대체됩니다.")

    def step(self):
        import numpy as np

        raw = self._sct.grab({
            "left": self.roi["x"], "top": self.roi["y"],
            "width": self.roi["w"], "height": self.roi["h"],
        })
        frame = np.array(raw)[:, :, :3]

        now = time.perf_counter() - self.t0
        res = self.rec.read_frame(frame)
        self.tracker.update(res.observations, now)

        if self.debug:
            self._log(res, now)
        return self.tracker.snapshot(now)

    def _log(self, res, now: float) -> None:
        if res.grid is None:
            if not self._warned_empty:
                print("[디버그] 그리드 미검출. ROI가 아이콘 행을 "
                      "포함하는지 확인하세요.")
                self._warned_empty = True
            return
        self._warned_empty = False

        gap = consistency_gap(res, len(self.tracker.active_ids))
        parts = [
            f"t={now:6.1f}",
            f"보임={res.visible}",
            f"추적={len(self.tracker.active_ids)}",
        ]
        if res.overflow is not None:
            parts.append(f"오버플로=+{res.overflow}")
        if gap is not None and gap != 0:
            parts.append(f"불일치={gap:+d}")
        ids = ",".join(o.buff_id for o in res.observations) or "-"
        print("  ".join(parts) + f"  [{ids}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true", help="오버레이 위치 조정")
    ap.add_argument("--debug", action="store_true", help="인식 결과 출력")
    ap.add_argument("--no-capture", action="store_true",
                    help="캡처 없이 더미 피드로 UI만 확인")
    ap.add_argument("--roi", default="buffbar")
    args = ap.parse_args()

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from overlay import WIDTH, BuffOverlay, DemoFeed, apply_win_flags, load_pos, save_pos

    app = QApplication(sys.argv)
    key = config.screen_key(app)

    if args.no_capture or args.setup:
        feed = DemoFeed()
    else:
        roi = config.get_roi(key, args.roi)
        if roi is None:
            print(f"[{key}] '{args.roi}' ROI가 없습니다. 먼저 실행하세요:")
            print(f"    python roi_picker.py --name {args.roi}")
            return 1
        tracker = BuffTracker(load_catalog())
        try:
            feed = CaptureFeed(roi, tracker, debug=args.debug)
        except ImportError:
            print("mss가 필요합니다:  pip install mss")
            return 1

    w = BuffOverlay(setup_mode=args.setup)
    pos = load_pos(key)
    if pos:
        w.move(*pos)
    else:
        geo = app.primaryScreen().geometry()
        w.move(geo.width() - WIDTH - 40, 120)

    w.show()
    apply_win_flags(w, click_through=not args.setup)

    timer = QTimer()
    timer.timeout.connect(lambda: w.set_rows(feed.step()))
    timer.start(int(1000 / FPS))

    if args.setup:
        app.aboutToQuit.connect(lambda: save_pos(key, w.x(), w.y()))
        print(f"[setup] 드래그로 위치 조정 후 Esc. 프리셋 키: {key}")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
