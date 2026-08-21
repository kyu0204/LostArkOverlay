"""
학습용 캡처 자동 수집

버프창이 처음 잡히는 순간부터 몇 분간 주기적으로 떠서 쌓는다.
아이콘/글자 템플릿이 지금 캡처 두 장에서만 나와 배경이 바뀌면
무너지는데, 그걸 고치려면 배경이 다양한 실전 캡처가 있어야 한다.

실행
----
    python collect.py                    # 첫 인식까지 대기 -> 3분 촬영
    python collect.py --minutes 2
    python collect.py --now              # 대기 없이 바로 촬영
    python collect.py --region 470,900,460,70

게임 중에는 콘솔로 못 돌아오므로 정해진 시간이 지나면 알아서 멈춘다.
Ctrl+C 로도 멈춘다.

첫 인식을 기다리면 로딩 화면이나 메뉴가 안 쌓인다. 다만 **인식이
안 되는 환경을 찍으려고 도는 도구**라 탐지에만 기대면 정작 필요한
장면을 못 건진다. 그래서 대기 시간이 지나면 안 잡혀도 촬영을
시작한다. 빈손으로 끝나는 경우를 없애는 쪽이 중요하다.

PNG로 저장한다. JPEG는 획 가장자리를 뭉개서 글자 템플릿에 못 쓴다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

OUT_DIR = Path(__file__).with_name("captures") / "auto"

# 버프창은 화면 하단 중앙, 스킬바 위에 뜬다. 1080p 기준 y 900~965이므로
# 높이의 78~92% 구간을 폭 전체로 뜨면 해상도가 달라도 들어온다.
# 폭을 안 자르는 이유: 버프가 늘면 창이 좌우로 자라고, 초과 표시(+)가
# 왼쪽 끝에 붙는다. 잘라 두면 나중에 기하를 다시 못 잡는다.
BAND_TOP, BAND_BOTTOM = 0.78, 0.92


# ----------------------------------------------------------------------
# 순수 함수 (화면 없이 테스트 가능)
# ----------------------------------------------------------------------

def band_region(width: int, height: int,
                top: float = BAND_TOP, bottom: float = BAND_BOTTOM) -> dict:
    """화면 크기 -> 버프창이 들어있을 띠 영역."""
    if width <= 0 or height <= 0:
        raise ValueError("화면 크기가 잘못됐습니다")
    if not 0.0 <= top < bottom <= 1.0:
        raise ValueError("띠 구간이 잘못됐습니다")
    y0 = int(height * top)
    y1 = int(height * bottom)
    return {"x": 0, "y": y0, "w": width, "h": max(1, y1 - y0)}


def parse_region(spec: str) -> dict:
    """'470,900,460,70' -> {x, y, w, h}"""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"영역 형식이 잘못됐습니다: {spec!r} (x,y,w,h)")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"영역은 정수 넷이어야 합니다: {spec!r}")
    if w <= 0 or h <= 0:
        raise ValueError("폭과 높이는 1 이상이어야 합니다")
    return {"x": x, "y": y, "w": w, "h": h}


def frame_diff(a: Optional[np.ndarray], b: np.ndarray) -> float:
    """두 프레임의 평균 절대 차이. 앞이 없으면 무한대(=반드시 저장)."""
    if a is None or a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def is_duplicate(prev: Optional[np.ndarray], cur: np.ndarray,
                 tol: float = 1.5) -> bool:
    """직전에 **저장한** 프레임과 사실상 같은가.

    문턱을 낮게 잡는다. 몇 초 간격이면 초 표기가 이미 바뀌어 있으므로,
    걸러내고 싶은 것은 '알탭했거나 로딩 중이라 화면이 멈춘' 경우뿐이다.
    비슷한 장면을 공격적으로 버리면 배경 다양성까지 같이 날아간다.
    """
    return frame_diff(prev, cur) < tol


def is_blank(frame: np.ndarray, tol: float = 3.0) -> bool:
    """거의 단색인가. 로딩 화면이나 암전 구간을 거른다."""
    return float(frame.std()) < tol


WAIT, RECORD, DONE = "대기", "촬영", "종료"


class Session:
    """대기 -> 촬영 -> 종료. 시간과 탐지 여부만 보고 다음 행동을 정한다.

    화면도 파일도 건드리지 않아서 그대로 테스트할 수 있다. 촬영을
    언제 시작하고 끝낼지가 이 도구의 유일한 판단이므로 따로 뺐다.
    """

    def __init__(self, minutes: float = 3.0, warmup: float = 300.0,
                 start_now: bool = False):
        self.duration = minutes * 60.0
        self.warmup = warmup
        self.start_now = start_now
        self.state = RECORD if start_now else WAIT
        self.started_at: Optional[float] = None
        self.trigger: Optional[str] = "즉시" if start_now else None

    def begin(self, now: float) -> None:
        """루프 시작 시각을 알려준다."""
        self.origin = now
        if self.state is RECORD:
            self.started_at = now

    def step(self, now: float, detected: bool) -> str:
        """'건너뜀' | '저장' | '종료'"""
        if self.state is DONE:
            return "종료"

        if self.state is WAIT:
            if detected:
                self.trigger = "버프창 인식"
            elif now - self.origin >= self.warmup:
                # 못 찾아도 시작한다. 인식이 안 되는 장면을 찍으려고
                # 도는 도구라, 탐지 실패가 곧 촬영 실패가 되면 안 된다.
                self.trigger = "대기 시간 초과 (인식 없이 시작)"
            else:
                return "건너뜀"
            self.state = RECORD
            self.started_at = now
            return "저장"

        if now - self.started_at >= self.duration:
            self.state = DONE
            return "종료"
        return "저장"


# ----------------------------------------------------------------------
# 수집 루프
# ----------------------------------------------------------------------

def load_user_settings():
    """설정 탭(buff_editor.py)에 등록해 둔 값을 읽는다.

    화면 키를 앱과 **똑같은 방법으로** 만들어야 해서 Qt를 띄운다.
    mss는 물리 해상도를, Qt는 논리 해상도와 배율을 보므로 키가
    갈린다(이 PC에서 1920x1080 물리 = 1536x864@1.25 논리).
    직접 계산하면 어긋나서 저장해 둔 설정을 못 찾는다.

    반환: (화면키, 설정 dict, 저장된 buffbar ROI 또는 None)
    """
    try:
        from PySide6.QtWidgets import QApplication

        import config

        app = QApplication.instance() or QApplication([])
        key = config.screen_key(app)
        return key, config.get_settings(key) or {}, config.get_roi(key, "buffbar")
    except Exception:
        # 설정을 못 읽어도 수집은 돌아야 한다. 칸수는 없으면 추정한다.
        return None, {}, None


def foreground_title() -> str:
    """지금 맨 앞에 있는 창 제목. Windows가 아니면 빈 문자열."""
    try:
        import ctypes

        u = ctypes.windll.user32
        h = u.GetForegroundWindow()
        n = u.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(h, buf, n + 1)
        return buf.value or ""
    except Exception:
        return ""


def _detect(band: np.ndarray, slots: Optional[int]) -> bool:
    """띠 안에 버프창으로 보이는 주기 구조가 있는가."""
    try:
        from grid_detect import locate_buff_bar
        # 이미 하단만 잘라 넘기므로 band_top은 0으로 둔다.
        return locate_buff_bar(band, slots=slots, band_top=0.0) is not None
    except Exception:
        # 탐지가 죽어도 수집은 계속되어야 한다. 대기 시간이 지나면
        # 어차피 촬영이 시작된다.
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="학습용 캡처 자동 수집")
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="첫 인식 후 몇 분간 촬영할지")
    ap.add_argument("--interval", type=float, default=5.0, help="촬영 간격(초)")
    ap.add_argument("--warmup", type=float, default=300.0,
                    help="인식을 이만큼 기다린 뒤에는 안 잡혀도 시작")
    ap.add_argument("--now", action="store_true", help="대기 없이 바로 촬영")
    ap.add_argument("--slots", type=int,
                    help="버프 표시 칸수. 없으면 설정 탭 등록값을 씀")
    ap.add_argument("--roi", action="store_true",
                    help="하단 띠 대신 등록된 buffbar ROI 주변을 뜬다")
    ap.add_argument("--margin", type=int, default=30,
                    help="--roi 일 때 사방으로 둘 여유(px)")
    ap.add_argument("--max", type=int, default=200, help="최대 장수")
    ap.add_argument("--full", action="store_true", help="화면 전체를 뜬다")
    ap.add_argument("--region", help="직접 지정: x,y,w,h")
    ap.add_argument("--tol", type=float, default=1.5,
                    help="이 값보다 차이가 작으면 같은 장면으로 보고 건너뜀")
    ap.add_argument("--window", default="LOST ARK",
                    help="이 글자가 든 창이 맨 앞일 때만 찍는다. 빈 문자열이면 항상")
    ap.add_argument("--out", default=str(OUT_DIR), help="저장 폴더")
    args = ap.parse_args()

    if args.interval <= 0:
        print("간격은 0보다 커야 합니다", file=sys.stderr)
        return 2
    if args.minutes <= 0:
        print("촬영 시간은 0보다 커야 합니다", file=sys.stderr)
        return 2

    import cv2
    import mss

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 설정 탭에 등록해 둔 값이 있으면 쓴다. 명령줄 인자가 항상 우선.
    key, settings, saved_roi = load_user_settings()
    slots = args.slots if args.slots else settings.get("slots")
    if key:
        if settings:
            print(f"등록된 설정 [{key}]: 배율 {settings.get('ui_scale') or '자동'}, "
                  f"칸수 {settings.get('slots') or '미지정'}")
        else:
            print(f"등록된 설정 없음 [{key}] - buff_editor.py 설정 탭에서 넣을 수 있습니다")
    if args.slots and settings.get("slots") and args.slots != settings["slots"]:
        print(f"  (명령줄 --slots {args.slots} 가 등록값 {settings['slots']} 보다 우선)")

    with mss.mss() as sct:
        mon = sct.monitors[1]
        if args.region:
            try:
                region = parse_region(args.region)
            except ValueError as e:
                print(e, file=sys.stderr)
                return 2
            how = "직접 지정"
        elif args.full:
            region = {"x": mon["left"], "y": mon["top"],
                      "w": mon["width"], "h": mon["height"]}
            how = "화면 전체"
        elif args.roi and saved_roi:
            # 저장된 ROI는 인식용이라 딱 맞게 잘려 있다. 수집용으로는
            # 여유를 준다. 나중에 기하를 다시 잡으려면 창 밖이 보여야 한다.
            m = args.margin
            region = {"x": max(0, saved_roi["x"] - m), "y": max(0, saved_roi["y"] - m),
                      "w": saved_roi["w"] + 2 * m, "h": saved_roi["h"] + 2 * m}
            how = f"등록된 ROI + 여유 {m}px"
        else:
            if args.roi and not saved_roi:
                print("등록된 ROI가 없어 하단 띠로 대신합니다.")
            region = band_region(mon["width"], mon["height"])
            region["x"] += mon["left"]
            region["y"] += mon["top"]
            how = f"하단 띠 (높이의 {BAND_TOP:.0%}~{BAND_BOTTOM:.0%})"

        print(f"화면 {mon['width']}x{mon['height']}")
        print(f"촬영 영역: {region['w']}x{region['h']} @ "
              f"({region['x']}, {region['y']})  [{how}]")
        if args.now:
            print(f"바로 촬영 시작. {args.minutes:g}분간 {args.interval:g}초 간격.")
        else:
            print(f"버프창이 잡히면 {args.minutes:g}분간 {args.interval:g}초 간격으로 촬영.")
            print(f"{args.warmup:g}초 안에 못 잡으면 그냥 시작합니다.")
        print(f"저장 위치: {out_dir}   (Ctrl+C 로 중단)")
        print()

        grab_box = {"left": region["x"], "top": region["y"],
                    "width": region["w"], "height": region["h"]}
        sess = Session(args.minutes, args.warmup, args.now)
        sess.begin(time.monotonic())
        prev: Optional[np.ndarray] = None
        saved = skipped = 0
        announced = args.now
        # 대기 중에는 자주 확인해야 시작 시점을 놓치지 않는다.
        wait_poll = min(1.0, args.interval)

        try:
            while saved < args.max:
                started = time.monotonic()
                try:
                    frame = np.array(sct.grab(grab_box))[:, :, :3]
                except Exception as e:
                    # 해상도 변경이나 모니터 분리로 영역이 화면 밖이 될 수 있다.
                    # 한 프레임 실패로 수집 전체를 끝낼 이유는 없다.
                    print(f"  캡처 실패, 건너뜀: {e}")
                    time.sleep(args.interval)
                    continue

                waiting = sess.state is WAIT
                detected = (not waiting) or _detect(frame, slots)
                action = sess.step(started, detected)

                if action == "종료":
                    print(f"\n{args.minutes:g}분이 지나 정지합니다.")
                    break

                if action == "저장" and args.window and args.window not in foreground_title():
                    # 알탭해서 다른 창을 보고 있으면 그 화면이 찍힌다.
                    # 학습용 자료에 섞이면 나중에 골라내기 번거롭다.
                    skipped += 1
                elif action == "저장":
                    if not announced:
                        print(f"촬영 시작 - {sess.trigger}")
                        announced = True
                    if is_blank(frame) or is_duplicate(prev, frame, args.tol):
                        skipped += 1
                    else:
                        name = time.strftime("buffbar_%Y%m%d_%H%M%S.png")
                        cv2.imwrite(str(out_dir / name), frame)
                        prev = frame
                        saved += 1
                        print(f"  [{saved:3d}] {name}", flush=True)

                gap = wait_poll if sess.state is WAIT else args.interval
                rest = gap - (time.monotonic() - started)
                if rest > 0:
                    time.sleep(rest)
        except KeyboardInterrupt:
            print("\n중단했습니다.")

        print(f"\n저장 {saved}장 / 건너뜀 {skipped}장  ->  {out_dir}")
        if saved == 0:
            print("한 장도 못 건졌습니다. --now 로 다시 해보세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
