"""해상도/DPI 조합별 설정 저장.

같은 사람이라도 창 모드를 바꾸거나 모니터를 옮기면 좌표가 전부 어긋난다.
테스터마다 환경이 다르므로 처음부터 프리셋 키로 분리해 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_PATH = Path(__file__).with_name("app_config.json")


def screen_key(app) -> str:
    """예: '2560x1440@1.25'"""
    scr = app.primaryScreen()
    g = scr.geometry()
    return f"{g.width()}x{g.height()}@{scr.devicePixelRatio():g}"


def load_all() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        # 설정이 깨졌다고 프로그램이 죽으면 안 된다. 기본값으로 간다.
        return {}


def save_all(data: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get(key: str, section: str, name: str, default=None):
    return load_all().get(key, {}).get(section, {}).get(name, default)


def put(key: str, section: str, name: str, value: Any) -> None:
    data = load_all()
    data.setdefault(key, {}).setdefault(section, {})[name] = value
    save_all(data)


def get_section(key: str, section: str) -> Dict[str, Any]:
    return load_all().get(key, {}).get(section, {})


# -- ROI 전용 헬퍼 ------------------------------------------------------

def get_roi(key: str, name: str) -> Optional[Dict[str, int]]:
    return get(key, "roi", name)


def put_roi(key: str, name: str, rect: Dict[str, int]) -> None:
    put(key, "roi", name, rect)


# -- 버프바 기하 프로필 --------------------------------------------------
#
# 왜 프로필인가
# -------------
# 버프바의 셀 크기와 간격은 해상도와 게임 UI 배율에 함께 달려 있다.
# 같은 1920x1080에서도 배율에 따라 21px / 26px로 갈린다.
#
# 그런데 배율값(100%, 120% ...)을 픽셀로 바꾸는 표는 우리에게 없다.
# 그래서 표를 지어내지 않고, **측정한 값을 배율별로 저장해 재사용**한다.
# 쓰다 보면 표가 채워지는 구조다.
#
# 저장이 필요한 이유는 실측으로 확인됐다. 버프가 적을 때 켜면 기하
# 자체를 잘못 잡는다:
#
#     버프 15개 -> pitch 28.9  (신뢰 0.69)
#     버프  3개 -> pitch 24.0  (신뢰 0.53)
#     버프  2개 -> pitch 14.0  (신뢰 0.43)
#
# 전투 전에 앱을 켜는 것은 흔한 일이므로, 잘 측정된 값을 한 번 저장해
# 두면 그 뒤로는 버프가 없어도 정확하게 시작한다.

def profile_key(scale: Any) -> str:
    """배율값을 프로필 이름으로. 입력이 없으면 'auto'.

    dict를 주면 기하에 영향을 주는 항목만 골라 조합한다. 게임에는
    배율 손잡이가 둘 있는데(HUD 크기, 버프 크기) 어느 쪽을 바꿔도
    픽셀이 달라진다. 하나만 이름으로 쓰면 서로 다른 기하가 같은
    이름 아래 덮어써진다.

    설정 탭이 없던 시절 저장된 값과 섞이지 않게, 스칼라를 받으면
    예전 방식 그대로 둔다.
    """
    if isinstance(scale, dict):
        hud = scale.get("hud_scale") or "auto"
        buff = scale.get("ui_scale") or "auto"
        return f"hud{hud}-buff{buff}"
    return "auto" if scale in (None, "") else str(scale)


def get_profile(key: str, scale: Any = None) -> Optional[Dict[str, Any]]:
    return get(key, "profile", profile_key(scale))


def put_profile(key: str, scale: Any, geom: Dict[str, Any]) -> None:
    put(key, "profile", profile_key(scale), geom)


def get_settings(key: str) -> Dict[str, Any]:
    """사용자가 설정 탭에서 넣은 값 (해상도/배율/칸수)."""
    return get_section(key, "settings")


def put_settings(key: str, values: Dict[str, Any]) -> None:
    data = load_all()
    data.setdefault(key, {})["settings"] = values
    save_all(data)
