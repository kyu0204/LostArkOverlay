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
