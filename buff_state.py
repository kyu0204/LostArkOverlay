"""
로스트아크 버프 오버레이 - 상태 머신 코어

설계 원칙
---------
1. 인식(observation)과 상태(state)를 분리한다.
   버프바에서 안 보인다고 해서 버프가 끝난 것이 아니다.
   밀려서 ROI 밖으로 나갔을 뿐일 수 있다.

2. 미관측은 절대 소멸 근거가 아니다.
   버프 제거 조건은 오직 `now >= expire_at` 하나뿐이다.

3. 관측될 때마다 실측값으로 보정한다.
   로아는 버프 갱신 시 앞으로 당겨지므로, 갱신은 자연히 재관측된다.

4. UI로 나가는 것은 직렬화 가능한 dict다.
   나중에 Tauri/WebSocket으로 옮길 때 UI 쪽을 건드리지 않기 위함.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


# 버프 타입
TYPE_BUFF = "buff"       # 지속시간이 있는 버프. 타이머 표시 대상
TYPE_INSTANT = "instant"  # 즉발형. 타이머 없음. 짧은 토스트만

# 신뢰도
CONF_OBSERVED = "observed"    # 지금 버프바에서 보이는 중. 실측 기반
CONF_PREDICTED = "predicted"  # 밀려서 안 보임. 최초 관측 기준 추정

DEFAULT_PRIORITY = 99


@dataclass(frozen=True)
class BuffDef:
    """카탈로그 항목. JSON에서 로드된다."""

    id: str
    name: str
    type: str = TYPE_BUFF
    duration: Optional[float] = None
    priority: int = DEFAULT_PRIORITY
    color: Optional[str] = None

    @staticmethod
    def from_dict(buff_id: str, raw: dict) -> "BuffDef":
        return BuffDef(
            id=buff_id,
            name=raw.get("name", buff_id),
            type=raw.get("type", TYPE_BUFF),
            duration=raw.get("duration"),
            priority=raw.get("priority", DEFAULT_PRIORITY),
            color=raw.get("color"),
        )


@dataclass
class BuffState:
    """추적 중인 버프 하나의 런타임 상태."""

    buff_id: str
    expire_at: float
    started_at: float
    last_seen: float
    total: float
    confidence: str = CONF_OBSERVED

    def remaining(self, now: float) -> float:
        return max(0.0, self.expire_at - now)

    def progress(self, now: float) -> float:
        """1.0 = 방금 걸림, 0.0 = 만료 직전. 게이지 렌더링용."""
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.remaining(now) / self.total))


@dataclass
class Observation:
    """한 프레임에서 인식된 버프 하나.

    remaining이 None이면 화면에서 남은 시간을 읽지 못했다는 뜻이다.
    (아이콘은 인식했으나 숫자/게이지 파싱 실패, 혹은 애초에 표시가 없음)
    이 경우 카탈로그의 duration으로 폴백한다.
    """

    buff_id: str
    remaining: Optional[float] = None


class BuffTracker:
    def __init__(
        self,
        catalog: Optional[Dict[str, dict]] = None,
        predict_after: float = 0.5,
    ):
        """
        catalog: buff_id -> 정의 dict. JSON에서 그대로 로드 가능.
        predict_after: 마지막 관측 후 이 시간이 지나면 confidence를
                       predicted로 낮춘다. 10Hz 캡처의 미스 프레임을
                       흡수하기 위한 여유값.
        """
        self.catalog: Dict[str, BuffDef] = {
            bid: BuffDef.from_dict(bid, raw) for bid, raw in (catalog or {}).items()
        }
        self.predict_after = predict_after
        self._active: Dict[str, BuffState] = {}

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _definition(self, buff_id: str) -> BuffDef:
        """카탈로그에 없는 버프도 추적은 한다. 알 수 없는 것을
        조용히 버리면 디버깅이 어려워진다."""
        return self.catalog.get(buff_id) or BuffDef(id=buff_id, name=buff_id)

    def _resolve_duration(self, obs: Observation, defn: BuffDef) -> Optional[float]:
        """실측값 우선, 없으면 카탈로그 폴백."""
        if obs.remaining is not None and obs.remaining > 0:
            return obs.remaining
        if obs.remaining is not None and obs.remaining <= 0:
            return 0.0
        return defn.duration

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def update(self, observations: Iterable[Observation], now: float) -> None:
        """한 프레임 처리.

        observations는 '이번 프레임에 실제로 보인 것'만 담는다.
        여기 없다고 해서 만료시키지 않는다는 점이 핵심이다.
        """
        for obs in observations:
            defn = self._definition(obs.buff_id)

            # 즉발형은 타이머 대상이 아니다. 별도 토스트 채널로 뺀다.
            if defn.type == TYPE_INSTANT:
                continue

            duration = self._resolve_duration(obs, defn)
            if duration is None:
                # 실측도 없고 카탈로그에도 없다. 추적 불가.
                continue

            existing = self._active.get(obs.buff_id)
            if existing is None:
                self._active[obs.buff_id] = BuffState(
                    buff_id=obs.buff_id,
                    expire_at=now + duration,
                    started_at=now,
                    last_seen=now,
                    total=duration,
                    confidence=CONF_OBSERVED,
                )
            else:
                # 보이는 동안은 항상 실측으로 덮어쓴다.
                # 갱신 시 앞으로 당겨지므로 이 경로로 자연히 처리된다.
                existing.expire_at = now + duration
                existing.last_seen = now
                # 실측 남은시간이 이전 total보다 크면 갱신된 것.
                # 게이지가 100%를 넘지 않도록 total도 올려준다.
                existing.total = max(existing.total, duration)
                existing.confidence = CONF_OBSERVED

        self._tick(now)

    def _tick(self, now: float) -> None:
        for buff_id in list(self._active):
            state = self._active[buff_id]
            if now >= state.expire_at:
                del self._active[buff_id]
            elif now - state.last_seen > self.predict_after:
                state.confidence = CONF_PREDICTED

    def flush(self) -> None:
        """사망 등으로 모든 버프가 확실히 사라졌을 때 호출."""
        self._active.clear()

    def snapshot(self, now: float) -> List[dict]:
        """UI로 보낼 직렬화 가능한 목록.

        정렬: priority 오름차순 -> 남은 시간 오름차순.
        곧 끝나는 것이 위로 온다.
        """
        self._tick(now)
        rows = []
        for state in self._active.values():
            defn = self._definition(state.buff_id)
            rows.append(
                {
                    "id": state.buff_id,
                    "name": defn.name,
                    "remaining": round(state.remaining(now), 3),
                    "total": round(state.total, 3),
                    "progress": round(state.progress(now), 4),
                    "confidence": state.confidence,
                    "priority": defn.priority,
                    "color": defn.color,
                }
            )
        rows.sort(key=lambda r: (r["priority"], r["remaining"]))
        return rows

    def event(self, now: float) -> dict:
        """WebSocket으로 그대로 쏠 수 있는 형태."""
        return {"type": "buff_update", "t": now, "buffs": self.snapshot(now)}

    @property
    def active_ids(self) -> List[str]:
        return list(self._active)
