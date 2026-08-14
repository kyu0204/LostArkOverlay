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
    """추적 중인 버프 하나의 런타임 상태.

    expire_at이 None이면 '걸려 있는 건 아는데 남은 시간을 모른다'는 뜻이다.
    화면의 지속시간 표기를 읽지 못한 경우로, 타이머는 못 돌리지만
    버프가 있다는 사실 자체는 알려줘야 하므로 목록에는 남긴다.
    """

    buff_id: str
    expire_at: Optional[float]
    started_at: float
    last_seen: float
    total: Optional[float]
    confidence: str = CONF_OBSERVED
    # 마지막으로 화면에서 읽은 표기값. 표기가 바뀌는 순간을 잡으려면
    # 이전 값과 비교해야 한다.
    last_reading: Optional[float] = None

    @property
    def duration_known(self) -> bool:
        return self.expire_at is not None

    def remaining(self, now: float) -> Optional[float]:
        if self.expire_at is None:
            return None
        return max(0.0, self.expire_at - now)

    def progress(self, now: float) -> float:
        """1.0 = 방금 걸림, 0.0 = 만료 직전. 게이지 렌더링용."""
        rem = self.remaining(now)
        if rem is None or not self.total or self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, rem / self.total))


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
        unknown_grace: float = 0.0,
    ):
        """
        catalog: buff_id -> 정의 dict. JSON에서 그대로 로드 가능.
        predict_after: 마지막 관측 후 이 시간이 지나면 confidence를
                       predicted로 낮춘다. 10Hz 캡처의 미스 프레임을
                       흡수하기 위한 여유값.
        unknown_grace: 남은 시간을 못 읽은 버프를, 화면에서 사라진 뒤
                       이 시간까지만 목록에 남긴다. 기본 0 - 타이머가
                       없으니 안 보이는 순간 근거가 사라진다. 시간을
                       읽어둔 버프는 이 경로를 타지 않으므로 '미관측은
                       소멸 근거가 아니다'는 원칙은 그대로다.
        """
        self.catalog: Dict[str, BuffDef] = {
            bid: BuffDef.from_dict(bid, raw) for bid, raw in (catalog or {}).items()
        }
        self.predict_after = predict_after
        self.unknown_grace = unknown_grace
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

    @staticmethod
    def _display_step(value: float) -> float:
        """표기 눈금. 1초 이상은 정수, 미만은 0.1초 단위로 보인다."""
        return 1.0 if value >= 1.0 else 0.1

    def _apply_measurement(
        self, state: BuffState, measured: float, now: float
    ) -> None:
        """화면에서 읽은 남은 시간을 반영한다.

        표기 'X초'는 정확히 X초가 아니라 **[X, X+눈금) 구간**이라는 뜻이다.
        매 프레임 `expire_at = now + X`로 덮어쓰면 같은 값이 눈금 하나만큼
        계속 읽히는 동안 타이머가 리셋되어 남은 시간이 줄지 않는다.

        그래서 세 가지로 나눠 다룬다.

        1. 표기가 바뀌는 순간(X+1 -> X)을 봤다면 지금이 가장 정확하다.
           방금 X+1 구간의 아래끝을 지났으므로 남은 시간은 X.9다.
        2. 예측이 표기가 허용하는 구간 안에 있으면 그대로 흐르게 둔다.
           버프는 갱신 아니면 만료로만 사라지므로 굳이 다시 잡을 이유가 없다.
        3. 구간을 벗어났다면(갱신되었거나 추정이 어긋남) 구간 안으로 맞춘다.
           이때는 아래끝을 쓴다. 실제보다 길게 보여주면 유저가 그걸 믿고
           딜사이클을 짜므로, 짧게 보여주는 쪽이 안전하다.
        """
        step = self._display_step(measured)
        predicted = state.remaining(now)
        prev = state.last_reading
        state.last_reading = measured

        if prev is not None and abs((prev - step) - measured) < step * 0.05:
            # 1. 표기가 방금 한 눈금 내려갔다
            state.expire_at = now + measured + step * 0.9
        elif predicted is None or not (measured <= predicted < measured + step):
            # 3. 구간 밖 -> 아래끝으로 보수적으로 맞춘다
            state.expire_at = now + measured
        # 2. 구간 안이면 아무것도 하지 않는다

        state.total = max(state.total or 0.0, measured + step)

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

            # 화면에서 실제로 읽은 값인가, 카탈로그 폴백인가.
            # 이 구분이 없으면 계속 보이는 버프의 타이머가 매 프레임
            # 리셋되어 남은 시간이 줄지 않는다.
            measured = obs.remaining is not None
            duration = self._resolve_duration(obs, defn)

            existing = self._active.get(obs.buff_id)
            if existing is None:
                # 남은 시간을 몰라도 목록에는 올린다. 버프가 걸려 있다는
                # 사실 자체가 정보이고, 나중에 읽히면 그때 타이머가 붙는다.
                self._active[obs.buff_id] = BuffState(
                    buff_id=obs.buff_id,
                    expire_at=None if duration is None else now + duration,
                    started_at=now,
                    last_seen=now,
                    total=duration,
                    confidence=CONF_OBSERVED,
                    # 첫 표기값을 남겨야 다음 프레임에 '표기가 바뀌었는지'를
                    # 알 수 있다. 비워두면 전환 순간을 영영 못 잡는다.
                    last_reading=obs.remaining if measured else None,
                )
            else:
                if measured:
                    self._apply_measurement(existing, duration, now)
                # 폴백값은 '아직 걸려 있다'는 사실만 알려줄 뿐
                # 남은 시간의 근거가 못 된다. 타이머는 그대로 흐르게 둔다.
                existing.last_seen = now
                existing.confidence = CONF_OBSERVED

        self._tick(now)

    def _tick(self, now: float) -> None:
        for buff_id in list(self._active):
            state = self._active[buff_id]
            if state.expire_at is not None and now >= state.expire_at:
                del self._active[buff_id]
                continue
            if state.expire_at is None and now - state.last_seen > self.unknown_grace:
                # 남은 시간을 끝내 읽지 못한 버프. 화면에서도 사라졌으면
                # 만료됐는지 밀렸는지 알 길이 없다. 근거 없이 계속 띄우면
                # 목록에 쌓이기만 하므로 여기서 뺀다.
                # (실측 타이머가 있는 버프는 이 경로를 타지 않는다 —
                #  미관측은 소멸 근거가 아니라는 원칙은 그대로다)
                del self._active[buff_id]
                continue
            if now - state.last_seen > self.predict_after:
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
            rem = state.remaining(now)
            rows.append(
                {
                    "id": state.buff_id,
                    "name": defn.name,
                    # 남은 시간을 못 읽었으면 None. UI는 숫자 대신
                    # '모름'을 표시해야 한다. 0으로 내려보내면 곧 만료되는
                    # 것처럼 보여 없느니만 못하다.
                    "remaining": None if rem is None else round(rem, 3),
                    "duration_known": state.duration_known,
                    "total": None if state.total is None else round(state.total, 3),
                    "progress": round(state.progress(now), 4),
                    "confidence": state.confidence,
                    "priority": defn.priority,
                    "color": defn.color,
                }
            )
        # 시간을 모르는 것은 맨 아래로. 곧 끝나는 것이 위에 와야 한다.
        rows.sort(
            key=lambda r: (
                r["priority"],
                r["remaining"] is None,
                r["remaining"] if r["remaining"] is not None else 0.0,
            )
        )
        return rows

    def event(self, now: float) -> dict:
        """WebSocket으로 그대로 쏠 수 있는 형태."""
        return {"type": "buff_update", "t": now, "buffs": self.snapshot(now)}

    @property
    def active_ids(self) -> List[str]:
        return list(self._active)
