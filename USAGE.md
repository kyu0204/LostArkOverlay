# 실행 방법

모든 명령은 프로젝트 폴더 안에서 실행합니다.

## 0. 준비

```bash
pip install PySide6 mss opencv-python numpy
python -m unittest discover -p "test_*.py"   # 140개 통과하면 정상
```

**게임을 '테두리 없는 창 모드'로 실행하세요.**
전체화면에서는 외부 오버레이가 보이지 않습니다.

---

## 1. 게임 없이 지금 할 수 있는 것

### UI 확인

```bash
python main.py --no-capture
```

더미 버프가 흐르는 오버레이가 뜹니다. 밀림 시나리오까지 재현되므로
점선(추정) / 실선(관측) 구분을 눈으로 볼 수 있습니다.

Windows에서 확인할 것:
- 게임 위에서 클릭이 통과되는지
- 알트탭 후에도 계속 위에 떠 있는지
- 작업표시줄에 안 뜨는지

### 글자 템플릿 수집

이미 받아둔 캡처로 지금 진행할 수 있습니다.

```bash
# 분할이 어떻게 되는지 먼저 확인 (captures/..._glyphs.png 생성)
python glyph_tool.py preview captures/combat_1080p.png \
    --y 940 960 --x0 390 --x1 870

# 라벨 붙이기
python glyph_tool.py label captures/combat_1080p.png \
    --y 940 960 --x0 390 --x1 870

# 진행 상황
python glyph_tool.py list
```

글자가 터미널에 아스키로 뜹니다.

```
--- 1/28  x=69 폭=4 높이=8 ---
####
#..#
...#
...#
..#.
.##.
##..
####
라벨> 7
```

입력 규칙:

| 입력 | 의미 |
|---|---|
| `0`~`9` | 숫자 |
| `.` | 소수점 |
| `ch` | 초 |
| (엔터) | 건너뛰기 |
| `q` | 중단하고 저장 |

**이미 아는 글자는 자동으로 건너뜁니다.** 한 프레임에 같은 숫자가
여러 번 나오므로, 2회차부터는 물어보는 개수가 확 줄어듭니다.

분할이 어긋난 조각에 억지로 라벨을 붙이지 마세요. 건너뛰는 편이
낫습니다. `0` `3` `4` `5` `6` `.`은 이 캡처에 없거나 적으니
나중에 다른 프레임에서 채우면 됩니다.

---

## 2. 게임 접속 후

### ROI 지정

```bash
python roi_picker.py --name buffbar
python roi_picker.py --list        # 저장된 프리셋 확인
```

화면이 어두워지고 드래그로 영역을 잡습니다. 커서 주변이 6배로
확대되니 픽셀 단위로 맞출 수 있습니다.

| 조작 | 동작 |
|---|---|
| 드래그 | 영역 선택 |
| 방향키 | 1px 이동 |
| Shift+방향키 | 크기 조절 |
| Enter | 저장 |
| Esc | 취소 |

**버프 아이콘 행과 그 아래 지속시간 텍스트를 모두 포함**해야 합니다.
1920×1080 기본 배율 기준으로 `x=692 y=947 w=188 h=30` 근처입니다.

### 캡처

```bash
# 버프가 쌓였다 사라지는 과정 담기
python capture.py --frames 30 --interval 1

# 셀 분할이 맞는지 확인
python capture.py --auto
```

`--auto`가 셀 크기와 피치를 자동으로 찾아 출력합니다.

### 아이콘 등록

```bash
# 셀 번호 확인 (..._cells.png 생성)
python icon_tool.py preview captures/파일.png --y 913 941

# 등록
python icon_tool.py add captures/파일.png --y 913 941 \
    --map 3=arcana_emperor 5=arcana_moon

# 확인
python icon_tool.py list
python icon_tool.py test captures/파일.png --y 913 941
```

`--y`는 **아이콘 행만** 지정합니다 (텍스트 행 제외).
배경이 다른 장면에서 같은 버프를 2~3장씩 등록하면 더 안정적입니다.

이미 등록된 것:

| ID | 내용 |
|---|---|
| `sup_red` / `sup_gold` / `sup_blue` / `sup_purple` | 서폿 버프 4종 |
| `overflow_counter` | 버프창 초과 표시 |

### buffs.json 채우기

`icon_tool.py`로 등록한 ID와 같은 키로 이름과 지속시간을 넣습니다.

```json
{
  "sup_red": {
    "name": "용맹의 축복",
    "type": "buff",
    "duration": 10.0,
    "priority": 1,
    "color": "#ff5252"
  }
}
```

`duration`은 **인게임에서 직접 재서** 넣으세요. 각인이나 밸패에 따라
달라지므로 추정치를 쓰면 안 됩니다.

글자 템플릿을 다 모았다면 화면에서 실시간으로 읽으므로 `duration`은
폴백용이 됩니다. 그래도 채워두는 편이 안전합니다.

---

## 3. 실행

```bash
python main.py            # 오버레이
python main.py --debug    # 인식 결과를 콘솔에 출력
python main.py --setup    # 오버레이 위치 조정 (드래그 후 Esc)
```

`--debug` 출력 예시:

```
t=  12.3  보임=9  추적=12  오버플로=+3  [sup_red,sup_gold,...]
t=  12.4  보임=9  추적=12  오버플로=+3  불일치=-1  [sup_red,...]
```

- `보임` — 이번 프레임에 화면에서 인식된 수
- `추적` — 타이머가 돌고 있는 수 (밀려서 안 보이는 것 포함)
- `불일치` — 음수면 인식을 놓쳤다는 뜻, 양수면 타이머가 실제보다
  오래 살아 있다는 뜻

---

## 문제가 생기면

**"ROI가 없습니다"** → `roi_picker.py`를 먼저 실행하세요.

**"icons.json이 비어 있습니다"** → `icon_tool.py add`로 등록하세요.

**"그리드 미검출"** → ROI가 아이콘 행을 포함하는지 확인하세요.
아이콘 테두리가 잘려 있으면 검출되지 않습니다.

**오버레이가 안 보임** → 게임이 전체화면 모드인지 확인하세요.
테두리 없는 창 모드여야 합니다.

**해상도나 UI 배율을 바꿨을 때** → ROI를 다시 잡으세요.
프리셋은 `1920x1080@1` 같은 키로 분리 저장되므로 기존 설정은
그대로 남습니다.
