"""
버프 정보 편집기

등록된 아이콘을 눈으로 보면서 이름/종류/지속시간을 고친다.

    python buff_editor.py

왜 GUI인가
----------
`icon_tool.py add` 로 등록할 때 붙이는 ID는 셀 번호에 기대는 임시
이름이라 나중에 무엇이 무엇인지 알 수 없다. 아이콘을 보지 않고
`buffs.json`을 직접 고치면 엉뚱한 버프에 이름을 붙이게 된다.
썸네일을 함께 띄우는 이유가 이것이다.

ID를 바꾸면 `icons.json`과 `icon_images/`의 파일명까지 함께 옮긴다.
세 곳이 어긋나면 인식은 되는데 이름이 안 뜨는 상태가 되기 때문이다.

`type`
------
    buff     타이머가 도는 일반 버프
    instant  즉발형. 타이머 대상에서 제외된다

고정 지속시간은 다루지 않는다. 같은 버프라도 어떤 스킬로 걸었느냐에
따라 길이가 달라져 고정값은 틀린 정보가 되기 때문이다. 남은 시간은
화면의 지속시간 표기를 읽어서만 채운다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from icon_match import ICON_DB_PATH, ICON_IMG_DIR
from pipeline import OVERFLOW_ID

CATALOG_PATH = Path(__file__).with_name("buffs.json")

TYPES = ["buff", "instant"]
THUMB = 40


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class BuffRow:
    """한 버프의 편집 위젯 묶음."""

    def __init__(self, buff_id: str, entry: dict, has_icon: bool):
        self.orig_id = buff_id
        self.has_icon = has_icon

        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB, THUMB)
        self.thumb.setAlignment(Qt.AlignCenter)
        path = ICON_IMG_DIR / f"{buff_id}.png"
        if path.exists():
            pix = QPixmap(str(path))
            if not pix.isNull():
                self.thumb.setPixmap(
                    pix.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.FastTransformation)
                )
        else:
            self.thumb.setText("—")
            self.thumb.setStyleSheet("color:#666;")

        self.id_edit = QLineEdit(buff_id)
        self.id_edit.setFixedWidth(150)

        self.name_edit = QLineEdit(str(entry.get("name") or ""))
        self.name_edit.setPlaceholderText("버프 이름")

        self.type_box = QComboBox()
        self.type_box.addItems(TYPES)
        t = entry.get("type") or "buff"
        self.type_box.setCurrentIndex(TYPES.index(t) if t in TYPES else 0)
        self.type_box.setFixedWidth(80)

        self.pri_edit = QLineEdit(str(entry.get("priority", 99)))
        self.pri_edit.setFixedWidth(50)

        self.color = str(entry.get("color") or "#7ec8ff")
        self.color_btn = QPushButton()
        self.color_btn.setFixedWidth(60)
        self._paint_color()
        self.color_btn.clicked.connect(self._pick_color)

        self.del_btn = QPushButton("삭제")
        self.del_btn.setFixedWidth(55)
        self.deleted = False

    def _paint_color(self) -> None:
        self.color_btn.setText(self.color)
        self.color_btn.setStyleSheet(
            f"background:{self.color}; color:#000; font-size:10px;"
        )

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.color))
        if c.isValid():
            self.color = c.name()
            self._paint_color()

    def widgets(self):
        return [
            self.thumb, self.id_edit, self.name_edit, self.type_box,
            self.pri_edit, self.color_btn, self.del_btn,
        ]

    def collect(self):
        """(새 ID, 엔트리 dict). 값이 잘못됐으면 ValueError."""
        new_id = self.id_edit.text().strip()
        if not new_id:
            raise ValueError(f"'{self.orig_id}': ID는 비울 수 없습니다")

        try:
            priority = int(self.pri_edit.text().strip() or 99)
        except ValueError:
            raise ValueError(f"'{new_id}': 우선순위가 숫자가 아닙니다")

        return new_id, {
            "name": self.name_edit.text().strip() or new_id,
            "type": self.type_box.currentText(),
            # 고정 지속시간은 두지 않는다. 같은 버프라도 어떤 스킬로
            # 걸었느냐에 따라 길이가 달라져 고정값은 틀린 정보가 된다.
            # 남은 시간은 화면 표기를 읽어서만 채운다.
            "duration": None,
            "priority": priority,
            "color": self.color,
        }


class Editor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("버프 정보 편집")
        self.resize(880, 620)

        self.catalog = load_json(CATALOG_PATH)
        self.icons = load_json(ICON_DB_PATH)
        self.comment = self.catalog.get("_comment")

        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(
            "등록된 아이콘을 보고 이름을 붙이세요.  "
            "남은 시간은 화면 표기를 읽어서만 채웁니다 (고정 지속시간 없음)."
        ))

        head = QHBoxLayout()
        for text, w in (("아이콘", THUMB + 8), ("ID", 155), ("이름", 200),
                        ("종류", 85), ("우선", 55),
                        ("색상", 65), ("", 60)):
            lb = QLabel(text)
            lb.setFixedWidth(w)
            lb.setStyleSheet("color:#888; font-size:11px;")
            head.addWidget(lb)
        head.addStretch()
        outer.addLayout(head)

        inner = QWidget()
        self.grid = QGridLayout(inner)
        self.grid.setVerticalSpacing(4)
        self.rows: list[BuffRow] = []

        # 아이콘이 등록된 것을 먼저, 그다음 카탈로그에만 있는 것.
        # 무시 대상(물약 퀵슬롯 등)과 오버플로 카운터는 버프가 아니므로 뺀다.
        # 카운터 칸의 숫자는 남은 시간이 아니라 '밀린 버프 수'라서
        # 파이프라인이 따로 해석한다. 이름을 붙일 대상이 아니다.
        icon_ids = [k for k, v in self.icons.items()
                    if not v.get("ignore") and k != OVERFLOW_ID]
        other_ids = [k for k in self.catalog
                     if not k.startswith("_") and k not in icon_ids
                     and k != OVERFLOW_ID]

        for bid in icon_ids + other_ids:
            self._add_row(bid, self.catalog.get(bid, {}), bid in icon_ids)

        self.grid.setRowStretch(self.grid.rowCount(), 1)
        area = QScrollArea()
        area.setWidget(inner)
        area.setWidgetResizable(True)
        outer.addWidget(area, 1)

        bottom = QHBoxLayout()
        self.status = QLabel("")
        bottom.addWidget(self.status, 1)
        save = QPushButton("저장")
        save.setFixedWidth(90)
        save.clicked.connect(self.save)
        bottom.addWidget(save)
        outer.addLayout(bottom)

    def _add_row(self, buff_id: str, entry: dict, has_icon: bool) -> None:
        row = BuffRow(buff_id, entry, has_icon)
        r = self.grid.rowCount()
        for c, w in enumerate(row.widgets()):
            self.grid.addWidget(w, r, c)
        row.del_btn.clicked.connect(lambda: self._delete(row))
        self.rows.append(row)

    def _delete(self, row: BuffRow) -> None:
        ok = QMessageBox.question(
            self, "삭제",
            f"'{row.orig_id}'를 목록에서 지웁니다.\n"
            "저장할 때 아이콘 등록도 함께 지워집니다. 계속할까요?",
        )
        if ok != QMessageBox.Yes:
            return
        row.deleted = True
        for w in row.widgets():
            w.setVisible(False)

    def save(self) -> None:
        try:
            collected = [(r, *r.collect()) for r in self.rows if not r.deleted]
        except ValueError as e:
            QMessageBox.warning(self, "확인 필요", str(e))
            return

        ids = [new_id for _, new_id, _ in collected]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            QMessageBox.warning(
                self, "확인 필요", f"ID가 겹칩니다: {', '.join(sorted(dupes))}"
            )
            return

        catalog = {"_comment": self.comment} if self.comment else {}
        icons = dict(self.icons)
        # 편집 대상이 아닌 항목은 손대지 않고 그대로 남긴다
        keep_ignored = {k: v for k, v in self.icons.items()
                        if v.get("ignore") or k == OVERFLOW_ID}

        renames: list[tuple[str, str]] = []
        for row, new_id, entry in collected:
            catalog[new_id] = entry
            if new_id != row.orig_id and row.has_icon:
                renames.append((row.orig_id, new_id))

        # 아이콘 DB: 삭제 반영 + ID 변경 반영
        alive = {new_id for _, new_id, _ in collected}
        new_icons = dict(keep_ignored)
        for row, new_id, _ in collected:
            if row.has_icon and row.orig_id in icons:
                new_icons[new_id] = icons[row.orig_id]

        try:
            CATALOG_PATH.write_text(
                json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            ICON_DB_PATH.write_text(
                json.dumps(new_icons, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            self._move_thumbs(renames, alive, keep_ignored)
        except Exception as e:
            QMessageBox.critical(self, "저장 실패", str(e))
            return

        # 저장 후 상태를 반영해 두어야 연속 저장이 어긋나지 않는다
        self.catalog, self.icons = catalog, new_icons
        for row, new_id, _ in collected:
            row.orig_id = new_id

        n_del = sum(1 for r in self.rows if r.deleted)
        msg = f"저장했습니다. 버프 {len(collected)}종"
        if renames:
            msg += f", 이름 변경 {len(renames)}건"
        if n_del:
            msg += f", 삭제 {n_del}건"
        self.status.setText(msg + "  (앱을 다시 켜면 반영됩니다)")

    def _move_thumbs(self, renames, alive, keep_ignored) -> None:
        """썸네일 파일을 ID 변경/삭제에 맞춘다."""
        if not ICON_IMG_DIR.exists():
            return
        for old, new in renames:
            src = ICON_IMG_DIR / f"{old}.png"
            if src.exists():
                src.replace(ICON_IMG_DIR / f"{new}.png")
        for f in ICON_IMG_DIR.glob("*.png"):
            if f.stem not in alive and f.stem not in keep_ignored:
                f.unlink()


class SettingsTab(QWidget):
    """해상도 / 버프 UI 배율 / 표시 칸수.

    왜 배율을 물어보는가
    --------------------
    셀 크기와 간격은 해상도와 게임 UI 배율에 함께 달려 있다. 같은
    1920x1080에서도 배율에 따라 21px / 26px로 갈린다.

    다만 배율값(100%, 120% ...)을 픽셀로 바꾸는 표는 없다. 그래서 배율은
    **프로필 이름**으로만 쓰고, 실제 픽셀 값은 측정해서 그 이름 아래에
    저장한다. 쓰다 보면 배율별 표가 채워진다.

    저장이 필요한 이유는 실측으로 확인됐다. 버프가 적을 때 켜면 기하를
    잘못 잡는다(버프 2개에서 pitch 14). 전투 전에 켜는 것은 흔한 일이고,
    한 번 잘못 잡으면 위상이 고정돼 그 세션 내내 인식이 죽는다.
    저장된 값이 있으면 버프가 없어도 정확하게 시작한다.
    """

    def __init__(self):
        super().__init__()
        import config

        self.config = config
        app = QApplication.instance()
        self.key = config.screen_key(app)
        saved = config.get_settings(self.key)

        form = QVBoxLayout(self)
        form.addWidget(QLabel(
            "게임 환경을 알려주면 그 조합으로 측정값을 따로 저장합니다.\n"
            "저장된 값이 있으면 버프가 하나도 없을 때 켜도 정확하게 시작합니다."
        ))

        grid = QGridLayout()
        detected = "-"
        if app is not None:
            geo = app.primaryScreen().geometry()
            detected = f"{geo.width()}x{geo.height()}"

        grid.addWidget(QLabel("해상도"), 0, 0)
        self.res_edit = QLineEdit(str(saved.get("resolution") or detected))
        self.res_edit.setFixedWidth(160)
        grid.addWidget(self.res_edit, 0, 1)
        grid.addWidget(QLabel(f"(자동 감지: {detected} / 프리셋 키: {self.key})"), 0, 2)

        grid.addWidget(QLabel("버프 UI 배율"), 1, 0)
        self.scale_edit = QLineEdit(str(saved.get("ui_scale") or ""))
        self.scale_edit.setPlaceholderText("예: 100")
        self.scale_edit.setFixedWidth(160)
        grid.addWidget(self.scale_edit, 1, 1)
        grid.addWidget(QLabel("게임 설정의 UI 배율. 측정값을 이 이름으로 저장합니다"), 1, 2)

        grid.addWidget(QLabel("버프 표시 칸수"), 2, 0)
        self.slots_edit = QLineEdit(str(saved.get("slots") or ""))
        self.slots_edit.setPlaceholderText("예: 16")
        self.slots_edit.setFixedWidth(160)
        grid.addWidget(self.slots_edit, 2, 1)
        grid.addWidget(QLabel("화면만 봐서는 알 수 없는 값이라 직접 받습니다"), 2, 2)
        form.addLayout(grid)

        self.geom_label = QLabel()
        self.geom_label.setStyleSheet("color:#888;")
        form.addWidget(self.geom_label)
        self._refresh_geom()

        row = QHBoxLayout()
        save = QPushButton("설정 저장")
        save.setFixedWidth(110)
        save.clicked.connect(self.save)
        row.addWidget(save)

        clear = QPushButton("측정값 지우기")
        clear.setFixedWidth(130)
        clear.clicked.connect(self.clear_geometry)
        row.addWidget(clear)

        self.status = QLabel("")
        row.addWidget(self.status, 1)
        form.addLayout(row)
        form.addStretch()

    def _scale(self):
        return self.scale_edit.text().strip() or None

    def _refresh_geom(self) -> None:
        g = self.config.get_profile(self.key, self._scale())
        if g:
            self.geom_label.setText(
                f"저장된 측정값: 셀 {g.get('cell')}px, 피치 {g.get('pitch')}, "
                f"세로오프셋 {g.get('icon_top')}, 위상 {g.get('phase')}"
            )
        else:
            self.geom_label.setText(
                "저장된 측정값 없음 — 버프가 여러 개 보일 때 앱을 켜면 그때 측정해 저장합니다."
            )

    def save(self) -> None:
        slots = self.slots_edit.text().strip()
        if slots:
            try:
                if int(slots) <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "확인 필요", "칸수는 1 이상의 숫자여야 합니다.")
                return
        self.config.put_settings(self.key, {
            "resolution": self.res_edit.text().strip(),
            "ui_scale": self._scale(),
            "slots": int(slots) if slots else None,
        })
        self._refresh_geom()
        self.status.setText("저장했습니다. 앱을 다시 켜면 반영됩니다.")

    def clear_geometry(self) -> None:
        """측정값만 버린다. UI 배율을 바꿨을 때 쓴다."""
        self.config.put_profile(self.key, self._scale(), {})
        self._refresh_geom()
        self.status.setText("측정값을 지웠습니다. 다음 실행에 다시 측정합니다.")


def main() -> int:
    app = QApplication(sys.argv)
    tabs = QTabWidget()
    tabs.setWindowTitle("로스트아크 버프 오버레이 설정")
    tabs.resize(920, 640)
    tabs.addTab(Editor(), "버프")
    tabs.addTab(SettingsTab(), "설정")
    tabs.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
