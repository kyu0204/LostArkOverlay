"""
글자 템플릿 수집 도구

캡처에서 지속시간 텍스트를 잘라내 라벨을 붙이고 `glyphs.json`에 쌓는다.
한 번 모아두면 이후 인식은 템플릿 대조만으로 끝난다.

실행
----
    # 1. 분할 결과를 눈으로 확인 (라벨 붙이기 전)
    python glyph_tool.py preview captures/frame.png --y 941 957

    # 2. 라벨 붙이기. 잘린 글자가 하나씩 뜨고 무엇인지 입력한다
    python glyph_tool.py label captures/frame.png --y 941 957

    # 3. 모은 템플릿 확인
    python glyph_tool.py list

라벨 입력 규칙
--------------
    0~9   숫자
    .     소수점
    ch    초
    (엔터) 건너뛰기 (글자가 아니거나 잘못 잘린 것)
    q     중단하고 저장

'분'(mn)은 기본 대상이 아니다. 추적 대상 버프가 모두 초 단위이고,
'분'과 '초'는 서로 오인되기 쉬운 형태라 아예 후보에서 뺐다.
필요해지면 --minutes 로 켤 수 있다.

분할이 어긋난 조각에 억지로 라벨을 붙이지 말 것. 건너뛰는 편이 낫다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from text_parse import (
    UNIT_MIN,
    UNIT_SEC,
    GlyphBook,
    binarize,
    group_glyphs,
    segment_glyphs,
)

LABEL_MAP = {"ch": UNIT_SEC, "mn": UNIT_MIN}
BASE_LABELS = set("0123456789.") | {UNIT_SEC}


def load_row(path: Path, y0: int, y1: int, x0: int, x1: int, thr):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"읽을 수 없습니다: {path}")
    x1 = x1 if x1 > 0 else img.shape[1]
    row = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    return row, binarize(gray, thr)


def cmd_preview(args) -> int:
    import cv2

    row, bw = load_row(Path(args.image), *args.y, args.x0, args.x1, args.thr)
    glyphs = segment_glyphs(bw)
    groups = group_glyphs(glyphs, max_gap=args.gap)

    canvas = cv2.cvtColor(bw * 255, cv2.COLOR_GRAY2BGR)
    for gi, grp in enumerate(groups):
        c = (0, 255, 0) if gi % 2 == 0 else (0, 180, 255)
        for g in grp:
            cv2.rectangle(canvas, (g.x, g.y), (g.x + g.w - 1, g.y + g.h - 1), c, 1)

    out = Path(args.image).with_name(Path(args.image).stem + "_glyphs.png")
    cv2.imwrite(str(out), cv2.resize(canvas, None, fx=args.scale, fy=args.scale,
                                     interpolation=cv2.INTER_NEAREST))
    print(f"글자 {len(glyphs)}개, 그룹 {len(groups)}개 -> {out}")
    for gi, grp in enumerate(groups):
        print(f"  g{gi}: x{grp[0].x}~{grp[-1].right} "
              f"폭={[int(g.w) for g in grp]}")
    return 0


def _show(img: np.ndarray) -> str:
    """터미널에 글자를 아스키로 그린다. 별도 뷰어 없이 라벨링하기 위함."""
    return "\n".join("".join("#" if v else "." for v in row) for row in img)


def cmd_label(args) -> int:
    row, bw = load_row(Path(args.image), *args.y, args.x0, args.x1, args.thr)
    glyphs = segment_glyphs(bw)
    if not glyphs:
        print("글자를 찾지 못했습니다. --thr 값을 조정해 보세요.")
        return 1

    valid = BASE_LABELS | ({UNIT_MIN} if args.minutes else set())
    hint = "숫자/./ch" + ("/mn" if args.minutes else "")

    book = GlyphBook.load()
    added = 0

    # 이미 확실히 아는 글자는 묻지 않는다. 한 프레임에 같은 숫자가
    # 여러 번 나오므로, 이게 없으면 45개를 전부 확인해야 한다.
    todo = []
    known = 0
    for g in glyphs:
        label, score = book.match(g.img)
        if label is not None and score >= args.skip_score:
            known += 1
            continue
        todo.append(g)

    if known:
        print(f"이미 아는 글자 {known}개는 건너뜁니다.")
    if not todo:
        print("새로 라벨을 붙일 글자가 없습니다.")
        return 0

    print(f"글자 {len(todo)}개. 라벨 입력 ({hint}, 엔터=건너뛰기, q=중단)\n")

    for i, g in enumerate(todo):
        print(f"--- {i + 1}/{len(todo)}  x={g.x} 폭={g.w} 높이={g.h} ---")
        print(_show(g.img))
        try:
            raw = input("라벨> ").strip()
        except EOFError:
            break
        if raw == "q":
            break
        if not raw:
            continue
        label = LABEL_MAP.get(raw, raw)
        if label not in valid:
            if label == UNIT_MIN:
                print("  무시: '분'은 기본 대상이 아닙니다 (--minutes 로 활성화)")
            else:
                print(f"  무시: '{raw}'는 유효한 라벨이 아닙니다")
            continue
        book.add(label, g.img)
        added += 1

    book.save()
    print(f"\n{added}개 추가. 총 {sum(len(v) for v in book.templates.values())}개 "
          f"({len(book.templates)}종) 저장됨.")
    return 0


def cmd_list(args) -> int:
    book = GlyphBook.load()
    if not book.templates:
        print("아직 모은 템플릿이 없습니다.")
        return 0
    need = BASE_LABELS | ({UNIT_MIN} if getattr(args, "minutes", False) else set())
    for label in sorted(book.templates, key=lambda s: (len(s), s)):
        print(f"  {label} : {len(book.templates[label])}개")
    missing = need - set(book.templates)
    if missing:
        print(f"\n아직 없는 글자: {' '.join(sorted(missing))}")
    else:
        print("\n필요한 글자를 모두 모았습니다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("preview", cmd_preview), ("label", cmd_label)):
        p = sub.add_parser(name)
        p.add_argument("image")
        p.add_argument("--y", nargs=2, type=int, required=True,
                       metavar=("Y0", "Y1"), help="텍스트 행의 y 범위")
        p.add_argument("--x0", type=int, default=0)
        p.add_argument("--x1", type=int, default=0, help="0이면 끝까지")
        p.add_argument("--thr", type=int, default=None,
                       help="이진화 임계값. 생략하면 Otsu")
        p.add_argument("--gap", type=int, default=3)
        p.add_argument("--scale", type=int, default=6)
        p.add_argument("--minutes", action="store_true",
                       help="'분' 라벨도 수집 (기본 제외)")
        p.add_argument("--skip-score", type=float, default=0.95,
                       help="이 점수 이상으로 매칭되면 묻지 않는다 (기본 0.95)")
        p.set_defaults(func=fn)

    p = sub.add_parser("list")
    p.add_argument("--minutes", action="store_true")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
