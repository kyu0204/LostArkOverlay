"""
아이콘 DB 수집 도구

캡처에서 버프 아이콘을 잘라 이름을 붙이고 `icons.json`에 쌓는다.

실행
----
    # 1. 셀 분할이 맞는지 먼저 확인 (타일 이미지가 저장된다)
    python icon_tool.py preview captures/frame.png --y 913 941

    # 2. 셀 번호를 지정해 등록
    python icon_tool.py add captures/frame.png --y 913 941 \\
        --cell 10 --id sup_purple

    # 여러 개를 한 번에
    python icon_tool.py add captures/frame.png --y 913 941 \\
        --map 10=sup_purple 11=sup_blue 12=sup_gold 13=sup_red

    # 3. 등록된 DB 확인 / 프레임 전체 인식 테스트
    python icon_tool.py list
    python icon_tool.py test captures/frame.png --y 913 941

같은 버프를 여러 프레임에서 등록하면 샘플이 쌓여 인식이 안정된다.
배경이 다른 장면에서 2~3장씩 등록하는 것을 권한다.

버프가 아닌 것이 확실한 칸은 --ignore 로 등록해두면 결과에서
자동으로 빠진다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grid_detect import detect_grid
from icon_match import ICON_IMG_DIR, IconBook, identify_cells


def load_row(path: Path, y0: int, y1: int, x0: int, x1: int):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"읽을 수 없습니다: {path}")
    x1 = x1 if x1 > 0 else img.shape[1]
    return img[y0:y1, x0:x1]


def resolve_grid(row, args):
    if args.grid:
        from capture import parse_grid
        from grid_detect import Grid

        cell, gap, off = parse_grid(args.grid)
        pitch = cell + gap
        count = 0
        while int(round(off + count * pitch)) + cell <= row.shape[1]:
            count += 1
        return Grid(left=off, cell=cell, pitch=pitch, count=count)

    g = detect_grid(row)
    if g is None:
        raise SystemExit(
            "그리드 자동 검출 실패. --y 로 아이콘 행만 잘랐는지 확인하거나\n"
            "--grid 21x2.33+0 형식으로 직접 지정하세요."
        )
    return g


def cmd_preview(args) -> int:
    import cv2
    import numpy as np

    row = load_row(Path(args.image), *args.y, args.x0, args.x1)
    g = resolve_grid(row, args)
    print(f"검출: {g}")

    tiles, h = [], row.shape[0]
    for i, (l, r) in enumerate(g.bounds()):
        t = row[:, l:r]
        big = cv2.resize(t, None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_NEAREST)
        label = np.full((22, big.shape[1], 3), 25, np.uint8)
        cv2.putText(label, str(i), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (120, 220, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label, big]))

    gap = np.full((tiles[0].shape[0], 6, 3), 40, np.uint8)
    canvas = tiles[0]
    for t in tiles[1:]:
        canvas = np.hstack([canvas, gap, t])

    out = Path(args.image).with_name(Path(args.image).stem + "_cells.png")
    cv2.imwrite(str(out), canvas)
    print(f"셀 {g.count}개 -> {out}")
    print("이미지에서 번호를 확인한 뒤 add 로 등록하세요.")
    return 0


def save_thumb(buff_id: str, tile) -> None:
    """오버레이에 띄울 아이콘 원본을 남긴다.

    해시는 되돌릴 수 없으므로, 등록 시점의 타일을 그대로 보관해야
    오버레이가 실제 아이콘을 그릴 수 있다. 같은 버프를 여러 번
    등록하면 마지막 것으로 덮어쓴다.
    """
    import cv2

    ICON_IMG_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(ICON_IMG_DIR / f"{buff_id}.png"), tile)


def cmd_add(args) -> int:
    row = load_row(Path(args.image), *args.y, args.x0, args.x1)
    g = resolve_grid(row, args)
    bounds = g.bounds()

    pairs = []
    if args.map:
        for item in args.map:
            if "=" not in item:
                print(f"  무시: '{item}' (형식은 셀번호=이름)")
                continue
            k, v = item.split("=", 1)
            pairs.append((int(k), v))
    if args.cell is not None:
        if not args.id:
            raise SystemExit("--cell 에는 --id 가 필요합니다.")
        pairs.append((args.cell, args.id))
    if not pairs:
        raise SystemExit("--cell/--id 또는 --map 중 하나가 필요합니다.")

    book = IconBook.load()
    n = 0
    for idx, buff_id in pairs:
        if not 0 <= idx < len(bounds):
            print(f"  건너뜀: 셀 {idx}는 범위 밖 (0~{len(bounds) - 1})")
            continue
        l, r = bounds[idx]
        tile = row[args.inset:row.shape[0] - args.inset,
                   l + args.inset:r - args.inset]
        if tile.size == 0:
            print(f"  건너뜀: 셀 {idx} 크롭이 비었습니다")
            continue
        book.add(buff_id, tile, ignore=args.ignore)
        if not args.ignore:
            # 표시용은 inset을 적용하지 않는다. 해시는 테두리를 빼는 편이
            # 안정적이지만, 눈으로 보는 아이콘은 잘리지 않은 편이 낫다.
            save_thumb(buff_id, row[:, l:r])
        print(f"  셀 {idx} -> {buff_id}" + ("  (무시 대상)" if args.ignore else ""))
        n += 1

    book.save()
    total = sum(len(e.hashes) for e in book.entries.values())
    print(f"\n{n}개 등록. DB: {len(book.entries)}종 / 샘플 {total}개")
    return 0


def cmd_list(_args) -> int:
    book = IconBook.load()
    if not book.entries:
        print("아직 등록된 아이콘이 없습니다.")
        return 0
    for bid in sorted(book.entries):
        e = book.entries[bid]
        hues = [f"{h:.0f}" for h in e.hues if h is not None]
        tag = "  [무시]" if e.ignore else ""
        print(f"  {bid:16s} 샘플 {len(e.hashes)}개  "
              f"Hue {'/'.join(hues) if hues else '무채색'}{tag}")
    return 0


def cmd_test(args) -> int:
    row = load_row(Path(args.image), *args.y, args.x0, args.x1)
    g = resolve_grid(row, args)
    book = IconBook.load()
    if not book.entries:
        print("DB가 비어 있습니다. 먼저 add 로 등록하세요.")
        return 1

    res = identify_cells(row, g.bounds(), book, inset=args.inset)
    print(f"셀 {g.count}개 중 {len(res)}개 인식\n")
    for i in range(g.count):
        r = res.get(i)
        if r is None:
            print(f"  셀 {i:2d}: -")
        else:
            print(f"  셀 {i:2d}: {r.buff_id:16s} 해밍={r.distance:2d} "
                  f"Hue차={r.hue_gap:5.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("image")
        p.add_argument("--y", nargs=2, type=int, required=True,
                       metavar=("Y0", "Y1"), help="아이콘 행의 y 범위")
        p.add_argument("--x0", type=int, default=0)
        p.add_argument("--x1", type=int, default=0, help="0이면 끝까지")
        p.add_argument("--grid", help="자동 검출 대신 직접 지정 (예: 26x3+2)")
        p.add_argument("--inset", type=int, default=2,
                       help="테두리를 몇 px 잘라낼지 (기본 2)")

    p = sub.add_parser("preview")
    common(p)
    p.add_argument("--scale", type=int, default=6)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("add")
    common(p)
    p.add_argument("--cell", type=int)
    p.add_argument("--id")
    p.add_argument("--map", nargs="+", metavar="N=ID")
    p.add_argument("--ignore", action="store_true",
                   help="버프가 아닌 것이 확실한 칸으로 등록. 정체를 모르면 쓰지 말 것")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("test")
    common(p)
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
