#!/usr/bin/env python3
"""Сборка `data/egp.jsonl` из выгрузки English Grammar Profile Online (.xlsx).

Источник — не сеть, а локальный файл: EGP не отдаётся по стабильному URL, его выгружают
руками с englishprofile.org. Поэтому путь к .xlsx — обязательный аргумент, а не константа.

    python3 tools/build_egp.py ~/Downloads/'English Grammar Profile Online.xlsx'

Требует openpyxl (`pip install openpyxl`). Результат коммитится: файл вкомпилирован в
крейт через include_str!, в рантайме ни сети, ни python не нужно.

ВНИМАНИЕ по лицензии — см. data/SOURCES.md. EGP принадлежит Cambridge; соседний rc-lex
сознательно отказался от English Vocabulary Profile из того же семейства.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter

LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}

# Хвостовая скобка примера: "(Germany; C2 MASTERY; 1993; German; Pass)".
# Состав полей плавает — бывает без страны, без года, иногда только L1 ("(Arabic - Other)").
# Поэтому поля не позиционные, а распознаются по виду.
META_RE = re.compile(r"\(([^()]*)\)\s*$")
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
LEVEL_RE = re.compile(r"^([A-C][12])\b")


def slug(s: str) -> str:
    """ASCII-слаг для идентификатора: 'REPORTED SPEECH' -> 'reported_speech'."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def parse_example(block: str) -> dict | None:
    """Один блок примера -> {text, level?, l1?, pass?}.

    Метаданные выкусываются из хвостовой скобки; если её нет, остаётся голый текст —
    это нормально, у части записей примеры без атрибуции.
    """
    block = block.strip()
    if not block:
        return None
    out: dict = {}
    m = META_RE.search(block)
    if m:
        text = block[: m.start()].strip()
        rest = []
        for field in (f.strip() for f in m.group(1).split(";")):
            if not field:
                continue
            if field in ("Pass", "Fail"):
                out["pass"] = field == "Pass"
            elif YEAR_RE.match(field):
                pass  # год не нужен никому из потребителей
            elif (lv := LEVEL_RE.match(field)) and lv.group(1) in LEVELS:
                out["level"] = lv.group(1)
            else:
                rest.append(field)
        # Из оставшихся полей последнее — родной язык автора, предыдущее — страна экзамена.
        # Страна нам не нужна, L1 нужен: он показывает, на каком L1 конструкция «взлетает».
        if rest:
            out["l1"] = rest[-1]
    else:
        text = block
    if not text:
        return None
    out["text"] = text
    return out


def parse_guideword(gw: str) -> tuple[str, str]:
    """Guideword -> (aspect, topic).

    EGP размечает записи префиксом FORM / USE / FORM-USE. Аспект важен практически: FORM
    описывает форму (её в принципе можно искать в тексте синтаксически), USE — намерение
    говорящего, которое из текста детерминированно не достаётся.
    """
    head, sep, tail = gw.partition(":")
    if not sep:
        return "other", gw.strip()
    key = head.strip().upper()
    aspect = {"FORM": "form", "USE": "use", "FORM/USE": "form_use"}.get(key)
    if aspect is None:
        return "other", gw.strip()
    return aspect, tail.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="выгрузка English Grammar Profile Online")
    ap.add_argument("-o", "--out", default="data/egp.jsonl")
    args = ap.parse_args()

    try:
        import openpyxl
    except ImportError:
        print("нужен openpyxl: pip install openpyxl", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    ws = wb["Data"]
    rows = ws.iter_rows(values_only=True)
    header = [str(c or "") for c in next(rows)]
    idx = {name: i for i, name in enumerate(header)}
    need = ["id", "SuperCategory", "SubCategory", "Level", "Lexical Range", "Guideword",
            "Can-do statement", "Example"]
    if missing := [n for n in need if n not in idx]:
        print(f"в выгрузке нет колонок: {missing}; есть {header}", file=sys.stderr)
        return 1

    seen: Counter[str] = Counter()
    records = []
    for row in rows:
        def cell(name: str) -> str:
            v = row[idx[name]]
            return "" if v is None else str(v).strip()

        if not cell("id"):
            continue
        level = cell("Level")
        if level not in LEVELS:
            print(f"пропущен уровень {level!r} у {cell('id')}", file=sys.stderr)
            continue
        cat, sub = cell("SuperCategory"), cell("SubCategory")
        aspect, topic = parse_guideword(cell("Guideword"))

        # Собственный ключ вместо родного `1741163706316x198445876...`: он читается в
        # маппинге на construct_key и в диффах. Родной сохраняем отдельно — по нему
        # сверяется пересборка, если EGP поменяет порядок строк.
        group = f"egp:{slug(cat)}.{slug(sub)}.{level.lower()}"
        seen[group] += 1
        rec = {
            "id": f"{group}.{seen[group]:02d}",
            "src_id": cell("id"),
            "level": level,
            "category": cat,
            "subcategory": sub,
            "aspect": aspect,
            "topic": topic,
            "can_do": cell("Can-do statement"),
        }
        if (lr := cell("Lexical Range")) and lr != "N/A":
            rec["lexical_range"] = int(float(lr))
        examples = [e for b in cell("Example").split("\n\n") if (e := parse_example(b))]
        if examples:
            rec["examples"] = examples
        records.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    by_level = Counter(r["level"] for r in records)
    by_aspect = Counter(r["aspect"] for r in records)
    print(f"{args.out}: {len(records)} записей")
    print("  уровни:", dict(sorted(by_level.items())))
    print("  аспекты:", dict(by_aspect.most_common()))
    print("  с примерами:", sum(1 for r in records if r.get("examples")))
    print("  ids уникальны:", len({r["id"] for r in records}) == len(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
