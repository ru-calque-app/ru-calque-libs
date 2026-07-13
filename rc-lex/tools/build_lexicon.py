#!/usr/bin/env python3
"""Офлайн-сборщик словаря `rc-lex` (запускается руками, результат коммитится).

Собирает `data/lexicon.tsv` из четырёх открытых источников (см. data/SOURCES.md):

  CEFR-J Vocabulary Profile 1.5   слово + POS + уровень A1..B2
  Octanove Vocabulary Profile     слово + POS + уровень C1..C2
  FrequencyWords (OpenSubtitles)  частотность → zipf
  Wiktionary (category API)       многословные единицы: фразовые глаголы, идиомы

Уровень многословных единиц ни в одном открытом словаре не размечен — выводим сами
(`cefr_source=derived`, см. `derive_mwe_cefr`), поэтому доверять ему как размеченному
CEFR-J нельзя: это оценка, а не факт.

    python3 tools/build_lexicon.py            # качает и пишет data/lexicon.tsv
    python3 tools/build_lexicon.py --cache X  # переиспользовать скачанное из X
"""

import argparse
import csv
import io
import json
import math
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Wikimedia требует контактный User-Agent и режет частые запросы (429).
USER_AGENT = "ru-calque-lexicon-builder/0.1 (https://github.com/ru-calque-app; gussman7777@gmail.com)"
THROTTLE_S = 0.5

CEFRJ_URL = "https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master/cefrj-vocabulary-profile-1.5.csv"
OCTANOVE_URL = "https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master/octanove-vocabulary-profile-c1c2-1.0.csv"
FREQ_URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"

# Категории Wiktionary → kind в словаре.
MWE_CATEGORIES = {
    "Category:English phrasal verbs": "phrasal_verb",
    "Category:English idioms": "idiom",
}

# POS в CEFR-J/Octanove пишутся словами; сводим к короткому набору.
POS_MAP = {
    "noun": "noun",
    "verb": "verb",
    "adjective": "adj",
    "adverb": "adv",
    "preposition": "prep",
    "pronoun": "pron",
    "determiner": "det",
    "conjunction": "conj",
    "interjection": "intj",
    "number": "num",
    "modal verb": "verb",
    "auxiliary verb": "verb",
}

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def fetch(url: str, cache: pathlib.Path | None, name: str) -> bytes:
    if cache:
        hit = cache / name
        if hit.exists():
            return hit.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            break
        except urllib.error.HTTPError as err:
            if err.code != 429 or attempt == 4:
                raise
            time.sleep(2**attempt)
    time.sleep(THROTTLE_S)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(body)
    return body


def load_cefr(blob: bytes) -> dict[tuple[str, str], str]:
    """(unit, pos) → уровень. Орфографические варианты через `/` — отдельные ключи."""
    out: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))):
        level = (row.get("CEFR") or "").strip().upper()
        pos = POS_MAP.get((row.get("pos") or "").strip().lower())
        if level not in LEVELS or not pos:
            continue
        for variant in (row.get("headword") or "").split("/"):
            unit = variant.strip().lower()
            if not unit:
                continue
            prev = out.get((unit, pos))
            # Слово может встретиться дважды — держим самый ранний уровень: он
            # определяет, когда ученик впервые обязан его знать.
            if prev is None or LEVELS.index(level) < LEVELS.index(prev):
                out[(unit, pos)] = level
    return out


def load_zipf(blob: bytes) -> dict[str, float]:
    """слово → zipf (log10 вхождений на миллиард)."""
    counts: dict[str, int] = {}
    for line in blob.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        counts[parts[0]] = int(parts[1])
    total = sum(counts.values())
    return {w: round(math.log10(c / total * 1e9), 2) for w, c in counts.items()}


def fetch_category(category: str, cache: pathlib.Path | None) -> list[str]:
    """Все страницы категории Wiktionary (основное пространство имён)."""
    titles: list[str] = []
    cont: dict[str, str] = {}
    page = 0
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmnamespace": "0",
            "format": "json",
            **cont,
        }
        name = f"wikt-{category.split(':')[1].replace(' ', '_')}-{page}.json"
        url = f"{WIKTIONARY_API}?{urllib.parse.urlencode(params)}"
        data = json.loads(fetch(url, cache, name))
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" not in data:
            return titles
        cont = data["continue"]
        page += 1


def derive_mwe_cefr(unit: str, cefr: dict[tuple[str, str], str], zipf: dict[str, float]) -> str:
    """Уровень многословной единицы: по головному слову, но на ступень выше.

    Фразовый глагол сложнее своих частей (`get` — A1, `get over` — точно не A1):
    значение неразложимо, и это ровно то, на чём ученик буксует. Головное слово вне
    словаря → идём от частотности; совсем редкое → C2.
    """
    head = unit.split()[0]
    levels = [lvl for (word, _), lvl in cefr.items() if word == head]
    if levels:
        base = min(levels, key=LEVELS.index)
        return LEVELS[min(LEVELS.index(base) + 1, len(LEVELS) - 1)]
    z = zipf.get(head, 0.0)
    if z >= 5.0:
        return "B1"
    if z >= 4.0:
        return "B2"
    if z >= 3.0:
        return "C1"
    return "C2"


def slug(unit: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in unit.lower())


def build(cache: pathlib.Path | None) -> list[dict[str, str]]:
    cefr = load_cefr(fetch(CEFRJ_URL, cache, "cefrj.csv"))
    for key, level in load_cefr(fetch(OCTANOVE_URL, cache, "octanove.csv")).items():
        cefr.setdefault(key, level)
    zipf = load_zipf(fetch(FREQ_URL, cache, "en_50k.txt"))
    print(f"cefr: {len(cefr)} записей, zipf: {len(zipf)} слов", file=sys.stderr)

    rows: list[dict[str, str]] = []
    for (unit, pos), level in sorted(cefr.items()):
        rows.append(
            {
                "id": f"w:{slug(unit)}|{pos}",
                "unit": unit,
                "kind": "word",
                "pos": pos,
                "cefr": level,
                "cefr_source": "cefrj",
                "zipf": f"{zipf.get(unit, 0.0):.2f}",
            }
        )

    seen = {(r["unit"], r["kind"]) for r in rows}
    for category, kind in MWE_CATEGORIES.items():
        titles = fetch_category(category, cache)
        print(f"{category}: {len(titles)}", file=sys.stderr)
        for title in titles:
            unit = title.strip().lower()
            # Единицы вида `abate of` (архаика) и мусор с заглавными/пунктуацией не
            # отсеиваем: они просто редко совпадут. Отсеиваем только неанглийское.
            if not unit or not all(c.isalpha() or c in " -'" for c in unit):
                continue
            if " " not in unit or (unit, kind) in seen:
                continue
            seen.add((unit, kind))
            prefix = "pv" if kind == "phrasal_verb" else "idm"
            rows.append(
                {
                    "id": f"{prefix}:{slug(unit)}",
                    "unit": unit,
                    "kind": kind,
                    "pos": "verb" if kind == "phrasal_verb" else "",
                    "cefr": derive_mwe_cefr(unit, cefr, zipf),
                    "cefr_source": "derived",
                    "zipf": f"{zipf.get(unit, 0.0):.2f}",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, help="каталог со скачанными источниками")
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "data" / "lexicon.tsv",
    )
    args = ap.parse_args()

    rows = build(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "unit", "kind", "pos", "cefr", "cefr_source", "zipf"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{args.out}: {len(rows)} единиц {kinds}", file=sys.stderr)


if __name__ == "__main__":
    main()
