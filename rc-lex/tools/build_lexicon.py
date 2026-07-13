#!/usr/bin/env python3
"""Офлайн-сборщик словаря `rc-lex` (запускается руками, результат коммитится).

Собирает `data/lexicon.tsv` из четырёх открытых источников (см. data/SOURCES.md):

  CEFR-J Vocabulary Profile 1.5   слово + POS + уровень A1..B2
  Octanove Vocabulary Profile     слово + POS + уровень C1..C2
  FrequencyWords (OpenSubtitles)  частотность слов → zipf
  Wiktionary (category API)       многословные единицы: фразовые глаголы, идиомы
  Tatoeba (корпус)                частотность многословных единиц → zipf → их уровень

Уровень многословных единиц ни в одном открытом словаре не размечен — выводим сами
(`cefr_source=derived`, см. `derive_mwe_cefr`), поэтому доверять ему как размеченному
CEFR-J нельзя: это оценка, а не факт.

    python3 tools/build_lexicon.py            # качает и пишет data/lexicon.tsv
    python3 tools/build_lexicon.py --cache X  # переиспользовать скачанное из X
"""

import argparse
import bz2
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
# Корпус для частотности многословных единиц: униграммный список их не покрывает, а без
# частотности уровень фразового глагола выводить не из чего (`move` — A1, но `move in` —
# точно не A1). Tatoeba: ~1.6 млн разговорных предложений, CC BY — можно коммерчески.
CORPUS_URL = "https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"

# Неправильные глаголы: `got up` должно считаться в `get up`. Правильные формы снимаются
# правилами (см. `verb_bases`), а эти — только списком. Взяты частотные головы фразовых
# глаголов; редкие пропустить не страшно — они и так уйдут в верхние уровни.
IRREGULAR = {
    "got": "get", "gotten": "get", "went": "go", "gone": "go", "came": "come",
    "took": "take", "taken": "take", "gave": "give", "given": "give", "made": "make",
    "did": "do", "done": "do", "had": "have", "was": "be", "were": "be", "been": "be",
    "put": "put", "ran": "run", "run": "run", "sat": "sit", "stood": "stand",
    "held": "hold", "kept": "keep", "brought": "bring", "broke": "break",
    "broken": "break", "cut": "cut", "found": "find", "left": "leave", "let": "let",
    "lay": "lie", "paid": "pay", "said": "say", "saw": "see", "seen": "see",
    "sent": "send", "set": "set", "shut": "shut", "spoke": "speak", "spent": "spend",
    "thought": "think", "threw": "throw", "thrown": "throw", "told": "tell",
    "wore": "wear", "won": "win", "wrote": "write", "written": "write", "grew": "grow",
    "drew": "draw", "fell": "fall", "fallen": "fall", "felt": "feel", "flew": "fly",
    "hung": "hang", "heard": "hear", "knew": "know", "known": "know", "lost": "lose",
    "meant": "mean", "met": "meet", "read": "read", "rode": "ride", "rose": "rise",
    "sold": "sell", "shot": "shoot", "sang": "sing", "slept": "sleep", "spread": "spread",
    "stuck": "stick", "swept": "sweep", "tore": "tear", "woke": "wake", "worn": "wear",
}

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


def verb_bases(token: str) -> set[str]:
    """Токен → возможные словарные формы глагола (`moving` → `move`, `got` → `get`).

    Правила снимают правильные окончания, список — неправильные формы. Лишние кандидаты
    безвредны: они просто не совпадут ни с одной единицей словаря.
    """
    out = {token}
    if token in IRREGULAR:
        out.add(IRREGULAR[token])
    if token.endswith("ies") and len(token) > 4:
        out.add(token[:-3] + "y")
    if token.endswith(("es", "ed")) and len(token) > 3:
        out.add(token[:-2])
        out.add(token[:-1])
    elif token.endswith("s") and len(token) > 2:
        out.add(token[:-1])
    if token.endswith("ing") and len(token) > 4:
        stem = token[:-3]
        out.add(stem)
        out.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:  # `putting` → `put`
            out.add(stem[:-1])
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:  # `stopped` → `stop`
            out.add(stem[:-1])
    return out


def count_mwe(blob: bytes, units: set[str]) -> dict[str, float]:
    """Частотность многословных единиц по корпусу → zipf.

    Униграммный частотный список тут бесполезен, а без частотности уровень фразового
    глагола брать неоткуда. Считаем словоформы (`got up`, `moving in` → `get up`, `move in`),
    хвост единицы (`up`, `in`, `away with`) неизменяем.
    """
    max_len = max(len(u.split()) for u in units)
    counts: dict[str, int] = {}
    total = 0
    for line in blob.decode("utf-8", "replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        tokens = [t for t in tokenize(parts[2]) if t]
        total += len(tokens)
        for i, token in enumerate(tokens):
            bases = verb_bases(token)
            for n in range(2, min(max_len, len(tokens) - i) + 1):
                tail = " ".join(tokens[i + 1 : i + n])
                for base in bases:
                    unit = f"{base} {tail}"
                    if unit in units:
                        counts[unit] = counts.get(unit, 0) + 1
                        break
    print(f"корпус: {total} токенов, найдено {len(counts)} многословных единиц", file=sys.stderr)
    return {u: round(math.log10(c / total * 1e9), 2) for u, c in counts.items()}


def tokenize(text: str) -> list[str]:
    return "".join(c.lower() if (c.isalnum() or c == "'") else " " for c in text).split()


# Уровень многословной единицы — по её РАНГУ частотности среди единиц того же типа, а не
# по абсолютному zipf: корпус небольшой, абсолютные значения сжаты, и по любому разумному
# порогу в A2 сваливается всё подряд. Ранг же устойчив.
#
# Границы откалиброваны по якорям, которые преподаватель разложит одинаково:
#   get up(25) come back(15) find out(20) wake up(34)  → A2, учат сразу;
#   look after(106) move in(173)                       → B1;
#   put up with(222) account for(308) attend to(499)   → B2;
#   дальше — хвост, который в речи не встречается.
# Идиомы сдвинуты на ступень: идиома уровня A2 — редкость, это заведомо не начальный слой.
#
# Это по-прежнему оценка (`derived`), но опирается она на частотность самой единицы, а не
# на её головной глагол: по голове `move in` выходил A2 («сначала mother/father, потом
# move in»), потому что `move` — A1.
PV_BANDS = [(50, "A2"), (200, "B1"), (800, "B2"), (2000, "C1")]
IDIOM_BANDS = [(100, "B1"), (500, "B2"), (1500, "C1")]


def rank_by_frequency(mwe: list[tuple[str, str]], mwe_zipf: dict[str, float]) -> dict[str, int]:
    """Единица → её ранг частотности среди единиц того же типа (1 — самая частая)."""
    ranks: dict[str, int] = {}
    for kind in {k for _, k in mwe}:
        units = sorted(
            (u for u, k in mwe if k == kind),
            key=lambda u: -mwe_zipf.get(u, 0.0),
        )
        ranks.update({u: i + 1 for i, u in enumerate(units)})
    return ranks


def derive_mwe_cefr(unit: str, kind: str, ranks: dict[str, int], mwe_zipf: dict[str, float]) -> str:
    """Уровень многословной единицы: из её ранга частотности в корпусе."""
    if mwe_zipf.get(unit, 0.0) <= 0.0:  # в корпусе не встретилась ни разу
        return "C2"
    bands = PV_BANDS if kind == "phrasal_verb" else IDIOM_BANDS
    rank = ranks[unit]
    for ceiling, level in bands:
        if rank <= ceiling:
            return level
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

    # Многословные единицы: сперва собираем инвентарь, потом считаем его частотность по
    # корпусу — уровень выводим уже из неё.
    mwe: list[tuple[str, str]] = []
    seen = {(r["unit"], r["kind"]) for r in rows}
    for category, kind in MWE_CATEGORIES.items():
        titles = fetch_category(category, cache)
        print(f"{category}: {len(titles)}", file=sys.stderr)
        for title in titles:
            unit = title.strip().lower()
            # Единицы вида `abate of` (архаика) не отсеиваем: они просто редко совпадут и
            # уйдут в верхние уровни. Отсеиваем только неанглийское.
            if not unit or not all(c.isalpha() or c in " -'" for c in unit):
                continue
            if " " not in unit or (unit, kind) in seen:
                continue
            seen.add((unit, kind))
            mwe.append((unit, kind))

    corpus = bz2.decompress(fetch(CORPUS_URL, cache, "eng_sentences.tsv.bz2"))
    mwe_zipf = count_mwe(corpus, {unit for unit, _ in mwe})
    ranks = rank_by_frequency(mwe, mwe_zipf)

    for unit, kind in mwe:
        prefix = "pv" if kind == "phrasal_verb" else "idm"
        rows.append(
            {
                "id": f"{prefix}:{slug(unit)}",
                "unit": unit,
                "kind": kind,
                "pos": "verb" if kind == "phrasal_verb" else "",
                "cefr": derive_mwe_cefr(unit, kind, ranks, mwe_zipf),
                "cefr_source": "derived",
                "zipf": f"{mwe_zipf.get(unit, 0.0):.2f}",
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
