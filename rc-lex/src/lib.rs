//! Лексический слой ru-calque: английский текст → семантические единицы со сложностью.
//!
//! Зачем: грамматической разметки мало. «When can I move in? Are utilities included in the
//! rent?» грамматически проста, но `utilities` — B2, и без словаря система считает такой
//! текст лёгким, а ученика — готовым. Крейт даёт объективную лексическую сторону:
//!
//! - **единица, а не слово**: `move in` — одна единица, а не `move` + `in`;
//! - **CEFR и частотность** на каждую единицу — из открытых словарей (см. `data/SOURCES.md`);
//! - **лемма**: `moved in` → `move in`, `utilities` → `utility`.
//!
//! Используется в двух местах: goals размечает текст при импорте колоды, eval разбирает
//! попытку ученика (какие единицы использованы, какие обойдены).
//!
//! ```
//! let a = rc_lex::analyze("Are utilities included in the rent?");
//! assert!(a.units.iter().any(|u| u.unit == "utility" && u.cefr == rc_lex::Cefr::B2));
//! ```
//!
//! Чего крейт **не** делает (сознательно): не размечает грамматику (нет POS-теггера) и не
//! ловит валентность вида `be included in` — это не лексема, а модель управления глагола,
//! её нет ни в одном открытом словаре. Оба слоя живут выше, в разметке текста.
//!
//! Известное ограничение (см. тест `literal_use_of_a_prepositional_verb_over_matches`):
//! Wiktionary держит в одной категории и настоящие фразовые глаголы (`move in`), и
//! предложные (`go to`, `attend to`), поэтому буквальное «go to the shop» матчится как
//! единица. Различить их можно только по разбору предложения — вернёмся к этому, когда
//! появится POS-теггер; пока лишняя единица просто быстро уходит в «знакомо».

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use wordnet_lemmatizer::{Lemmatizer, Pos};

/// Словарь: собран `tools/build_lexicon.py`, вкомпилирован в бинарь.
const LEXICON_TSV: &str = include_str!("../data/lexicon.tsv");

/// Максимальная длина многословной единицы в токенах (`get away with` — 3, запас на 4).
const MAX_MWE_LEN: usize = 4;

/// Уровень CEFR. Порядок значим: `A1 < C2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Cefr {
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
}

impl Cefr {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "A1" => Self::A1,
            "A2" => Self::A2,
            "B1" => Self::B1,
            "B2" => Self::B2,
            "C1" => Self::C1,
            "C2" => Self::C2,
            _ => return None,
        })
    }
}

/// Тип семантической единицы.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Word,
    PhrasalVerb,
    Idiom,
}

/// Единица словаря.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lexeme {
    /// Стабильный ключ: `w:utility|noun`, `pv:move_in`. По нему ссылается `user_lexicon`.
    pub id: String,
    /// Каноничная форма: `utility`, `move in`.
    pub unit: String,
    pub kind: Kind,
    /// Часть речи (у идиом пусто).
    pub pos: Option<String>,
    pub cefr: Cefr,
    /// `true` — уровень выведен нами (у многословных единиц его нет ни в одном словаре),
    /// а не взят из CEFR-J. Оценка, а не факт: на неё нельзя опираться как на разметку.
    pub cefr_derived: bool,
    /// Частотность (log10 вхождений на миллиард). `0.0` — единицы нет в частотном списке;
    /// у многословных так всегда: список униграммный.
    pub zipf: f32,
}

/// Найденная в тексте единица.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Found {
    pub id: String,
    pub unit: String,
    pub kind: Kind,
    pub cefr: Cefr,
    pub cefr_derived: bool,
    pub zipf: f32,
    /// Как единица выглядела в тексте (`utilities`, `moved in`).
    pub surface: String,
}

/// Разбор текста.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    /// Единицы в порядке появления, без повторов.
    pub units: Vec<Found>,
    /// Слова, которых нет в словаре (имена, опечатки, узкая лексика).
    pub unknown: Vec<String>,
}

impl Analysis {
    /// Лексическая сложность текста: уровень, на котором ученик поймёт почти всё.
    ///
    /// Берём 90-й перцентиль по знаменательным единицам, а не максимум: одно редкое слово
    /// не должно делать текст C2 — но и медиана не годится, она прячет ровно те единицы,
    /// на которых ученик и споткнётся.
    pub fn difficulty(&self) -> Option<Cefr> {
        let mut levels: Vec<Cefr> = self
            .units
            .iter()
            .filter(|u| u.is_content())
            .map(|u| u.cefr)
            .collect();
        if levels.is_empty() {
            return None;
        }
        levels.sort_unstable();
        let idx = ((levels.len() as f32 * 0.9).ceil() as usize).saturating_sub(1);
        Some(levels[idx])
    }
}

impl Found {
    /// Знаменательная единица (не артикль/предлог/местоимение): служебные слова есть в
    /// каждом тексте и о сложности не говорят ничего.
    fn is_content(&self) -> bool {
        !matches!(self.pos_str(), "det" | "prep" | "pron" | "conj" | "num")
    }

    fn pos_str(&self) -> &str {
        self.id.split('|').nth(1).unwrap_or("")
    }
}

/// Словарь целиком (~25k единиц), парсится один раз.
pub struct Dict {
    by_unit: HashMap<String, Vec<Lexeme>>,
    lemmatizer: Lemmatizer,
}

/// Глобальный словарь.
pub fn dict() -> &'static Dict {
    static DICT: OnceLock<Dict> = OnceLock::new();
    DICT.get_or_init(Dict::load)
}

/// Разобрать английский текст на единицы (сахар над [`Dict::analyze`]).
pub fn analyze(text: &str) -> Analysis {
    dict().analyze(text)
}

impl Dict {
    fn load() -> Self {
        let mut by_unit: HashMap<String, Vec<Lexeme>> = HashMap::new();
        for line in LEXICON_TSV.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let (Some(cefr), 7) = (f.get(4).and_then(|s| Cefr::parse(s)), f.len()) else {
                continue;
            };
            let lex = Lexeme {
                id: f[0].to_string(),
                unit: f[1].to_string(),
                kind: match f[2] {
                    "phrasal_verb" => Kind::PhrasalVerb,
                    "idiom" => Kind::Idiom,
                    _ => Kind::Word,
                },
                pos: (!f[3].is_empty()).then(|| f[3].to_string()),
                cefr,
                cefr_derived: f[5] == "derived",
                zipf: f[6].parse().unwrap_or(0.0),
            };
            by_unit.entry(lex.unit.clone()).or_default().push(lex);
        }
        Self {
            by_unit,
            lemmatizer: Lemmatizer::embedded(),
        }
    }

    /// Единица по каноничной форме. Омонимы (`cover` глагол/существительное) различаем не
    /// POS-теггером, которого нет, а частотностью: берём самое частое чтение. При равной
    /// частотности (у многословных она всегда нулевая — частотный список униграммный)
    /// решает тип: `go to` есть и в фразовых глаголах, и в идиомах, и без явного порядка
    /// выбор зависел бы от порядка строк в файле.
    pub fn lookup(&self, unit: &str) -> Option<&Lexeme> {
        self.by_unit.get(unit)?.iter().max_by(|a, b| {
            a.zipf
                .total_cmp(&b.zipf)
                .then(rank(b.kind).cmp(&rank(a.kind)))
        })
    }

    /// Все единицы словаря — для сида таблицы `lexemes` в сервисе.
    pub fn lexemes(&self) -> impl Iterator<Item = &Lexeme> {
        self.by_unit.values().flatten()
    }

    /// Разобрать текст. Многословные единицы ищем жадно, от длинных к коротким: `move in`
    /// должен победить `move`, иначе фразовый глагол рассыплется на составляющие и мы
    /// решим, что ученику здесь нечего учить.
    pub fn analyze(&self, text: &str) -> Analysis {
        let tokens = tokenize(text);
        let mut units: Vec<Found> = Vec::new();
        let mut unknown: Vec<String> = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            let mut hit = None;
            for len in (1..=MAX_MWE_LEN.min(tokens.len() - i)).rev() {
                let window = &tokens[i..i + len];
                if let Some(lex) = self.match_window(window) {
                    hit = Some((lex, window.join(" "), len));
                    break;
                }
            }
            match hit {
                Some((lex, surface, len)) => {
                    if !units.iter().any(|u| u.id == lex.id) {
                        units.push(Found {
                            id: lex.id.clone(),
                            unit: lex.unit.clone(),
                            kind: lex.kind,
                            cefr: lex.cefr,
                            cefr_derived: lex.cefr_derived,
                            zipf: lex.zipf,
                            surface,
                        });
                    }
                    i += len;
                }
                None => {
                    let w = tokens[i].clone();
                    if !unknown.contains(&w) {
                        unknown.push(w);
                    }
                    i += 1;
                }
            }
        }
        Analysis { units, unknown }
    }

    /// Окно токенов → единица: сперва как есть, потом с леммой головного слова
    /// (`moved in` → `move in`, `utilities` → `utility`).
    fn match_window(&self, window: &[String]) -> Option<&Lexeme> {
        let surface = window.join(" ");
        if let Some(lex) = self.lookup(&surface) {
            return Some(lex);
        }
        // Голова единицы — первый токен: именно он склоняется/спрягается, хвост
        // (`in`, `after`, `up with`) неизменяем.
        let head = &window[0];
        for pos in [Pos::Verb, Pos::Noun, Pos::Adj, Pos::Adv] {
            let lemma = self.lemmatizer.lemmatize(head, pos);
            if lemma == *head {
                continue;
            }
            let candidate = if window.len() == 1 {
                lemma
            } else {
                format!("{lemma} {}", window[1..].join(" "))
            };
            if let Some(lex) = self.lookup(&candidate) {
                return Some(lex);
            }
        }
        None
    }
}

/// Приоритет типа при равной частотности: чем меньше, тем предпочтительнее.
fn rank(kind: Kind) -> u8 {
    match kind {
        Kind::Word => 0,
        Kind::PhrasalVerb => 1,
        Kind::Idiom => 2,
    }
}

/// Токенизация: слова в нижнем регистре, апостроф внутри слова сохраняем (`don't`).
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Текст из колоды про аренду: грамматика простая, лексика — нет.
    const RENT: &str =
        "When can I move in? Are utilities, internet and heating included in the rent?";

    fn found<'a>(a: &'a Analysis, unit: &str) -> &'a Found {
        a.units
            .iter()
            .find(|u| u.unit == unit)
            .unwrap_or_else(|| panic!("нет единицы {unit}: {:?}", a.units))
    }

    #[test]
    fn phrasal_verb_is_one_unit_not_two_words() {
        let a = analyze(RENT);
        let pv = found(&a, "move in");
        assert_eq!(pv.kind, Kind::PhrasalVerb);
        assert_eq!(pv.id, "pv:move_in");
        // Голова единицы не должна отдельно попасть в разбор: иначе мы бы решили, что текст
        // тренирует `move`, и не заметили фразовый глагол. (`in` в разборе есть — но из
        // `included in the rent`, это другое вхождение.)
        assert!(!a.units.iter().any(|u| u.unit == "move"));
    }

    /// Обратная сторона жадного матчинга: `go to the shop` — буквальное «пойти в», но в
    /// Wiktionary `go to` лежит в той же категории, что и настоящие фразовые глаголы.
    /// Тест фиксирует поведение, чтобы оно было видно, а не всплыло сюрпризом.
    #[test]
    fn literal_use_of_a_prepositional_verb_over_matches() {
        let a = analyze("I can go to the shop.");
        assert_eq!(found(&a, "go to").kind, Kind::PhrasalVerb);
    }

    #[test]
    fn inflected_forms_map_to_the_canonical_unit() {
        let a = analyze(RENT);
        assert_eq!(found(&a, "utility").surface, "utilities");
        assert_eq!(found(&a, "utility").id, "w:utility|noun");
        assert_eq!(analyze("He moved in yesterday.").units[1].unit, "move in");
    }

    /// Главное, ради чего всё затевалось: простой по грамматике текст не должен выглядеть
    /// лёгким, если в нём B2-лексика.
    #[test]
    fn lexical_difficulty_is_driven_by_the_rare_word() {
        let a = analyze(RENT);
        assert_eq!(found(&a, "utility").cefr, Cefr::B2);
        assert_eq!(found(&a, "rent").cefr, Cefr::A2);
        assert_eq!(a.difficulty(), Some(Cefr::B2));
    }

    /// Служебные слова (`the`, `in`, `I`) есть в любом тексте — сложность должны задавать
    /// знаменательные единицы. Здесь потолок держит `shop` (B1 по CEFR-J), а не `the` (A1).
    #[test]
    fn function_words_do_not_drive_difficulty() {
        let a = analyze("I can go to the shop.");
        assert_eq!(found(&a, "the").cefr, Cefr::A1);
        assert_eq!(a.difficulty(), Some(Cefr::B1));
    }

    #[test]
    fn unknown_words_are_reported_not_swallowed() {
        let a = analyze("Kropotkin visited Tbilisi.");
        assert!(a.unknown.iter().any(|w| w == "kropotkin"));
    }

    /// Уровень фразового глагола должен идти от частотности САМОГО глагола, а не от его
    /// головы. Пока он выводился по голове, `move in` получал A2 (`move` — A1): выходило,
    /// что фразовый глагол учат сразу после `mother`/`father`. Якоря — порядок, в котором
    /// эти единицы реально появляются в обучении.
    #[test]
    fn phrasal_verb_level_follows_its_own_frequency() {
        let level = |unit: &str| dict().lookup(unit).unwrap().cefr;
        assert_eq!(level("get up"), Cefr::A2);
        assert_eq!(level("move in"), Cefr::B1);
        assert_eq!(level("put up with"), Cefr::B2);
        assert!(level("get up") < level("move in"));
        assert!(level("move in") < level("put up with"));
    }

    #[test]
    fn mwe_level_is_marked_as_derived() {
        // У фразовых глаголов CEFR нет ни в одном открытом словаре — наш вывод должен быть
        // честно помечен, иначе через полгода никто не вспомнит, что это оценка.
        assert!(found(&analyze(RENT), "move in").cefr_derived);
        assert!(!found(&analyze(RENT), "utility").cefr_derived);
    }

    #[test]
    fn dict_covers_the_expected_scale() {
        let n = dict().lexemes().count();
        assert!(n > 20_000, "словарь подозрительно мал: {n}");
    }
}
