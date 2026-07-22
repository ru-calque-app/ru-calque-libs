//! Грамматический инвентарь ru-calque: какие конструкции существуют и на каком они уровне.
//!
//! Зачем: уровень ученика сейчас целиком выдаёт LLM-eval, и проверить его нечем. Этот крейт
//! даёт вторую опору — инвентарь из 1222 конструкций с уровнями, выведенными из Cambridge
//! Learner Corpus (см. `data/SOURCES.md`). По нему можно спросить: что вообще бывает на B1,
//! что идёт следующей ступенью, чему соответствует наш `construct_key`.
//!
//! Зеркало [`rc_lex`] для грамматики — с одним важным отличием.
//!
//! # Чего крейт НЕ делает: он не смотрит в текст
//!
//! `rc_lex::analyze` умеет разобрать текст на единицы. Здесь такого нет и в v1 не будет —
//! не «руки не дошли», а по устройству данных:
//!
//! - **295 записей ([`Aspect::Use`]) описывают намерение, а не форму** («чтобы подчеркнуть»,
//!   «как сигнал завершения»). Их не найти в тексте ни сейчас, ни синтаксическим разбором
//!   потом — это суждение о смысле;
//! - реально матчибельны [`Aspect::Form`] (657) и часть [`Aspect::FormUse`] (264), и то
//!   при наличии POS-тегов и зависимостей, которых у крейта нет;
//! - слепое пятно при этом распределено по уровням почти ровно (18% на A1, 27% на B1–B2,
//!   23% на C1–C2), то есть детектор не «доломается на высоких уровнях» — он всюду
//!   систематически недосчитывает около четверти инвентаря.
//!
//! Детектор — отдельный слой поверх разбора предложения. Пока его нет, [`Feature::markers`]
//! отдаёт сырьё для него: литералы, которые EGP называет явно.
//!
//! # Умения адресуются иерархически
//!
//! Инвентарь — это 1222 умения («студент умеет ...»), и хранить наблюдения надо именно на
//! этом разрешении: свернуть потом можно, развернуть — уже нет. Поэтому ключ [`SkillId`]
//! иерархичен, а более грубый адрес — просто его префикс ([`Grain`]):
//!
//! ```text
//! egp:clauses                     Category    19 групп
//! egp:clauses.conditional         Node        91
//! egp:clauses.conditional.c2      Rung       399
//! egp:clauses.conditional.c2.01   Feature   1222
//! ```
//!
//! Отсюда две вещи. Разметчик, который видит «условное на C2», но не может показать
//! пальцем в конкретную запись, кладёт `Rung` — и это честное наблюдение, а не догадка
//! ([`Inventory::under`] потом покажет, какие записи он мог иметь в виду). А анализ и
//! отображение берут любое разрешение через [`Inventory::rollup`], не трогая данные.
//!
//! Оценивать 1222 независимых состояния по полусотне наблюдений всё равно нельзя — но это
//! ограничение модели на чтении (пулинг по ступени/узлу), а не повод обеднять запись.
//!
//! # Пример
//!
//! ```
//! use rc_gram::{inventory, Aspect, Cefr};
//!
//! let inv = inventory();
//! assert_eq!(inv.len(), 1222);
//!
//! // Что ждёт ученика следующей ступенью после B1.
//! let next: Vec<_> = inv.at_level(Cefr::B2).collect();
//! assert!(next.len() > 100);
//!
//! // Конкретная конструкция — с корпусным уровнем и живым примером.
//! let f = inv.get("egp:conjunctions.coordinating.c2.02").unwrap();
//! assert_eq!(f.level, Cefr::C2);
//! assert_eq!(f.aspect, Aspect::FormUse);
//! assert_eq!(f.markers(), vec!["AND YET"]);
//! assert!(f.examples.iter().any(|e| e.text.contains("And yet")));
//! ```

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Уровень CEFR. Реэкспорт из [`rc_lex`], а не своя копия: уровень грамматической
/// конструкции сравнивают с уровнем лексики в одном и том же расчёте, и два разных `Cefr`
/// в графе зависимостей означали бы конверсию на каждой границе (см. `rc_lex` в
/// `rc-lexpred`, там на это уже наступали).
pub use rc_lex::Cefr;

/// Словарь форм. Реэкспорт обязателен по той же причине, что и в `rc-lexpred`: `Cefr` из
/// `rc_lex` торчит в публичном API, и без реэкспорта потребитель заводит собственную
/// зависимость на `rc-lex` — причём РОВНО того же тега, иначе в графе окажутся две
/// несовместимые копии одного типа.
pub use rc_lex;

/// Инвентарь, собранный `tools/build_egp.py`, вкомпилирован в бинарь.
const EGP_JSONL: &str = include_str!("../data/egp.jsonl");

/// Что именно описывает запись — форму или намерение.
///
/// Различие практическое, а не филологическое: [`Aspect::Form`] в принципе можно искать в
/// тексте синтаксически, [`Aspect::Use`] — нельзя никогда.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aspect {
    /// Форма: «FORM: 'AND YET'», «FORM: + -IER».
    Form,
    /// Употребление: «USE: LINKING», «USE: FOCUS». Из текста не выводится.
    Use,
    /// И то и другое.
    FormUse,
    /// Разметка сломана в самом источнике (6 записей). Не выбрасываем, чтобы не терять
    /// конструкцию из-за опечатки Cambridge.
    Other,
}

impl Aspect {
    /// Можно ли в принципе надеяться найти такую конструкцию в тексте разбором.
    ///
    /// Не обещание, что найдём: у крейта нет ни POS-тегера, ни зависимостей. Это верхняя
    /// граница того, на что вообще стоит тратить силы, когда детектор появится.
    pub fn is_detectable_in_principle(self) -> bool {
        matches!(self, Self::Form | Self::FormUse)
    }
}

/// Пример из Cambridge Learner Corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    pub text: String,
    /// Уровень работы, из которой взят пример. **Не** уровень конструкции: в примерах к
    /// записи C2 встречаются работы B2 — пример показывает употребление, а не освоенность.
    #[serde(default)]
    pub level: Option<Cefr>,
    /// Родной язык автора работы. Русских — 87 из 3610.
    #[serde(default)]
    pub l1: Option<String>,
    /// Сдана ли работа. `Some(false)` — конструкция употреблена, но экзамен провален:
    /// такой пример нельзя показывать ученику как образец.
    #[serde(default)]
    pub pass: Option<bool>,
}

impl Example {
    /// Годится ли как образец для показа: сданная работа.
    pub fn is_model(&self) -> bool {
        self.pass == Some(true)
    }
}

/// Грамматическая конструкция.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Feature {
    /// Наш стабильный ключ: `egp:conjunctions.coordinating.c2.02`. По нему ссылается
    /// маппинг на `construct_key` контракта.
    pub id: String,
    /// Родной ключ EGP. Нужен только для сверки при пересборке из новой выгрузки.
    pub src_id: String,
    /// Уровень, на котором конструкция впервые надёжно появляется в корпусе.
    pub level: Cefr,
    /// `MODALITY`, `CLAUSES`, `PAST` — 19 значений.
    pub category: String,
    /// `combining`, `conditional` — 91 значение внутри категорий.
    pub subcategory: String,
    pub aspect: Aspect,
    /// Тема без префикса аспекта: `'AND YET', CONCESSIVE`.
    pub topic: String,
    /// Формулировка «Can use ...» — готовый текст для показа ученику.
    pub can_do: String,
    /// Шкала 1–3, есть лишь у 231 записи и толком не документирована в EGP. Решений на ней
    /// не строить.
    #[serde(default)]
    pub lexical_range: Option<u8>,
    #[serde(default)]
    pub examples: Vec<Example>,
}

impl Feature {
    /// Литералы, названные в теме явно: `'not only … but also'` → `["NOT ONLY … BUT ALSO"]`.
    ///
    /// Это **не детектор**, а сырьё для него: EGP сам выделяет кавычками те слова, без
    /// которых конструкции не бывает. Такие есть у ~540 записей из 1222; у остальных тема
    /// описательная (`FORM: COMPOUND ADJECTIVES`) и опорных слов не содержит вовсе.
    ///
    /// Совпадение литерала в тексте НЕ означает, что конструкция употреблена: `'have to'`
    /// найдётся и там, где речь о другом. Нужен разбор предложения.
    pub fn markers(&self) -> Vec<&str> {
        self.topic
            .split('\'')
            .skip(1)
            .step_by(2)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Примеры, годные для показа ученику: только из сданных работ.
    pub fn model_examples(&self) -> impl Iterator<Item = &Example> {
        self.examples.iter().filter(|e| e.is_model())
    }
}

/// Уровень детализации в иерархии ключей.
///
/// Ключ записи иерархичен по построению: `egp:clauses.conditional.c2.01`. Каждый более
/// грубый адрес — **префикс** точного, в том же неймспейсе:
///
/// ```text
/// egp:clauses                     Category  19
/// egp:clauses.conditional         Node      91
/// egp:clauses.conditional.c2      Rung     399
/// egp:clauses.conditional.c2.01   Feature 1222
/// ```
///
/// Поэтому второй словарь и таблица соответствий не нужны: свёртка — операция над
/// префиксом. И разметчик, который видит «условное на C2», но не может показать пальцем в
/// конкретную запись, честно кладёт `Rung` вместо того, чтобы гадать.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grain {
    Category,
    Node,
    Rung,
    Feature,
}

impl Grain {
    /// Сколько сегментов после `egp:` соответствует этой детализации.
    fn segments(self) -> usize {
        match self {
            Self::Category => 1,
            Self::Node => 2,
            Self::Rung => 3,
            Self::Feature => 4,
        }
    }
}

/// Ключ умения — адрес в инвентаре любой детализации.
///
/// Именно это значение хранит состояние ученика («умеет / обошёл / сломался»), поэтому оно
/// вынесено в отдельный тип, а не остаётся строкой: `SkillId` умеет то, ради чего иерархия
/// и заводилась — обрезаться до нужной детализации и проверять вложенность.
///
/// Инвентарь — про то, что бывает в языке; `SkillId` — про то, что мы записали про ученика.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SkillId(pub String);

const NS: &str = "egp:";

impl SkillId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Сегменты после `egp:`. Пусто, если ключ не из нашего неймспейса.
    fn parts(&self) -> Vec<&str> {
        self.0
            .strip_prefix(NS)
            .map(|s| s.split('.').filter(|p| !p.is_empty()).collect())
            .unwrap_or_default()
    }

    /// Детализация ключа. `None` — ключ чужой или пустой.
    pub fn grain(&self) -> Option<Grain> {
        Some(match self.parts().len() {
            1 => Grain::Category,
            2 => Grain::Node,
            3 => Grain::Rung,
            4 => Grain::Feature,
            _ => return None,
        })
    }

    /// Обрезать до более грубой детализации. `None`, если ключ уже грубее запрошенной.
    pub fn truncate(&self, grain: Grain) -> Option<Self> {
        let parts = self.parts();
        let n = grain.segments();
        (parts.len() >= n).then(|| Self(format!("{NS}{}", parts[..n].join("."))))
    }

    /// Адресует ли этот ключ `other` — сам себя или что-то под собой.
    ///
    /// Сравнение по границе сегмента, а не по символам: `egp:past` не должен «покрывать»
    /// `egp:passives`.
    pub fn covers(&self, other: &Self) -> bool {
        other.0 == self.0 || other.0.starts_with(&format!("{}.", self.0))
    }
}

impl std::fmt::Display for SkillId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SkillId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Свёрнутая группа: адрес плюс всё, что под него попало.
#[derive(Debug, Clone)]
pub struct Group<'a> {
    pub id: SkillId,
    pub features: Vec<&'a Feature>,
}

impl Group<'_> {
    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Размах уровней внутри группы — лестница трудности. У большинства узлов это 4–5
    /// ступеней CEFR, и именно поэтому «умеет условные предложения» без уровня
    /// бессмысленно.
    pub fn level_span(&self) -> Option<(Cefr, Cefr)> {
        let mut it = self.features.iter().map(|f| f.level);
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), l| (lo.min(l), hi.max(l))))
    }
}

/// Инвентарь целиком (1222 конструкции), парсится один раз.
pub struct Inventory {
    features: Vec<Feature>,
    by_id: HashMap<String, usize>,
}

/// Глобальный инвентарь.
pub fn inventory() -> &'static Inventory {
    static INV: OnceLock<Inventory> = OnceLock::new();
    INV.get_or_init(Inventory::load)
}

impl Inventory {
    fn load() -> Self {
        let features: Vec<Feature> = EGP_JSONL
            .lines()
            .filter(|l| !l.trim().is_empty())
            // Битая строка — дефект сборки данных, а не рантайма: файл вкомпилирован и
            // проверен тестом `inventory_covers_the_expected_scale`. Молча пропускаем,
            // чтобы одна опечатка в выгрузке не роняла сервис на старте.
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let by_id = features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.id.clone(), i))
            .collect();
        Self { features, by_id }
    }

    /// Конструкция по нашему ключу.
    pub fn get(&self, id: &str) -> Option<&Feature> {
        self.by_id.get(id).map(|&i| &self.features[i])
    }

    /// Все конструкции в порядке файла.
    pub fn all(&self) -> impl Iterator<Item = &Feature> {
        self.features.iter()
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Конструкции ровно этого уровня — «что осваивают на B2».
    pub fn at_level(&self, level: Cefr) -> impl Iterator<Item = &Feature> {
        self.features.iter().filter(move |f| f.level == level)
    }

    /// Всё до уровня включительно — «что ученик к этой ступени уже должен уметь».
    pub fn up_to(&self, level: Cefr) -> impl Iterator<Item = &Feature> {
        self.features.iter().filter(move |f| f.level <= level)
    }

    /// Конструкции категории (`MODALITY`), регистронезависимо.
    pub fn in_category<'a>(&'a self, category: &'a str) -> impl Iterator<Item = &'a Feature> {
        self.features
            .iter()
            .filter(move |f| f.category.eq_ignore_ascii_case(category))
    }

    /// Всё, что адресует ключ любой детализации: `egp:clauses.conditional` → 27 записей,
    /// `egp:clauses.conditional.c2.01` → одна.
    ///
    /// Это способ прочитать грубое наблюдение: разметчик положил ступень — здесь видно,
    /// какие конкретно записи он мог иметь в виду.
    pub fn under<'a>(&'a self, prefix: &'a SkillId) -> impl Iterator<Item = &'a Feature> {
        self.features
            .iter()
            .filter(move |f| prefix.covers(&SkillId(f.id.clone())))
    }

    /// Свернуть инвентарь до нужной детализации — то, ради чего ключи иерархические.
    ///
    /// `Category` → 19 групп, `Node` → 91, `Rung` → 399, `Feature` → 1222. Хранение при
    /// этом всегда полное: свёртка живёт на чтении и её можно менять, не трогая данные.
    ///
    /// Группы отсортированы по ключу — порядок устойчив между запусками.
    pub fn rollup(&self, grain: Grain) -> Vec<Group<'_>> {
        let mut by_key: HashMap<SkillId, Vec<&Feature>> = HashMap::new();
        let keyed = self
            .features
            .iter()
            .filter_map(|f| Some((SkillId(f.id.clone()).truncate(grain)?, f)));
        for (key, f) in keyed {
            by_key.entry(key).or_default().push(f);
        }
        let mut out: Vec<Group<'_>> = by_key
            .into_iter()
            .map(|(id, features)| Group { id, features })
            .collect();
        out.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Лестница узла: ступени снизу вверх. `egp:clauses.conditional` → A2, B1, B2, C1, C2.
    ///
    /// Основной способ спросить «а что дальше»: ступень выше текущей — это и есть край
    /// роста ученика внутри этой темы.
    pub fn ladder(&self, node: &SkillId) -> Vec<Group<'_>> {
        let mut rungs: Vec<Group<'_>> = self
            .rollup(Grain::Rung)
            .into_iter()
            .filter(|g| node.covers(&g.id))
            .collect();
        rungs.sort_unstable_by_key(|g| g.features.first().map(|f| f.level));
        rungs
    }

    /// Названия категорий, по алфавиту.
    pub fn categories(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.features.iter().map(|f| f.category.as_str()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv() -> &'static Inventory {
        inventory()
    }

    #[test]
    fn inventory_covers_the_expected_scale() {
        // Парсинг молча пропускает битые строки — этот тест единственное, что отличает
        // «файл распарсился» от «файл распарсился наполовину».
        assert_eq!(inv().len(), 1222);
        assert_eq!(inv().categories().len(), 19);
        assert_eq!(inv().all().filter(|f| !f.examples.is_empty()).count(), 1218);
    }

    #[test]
    fn levels_are_comparable_with_lexical_levels() {
        // Ради этого крейт зависит от rc-lex: уровень конструкции и уровень слова должны
        // складываться в один расчёт без конверсий.
        let g = inv()
            .get("egp:conjunctions.coordinating.c2.02")
            .unwrap()
            .level;
        let l = rc_lex::dict().lookup_pos("utility", "noun").unwrap().cefr;
        assert!(l < g, "B2-лексика должна быть ниже C2-конструкции");
    }

    #[test]
    fn a_feature_carries_level_category_and_corpus_examples() {
        let f = inv().get("egp:conjunctions.coordinating.c2.02").unwrap();
        assert_eq!(f.level, Cefr::C2);
        assert_eq!(f.category, "CONJUNCTIONS");
        assert_eq!(f.subcategory, "coordinating");
        assert_eq!(f.aspect, Aspect::FormUse);
        assert!(f.can_do.starts_with("Can use 'And yet'"));
        assert!(f.examples.iter().any(|e| e.text.contains("And yet")));
    }

    #[test]
    fn markers_extract_the_literals_egp_names_itself() {
        let f = inv().get("egp:conjunctions.coordinating.c1.04").unwrap();
        assert_eq!(f.markers(), vec!["NOT ONLY … BUT ALSO"]);
        // Описательная тема опорных слов не содержит — и это не ошибка, а половина
        // инвентаря. Детектор, построенный только на markers, увидит меньшую его часть.
        let descriptive = inv()
            .all()
            .find(|f| f.topic == "COMPOUND ADJECTIVES")
            .unwrap();
        assert!(descriptive.markers().is_empty());
        let with_markers = inv().all().filter(|f| !f.markers().is_empty()).count();
        assert!(
            (500..600).contains(&with_markers),
            "литералы есть у {with_markers} записей"
        );
    }

    /// Главное, ради чего инвентарь заведён: «что дальше» становится вопросом к данным,
    /// а не к LLM.
    #[test]
    fn next_step_after_a_level_is_a_finite_answer() {
        let b1: Vec<_> = inv().at_level(Cefr::B1).collect();
        let b2: Vec<_> = inv().at_level(Cefr::B2).collect();
        assert_eq!(b1.len(), 338);
        assert_eq!(b2.len(), 243);
        assert!(b1.iter().all(|f| f.level == Cefr::B1));
        // Накопительный срез включает всё, что ниже.
        assert_eq!(inv().up_to(Cefr::A2).count(), 109 + 291);
        assert_eq!(inv().up_to(Cefr::C2).count(), inv().len());
    }

    #[test]
    fn modality_is_the_biggest_category_and_is_queryable() {
        let m: Vec<_> = inv().in_category("MODALITY").collect();
        assert_eq!(m.len(), 239);
        assert_eq!(
            inv().in_category("modality").count(),
            239,
            "регистр не важен"
        );
        assert!(inv().in_category("NOSUCHTHING").next().is_none());
    }

    /// Аспект — не украшение: он задаёт потолок любого будущего детектора.
    #[test]
    fn use_features_are_marked_as_never_detectable() {
        let total = inv().len();
        let detectable = inv()
            .all()
            .filter(|f| f.aspect.is_detectable_in_principle())
            .count();
        assert_eq!(inv().all().filter(|f| f.aspect == Aspect::Use).count(), 295);
        assert_eq!(detectable, 657 + 264);
        assert!(detectable * 100 / total < 80, "потолок детектора — не 100%");
    }

    /// Примеры из проваленных работ содержат конструкцию, но образцом быть не могут.
    #[test]
    fn failed_scripts_are_excluded_from_model_examples() {
        let failed: Vec<_> = inv()
            .all()
            .filter(|f| f.examples.iter().any(|e| e.pass == Some(false)))
            .collect();
        assert!(!failed.is_empty());
        let f = failed[0];
        assert!(f.model_examples().all(|e| e.pass == Some(true)));
        assert!(f.model_examples().count() < f.examples.len());
    }

    /// Хранение полное, свёртка — на чтении. Числа фиксируют все четыре разрешения сразу.
    #[test]
    fn rollup_gives_every_resolution_from_the_same_data() {
        assert_eq!(inv().rollup(Grain::Category).len(), 19);
        assert_eq!(inv().rollup(Grain::Node).len(), 91);
        assert_eq!(inv().rollup(Grain::Rung).len(), 399);
        assert_eq!(inv().rollup(Grain::Feature).len(), 1222);
        // Свёртка ничего не теряет и не двоит на любом разрешении.
        for g in [Grain::Category, Grain::Node, Grain::Rung] {
            let total: usize = inv().rollup(g).iter().map(Group::len).sum();
            assert_eq!(total, inv().len(), "разрешение {g:?} потеряло записи");
        }
    }

    /// Ключ грубого наблюдения — префикс точного, и это единственное, что нужно, чтобы
    /// связать «условное на C2» с конкретными записями.
    #[test]
    fn a_coarse_key_addresses_the_precise_ones() {
        let node = SkillId::from("egp:clauses.conditional");
        let rung = SkillId::from("egp:clauses.conditional.c2");
        let feat = SkillId::from("egp:clauses.conditional.c2.01");

        assert_eq!(node.grain(), Some(Grain::Node));
        assert_eq!(feat.grain(), Some(Grain::Feature));
        assert_eq!(feat.truncate(Grain::Node), Some(node.clone()));
        assert_eq!(feat.truncate(Grain::Rung), Some(rung.clone()));
        // Обрезать до более мелкого, чем есть, нечего.
        assert_eq!(node.truncate(Grain::Feature), None);

        assert!(node.covers(&feat));
        assert!(rung.covers(&feat));
        assert!(!rung.covers(&SkillId::from("egp:clauses.conditional.b1.01")));
        assert_eq!(inv().under(&node).count(), 27);
        assert_eq!(inv().under(&rung).count(), 8);
        assert_eq!(inv().under(&feat).count(), 1);
    }

    /// Сравнение по границе сегмента, а не по символам: иначе `egp:past` проглотил бы
    /// `egp:passives` и статистика по временам молча вобрала бы пассивы.
    #[test]
    fn prefix_matching_respects_segment_boundaries() {
        let past = SkillId::from("egp:past");
        assert!(!past.covers(&SkillId::from("egp:passives.passives_form.b1.01")));
        assert!(past.covers(&SkillId::from("egp:past.past_simple.a2.01")));
        assert_eq!(inv().under(&past).count(), 93);
        assert_eq!(inv().under(&SkillId::from("egp:passives")).count(), 40);
    }

    /// Лестница — основной способ спросить «что дальше»: ступени идут снизу вверх.
    #[test]
    fn a_node_is_a_ladder_of_rungs_bottom_up() {
        let rungs = inv().ladder(&SkillId::from("egp:clauses.conditional"));
        let levels: Vec<Cefr> = rungs.iter().map(|g| g.level_span().unwrap().0).collect();
        assert_eq!(
            levels,
            vec![Cefr::A2, Cefr::B1, Cefr::B2, Cefr::C1, Cefr::C2]
        );
        assert!(levels.windows(2).all(|w| w[0] < w[1]));
        // Внутри ступени уровень один — на то она и ступень.
        for g in &rungs {
            let (lo, hi) = g.level_span().unwrap();
            assert_eq!(lo, hi);
        }
    }

    /// Ради чего вся иерархия: «умеет условные предложения» без уровня — не факт, а каша
    /// из пяти ступеней CEFR.
    #[test]
    fn a_node_without_a_level_spans_the_whole_scale() {
        let nodes = inv().rollup(Grain::Node);
        let wide = nodes
            .iter()
            .filter(|g| {
                let (lo, hi) = g.level_span().unwrap();
                (hi as u8) - (lo as u8) >= 3
            })
            .count();
        assert_eq!(wide, 78, "узлов с размахом >=3 ступеней");
    }

    /// Родной ключ EGP сохранён: без него пересборку из новой выгрузки не сверить.
    #[test]
    fn source_ids_are_preserved_and_unique() {
        let n = inv().all().map(|f| f.src_id.as_str()).collect::<Vec<_>>();
        let uniq: std::collections::HashSet<_> = n.iter().collect();
        assert_eq!(uniq.len(), inv().len());
        assert!(n.iter().all(|s| !s.is_empty()));
    }
}
