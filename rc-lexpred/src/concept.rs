//! Концепт — способность выразить смысл, а не знание слова.
//!
//! Различие принципиальное. «Знает ли ученик `decide`» — вопрос про форму, и ответ на него
//! мало что даёт: смысл «принять решение» можно передать и через `make a decision`. Учёт
//! идёт по смыслу, реализации — взаимозаменяемые способы его выразить. Поэтому успех
//! засчитывается концепту, какой бы из допустимых реализаций ученик ни воспользовался.
//!
//! Обратная сторона того же правила: знание составных частей НЕ означает знания концепта.
//! `make` (A1) и `decision` (A2) по отдельности ничего не говорят о том, что ученик скажет
//! `make a decision`, а не `take a decision`. Поэтому коллокация — самостоятельный концепт
//! со своей статистикой, а не производная от лексем-компонентов.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Стабильный идентификатор концепта.
///
/// Newtype над строкой — как `ConstructKey` в контракте, и по той же причине: значения
/// приходят из каталога, а не из фиксированного перечисления в коде.
///
/// **Стабильность обязательна.** Состояние ученика ссылается на концепты только по id;
/// если id поплывёт между версиями каталога, накопленная статистика молча привяжется не к
/// тому смыслу. Каталог обязан быть курируемым — генерировать id на лету под каждый текст
/// нельзя.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConceptId(pub String);

impl ConceptId {
    /// Заимствованный вид — чтобы не клонировать ради поиска в мапе.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ConceptId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ConceptId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for ConceptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Идентификатор группы, по которой копится агрегат для fallback.
///
/// Группы намеренно не перечислены enum'ом: измерения будут добавляться (тема, источник
/// колоды, частотный диапазон), и каждое новое измерение не должно ломать сериализованное
/// состояние. Конструкторы ниже задают соглашение об именовании префиксом.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupId(pub String);

impl GroupId {
    /// Группа «вид лексики»: `kind:collocation`.
    pub fn kind(kind: ConceptKind) -> Self {
        Self(format!("kind:{}", kind.as_str()))
    }

    /// Группа «уровень сложности»: `cefr:B2`.
    pub fn cefr(cefr: rc_lex::Cefr) -> Self {
        Self(format!("cefr:{cefr:?}"))
    }

    /// Группа «частотный диапазон»: `zipf:4` (целая часть zipf).
    pub fn frequency_band(zipf: f32) -> Self {
        Self(format!("zipf:{}", zipf.trunc() as i32))
    }

    /// Словообразовательное семейство или иная произвольная категория каталога.
    pub fn family(name: &str) -> Self {
        Self(format!("family:{name}"))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Вид лексической единицы.
///
/// Расширяет `rc_lex::Kind` теми видами, которых в словаре форм нет: коллокация и
/// устойчивое сочетание там не размечены (нет открытого источника — см. `rc-lex/data/
/// SOURCES.md`), а `Family` вообще не единица, а узел для fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConceptKind {
    /// Отдельная лемма в конкретном значении.
    Lemma,
    /// Устойчивое словосочетание, воспроизводимое целиком (`as far as I know`).
    Phrase,
    /// Коллокация: свободное по смыслу, но несвободное по выбору слов (`make a decision`).
    Collocation,
    /// Фразовый глагол (`turn down`).
    PhrasalVerb,
    /// Идиоматическая конструкция, смысл которой не выводится из частей.
    Idiom,
    /// Словообразовательное семейство или иная обобщающая группа — используется как
    /// уровень fallback, а не как самостоятельная цель.
    Family,
}

impl ConceptKind {
    /// Строковый вид для ключей групп и отчётов.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lemma => "lemma",
            Self::Phrase => "phrase",
            Self::Collocation => "collocation",
            Self::PhrasalVerb => "phrasal_verb",
            Self::Idiom => "idiom",
            Self::Family => "family",
        }
    }

    /// Многословная ли единица. Многословные — отдельный риск: их нельзя собрать из
    /// известных слов, поэтому они считаются отдельно в фичах текста.
    pub fn is_multiword(self) -> bool {
        matches!(
            self,
            Self::Phrase | Self::Collocation | Self::PhrasalVerb | Self::Idiom
        )
    }

    /// Соответствие виду из словаря форм.
    pub fn from_lex(kind: rc_lex::Kind) -> Self {
        match kind {
            rc_lex::Kind::Word => Self::Lemma,
            rc_lex::Kind::PhrasalVerb => Self::PhrasalVerb,
            rc_lex::Kind::Idiom => Self::Idiom,
        }
    }
}

/// Допустимая реализация концепта — конкретный английский способ выразить смысл.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Realization {
    /// Каноничная форма: `make a decision`, `turn down`.
    pub text: String,
    /// Ссылка на единицу словаря форм (`rc_lex::Lexeme::id`), если она там есть.
    /// `None` — реализация есть только в нашем каталоге (коллокаций в `rc-lex` нет).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexeme_id: Option<String>,
}

impl Realization {
    /// Реализация без привязки к словарю форм.
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            lexeme_id: None,
        }
    }

    /// Реализация, связанная с единицей `rc-lex`.
    pub fn from_lexeme(text: &str, lexeme_id: &str) -> Self {
        Self {
            text: text.to_string(),
            lexeme_id: Some(lexeme_id.to_string()),
        }
    }
}

/// Запись глобального каталога: всё, что известно о концепте безотносительно ученика.
///
/// В состоянии ученика этого нет и быть не должно — там только id и накопленные
/// свидетельства. Описание, сложность и реализации живут здесь и меняются вместе с
/// каталогом, не трогая состояния учеников.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Concept {
    pub id: ConceptId,
    pub kind: ConceptKind,
    /// Априорная вероятность, что средний ученик выразит этот смысл. Не «сложность»:
    /// шкала направлена в сторону успеха, как и всё остальное в модели.
    pub base_probability: f64,
    /// Взаимозаменяемые способы выразить смысл. Пустой список допустим: концепт может
    /// учитываться и без перечисления форм, если наблюдения приходят уже по смыслу.
    #[serde(default)]
    pub realizations: Vec<Realization>,
    /// Группы, в агрегаты которых идут наблюдения по этому концепту.
    #[serde(default)]
    pub group_ids: Vec<GroupId>,
    /// Уровень по CEFR, если каталог его знает.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cefr: Option<rc_lex::Cefr>,
    /// Произвольные метаданные каталога — либа их не интерпретирует, но и не теряет.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl Concept {
    /// Минимальный концепт: id, вид и prior. Остальное — через `with_*`.
    pub fn new(id: impl Into<ConceptId>, kind: ConceptKind, base_probability: f64) -> Self {
        Self {
            id: id.into(),
            kind,
            base_probability,
            realizations: Vec::new(),
            group_ids: Vec::new(),
            cefr: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_realizations(mut self, items: &[&str]) -> Self {
        self.realizations = items.iter().map(|t| Realization::new(t)).collect();
        self
    }

    pub fn with_groups(mut self, groups: Vec<GroupId>) -> Self {
        self.group_ids = groups;
        self
    }

    pub fn with_cefr(mut self, cefr: rc_lex::Cefr) -> Self {
        self.cefr = Some(cefr);
        self
    }

    /// Группы, в которые идёт наблюдение: явные из каталога плюс выводимые из вида и
    /// уровня. Дубликаты убираются, порядок стабилен — от него зависит воспроизводимость.
    pub fn effective_groups(&self) -> Vec<GroupId> {
        let mut out = self.group_ids.clone();
        out.push(GroupId::kind(self.kind));
        if let Some(c) = self.cefr {
            out.push(GroupId::cefr(c));
        }
        out.sort();
        out.dedup();
        out
    }
}

/// Источник метаданных о концептах.
///
/// Каталог живёт вне библиотеки: в БД, в файле, в памяти сервиса — либе всё равно.
/// `Send + Sync` — чтобы состояние и прогноз считались из обработчиков axum.
pub trait ConceptCatalog: Send + Sync {
    /// Метаданные концепта. `None` — концепта в каталоге нет; что с этим делать, решает
    /// `Config::missing_concept`, а не каталог.
    fn get(&self, id: &ConceptId) -> Option<Concept>;
}

/// Каталог в памяти.
///
/// Не тестовая заглушка: сервис наполняет его строками из таблицы `lexemes` и курируемого
/// файла связок «смысл → реализации». В тестах используется он же — поэтому production-код
/// не зависит от отдельной mock-реализации.
#[derive(Debug, Default, Clone)]
pub struct MapCatalog {
    by_id: BTreeMap<ConceptId, Concept>,
}

impl MapCatalog {
    pub fn new(concepts: impl IntoIterator<Item = Concept>) -> Self {
        Self {
            by_id: concepts.into_iter().map(|c| (c.id.clone(), c)).collect(),
        }
    }

    pub fn insert(&mut self, concept: Concept) {
        self.by_id.insert(concept.id.clone(), concept);
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

impl ConceptCatalog for MapCatalog {
    fn get(&self, id: &ConceptId) -> Option<Concept> {
        self.by_id.get(id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::{Concept, ConceptCatalog, ConceptId, ConceptKind, GroupId, MapCatalog};

    #[test]
    fn concept_id_serializes_as_a_bare_string() {
        let j = serde_json::to_string(&ConceptId::from("MAKE_DECISION")).unwrap();
        assert_eq!(j, "\"MAKE_DECISION\"");
    }

    /// Виды-единицы делятся на однословные и многословные: многословные нельзя собрать
    /// из известных слов, поэтому в фичах текста они считаются отдельно.
    #[test]
    fn multiword_kinds_are_marked_as_such() {
        assert!(!ConceptKind::Lemma.is_multiword());
        assert!(ConceptKind::Collocation.is_multiword());
        assert!(ConceptKind::PhrasalVerb.is_multiword());
        assert!(ConceptKind::Idiom.is_multiword());
    }

    /// Группы должны выводиться детерминированно и без дублей: их порядок влияет на
    /// то, какие агрегаты обновятся, а значит — на воспроизводимость прогноза.
    #[test]
    fn effective_groups_are_sorted_deduped_and_include_derived_ones() {
        let c = Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.4)
            .with_groups(vec![GroupId::family("offer"), GroupId::family("offer")])
            .with_cefr(rc_lex::Cefr::B2);
        let groups = c.effective_groups();
        assert_eq!(
            groups,
            vec![
                GroupId("cefr:B2".into()),
                GroupId("family:offer".into()),
                GroupId("kind:collocation".into()),
            ]
        );
    }

    #[test]
    fn map_catalog_returns_none_for_unknown_ids() {
        let cat = MapCatalog::new([Concept::new("A", ConceptKind::Lemma, 0.5)]);
        assert!(cat.get(&ConceptId::from("A")).is_some());
        assert!(cat.get(&ConceptId::from("B")).is_none());
    }
}
