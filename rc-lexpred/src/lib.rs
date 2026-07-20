//! Лексическое знание ученика: попытки → компактное состояние → прогноз по новому тексту.
//!
//! Отвечает на вопрос «сможет ли этот ученик передать смысл вот этого русского текста
//! по-английски — с лексической стороны». Грамматику, произношение и пунктуацию крейт не
//! оценивает: качество распознавания попадает сюда только как `confidence` наблюдения.
//!
//! # Что здесь считается концептом
//!
//! Единица учёта — **способность выразить смысл**, а не знание слова. `ОТКЛОНИТЬ_ПРЕДЛОЖЕНИЕ`
//! — один концепт с несколькими взаимозаменяемыми реализациями (`reject` / `decline` /
//! `turn down an offer`); ученику достаточно уметь хоть одну. Обратная сторона: знание
//! `make` и `decision` по отдельности не означает знания `make a decision`, поэтому
//! коллокация — самостоятельный концепт, а не производная от компонентов.
//!
//! # Что крейт делает сам, а что обязан получить снаружи
//!
//! Сам — только арифметику и бухгалтерию: веса, накопление, забывание, fallback,
//! компактизацию, сведе́ние в оценку. Всё это детерминированно и проверяемо.
//!
//! Снаружи приходят три вещи, которые детерминированно не выводятся:
//!
//! 1. **Требования текста** ([`LexicalExtractor`]) — какие смыслы нужны, чтобы перевести
//!    русский текст, и насколько каждый важен. Переход «русский смысл → английский концепт»
//!    требует двуязычной онтологии; в ru-calque это делается через канонические переводы.
//! 2. **Разбор попытки** ([`observe`]) — факт «форма прозвучала» крейт устанавливает сам
//!    словарём `rc-lex`, но «донесён ли смысл перифразом» приходит снаружи как факт
//!    разбора. Это суждение, а не факт, и выдумывать его здесь нечем.
//! 3. **Каталог концептов** ([`ConceptCatalog`]) — описания, реализации, группы, prior'ы.
//!    Крейт не знает, лежит каталог в БД, в файле или в памяти.
//!
//! Персистентность состояния — тоже снаружи: крейт даёт [`LexicalState::to_json`] и
//! [`LexicalState::from_json`], а где это хранить, решает сервис.
//!
//! # Жизненный цикл состояния
//!
//! ```text
//! загрузить состояние → применить наблюдения → сохранить состояние
//! загрузить состояние → получить требования → предсказать → показать
//! ```
//!
//! Состояние компактно и не растёт бесконечно: [`compact_state`] держит число
//! поконцептных записей в пределах [`Config::max_concepts`]. Выброшенные записи не теряют
//! статистического вклада — он остаётся в групповых агрегатах.
//!
//! Обновление и прогноз стоят по числу концептов в текущей попытке или тексте, а не по
//! объёму истории ученика: всё состояние просматривается только при явной компактизации.
//!
//! # Пример
//!
//! ```
//! use chrono::{TimeZone, Utc};
//! use rc_lexpred::{
//!     apply_observations, predict_requirements, BaselineScorer, Concept, ConceptKind,
//!     Config, LexicalObservation, LexicalRequirement, LexicalState, MapCatalog,
//! };
//!
//! let at = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
//! let catalog = MapCatalog::new([
//!     Concept::new("MAKE_DECISION", ConceptKind::Collocation, 0.5),
//! ]);
//! let cfg = Config::default();
//!
//! // Попытка разобрана снаружи: ученик выразил смысл верно.
//! let observations = vec![LexicalObservation::new("MAKE_DECISION", 1.0, 1.0, at).unwrap()];
//! let state = apply_observations(&LexicalState::empty(at), &observations, &catalog, &cfg).unwrap();
//!
//! // Новый текст требует того же смысла.
//! let reqs = vec![LexicalRequirement::new("MAKE_DECISION", ConceptKind::Collocation, 1.0).unwrap()];
//! let p = predict_requirements(&state, &reqs, &catalog, &cfg, &BaselineScorer::default()).unwrap();
//! assert!(p.expected_lexical_coverage > 0.5);
//! ```
//!
//! # Чего крейт сознательно не делает
//!
//! Не ходит в БД, не грузит словарь из файлов, не знает ни одной LLM и не хранит истории
//! попыток и исходных текстов. И не притворяется, что
//! [`LexicalPrediction::probability_lexically_acceptable`] — измеренная частота: baseline
//! ранжирует тексты по риску, но не откалиброван (см. [`scorer`]).

#![forbid(unsafe_code)]

pub mod compact;
pub mod concept;
pub mod config;
pub mod error;
pub mod observe;
pub mod predict;
pub mod report;
pub mod requirement;
pub mod scorer;
pub mod state;
pub mod update;

/// Словарь форм. Реэкспорт обязателен: `rc_lex::Cefr` и `rc_lex::Kind` торчат в публичном
/// API этого крейта, и без реэкспорта потребитель обязан завести собственную зависимость на
/// `rc-lex` — причём РОВНО того же тега, иначе в графе окажутся две несовместимые копии
/// одних и тех же типов. На эти грабли наступили оба сервиса при подключении.
pub use rc_lex;

pub use compact::compact_state;
pub use concept::{
    derived_groups, Concept, ConceptCatalog, ConceptId, ConceptKind, GroupId, MapCatalog,
    Realization,
};
pub use config::{Config, MissingConcept};
pub use error::{LexError, LexResult};
pub use observe::{
    observe, observe_detailed, Attempt, Correction, Impact, ObserveConfig, Observed, Use,
};
pub use predict::{predict_requirements, ConceptPrediction, LexicalPrediction, ProbabilitySource};
pub use requirement::{LexicalErrorKind, LexicalObservation, LexicalRequirement};
pub use scorer::{BaselineScorer, SentenceScorer, SummaryFeatures};
pub use state::{ConceptKnowledge, Evidence, LexicalState, SCHEMA_VERSION};
pub use update::apply_observations;

/// Источник лексических требований русского текста.
///
/// Реализуется снаружи: детерминированно перейти от русского текста к английским концептам
/// нечем — нужна двуязычная онтология либо канонические переводы. В ru-calque боевая
/// реализация идёт через разметку колоды в goals.
pub trait LexicalExtractor: Send + Sync {
    /// Требования текста. Ошибка внешнего анализатора приезжает как
    /// [`LexError::Extractor`].
    fn extract(&self, source_text: &str) -> LexResult<Vec<LexicalRequirement>>;
}

/// Оценщик попытки: русский исходник + английский ответ → наблюдения.
///
/// Отдельный трейт нужен там, где разбор делает внешняя система целиком. Если разбор уже
/// есть в виде правок, дешевле собрать наблюдения детерминированно через [`observe`].
pub trait AttemptEvaluator: Send + Sync {
    /// Наблюдения по требованиям. Ошибка приезжает как [`LexError::Evaluator`].
    fn evaluate(
        &self,
        source_text: &str,
        student_answer: &str,
        requirements: &[LexicalRequirement],
    ) -> LexResult<Vec<LexicalObservation>>;
}

/// Внешние зависимости для сквозного сценария.
pub struct Services<'a> {
    pub extractor: &'a dyn LexicalExtractor,
    pub catalog: &'a dyn ConceptCatalog,
    pub scorer: &'a dyn SentenceScorer,
}

/// Удобная обёртка: русский текст → прогноз.
///
/// Только связывает этапы. Никакой логики сверх [`predict_requirements`] здесь нет и быть
/// не должно — иначе появится второй способ посчитать то же самое.
pub fn predict_text(
    state: &LexicalState,
    source_text: &str,
    services: &Services<'_>,
    cfg: &Config,
) -> LexResult<LexicalPrediction> {
    let requirements = services.extractor.extract(source_text)?;
    predict_requirements(state, &requirements, services.catalog, cfg, services.scorer)
}
