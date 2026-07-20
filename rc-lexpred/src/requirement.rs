//! Лексические требования текста и наблюдения из попыток — нормализованный вход модели.
//!
//! Оба типа намеренно «глупые»: это данные, а не поведение. Кто именно их произвёл — LLM,
//! разметка колоды, ручная курация — либе неизвестно и неважно.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::concept::{ConceptId, ConceptKind, Realization};
use crate::error::{check_probability, check_weight, LexResult};

/// Что текст требует от ученика лексически.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalRequirement {
    pub concept_id: ConceptId,
    /// Насколько концепт важен для передачи смысла ЭТОГО текста. Не сложность концепта:
    /// частотное слово может нести всё предложение, редкое — быть проходным.
    pub importance_weight: f64,
    pub concept_kind: ConceptKind,
    /// Реализации, допустимые именно здесь. Пусто — значит берём из каталога: список в
    /// требовании нужен, когда текст сужает выбор (в этом контексте годится не любой синоним).
    #[serde(default)]
    pub acceptable_realizations: Vec<Realization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl LexicalRequirement {
    /// Требование с проверкой веса. Единственный способ его получить в обход десериализации.
    pub fn new(
        concept_id: impl Into<ConceptId>,
        concept_kind: ConceptKind,
        importance_weight: f64,
    ) -> LexResult<Self> {
        check_weight("importance_weight", importance_weight)?;
        Ok(Self {
            concept_id: concept_id.into(),
            importance_weight,
            concept_kind,
            acceptable_realizations: Vec::new(),
            note: None,
        })
    }

    pub fn with_realizations(mut self, items: Vec<Realization>) -> Self {
        self.acceptable_realizations = items;
        self
    }

    /// Проверка требования, пришедшего десериализацией (там конструктор не вызывается).
    pub fn validate(&self) -> LexResult<()> {
        check_weight("importance_weight", self.importance_weight)?;
        Ok(())
    }
}

/// Почему концепт не был выражен как надо. Совпадает по смыслу со шкалой вреда из разбора
/// (`CommunicativeImpact` в контракте) плюс два лексических исхода, которых там нет.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalErrorKind {
    /// Ученик не сказал единицу, но донёс смысл перифразом. Не ошибка: верх лестницы.
    Workaround,
    /// Не сказал и смысл не донёс.
    Skipped,
    /// Сказал не то: выбрал неверную реализацию.
    WrongChoice,
    /// Сказал верную единицу в неверной форме/сочетаемости.
    WrongForm,
}

/// Свидетельство об одном концепте из одной попытки.
///
/// `outcome` — насколько смысл выражен, `confidence` — насколько мы этому свидетельству
/// верим. Разделение принципиальное: сорванное распознавание речи должно ронять
/// `confidence`, а не превращаться в `outcome = 0`. Иначе ученик получает провал за
/// качество микрофона, и модель уверенно учится ерунде.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalObservation {
    pub concept_id: ConceptId,
    /// Насколько смысл выражен, `0.0..=1.0`:
    /// `1.0` — верно; `0.75..0.9` — приемлемый парафраз; `0.4..0.7` — смысл примерно
    /// передан, выбор неточен; `0.0` — пропущено или неверно.
    pub outcome: f64,
    /// Насколько наблюдению можно верить, `0.0..=1.0`.
    pub confidence: f64,
    pub observed_at: DateTime<Utc>,
    /// Какой именно реализацией ученик воспользовался, если это известно.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_realization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<LexicalErrorKind>,
}

impl LexicalObservation {
    /// Наблюдение с проверкой обоих чисел.
    pub fn new(
        concept_id: impl Into<ConceptId>,
        outcome: f64,
        confidence: f64,
        observed_at: DateTime<Utc>,
    ) -> LexResult<Self> {
        check_probability("outcome", outcome)?;
        check_probability("confidence", confidence)?;
        Ok(Self {
            concept_id: concept_id.into(),
            outcome,
            confidence,
            observed_at,
            selected_realization: None,
            error_kind: None,
        })
    }

    pub fn with_realization(mut self, realization: &str) -> Self {
        self.selected_realization = Some(realization.to_string());
        self
    }

    pub fn with_error(mut self, kind: LexicalErrorKind) -> Self {
        self.error_kind = Some(kind);
        self
    }

    /// Проверка наблюдения, пришедшего десериализацией.
    pub fn validate(&self) -> LexResult<()> {
        check_probability("outcome", self.outcome)?;
        check_probability("confidence", self.confidence)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::{ConceptKind, LexicalObservation, LexicalRequirement};

    fn now() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    /// Числа, пришедшие снаружи, обязаны проверяться на границе. NaN, пролезший в
    /// `outcome`, отравит все агрегаты, куда попадёт, и найти его потом негде.
    #[test]
    fn observations_with_impossible_numbers_are_rejected() {
        assert!(LexicalObservation::new("A", 1.5, 1.0, now()).is_err());
        assert!(LexicalObservation::new("A", f64::NAN, 1.0, now()).is_err());
        assert!(LexicalObservation::new("A", 0.5, -0.1, now()).is_err());
        assert!(LexicalObservation::new("A", 0.5, 1.0, now()).is_ok());
    }

    #[test]
    fn requirements_reject_negative_importance() {
        assert!(LexicalRequirement::new("A", ConceptKind::Lemma, -1.0).is_err());
        assert!(LexicalRequirement::new("A", ConceptKind::Lemma, 2.0).is_ok());
    }

    /// Валидация должна работать и для объектов из JSON: конструктор там не вызывается,
    /// а данные приходят из чужого сервиса.
    #[test]
    fn validate_catches_bad_numbers_that_bypassed_the_constructor() {
        let bad: LexicalObservation = serde_json::from_value(serde_json::json!({
            "concept_id": "A",
            "outcome": 3.0,
            "confidence": 1.0,
            "observed_at": "2026-07-20T12:00:00Z"
        }))
        .unwrap();
        assert!(bad.validate().is_err());
    }
}
