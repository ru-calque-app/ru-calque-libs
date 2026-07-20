//! Компактное лексическое состояние ученика.
//!
//! Что здесь есть: свёрнутые свидетельства на трёх уровнях — общий, групповой, поконцептный.
//! Чего здесь нет и не должно быть: сырых попыток, исходных текстов, описаний концептов.
//! Состояние ссылается на концепты только по id, всё остальное приезжает из каталога.
//! Поэтому правка каталога (новая реализация, другой prior) не требует трогать состояния
//! учеников — а их будут десятки тысяч.
//!
//! Забывание считается **лениво**: у каждой записи есть отметка времени, и коэффициент
//! распада применяется в момент чтения или обновления именно этой записи. Регулярно
//! переписывать всё состояние ради decay нельзя — это превращает «посмотреть прогноз» в
//! запись в БД и ломает воспроизводимость.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::concept::{ConceptId, GroupId};
use crate::error::{LexError, LexResult};

/// Версия схемы состояния. Растёт при несовместимом изменении раскладки полей.
pub const SCHEMA_VERSION: u16 = 1;

/// Свёрнутые свидетельства: сколько «успеха» и «неуспеха» накоплено.
///
/// Числа дробные, потому что каждое наблюдение входит со своим весом (доверие к
/// распознаванию, свежесть). Счётчик наблюдений целый и не распадается — он нужен для
/// диагностики и для решения о компактизации, а не для вероятности.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub weighted_successes: f64,
    pub weighted_failures: f64,
    pub observation_count: u32,
    pub updated_at: DateTime<Utc>,
}

impl Evidence {
    /// Пустое свидетельство с отметкой времени.
    pub fn empty(at: DateTime<Utc>) -> Self {
        Self {
            weighted_successes: 0.0,
            weighted_failures: 0.0,
            observation_count: 0,
            updated_at: at,
        }
    }

    /// Сколько всего весa накоплено.
    pub fn total(&self) -> f64 {
        self.weighted_successes + self.weighted_failures
    }

    /// Свидетельства, приведённые к моменту `now`.
    ///
    /// `half_life` в днях; `None` — забывания нет. Отметка времени в прошлом относительно
    /// записи (часы разъехались, попытка приехала задним числом) не должна усиливать
    /// свидетельство — коэффициент зажат сверху единицей.
    pub fn decayed(&self, now: DateTime<Utc>, half_life_days: Option<f64>) -> (f64, f64) {
        let Some(half_life) = half_life_days else {
            return (self.weighted_successes, self.weighted_failures);
        };
        let days = (now - self.updated_at).num_seconds() as f64 / 86_400.0;
        if days <= 0.0 {
            return (self.weighted_successes, self.weighted_failures);
        }
        let factor = 0.5_f64.powf(days / half_life);
        (
            self.weighted_successes * factor,
            self.weighted_failures * factor,
        )
    }

    /// Добавить взвешенное наблюдение, предварительно приведя накопленное к `at`.
    ///
    /// Счётчики не могут уйти в минус: `outcome` и `weight` уже проверены на границе, но
    /// защита от накопленной ошибки округления дешевле, чем поиск отрицательного веса в
    /// проде.
    pub fn add(&mut self, outcome: f64, weight: f64, at: DateTime<Utc>, half_life: Option<f64>) {
        let (s, f) = self.decayed(at, half_life);
        self.weighted_successes = (s + weight * outcome).max(0.0);
        self.weighted_failures = (f + weight * (1.0 - outcome)).max(0.0);
        self.observation_count = self.observation_count.saturating_add(1);
        self.updated_at = at.max(self.updated_at);
    }
}

/// Что известно про конкретный концепт. Ровно доказательства и отметки времени — ничего
/// из каталога.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptKnowledge {
    #[serde(flatten)]
    pub evidence: Evidence,
    /// Когда концепт встретился впервые — нужен для диагностики «давно ли учим».
    pub first_seen_at: DateTime<Utc>,
}

impl ConceptKnowledge {
    pub fn new(at: DateTime<Utc>) -> Self {
        Self {
            evidence: Evidence::empty(at),
            first_seen_at: at,
        }
    }

    /// Когда концепт встречался в последний раз.
    pub fn last_seen_at(&self) -> DateTime<Utc> {
        self.evidence.updated_at
    }
}

/// Лексическое состояние ученика целиком.
///
/// `BTreeMap`, а не `HashMap`: порядок обхода и байты сериализации должны быть
/// воспроизводимы. С `HashMap` два одинаковых состояния дают разный JSON, и любой
/// golden-тест или сверка хэшей превращаются в лотерею.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalState {
    pub schema_version: u16,
    /// Общая лексическая статистика ученика — последний уровень fallback.
    pub overall: Evidence,
    /// Агрегаты по группам: вид лексики, уровень, семейство, что угодно ещё.
    pub groups: BTreeMap<GroupId, Evidence>,
    /// Статистика по конкретным концептам. Размер ограничен компактизацией.
    pub concepts: BTreeMap<ConceptId, ConceptKnowledge>,
}

impl LexicalState {
    /// Пустое состояние нового ученика.
    pub fn empty(at: DateTime<Utc>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            overall: Evidence::empty(at),
            groups: BTreeMap::new(),
            concepts: BTreeMap::new(),
        }
    }

    /// Разбор сохранённого состояния.
    ///
    /// Чужую версию схемы молча принимать нельзя: поля разъедутся тихо, счётчики поедут, и
    /// заметим это через месяц по кривым прогнозам. Лучше явная ошибка на загрузке.
    pub fn from_json(raw: &str) -> LexResult<Self> {
        let state: Self = serde_json::from_str(raw)
            .map_err(|e| LexError::Config(format!("состояние не разбирается: {e}")))?;
        state.check_version()?;
        Ok(state)
    }

    /// Сериализация состояния.
    pub fn to_json(&self) -> LexResult<String> {
        serde_json::to_string(self)
            .map_err(|e| LexError::Config(format!("состояние не сериализуется: {e}")))
    }

    /// Совместима ли версия схемы с этим кодом.
    pub fn check_version(&self) -> LexResult<()> {
        if self.schema_version == SCHEMA_VERSION {
            return Ok(());
        }
        Err(LexError::IncompatibleSchema {
            found: self.schema_version,
            supported: SCHEMA_VERSION,
        })
    }

    /// Сколько индивидуальных концептов сейчас в состоянии.
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone, Utc};

    use super::{Evidence, LexicalState, SCHEMA_VERSION};
    use crate::error::LexError;

    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    #[test]
    fn empty_state_carries_the_current_schema_version() {
        assert_eq!(LexicalState::empty(at()).schema_version, SCHEMA_VERSION);
    }

    /// Состояние чужой версии обязано падать на загрузке, а не молча разбираться в
    /// сегодняшнюю структуру: тихо разъехавшиеся счётчики не найдёт никто.
    #[test]
    fn unknown_schema_version_is_rejected_on_load() {
        let mut s = LexicalState::empty(at());
        s.schema_version = 99;
        let raw = serde_json::to_string(&s).unwrap();
        assert_eq!(
            LexicalState::from_json(&raw),
            Err(LexError::IncompatibleSchema {
                found: 99,
                supported: SCHEMA_VERSION
            })
        );
    }

    #[test]
    fn state_round_trips_through_json_unchanged() {
        let s = LexicalState::empty(at());
        let back = LexicalState::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(s, back);
    }

    /// Забывание: свидетельство ровно одного периода полураспада весит вдвое меньше.
    #[test]
    fn evidence_halves_after_one_half_life() {
        let mut e = Evidence::empty(at());
        e.add(1.0, 1.0, at(), Some(30.0));
        let (s, f) = e.decayed(at() + Duration::days(30), Some(30.0));
        assert!((s - 0.5).abs() < 1e-9, "успехи не распались: {s}");
        assert!(f.abs() < 1e-9);
    }

    /// Без настроенного полураспада состояние не забывается вовсе.
    #[test]
    fn evidence_is_kept_intact_when_decay_is_disabled() {
        let mut e = Evidence::empty(at());
        e.add(1.0, 1.0, at(), None);
        let (s, _) = e.decayed(at() + Duration::days(3650), None);
        assert!((s - 1.0).abs() < 1e-9);
    }

    /// Часы могут разъехаться, а попытка — приехать задним числом. Отрицательный интервал
    /// не должен усиливать свидетельство: иначе «старое» наблюдение станет весомее нового.
    #[test]
    fn a_timestamp_from_the_past_never_amplifies_evidence() {
        let mut e = Evidence::empty(at());
        e.add(1.0, 1.0, at(), Some(30.0));
        let (s, _) = e.decayed(at() - Duration::days(100), Some(30.0));
        assert!((s - 1.0).abs() < 1e-9, "свидетельство выросло: {s}");
    }

    #[test]
    fn counters_never_go_negative() {
        let mut e = Evidence::empty(at());
        e.add(0.0, 1.0, at(), None);
        assert!(e.weighted_successes >= 0.0);
        assert!(e.weighted_failures >= 0.0);
    }
}
