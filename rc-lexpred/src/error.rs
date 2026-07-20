//! Типизированные ошибки. Из публичного API не летят паники: всё, что может не сойтись
//! (чужие числа, чужой JSON, чужой каталог), возвращает `Result`.

use thiserror::Error;

use crate::concept::ConceptId;

/// Что пошло не так при работе с лексическим состоянием и прогнозом.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum LexError {
    /// Концепта нет в каталоге, а конфигурация велит на этом останавливаться
    /// (`MissingConcept::Fail`).
    #[error("концепта {0} нет в каталоге")]
    UnknownConcept(ConceptId),

    /// Вероятность вне [0, 1] либо NaN/inf. Поле — чтобы было видно, чья именно.
    #[error("некорректная вероятность в поле {field}: {value}")]
    InvalidProbability { field: &'static str, value: String },

    /// Вес важности/наблюдения вне допустимого диапазона либо NaN/inf.
    #[error("некорректный вес в поле {field}: {value}")]
    InvalidWeight { field: &'static str, value: String },

    /// Состояние сохранено схемой, которую этот код не понимает. Молча принимать нельзя:
    /// счётчики разъедутся тихо, а заметим через месяц.
    #[error("несовместимая версия состояния: {found}, поддерживается {supported}")]
    IncompatibleSchema { found: u16, supported: u16 },

    /// Внешний extractor не смог разметить текст.
    #[error("extractor не отработал: {0}")]
    Extractor(String),

    /// Внешний evaluator не смог разобрать попытку.
    #[error("evaluator не отработал: {0}")]
    Evaluator(String),

    /// Прогнозировать нечего: у текста нет ни одного лексического требования.
    #[error("пустой набор требований: прогнозировать нечего")]
    EmptyRequirements,

    /// Конфигурация не проходит собственную валидацию.
    #[error("некорректная конфигурация: {0}")]
    Config(String),
}

/// Результат операций крейта.
pub type LexResult<T> = Result<T, LexError>;

/// Проверка «это вероятность»: конечное число в [0, 1].
pub(crate) fn check_probability(field: &'static str, value: f64) -> LexResult<f64> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        return Ok(value);
    }
    Err(LexError::InvalidProbability {
        field,
        value: value.to_string(),
    })
}

/// Проверка «это вес»: конечное неотрицательное число. Верхней границы у веса нет —
/// важность концепта в тексте может быть любой положительной.
pub(crate) fn check_weight(field: &'static str, value: f64) -> LexResult<f64> {
    if value.is_finite() && value >= 0.0 {
        return Ok(value);
    }
    Err(LexError::InvalidWeight {
        field,
        value: value.to_string(),
    })
}

/// Проверка строго положительного параметра конфигурации.
pub(crate) fn check_positive(field: &str, value: f64) -> LexResult<f64> {
    if value.is_finite() && value > 0.0 {
        return Ok(value);
    }
    Err(LexError::Config(format!(
        "{field} должен быть конечным положительным числом, получено {value}"
    )))
}

#[cfg(test)]
mod tests {
    use super::{check_probability, check_weight, LexError};

    /// NaN — главный враг такой модели: он проходит любое сравнение и заражает все
    /// агрегаты, в которые попал. Ловим на входе, а не в отчёте через месяц.
    #[test]
    fn nan_and_infinity_are_rejected_as_probabilities() {
        assert!(check_probability("outcome", f64::NAN).is_err());
        assert!(check_probability("outcome", f64::INFINITY).is_err());
        assert!(check_probability("outcome", -0.1).is_err());
        assert!(check_probability("outcome", 1.1).is_err());
        assert_eq!(check_probability("outcome", 0.0), Ok(0.0));
        assert_eq!(check_probability("outcome", 1.0), Ok(1.0));
    }

    #[test]
    fn negative_weight_is_rejected_but_large_one_is_fine() {
        assert!(check_weight("importance", -1.0).is_err());
        assert!(check_weight("importance", f64::NAN).is_err());
        assert_eq!(check_weight("importance", 3.0), Ok(3.0));
    }

    #[test]
    fn error_messages_name_the_offending_field() {
        let e = check_probability("confidence", 2.0).unwrap_err();
        assert!(matches!(
            e,
            LexError::InvalidProbability {
                field: "confidence",
                ..
            }
        ));
    }
}
