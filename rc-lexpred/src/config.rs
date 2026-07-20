//! Настройки модели. Коэффициенты живут здесь, а не в глубине алгоритма: их придётся
//! калибровать по реальным попыткам, и делать это правкой формул — гарантированный способ
//! однажды поменять поведение и не понять почему.

use serde::{Deserialize, Serialize};

use crate::error::{check_positive, check_probability, LexResult};

/// Что делать, если требование ссылается на концепт, которого нет в каталоге.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingConcept {
    /// Вернуть ошибку. Для батчей и импорта: дыра в каталоге — это баг данных.
    Fail,
    /// Взять дефолтный prior и пометить источник как `DefaultPrior`. Для боевого прогноза:
    /// одна незнакомая единица не повод не показать ученику оценку.
    UsePrior,
    /// Пропустить требование, сложив его в диагностику. Прогноз считается по остальным.
    Skip,
}

/// Параметры обновления знания и прогноза.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    /// Сила prior на уровне конкретного концепта — во сколько «виртуальных наблюдений»
    /// оценивается априорное мнение. `2.0` + `default_prior = 0.5` дают ровно правило
    /// Лапласа `(s+1)/(shows+2)` из `mastery::domain::score`.
    pub prior_strength: f64,
    /// Сила prior на групповом уровне и на уровне вида лексики.
    pub group_prior_strength: f64,
    /// Сила prior на уровне общей статистики ученика.
    pub overall_prior_strength: f64,
    /// Априорная вероятность, когда о концепте не известно вообще ничего.
    pub default_prior: f64,
    /// Период полураспада доказательств в днях. `None` — забывания нет.
    pub half_life_days: Option<f64>,
    /// Минимальный вес наблюдения, ниже которого оно не пишется в состояние: наблюдение
    /// с `confidence` около нуля — это шум распознавания, а не свидетельство.
    pub min_effective_weight: f64,
    /// Порог «слабого» концепта для `low_probability_concept_count` и `weakest_concepts`.
    pub low_probability_threshold: f64,
    /// Сколько индивидуальных концептов держать в состоянии.
    pub max_concepts: usize,
    /// Поведение при отсутствии концепта в каталоге.
    pub missing_concept: MissingConcept,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Совпадение с правилом Лапласа — не совпадение: см. тест
            // `matches_the_existing_laplace_score_when_unweighted`.
            prior_strength: 2.0,
            // Групповое свидетельство размывать сильнее: оно косвенное.
            group_prior_strength: 4.0,
            overall_prior_strength: 6.0,
            default_prior: 0.5,
            // ~2 месяца: попытка полугодовой давности весит вчетверо меньше вчерашней.
            half_life_days: Some(60.0),
            min_effective_weight: 0.01,
            low_probability_threshold: 0.5,
            max_concepts: 2_000,
            missing_concept: MissingConcept::UsePrior,
        }
    }
}

impl Config {
    /// Проверка на вменяемость. Вызывается всеми публичными операциями: конфиг приезжает
    /// из env сервиса, и нулевая `prior_strength` там — это деление на ноль здесь.
    pub fn validate(&self) -> LexResult<()> {
        check_positive("prior_strength", self.prior_strength)?;
        check_positive("group_prior_strength", self.group_prior_strength)?;
        check_positive("overall_prior_strength", self.overall_prior_strength)?;
        check_probability("default_prior", self.default_prior)?;
        check_probability("low_probability_threshold", self.low_probability_threshold)?;
        if let Some(h) = self.half_life_days {
            check_positive("half_life_days", h)?;
        }
        if self.max_concepts == 0 {
            return Err(crate::error::LexError::Config(
                "max_concepts = 0: состояние не сможет хранить ничего".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, MissingConcept};

    #[test]
    fn default_config_is_valid() {
        assert!(Config::default().validate().is_ok());
    }

    /// Нулевая сила prior — деление на ноль в постериоре. Отсекаем на границе, иначе
    /// получим NaN глубоко внутри и будем искать его в отчёте.
    #[test]
    fn zero_prior_strength_is_rejected() {
        let cfg = Config {
            prior_strength: 0.0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn out_of_range_default_prior_is_rejected() {
        let cfg = Config {
            default_prior: 1.5,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_capacity_state_is_rejected() {
        let cfg = Config {
            max_concepts: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn negative_half_life_is_rejected() {
        let cfg = Config {
            half_life_days: Some(-1.0),
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
        assert!(Config {
            half_life_days: None,
            ..Config::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn missing_concept_policy_round_trips_as_snake_case() {
        let j = serde_json::to_string(&MissingConcept::UsePrior).unwrap();
        assert_eq!(j, "\"use_prior\"");
    }
}
