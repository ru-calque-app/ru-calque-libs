//! Сведение прогноза по концептам в одно число — и честная граница этого числа.
//!
//! Здесь живёт единственное место, где модель делает содержательно недоказуемое
//! утверждение: «вероятность, что перевод в целом окажется лексически приемлемым».
//! Произведением вероятностей её называть нельзя — концепты зависимы (не знаешь темы —
//! проваливаешь сразу несколько), а ученик может обойтись перифразом.
//!
//! **Ограничение baseline'а, которое надо знать.** `BaselineScorer` — интерпретируемая
//! логистическая свёртка нескольких признаков с коэффициентами, выставленными на глаз. Она
//! **ранжирует**: текст с оценкой 0.3 действительно рискованнее текста с 0.8. Но она
//! **не откалибрована**: из «0.67» не следует, что в 67 попытках из 100 перевод окажется
//! приемлемым. Чтобы это стало правдой, нужны размеченные исходы (прогноз → реальная
//! попытка → сверка), а не другая формула.
//!
//! Поэтому скорер — трейт. Когда данные накопятся, на их месте появится обученная
//! калибровка (хоть логистическая регрессия на этих же признаках, хоть изотоническая
//! поверх baseline'а), и меняется ровно одна реализация — остальной API не трогается.

use serde::{Deserialize, Serialize};

/// Признаки текста, из которых считается итоговая оценка.
///
/// Отдельный сериализуемый тип, а не набор аргументов: это ровно тот вектор, который
/// пойдёт в обучение калибровочной модели, и он должен быть виден в диагностике.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummaryFeatures {
    /// Взвешенное по важности среднее вероятностей.
    pub weighted_mean_probability: f64,
    /// Взвешенное геометрическое среднее — жёстче реагирует на одно слабое место.
    pub weighted_geometric_mean: f64,
    /// Худшая вероятность среди требований.
    pub minimum_probability: f64,
    /// Сколько концептов ученик ни разу не встречал.
    pub unknown_concept_count: usize,
    /// Сколько концептов ниже порога `Config::low_probability_threshold`.
    pub low_probability_concept_count: usize,
    /// Всего концептов в тексте.
    pub concept_count: usize,
    /// Из них многословных — их нельзя собрать из известных слов.
    pub multiword_concept_count: usize,
}

impl SummaryFeatures {
    /// Доля слабых концептов. Нормируется, чтобы признак не зависел от длины текста.
    pub fn low_probability_ratio(&self) -> f64 {
        if self.concept_count == 0 {
            return 0.0;
        }
        self.low_probability_concept_count as f64 / self.concept_count as f64
    }
}

/// Стратегия сведения признаков в вероятность приемлемого перевода.
pub trait SentenceScorer: Send + Sync {
    /// Оценка в `0.0..=1.0`. Реализация обязана возвращать число из этого диапазона —
    /// вызывающий код на это полагается.
    fn score(&self, features: &SummaryFeatures) -> f64;
    /// Имя стратегии для диагностики: по отчёту должно быть видно, чем считали.
    fn name(&self) -> &'static str;
}

/// Baseline: логистическая функция от среднего, минимума и доли слабых концептов.
///
/// Коэффициенты вынесены в поля, а не зашиты: их придётся двигать при калибровке, и делать
/// это правкой формулы — способ однажды изменить поведение и не понять почему.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineScorer {
    pub bias: f64,
    pub weight_mean: f64,
    /// Вес худшего звена: одно незнакомое ключевое слово способно уронить перевод целиком.
    pub weight_minimum: f64,
    /// Вес доли слабых концептов — отрицательный.
    pub weight_low_ratio: f64,
}

impl Default for BaselineScorer {
    fn default() -> Self {
        // Значения выставлены так, чтобы «всё знаю» давало ~0.9, «половину знаю» ~0.4,
        // «ничего не знаю» ~0.05. Это калибровка на глаз — см. предупреждение в шапке.
        Self {
            bias: -3.0,
            weight_mean: 3.0,
            weight_minimum: 2.0,
            weight_low_ratio: -2.0,
        }
    }
}

impl SentenceScorer for BaselineScorer {
    fn score(&self, f: &SummaryFeatures) -> f64 {
        if f.concept_count == 0 {
            return 0.0;
        }
        let z = self.weight_mean.mul_add(
            f.weighted_mean_probability,
            self.weight_minimum.mul_add(
                f.minimum_probability,
                self.weight_low_ratio
                    .mul_add(f.low_probability_ratio(), self.bias),
            ),
        );
        logistic(z)
    }

    fn name(&self) -> &'static str {
        "baseline-logistic"
    }
}

fn logistic(z: f64) -> f64 {
    let p = 1.0 / (1.0 + (-z).exp());
    p.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{BaselineScorer, SentenceScorer, SummaryFeatures};

    fn features(mean: f64, min: f64, low: usize, total: usize) -> SummaryFeatures {
        SummaryFeatures {
            weighted_mean_probability: mean,
            weighted_geometric_mean: mean,
            minimum_probability: min,
            unknown_concept_count: 0,
            low_probability_concept_count: low,
            concept_count: total,
            multiword_concept_count: 0,
        }
    }

    #[test]
    fn a_stronger_text_always_scores_higher() {
        let s = BaselineScorer::default();
        let weak = s.score(&features(0.4, 0.1, 3, 5));
        let strong = s.score(&features(0.9, 0.8, 0, 5));
        assert!(strong > weak, "strong={strong} weak={weak}");
    }

    /// Одно провальное звено обязано ронять оценку даже при хорошем среднем: именно из-за
    /// него перевод и разваливается.
    #[test]
    fn a_single_weak_link_pulls_the_score_down() {
        let s = BaselineScorer::default();
        let even = s.score(&features(0.8, 0.75, 0, 4));
        let with_hole = s.score(&features(0.8, 0.1, 1, 4));
        assert!(with_hole < even, "even={even} with_hole={with_hole}");
    }

    #[test]
    fn the_score_always_stays_a_probability() {
        let s = BaselineScorer {
            bias: 500.0,
            ..BaselineScorer::default()
        };
        assert!((0.0..=1.0).contains(&s.score(&features(1.0, 1.0, 0, 3))));
        let s = BaselineScorer {
            bias: -500.0,
            ..BaselineScorer::default()
        };
        assert!((0.0..=1.0).contains(&s.score(&features(0.0, 0.0, 3, 3))));
    }

    #[test]
    fn an_empty_text_scores_zero_rather_than_nan() {
        assert_eq!(
            BaselineScorer::default().score(&features(0.0, 0.0, 0, 0)),
            0.0
        );
    }
}
