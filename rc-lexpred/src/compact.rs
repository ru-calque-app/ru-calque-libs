//! Ограничение размера состояния.
//!
//! Состояние ученика лежит одной строкой в БД и читается на каждый прогноз, поэтому расти
//! бесконечно оно не может. Компактизация выбрасывает наименее полезные поконцептные
//! записи, оставляя ровно `Config::max_concepts` штук.
//!
//! Выброс ничего не теряет статистически: групповые агрегаты и общая статистика ведутся
//! параллельно поконцептным (см. `update::bump_groups`), поэтому вклад выброшенного
//! концепта остаётся в группе и продолжает работать через fallback. Теряется только
//! возможность сказать про этот концепт что-то индивидуальное.
//!
//! Опорная точка времени берётся из самого состояния, а не из часов: иначе одна и та же
//! компактизация давала бы разный результат в разные минуты, и воспроизводимость исчезла бы.

use chrono::{DateTime, Utc};

use crate::concept::ConceptId;
use crate::config::Config;
use crate::state::{ConceptKnowledge, LexicalState};

/// Вклад частоты встреч: концепт, который попадался часто, полезнее случайного.
pub const WEIGHT_FREQUENCY: f64 = 1.0;
/// Вклад неопределённости: про концепт с парой наблюдений выгоднее продолжать копить.
pub const WEIGHT_UNCERTAINTY: f64 = 0.7;
/// Вклад выраженности результата: уверенно провальный или уверенно освоенный концепт
/// сильнее влияет на будущие прогнозы, чем болтающийся у 0.5.
pub const WEIGHT_EXTREMENESS: f64 = 0.5;

/// Урезать состояние до лимита.
///
/// Вызывается автоматически после обновления, но доступна и отдельно — политику надо уметь
/// проверять на собранном состоянии, не гоняя наблюдения.
pub fn compact_state(state: &LexicalState, cfg: &Config) -> LexicalState {
    if state.concepts.len() <= cfg.max_concepts {
        return state.clone();
    }
    let now = newest_timestamp(state);
    let mut ranked: Vec<(ConceptId, f64)> = state
        .concepts
        .iter()
        .map(|(id, k)| (id.clone(), retention_value(k, now, cfg)))
        .collect();
    // Ценность по убыванию, при равенстве — по id: без второго ключа порядок при
    // одинаковых ценностях зависел бы от чисел с плавающей точкой, и компактизация
    // перестала бы быть воспроизводимой.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(cfg.max_concepts);

    let keep: std::collections::BTreeSet<ConceptId> =
        ranked.into_iter().map(|(id, _)| id).collect();
    let mut out = state.clone();
    out.concepts.retain(|id, _| keep.contains(id));
    out
}

/// Самая поздняя отметка времени в состоянии — опора для оценки свежести.
fn newest_timestamp(state: &LexicalState) -> DateTime<Utc> {
    state
        .concepts
        .values()
        .map(ConceptKnowledge::last_seen_at)
        .fold(state.overall.updated_at, DateTime::max)
}

/// Насколько запись стоит держать. Больше — ценнее.
fn retention_value(k: &ConceptKnowledge, now: DateTime<Utc>, cfg: &Config) -> f64 {
    let (s, f) = k.evidence.decayed(now, cfg.half_life_days);
    let total = s + f;
    // Свежесть уже сидит в распавшихся весах: чем старее запись, тем меньше `total`.
    let frequency = f64::from(k.evidence.observation_count).ln_1p();
    let uncertainty = 1.0 / (1.0 + total);
    let extremeness = if total > 0.0 {
        (s / total - 0.5).abs() * 2.0
    } else {
        0.0
    };
    let recency = 1.0 + total;
    recency
        * WEIGHT_FREQUENCY.mul_add(
            frequency,
            WEIGHT_UNCERTAINTY.mul_add(uncertainty, WEIGHT_EXTREMENESS * extremeness),
        )
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone, Utc};

    use super::compact_state;
    use crate::concept::{Concept, ConceptId, ConceptKind, MapCatalog};
    use crate::config::Config;
    use crate::requirement::LexicalObservation;
    use crate::state::LexicalState;
    use crate::update::apply_observations;

    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    /// Одно наблюдение по концепту `i`, повтор `rep`.
    fn observation(i: usize, rep: usize) -> LexicalObservation {
        LexicalObservation::new(
            format!("C{i:03}"),
            1.0,
            1.0,
            at() + Duration::days((i + rep) as i64),
        )
        .unwrap()
    }

    /// Состояние с `n` концептами: чем больше индекс, тем свежее и чаще встречался.
    fn grown(n: usize, cfg: &Config) -> LexicalState {
        let concepts: Vec<Concept> = (0..n)
            .map(|i| Concept::new(format!("C{i:03}"), ConceptKind::Lemma, 0.5))
            .collect();
        let catalog = MapCatalog::new(concepts);
        // Чем больше индекс, тем чаще и свежее встречался концепт.
        let obs: Vec<LexicalObservation> = (0..n)
            .flat_map(|i| (0..=(i % 3)).map(move |rep| (i, rep)))
            .map(|(i, rep)| observation(i, rep))
            .collect();
        apply_observations(&LexicalState::empty(at()), &obs, &catalog, cfg).unwrap()
    }

    #[test]
    fn compaction_respects_the_limit() {
        let cfg = Config {
            max_concepts: 10,
            ..Config::default()
        };
        assert_eq!(grown(50, &cfg).concept_count(), 10);
    }

    /// Компактизация обязана быть воспроизводимой: на неё завязано то, какие концепты
    /// вообще имеют индивидуальную историю, и «то так, то эдак» здесь недопустимо.
    #[test]
    fn compaction_is_deterministic() {
        let cfg = Config {
            max_concepts: 7,
            ..Config::default()
        };
        let state = grown(40, &Config::default());
        let a = compact_state(&state, &cfg);
        let b = compact_state(&state, &cfg);
        assert_eq!(a, b);
        // И повторный прогон по уже сжатому не меняет набор.
        assert_eq!(compact_state(&a, &cfg), a);
    }

    /// Выброшенный концепт не должен уносить статистику с собой: его вклад остаётся в
    /// группах и в общем счётчике, иначе компактизация тихо обнуляла бы историю ученика.
    #[test]
    fn dropping_concepts_keeps_their_contribution_in_the_aggregates() {
        let big = Config::default();
        let full = grown(30, &big);
        let tight = Config {
            max_concepts: 5,
            ..Config::default()
        };
        let small = compact_state(&full, &tight);
        assert_eq!(small.concept_count(), 5);
        assert_eq!(small.overall, full.overall);
        assert_eq!(small.groups, full.groups);
    }

    /// Свежий и часто встречавшийся концепт должен пережить редкий и давний — иначе
    /// компактизация выбрасывает ровно то, что нужнее всего для ближайшего прогноза.
    #[test]
    fn a_frequent_recent_concept_survives_a_stale_rare_one() {
        let cfg = Config {
            max_concepts: 1,
            ..Config::default()
        };
        let catalog = MapCatalog::new([
            Concept::new("STALE", ConceptKind::Lemma, 0.5),
            Concept::new("FRESH", ConceptKind::Lemma, 0.5),
        ]);
        let mut obs = vec![LexicalObservation::new("STALE", 1.0, 1.0, at()).unwrap()];
        for d in 0..5 {
            obs.push(
                LexicalObservation::new("FRESH", 1.0, 1.0, at() + Duration::days(300 + d)).unwrap(),
            );
        }
        let s = apply_observations(&LexicalState::empty(at()), &obs, &catalog, &cfg).unwrap();
        assert!(s.concepts.contains_key(&ConceptId::from("FRESH")));
        assert!(!s.concepts.contains_key(&ConceptId::from("STALE")));
    }
}
