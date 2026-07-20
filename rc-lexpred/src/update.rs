//! Обновление состояния наблюдениями.
//!
//! Функция чистая: то же состояние + те же наблюдения + тот же каталог + тот же конфиг
//! дают тот же результат. Скрытых глобальных счётчиков и обращений к часам здесь нет —
//! время всегда приходит из наблюдения.
//!
//! Стоимость обновления зависит от числа наблюдений в попытке, а не от объёма истории:
//! трогаются только затронутые записи. Всё состояние просматривается ровно в одном месте —
//! в явной компактизации.

use chrono::{DateTime, Utc};

use crate::compact::compact_state;
use crate::concept::{Concept, ConceptCatalog, ConceptId};
use crate::config::{Config, MissingConcept};
use crate::error::{LexError, LexResult};
use crate::requirement::LexicalObservation;
use crate::state::{ConceptKnowledge, Evidence, LexicalState};

/// Применить наблюдения к состоянию.
///
/// Наблюдения обрабатываются в порядке времени, а не в порядке поступления: попытки
/// приезжают из брокера и могут прийти вперемешку, а результат от этого зависеть не должен.
pub fn apply_observations(
    previous: &LexicalState,
    observations: &[LexicalObservation],
    catalog: &dyn ConceptCatalog,
    cfg: &Config,
) -> LexResult<LexicalState> {
    cfg.validate()?;
    previous.check_version()?;
    let mut state = previous.clone();
    for obs in ordered(observations)? {
        apply_one(&mut state, obs, catalog, cfg)?;
    }
    Ok(compact_state(&state, cfg))
}

/// Наблюдения, отсортированные по времени и id — устойчиво к порядку доставки.
fn ordered(observations: &[LexicalObservation]) -> LexResult<Vec<&LexicalObservation>> {
    for o in observations {
        o.validate()?;
    }
    let mut out: Vec<&LexicalObservation> = observations.iter().collect();
    out.sort_by(|a, b| {
        a.observed_at
            .cmp(&b.observed_at)
            .then_with(|| a.concept_id.cmp(&b.concept_id))
    });
    Ok(out)
}

/// Одно наблюдение: концепт, его группы, общая статистика.
fn apply_one(
    state: &mut LexicalState,
    obs: &LexicalObservation,
    catalog: &dyn ConceptCatalog,
    cfg: &Config,
) -> LexResult<()> {
    let Some(concept) = resolve(&obs.concept_id, catalog, cfg)? else {
        return Ok(());
    };
    // Вес наблюдения — это доверие к нему. Свежесть отдельным множителем не нужна: она
    // уже учтена распадом накопленного при добавлении (см. `Evidence::add`).
    let weight = obs.confidence;
    if weight < cfg.min_effective_weight {
        return Ok(());
    }
    bump_concept(state, obs, weight, cfg);
    bump_groups(state, concept.as_ref(), obs, weight, cfg);
    state
        .overall
        .add(obs.outcome, weight, obs.observed_at, cfg.half_life_days);
    Ok(())
}

/// Что делать с концептом, которого нет в каталоге, — решает конфиг, а не каталог.
///
/// `Ok(None)` — наблюдение пропускается. `Ok(Some(None))` — концепт учитывается, но без
/// групп: их взять неоткуда.
type Resolved = Option<Option<Concept>>;

fn resolve(id: &ConceptId, catalog: &dyn ConceptCatalog, cfg: &Config) -> LexResult<Resolved> {
    match (catalog.get(id), cfg.missing_concept) {
        (Some(c), _) => Ok(Some(Some(c))),
        (None, MissingConcept::Fail) => Err(LexError::UnknownConcept(id.clone())),
        (None, MissingConcept::UsePrior) => Ok(Some(None)),
        (None, MissingConcept::Skip) => Ok(None),
    }
}

fn bump_concept(state: &mut LexicalState, obs: &LexicalObservation, weight: f64, cfg: &Config) {
    let entry = state
        .concepts
        .entry(obs.concept_id.clone())
        .or_insert_with(|| ConceptKnowledge::new(obs.observed_at));
    entry.first_seen_at = entry.first_seen_at.min(obs.observed_at);
    entry
        .evidence
        .add(obs.outcome, weight, obs.observed_at, cfg.half_life_days);
}

/// Групповые агрегаты ведутся параллельно поконцептным, а не собираются из них.
///
/// Поэтому выброс концепта при компактизации ничего не теряет: его вклад уже сложен в
/// группы и в общую статистику. Плата за это — концепт учитывается на нескольких уровнях
/// сразу, то есть цепочка постериоров не строгий Байес, а интерпретируемое сглаживание.
fn bump_groups(
    state: &mut LexicalState,
    concept: Option<&Concept>,
    obs: &LexicalObservation,
    weight: f64,
    cfg: &Config,
) {
    let Some(concept) = concept else { return };
    for group in concept.effective_groups() {
        state
            .groups
            .entry(group)
            .or_insert_with(|| Evidence::empty(obs.observed_at))
            .add(obs.outcome, weight, obs.observed_at, cfg.half_life_days);
    }
}

/// Время последнего обновления состояния — максимум по всем уровням.
pub fn last_updated_at(state: &LexicalState) -> DateTime<Utc> {
    state
        .concepts
        .values()
        .map(ConceptKnowledge::last_seen_at)
        .chain(state.groups.values().map(|e| e.updated_at))
        .fold(state.overall.updated_at, DateTime::max)
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone, Utc};

    use super::apply_observations;
    use crate::concept::{Concept, ConceptId, ConceptKind, MapCatalog};
    use crate::config::{Config, MissingConcept};
    use crate::error::LexError;
    use crate::requirement::LexicalObservation;
    use crate::state::LexicalState;

    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    fn catalog() -> MapCatalog {
        MapCatalog::new([
            Concept::new("MAKE_DECISION", ConceptKind::Collocation, 0.5),
            Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.4),
        ])
    }

    fn obs(
        id: &str,
        outcome: f64,
        confidence: f64,
        at: chrono::DateTime<Utc>,
    ) -> LexicalObservation {
        LexicalObservation::new(id, outcome, confidence, at).unwrap()
    }

    fn successes_of(state: &LexicalState, id: &str) -> f64 {
        state.concepts[&ConceptId::from(id)]
            .evidence
            .weighted_successes
    }

    #[test]
    fn a_success_accumulates_evidence_on_the_concept_group_and_overall() {
        let cfg = Config::default();
        let s = apply_observations(
            &LexicalState::empty(at()),
            &[obs("MAKE_DECISION", 1.0, 1.0, at())],
            &catalog(),
            &cfg,
        )
        .unwrap();
        assert!((successes_of(&s, "MAKE_DECISION") - 1.0).abs() < 1e-9);
        assert!((s.overall.weighted_successes - 1.0).abs() < 1e-9);
        assert_eq!(s.groups.len(), 1, "должна появиться группа по виду лексики");
    }

    /// Низкое доверие обязано слабо двигать состояние: сорванное распознавание — не
    /// свидетельство о знании.
    #[test]
    fn a_low_confidence_observation_barely_moves_the_state() {
        let cfg = Config::default();
        let weak = apply_observations(
            &LexicalState::empty(at()),
            &[obs("MAKE_DECISION", 1.0, 0.1, at())],
            &catalog(),
            &cfg,
        )
        .unwrap();
        let strong = apply_observations(
            &LexicalState::empty(at()),
            &[obs("MAKE_DECISION", 1.0, 1.0, at())],
            &catalog(),
            &cfg,
        )
        .unwrap();
        assert!(
            successes_of(&weak, "MAKE_DECISION") < successes_of(&strong, "MAKE_DECISION") / 5.0
        );
    }

    /// Наблюдение с доверием ниже порога вообще не пишется: иначе состояние засоряется
    /// записями, которые ничего не значат, и compaction выбрасывает полезные ради шума.
    #[test]
    fn observations_below_the_weight_floor_are_not_recorded() {
        let cfg = Config {
            min_effective_weight: 0.2,
            ..Config::default()
        };
        let s = apply_observations(
            &LexicalState::empty(at()),
            &[obs("MAKE_DECISION", 1.0, 0.05, at())],
            &catalog(),
            &cfg,
        )
        .unwrap();
        assert!(s.concepts.is_empty());
    }

    #[test]
    fn repeated_observations_accumulate() {
        let cfg = Config {
            half_life_days: None,
            ..Config::default()
        };
        let s = apply_observations(
            &LexicalState::empty(at()),
            &[
                obs("MAKE_DECISION", 1.0, 1.0, at()),
                obs("MAKE_DECISION", 1.0, 1.0, at() + Duration::days(1)),
                obs("MAKE_DECISION", 0.0, 1.0, at() + Duration::days(2)),
            ],
            &catalog(),
            &cfg,
        )
        .unwrap();
        let e = &s.concepts[&ConceptId::from("MAKE_DECISION")].evidence;
        assert_eq!(e.observation_count, 3);
        assert!((e.weighted_successes - 2.0).abs() < 1e-9);
        assert!((e.weighted_failures - 1.0).abs() < 1e-9);
    }

    /// Порядок доставки из брокера случаен, а результат от него зависеть не должен.
    #[test]
    fn delivery_order_does_not_change_the_result() {
        let cfg = Config::default();
        let a = obs("MAKE_DECISION", 1.0, 1.0, at());
        let b = obs("MAKE_DECISION", 0.0, 1.0, at() + Duration::days(5));
        let forward = apply_observations(
            &LexicalState::empty(at()),
            &[a.clone(), b.clone()],
            &catalog(),
            &cfg,
        )
        .unwrap();
        let backward =
            apply_observations(&LexicalState::empty(at()), &[b, a], &catalog(), &cfg).unwrap();
        assert_eq!(forward, backward);
    }

    /// Свежая попытка должна весить больше старой при включённом забывании.
    #[test]
    fn a_recent_attempt_outweighs_an_old_one() {
        let cfg = Config {
            half_life_days: Some(30.0),
            ..Config::default()
        };
        // Успех давно, провал недавно.
        let s = apply_observations(
            &LexicalState::empty(at()),
            &[
                obs("MAKE_DECISION", 1.0, 1.0, at()),
                obs("MAKE_DECISION", 0.0, 1.0, at() + Duration::days(120)),
            ],
            &catalog(),
            &cfg,
        )
        .unwrap();
        let e = &s.concepts[&ConceptId::from("MAKE_DECISION")].evidence;
        assert!(
            e.weighted_failures > e.weighted_successes,
            "старый успех должен был распасться: s={} f={}",
            e.weighted_successes,
            e.weighted_failures
        );
    }

    #[test]
    fn an_unknown_concept_fails_or_is_skipped_per_config() {
        let unknown = [obs("NO_SUCH", 1.0, 1.0, at())];
        let strict = Config {
            missing_concept: MissingConcept::Fail,
            ..Config::default()
        };
        assert_eq!(
            apply_observations(&LexicalState::empty(at()), &unknown, &catalog(), &strict),
            Err(LexError::UnknownConcept(ConceptId::from("NO_SUCH")))
        );

        let skip = Config {
            missing_concept: MissingConcept::Skip,
            ..Config::default()
        };
        let s =
            apply_observations(&LexicalState::empty(at()), &unknown, &catalog(), &skip).unwrap();
        assert!(s.concepts.is_empty());
        assert_eq!(s.overall.observation_count, 0);
    }

    /// Наблюдение с NaN не должно попасть в состояние даже в обход конструктора.
    #[test]
    fn invalid_observations_are_rejected_before_touching_the_state() {
        let bad: LexicalObservation = serde_json::from_value(serde_json::json!({
            "concept_id": "MAKE_DECISION",
            "outcome": 1.0,
            "confidence": 5.0,
            "observed_at": "2026-07-20T12:00:00Z"
        }))
        .unwrap();
        let r = apply_observations(
            &LexicalState::empty(at()),
            &[bad],
            &catalog(),
            &Config::default(),
        );
        assert!(matches!(r, Err(LexError::InvalidProbability { .. })));
    }
}
