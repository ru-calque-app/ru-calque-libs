//! Прогноз по новому тексту.
//!
//! Вероятность знания концепта строится **цепочкой постериоров** от широкого к узкому:
//! дефолт → глобальный prior концепта → общая статистика ученика → его статистика по виду
//! лексики → по группам → по самому концепту. Постериор каждого уровня становится prior
//! следующего.
//!
//! Такая форма даёт требуемое поведение сама, без отдельного костыля: чем больше
//! индивидуальных наблюдений, тем меньше вклад всего, что выше по цепочке, — просто потому
//! что в формуле `(k·p + s) / (k + s + f)` растёт `s + f`.
//!
//! Честная оговорка: строгим Байесом это не является. Одно наблюдение попадает и в концепт,
//! и в его группы, и в общий счётчик, поэтому уровни не независимы и свидетельство
//! учитывается не по одному разу. Это осознанный размен — интерпретируемость и дешёвое
//! обновление против формальной корректности.

use serde::{Deserialize, Serialize};

use crate::concept::{Concept, ConceptCatalog, ConceptId, GroupId};
use crate::config::{Config, MissingConcept};
use crate::error::{LexError, LexResult};
use crate::requirement::LexicalRequirement;
use crate::scorer::{SentenceScorer, SummaryFeatures};
use crate::state::LexicalState;

/// Сколько слабых концептов показывать в отчёте.
pub const WEAKEST_LIMIT: usize = 5;

/// Откуда взялась вероятность. Нужен, чтобы отчёт объяснял оценку, а не только называл её.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbabilitySource {
    /// Индивидуальная история концепта.
    Concept,
    /// Агрегат группы (семейство, уровень, тема).
    Group,
    /// Агрегат по виду лексики.
    Kind,
    /// Общая лексическая статистика ученика.
    Overall,
    /// Глобальный prior концепта из каталога.
    GlobalPrior,
    /// Дефолт из конфигурации: не известно вообще ничего.
    DefaultPrior,
}

/// Прогноз по одному концепту.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptPrediction {
    pub concept_id: ConceptId,
    pub probability: f64,
    pub importance_weight: f64,
    /// Сколько индивидуальных наблюдений стоит за оценкой. `0` — оценка целиком с
    /// вышестоящих уровней.
    pub evidence_count: u32,
    pub probability_source: ProbabilitySource,
    /// Какая реализация дала лучшую вероятность, если выигрыш пришёл от неё.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched_realization: Option<String>,
    pub is_multiword: bool,
}

/// Результат прогноза по тексту.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalPrediction {
    /// Вероятность лексически приемлемого перевода. **Ранжирует, но не откалибрована** —
    /// см. документацию `scorer`.
    pub probability_lexically_acceptable: f64,
    /// Ожидаемая доля лексического содержания, которую ученик передаст. В отличие от
    /// предыдущего числа интерпретируется буквально: это взвешенное среднее.
    pub expected_lexical_coverage: f64,
    /// Насколько прогноз опирается на личную историю, а не на prior'ы: `0.0` — сплошные
    /// догадки, `1.0` — по каждому концепту есть своя история.
    pub evidence_quality: f64,
    /// По каждому требованию, в порядке требований.
    pub concept_predictions: Vec<ConceptPrediction>,
    /// Самые рискованные концепты, отсортированные детерминированно.
    pub weakest_concepts: Vec<ConceptPrediction>,
    pub summary_features: SummaryFeatures,
    /// Концепты, которых не нашлось в каталоге и которые были пропущены
    /// (`MissingConcept::Skip`). Пусто при остальных политиках.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_concepts: Vec<ConceptId>,
    /// Чем считалась итоговая вероятность.
    pub scorer: String,
}

/// Общий контекст расчёта: три вещи, которые ходят вместе через все уровни fallback.
struct Ctx<'a> {
    state: &'a LexicalState,
    catalog: &'a dyn ConceptCatalog,
    cfg: &'a Config,
}

/// Оценить новый текст по готовым требованиям.
pub fn predict_requirements(
    state: &LexicalState,
    requirements: &[LexicalRequirement],
    catalog: &dyn ConceptCatalog,
    cfg: &Config,
    scorer: &dyn SentenceScorer,
) -> LexResult<LexicalPrediction> {
    cfg.validate()?;
    state.check_version()?;
    if requirements.is_empty() {
        return Err(LexError::EmptyRequirements);
    }
    let ctx = Ctx {
        state,
        catalog,
        cfg,
    };
    let mut predictions = Vec::with_capacity(requirements.len());
    let mut skipped = Vec::new();
    for req in requirements {
        req.validate()?;
        match predict_one(&ctx, req)? {
            Some(p) => predictions.push(p),
            None => skipped.push(req.concept_id.clone()),
        }
    }
    if predictions.is_empty() {
        return Err(LexError::EmptyRequirements);
    }
    Ok(assemble(predictions, skipped, cfg, scorer))
}

/// Прогноз по одному требованию. `None` — требование пропущено по политике каталога.
fn predict_one(ctx: &Ctx<'_>, req: &LexicalRequirement) -> LexResult<Option<ConceptPrediction>> {
    let concept = match (ctx.catalog.get(&req.concept_id), ctx.cfg.missing_concept) {
        (Some(c), _) => Some(c),
        (None, MissingConcept::Fail) => {
            return Err(LexError::UnknownConcept(req.concept_id.clone()))
        }
        (None, MissingConcept::Skip) => return Ok(None),
        (None, MissingConcept::UsePrior) => None,
    };
    let (direct_p, direct_source) = probability_of(ctx, &req.concept_id, concept.as_ref(), req);
    let best = best_realization(ctx, req, concept.as_ref(), direct_p);
    let evidence_count = ctx
        .state
        .concepts
        .get(&req.concept_id)
        .map_or(0, |k| k.evidence.observation_count);
    Ok(Some(ConceptPrediction {
        concept_id: req.concept_id.clone(),
        probability: best.probability,
        importance_weight: req.importance_weight,
        evidence_count,
        // Если выиграла реализация, оценка построена на ЕЁ индивидуальной истории —
        // источник должен показывать это, иначе отчёт объясняет число не тем.
        probability_source: best.source.unwrap_or(direct_source),
        matched_realization: best.realization,
        is_multiword: req.concept_kind.is_multiword(),
    }))
}

/// Вероятность знания концепта: цепочка постериоров от широкого к узкому.
fn probability_of(
    ctx: &Ctx<'_>,
    id: &ConceptId,
    concept: Option<&Concept>,
    req: &LexicalRequirement,
) -> (f64, ProbabilitySource) {
    let (state, cfg) = (ctx.state, ctx.cfg);
    let mut p = cfg.default_prior;
    let mut source = ProbabilitySource::DefaultPrior;
    if let Some(c) = concept {
        p = c.base_probability.clamp(0.0, 1.0);
        source = ProbabilitySource::GlobalPrior;
    }
    if state.overall.observation_count > 0 {
        let (s, f) = state.overall.decayed(reference(state), cfg.half_life_days);
        p = posterior(p, cfg.overall_prior_strength, s, f);
        source = ProbabilitySource::Overall;
    }
    if let Some(e) = state.groups.get(&GroupId::kind(req.concept_kind)) {
        let (s, f) = e.decayed(reference(state), cfg.half_life_days);
        p = posterior(p, cfg.group_prior_strength, s, f);
        source = ProbabilitySource::Kind;
    }
    if let Some((s, f)) = other_groups(ctx, concept, req) {
        p = posterior(p, cfg.group_prior_strength, s, f);
        source = ProbabilitySource::Group;
    }
    if let Some(k) = state.concepts.get(id) {
        let (s, f) = k.evidence.decayed(reference(state), cfg.half_life_days);
        p = posterior(p, cfg.prior_strength, s, f);
        source = ProbabilitySource::Concept;
    }
    (p.clamp(0.0, 1.0), source)
}

/// Свидетельства по группам концепта, кроме группы вида (она учтена отдельной ступенью).
/// Складываются, а не применяются по очереди: последовательное применение переучитывало бы
/// одно и то же наблюдение столько раз, во скольких группах состоит концепт.
fn other_groups(
    ctx: &Ctx<'_>,
    concept: Option<&Concept>,
    req: &LexicalRequirement,
) -> Option<(f64, f64)> {
    let (state, cfg) = (ctx.state, ctx.cfg);
    let concept = concept?;
    let kind_group = GroupId::kind(req.concept_kind);
    let mut total = (0.0, 0.0);
    for g in concept
        .effective_groups()
        .iter()
        .filter(|g| **g != kind_group)
    {
        if let Some(e) = state.groups.get(g) {
            let (s, f) = e.decayed(reference(state), cfg.half_life_days);
            total = (total.0 + s, total.1 + f);
        }
    }
    (total.0 + total.1 > 0.0).then_some(total)
}

/// Опорное время для распада — самая поздняя отметка в состоянии.
///
/// Прогноз намеренно не смотрит на часы: одинаковые входы обязаны давать одинаковый
/// результат, а «сколько сейчас времени» — не вход. Забывание учитывается относительно
/// последней активности ученика.
fn reference(state: &LexicalState) -> chrono::DateTime<chrono::Utc> {
    state
        .concepts
        .values()
        .map(|k| k.evidence.updated_at)
        .chain(state.groups.values().map(|e| e.updated_at))
        .fold(state.overall.updated_at, chrono::DateTime::max)
}

/// Лучшая из допустимых реализаций.
///
/// Достаточно знать одну — но обычный noisy-OR по синонимам сильно завышает: знания
/// реализаций зависимы (кто не знает `decline`, чаще не знает и `turn down`), и «хотя бы
/// одну из пяти» такая формула выводит почти в единицу. Поэтому берётся максимум:
/// консервативно и объяснимо. Стратегия локализована здесь — заменить её можно, не трогая
/// остального.
fn best_realization(
    ctx: &Ctx<'_>,
    req: &LexicalRequirement,
    concept: Option<&Concept>,
    base: f64,
) -> BestRealization {
    let realizations = if req.acceptable_realizations.is_empty() {
        concept.map(|c| c.realizations.clone()).unwrap_or_default()
    } else {
        req.acceptable_realizations.clone()
    };
    let mut best = BestRealization {
        probability: base,
        realization: None,
        source: None,
    };
    for r in realizations {
        let Some(lexeme_id) = r.lexeme_id.as_ref() else {
            continue;
        };
        let id = ConceptId::from(lexeme_id.as_str());
        if !ctx.state.concepts.contains_key(&id) {
            continue;
        }
        let (p, _) = probability_of(ctx, &id, ctx.catalog.get(&id).as_ref(), req);
        if p > best.probability {
            best = BestRealization {
                probability: p,
                realization: Some(r.text.clone()),
                source: Some(ProbabilitySource::Concept),
            };
        }
    }
    best
}

/// Победившая реализация вместе с тем, откуда взялась её вероятность.
struct BestRealization {
    probability: f64,
    realization: Option<String>,
    source: Option<ProbabilitySource>,
}

/// Постериор Beta: `(k·p + s) / (k + s + f)`.
///
/// При `k = 2`, `p = 0.5` и целых счётчиках совпадает с правилом Лапласа
/// `(s + 1) / (shows + 2)` из `mastery::domain::score` — эта модель его обобщает, а не
/// заменяет. Пиннится тестом `matches_the_existing_laplace_score_when_unweighted`.
fn posterior(prior_p: f64, prior_strength: f64, successes: f64, failures: f64) -> f64 {
    let denominator = prior_strength + successes + failures;
    if denominator <= 0.0 {
        return prior_p;
    }
    prior_strength.mul_add(prior_p, successes) / denominator
}

/// Сборка итога из поконцептных прогнозов.
fn assemble(
    predictions: Vec<ConceptPrediction>,
    skipped: Vec<ConceptId>,
    cfg: &Config,
    scorer: &dyn SentenceScorer,
) -> LexicalPrediction {
    let features = summarize(&predictions, cfg);
    let coverage = features.weighted_mean_probability;
    LexicalPrediction {
        probability_lexically_acceptable: scorer.score(&features),
        expected_lexical_coverage: coverage,
        evidence_quality: evidence_quality(&predictions),
        weakest_concepts: weakest(&predictions),
        concept_predictions: predictions,
        summary_features: features,
        skipped_concepts: skipped,
        scorer: scorer.name().to_string(),
    }
}

/// Признаки текста. Веса важности учитываются везде, где это осмысленно.
fn summarize(predictions: &[ConceptPrediction], cfg: &Config) -> SummaryFeatures {
    let total_weight: f64 = predictions.iter().map(|p| p.importance_weight).sum();
    let (mean, geometric) = if total_weight > 0.0 {
        let sum: f64 = predictions
            .iter()
            .map(|p| p.importance_weight * p.probability)
            .sum();
        // Через логарифмы: произведение вероятностей длинного текста иначе схлопывается в
        // ноль на арифметике с плавающей точкой.
        let log_sum: f64 = predictions
            .iter()
            .map(|p| p.importance_weight * p.probability.max(1e-12).ln())
            .sum();
        (sum / total_weight, (log_sum / total_weight).exp())
    } else {
        (0.0, 0.0)
    };
    SummaryFeatures {
        weighted_mean_probability: mean,
        weighted_geometric_mean: geometric,
        minimum_probability: predictions
            .iter()
            .map(|p| p.probability)
            .fold(f64::INFINITY, f64::min),
        unknown_concept_count: predictions.iter().filter(|p| p.evidence_count == 0).count(),
        low_probability_concept_count: predictions
            .iter()
            .filter(|p| p.probability < cfg.low_probability_threshold)
            .count(),
        concept_count: predictions.len(),
        multiword_concept_count: predictions.iter().filter(|p| p.is_multiword).count(),
    }
}

/// Доля важности, обеспеченная личной историей. Три наблюдения считаем полным
/// свидетельством: дальше растёт точность, но не сам факт «мы про это что-то знаем».
fn evidence_quality(predictions: &[ConceptPrediction]) -> f64 {
    let total: f64 = predictions.iter().map(|p| p.importance_weight).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let covered: f64 = predictions
        .iter()
        .map(|p| p.importance_weight * (f64::from(p.evidence_count) / 3.0).min(1.0))
        .sum();
    covered / total
}

/// Самые слабые места. Порядок полностью детерминирован: вероятность по возрастанию, при
/// равенстве — важность по убыванию, при равенстве — id. Без последнего ключа отчёт
/// «прыгал» бы между одинаковыми концептами.
fn weakest(predictions: &[ConceptPrediction]) -> Vec<ConceptPrediction> {
    let mut out = predictions.to_vec();
    out.sort_by(|a, b| {
        a.probability
            .total_cmp(&b.probability)
            .then_with(|| b.importance_weight.total_cmp(&a.importance_weight))
            .then_with(|| a.concept_id.cmp(&b.concept_id))
    });
    out.truncate(WEAKEST_LIMIT);
    out
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::{posterior, predict_requirements, LexicalPrediction, ProbabilitySource};
    use crate::concept::{Concept, ConceptKind, GroupId, MapCatalog, Realization};
    use crate::config::Config;
    use crate::error::LexError;
    use crate::requirement::{LexicalObservation, LexicalRequirement};
    use crate::scorer::BaselineScorer;
    use crate::state::LexicalState;
    use crate::update::apply_observations;

    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    fn catalog() -> MapCatalog {
        MapCatalog::new([
            Concept::new("MAKE_DECISION", ConceptKind::Collocation, 0.5)
                .with_groups(vec![GroupId::family("decision")]),
            Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.4)
                .with_groups(vec![GroupId::family("offer")]),
            Concept::new("TAKE_RESPONSIBILITY", ConceptKind::Collocation, 0.45)
                .with_groups(vec![GroupId::family("responsibility")]),
            Concept::new("MAKE", ConceptKind::Lemma, 0.9),
            Concept::new("DECISION", ConceptKind::Lemma, 0.85),
            // Посторонние леммы с теми же prior'ами, что у компонентов: нужны, чтобы
            // сравнивать двух учеников с одинаковой по силе общей историей.
            Concept::new("WEATHER", ConceptKind::Lemma, 0.9),
            Concept::new("BRIDGE", ConceptKind::Lemma, 0.85),
        ])
    }

    fn req(id: &str, kind: ConceptKind) -> LexicalRequirement {
        LexicalRequirement::new(id, kind, 1.0).unwrap()
    }

    fn state_with(observations: &[(&str, f64)]) -> LexicalState {
        let obs: Vec<_> = observations
            .iter()
            .map(|(id, outcome)| LexicalObservation::new(*id, *outcome, 1.0, at()).unwrap())
            .collect();
        apply_observations(
            &LexicalState::empty(at()),
            &obs,
            &catalog(),
            &Config::default(),
        )
        .unwrap()
    }

    /// Прогноз с дефолтными конфигом и скорером — самый частый вызов в тестах.
    fn run(
        state: &LexicalState,
        reqs: &[LexicalRequirement],
        cat: &MapCatalog,
    ) -> LexicalPrediction {
        predict_requirements(
            state,
            reqs,
            cat,
            &Config::default(),
            &BaselineScorer::default(),
        )
        .unwrap()
    }

    fn probability(state: &LexicalState, id: &str, kind: ConceptKind) -> f64 {
        predict_requirements(
            state,
            &[req(id, kind)],
            &catalog(),
            &Config::default(),
            &BaselineScorer::default(),
        )
        .unwrap()
        .concept_predictions[0]
            .probability
    }

    /// Модель обязана обобщать существующую оценку освоенности, а не быть вторым способом
    /// считать то же самое: при силе prior 2, prior 0.5 и единичных весах постериор
    /// совпадает с правилом Лапласа из `mastery::domain::score`.
    #[test]
    fn matches_the_existing_laplace_score_when_unweighted() {
        let laplace = |s: f64, e: f64| (s + 1.0) / (s + e + 2.0);
        for (s, f) in [(0.0, 0.0), (1.0, 0.0), (5.0, 5.0), (3.0, 7.0)] {
            let got = posterior(0.5, 2.0, s, f);
            assert!((got - laplace(s, f)).abs() < 1e-12, "s={s} f={f}: {got}");
        }
    }

    #[test]
    fn a_success_raises_and_a_failure_lowers_the_probability() {
        let base = probability(
            &LexicalState::empty(at()),
            "MAKE_DECISION",
            ConceptKind::Collocation,
        );
        let good = probability(
            &state_with(&[("MAKE_DECISION", 1.0)]),
            "MAKE_DECISION",
            ConceptKind::Collocation,
        );
        let bad = probability(
            &state_with(&[("MAKE_DECISION", 0.0)]),
            "MAKE_DECISION",
            ConceptKind::Collocation,
        );
        assert!(good > base, "успех не поднял: {base} → {good}");
        assert!(bad < base, "провал не опустил: {base} → {bad}");
    }

    /// Знание составных слов НЕ означает знания коллокации: `make` и `decision` по
    /// отдельности ничего не говорят о том, что ученик скажет `make a decision`, а не
    /// `take a decision`.
    ///
    /// Проверяется сравнением двух учеников с ОДИНАКОВОЙ общей историей (по 6 успехов на
    /// леммах), но у одного успехи именно на компонентах коллокации, у другого — на
    /// посторонних словах. Если компоненты протекают в коллокацию, у первого прогноз
    /// окажется выше. Такая форма нужна, чтобы тест не путал протечку компонентов с
    /// законным подъёмом от общей статистики: последний одинаков у обоих.
    #[test]
    fn knowing_the_component_words_does_not_make_a_collocation_known() {
        let components = state_with(&[
            ("MAKE", 1.0),
            ("MAKE", 1.0),
            ("MAKE", 1.0),
            ("DECISION", 1.0),
            ("DECISION", 1.0),
            ("DECISION", 1.0),
        ]);
        let unrelated = state_with(&[
            ("WEATHER", 1.0),
            ("WEATHER", 1.0),
            ("WEATHER", 1.0),
            ("BRIDGE", 1.0),
            ("BRIDGE", 1.0),
            ("BRIDGE", 1.0),
        ]);
        let with_components = probability(&components, "MAKE_DECISION", ConceptKind::Collocation);
        let without = probability(&unrelated, "MAKE_DECISION", ConceptKind::Collocation);
        assert!(
            (with_components - without).abs() < 1e-12,
            "компоненты протекли в коллокацию: {with_components} против {without}"
        );
        // И сама коллокация всё равно должна оставаться ниже освоенного слова.
        let word = probability(&components, "MAKE", ConceptKind::Lemma);
        assert!(
            with_components < word,
            "коллокация подтянулась за словами: {with_components} vs {word}"
        );
    }

    /// Без личной истории концепта должна работать групповая: провалы в семействе
    /// «offer» обязаны ронять прогноз по другому концепту того же семейства.
    /// Каталог из двух концептов одного семейства и состояние, где один из них стабильно
    /// провален.
    fn family_setup() -> (MapCatalog, LexicalState) {
        let catalog = MapCatalog::new([
            Concept::new("DECLINE_INVITATION", ConceptKind::Collocation, 0.4)
                .with_groups(vec![GroupId::family("offer")]),
            Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.4)
                .with_groups(vec![GroupId::family("offer")]),
        ]);
        let obs: Vec<_> = (0..6)
            .map(|_| LexicalObservation::new("REJECT_OFFER", 0.0, 1.0, at()).unwrap())
            .collect();
        let state = apply_observations(
            &LexicalState::empty(at()),
            &obs,
            &catalog,
            &Config::default(),
        )
        .unwrap();
        (catalog, state)
    }

    #[test]
    fn group_history_is_used_when_the_concept_itself_is_unseen() {
        let (catalog, state) = family_setup();
        let p = predict_requirements(
            &state,
            &[req("DECLINE_INVITATION", ConceptKind::Collocation)],
            &catalog,
            &Config::default(),
            &BaselineScorer::default(),
        )
        .unwrap();
        let pred = &p.concept_predictions[0];
        assert_eq!(pred.evidence_count, 0, "личной истории быть не должно");
        assert!(
            pred.probability < 0.4,
            "групповые провалы не учтены: {}",
            pred.probability
        );
        assert!(matches!(
            pred.probability_source,
            ProbabilitySource::Group | ProbabilitySource::Kind
        ));
    }

    /// Совсем незнакомый концепт без групповой истории берёт глобальный prior каталога.
    #[test]
    fn a_wholly_unseen_concept_falls_back_to_the_catalog_prior() {
        let p = predict_requirements(
            &LexicalState::empty(at()),
            &[req("REJECT_OFFER", ConceptKind::Collocation)],
            &catalog(),
            &Config::default(),
            &BaselineScorer::default(),
        )
        .unwrap();
        let pred = &p.concept_predictions[0];
        assert_eq!(pred.probability_source, ProbabilitySource::GlobalPrior);
        assert!((pred.probability - 0.4).abs() < 1e-9);
    }

    /// Общая статистика ученика — уровень fallback между видом лексики и глобальным
    /// prior'ом: ученик, который вообще всё проваливает, должен получать оценку ниже
    /// каталожной даже по невиданному концепту.
    #[test]
    fn overall_history_shifts_an_unseen_concept() {
        let alternating = |i: usize| if i % 2 == 0 { "MAKE" } else { "DECISION" };
        let obs: Vec<_> = (0..10)
            .map(|i| LexicalObservation::new(alternating(i), 0.0, 1.0, at()).unwrap())
            .collect();
        let state = apply_observations(
            &LexicalState::empty(at()),
            &obs,
            &catalog(),
            &Config::default(),
        )
        .unwrap();
        let p = probability(&state, "REJECT_OFFER", ConceptKind::Collocation);
        assert!(p < 0.4, "общая статистика не учтена: {p}");
    }

    /// Порядок прогнозов повторяет порядок требований, а слабейшие отсортированы
    /// детерминированно — на этом строится отчёт и снапшоты.
    #[test]
    fn prediction_and_weakest_ordering_is_stable() {
        let reqs = vec![
            req("TAKE_RESPONSIBILITY", ConceptKind::Collocation),
            req("REJECT_OFFER", ConceptKind::Collocation),
            req("MAKE_DECISION", ConceptKind::Collocation),
        ];
        let state = state_with(&[("MAKE_DECISION", 1.0), ("MAKE_DECISION", 1.0)]);
        let a = run(&state, &reqs, &catalog());
        let b = run(&state, &reqs, &catalog());
        assert_eq!(a, b);
        let ids: Vec<_> = a
            .concept_predictions
            .iter()
            .map(|p| p.concept_id.as_str().to_string())
            .collect();
        assert_eq!(
            ids,
            vec!["TAKE_RESPONSIBILITY", "REJECT_OFFER", "MAKE_DECISION"]
        );
        // Слабейший — тот, у кого ниже вероятность.
        assert!(a.weakest_concepts[0].probability <= a.weakest_concepts[1].probability);
    }

    /// Знание одной реализации подтягивает концепт: ученику достаточно уметь выразить
    /// смысл хоть как-то.
    #[test]
    fn knowing_one_realization_lifts_the_concept() {
        let catalog = MapCatalog::new([
            Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.3),
            Concept::new("pv:turn_down", ConceptKind::PhrasalVerb, 0.3),
        ]);
        let obs: Vec<_> = (0..6)
            .map(|_| LexicalObservation::new("pv:turn_down", 1.0, 1.0, at()).unwrap())
            .collect();
        let state = apply_observations(
            &LexicalState::empty(at()),
            &obs,
            &catalog,
            &Config::default(),
        )
        .unwrap();
        let requirement = req("REJECT_OFFER", ConceptKind::Collocation).with_realizations(vec![
            Realization::from_lexeme("turn down the offer", "pv:turn_down"),
        ]);
        let p = run(&state, &[requirement], &catalog);
        let pred = &p.concept_predictions[0];
        assert_eq!(
            pred.matched_realization.as_deref(),
            Some("turn down the offer")
        );
        assert!(
            pred.probability > 0.5,
            "известная реализация не подтянула: {}",
            pred.probability
        );
    }

    #[test]
    fn an_empty_requirement_set_is_an_error() {
        assert_eq!(
            predict_requirements(
                &LexicalState::empty(at()),
                &[],
                &catalog(),
                &Config::default(),
                &BaselineScorer::default()
            ),
            Err(LexError::EmptyRequirements)
        );
    }

    /// Сериализация состояния не должна менять прогноз: состояние ездит через БД на
    /// каждый запрос, и расхождение здесь означало бы, что прогноз зависит от того,
    /// сохранялись мы или нет.
    ///
    /// Сравнение с допуском намеренно: `serde_json` не round-trip'ит f64 побитово (1 ulp
    /// теряется на числах вроде `0.11999999999999997`). Побитового равенства здесь
    /// требовать нельзя — оно зависит от чужой арифметики, а не от нашей модели.
    #[test]
    fn serialization_round_trip_does_not_change_the_prediction() {
        let state = state_with(&[("MAKE_DECISION", 1.0), ("REJECT_OFFER", 0.0)]);
        let reqs = vec![
            req("MAKE_DECISION", ConceptKind::Collocation),
            req("REJECT_OFFER", ConceptKind::Collocation),
        ];
        let before = run(&state, &reqs, &catalog());
        let restored = LexicalState::from_json(&state.to_json().unwrap()).unwrap();
        let after = run(&restored, &reqs, &catalog());
        for (x, y) in before
            .concept_predictions
            .iter()
            .zip(&after.concept_predictions)
        {
            assert_eq!(x.concept_id, y.concept_id);
            assert_eq!(x.probability_source, y.probability_source);
            assert!(
                (x.probability - y.probability).abs() < 1e-9,
                "{x:?} vs {y:?}"
            );
        }
        assert!((before.expected_lexical_coverage - after.expected_lexical_coverage).abs() < 1e-9);
    }
}
