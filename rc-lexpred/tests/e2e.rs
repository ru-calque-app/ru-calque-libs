//! Сквозной сценарий из фикстуры: пустое состояние → попытки → сериализация → прогноз.
//!
//! Тест параметризован каталогом `tests/fixtures`: чтобы прогнать новый реальный набор
//! попыток, достаточно положить рядом ещё один JSON — ни нового mock-класса, ни правки
//! этого файла не нужно.
//!
//! Проверяются два разных типа утверждений. Свойства («знакомый концепт выше ранее
//! проваленного», «после успеха прогноз улучшается») живут в коде: они должны выполняться
//! на любой фикстуре. Точные числа — в golden-файле `tests/golden/<name>.txt`, который
//! обновляется прогоном с `UPDATE_GOLDEN=1`. Так изменение модели видно построчно в диффе,
//! а не превращается в спор о том, стало лучше или хуже.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rc_lexpred::{
    apply_observations, observe, predict_requirements, report, Attempt, BaselineScorer, Concept,
    Config, Correction, Impact, LexicalObservation, LexicalPrediction, LexicalRequirement,
    LexicalState, MapCatalog, ObserveConfig,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    #[allow(dead_code)]
    comment: String,
    concepts: Vec<Concept>,
    attempts: Vec<FixtureAttempt>,
    target: Target,
    expect: Expectations,
}

#[derive(Debug, Deserialize)]
struct FixtureAttempt {
    #[allow(dead_code)]
    source: String,
    answer: String,
    observed_at: DateTime<Utc>,
    requirements: Vec<LexicalRequirement>,
    corrections: Vec<FixtureCorrection>,
}

/// Владеющая копия правки: `Correction` заимствует строки, а фикстуре нужно их хранить.
#[derive(Debug, Deserialize)]
struct FixtureCorrection {
    item: Option<String>,
    correct: String,
    impact: Impact,
}

#[derive(Debug, Deserialize)]
struct Target {
    #[allow(dead_code)]
    source: String,
    requirements: Vec<LexicalRequirement>,
}

#[derive(Debug, Deserialize)]
struct Expectations {
    /// Какой концепт обязан оказаться самым слабым.
    weakest_concept: String,
    /// Пара «этот должен быть выше того».
    stronger: String,
    weaker: String,
    coverage_min: f64,
    coverage_max: f64,
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{name}.txt"))
}

fn load_fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(fixtures_dir()).expect("каталога фикстур нет");
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    // Порядок обхода каталога не гарантирован — сортируем, иначе тест «то так, то эдак».
    paths.sort();
    for p in paths {
        let raw = std::fs::read_to_string(&p).unwrap();
        let fixture: Fixture = serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("фикстура {p:?} не читается: {e}"));
        out.push(fixture);
    }
    assert!(!out.is_empty(), "в tests/fixtures нет ни одной фикстуры");
    out
}

/// Прогон всех попыток фикстуры в состояние.
fn build_state(fixture: &Fixture, catalog: &MapCatalog, cfg: &Config) -> LexicalState {
    let start = fixture
        .attempts
        .first()
        .map_or_else(Utc::now, |a| a.observed_at);
    let mut state = LexicalState::empty(start);
    for attempt in &fixture.attempts {
        let corrections: Vec<Correction<'_>> = attempt
            .corrections
            .iter()
            .map(|c| Correction {
                item: c.item.as_deref(),
                correct: &c.correct,
                impact: c.impact,
            })
            .collect();
        let a = Attempt::new(&attempt.answer, &corrections, attempt.observed_at);
        let observations =
            observe(&attempt.requirements, &a, &ObserveConfig::default()).expect("разбор попытки");
        state = apply_observations(&state, &observations, catalog, cfg).expect("обновление");
    }
    state
}

fn predict(
    state: &LexicalState,
    fixture: &Fixture,
    catalog: &MapCatalog,
    cfg: &Config,
) -> LexicalPrediction {
    predict_requirements(
        state,
        &fixture.target.requirements,
        catalog,
        cfg,
        &BaselineScorer::default(),
    )
    .expect("прогноз")
}

fn probabilities(prediction: &LexicalPrediction) -> BTreeMap<String, f64> {
    prediction
        .concept_predictions
        .iter()
        .map(|p| (p.concept_id.as_str().to_string(), p.probability))
        .collect()
}

#[test]
fn every_fixture_runs_the_full_flow_and_holds_its_properties() {
    let cfg = Config::default();
    for fixture in load_fixtures() {
        let catalog = MapCatalog::new(fixture.concepts.clone());
        let state = build_state(&fixture, &catalog, &cfg);

        // Состояние обязано пережить круг через хранилище без изменения прогноза: в бою
        // оно ездит через БД на каждый запрос.
        //
        // Сравнение с допуском, а не побитовое: `serde_json` не гарантирует точного
        // round-trip для f64 (проверено — `0.11999999999999997` возвращается как
        // `...95`). Расхождение порядка 1 ulp, и ни одно решение модели на нём не стоит,
        // но требовать здесь `assert_eq!` значит завязать тест на чужую арифметику.
        let restored = LexicalState::from_json(&state.to_json().unwrap()).unwrap();
        assert_eq!(
            state.concepts.keys().collect::<Vec<_>>(),
            restored.concepts.keys().collect::<Vec<_>>(),
            "[{}] после круга через JSON изменился набор концептов",
            fixture.name
        );

        let prediction = predict(&restored, &fixture, &catalog, &cfg);
        assert_close(
            &prediction,
            &predict(&state, &fixture, &catalog, &cfg),
            &fixture.name,
        );

        check_properties(&fixture, &prediction);
        check_improvement(&fixture, &catalog, &cfg, &restored, &prediction);
        check_golden(&fixture, &prediction);
    }
}

/// Прогнозы совпадают с точностью, далеко превосходящей любую содержательную.
///
/// Порог `1e-9` выбран так, чтобы ловить настоящие расхождения (любая ошибка в модели даёт
/// на порядки больше) и не спотыкаться о последний разряд f64 после JSON.
fn assert_close(a: &LexicalPrediction, b: &LexicalPrediction, name: &str) {
    assert_eq!(
        a.concept_predictions.len(),
        b.concept_predictions.len(),
        "[{name}] разное число концептов"
    );
    for (x, y) in a.concept_predictions.iter().zip(&b.concept_predictions) {
        assert_eq!(
            x.concept_id, y.concept_id,
            "[{name}] порядок концептов разъехался"
        );
        assert!(
            (x.probability - y.probability).abs() < 1e-9,
            "[{name}] {}: {} против {}",
            x.concept_id,
            x.probability,
            y.probability
        );
        assert_eq!(
            x.probability_source, y.probability_source,
            "[{name}] {}: разный источник вероятности",
            x.concept_id
        );
    }
    assert!(
        (a.expected_lexical_coverage - b.expected_lexical_coverage).abs() < 1e-9,
        "[{name}] разное покрытие"
    );
    assert!(
        (a.probability_lexically_acceptable - b.probability_lexically_acceptable).abs() < 1e-9,
        "[{name}] разная итоговая оценка"
    );
}

/// Свойства, которые обязаны выполняться на любой фикстуре.
fn check_properties(fixture: &Fixture, prediction: &LexicalPrediction) {
    let name = &fixture.name;
    let probs = probabilities(prediction);
    let e = &fixture.expect;

    let stronger = probs[&e.stronger];
    let weaker = probs[&e.weaker];
    assert!(
        stronger > weaker,
        "[{name}] {} ({stronger:.3}) должен быть выше {} ({weaker:.3})",
        e.stronger,
        e.weaker
    );

    assert_eq!(
        prediction.weakest_concepts[0].concept_id.as_str(),
        e.weakest_concept,
        "[{name}] не тот слабейший концепт"
    );

    let coverage = prediction.expected_lexical_coverage;
    assert!(
        coverage >= e.coverage_min && coverage <= e.coverage_max,
        "[{name}] coverage {coverage:.3} вне [{}, {}]",
        e.coverage_min,
        e.coverage_max
    );

    // Все числа обязаны остаться числами: NaN здесь означал бы, что что-то поделилось на ноль.
    assert!(prediction.probability_lexically_acceptable.is_finite());
    assert!((0.0..=1.0).contains(&prediction.probability_lexically_acceptable));
    assert!((0.0..=1.0).contains(&coverage));
    assert!((0.0..=1.0).contains(&prediction.evidence_quality));
}

/// Успешная попытка по слабейшему концепту обязана улучшать прогноз по связанному тексту.
/// Это главное, ради чего всё считается: модель должна реагировать на прогресс ученика.
fn check_improvement(
    fixture: &Fixture,
    catalog: &MapCatalog,
    cfg: &Config,
    state: &LexicalState,
    before: &LexicalPrediction,
) {
    let weakest = &before.weakest_concepts[0].concept_id;
    let later = before.concept_predictions.iter().map(|_| ()).count().max(1) as i64;
    let at = state.overall.updated_at + chrono::Duration::days(later);
    let success = vec![LexicalObservation::new(weakest.clone(), 1.0, 1.0, at).unwrap()];
    let improved_state = apply_observations(state, &success, catalog, cfg).unwrap();
    let after = predict(&improved_state, fixture, catalog, cfg);

    let before_p = probabilities(before)[weakest.as_str()];
    let after_p = probabilities(&after)[weakest.as_str()];
    assert!(
        after_p > before_p,
        "[{}] успех не поднял {weakest}: {before_p:.3} → {after_p:.3}",
        fixture.name
    );
    assert!(
        after.expected_lexical_coverage > before.expected_lexical_coverage,
        "[{}] успех не поднял общее покрытие",
        fixture.name
    );
}

/// Сверка человекочитаемого отчёта с эталоном. Обновление — `UPDATE_GOLDEN=1 cargo test`.
fn check_golden(fixture: &Fixture, prediction: &LexicalPrediction) {
    let rendered = report::render(prediction);
    let path = golden_path(&fixture.name);
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &rendered).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("нет эталона {path:?} — прогони один раз с UPDATE_GOLDEN=1"));
    assert_eq!(
        rendered, expected,
        "[{}] отчёт разошёлся с эталоном (принять: UPDATE_GOLDEN=1)",
        fixture.name
    );
}
