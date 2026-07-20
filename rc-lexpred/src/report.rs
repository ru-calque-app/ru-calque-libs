//! Человекочитаемый отчёт по прогнозу.
//!
//! Нужен для ручной проверки поведения на реальных наборах попыток: числа сами по себе
//! ничего не говорят, а «почему у этого концепта 0.43» видно только рядом с источником и
//! объёмом доказательств.
//!
//! Отчёт — диагностика, а не часть контракта. Ни один расчёт от его форматирования не
//! зависит, и менять формат можно свободно (сломается только golden-файл теста, что и
//! требуется).

use std::fmt::Write;

use crate::predict::{ConceptPrediction, LexicalPrediction, ProbabilitySource};

/// Человекочитаемое имя источника вероятности.
fn source_label(source: ProbabilitySource) -> &'static str {
    match source {
        ProbabilitySource::Concept => "concept history",
        ProbabilitySource::Group => "group fallback",
        ProbabilitySource::Kind => "kind fallback",
        ProbabilitySource::Overall => "overall history",
        ProbabilitySource::GlobalPrior => "catalog prior",
        ProbabilitySource::DefaultPrior => "default prior",
    }
}

/// Отрисовать прогноз.
pub fn render(prediction: &LexicalPrediction) -> String {
    let mut out = String::new();
    out.push_str("Lexical prediction\n------------------\n");
    let _ = writeln!(
        out,
        "Acceptable translation probability: {:.2}  (scorer: {}, not calibrated)",
        prediction.probability_lexically_acceptable, prediction.scorer
    );
    let _ = writeln!(
        out,
        "Expected lexical coverage:          {:.2}",
        prediction.expected_lexical_coverage
    );
    let _ = writeln!(
        out,
        "Evidence quality:                   {:.2}",
        prediction.evidence_quality
    );
    out.push_str("\nConcepts:\n");
    for p in &prediction.concept_predictions {
        push_concept(&mut out, p);
    }
    push_weakest(&mut out, &prediction.weakest_concepts);
    push_skipped(&mut out, prediction);
    out
}

fn push_concept(out: &mut String, p: &ConceptPrediction) {
    let _ = writeln!(out, "- {}: {:.2}", p.concept_id, p.probability);
    let _ = writeln!(out, "  source: {}", source_label(p.probability_source));
    let _ = writeln!(out, "  observations: {}", p.evidence_count);
    if let Some(r) = &p.matched_realization {
        let _ = writeln!(out, "  via realization: {r}");
    }
}

fn push_weakest(out: &mut String, weakest: &[ConceptPrediction]) {
    if weakest.is_empty() {
        return;
    }
    out.push_str("\nWeakest concepts:\n");
    for (i, p) in weakest.iter().enumerate() {
        let _ = writeln!(out, "{}. {} — {:.2}", i + 1, p.concept_id, p.probability);
    }
}

fn push_skipped(out: &mut String, prediction: &LexicalPrediction) {
    if prediction.skipped_concepts.is_empty() {
        return;
    }
    out.push_str("\nSkipped (absent from catalog):\n");
    for id in &prediction.skipped_concepts {
        let _ = writeln!(out, "- {id}");
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::render;
    use crate::concept::{Concept, ConceptKind, MapCatalog};
    use crate::config::Config;
    use crate::predict::predict_requirements;
    use crate::requirement::LexicalRequirement;
    use crate::scorer::BaselineScorer;
    use crate::state::LexicalState;

    /// Отчёт должен объяснять оценку, а не только называть её: без источника и числа
    /// наблюдений «0.43» невозможно ни проверить, ни оспорить.
    #[test]
    fn the_report_explains_where_each_number_came_from() {
        let at = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let catalog =
            MapCatalog::new([Concept::new("REJECT_OFFER", ConceptKind::Collocation, 0.4)]);
        let reqs =
            vec![LexicalRequirement::new("REJECT_OFFER", ConceptKind::Collocation, 1.0).unwrap()];
        let p = predict_requirements(
            &LexicalState::empty(at),
            &reqs,
            &catalog,
            &Config::default(),
            &BaselineScorer::default(),
        )
        .unwrap();
        let text = render(&p);
        assert!(text.contains("REJECT_OFFER: 0.40"), "{text}");
        assert!(text.contains("source: catalog prior"), "{text}");
        assert!(text.contains("observations: 0"), "{text}");
        // Из отчёта должно быть видно, что верхнее число некалибровано.
        assert!(text.contains("not calibrated"), "{text}");
    }
}
