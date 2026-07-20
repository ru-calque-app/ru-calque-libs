//! Попытка ученика → наблюдения по концептам.
//!
//! Раньше это жило в `ru-calque-eval/src/domain/lexis.rs`. Логика переехала сюда целиком,
//! чтобы лексические вычисления были в одном месте, а не размазаны по сервисам: eval
//! теперь только переводит свой `Report` в нейтральные `Correction` и зовёт `observe`.
//!
//! Ключевое решение (сохранено из eval): **обход — не всегда провал.** Если ученик не
//! сказал точную единицу, но смысл донёс перифразом — это верх лестницы, ось точности, а
//! не «не смог». Если же смысл не донёс — тот же провал, что молчание. Различает их факт
//! наличия разлома в разборе. Раньше любой обход считался провалом, и «пылесос → штука для
//! мусора» роняло готовность как молчание; это был баг.
//!
//! Чего этот модуль **не** делает: не судит о смысле сам. «Донесён ли смысл» приходит
//! снаружи как факт разбора, потому что определить это детерминированно нельзя. Код здесь
//! отвечает только на вопрос «есть ли форма в речи» — а это факт, а не мнение.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{check_probability, LexResult};
use crate::requirement::{LexicalErrorKind, LexicalObservation, LexicalRequirement};

/// Степень вреда от ошибки — та же порядковая шкала, что `CommunicativeImpact` в контракте.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Impact {
    None,
    MinorNoise,
    Distortion,
    Breakdown,
}

/// Правка из разбора: что ученик сказал не так и насколько это повредило.
#[derive(Debug, Clone, PartialEq)]
pub struct Correction<'a> {
    /// Конкретная единица, которую правит разбор, если он её назвал.
    pub item: Option<&'a str>,
    /// Эталонный вариант («как надо»).
    pub correct: &'a str,
    pub impact: Impact,
}

/// Попытка в нейтральном виде: что прозвучало и что разбор поправил.
#[derive(Debug, Clone)]
pub struct Attempt<'a> {
    /// Расшифровка ответа ученика по-английски.
    pub transcript: &'a str,
    pub corrections: &'a [Correction<'a>],
    /// Насколько можно верить расшифровке (качество распознавания). Множится в
    /// `confidence` каждого наблюдения — плохой звук не должен выглядеть незнанием слова.
    pub transcript_confidence: f64,
    pub observed_at: DateTime<Utc>,
}

impl<'a> Attempt<'a> {
    pub fn new(transcript: &'a str, corrections: &'a [Correction<'a>], at: DateTime<Utc>) -> Self {
        Self {
            transcript,
            corrections,
            transcript_confidence: 1.0,
            observed_at: at,
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.transcript_confidence = confidence;
        self
    }

    /// Донёс ли ученик смысл в принципе. Прокси — отсутствие разлома в разборе:
    /// `breakdown` и есть «понять нельзя». Грубо (по всей попытке, не по предложениям) —
    /// ровно как было в eval; уточняется вместе со ступенями готовности.
    fn meaning_conveyed(&self) -> bool {
        !self
            .corrections
            .iter()
            .any(|c| c.impact == Impact::Breakdown)
    }
}

/// Как ученик обошёлся с требуемым концептом.
#[derive(Debug, Clone, PartialEq)]
pub enum Use {
    /// Выразил — одна из допустимых реализаций есть в речи.
    Expressed { realization: String },
    /// Покусился, но выбрал не то: разбор правит именно этот концепт.
    WrongChoice {
        impact: Impact,
        /// Индекс сработавшей правки в `Attempt::corrections` — чтобы вызывающий взял из
        /// неё свои поля, а не искал её заново.
        correction_index: usize,
    },
    /// Не сказал точную единицу, но смысл донёс: удачный обход.
    Workaround,
    /// Не сказал и смысл не донёс.
    Skipped,
}

/// Коэффициенты перевода исхода в числа. Вынесены из алгоритма: это оценочные значения,
/// которые придётся калибровать, и менять их правкой формул — плохая идея.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObserveConfig {
    /// Смысл донесён перифразом: высоко, но не единица.
    pub outcome_workaround: f64,
    /// Насколько верим выводу «это был обход». Вывод косвенный (через отсутствие разлома
    /// во всей попытке), поэтому вес свидетельства снижен осознанно.
    pub confidence_workaround: f64,
    /// Исход при ошибке — по степени вреда.
    pub outcome_impact_none: f64,
    pub outcome_impact_minor: f64,
    pub outcome_impact_distortion: f64,
}

impl Default for ObserveConfig {
    fn default() -> Self {
        Self {
            outcome_workaround: 0.8,
            confidence_workaround: 0.6,
            // Ошибка без вреда — выбор неточен, но смысл на месте.
            outcome_impact_none: 0.7,
            outcome_impact_minor: 0.5,
            outcome_impact_distortion: 0.3,
            // `Breakdown` даёт 0.0 и потому в конфиге не нужен: ниже нуля исхода нет.
        }
    }
}

impl ObserveConfig {
    fn outcome_for(&self, impact: Impact) -> f64 {
        match impact {
            Impact::None => self.outcome_impact_none,
            Impact::MinorNoise => self.outcome_impact_minor,
            Impact::Distortion => self.outcome_impact_distortion,
            Impact::Breakdown => 0.0,
        }
    }

    fn validate(&self) -> LexResult<()> {
        check_probability("outcome_workaround", self.outcome_workaround)?;
        check_probability("confidence_workaround", self.confidence_workaround)?;
        check_probability("outcome_impact_none", self.outcome_impact_none)?;
        check_probability("outcome_impact_minor", self.outcome_impact_minor)?;
        check_probability("outcome_impact_distortion", self.outcome_impact_distortion)?;
        Ok(())
    }
}

/// Наблюдения по требованиям текста из одной попытки.
///
/// Порядок наблюдений повторяет порядок требований — воспроизводимость важнее удобства.
///
/// Требования **без единой допустимой реализации пропускаются**: сопоставлять не с чем, а
/// значит и сказать про такой концепт нечего. Молчать здесь честнее, чем выдать
/// правдоподобное число: раньше такое требование неизбежно классифицировалось как «обход»
/// и приносило ученику 0.8 из воздуха.
pub fn observe(
    requirements: &[LexicalRequirement],
    attempt: &Attempt<'_>,
    cfg: &ObserveConfig,
) -> LexResult<Vec<LexicalObservation>> {
    Ok(observe_detailed(requirements, attempt, cfg)?
        .into_iter()
        .map(|o| o.observation)
        .collect())
}

/// Наблюдение вместе с тем, как оно было получено.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    /// Индекс требования во входном срезе — по нему вызывающий находит свои данные.
    pub requirement_index: usize,
    pub observation: LexicalObservation,
    pub used: Use,
}

/// То же, что [`observe`], но с сохранением классификации.
///
/// Нужно вызывающему, которому мало числа: `ru-calque-eval` строит из наблюдения событие
/// мастерства, и ему требуются степень вреда и та самая правка разбора. Без этого он искал
/// бы правку повторно — то есть держал бы вторую копию логики сопоставления.
pub fn observe_detailed(
    requirements: &[LexicalRequirement],
    attempt: &Attempt<'_>,
    cfg: &ObserveConfig,
) -> LexResult<Vec<Observed>> {
    cfg.validate()?;
    check_probability("transcript_confidence", attempt.transcript_confidence)?;
    let said = said_units(attempt.transcript);
    requirements
        .iter()
        .enumerate()
        .filter(|(_, req)| !req.acceptable_realizations.is_empty())
        .map(|(requirement_index, req)| {
            let used = classify(req, &said, attempt);
            Ok(Observed {
                requirement_index,
                observation: observation(req, &used, attempt, cfg)?,
                used,
            })
        })
        .collect()
}

/// Каноничные единицы, прозвучавшие в ответе. Разбор тем же словарём, которым размечался
/// текст, — поэтому формы совпадают по построению (`moved in` → `move in`).
fn said_units(transcript: &str) -> HashSet<String> {
    rc_lex::analyze(transcript)
        .units
        .into_iter()
        .map(|u| u.unit)
        .collect()
}

/// Куда отнести концепт: выражен, выражен неверно, обойдён, пропущен.
fn classify(req: &LexicalRequirement, said: &HashSet<String>, attempt: &Attempt<'_>) -> Use {
    if let Some(r) = matched_realization(req, said, attempt.transcript) {
        return Use::Expressed { realization: r };
    }
    // Единицы нет — но, может, ученик её ПЫТАЛСЯ взять, и разбор поправил именно её
    // («I make a photo» → «I took a photo»). Это не обход, а ошибка: концепт он знает,
    // просто выбрал не ту реализацию.
    if let Some((i, c)) = attempt
        .corrections
        .iter()
        .enumerate()
        .find(|(_, c)| corrects(c, req))
    {
        return Use::WrongChoice {
            impact: c.impact,
            correction_index: i,
        };
    }
    if attempt.meaning_conveyed() {
        Use::Workaround
    } else {
        Use::Skipped
    }
}

/// Первая допустимая реализация, найденная в речи. Порядок обхода — из требования,
/// поэтому результат детерминирован.
fn matched_realization(
    req: &LexicalRequirement,
    said: &HashSet<String>,
    transcript: &str,
) -> Option<String> {
    let normalized = normalize(transcript);
    req.acceptable_realizations
        .iter()
        .find(|r| mentions(&r.text, said, &normalized))
        .map(|r| r.text.clone())
}

/// Есть ли реализация в речи. Два способа, оба нужны: буквальное вхождение ловит
/// многословные сочетания, которых нет в словаре форм (коллокаций в `rc-lex` нет), а
/// совпадение по леммам ловит словоизменение (`moved in` → `move in`).
///
/// Известное огрубление: у многословной реализации, отсутствующей в словаре, совпадение по
/// леммам не проверяет порядок и близость слов — `make` и `decision` в разных концах
/// длинного ответа зачтутся за `make a decision`. Пиннится тестом
/// `scattered_words_still_count_as_a_collocation`.
fn mentions(realization: &str, said: &HashSet<String>, normalized_transcript: &str) -> bool {
    let target = normalize(realization);
    if target.is_empty() {
        return false;
    }
    if normalized_transcript.contains(&target) {
        return true;
    }
    let lemmas = rc_lex::analyze(realization);
    let content: Vec<_> = lemmas.units.iter().filter(|u| u.is_content()).collect();
    !content.is_empty() && content.iter().all(|u| said.contains(&u.unit))
}

/// Правит ли эта правка именно этот концепт: она названа по имени концепта, по одной из
/// его реализаций, либо эталон содержит одну из реализаций.
fn corrects(c: &Correction<'_>, req: &LexicalRequirement) -> bool {
    let item = c.item.map(str::trim);
    if item == Some(req.concept_id.as_str()) {
        return true;
    }
    let correct_units = said_units(c.correct);
    let correct_norm = normalize(c.correct);
    req.acceptable_realizations
        .iter()
        .any(|r| item == Some(r.text.as_str()) || mentions(&r.text, &correct_units, &correct_norm))
}

/// Одна и та же нормализация для речи и для реализаций: нижний регистр, схлопнутые
/// пробелы, пробелы по краям — чтобы `move  In ` и `move in` были одним и тем же.
fn normalize(text: &str) -> String {
    let words: Vec<String> = text
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();
    format!(" {} ", words.join(" "))
}

/// Наблюдение из классификации.
fn observation(
    req: &LexicalRequirement,
    used: &Use,
    attempt: &Attempt<'_>,
    cfg: &ObserveConfig,
) -> LexResult<LexicalObservation> {
    let (outcome, confidence_factor, error, realization) = match used {
        Use::Expressed { realization } => (1.0, 1.0, None, Some(realization.clone())),
        Use::WrongChoice { impact, .. } => (
            cfg.outcome_for(*impact),
            1.0,
            Some(LexicalErrorKind::WrongChoice),
            None,
        ),
        Use::Workaround => (
            cfg.outcome_workaround,
            cfg.confidence_workaround,
            Some(LexicalErrorKind::Workaround),
            None,
        ),
        Use::Skipped => (0.0, 1.0, Some(LexicalErrorKind::Skipped), None),
    };
    let confidence = attempt.transcript_confidence * confidence_factor;
    let mut obs = LexicalObservation::new(
        req.concept_id.clone(),
        outcome,
        confidence,
        attempt.observed_at,
    )?;
    obs.error_kind = error;
    obs.selected_realization = realization;
    Ok(obs)
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::{observe, Attempt, Correction, Impact, ObserveConfig};
    use crate::concept::{ConceptKind, Realization};
    use crate::requirement::{LexicalErrorKind, LexicalObservation, LexicalRequirement};

    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap()
    }

    fn req(id: &str, kind: ConceptKind, realizations: &[&str]) -> LexicalRequirement {
        LexicalRequirement::new(id, kind, 1.0)
            .unwrap()
            .with_realizations(realizations.iter().map(|r| Realization::new(r)).collect())
    }

    fn requirements() -> Vec<LexicalRequirement> {
        vec![
            req("MOVE_IN", ConceptKind::PhrasalVerb, &["move in"]),
            req("UTILITIES", ConceptKind::Lemma, &["utilities"]),
            req("DEPOSIT", ConceptKind::Lemma, &["deposit"]),
        ]
    }

    fn by_id<'a>(obs: &'a [LexicalObservation], id: &str) -> &'a LexicalObservation {
        obs.iter()
            .find(|o| o.concept_id.as_str() == id)
            .unwrap_or_else(|| panic!("нет наблюдения по {id}"))
    }

    /// Перенос поведения из eval: обход ПРИ ЦЕЛОМ СМЫСЛЕ — это высокий исход, а не провал,
    /// а покусившийся-но-промазавший получает исход по степени вреда.
    #[test]
    fn avoiding_a_unit_while_conveying_meaning_scores_high_not_zero() {
        let corrections = [Correction {
            item: Some("deposit"),
            correct: "I pay a deposit",
            impact: Impact::Distortion,
        }];
        let attempt = Attempt::new(
            "When can I moved in? Is water and light included? I pay a pledge.",
            &corrections,
            at(),
        );
        let obs = observe(&requirements(), &attempt, &ObserveConfig::default()).unwrap();

        // `moved in` — та же единица, что `move in`: словарь лемматизирует.
        assert_eq!(by_id(&obs, "MOVE_IN").outcome, 1.0);
        // `utilities` обойдено описанием, смысл донесён — высокий исход, но пониженная
        // уверенность: вывод косвенный.
        let util = by_id(&obs, "UTILITIES");
        assert_eq!(util.outcome, 0.8);
        assert_eq!(util.confidence, 0.6);
        assert_eq!(util.error_kind, Some(LexicalErrorKind::Workaround));
        // `deposit` пытались взять и промазали — исход по вреду, а не ноль.
        let dep = by_id(&obs, "DEPOSIT");
        assert_eq!(dep.outcome, 0.3);
        assert_eq!(dep.error_kind, Some(LexicalErrorKind::WrongChoice));
    }

    /// Тот же обход, но на фоне разлома: смысл не донесён — это провал, а не перифраз.
    /// Иначе провалившаяся попытка получала бы кредит за то, что просто не назвала единицу.
    #[test]
    fn avoiding_a_unit_amid_a_breakdown_is_a_miss() {
        let corrections = [Correction {
            item: None,
            correct: "I want to understand the date",
            impact: Impact::Breakdown,
        }];
        let attempt = Attempt::new("I no understand when live.", &corrections, at());
        let obs = observe(&requirements(), &attempt, &ObserveConfig::default()).unwrap();
        let util = by_id(&obs, "UTILITIES");
        assert_eq!(util.outcome, 0.0);
        assert_eq!(util.error_kind, Some(LexicalErrorKind::Skipped));
    }

    /// Любая из допустимых реализаций засчитывается за успех: учёт идёт по смыслу, а не по
    /// тому, угадал ли ученик наш вариант перевода.
    #[test]
    fn any_acceptable_realization_counts_as_success() {
        let reqs = vec![req(
            "REJECT_OFFER",
            ConceptKind::Collocation,
            &[
                "reject the offer",
                "decline the offer",
                "turn down the offer",
            ],
        )];
        let attempt = Attempt::new("He turned down the offer politely.", &[], at());
        let obs = observe(&reqs, &attempt, &ObserveConfig::default()).unwrap();
        assert_eq!(obs[0].outcome, 1.0);
        assert_eq!(
            obs[0].selected_realization.as_deref(),
            Some("turn down the offer")
        );
    }

    /// Плохое распознавание должно ронять доверие к наблюдению, а не исход: иначе ученик
    /// получает провал за качество микрофона, и модель уверенно учится ерунде.
    #[test]
    fn poor_transcription_lowers_confidence_not_outcome() {
        let attempt = Attempt::new("I moved in yesterday.", &[], at()).with_confidence(0.3);
        let reqs = vec![req("MOVE_IN", ConceptKind::PhrasalVerb, &["move in"])];
        let obs = observe(&reqs, &attempt, &ObserveConfig::default()).unwrap();
        assert_eq!(obs[0].outcome, 1.0);
        assert_eq!(obs[0].confidence, 0.3);
    }

    /// Известное огрубление, зафиксированное намеренно: у многословной реализации, которой
    /// нет в словаре форм, совпадение по леммам не смотрит на порядок и расстояние. Тест
    /// нужен, чтобы поведение было видно, а не всплыло сюрпризом при разборе жалобы.
    #[test]
    fn scattered_words_still_count_as_a_collocation() {
        let reqs = vec![req(
            "MAKE_DECISION",
            ConceptKind::Collocation,
            &["make a decision"],
        )];
        let attempt = Attempt::new(
            "I make coffee every morning and the decision was hard.",
            &[],
            at(),
        );
        let obs = observe(&reqs, &attempt, &ObserveConfig::default()).unwrap();
        assert_eq!(obs[0].outcome, 1.0, "огрубление: слова засчитаны вразбивку");
    }

    /// Требование, у которого не перечислено ни одной реализации, наблюдать нечем.
    /// Раньше оно молча становилось «обходом» и приносило ученику 0.8 из ничего — число
    /// выглядело осмысленным и было полностью выдуманным.
    #[test]
    fn a_requirement_without_realizations_yields_no_observation() {
        let reqs = vec![LexicalRequirement::new("MYSTERY", ConceptKind::Collocation, 1.0).unwrap()];
        let attempt = Attempt::new("I said something entirely different.", &[], at());
        let obs = observe(&reqs, &attempt, &ObserveConfig::default()).unwrap();
        assert!(
            obs.is_empty(),
            "выдумали наблюдение на пустом месте: {obs:?}"
        );
    }

    #[test]
    fn observation_order_follows_requirement_order() {
        let attempt = Attempt::new("nothing relevant here", &[], at());
        let obs = observe(&requirements(), &attempt, &ObserveConfig::default()).unwrap();
        let ids: Vec<_> = obs
            .iter()
            .map(|o| o.concept_id.as_str().to_string())
            .collect();
        assert_eq!(ids, vec!["MOVE_IN", "UTILITIES", "DEPOSIT"]);
    }
}
