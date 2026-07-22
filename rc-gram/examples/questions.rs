//! На какие вопросы инвентарь отвечает сегодня — исполняемая версия документации.
//!
//! `cargo run -p rc-gram --example questions`

use rc_gram::{inventory, Aspect, Cefr, Grain, Group, Inventory, SkillId};

fn main() {
    let inv = inventory();
    what_is_at_a_level(inv);
    how_much_by_a_level(inv);
    categories(inv);
    one_feature_in_full(inv);
    russian_l1_examples(inv);
    detector_ceiling(inv);
    markers(inv);
    rollup(inv);
    ladder(inv);
    coarse_keys(inv);
    shared_scale_with_lexis(inv);
}

/// Что осваивают на уровне — он же край роста для ступени ниже.
fn what_is_at_a_level(inv: &Inventory) {
    println!("=== 1. Что осваивают на B2 (край роста B1-ученика) ===");
    println!("  всего на B2: {}", inv.at_level(Cefr::B2).count());
    for f in inv.at_level(Cefr::B2).take(3) {
        println!("  [{}] {} — {}", f.category, f.topic, f.can_do);
    }
}

/// Накопительный объём: что ученик к ступени уже должен уметь.
fn how_much_by_a_level(inv: &Inventory) {
    println!("\n=== 2. Объём к ступени (накопительно, складывать нельзя) ===");
    for lv in [Cefr::A1, Cefr::A2, Cefr::B1, Cefr::B2, Cefr::C1, Cefr::C2] {
        println!(
            "  к {lv:?}: {:4} всего · {:3} добавилось на этой ступени",
            inv.up_to(lv).count(),
            inv.at_level(lv).count()
        );
    }
}

fn categories(inv: &Inventory) {
    println!("\n=== 3. Категории и их объём ===");
    let mut cats: Vec<_> = inv
        .categories()
        .into_iter()
        .map(|c| (inv.in_category(c).count(), c))
        .collect();
    cats.sort_unstable_by(|a, b| b.cmp(a));
    for (n, c) in cats.iter().take(5) {
        println!("  {n:4}  {c}");
    }
}

/// Досье на конструкцию плюс образцы для показа ученику.
fn one_feature_in_full(inv: &Inventory) {
    println!("\n=== 4. Одна конструкция целиком ===");
    let f = inv.get("egp:clauses.conditional.c2.01").unwrap();
    println!("  id       {}", f.id);
    println!("  уровень  {:?}", f.level);
    println!("  узел     {}.{}", f.category, f.subcategory);
    println!("  аспект   {:?}", f.aspect);
    println!("  опоры    {:?}", f.markers());
    println!("  can-do   {}", f.can_do);
    println!(
        "\n=== 5. Образцы ({} примеров, годных {} — проваленные работы отсеяны) ===",
        f.examples.len(),
        f.model_examples().count()
    );
    for e in f.model_examples().take(2) {
        println!("  · {}", e.text);
    }
}

/// Как конструкция звучит именно у русскоязычных.
fn russian_l1_examples(inv: &Inventory) {
    println!("\n=== 6. Примеры от русскоязычных авторов ===");
    let ru: Vec<_> = inv
        .all()
        .flat_map(|f| f.examples.iter().map(move |e| (f, e)))
        .filter(|(_, e)| e.l1.as_deref() == Some("Russian"))
        .collect();
    println!("  всего: {}", ru.len());
    for (f, e) in ru.iter().filter(|(f, _)| f.level >= Cefr::C1).take(3) {
        println!("  [{:?} {}] {}", f.level, f.subcategory, e.text);
    }
}

/// Верхняя граница того, что вообще можно найти разбором текста.
fn detector_ceiling(inv: &Inventory) {
    println!("\n=== 7. Сколько инвентаря в принципе детектируемо ===");
    for a in [Aspect::Form, Aspect::FormUse, Aspect::Use, Aspect::Other] {
        let n = inv.all().filter(|f| f.aspect == a).count();
        let verdict = if a.is_detectable_in_principle() {
            "матчибельно"
        } else {
            "— никогда"
        };
        println!("  {a:?}: {n:4}  {verdict}");
    }
    let d = inv
        .all()
        .filter(|f| f.aspect.is_detectable_in_principle())
        .count();
    println!("  потолок: {}%", d * 100 / inv.len());
}

/// Опорные слова — сырьё для будущего детектора.
fn markers(inv: &Inventory) {
    println!("\n=== 8. Конструкции с явными опорными словами ===");
    let with = inv.all().filter(|f| !f.markers().is_empty()).count();
    println!("  {with} из {} записей", inv.len());
    for f in inv.all().filter(|f| f.markers().len() > 1).take(3) {
        println!("  {:?} {:?}", f.markers(), f.topic);
    }
}

/// Одни и те же данные на четырёх разрешениях — свёртка живёт на чтении.
fn rollup(inv: &Inventory) {
    println!("\n=== 9. Свёртка: хранение полное, разрешение по требованию ===");
    for g in [Grain::Category, Grain::Node, Grain::Rung, Grain::Feature] {
        let r = inv.rollup(g);
        let total: usize = r.iter().map(Group::len).sum();
        println!("  {g:?}: {:4} групп, записей внутри {total}", r.len());
    }
}

/// Лестница узла — «что дальше» внутри темы.
fn ladder(inv: &Inventory) {
    println!("\n=== 10. Лестница egp:clauses.conditional ===");
    for rung in inv.ladder(&SkillId::from("egp:clauses.conditional")) {
        let (lo, _) = rung.level_span().unwrap();
        println!("  {lo:?}  {:32} {} записей", rung.id.as_str(), rung.len());
    }
}

/// Грубое наблюдение адресует точные — префиксом, без таблицы соответствий.
fn coarse_keys(inv: &Inventory) {
    println!("\n=== 11. Разметчик не уверен → кладёт префикс ===");
    for key in [
        "egp:clauses",
        "egp:clauses.conditional",
        "egp:clauses.conditional.c2",
        "egp:clauses.conditional.c2.01",
    ] {
        let id = SkillId::from(key);
        println!(
            "  {key:32} {:?}\t→ {} записей",
            id.grain().unwrap(),
            inv.under(&id).count()
        );
    }
}

/// Уровень грамматики и уровень лексики — один тип, сравниваются напрямую.
fn shared_scale_with_lexis(inv: &Inventory) {
    println!("\n=== 12. Общая шкала с rc-lex ===");
    let g = inv.get("egp:clauses.conditional.c2.01").unwrap().level;
    let l = rc_gram::rc_lex::dict()
        .lookup_pos("utility", "noun")
        .unwrap()
        .cefr;
    println!(
        "  конструкция {g:?} vs слово 'utility' {l:?} → лексика ниже: {}",
        l < g
    );
}
