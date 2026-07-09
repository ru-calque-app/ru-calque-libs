//! Postgres-пул и раннер идемпотентных миграций.
//!
//! **Декларация миграций остаётся в сервисе** — список `(имя, sql)` со своими
//! `include_str!` (вшивается на сборке), здесь только логика прогона. DDL идемпотентный
//! (`IF NOT EXISTS`), гоняется на старте по порядку.
//!
//! ```ignore
//! const MIGRATIONS: &[(&str, &str)] = &[
//!     ("01_init", include_str!("../../migrations/01_init.sql")),
//! ];
//! let pool = rc_db::connect(&url).await?;
//! rc_db::migrate(&pool, MIGRATIONS).await?;
//! ```

use anyhow::Result;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

/// Открыть пул соединений (до 5).
pub async fn connect(database_url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;
    Ok(pool)
}

/// Прогнать миграции по порядку. Каждая — идемпотентный DDL (`IF NOT EXISTS`).
pub async fn migrate(pool: &PgPool, migrations: &[(&str, &str)]) -> Result<()> {
    for (name, sql) in migrations {
        sqlx::raw_sql(sql).execute(pool).await?;
        tracing::info!(migration = name, "применена миграция");
    }
    Ok(())
}
