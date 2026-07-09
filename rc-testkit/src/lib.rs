//! Интеграционный тест-стенд Rust-сервисов ru-calque: роутер на эфемерном порту поверх
//! локального Postgres с изоляцией **схема-на-тест**. Токены подписываются встроенной
//! ES256-фикстурой (в проде их выпускает auth) — сервис проверяет [`public_pem`].
//!
//! Сервис-специфику (сборку `AppState`/router) передаёшь замыканием `build_app`:
//! ```ignore
//! let h = Harness::spin("ru_calque_goals", MIGRATIONS, |pool| {
//!     let jwt = rc_authn::Jwt::verifier(rc_testkit::public_pem(), ISS.into(), AUD.into()).unwrap();
//!     crate::http::router(Arc::new(AppStateInner { pool, cfg: test_config(), jwt }))
//! }).await;
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use sqlx::postgres::PgPoolOptions;
use sqlx::{Connection, PgConnection, PgPool};
use uuid::Uuid;

/// Issuer/audience фикстур — совпадают с тем, что кладёт auth и ждут сервисы.
pub const ISSUER: &str = "ru-calque-auth";
pub const AUDIENCE: &str = "ru-calque-api";

static SCHEMA_SEQ: AtomicU64 = AtomicU64::new(0);

const PRIVATE_PEM: &[u8] = include_bytes!("../keys/es256_private.pem");

/// Публичный ключ фикстуры — из него сервис строит свой verify-`Jwt` в `build_app`.
pub fn public_pem() -> &'static [u8] {
    include_bytes!("../keys/es256_public.pem")
}

fn test_database_url(default_db: &str) -> String {
    std::env::var("TEST_DATABASE_URL")
        .unwrap_or_else(|_| format!("postgres://msk-hq-nb-1221@127.0.0.1:5432/{default_db}"))
}

/// Живой стенд: базовый URL, HTTP-клиент, пул на изолированной схеме.
pub struct Harness {
    base_url: String,
    http: reqwest::Client,
    pub pool: PgPool,
}

impl Harness {
    /// Поднять стенд: изолированная схема → миграции → сервер из `build_app`.
    /// `default_db` — имя БД по умолчанию (если нет `TEST_DATABASE_URL`).
    pub async fn spin<F>(default_db: &str, migrations: &[(&str, &str)], build_app: F) -> Self
    where
        F: FnOnce(PgPool) -> Router,
    {
        let pool = isolated_pool(default_db, migrations).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = build_app(pool.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            base_url: format!("http://{addr}"),
            http: reqwest::Client::new(),
            pool,
        }
    }

    /// Подписать access-токен для ученика (то, что в проде делает auth).
    pub fn token(&self, user_id: Uuid) -> String {
        use jsonwebtoken::{encode, Algorithm, EncodingKey, Header};
        let now = chrono::Utc::now().timestamp();
        let claims = serde_json::json!({
            "sub": user_id.to_string(), "iss": ISSUER,
            "aud": AUDIENCE, "iat": now, "exp": now + 3600,
        });
        let key = EncodingKey::from_ec_pem(PRIVATE_PEM).unwrap();
        encode(&Header::new(Algorithm::ES256), &claims, &key).unwrap()
    }

    pub async fn get(&self, path: &str, token: &str) -> reqwest::Response {
        self.req(reqwest::Method::GET, path, token, None).await
    }

    /// GET без Bearer-токена (публичные/dev-эндпоинты).
    pub async fn get_anon(&self, path: &str) -> reqwest::Response {
        self.http.get(self.url(path)).send().await.unwrap()
    }

    /// POST без Bearer-токена, JSON-тело (напр. dev-seed).
    pub async fn post_anon(&self, path: &str, body: serde_json::Value) -> reqwest::Response {
        self.http
            .post(self.url(path))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    /// Полный URL по пути — escape-hatch для полностью кастомных запросов.
    pub fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    /// HTTP-клиент стенда — escape-hatch (свои заголовки и т.п.).
    pub fn client(&self) -> &reqwest::Client {
        &self.http
    }

    /// GET внутреннего эндпоинта с `X-Internal-Key` (без Bearer). `None` — без ключа.
    pub async fn internal_get(&self, path: &str, key: Option<&str>) -> reqwest::Response {
        let mut r = self.http.get(format!("{}{path}", self.base_url));
        if let Some(k) = key {
            r = r.header("x-internal-key", k);
        }
        r.send().await.unwrap()
    }

    pub async fn post(
        &self,
        path: &str,
        token: &str,
        body: serde_json::Value,
    ) -> reqwest::Response {
        self.req(reqwest::Method::POST, path, token, Some(body))
            .await
    }

    pub async fn put(&self, path: &str, token: &str, body: serde_json::Value) -> reqwest::Response {
        self.req(reqwest::Method::PUT, path, token, Some(body))
            .await
    }

    pub async fn patch(
        &self,
        path: &str,
        token: &str,
        body: serde_json::Value,
    ) -> reqwest::Response {
        self.req(reqwest::Method::PATCH, path, token, Some(body))
            .await
    }

    pub async fn delete(&self, path: &str, token: &str) -> reqwest::Response {
        self.req(reqwest::Method::DELETE, path, token, None).await
    }

    async fn req(
        &self,
        method: reqwest::Method,
        path: &str,
        token: &str,
        body: Option<serde_json::Value>,
    ) -> reqwest::Response {
        let mut b = self
            .http
            .request(method, format!("{}{path}", self.base_url))
            .bearer_auth(token);
        if let Some(body) = body {
            b = b.json(&body);
        }
        b.send().await.unwrap()
    }
}

/// Пул на изолированной схеме (`t_<pid>_<seq>`) с прогнанными миграциями.
async fn isolated_pool(default_db: &str, migrations: &[(&str, &str)]) -> PgPool {
    let url = test_database_url(default_db);
    let seq = SCHEMA_SEQ.fetch_add(1, Ordering::SeqCst);
    let schema = format!("t_{}_{}", std::process::id(), seq);

    let mut admin = PgConnection::connect(&url)
        .await
        .expect("нет Postgres: задай TEST_DATABASE_URL");
    sqlx::query(&format!("CREATE SCHEMA \"{schema}\""))
        .execute(&mut admin)
        .await
        .unwrap();
    admin.close().await.ok();

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .after_connect(move |conn, _| {
            let schema = schema.clone();
            Box::pin(async move {
                sqlx::query(&format!("SET search_path TO \"{schema}\""))
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .connect(&url)
        .await
        .unwrap();
    rc_db::migrate(&pool, migrations).await.unwrap();
    pool
}
