//! Общие HTTP-примитивы Rust-сервисов ru-calque.
//!
//! - [`AppError`] / [`ApiResult`] — доменная ошибка запроса в формате RFC 9457
//!   (`application/problem+json`), как в контракте.
//! - [`health`] — тривиальный health-хендлер (`200 "ok"`).

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

/// Доменная ошибка запроса. Каждый вариант маппится на HTTP-код и `Problem`.
/// Часть вариантов может не конструироваться конкретным сервисом — оставлены как
/// готовая поверхность API (сервисы глушат dead_code при желании).
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

/// Тело ответа RFC 9457.
#[derive(Serialize)]
struct Problem {
    #[serde(rename = "type")]
    type_: String,
    title: String,
    status: u16,
    detail: String,
}

impl AppError {
    fn parts(&self) -> (StatusCode, &'static str, &'static str) {
        match self {
            Self::Validation(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "validation",
                "Validation failed",
            ),
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad-request", "Bad request"),
            Self::Unauthorized(_) => (StatusCode::UNAUTHORIZED, "unauthorized", "Unauthorized"),
            Self::Forbidden(_) => (StatusCode::FORBIDDEN, "forbidden", "Forbidden"),
            Self::NotFound(_) => (StatusCode::NOT_FOUND, "not-found", "Not found"),
            Self::Conflict(_) => (StatusCode::CONFLICT, "conflict", "Conflict"),
            Self::PayloadTooLarge(_) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload-too-large",
                "Payload too large",
            ),
            Self::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "Internal server error",
            ),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, slug, title) = self.parts();
        // Внутренние ошибки логируем полностью, но клиенту не раскрываем.
        let detail = match &self {
            Self::Internal(e) => {
                tracing::error!(error = ?e, "внутренняя ошибка");
                "Internal server error.".to_string()
            }
            Self::Validation(m)
            | Self::BadRequest(m)
            | Self::Unauthorized(m)
            | Self::Forbidden(m)
            | Self::NotFound(m)
            | Self::Conflict(m)
            | Self::PayloadTooLarge(m) => m.clone(),
        };
        let body = Problem {
            type_: format!("https://errors.ru-calque.app/{slug}"),
            title: title.to_string(),
            status: status.as_u16(),
            detail,
        };
        (
            status,
            [(header::CONTENT_TYPE, "application/problem+json")],
            Json(body),
        )
            .into_response()
    }
}

/// Тип-алиас для хендлеров: `Result<T, AppError>`.
pub type ApiResult<T> = Result<T, AppError>;

/// Health-хендлер: `200 "ok"`. Роут: `.route("/health", get(rc_http::health))`.
pub async fn health() -> &'static str {
    "ok"
}
