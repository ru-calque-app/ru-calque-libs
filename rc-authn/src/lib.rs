//! Client-side JWT ru-calque (ES256).
//!
//! Токены выпускает **только** `ru-calque-auth` (приватным ключом); остальные сервисы
//! их проверяют публичным. Один тип [`Jwt`] покрывает обе роли: verify-сервисы строят
//! его как [`Jwt::verifier`] (без приватного ключа), auth — как [`Jwt::signer`].
//!
//! Экстрактор [`AuthUser`] достаёт `user_id` из `Authorization: Bearer`, проверяя токен
//! через состояние приложения (`S: `[`HasJwt`]) и отвергая [`rc_http::AppError`].

use anyhow::{Context, Result};
use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use rc_http::AppError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Полезная нагрузка access-токена (совпадает с тем, что кладёт auth).
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String, // user id (UUID)
    pub iss: String,
    pub aud: String,
    pub iat: i64,
    pub exp: i64,
}

/// Держатель ключей и параметров ES256. `encoding` есть только у подписанта (auth).
pub struct Jwt {
    decoding: DecodingKey,
    encoding: Option<EncodingKey>,
    issuer: String,
    audience: String,
    access_ttl_secs: i64,
}

impl Jwt {
    /// Только проверка (verify-сервисы): публичный ключ из PEM-байтов.
    pub fn verifier(pub_pem: &[u8], issuer: String, audience: String) -> Result<Self> {
        Ok(Self {
            decoding: DecodingKey::from_ec_pem(pub_pem).context("публичный ключ не ES256 PEM")?,
            encoding: None,
            issuer,
            audience,
            access_ttl_secs: 0,
        })
    }

    /// Только проверка: публичный ключ из PEM-файла.
    pub fn verifier_from_path(path: &str, issuer: String, audience: String) -> Result<Self> {
        let pem =
            std::fs::read(path).with_context(|| format!("не читается публичный ключ {path}"))?;
        Self::verifier(&pem, issuer, audience)
    }

    /// Выпуск + проверка (auth): приватный и публичный ключи из PEM-байтов.
    pub fn signer(
        priv_pem: &[u8],
        pub_pem: &[u8],
        issuer: String,
        audience: String,
        access_ttl_secs: i64,
    ) -> Result<Self> {
        Ok(Self {
            decoding: DecodingKey::from_ec_pem(pub_pem).context("публичный ключ не ES256 PEM")?,
            encoding: Some(
                EncodingKey::from_ec_pem(priv_pem).context("приватный ключ не ES256 PEM")?,
            ),
            issuer,
            audience,
            access_ttl_secs,
        })
    }

    /// Выпуск + проверка (auth): ключи из PEM-файлов.
    pub fn signer_from_paths(
        priv_path: &str,
        pub_path: &str,
        issuer: String,
        audience: String,
        access_ttl_secs: i64,
    ) -> Result<Self> {
        let priv_pem = std::fs::read(priv_path)
            .with_context(|| format!("не читается приватный ключ {priv_path}"))?;
        let pub_pem = std::fs::read(pub_path)
            .with_context(|| format!("не читается публичный ключ {pub_path}"))?;
        Self::signer(&priv_pem, &pub_pem, issuer, audience, access_ttl_secs)
    }

    /// Выпустить access-токен. Возвращает `(jwt, expires_in_secs)`. Ошибка, если
    /// экземпляр без приватного ключа (verify-only).
    pub fn issue_access(&self, user_id: Uuid) -> Result<(String, i64)> {
        let encoding = self
            .encoding
            .as_ref()
            .context("этот Jwt только для проверки (нет приватного ключа)")?;
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            sub: user_id.to_string(),
            iss: self.issuer.clone(),
            aud: self.audience.clone(),
            iat: now,
            exp: now + self.access_ttl_secs,
        };
        let token = encode(&Header::new(Algorithm::ES256), &claims, encoding)
            .context("подпись access-токена")?;
        Ok((token, self.access_ttl_secs))
    }

    /// Проверить access-токен: подпись, `iss`, `aud`, срок.
    pub fn verify(&self, token: &str) -> Result<Claims> {
        let mut v = Validation::new(Algorithm::ES256);
        v.set_issuer(&[self.issuer.as_str()]);
        v.set_audience(&[self.audience.as_str()]);
        let data =
            decode::<Claims>(token, &self.decoding, &v).context("невалидный access-токен")?;
        Ok(data.claims)
    }
}

/// Состояние приложения, из которого экстрактор берёт проверяющий ключ.
/// Сервис реализует его на своём `AppStateInner`; для `Arc<AppStateInner>`
/// (типичный axum-`State`) работает через blanket-impl ниже.
pub trait HasJwt {
    fn jwt(&self) -> &Jwt;
}

impl<T: HasJwt> HasJwt for std::sync::Arc<T> {
    fn jwt(&self) -> &Jwt {
        (**self).jwt()
    }
}

/// Извлекается из запроса: проверяет access-токен (ES256) и даёт `user_id`.
pub struct AuthUser(pub Uuid);

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: HasJwt + Send + Sync,
{
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| AppError::Unauthorized("нет заголовка Authorization".into()))?;
        let token = header
            .strip_prefix("Bearer ")
            .ok_or_else(|| AppError::Unauthorized("ожидается Bearer-токен".into()))?;
        let claims = state
            .jwt()
            .verify(token)
            .map_err(|_| AppError::Unauthorized("невалидный или просроченный токен".into()))?;
        let id = Uuid::parse_str(&claims.sub)
            .map_err(|_| AppError::Unauthorized("битый sub в токене".into()))?;
        Ok(AuthUser(id))
    }
}
