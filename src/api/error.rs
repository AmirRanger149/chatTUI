//! Failure classification shared by every backend: turning an HTTP status and
//! a response body into a [`Failure`], and distilling the API's error text.

use crate::api::types::Failure;
use serde_json::Value;

/// Decide whether a failed chat response could succeed with a different
/// model. `status` is the HTTP status, or `0` when the API failed inside a
/// 200-OK stream and only the error text is known.
pub(crate) fn classify_failure(status: u16, body: &str) -> Failure {
    let lower = body.to_ascii_lowercase();

    // Wrong credentials are fatal no matter which model is asked.
    const AUTH: &[&str] = &[
        "invalid api key",
        "invalid_api_key",
        "incorrect api key",
        "missing api key",
        "api key not valid",
        "invalid x-api-key",
        "unauthorized",
        "authentication",
        "not authenticated",
        "forbidden",
    ];
    if AUTH.iter().any(|phrase| lower.contains(phrase)) {
        return Failure::Fatal(compact_failure("the API rejected the credentials", status, body));
    }

    // These statuses are the API saying "not this model / not right now":
    // bad requests against the model, missing models, throttling and
    // upstream outages. A different model may well work.
    if matches!(status, 400 | 404 | 408 | 409 | 413 | 422 | 425 | 429)
        || (500..=599).contains(&status)
    {
        return Failure::Retryable(compact_failure("the model rejected the request", status, body));
    }

    // Any other status can still mean "model busy / gone" if the body says so,
    // like the usual "currently experiencing high demand" line.
    const MODEL_TROUBLE: &[&str] = &[
        "high demand",
        "overload",
        "at capacity",
        "capacity",
        "busy",
        "rate limit",
        "too many requests",
        "quota",
        "temporarily unavailable",
        "unavailable",
        "try again",
        "model not found",
        "does not exist",
        "invalid model",
        "not a valid model",
        "no longer available",
        "not available",
        "decommissioned",
        "deprecated",
    ];
    if MODEL_TROUBLE.iter().any(|phrase| lower.contains(phrase)) {
        return Failure::Retryable(compact_failure("the model is not available", status, body));
    }

    // Project/key valid but this model is not enabled (common on 403).
    if status == 403
        && (lower.contains("model") || lower.contains("does not have access"))
    {
        return Failure::Retryable(compact_failure("the model rejected the request", status, body));
    }

    Failure::Fatal(compact_failure("the request failed", status, body))
}

/// One-line reason for a failed attempt: prefer the API's own error message,
/// fall back to the raw body, and keep it short.
fn compact_failure(label: &str, status: u16, body: &str) -> String {
    let detail = error_detail(body);
    if status == 0 {
        format!("{label}: {detail}")
    } else {
        format!("{label} (HTTP {status}: {detail})")
    }
}

/// Extract a short error message from a failed response body: use the
/// OpenAI/Anthropic/Gemini `error.message` when present, otherwise the
/// trimmed body.
pub(crate) fn error_detail(body: &str) -> String {
    let trimmed = body.trim();
    let detail = serde_json::from_str::<Value>(trimmed)
        .ok()
        .and_then(|value| {
            value["error"]["message"]
                .as_str()
                .or_else(|| value["message"].as_str())
                .or_else(|| value["detail"].as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| trimmed.to_string());
    truncate_chars(detail.trim(), 200)
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_retryable_and_fatal_failures() {
        // Bad request for the model → try another one.
        assert!(matches!(
            classify_failure(400, "{\"error\":{\"message\":\"Model not found\"}}"),
            Failure::Retryable(_)
        ));
        // The classic "high demand" overload line.
        assert!(matches!(
            classify_failure(503, "The model is currently experiencing high demand"),
            Failure::Retryable(_)
        ));
        assert!(matches!(classify_failure(429, "slow down"), Failure::Retryable(_)));
        assert!(matches!(classify_failure(500, "oops"), Failure::Retryable(_)));
        // Credentials never get better by switching models.
        assert!(matches!(
            classify_failure(401, "{\"error\":{\"message\":\"Invalid API key\"}}"),
            Failure::Fatal(_)
        ));
        assert!(matches!(
            classify_failure(401, "{\"error\":{\"message\":\"invalid x-api-key\"}}"),
            Failure::Fatal(_)
        ));
        assert!(matches!(classify_failure(405, "method not allowed"), Failure::Fatal(_)));
        assert!(matches!(
            classify_failure(
                403,
                "Project `proj_x` does not have access to model `gpt-4o-mini`"
            ),
            Failure::Retryable(_)
        ));
        // Mid-stream failure texts follow the same rules.
        assert!(matches!(
            classify_failure(0, "Model is currently getting high demand, try later"),
            Failure::Retryable(_)
        ));
        assert!(matches!(classify_failure(0, "unknown Explosion"), Failure::Fatal(_)));
    }

    #[test]
    fn error_detail_prefers_the_api_message() {
        assert_eq!(error_detail("{\"error\":{\"message\":\"boom\"}}"), "boom");
        assert_eq!(error_detail("{\"detail\":\"detailed\"}"), "detailed");
        assert_eq!(error_detail("  plain text  "), "plain text");
        assert_eq!(error_detail(""), "");
        assert!(error_detail(&"x".repeat(500)).chars().count() <= 201);
    }
}
