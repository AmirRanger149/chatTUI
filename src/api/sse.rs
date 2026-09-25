//! Server-Sent Events framing shared by every provider. OpenAI, Anthropic and
//! Gemini all stream their tokens as `data:` lines over SSE, so the
//! incremental line reader lives here instead of in each backend.

/// Extract the payload of an SSE `data:` line. Returns `None` for blank
/// lines, `event:` lines, or anything that isn't valid UTF-8.
pub(crate) fn sse_data(line: &[u8]) -> Option<&str> {
    std::str::from_utf8(line)
        .ok()?
        .trim_end_matches(['\r', '\n'])
        .strip_prefix("data:")
        .map(str::trim)
}

/// Finish reasons that mean "the model stopped because it ran out of output
/// budget", per wire protocol: OpenAI-compatible (`length`), Anthropic
/// (`max_tokens`) and Gemini (`MAX_TOKENS`).
const TRUNCATED_BY_LIMIT: &[&str] = &["length", "max_tokens", "MAX_TOKENS"];

/// Finish reasons that mean "the provider's filter ended the response", per
/// wire protocol: OpenAI-compatible (`content_filter`) and Gemini's safety
/// reasons.
const ENDED_BY_FILTER: &[&str] = &[
    "content_filter",
    "SAFETY",
    "RECITATION",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
];

/// Explain a stream that produced output but did not end the way the provider
/// intended. `None` means the ending needs no explanation.
///
/// A response is *complete* when the provider sent its end-of-stream marker
/// (`saw_terminator`) **or** reported a finish reason for its last chunk —
/// gateways differ in which of the two they send, and some send both, so
/// either one counts. Anything else is a connection that closed mid-answer.
///
/// This exists because the alternative is silent: a truncated answer looks
/// exactly like a finished one, and the user has no way to tell that the
/// sentence stopped because the provider hung up.
pub(crate) fn abnormal_stream_end(
    saw_terminator: bool,
    finish_reason: Option<&str>,
) -> Option<String> {
    if !saw_terminator && finish_reason.is_none() {
        return Some(
            "⚠ the provider closed the connection before the response finished — the answer above may be incomplete".to_string(),
        );
    }
    match finish_reason {
        Some(reason) if TRUNCATED_BY_LIMIT.contains(&reason) => Some(format!(
            "⚠ the response stopped early: the model reached its output limit ({reason}) — the answer above may be incomplete. Ask it to continue, or raise the model's max output tokens."
        )),
        Some(reason) if ENDED_BY_FILTER.contains(&reason) => Some(format!(
            "⚠ the provider's content filter ended the response early ({reason}) — the answer above may be incomplete"
        )),
        _ => None,
    }
}

/// Whether a response ended because it ran out of output budget, as opposed
/// to the connection simply closing. Worth telling apart because the advice
/// differs: an output cap is fixed by writing less (or by raising
/// `max_output_tokens`), a dropped connection by asking again.
pub(crate) fn ended_by_output_limit(finish_reason: Option<&str>) -> bool {
    finish_reason.map_or(false, |reason| TRUNCATED_BY_LIMIT.contains(&reason))
}

/// Whether a pre-first-token wait was long enough that retrying it would be
/// wasteful. Reasoning models on some gateways buffer their whole chain of
/// thought and send nothing for many minutes; a retry would restart that wait
/// from zero, so past this point the failure is reported instead of retried.
const LONG_FIRST_TOKEN_WAIT_SECS: u64 = 120;

/// Classify "the connection went quiet before the model produced anything".
///
/// A short wait means a dead connection, which a retry can fix. A long wait
/// means the model was almost certainly still working, and a retry would
/// throw that work away and start the same wait again — so it is reported as
/// final instead.
pub(crate) fn first_token_timeout_failure(waited: std::time::Duration) -> crate::api::types::Failure {
    use crate::api::types::Failure;
    let secs = waited.as_secs();
    if secs >= LONG_FIRST_TOKEN_WAIT_SECS {
        Failure::Fatal(format!(
            "no response after {secs}s — the model may still have been working; not retrying, because a retry would restart the same wait"
        ))
    } else {
        Failure::Transient(format!(
            "the connection went quiet for {secs}s before any output"
        ))
    }
}

/// Incremental SSE reader: feed raw bytes as they arrive and drain every
/// complete `data:` payload they produced.
pub(crate) struct SseReader {
    buffer: Vec<u8>,
}

impl SseReader {
    pub(crate) fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        self.buffer.extend_from_slice(chunk);
        while let Some(end) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=end).collect::<Vec<_>>();
            if let Some(data) = sse_data(&line) {
                out.push(data.to_string());
            }
        }
        out
    }

    /// Drain a trailing frame that ended without a final newline. Providers
    /// that close the connection mid-frame must not silently lose the last
    /// event — that is exactly how tool-call arguments get truncated.
    pub(crate) fn flush(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if let Some(data) = sse_data(&line) {
                out.push(data.to_string());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_sse_data() {
        assert_eq!(
            sse_data(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\r\n"),
            Some("{\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}")
        );
    }

    #[test]
    fn ignores_non_data_sse_lines() {
        assert_eq!(sse_data(b"event: message\n"), None);
        assert_eq!(sse_data(b"\n"), None);
    }

    #[test]
    fn reader_drains_data_lines_across_chunks() {
        let mut reader = SseReader::new();
        let mut out = reader.feed(b"event: message\ndata: {\"a\":1}\n\n");
        out.extend(reader.feed(b"data: {\"b\":"));
        out.extend(reader.feed(b"2}\ndata: done\n"));
        assert_eq!(
            out,
            vec![
                "{\"a\":1}".to_string(),
                "{\"b\":2}".to_string(),
                "done".to_string()
            ]
        );
    }

    #[test]
    fn flush_recovers_a_trailing_frame_without_newline() {
        let mut reader = SseReader::new();
        assert!(reader.feed(b"data: {\"tail\":1}").is_empty());
        assert_eq!(reader.flush(), vec!["{\"tail\":1}".to_string()]);
        // Nothing is left behind, and flushing again is harmless.
        assert!(reader.flush().is_empty());
    }

    #[test]
    fn flush_ignores_non_data_trailing_lines() {
        let mut reader = SseReader::new();
        reader.feed(b"event: ping");
        assert!(reader.flush().is_empty());
    }

    #[test]
    fn a_stream_with_a_terminator_needs_no_explanation() {
        // `[DONE]` seen, or a finish reason reported: both are clean endings,
        // even for gateways that only send one of the two.
        assert_eq!(abnormal_stream_end(true, None), None);
        assert_eq!(abnormal_stream_end(true, Some("stop")), None);
        assert_eq!(abnormal_stream_end(false, Some("stop")), None);
        assert_eq!(abnormal_stream_end(false, Some("end_turn")), None);
        assert_eq!(abnormal_stream_end(false, Some("STOP")), None);
        assert_eq!(abnormal_stream_end(false, Some("tool_calls")), None);
    }

    #[test]
    fn a_stream_that_closed_with_no_terminator_is_reported() {
        let notice = abnormal_stream_end(false, None).expect("premature close must be reported");
        assert!(notice.contains("closed the connection"), "{notice}");
        assert!(notice.contains("incomplete"), "{notice}");
    }

    #[test]
    fn output_limit_and_filter_endings_are_reported_with_their_reason() {
        for reason in ["length", "max_tokens", "MAX_TOKENS"] {
            let notice = abnormal_stream_end(true, Some(reason)).unwrap_or_else(|| {
                panic!("{reason} must be reported as a truncated response")
            });
            assert!(notice.contains("output limit"), "{reason}: {notice}");
            assert!(notice.contains(reason), "{reason}: {notice}");
        }
        for reason in ["content_filter", "SAFETY"] {
            let notice = abnormal_stream_end(true, Some(reason)).unwrap_or_else(|| {
                panic!("{reason} must be reported as a filtered response")
            });
            assert!(notice.contains("content filter"), "{reason}: {notice}");
        }
    }

    #[test]
    fn a_short_first_token_wait_is_retryable_and_a_long_one_is_not() {
        use crate::api::types::Failure;
        // A dead connection: retrying is cheap and may work.
        assert!(matches!(
            first_token_timeout_failure(std::time::Duration::from_secs(90)),
            Failure::Transient(_)
        ));
        // A reasoning model that buffered for half an hour: retrying would
        // restart the same wait, so it is reported instead.
        assert!(matches!(
            first_token_timeout_failure(std::time::Duration::from_secs(1800)),
            Failure::Fatal(_)
        ));
    }
}
