//! The Google Gemini backend (`POST /v1beta/models/{model}:streamGenerateContent`).
//!
//! Gemini's dialect differs from OpenAI's in several ways the adapter papers
//! over:
//! - streaming is enabled with a `?alt=sse` query parameter (without it the
//!   endpoint returns one JSON array instead of a stream);
//! - the system prompt goes in `systemInstruction` (a single content block);
//! - turns are `contents` with a `parts: [{text}]` list, alternating
//!   `user`/`model` (an assistant reply is sent as `role: "model"`);
//! - the model id is part of the URL (the `{model}` from the endpoint may be
//!   a `models/`-prefixed display name, so the prefix is normalised away);
//! - the key travels as an `x-goog-api-key` header, not a bearer token;
//! - the streamed `parts[].text` is *cumulative* (each chunk repeats the whole
//!   answer so far), so only the new suffix is emitted, and thinking models
//!   put their reasoning in `"thought": true` parts which are skipped;
//! - the model list lives under `models[]` with a `displayName` (usually the
//!   full `models/{name}` path).
//! - Tool calling uses `functionDeclarations` and `functionCall`/`functionResponse`.

use super::{bounded_body, check_request_size, ChatBackend, HttpTimeouts};
use crate::api::error::{classify_failure, error_detail};
use crate::api::sse::{SseReader, MAX_CONSECUTIVE_SSE_PARSE_FAILURES};
use crate::api::types::{
    CompletionRequest, Failure, Role, StreamEvent, ToolCall, MAX_RESPONSE_TEXT_BYTES,
    MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_CALLS_PER_TURN,
};
use crate::tools;
use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc::Sender;

pub struct GeminiBackend {
    http: Client,
    api_key: String,
    base_url: String,
    /// How long the stream may stay quiet between chunks before the
    /// connection counts as dead. There is deliberately no whole-request
    /// timeout: it would cut long generations off mid-answer.
    idle_timeout: Duration,
}

impl GeminiBackend {
    pub fn new(api_key: String, base_url: String, timeouts: HttpTimeouts) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(timeouts.connect)
                .build()
                .expect("HTTP client"),
            api_key,
            base_url: base_url.trim_end_matches('/').into(),
            idle_timeout: timeouts.idle,
        }
    }

    /// Turn a model id into the `{model}` URL segment. Accepts a bare id
    /// (`some-model-2.0`), a `models/…`-prefixed display name, or a full
    /// `publishers/…/models/…` path.
    fn path_model(model: &str) -> String {
        let path = model.trim();
        if let Some(pos) = path.rfind("models/") {
            return path[pos + "models/".len()..].to_string();
        }
        path.to_string()
    }

    async fn list_models_inner(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        let response = tokio::time::timeout(
            self.idle_timeout,
            self.http
                .get(url)
                .header("x-goog-api-key", &self.api_key)
                .send(),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "model list request timed out after {}s",
                self.idle_timeout.as_secs()
            )
        })?
        .context("sending GET /models request")?;
        let status = response.status();
        let bytes = bounded_body(response, self.idle_timeout).await
            .context("reading the model list response")?;
        if !status.is_success() {
            return Err(anyhow!(
                "model list request failed (HTTP {}): {}",
                status,
                error_detail(&String::from_utf8_lossy(&bytes))
            ));
        }
        let value: Value = serde_json::from_slice(&bytes)
            .context("parsing the model list response")?;
        let mut ids = value["models"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .filter_map(|item| {
                item["name"]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| item["displayName"].as_str().map(str::to_string))
            })
            .collect::<Vec<_>>();
        ids.sort_by_key(|id| id.to_ascii_lowercase());
        if ids.is_empty() {
            return Err(anyhow!("the endpoint returned no models"));
        }
        Ok(ids)
    }

    async fn stream_inner(
        &self,
        request: CompletionRequest,
        tx: Sender<StreamEvent>,
    ) -> Result<(), Failure> {
        let mut system: Option<String> = None;
        let mut contents = Vec::with_capacity(request.messages.len());
        for message in &request.messages {
            match message.role {
                Role::System => {
                    system = Some(message.content.clone());
                }
                Role::Assistant => {
                    if let Some(tool_calls) = &message.tool_calls {
                        let mut parts = Vec::new();
                        if !message.content.is_empty() {
                            parts.push(json!({ "text": message.content }));
                        }
                        for tc in tool_calls {
                            parts.push(function_call_part(tc));
                        }
                        contents.push(json!({
                            "role": "model",
                            "parts": parts
                        }));
                    } else {
                        contents.push(json!({
                            "role": "model",
                            "parts": [{ "text": message.content }],
                        }));
                    }
                }
                Role::User => contents.push(json!({
                    "role": "user",
                    "parts": [{ "text": message.content }],
                })),
                Role::Tool => {
                    // Tool result -> functionResponse
                    let tool_name = message
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string());
                    // tool_call_id actually holds the function name for Gemini? We need to parse.
                    // For Gemini, we store id as name for simplicity, but we need to handle.
                    // We'll use the content as response, and need to know function name.
                    // We'll store tool name in tool_call_id as "name:id" or just name.
                    let (func_name, _) = if let Some((name, _id)) = tool_name.split_once(':') {
                        (name, _id)
                    } else {
                        (tool_name.as_str(), "")
                    };
                    let func_name = if func_name.is_empty() {
                        "unknown"
                    } else {
                        func_name
                    };
                    contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": func_name,
                                "response": { "result": message.content }
                            }
                        }]
                    }));
                }
            }
        }
        let mut body = json!({
            "contents": contents,
            "generationConfig": { "temperature": request.temperature },
        });
        if let Some(text) = system {
            body["systemInstruction"] = json!({ "parts": [{ "text": text }] });
        }

        // Add tools if present
        if !request.tools.is_empty() {
            let tool_defs: Vec<crate::tools::ToolDefinition> = request
                .tools
                .iter()
                .map(|t| {
                    crate::tools::ToolDefinition::new(&t.name, &t.description, t.parameters.clone())
                })
                .collect();
            body["tools"] = tools::to_gemini_tools(&tool_defs);
        }

        check_request_size(&body)?;

        let model = Self::path_model(&request.model);
        let url = format!(
            "{}/models/{model}:streamGenerateContent?alt=sse",
            self.base_url
        );
        let response = self
            .http
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send();
        let response = tokio::time::timeout(self.idle_timeout, response)
            .await
            .map_err(|_| Failure::Transient(format!(
                "request headers timed out after {}s", self.idle_timeout.as_secs()
            )))?
            .map_err(|error| Failure::Transient(format!("request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = bounded_body(response, self.idle_timeout).await
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_else(|error| format!("reading error response failed: {error}"));
            return Err(classify_failure(status.as_u16(), &body));
        }
        let mut stream = response.bytes_stream();
        let mut reader = SseReader::new();
        // Keep-alive pings do not count as progress: require actual answer
        // text or tool-call data within the configured progress window.
        let mut progress_deadline = std::time::Instant::now() + self.idle_timeout;
        // Cumulative text of the CURRENT answer segment (see text_delta).
        let mut segment_text = String::new();
        let mut any_text = false;
        let mut finish_reason = String::new();
        let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
        let mut previous_frame_calls: Vec<ToolCall> = Vec::new();
        let mut bad_lines = 0usize;

        let mut stream_ended = false;
        let mut saw_done = false;
        loop {
            // Idle-timeout guard: a stream may run as long as it keeps
            // producing chunks, but a connection that stays quiet past
            // `idle_timeout` is dead and must not hang the UI forever.
            // When the stream closes cleanly, whatever trailing frame the
            // reader still holds (a provider may end without a final
            // newline) is processed too — losing it can truncate tool-call
            // arguments.
            let payloads = if stream_ended {
                break;
            } else {
                let remaining = progress_deadline.saturating_duration_since(std::time::Instant::now());
                match tokio::time::timeout(remaining, stream.next()).await {
                    Ok(Some(Ok(chunk))) => reader.feed(&chunk)
                        .map_err(|error| Failure::Fatal(error.into()))?,
                    Ok(Some(Err(error))) => {
                        return Err(if any_text {
                            Failure::Fatal(format!(
                                "stream interrupted after output started: {error}"
                            ))
                        } else {
                            Failure::Transient(format!(
                                "stream interrupted before any output: {error}"
                            ))
                        });
                    }
                    Ok(None) => {
                        stream_ended = true;
                        reader.flush()
                    }
                    Err(_elapsed) => {
                        let secs = self.idle_timeout.as_secs();
                        return Err(if any_text {
                            Failure::Fatal(format!(
                                "no model progress for {secs}s after output started"
                            ))
                        } else {
                            Failure::Transient(format!(
                                "no model progress for {secs}s before any output"
                            ))
                        });
                    }
                }
            };
            for data in payloads {
                if data == "[DONE]" {
                    saw_done = true;
                    stream_ended = true;
                    break;
                }
                let value: Value = match serde_json::from_str(&data) {
                    Ok(value) => {
                        bad_lines = 0;
                        value
                    }
                    Err(_) => {
                        // Gateways occasionally emit junk or keep-alive
                        // lines; a single one must not kill the stream. Only
                        // a run of consecutive garbage is treated as broken.
                        bad_lines += 1;
                        if bad_lines >= MAX_CONSECUTIVE_SSE_PARSE_FAILURES {
                            return Err(Failure::Fatal(format!(
                                "the stream sent {bad_lines} unparseable lines in a row (last: {data})"
                            )));
                        }
                        continue;
                    }
                };
                if let Some(reason) = value["promptFeedback"]["blockReason"].as_str() {
                    return Err(Failure::Fatal(format!(
                        "Gemini blocked the prompt: {reason}"
                    )));
                }
                if let Some(message) = value["error"]["message"].as_str() {
                    if any_text {
                        return Err(Failure::Fatal(format!("API error mid-stream: {message}")));
                    }
                    return Err(classify_failure(0, message));
                }
                if let Some(reason) = value["candidates"][0]["finishReason"].as_str() {
                    finish_reason = reason.to_string();
                }

                // Collect one frame first: Gemini may repeat cumulative
                // functionCall parts in later frames. A completed call must
                // appear once in the executable result, not once per chunk.
                let mut frame_calls = Vec::new();
                if let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() {
                    for part in parts {
                        if let Some(func_call) = part.get("functionCall") {
                            if let Some(name) = func_call["name"].as_str() {
                                let args = func_call["args"].clone();
                                if name.len() > 256 || frame_calls.len() >= MAX_TOOL_CALLS_PER_TURN {
                                    return Err(Failure::Fatal("tool call exceeded the response limit".into()));
                                }
                                let args_str = if args.is_object() {
                                    serde_json::to_string(&args)
                                        .unwrap_or_else(|_| "{}".to_string())
                                } else {
                                    "{}".to_string()
                                };
                                if args_str.len() > MAX_TOOL_ARGUMENT_BYTES {
                                    return Err(Failure::Fatal("tool arguments exceeded the size limit".into()));
                                }
                                let id = format!("{}:{}", name, frame_calls.len());
                                let mut tc = ToolCall::new(id, name, args_str);
                                // The part's opaque reasoning signature is the
                                // only carrier of the model's thinking state;
                                // keep it so replay can echo it back verbatim
                                // (Gemini rejects turns that omit it: HTTP 400
                                // "missing a thought_signature").
                                if let Some(signature) = part["thoughtSignature"].as_str() {
                                    if signature.len() > 64 * 1024 {
                                        return Err(Failure::Fatal("tool signature exceeded the size limit".into()));
                                    }
                                    tc = tc.with_signature(signature.to_string());
                                }
                                frame_calls.push(tc);
                            }
                        }
                    }
                }
                if !frame_calls.is_empty()
                    && append_new_frame_calls(&mut pending_tool_calls, &mut previous_frame_calls, frame_calls)?
                {
                    progress_deadline = std::time::Instant::now() + self.idle_timeout;
                }

                // Text handling. Gemini normally streams cumulative text:
                // each frame repeats the whole answer so far, and only the
                // new suffix is emitted. With interleaved thinking, though,
                // the accumulation resets after a thinking segment — the
                // next answer piece arrives as a fresh part that does not
                // extend the previous frame. Dropping those frames is what
                // cut answers short after a few words.
                let mut full = String::new();
                if let Some(parts) = value["candidates"][0]["content"]["parts"].as_array() {
                    for part in parts {
                        if part["thought"].as_bool() == Some(true) {
                            continue;
                        }
                        if part.get("functionCall").is_some() {
                            continue; // Skip function calls for text
                        }
                        if let Some(text) = part["text"].as_str() {
                            if full.len().saturating_add(text.len()) > MAX_RESPONSE_TEXT_BYTES {
                                return Err(Failure::Fatal("response exceeded the size limit".into()));
                            }
                            full.push_str(text);
                        }
                    }
                }
                if let Some(delta) = text_delta(&segment_text, &full) {
                    tx.send(StreamEvent::Delta(delta))
                        .await
                        .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
                    segment_text = full;
                    any_text = true;
                    progress_deadline = std::time::Instant::now() + self.idle_timeout;
                }
            }
        }

        if !finish_reason.is_empty() && finish_reason != "STOP" {
            return Err(Failure::Fatal(format!(
                "Gemini stopped before completing its response ({finish_reason}); tool calls were not run"
            )));
        }
        if !pending_tool_calls.is_empty() && finish_reason.is_empty() && !saw_done {
            return Err(Failure::Fatal("tool-call stream closed without a completion marker; no calls were run".into()));
        }
        if !any_text && pending_tool_calls.is_empty() {
            return Err(Failure::Fatal("Gemini returned an empty response".into()));
        }
        // Publish executable calls only on successful completion. Buffered
        // function calls from a failed attempt can now be discarded safely.
        for tc in pending_tool_calls {
            tx.send(StreamEvent::ToolCall(tc)).await
                .map_err(|_| Failure::Fatal("stream receiver closed".into()))?;
        }
        Ok(())
    }
}

impl ChatBackend for GeminiBackend {
    fn list_models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>> {
        Box::pin(self.list_models_inner())
    }

    fn stream_completion(
        &self,
        request: &CompletionRequest,
        tx: Sender<StreamEvent>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Failure>> + Send + '_>> {
        Box::pin(self.stream_inner(request.clone(), tx))
    }
}

/// Compare function-call identity without the opaque thought signature: some
/// cumulative frames add that signature only after an earlier frame.
fn same_frame_call(a: &ToolCall, b: &ToolCall) -> bool {
    a.name == b.name && a.arguments == b.arguments
}

/// Ignore repeated/cumulative function-call parts across frames, including
/// a sliding window of already-seen parts. Keep duplicates within one frame:
/// they can be distinct intended calls. Contradictory cumulative prefixes
/// fail closed rather than executing both versions of an ambiguous call.
fn append_new_frame_calls(
    pending: &mut Vec<ToolCall>,
    previous: &mut Vec<ToolCall>,
    frame: Vec<ToolCall>,
) -> Result<bool, Failure> {
    let with_pending = frame.iter().zip(pending.iter())
        .take_while(|(a, b)| same_frame_call(a, b)).count();
    let with_previous = frame.iter().zip(previous.iter())
        .take_while(|(a, b)| same_frame_call(a, b)).count();
    let (fresh_start, match_start) = if with_pending == frame.len()
        || with_pending == pending.len()
    {
        (with_pending, 0)
    } else if with_previous == frame.len() || with_previous == previous.len() {
        (with_previous, pending.len().saturating_sub(previous.len()))
    } else if with_pending > 0 || with_previous > 0 {
        return Err(Failure::Fatal("inconsistent cumulative tool-call frames; no calls were run".into()));
    } else {
        let overlap = (1..=previous.len().min(frame.len())).rev()
            .find(|&n| previous[previous.len() - n..].iter()
                .zip(&frame[..n]).all(|(a, b)| same_frame_call(a, b)))
            .unwrap_or(0);
        (overlap, pending.len().saturating_sub(overlap))
    };
    let fresh = &frame[fresh_start..];
    if pending.len().saturating_add(fresh.len()) > MAX_TOOL_CALLS_PER_TURN {
        return Err(Failure::Fatal("too many tool calls in one response".into()));
    }
    // Retain a signature received in a later cumulative frame, even though
    // the executable call was already buffered by an earlier frame.
    let mut signature_updated = false;
    for (index, call) in frame[..fresh_start].iter().enumerate() {
        if let (Some(target), Some(signature)) =
            (pending.get_mut(match_start + index), call.signature.as_ref())
        {
            if target.signature.as_deref() != Some(signature) {
                target.signature = Some(signature.clone());
                signature_updated = true;
            }
        }
    }
    let had_new = !fresh.is_empty();
    for call in fresh {
        let mut call = call.clone();
        call.id = format!("{}:{}", call.name, pending.len());
        pending.push(call);
    }
    *previous = frame;
    Ok(had_new || signature_updated)
}

/// Advance the streaming-text state with one frame's visible text and
/// return the delta to emit, if any.
///
/// Within an answer segment Gemini streams cumulative text, so a frame that
/// extends `segment_text` yields only the new suffix. With interleaved
/// thinking the accumulation resets after a thinking segment: a frame that
/// does NOT extend the current segment is the start of a new one and must
/// be emitted whole — silently dropping it truncates the answer.
fn text_delta(segment_text: &str, full: &str) -> Option<String> {
    if full.is_empty() {
        return None;
    }
    let delta = match full.strip_prefix(segment_text) {
        Some(rest) => rest.to_string(),
        None => full.to_string(),
    };
    if delta.is_empty() {
        None
    } else {
        Some(delta)
    }
}

/// Build one replayed `model`-turn part for a tool call. The model's opaque
/// reasoning signature travels with the call and is echoed back verbatim —
/// Gemini's thinking models reject the whole turn with 400 INVALID_ARGUMENT
/// when it is missing.
fn function_call_part(tc: &ToolCall) -> Value {
    let args: Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
    let mut part = json!({
        "functionCall": {
            "name": tc.name,
            "args": args
        }
    });
    if let Some(signature) = &tc.signature {
        part["thoughtSignature"] = json!(signature);
    }
    part
}

#[cfg(test)]
mod cumulative_call_tests {
    use super::*;

    #[test]
    fn cumulative_frames_do_not_duplicate_an_executable_call() {
        let mut pending = Vec::new();
        let mut previous = Vec::new();
        let a = ToolCall::new("a:0", "a", "{}");
        let b = ToolCall::new("b:1", "b", "{}");
        assert!(append_new_frame_calls(&mut pending, &mut previous, vec![a.clone()]).unwrap());
        assert!(append_new_frame_calls(&mut pending, &mut previous, vec![a.clone().with_signature("sig")]).unwrap());
        assert_eq!(pending[0].signature.as_deref(), Some("sig"));
        assert!(append_new_frame_calls(&mut pending, &mut previous, vec![a, b.clone()]).unwrap());
        assert!(!append_new_frame_calls(&mut pending, &mut previous, vec![b]).unwrap());
        assert_eq!(pending.iter().map(|call| call.name.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(pending[0].id, "a:0");
        assert_eq!(pending[1].id, "b:1");
        // A fresh non-cumulative frame also receives a distinct id.
        let c = ToolCall::new("c:0", "c", "{}");
        assert!(append_new_frame_calls(&mut pending, &mut previous, vec![c]).unwrap());
        assert_eq!(pending[2].id, "c:2");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_call_part_echoes_the_thought_signature() {
        let signed = ToolCall::new("list_files:0", "list_files", "{}").with_signature("sig-bytes");
        let part = function_call_part(&signed);
        assert_eq!(part["thoughtSignature"], json!("sig-bytes"));
        assert_eq!(part["functionCall"]["name"], json!("list_files"));

        // Calls without a signature (other providers, legacy sessions) stay
        // exactly as they always were.
        let unsigned = ToolCall::new("list_files:1", "list_files", "{}");
        let part = function_call_part(&unsigned);
        assert!(part.get("thoughtSignature").is_none());
    }

    #[test]
    fn function_call_part_falls_back_to_empty_args_on_bad_json() {
        let tc = ToolCall::new("x:0", "read_file", r#"{"path": "#).with_signature("s");
        let part = function_call_part(&tc);
        assert_eq!(part["functionCall"]["args"], json!({}));
        assert_eq!(part["thoughtSignature"], json!("s"));
    }

    #[test]
    fn text_delta_streams_cumulative_suffixes() {
        assert_eq!(text_delta("", "Hello").as_deref(), Some("Hello"));
        assert_eq!(
            text_delta("Hello", "Hello world").as_deref(),
            Some(" world")
        );
        // Duplicate frame — nothing new.
        assert_eq!(text_delta("Hello world", "Hello world"), None);
        // Thought-only frame — no visible text at all.
        assert_eq!(text_delta("Hello", ""), None);
    }

    #[test]
    fn text_delta_treats_a_reset_as_a_new_segment() {
        // Interleaved thinking: the cumulative text restarts after a
        // thinking block. The fresh piece must be emitted whole, not
        // dropped — dropping it is what cut answers short.
        assert_eq!(
            text_delta("Sure", "Here is the rest").as_deref(),
            Some("Here is the rest")
        );
    }
}
