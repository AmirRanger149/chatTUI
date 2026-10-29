//! Server-Sent Events framing shared by every provider. OpenAI, Anthropic and
//! Gemini all stream their tokens as `data:` lines over SSE, so the
//! incremental line reader lives here instead of in each backend.

/// How many consecutive unparseable SSE `data:` lines a backend tolerates
/// before declaring the stream broken. Gateways occasionally emit junk or
/// keep-alive lines; a single bad line must not kill an otherwise healthy
/// stream.
pub(crate) const MAX_CONSECUTIVE_SSE_PARSE_FAILURES: usize = 5;

/// Extract the payload of an SSE `data:` line. Returns `None` for blank
/// lines, `event:` lines, or anything that isn't valid UTF-8.
pub(crate) fn sse_data(line: &[u8]) -> Option<&str> {
    std::str::from_utf8(line)
        .ok()?
        .trim_end_matches(['\r', '\n'])
        .strip_prefix("data:")
        .map(str::trim)
}

/// Maximum size of one SSE line and of one network chunk to process in a
/// single call. Oversize frames fail explicitly instead of growing a buffer
/// for an attacker-controlled endpoint indefinitely.
const MAX_SSE_LINE_BYTES: usize = 2 * 1024 * 1024;
const MAX_SSE_CHUNK_BYTES: usize = 2 * 1024 * 1024;
const MAX_SSE_EVENTS_PER_CHUNK: usize = 4096;

/// Incremental SSE reader: feed raw bytes as they arrive and drain every
/// complete `data:` payload they produced.
pub(crate) struct SseReader {
    buffer: Vec<u8>,
}

impl SseReader {
    pub(crate) fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<String>, &'static str> {
        if chunk.len() > MAX_SSE_CHUNK_BYTES {
            return Err("SSE chunk exceeded the 2 MiB limit");
        }
        let mut out = Vec::new();
        for part in chunk.split_inclusive(|byte| *byte == b'\n') {
            if self.buffer.len().saturating_add(part.len()) > MAX_SSE_LINE_BYTES {
                return Err("SSE line exceeded the 2 MiB limit");
            }
            self.buffer.extend_from_slice(part);
            if part.ends_with(b"\n") {
                let line = std::mem::take(&mut self.buffer);
                if let Some(data) = sse_data(&line) {
                    out.push(data.to_string());
                    if out.len() > MAX_SSE_EVENTS_PER_CHUNK {
                        return Err("too many SSE events in one chunk");
                    }
                }
            }
        }
        Ok(out)
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
        let mut out = reader.feed(b"event: message\ndata: {\"a\":1}\n\n").unwrap();
        out.extend(reader.feed(b"data: {\"b\":").unwrap());
        out.extend(reader.feed(b"2}\ndata: done\n").unwrap());
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
        assert!(reader.feed(b"data: {\"tail\":1}").unwrap().is_empty());
        assert_eq!(reader.flush(), vec!["{\"tail\":1}".to_string()]);
        // Nothing is left behind, and flushing again is harmless.
        assert!(reader.flush().is_empty());
    }

    #[test]
    fn oversized_unterminated_frame_is_rejected() {
        let mut reader = SseReader::new();
        assert!(reader.feed(&vec![b'x'; MAX_SSE_LINE_BYTES + 1]).is_err());
    }

    #[test]
    fn flush_ignores_non_data_trailing_lines() {
        let mut reader = SseReader::new();
        reader.feed(b"event: ping").unwrap();
        assert!(reader.flush().is_empty());
    }
}
