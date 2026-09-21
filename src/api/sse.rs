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
}
