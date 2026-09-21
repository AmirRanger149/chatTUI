//! Protocol-neutral request/response vocabulary shared by every chat backend.

/// A tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// Tool definition for CompletionRequest.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Events streamed back from a chat request. Errors end the stream; notices
/// are informational rows (e.g. a model-fallback announcement) that appear in
/// the transcript while the stream keeps going.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// A piece of assistant text.
    Delta(String),
    /// A completed tool call.
    ToolCall(ToolCall),
    /// Progress information, shown as a quiet transcript row.
    Notice(String),
    /// A same-model retry starts after a backoff. The UI shows an animated
    /// countdown while the client waits out `wait_ms`; the next `Delta`,
    /// `ToolCall` or `Error` clears it.
    Retry {
        attempt: u8,
        max: u8,
        wait_ms: u64,
        reason: String,
    },
    /// The request failed for good.
    Error(String),
}

/// Why a chat attempt failed, and what the orchestrator may do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The same model may answer fine if asked again: network failures,
    /// connect problems, idle timeouts, HTTP 408/429/5xx. The orchestrator
    /// retries the *same* model with backoff before considering other
    /// models.
    Transient(String),
    /// This specific model is the problem: bad requests against it, missing
    /// or decommissioned models, "high demand"/overload. Another model may
    /// succeed.
    Retryable(String),
    /// Switching models will not help: rejected credentials, protocol
    /// errors, a stream that broke after output started.
    Fatal(String),
}

/// How many models a single message may try before giving up: the requested
/// model plus up to three fallbacks.
pub const MAX_MODELS_PER_SEND: usize = 4;

/// How many times a single message may retry the *same* model on
/// [`Failure::Transient`] errors (network, timeouts, throttling) before the
/// orchestrator falls back to other models.
pub const MAX_RETRIES_PER_MODEL: usize = 3;

/// A message's speaker. Each backend maps these onto its own wire roles
/// (OpenAI `system/assistant/user`, Anthropic's top-level `system` + content
/// blocks, Gemini `user/model` + `systemInstruction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Assistant,
    User,
    Tool,
}

impl From<&str> for Role {
    fn from(role: &str) -> Self {
        match role {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        }
    }
}

impl Role {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Assistant => "assistant",
            Role::User => "user",
            Role::Tool => "tool",
        }
    }
}

/// One message in a conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn with_tool_call_id(mut self, id: impl Into<String>) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        self.tool_calls = Some(calls);
        self
    }

    /// Helper for tool result messages.
    #[allow(dead_code)]
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }

    /// Helper for assistant message that contains tool calls.
    #[allow(dead_code)]
    pub fn assistant_with_tools(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }
}

/// Everything a backend needs to run one completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub model: String,
    pub temperature: f32,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<String>, // "auto", "none", etc.
}

impl CompletionRequest {
    pub fn new(messages: Vec<Message>, model: String, temperature: f32) -> Self {
        Self {
            messages,
            model,
            temperature,
            tools: Vec::new(),
            tool_choice: None,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        if !self.tools.is_empty() {
            self.tool_choice = Some("auto".to_string());
        }
        self
    }
}
