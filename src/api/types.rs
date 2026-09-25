//! Protocol-neutral request/response vocabulary shared by every chat backend.

/// A tool call requested by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
    /// The provider's opaque reasoning signature for this call, when it
    /// gives one (Gemini's `thoughtSignature` on the functionCall part).
    /// It is the only carrier of the model's private reasoning state and
    /// must be echoed back verbatim when replaying the call, or Gemini's
    /// thinking models reject the turn with 400. Other providers leave it
    /// `None`.
    pub signature: Option<String>,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            signature: None,
        }
    }

    /// Attach the provider's opaque reasoning signature. Treated as a black
    /// box: never parsed, never modified, only round-tripped.
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }
}

/// Tool definition for CompletionRequest.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// What to do with a model's reasoning when history is replayed.
///
/// The two answers are genuinely different, and picking wrong costs quality
/// either way:
///
/// * [`Strip`](ReasoningReplay::Strip) — most OpenAI-compatible endpoints
///   reject or ignore reasoning on input, and sending it back doubles the
///   token bill for text the model will not use.
/// * [`Opaque`](ReasoningReplay::Opaque) — providers whose thinking blocks
///   carry a signature the next turn is validated against. Dropping them
///   during a tool loop silently degrades the answer, or fails the request.
///
/// The payload is deliberately a black box: chatTUI never interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningReplay {
    /// Remove reasoning before replay. The default.
    #[default]
    Strip,
    /// Send it back exactly as it arrived.
    Opaque,
}

impl ReasoningReplay {
    /// Resolve the configured value. `"auto"` defers to the provider kind,
    /// because the right answer is a property of the wire format.
    pub fn resolve(configured: Option<&str>, provider_is_anthropic: bool) -> Self {
        match configured.map(|raw| raw.trim().to_ascii_lowercase()) {
            Some(value) if !value.is_empty() => match value.as_str() {
                "opaque" | "keep" | "round-trip" => ReasoningReplay::Opaque,
                "strip" | "none" | "off" => ReasoningReplay::Strip,
                // An unrecognised value falls back to auto rather than
                // guessing one of the two extremes.
                _ => Self::auto(provider_is_anthropic),
            },
            _ => Self::auto(provider_is_anthropic),
        }
    }

    fn auto(provider_is_anthropic: bool) -> Self {
        if provider_is_anthropic {
            ReasoningReplay::Opaque
        } else {
            ReasoningReplay::Strip
        }
    }
}

/// Token counts reported by the provider for one request.
///
/// Real numbers where the provider sends them, and the only trustworthy input
/// to a "how full is the context window" decision: a `chars / 4` estimate is
/// wrong by 2x often enough to overflow on a long run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Tokens in the request — the whole replayed history plus tools.
    pub input_tokens: u64,
    /// Tokens in the response.
    pub output_tokens: u64,
}

/// Pull a `usage` object out of a provider payload, in whichever shape it
/// arrived. Providers differ in field names and in which event carries them.
pub fn usage_from(value: &serde_json::Value) -> Option<Usage> {
    let usage = &value["usage"];
    let input = usage["prompt_tokens"]
        .as_u64()
        .or_else(|| usage["input_tokens"].as_u64())
        .or_else(|| value["usageMetadata"]["promptTokenCount"].as_u64());
    let output = usage["completion_tokens"]
        .as_u64()
        .or_else(|| usage["output_tokens"].as_u64())
        .or_else(|| value["usageMetadata"]["candidatesTokenCount"].as_u64());
    match (input, output) {
        (Some(input), Some(output)) => Some(Usage {
            input_tokens: input,
            output_tokens: output,
        }),
        // Half a report is still better than an estimate for the half we have.
        (Some(input), None) => Some(Usage {
            input_tokens: input,
            output_tokens: 0,
        }),
        (None, Some(output)) => Some(Usage {
            input_tokens: 0,
            output_tokens: output,
        }),
        (None, None) => None,
    }
}

/// A reasoning payload carried through history so it can be replayed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Reasoning {
    /// The reasoning text, verbatim.
    pub text: String,
    /// The provider's signature over it, when it sends one.
    pub signature: Option<String>,
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
    /// One tool call finished. Emitted by the tool-execution task rather
    /// than produced by the model, so the transcript fills in while a
    /// command is still running instead of after it.
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
    /// Every tool call in the round has reported (or the round ended
    /// early). Carries the round's fingerprint and verdicts so the agent
    /// health guards can stay on the app, where their counters live.
    ToolRoundDone {
        round_key: String,
        any_error: bool,
        all_error: bool,
        /// True when the round stopped to wait for the user rather than
        /// because the tools finished. The health guards must not judge a
        /// round that is only parked, and the round resumes where it left
        /// off once the user answers.
        suspended: bool,
    },
    /// A tool call is waiting for the user's permission. Nothing runs until
    /// [`crate::app::App::resolve_approval`] answers it.
    ApprovalNeeded {
        call_id: String,
        /// One line saying what the call would do, for the prompt.
        summary: String,
        /// Why permission is needed, in the model's and the user's terms.
        reason: String,
        /// True when the sandbox already refused this call and the question
        /// is whether to run it unconfined instead.
        escalated: bool,
        /// For an escalated call: what the sandboxed attempt produced. The
        /// round stopped without recording a result, so this is what the
        /// model gets if the user says no.
        output: Option<String>,
    },
    /// The provider reported token counts for this request.
    Usage(Usage),
    /// The provider signed this turn's reasoning. Stored with the assistant
    /// message so the next request can replay the block it belongs to.
    ReasoningSignature(String),
    /// The model called `ask_user`: it needs a decision only the user can
    /// make. The round parks until [`crate::app::App`] has collected answers.
    QuestionsNeeded {
        call_id: String,
        questions: Vec<Question>,
    },
    /// A same-model retry starts after a backoff. The UI shows an animated
    /// countdown while the client waits out `wait_ms`; the next `Delta`,
    /// `ToolCall` or `Error` clears it.
    Retry {
        attempt: u8,
        max: u8,
        wait_ms: u64,
        reason: String,
    },
    /// The response ended before the model was finished. `output_limit` is
    /// true when the provider reported a budget finish reason
    /// (`length` / `max_tokens`) rather than the connection closing. Kept
    /// separate from [`StreamEvent::Notice`] because the notice is prose for
    /// the user, while a tool call cut off by the output cap needs different
    /// recovery advice from one cut off by a dead connection.
    EndedEarly {
        output_limit: bool,
    },
    /// The request failed for good.
    Error(String),
}

/// One question the model wants the user to answer.
///
/// Fields are defaulted rather than required because this arrives from a
/// model: a question with no id, no header or no options is still a question
/// worth putting in front of the user.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Deserialize)]
pub struct Question {
    #[serde(default)]
    pub id: String,
    /// Short label for the topic ("Auth", "Scope") — a few words at most.
    #[serde(default)]
    pub header: String,
    #[serde(default)]
    pub question: String,
    #[serde(default)]
    pub options: Vec<QuestionOption>,
}

/// One of the answers offered for a [`Question`]. The user can always type
/// something else instead.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Deserialize)]
pub struct QuestionOption {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub description: String,
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
    /// Reasoning to replay alongside this message, when the policy says to
    /// keep it. Opaque to chatTUI: it is stored as it arrived and sent back
    /// as it arrived.
    pub reasoning: Option<Reasoning>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
            reasoning: None,
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

    /// Attach the reasoning payload to replay with this message.
    pub fn with_reasoning(mut self, reasoning: Reasoning) -> Self {
        self.reasoning = Some(reasoning);
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
            reasoning: None,
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
            reasoning: None,
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
