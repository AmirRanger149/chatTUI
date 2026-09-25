//! Running a chat request: spawning the background stream, consuming its
//! token events frame by frame, and reporting elapsed time while it runs.
//! Now with agent loop for tool calling.

use crate::api::types::{
    Message, Question, QuestionOption, Reasoning, ReasoningReplay, Role, StreamEvent, ToolCall,
    ToolDefinition, Usage,
};
use crate::app::{
    App, Approval, Cell, PendingApproval, PendingQuestions, RetryView,
    AGENT_CONSECUTIVE_FAILURE_LIMIT, AGENT_STAGNATION_LIMIT,
};
use crate::sandbox::permissions::Gate;
use crate::sandbox::{strip_denial_marker, Sandbox, SANDBOX_DENIAL_MARKER};
use crate::session::manager::ToolCallRecord;
use crate::tools::{self, MAX_OPTIONS, MAX_QUESTIONS};
use anyhow::Result;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, error::TryRecvError};

impl App {
    pub fn elapsed_secs(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_secs())
            .unwrap_or(0)
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.stream_started
            .map(|started| started.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    pub(crate) fn start_stream(&mut self) -> Result<()> {
        if self.config.api_key.is_none() {
            let env_key = self
                .config
                .find_provider(&self.config.provider)
                .map(|p| p.env_key)
                .unwrap_or_else(|| "API_KEY".to_string());
            return Err(anyhow::anyhow!(
                "{env_key} is not configured — set it in config.json or the environment"
            ));
        }
        let (tx, rx) = mpsc::channel(64);
        self.pending_reasoning_signature = None;
        // Before the request is built, not after it fails: an overflowed
        // request is rejected outright and the work in flight goes with it.
        self.maybe_compact();
        let mut messages: Vec<Message> = Vec::with_capacity(self.sessions.current().messages.len());
        let mut repaired_args = 0usize;
        // One decision for the whole replay, not per message: the wire format
        // is a property of the provider.
        let replay_reasoning = self.reasoning_replay();
        for m in &self.sessions.current().messages {
            let role = Role::from(m.role.as_str());
            let mut msg = Message::new(
                role,
                crate::ui::thinking::strip(&m.content),
            );
            // For providers that sign their reasoning, the reasoning goes back
            // with the message it came from. Dropping it mid-tool-loop is what
            // silently degrades an answer: the model loses the chain of
            // thought it is supposed to continue from.
            if replay_reasoning == ReasoningReplay::Opaque && role == Role::Assistant {
                let text = crate::ui::thinking::reasoning_text(&m.content).unwrap_or_default();
                if !text.is_empty() || m.reasoning_signature.is_some() {
                    msg = msg.with_reasoning(Reasoning {
                        text,
                        signature: m.reasoning_signature.clone(),
                    });
                }
            }
            if let Some(id) = &m.tool_call_id {
                msg = msg.with_tool_call_id(id.clone());
            }
            if let Some(tcs) = &m.tool_calls {
                // History integrity: tool-call arguments must be valid JSON
                // before they are replayed. A truncated call (a stream cut
                // mid-arguments) would otherwise poison this request and
                // every later one in the session.
                let tool_calls: Vec<crate::api::types::ToolCall> = tcs
                    .iter()
                    .map(|tc| {
                        let arguments = if serde_json::from_str::<serde_json::Value>(&tc.arguments).is_ok() {
                            tc.arguments.clone()
                        } else {
                            repaired_args += 1;
                            "{}".to_string()
                        };
                        let mut call = crate::api::types::ToolCall::new(
                            tc.id.clone(),
                            tc.name.clone(),
                            arguments,
                        );
                        // Provider reasoning signatures (Gemini) must survive
                        // replay verbatim, or the turn is rejected with 400.
                        if let Some(signature) = tc.signature.clone() {
                            call = call.with_signature(signature);
                        }
                        call
                    })
                    .collect();
                msg = msg.with_tool_calls(tool_calls);
            }
            messages.push(msg);
        }

        // History integrity: every assistant tool call needs a paired tool
        // result, or OpenAI-compatible APIs reject the whole conversation.
        // Backfill anything an interrupted round left unanswered.
        let answered: HashSet<String> = messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .filter_map(|m| m.tool_call_id.clone())
            .collect();
        let mut backfilled = 0usize;
        let mut paired: Vec<Message> = Vec::with_capacity(messages.len());
        for message in messages {
            let missing: Vec<String> = if message.role == Role::Assistant {
                message
                    .tool_calls
                    .as_ref()
                    .map(|calls| {
                        calls
                            .iter()
                            .map(|tc| tc.id.clone())
                            .filter(|id| !answered.contains(id))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            paired.push(message);
            for id in missing {
                paired.push(Message::tool_result(id, "interrupted before this tool could run"));
                backfilled += 1;
            }
        }
        let mut messages = paired;
        if repaired_args > 0 {
            self.push_notice(format!(
                "repaired {repaired_args} truncated tool call(s) in history before replay"
            ));
        }
        if backfilled > 0 {
            self.push_notice(format!(
                "backfilled {backfilled} missing tool result(s) from an interrupted round"
            ));
        }

        // Inject system prompt for agent mode if enabled
        let want_tools = self.agent_mode && self.config.sandbox.enabled && self.sandbox.config.enabled;
        let tools_enabled = want_tools && self.sandbox.has_target();
        if want_tools && !self.sandbox.has_target() {
            self.push_notice(
                "agent tools off — set a target with /sandbox <path> (tools will not use this app's directory)".into(),
            );
        }
        if tools_enabled {
            let has_system = messages.iter().any(|m| m.role == Role::System);
            if !has_system {
                // Built from the policy the program actually enforces, plus
                // any instructions the project itself carries — see
                // `crate::instructions`. Nothing here is typed twice: change
                // the timeout or the isolation mode in config.json and the
                // model is told the new value, not the old prose.
                let system_content = crate::instructions::agent_system_prompt(&self.sandbox.config);
                messages.insert(0, Message::new(Role::System, system_content));
            }
        }
        let model = self.config.model.clone();
        let temperature = self.config.temperature;
        let client = self.config.api_client();

        // Prepare tools if agent mode enabled (reuse tools_enabled from above)
        let tool_defs: Vec<ToolDefinition> = if tools_enabled {
            tools::all_tools()
                .into_iter()
                .map(|t| ToolDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect()
        } else {
            Vec::new()
        };

        let task = tokio::spawn(async move {
            if let Err(error) = client
                .stream_chat(&messages, &model, temperature, tool_defs, tx.clone())
                .await
            {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        self.stream_task = Some(task);
        self.tokens = Some(rx);
        self.streaming = true;
        self.output_limit_hit = false;
        self.stream_started = Some(Instant::now());
        Ok(())
    }

    pub async fn receive_token(&mut self) {
        let Some(mut rx) = self.tokens.take() else {
            return;
        };
        let mut has_tool_calls = false;
        loop {
            match rx.try_recv() {
                Ok(StreamEvent::Delta(token)) => {
                    // Tokens are flowing again: the retry countdown is over.
                    self.retry_state = None;
                    self.response.push_str(&token);
                }
                Ok(StreamEvent::ToolCall(tc)) => {
                    self.retry_state = None;
                    // Detect truncated tool calls at ingestion: arguments
                    // that do not parse were cut mid-stream. Record the id
                    // so execution returns an explicit error and the stored
                    // record keeps replayable JSON.
                    if serde_json::from_str::<serde_json::Value>(&tc.arguments).is_err() {
                        self.truncated_tool_calls.insert(tc.id.clone());
                    }
                    self.pending_tool_calls.push(tc);
                    has_tool_calls = true;
                }
                Ok(StreamEvent::Usage(usage)) => {
                    // Anthropic reports input on message_start and output on
                    // message_delta, so merge rather than overwrite.
                    let merged = match self.last_usage {
                        Some(previous) => Usage {
                            input_tokens: previous.input_tokens.max(usage.input_tokens),
                            output_tokens: previous.output_tokens.max(usage.output_tokens),
                        },
                        None => usage,
                    };
                    self.last_usage = Some(merged);
                }
                Ok(StreamEvent::ReasoningSignature(signature)) => {
                    self.pending_reasoning_signature = Some(signature);
                }
                Ok(StreamEvent::Notice(message)) => {
                    self.push_notice(message);
                }
                Ok(StreamEvent::EndedEarly { output_limit }) => {
                    // Recorded, not shown: the matching `Notice` already told
                    // the user. This is for the tool-call error below, which
                    // has to explain a truncated call differently depending on
                    // whether the provider hit its output cap or hung up.
                    self.output_limit_hit = output_limit;
                }
                Ok(StreamEvent::ToolResult {
                    id,
                    content,
                    is_error,
                }) => {
                    self.answered_tool_ids.insert(id.clone());
                    self.sessions
                        .add_tool_result(id.clone(), content.clone(), is_error);
                    self.push_tool_result(id, content, is_error);
                }
                Ok(StreamEvent::ApprovalNeeded {
                    call_id,
                    summary,
                    reason,
                    escalated,
                    output,
                }) => {
                    // Park here. The tool task stops itself right after this
                    // event, and the round resumes — skipping the calls that
                    // already ran — once the user answers.
                    self.awaiting_approval = Some(PendingApproval {
                        call_id,
                        summary,
                        reason,
                        escalated,
                        output,
                    });
                }
                Ok(StreamEvent::QuestionsNeeded { call_id, questions }) => {
                    // One answer slot per question; they fill in as the user
                    // goes, and the tool result is built from them at the end.
                    let slots = questions.len();
                    self.awaiting_questions = Some(PendingQuestions {
                        call_id,
                        questions,
                        current: 0,
                        answers: vec![None; slots],
                    });
                }
                Ok(StreamEvent::ToolRoundDone {
                    round_key,
                    any_error,
                    all_error,
                    suspended,
                }) => {
                    self.finish_tool_round(round_key, any_error, all_error, suspended);
                    return;
                }
                Ok(StreamEvent::Retry {
                    attempt,
                    max,
                    wait_ms,
                    reason,
                }) => {
                    // Drive the animated status row: the countdown is
                    // computed from `started` on every redraw.
                    self.retry_state = Some(RetryView {
                        started: Instant::now(),
                        wait: Duration::from_millis(wait_ms),
                        attempt,
                        max,
                        reason,
                    });
                }
                Ok(StreamEvent::Error(error)) => {
                    self.retry_state = None;
                    self.tool_task = None;
                    self.finish_partial();
                    self.streaming = false;
                    self.stream_started = None;
                    self.pending_tool_calls.clear();
                    self.reset_agent_health();
                    self.push_error(error);
                    return;
                }
                Err(TryRecvError::Empty) => {
                    self.tokens = Some(rx);
                    return;
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }

        // The stream ended cleanly; drop any leftover retry indicator.
        self.retry_state = None;

        // Stream finished - check for agent loop
        if self.streaming && has_tool_calls && !self.pending_tool_calls.is_empty() {
            // Save assistant message with tool calls
            let response_text = std::mem::take(&mut self.response);
            let pending = std::mem::take(&mut self.pending_tool_calls);
            let tool_records: Vec<ToolCallRecord> = pending
                .iter()
                .map(|tc| ToolCallRecord {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    // Truncated arguments are stored as "{}" so the saved
                    // history never carries unparseable JSON.
                    arguments: if self.truncated_tool_calls.contains(&tc.id) {
                        "{}".to_string()
                    } else {
                        tc.arguments.clone()
                    },
                    signature: tc.signature.clone(),
                })
                .collect();

            if !response_text.is_empty() {
                self.cells.push(Cell::Assistant(response_text.clone()));
            }
            for tc in &pending {
                self.push_tool_call(tc);
            }
            self.sessions.add_assistant_with_tools(response_text, tool_records);
            self.attach_reasoning_signature();

            // Execute the round's tools on their own task.
            //
            // Awaiting them here — as this used to — blocks the draw loop
            // for as long as the slowest command runs, up to the shell
            // timeout: a 30s build froze the entire UI and `Esc` could not
            // reach it. Results now come back over a fresh channel as they
            // land, so the transcript fills in live, the status row keeps
            // counting, and an interrupt stops the round.
            let sandbox = self.sandbox.clone();
            // Only this round's truncation verdicts travel with it; the app
            // keeps its own set until the round reports back.
            let truncated: HashSet<String> = self
                .truncated_tool_calls
                .iter()
                .filter(|id| pending.iter().any(|tc| &tc.id == *id))
                .cloned()
                .collect();
            let (tx, rx) = mpsc::channel(16);
            self.active_round_calls = pending.clone();
            self.answered_tool_ids.clear();
            let output_limit = self.output_limit_hit;
            self.tool_task = Some(tokio::spawn(run_tool_round(
                sandbox,
                pending,
                truncated,
                output_limit,
                HashSet::new(),
                None,
                None,
                tx,
            )));
            self.tokens = Some(rx);
            // `streaming` stays true: from the user's point of view the
            // agent is still working, and the elapsed clock keeps running.
            return;
        }

        self.streaming = false;
        self.stream_started = None;
        self.tool_task = None;
        self.finish_partial();
        self.reset_agent_health();
        self.pending_tool_calls.clear();
    }

    /// The round's tools have all reported back: decide whether the agent
    /// may keep going.
    ///
    /// Progress, not a step counter, decides — a healthy run continues for
    /// as long as it is making progress. These guards stop the specific
    /// broken patterns instead: the same failing call coming round after
    /// round, every call failing, and a generous backstop ceiling.
    fn finish_tool_round(
        &mut self,
        round_key: String,
        any_error: bool,
        all_error: bool,
        suspended: bool,
    ) {
        // Parked waiting for the user, not finished. Keep the round's
        // bookkeeping — it says which calls already ran — and judge nothing:
        // a round that stopped to ask is not a round that failed.
        if suspended {
            self.tool_task = None;
            return;
        }

        // The round is over: drop its bookkeeping and the truncation
        // verdicts it consumed.
        let round_ids: Vec<String> = self
            .active_round_calls
            .iter()
            .map(|call| call.id.clone())
            .collect();
        self.active_round_calls.clear();
        self.answered_tool_ids.clear();
        for id in round_ids {
            self.truncated_tool_calls.remove(&id);
        }

        // Stagnation: the same failing round fingerprint coming back.
        if any_error && self.agent_last_round_key.as_deref() == Some(round_key.as_str()) {
            self.agent_repeat_failures += 1;
        } else {
            self.agent_repeat_failures = usize::from(any_error);
        }
        self.agent_last_round_key = Some(round_key);

        // Consecutive rounds in which every tool errored.
        if all_error {
            self.agent_consecutive_failures += 1;
        } else {
            self.agent_consecutive_failures = 0;
        }

        let stop_reason: Option<String> =
            if self.agent_repeat_failures >= AGENT_STAGNATION_LIMIT {
                Some(format!(
                    "agent stopped: the same tool call failed {AGENT_STAGNATION_LIMIT} rounds in a row — it is not making progress"
                ))
            } else if self.agent_consecutive_failures >= AGENT_CONSECUTIVE_FAILURE_LIMIT {
                Some(format!(
                    "agent stopped: {AGENT_CONSECUTIVE_FAILURE_LIMIT} consecutive rounds failed — check the workspace and try again"
                ))
            } else if self.agent_iterations >= self.agent_max_rounds {
                // Safety net only — healthy runs are stopped by the guards
                // above, not by this ceiling.
                Some(format!(
                    "agent stopped after {} rounds — raise agent.max_rounds in config.json if this task needs more",
                    self.agent_max_rounds
                ))
            } else {
                None
            };

        if let Some(reason) = stop_reason {
            self.push_notice(reason);
            self.streaming = false;
            self.stream_started = None;
            self.tool_task = None;
            self.finish_partial();
            self.reset_agent_health();
            return;
        }

        self.agent_iterations += 1;
        if let Err(error) = self.start_stream() {
            self.push_error(error.to_string());
            self.streaming = false;
            self.stream_started = None;
            self.tool_task = None;
            self.reset_agent_health();
        }
    }

    /// Answer the pending approval and let the round carry on.
    ///
    /// A denial is reported back as a tool result rather than swallowed:
    /// without it the model has no way to know the call was refused and
    /// simply tries again.
    pub fn resolve_approval(&mut self, decision: Approval) {
        let pending = match self.awaiting_approval.take() {
            Some(pending) => pending,
            None => return,
        };
        let call = match self
            .active_round_calls
            .iter()
            .find(|call| call.id == pending.call_id)
        {
            Some(call) => call.clone(),
            // Its round is already gone — interrupted, or an error closed
            // the stream. Nothing left to resume.
            None => return,
        };
        let key = self.sandbox.allowance_key_for(&call);
        let allow_once = match decision {
            Approval::Always => {
                self.sandbox.allow(key);
                None
            }
            Approval::Once => Some(key),
            Approval::Deny => None,
        };

        let verb = match decision {
            Approval::Once => "approved once",
            Approval::Always => "approved for this session",
            Approval::Deny => "denied",
        };
        self.push_notice(format!("{verb}: {}", pending.summary));

        if decision == Approval::Deny {
            // An escalated call already has output; without it the model
            // would only learn "denied" and not what the sandbox objected to.
            let content = match &pending.output {
                Some(output) => format!(
                    "Error: the user declined to run this outside the sandbox. The sandboxed attempt said: {output}\n\nWork inside the workspace without network access, or ask the user to change the sandbox settings — do not retry the same command."
                ),
                None => format!(
                    "Error: the user denied this call ({}). Do not retry it — ask what they would like instead.",
                    pending.reason
                ),
            };
            self.sessions
                .add_tool_result(call.id.clone(), content.clone(), true);
            self.push_tool_result(call.id.clone(), content, true);
            self.answered_tool_ids.insert(call.id.clone());
        }

        // Approving an escalation means "run that call again, unconfined".
        let escalate = match decision {
            Approval::Deny => None,
            _ if pending.escalated => Some(call.id.clone()),
            _ => None,
        };
        self.resume_round(allow_once, escalate);
    }

    /// Record an answer to the question on screen and move to the next one.
    ///
    /// Empty answers are ignored rather than recorded: pressing enter on an
    /// empty composer should not silently answer "nothing" to a question the
    /// model is blocked on.
    pub fn answer_current_question(&mut self, answer: String) {
        let answer = answer.trim();
        if answer.is_empty() {
            return;
        }
        let finished = match self.awaiting_questions.as_mut() {
            Some(pending) => {
                if let Some(slot) = pending.answers.get_mut(pending.current) {
                    *slot = Some(answer.to_string());
                }
                pending.current += 1;
                pending.current >= pending.questions.len()
            }
            None => return,
        };
        if finished {
            self.finish_questions();
        }
    }

    /// Stop asking: answer nothing more and let the round carry on.
    ///
    /// The model still gets a tool result, so the transcript stays valid and
    /// it learns the questions went unanswered instead of hanging.
    pub fn skip_questions(&mut self) {
        if self.awaiting_questions.is_none() {
            return;
        }
        self.finish_questions();
    }

    /// Build the `ask_user` result from the answers collected and resume.
    fn finish_questions(&mut self) {
        let pending = match self.awaiting_questions.take() {
            Some(pending) => pending,
            None => return,
        };
        let skipped = pending.answers.iter().all(|answer| answer.is_none());
        let mut lines: Vec<String> = Vec::new();
        for (index, question) in pending.questions.iter().enumerate() {
            let answer = pending
                .answers
                .get(index)
                .and_then(|answer| answer.as_deref())
                .unwrap_or("(the user did not answer this one)");
            let topic = if question.header.is_empty() {
                question.question.clone()
            } else {
                format!("{} — {}", question.header, question.question)
            };
            lines.push(format!("{}. {topic}\n   answer: {answer}", index + 1));
        }
        let content = if skipped {
            format!(
                "The user dismissed these questions without answering. Pick the most conservative option yourself and say which one you chose:\n{}",
                lines.join("\n")
            )
        } else {
            format!("The user answered:\n{}", lines.join("\n"))
        };
        self.sessions
            .add_tool_result(pending.call_id.clone(), content.clone(), false);
        self.push_tool_result(pending.call_id.clone(), content, false);
        self.answered_tool_ids.insert(pending.call_id.clone());
        self.resume_round(None, None);
    }

    /// Restart a round that stopped for an approval, from where it stopped.
    ///
    /// Calls that already reported are skipped, so saying yes to one call
    /// never re-runs the ones the user has already seen.
    fn resume_round(&mut self, allow_once: Option<String>, escalate: Option<String>) {
        if self.active_round_calls.is_empty() {
            self.streaming = false;
            self.stream_started = None;
            self.tool_task = None;
            self.finish_partial();
            self.reset_agent_health();
            return;
        }
        let calls = self.active_round_calls.clone();
        let skip: HashSet<String> = self.answered_tool_ids.clone();
        let truncated: HashSet<String> = self
            .truncated_tool_calls
            .iter()
            .filter(|id| calls.iter().any(|call| &call.id == *id))
            .cloned()
            .collect();
        let sandbox = self.sandbox.clone();
        let (tx, rx) = mpsc::channel(16);
        let output_limit = self.output_limit_hit;
        self.tool_task = Some(tokio::spawn(run_tool_round(
            sandbox,
            calls,
            truncated,
            output_limit,
            skip,
            allow_once,
            escalate,
            tx,
        )));
        self.tokens = Some(rx);
        self.streaming = true;
    }
}

/// Run every tool call of one round, reporting each result as it lands.
///
/// This is the work that used to block the UI thread. Each call is checked
/// against the permission layer *before* it runs, so a call the user has to
/// approve stops the round instead of starting and being refused: the round
/// reports itself `suspended` and is resumed by [`App::resolve_approval`]
/// with the calls that already ran in `skip`. Every other failure — timeout,
/// path escape, permission denial, truncated arguments — becomes a tool error
/// the model can read and react to rather than an abort.
///
/// The round fingerprint and verdicts are computed here, where the calls are,
/// and sent back at the end: the health guards own state on `App`, so the
/// decision stays there.
async fn run_tool_round(
    sandbox: Sandbox,
    calls: Vec<ToolCall>,
    truncated: HashSet<String>,
    // True when this turn's stream stopped on the provider's output cap.
    // Changes the wording of a truncation error, nothing else.
    output_limit: bool,
    // Calls that already reported during an earlier attempt at this round.
    // Empty the first time through; after an approval it keeps a resumed
    // round from re-running work the user has already seen.
    skip: HashSet<String>,
    // An approval granted for this attempt only.
    allow_once: Option<String>,
    // The id of a call the user agreed to re-run outside the kernel sandbox.
    escalate: Option<String>,
    tx: mpsc::Sender<StreamEvent>,
) {
    // Retire sessions that outlived their lifetime. Checked here rather than
    // on a timer: this is the only moment a session's owner is listening.
    for killed in sandbox.sessions.reap_expired() {
        let _ = tx.send(StreamEvent::Notice(killed)).await;
    }

    let mut any_error = false;
    let mut all_error = true;
    let mut executed = 0usize;

    for call in &calls {
        if skip.contains(&call.id) {
            continue;
        }
        // Computed once: a truncated call must not be approved, must not be
        // asked about, and must not be executed.
        let truncated_args = truncated.contains(&call.id);

        // `ask_user` is answered by the human, not executed. Park the round
        // and hand the questions to the UI; the tool result is written by the
        // app once every question has an answer.
        if call.name == "ask_user" && !truncated_args {
            let questions = parse_questions(&call.arguments);
            if tx
                .send(StreamEvent::QuestionsNeeded {
                    call_id: call.id.clone(),
                    questions,
                })
                .await
                .is_err()
            {
                return;
            }
            let _ = tx
                .send(StreamEvent::ToolRoundDone {
                    round_key: round_fingerprint(&calls),
                    any_error,
                    all_error: false,
                    suspended: true,
                })
                .await;
            return;
        }
        // Ask the permission layer *before* running anything: a call the
        // user has to approve must never start and be denied afterwards.
        // The round stops at the first such call and resumes from there.
        // A call whose arguments were cut mid-stream can never run, so there
        // is nothing to approve. Gating it anyway is what produced
        // "approved: write_file: (no path)" followed by a truncation error —
        // the user was asked about a call that did not exist.
        if !truncated_args {
            let verdict = sandbox.gate_call(call, allow_once.as_deref());
            if let Gate::Ask { reason } = &verdict.gate {
                let reason = reason.clone();
                let summary = verdict.summary.clone();
                if tx
                    .send(StreamEvent::ApprovalNeeded {
                        call_id: call.id.clone(),
                        summary,
                        reason,
                        escalated: false,
                        // Nothing ran, so there is no output to carry.
                        output: None,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
                let _ = tx
                    .send(StreamEvent::ToolRoundDone {
                        round_key: round_fingerprint(&calls),
                        any_error,
                        // Not a failed round — a paused one. The verdicts are
                        // ignored while suspended, but "every tool failed"
                        // would be a lie either way.
                        all_error: false,
                        suspended: true,
                    })
                    .await;
                return;
            }
        }
        // Never silently fall back to empty arguments: unparseable JSON
        // almost always means the call was cut mid-stream, and the model
        // deserves an error that says so and how to recover.
        let outcome = if truncated_args {
            // Two causes, two recoveries. Sending the model the wrong one
            // just makes it repeat a call that will be cut off again.
            let cause = if output_limit {
                "the response hit its output token limit before the tool call's JSON was complete"
            } else {
                "the stream was cut mid-call"
            };
            Err(anyhow::anyhow!(
                "tool call arguments arrived truncated: {cause}. Do NOT repeat this call unchanged — use apply_patch with small hunks, or split the change into several smaller calls, instead of sending one whole file in a single write_file. (If large writes keep getting cut, raising max_output_tokens in config.json gives the model more room per response.)"
            ))
        } else if escalate.as_deref() == Some(call.id.as_str()) {
            // The user agreed to this one outside the kernel sandbox. The
            // workspace restriction and the sensitive-file policy still
            // apply; only Landlock/seccomp are lifted.
            sandbox
                .without_isolation()
                .execute_tool(call, allow_once.as_deref())
                .await
        } else {
            sandbox.execute_tool(call, allow_once.as_deref()).await
        };

        // The kernel sandbox stopped this one. That is a question for the
        // user, not an error to hand back: left alone the model retries the
        // same command until the stagnation guard kills the run.
        if let Ok(output) = &outcome {
            if output.contains(SANDBOX_DENIAL_MARKER) {
                let shown = strip_denial_marker(output);
                if tx
                    .send(StreamEvent::ApprovalNeeded {
                        call_id: call.id.clone(),
                        summary: format!(
                            "{} — the sandbox blocked it",
                            sandbox.gate_call(call, None).summary
                        ),
                        reason: "the kernel sandbox refused this command".to_string(),
                        escalated: true,
                        output: Some(shown),
                    })
                    .await
                    .is_err()
                {
                    return;
                }
                let _ = tx
                    .send(StreamEvent::ToolRoundDone {
                        round_key: round_fingerprint(&calls),
                        any_error,
                        all_error: false,
                        suspended: true,
                    })
                    .await;
                return;
            }
        }

        let (content, is_error) = match outcome {
            Ok(output) => (output, false),
            Err(error) => (format!("Error: {error}"), true),
        };
        executed += 1;
        any_error |= is_error;
        all_error &= is_error;
        if tx
            .send(StreamEvent::ToolResult {
                id: call.id.clone(),
                content,
                is_error,
            })
            .await
            .is_err()
        {
            // The receiver is gone: the user interrupted. Whatever is
            // already running finishes under its own timeout; the app
            // backfills a result for the calls that never started.
            return;
        }
    }

    // Nothing ran (an empty round, or an immediate cancel) is not "every
    // tool failed" — that verdict would trip the failure guard on a round
    // that never happened.
    if executed == 0 {
        all_error = false;
    }

    let _ = tx
        .send(StreamEvent::ToolRoundDone {
            round_key: round_fingerprint(&calls),
            any_error,
            all_error,
            suspended: false,
        })
        .await;
}

/// Parse the model's questions, tolerating the shapes models actually send.
///
/// A question with no options is fine — the user answers free-form. Missing
/// ids and headers are filled in. Anything that cannot be read at all becomes
/// one open question rather than an error: a model that asked for help and got
/// a JSON complaint back will not ask again.
fn parse_questions(arguments: &str) -> Vec<Question> {
    let parsed: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let mut questions: Vec<Question> = Vec::new();
    for (index, item) in parsed["questions"]
        .as_array()
        .map(|items| items.as_slice())
        .unwrap_or(&[])
        .iter()
        .enumerate()
        .take(MAX_QUESTIONS)
    {
        let text = item["question"].as_str().unwrap_or("").trim().to_string();
        if text.is_empty() {
            continue;
        }
        let mut options: Vec<QuestionOption> = Vec::new();
        for option in item["options"]
            .as_array()
            .map(|items| items.as_slice())
            .unwrap_or(&[])
            .iter()
            .take(MAX_OPTIONS)
        {
            let label = option["label"].as_str().unwrap_or("").trim().to_string();
            if label.is_empty() {
                continue;
            }
            options.push(QuestionOption {
                label,
                description: option["description"].as_str().unwrap_or("").trim().to_string(),
            });
        }
        let id = item["id"].as_str().unwrap_or("").trim();
        questions.push(Question {
            id: if id.is_empty() {
                format!("q{}", index + 1)
            } else {
                id.to_string()
            },
            header: item["header"].as_str().unwrap_or("").trim().to_string(),
            question: text,
            options,
        });
    }
    if questions.is_empty() {
        // Whatever the model meant, the user can still say something useful.
        questions.push(Question {
            id: "q1".to_string(),
            header: String::new(),
            question: "The agent asked for input, but its question could not be read. What should it do?"
                .to_string(),
            options: Vec::new(),
        });
    }
    questions
}

/// Fingerprint a round so the stagnation guard can recognise the same
/// failing calls coming back.
fn round_fingerprint(calls: &[ToolCall]) -> String {
    let mut keys: Vec<String> = calls
        .iter()
        .map(|call| format!("{}\u{0}{}", call.name, call.arguments))
        .collect();
    keys.sort();
    keys.join("\u{1f}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ToolCall;
    use crate::app::{App, Cell};
    use crate::config::Config;
    use crate::sandbox::permissions::PermissionMode;
    use crate::session::manager::SessionManager;

    fn test_app() -> App {
        App::new(Config::default(), SessionManager::for_tests())
    }

    /// An app with agent tools enabled against an empty temp workspace and
    /// no API key: a tool round that tries to continue the agent loop fails
    /// its restart immediately instead of touching the network.
    fn agent_app(name: &str) -> App {
        let config: Config = serde_json::from_str("{}").unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir = std::env::temp_dir().join(format!("chatTUI_loop_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;
        app
    }

    fn tool_call_event(id: &str, name: &str, arguments: &str) -> StreamEvent {
        StreamEvent::ToolCall(ToolCall::new(id, name, arguments))
    }

    /// Run a tool round to completion: let the tool task have turns on the
    /// runtime, draining its events as they arrive, until the round has been
    /// processed.
    ///
    /// Tools execute on their own task now, so one `receive_token` no longer
    /// completes a round — the task has to be scheduled before its results
    /// exist. `tool_task` is cleared when the round is processed, which is
    /// what this waits for.
    async fn drain_tools(app: &mut App) {
        for _ in 0..500 {
            app.receive_token().await;
            if app.tool_task.is_none() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        panic!("tool round did not settle");
    }

    /// Feed one completed stream containing `events` into the app.
    fn feed(app: &mut App, events: Vec<StreamEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        for event in events {
            tx.try_send(event).unwrap();
        }
        drop(tx);
        app.streaming = true;
        app.tokens = Some(rx);
    }

    #[test]
    fn stream_notices_become_transcript_rows() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tx.try_send(StreamEvent::Notice("switching to b/2".into()))
            .unwrap();
        tx.try_send(StreamEvent::Delta("hel".into())).unwrap();
        let mut app = test_app();
        app.streaming = true;
        app.tokens = Some(rx);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(app.receive_token());
        assert_eq!(app.response, "hel");
        assert_eq!(app.cells.len(), 1);
        assert!(matches!(app.cells.last(), Some(Cell::Notice(_))));

        // Dropping the sender ends the stream and commits the partial answer.
        drop(tx);
        runtime.block_on(app.receive_token());
        assert!(!app.streaming);
        assert_eq!(app.cells.len(), 2);
        assert!(matches!(app.cells[0], Cell::Notice(_)));
        assert_eq!(app.cells[1], Cell::Assistant("hel".into()));
    }

    #[tokio::test]
    async fn a_running_tool_does_not_block_the_ui_thread() {
        let mut app = agent_app("nonblocking");
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);

        // One pass over the stream hands the round to a task and returns.
        // This is the whole point of the change: the old code awaited the
        // tool here, so a 30s command froze the draw loop and `Esc` could
        // not reach it.
        app.receive_token().await;
        assert!(app.streaming, "the agent is still working");
        assert!(
            app.tool_task.is_some(),
            "the round should be running on its own task"
        );
        assert!(
            !app.cells.iter().any(|c| matches!(c, Cell::ToolResult { .. })),
            "no result may exist yet — nothing has awaited the tool"
        );

        // And the round still completes, with its result in the transcript.
        drain_tools(&mut app).await;
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: false, .. }
        )));
    }

    #[tokio::test]
    async fn interrupting_a_round_backfills_a_result_for_every_call() {
        let mut app = agent_app("interrupt-round");
        feed(
            &mut app,
            vec![
                tool_call_event("c1", "list_files", "{}"),
                tool_call_event("c2", "list_files", "{}"),
            ],
        );
        app.receive_token().await;
        // Cancel before the round reports back.
        app.interrupt();

        assert!(!app.streaming);
        // Every tool call needs a result, or the provider rejects the whole
        // conversation on the next request.
        let results: Vec<&crate::session::manager::Message> = app
            .sessions
            .current()
            .messages
            .iter()
            .filter(|message| message.role == "tool")
            .collect();
        assert_eq!(results.len(), 2, "each call needs exactly one result");
        assert!(
            results.iter().all(|m| m.is_error == Some(true)),
            "an interrupted call is not a success"
        );
        assert!(
            results.iter().all(|m| m.content.contains("interrupted")),
            "the reason must be recorded, not left blank"
        );
    }

    #[tokio::test]
    async fn tool_rounds_execute_and_stop_predictably_when_restart_fails() {
        let mut app = agent_app("restart-fail");
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        drain_tools(&mut app).await;

        // The tool actually ran.
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: false, .. }
        )));
        // The loop tried to continue, the restart failed (no API key), and
        // the agent stopped with a reported error instead of looping.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(matches!(app.cells.last(), Some(Cell::Error(_))));
    }

    #[tokio::test]
    async fn agent_loop_stops_at_the_configured_ceiling() {
        let mut app = agent_app("ceiling");
        app.agent_iterations = app.agent_max_rounds;
        feed(&mut app, vec![tool_call_event("c1", "list_files", "{}")]);
        drain_tools(&mut app).await;

        // The final round's tool still ran and its result was recorded…
        assert!(app
            .cells
            .iter()
            .any(|c| matches!(c, Cell::ToolResult { .. })));
        // …but no further LLM request was started and the agent stopped.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped") && text.contains("max_rounds")
        )));
    }

    #[tokio::test]
    async fn successful_rounds_run_until_the_ceiling() {
        // Restart streams succeed (the legacy `api_key` field satisfies
        // start_stream; the endpoint is a closed local port that fails
        // fast), so the agent loop really runs round after round — exactly
        // the setup that would spin forever without a backstop. Identical
        // SUCCEEDING rounds must not trip the stagnation guard.
        let config: Config = serde_json::from_str(
            r#"{"provider":"openai","base_url":"http://127.0.0.1:9/v1","api_key":"test-key","agent":{"max_rounds":12}}"#,
        )
        .unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir =
            std::env::temp_dir().join(format!("chatTUI_loop_many_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;

        let mut rounds = 0;
        loop {
            // The round counter may reach but never exceed the ceiling.
            assert!(
                app.agent_iterations <= app.agent_max_rounds,
                "round counter exceeded its ceiling"
            );
            feed(
                &mut app,
                vec![tool_call_event(&format!("c{rounds}"), "list_files", "{}")],
            );
            drain_tools(&mut app).await;
            rounds += 1;
            assert!(rounds < 100, "agent loop failed to terminate");
            if !app.streaming {
                break;
            }
        }

        assert_eq!(rounds, app.agent_max_rounds + 1);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("agent stopped")
        )));
        assert_eq!(app.agent_iterations, 0);
    }

    #[tokio::test]
    async fn tool_errors_cannot_bypass_the_iteration_limit() {
        // Every round fails (permission denied); the counter must still
        // advance and the loop must still terminate.
        let mut app = agent_app("errors-bounded");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        let rounds = 3;
        for round in 0..rounds {
            assert!(app.agent_iterations < app.agent_max_rounds);
            feed(
                &mut app,
                vec![tool_call_event(&format!("c{round}"), "write_file", r#"{"path":"x","content":"y"}"#)],
            );
            drain_tools(&mut app).await;
            assert!(!app.streaming);
            assert!(app.cells.iter().any(|c| matches!(
                c,
                Cell::ToolResult { is_error: true, content, .. }
                    if content.contains("permission denied")
            )));
        }
        assert!(!app.streaming);
    }

    #[tokio::test]
    async fn permission_denials_reach_the_model_as_tool_errors() {
        let mut app = agent_app("permission");
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        feed(
            &mut app,
            vec![tool_call_event("c1", "bash", r#"{"command":"echo hi"}"#)],
        );
        drain_tools(&mut app).await;
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. }
                if content.contains("permission denied")
        )));
    }

    #[tokio::test]
    async fn truncated_tool_arguments_become_an_explicit_tool_error() {
        let mut app = agent_app("truncated");
        // Unterminated JSON — a tool call whose stream was cut mid-arguments.
        feed(
            &mut app,
            vec![StreamEvent::ToolCall(ToolCall::new(
                "c1",
                "edit_file",
                r#"{"new_string": "#,
            ))],
        );
        drain_tools(&mut app).await;

        // The model sees an error that names the real problem…
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::ToolResult { is_error: true, content, .. } if content.contains("truncated")
        )));
        // …and the stored record carries clean JSON, never the broken text.
        assert!(app.sessions.current().messages.iter().any(|m| m
            .tool_calls
            .as_ref()
            .is_some_and(|calls| calls.iter().any(|tc| tc.arguments == "{}"))));
    }

    #[tokio::test]
    async fn replay_repairs_truncated_arguments_and_missing_tool_results() {
        let mut config = Config::default();
        config.api_key = Some("test-key".into());
        config.base_url = "http://127.0.0.1:9/v1".into();
        let mut app = App::new(config, SessionManager::for_tests());

        // Doubly broken history: unterminated arguments AND no tool result.
        app.sessions.add_assistant_with_tools(
            "",
            vec![ToolCallRecord {
                id: "c1".into(),
                name: "edit_file".into(),
                arguments: r#"{"new_string": "#.into(),
                signature: None,
            }],
        );

        app.start_stream().expect("stream starts");

        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("repaired 1 truncated tool call")
        )));
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("backfilled 1 missing tool result")
        )));
        app.interrupt();
    }

    /// An app with a satisfied API key (pointed at a closed endpoint), a
    /// temp workspace and read-only permissions — for exercising the agent
    /// health guards round after round without real network traffic.
    fn guarded_agent_app(name: &str) -> App {
        let config: Config = serde_json::from_str(
            r#"{"provider":"openai","base_url":"http://127.0.0.1:9/v1","api_key":"test-key"}"#,
        )
        .unwrap();
        let mut app = App::new(config, SessionManager::for_tests());
        let dir =
            std::env::temp_dir().join(format!("chatTUI_guard_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        app.sandbox.set_target(dir.to_str().unwrap()).unwrap();
        app.agent_mode = true;
        app.sandbox.config.permission_mode = PermissionMode::ReadOnly;
        app
    }

    #[tokio::test]
    async fn identical_failing_rounds_stop_the_agent() {
        let mut app = guarded_agent_app("stagnant");
        for round in 0..3 {
            feed(
                &mut app,
                vec![tool_call_event(
                    &format!("c{round}"),
                    "write_file",
                    r#"{"path":"x","content":"y"}"#,
                )],
            );
            drain_tools(&mut app).await;
        }

        // Third identical failing round: the stagnation guard stops the run.
        assert!(!app.streaming);
        assert_eq!(app.agent_iterations, 0);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("not making progress")
        )));
    }

    #[tokio::test]
    async fn consecutive_failing_rounds_stop_the_agent() {
        let mut app = guarded_agent_app("failing");
        // A different call each round keeps the stagnation guard quiet;
        // every round still fails (read-only permission).
        for round in 0..4 {
            feed(
                &mut app,
                vec![tool_call_event(
                    &format!("c{round}"),
                    "write_file",
                    &format!(r#"{{"path":"f{round}","content":"y"}}"#),
                )],
            );
            drain_tools(&mut app).await;
        }

        assert!(!app.streaming);
        assert!(app.cells.iter().any(|c| matches!(
            c,
            Cell::Notice(text) if text.contains("consecutive rounds failed")
        )));
    }
}
