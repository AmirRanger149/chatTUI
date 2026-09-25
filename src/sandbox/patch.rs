//! Patch-format file editing.
//!
//! `edit_file` replaces one exact string, which means a single stray space
//! between what the model believes the file contains and what it contains
//! fails the edit — and a change spanning several files is several
//! independent writes, each able to fail halfway and leave the tree
//! half-edited.
//!
//! A patch fixes both. It describes every change to every file in one
//! document, anchored on surrounding context rather than on a unique exact
//! string, and it is planned in full before anything is written: if any
//! hunk does not apply, nothing is written at all.
//!
//! ## Format
//!
//! ```text
//! *** Begin Patch
//! *** Add File: src/new.rs
//! +fn main() {}
//! *** Update File: src/main.rs
//! @@ fn main
//!  fn main() {
//! -    println!("hi");
//! +    println!("hello");
//!  }
//! *** Delete File: src/old.rs
//! *** End Patch
//! ```
//!
//! - A body line starts with `+` (add), `-` (remove) or a single space
//!   (context, unchanged). An empty line inside a hunk is an empty context
//!   line, i.e. a blank line in the file.
//! - `@@` is an optional search hint: a string that must appear at or above
//!   the hunk. Use it when the surrounding context repeats in the file.
//! - Paths are relative to the workspace and go through the same
//!   workspace-restriction and sensitive-file checks as every other file
//!   tool.
//!
//! This module is deliberately pure — parsing and hunk arithmetic only, no
//! filesystem and no policy. [`crate::sandbox::Sandbox::apply_patch`] owns
//! path resolution, permissions and I/O.

use anyhow::{anyhow, Result};

const BEGIN: &str = "*** Begin Patch";
const END: &str = "*** End Patch";
const ADD: &str = "*** Add File:";
const UPDATE: &str = "*** Update File:";
const DELETE: &str = "*** Delete File:";
const ANCHOR: &str = "@@";

/// One line of an update hunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PatchLine {
    /// Unchanged; must be present in the file for the hunk to match.
    Context(String),
    /// Present in the file, absent afterwards.
    Remove(String),
    /// Absent from the file, present afterwards.
    Add(String),
}

/// One file operation inside a patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileOp {
    /// Create a file that must not exist yet.
    Add { path: String, content: Vec<String> },
    /// Change an existing file, anchored on context.
    Update {
        path: String,
        anchor: Option<String>,
        body: Vec<PatchLine>,
    },
    /// Remove a file that must exist.
    Delete { path: String },
}

/// A parsed patch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Patch {
    pub ops: Vec<FileOp>,
}

/// What one file becomes once the patch is applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedFile {
    /// The path exactly as the patch wrote it, for reporting.
    pub path: String,
    /// New contents, or `None` to delete the file.
    pub content: Option<String>,
    pub added: usize,
    pub removed: usize,
    /// `"added"`, `"updated"` or `"deleted"`.
    pub kind: &'static str,
}

/// Parse a patch document.
///
/// Tolerates the decorations models like to add — a markdown fence, prose
/// before or after — by looking for the begin/end markers rather than
/// requiring the input to be nothing but the patch.
pub(crate) fn parse(input: &str) -> Result<Patch> {
    let raw: Vec<&str> = input.lines().collect();
    let begin = raw
        .iter()
        .position(|line| line.trim() == BEGIN)
        .ok_or_else(|| anyhow!("no `{BEGIN}` line — the patch must be wrapped in `{BEGIN}` … `{END}`"))?;
    let end = raw
        .iter()
        .rposition(|line| line.trim() == END)
        .ok_or_else(|| anyhow!("no `{END}` line — the patch was probably cut off; resend it"))?;
    if end <= begin {
        return Err(anyhow!("`{END}` comes before `{BEGIN}`"));
    }

    let mut ops: Vec<FileOp> = Vec::new();
    let mut index = begin + 1;
    while index < end {
        let trimmed = raw[index].trim();
        if trimmed.is_empty() {
            index += 1;
            continue;
        }
        if let Some(path) = trimmed.strip_prefix(ADD) {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err(anyhow!("{ADD} needs a path"));
            }
            let (content, next) = read_added(&raw, index + 1, end)?;
            ops.push(FileOp::Add { path, content });
            index = next;
        } else if let Some(path) = trimmed.strip_prefix(UPDATE) {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err(anyhow!("{UPDATE} needs a path"));
            }
            let (anchor, body, next) = read_update(&raw, index + 1, end)?;
            if body.is_empty() {
                return Err(anyhow!("{UPDATE} {path} has no changes"));
            }
            ops.push(FileOp::Update { path, anchor, body });
            index = next;
        } else if let Some(path) = trimmed.strip_prefix(DELETE) {
            let path = path.trim().to_string();
            if path.is_empty() {
                return Err(anyhow!("{DELETE} needs a path"));
            }
            ops.push(FileOp::Delete { path });
            index += 1;
        } else {
            return Err(anyhow!(
                "unexpected line in the patch: {:?} — expected `{ADD}`, `{UPDATE}` or `{DELETE}`",
                raw[index]
            ));
        }
    }

    if ops.is_empty() {
        return Err(anyhow!("the patch contains no file operations"));
    }
    Ok(Patch { ops })
}

/// Collect the `+` lines of an `Add File` block.
fn read_added(raw: &[&str], mut index: usize, end: usize) -> Result<(Vec<String>, usize)> {
    let mut content = Vec::new();
    while index < end {
        let line = raw[index];
        if line.trim().starts_with("***") {
            break;
        }
        match line.strip_prefix('+') {
            Some(text) => content.push(text.to_string()),
            None if line.is_empty() => content.push(String::new()),
            None => {
                return Err(anyhow!(
                    "line {index} of an added file must start with '+': {line:?}"
                ))
            }
        }
        index += 1;
    }
    Ok((content, index))
}

/// Collect the optional `@@` hint and the body of an `Update File` block.
fn read_update(
    raw: &[&str],
    mut index: usize,
    end: usize,
) -> Result<(Option<String>, Vec<PatchLine>, usize)> {
    let mut anchor = None;
    let mut body = Vec::new();
    while index < end {
        let line = raw[index];
        if line.trim().starts_with("***") {
            break;
        }
        if let Some(hint) = line.strip_prefix(ANCHOR) {
            if !body.is_empty() {
                return Err(anyhow!("`{ANCHOR}` must come before the hunk, not inside it"));
            }
            if anchor.is_some() {
                return Err(anyhow!("a hunk can only have one `{ANCHOR}` hint"));
            }
            anchor = Some(hint.trim().to_string());
            index += 1;
            continue;
        }
        if let Some(text) = line.strip_prefix('-') {
            body.push(PatchLine::Remove(text.to_string()));
        } else if let Some(text) = line.strip_prefix('+') {
            body.push(PatchLine::Add(text.to_string()));
        } else if let Some(text) = line.strip_prefix(' ') {
            body.push(PatchLine::Context(text.to_string()));
        } else if line.is_empty() {
            // A blank line inside a hunk is a blank line in the file.
            body.push(PatchLine::Context(String::new()));
        } else {
            return Err(anyhow!(
                "line {index} must start with ' ', '+' or '-': {line:?}"
            ));
        }
        index += 1;
    }
    Ok((anchor, body, index))
}

/// Work out what every file becomes, without touching the filesystem.
///
/// `read` returns the current contents of a path, or `None` when the file
/// does not exist. Several hunks may target the same file; they are applied
/// in the order written, each against the result of the previous one.
///
/// Every hunk is validated before any of them is returned, so a caller that
/// writes the result writes a patch that fully applies or not at all.
pub(crate) fn plan<F>(patch: &Patch, mut read: F) -> Result<Vec<PlannedFile>>
where
    F: FnMut(&str) -> Option<String>,
{
    let mut planned: Vec<PlannedFile> = Vec::new();
    // Files this patch has already changed, so a later hunk for the same
    // path sees the earlier one's result.
    let mut staged: Vec<(String, Option<String>)> = Vec::new();

    let mut current = |path: &str, staged: &[(String, Option<String>)]| -> Option<String> {
        if let Some((_, content)) = staged.iter().rev().find(|(p, _)| p == path) {
            return content.clone();
        }
        read(path)
    };

    for op in &patch.ops {
        match op {
            FileOp::Add { path, content } => {
                if current(path, &staged).is_some() {
                    return Err(anyhow!(
                        "`{ADD} {path}` but that file already exists — use `{UPDATE}` to change it"
                    ));
                }
                let text = join_lines(content);
                let added = content.len();
                staged.push((path.clone(), Some(text.clone())));
                planned.push(PlannedFile {
                    path: path.clone(),
                    content: Some(text),
                    added,
                    removed: 0,
                    kind: "added",
                });
            }
            FileOp::Delete { path } => {
                let existing = current(path, &staged)
                    .ok_or_else(|| anyhow!("`{DELETE} {path}` but that file does not exist"))?;
                let removed = existing.lines().count();
                staged.push((path.clone(), None));
                planned.push(PlannedFile {
                    path: path.clone(),
                    content: None,
                    added: 0,
                    removed,
                    kind: "deleted",
                });
            }
            FileOp::Update { path, anchor, body } => {
                let existing = current(path, &staged).ok_or_else(|| {
                    anyhow!("`{UPDATE} {path}` but that file does not exist — use `{ADD}` to create it")
                })?;
                let (text, added, removed) = apply_hunk(&existing, body, anchor.as_deref(), path)?;
                staged.push((path.clone(), Some(text.clone())));
                planned.push(PlannedFile {
                    path: path.clone(),
                    content: Some(text),
                    added,
                    removed,
                    kind: "updated",
                });
            }
        }
    }

    Ok(planned)
}

/// Apply one update hunk to a file's contents.
fn apply_hunk(
    existing: &str,
    body: &[PatchLine],
    anchor: Option<&str>,
    path: &str,
) -> Result<(String, usize, usize)> {
    let lines: Vec<&str> = existing.lines().collect();

    // The lines the hunk expects to find, in order.
    let mut expected = expected_lines(body);
    if expected.is_empty() {
        return Err(anyhow!(
            "the hunk for {path} has no context or removed lines to match against"
        ));
    }

    // Models frequently end a hunk with a stray blank line. Retry without
    // trailing empty context lines before giving up: dropping an empty
    // context line at the very end cannot move the match, so it is the one
    // ambiguity that is safe to resolve automatically.
    let trimmed = trim_trailing_empty_context(body);
    let mut matches = find_matches(&lines, &expected, anchor);
    let mut use_trimmed = false;
    if matches.is_empty() && trimmed.len() != body.len() {
        let retry = expected_lines(&trimmed);
        let found = find_matches(&lines, &retry, anchor);
        if !found.is_empty() {
            expected = retry;
            matches = found;
            use_trimmed = true;
        }
    }

    let start = match matches.len() {
        1 => matches[0],
        0 => return Err(not_found_error(path, anchor, &expected)),
        count => {
            return Err(anyhow!(
                "the hunk for {path} matches {count} places in the file — add a `{ANCHOR}` hint                  or more context lines so it is unambiguous"
            ))
        }
    };

    let effective: &[PatchLine] = if use_trimmed { &trimmed } else { body };
    let replaced = expected.len();
    let replacement: Vec<&str> = effective
        .iter()
        .filter_map(|line| match line {
            PatchLine::Context(text) | PatchLine::Add(text) => Some(text.as_str()),
            PatchLine::Remove(_) => None,
        })
        .collect();

    let mut out: Vec<&str> = Vec::with_capacity(lines.len() - replaced + replacement.len());
    out.extend_from_slice(&lines[..start]);
    out.extend_from_slice(&replacement);
    out.extend_from_slice(&lines[start + replaced..]);

    // Preserve the file's trailing-newline convention.
    let mut text = out.join("\n");
    if !text.is_empty() && existing.ends_with('\n') {
        text.push('\n');
    }

    let added = effective
        .iter()
        .filter(|line| matches!(line, PatchLine::Add(_)))
        .count();
    let removed = effective
        .iter()
        .filter(|line| matches!(line, PatchLine::Remove(_)))
        .count();
    Ok((text, added, removed))
}

/// The lines a hunk expects to find in the file: context and removals.
fn expected_lines(body: &[PatchLine]) -> Vec<&str> {
    body.iter()
        .filter_map(|line| match line {
            PatchLine::Context(text) | PatchLine::Remove(text) => Some(text.as_str()),
            PatchLine::Add(_) => None,
        })
        .collect()
}

/// Every offset at which `expected` occurs in `lines`, optionally restricted
/// to occurrences at or after the first line containing `anchor`.
fn find_matches(lines: &[&str], expected: &[&str], anchor: Option<&str>) -> Vec<usize> {
    let floor = anchor.and_then(|needle| {
        lines
            .iter()
            .position(|line| line.contains(needle))
            // The anchor may sit below the last possible start; clamping to
            // the end simply yields no matches, which is the honest answer.
    });
    let mut found = Vec::new();
    if expected.len() > lines.len() {
        return found;
    }
    for start in 0..=lines.len() - expected.len() {
        if let Some(floor) = floor {
            if start + expected.len() <= floor {
                continue;
            }
        }
        if lines[start..start + expected.len()] == *expected {
            found.push(start);
        }
    }
    found
}

/// Drop trailing empty context lines — the usual artefact of a model ending
/// a hunk with a blank line.
fn trim_trailing_empty_context(body: &[PatchLine]) -> Vec<PatchLine> {
    let mut out = body.to_vec();
    while matches!(out.last(), Some(PatchLine::Context(text)) if text.is_empty()) {
        out.pop();
    }
    out
}

fn not_found_error(path: &str, anchor: Option<&str>, expected: &[&str]) -> anyhow::Error {
    let first = expected.first().copied().unwrap_or("");
    let mut message = format!(
        "the hunk for {path} does not match the file — the file has changed since you read it, \
         or the context lines differ. Re-read {path} and send the hunk again"
    );
    if let Some(anchor) = anchor {
        message.push_str(&format!(" (the `{ANCHOR} {anchor}` hint was not found either)"));
    } else if !first.is_empty() {
        message.push_str(&format!("; first expected line: {first:?}"));
    }
    anyhow!(message)
}

/// Join lines the way a text file ends: one newline between them and one at
/// the end, unless there is nothing to write.
fn join_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

/// Render the per-file summary that goes back to the model and into the
/// transcript.
pub(crate) fn format_report(files: &[PlannedFile]) -> String {
    let mut out = String::from("applied patch:");
    for file in files {
        out.push_str(&format!(
            "\n  {} {} (+{} -{})",
            file.kind, file.path, file.added, file.removed
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Plan against an in-memory filesystem.
    fn plan_against(patch: &str, files: &[(&str, &str)]) -> Result<Vec<PlannedFile>> {
        let map: HashMap<String, String> = files
            .iter()
            .map(|(path, content)| (path.to_string(), content.to_string()))
            .collect();
        let parsed = parse(patch)?;
        plan(&parsed, |path| map.get(path).cloned())
    }

    // Patch bodies are raw strings throughout: a context line starts with a
    // single space, and an escaped string literal with `\`-continuations
    // would strip exactly that space.

    #[test]
    fn adds_a_new_file() {
        let out = plan_against(
            r#"*** Begin Patch
*** Add File: src/new.rs
+fn main() {}
*** End Patch"#,
            &[],
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, "added");
        assert_eq!(out[0].content.as_deref(), Some("fn main() {}\n"));
        assert_eq!((out[0].added, out[0].removed), (1, 0));
    }

    #[test]
    fn adding_an_existing_file_is_refused() {
        let error = plan_against(
            "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch",
            &[("a.txt", "old\n")],
        )
        .unwrap_err();
        assert!(error.to_string().contains("already exists"), "{error}");
    }

    #[test]
    fn updates_using_context() {
        let file = "fn main() {\n    println!(\"hi\");\n}\n";
        let out = plan_against(
            r#"*** Begin Patch
*** Update File: main.rs
@@ fn main
 fn main() {
-    println!("hi");
+    println!("hello");
 }
*** End Patch"#,
            &[("main.rs", file)],
        )
        .unwrap();
        assert_eq!(out[0].kind, "updated");
        assert_eq!(
            out[0].content.as_deref(),
            Some("fn main() {\n    println!(\"hello\");\n}\n")
        );
        assert_eq!((out[0].added, out[0].removed), (1, 1));
    }

    #[test]
    fn a_hunk_that_does_not_match_is_refused_with_guidance() {
        let error = plan_against(
            "*** Begin Patch\n*** Update File: a.txt\n-not here\n+gone\n*** End Patch",
            &[("a.txt", "something else\n")],
        )
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("does not match"), "{text}");
        assert!(text.contains("Re-read"), "{text}");
    }

    #[test]
    fn an_ambiguous_hunk_is_refused() {
        let error = plan_against(
            "*** Begin Patch\n*** Update File: a.txt\n-dup\n+one\n*** End Patch",
            &[("a.txt", "dup\ndup\n")],
        )
        .unwrap_err();
        assert!(error.to_string().contains("matches 2 places"), "{error}");
    }

    #[test]
    fn an_anchor_disambiguates() {
        let out = plan_against(
            r#"*** Begin Patch
*** Update File: a.txt
@@ header two
-dup
+two
*** End Patch"#,
            &[("a.txt", "header one\ndup\nheader two\ndup\n")],
        )
        .unwrap();
        assert_eq!(
            out[0].content.as_deref(),
            Some("header one\ndup\nheader two\ntwo\n")
        );
    }

    #[test]
    fn trailing_blank_context_is_tolerated() {
        let out = plan_against(
            "*** Begin Patch\n*** Update File: a.txt\n-one\n+two\n\n*** End Patch",
            &[("a.txt", "one\n")],
        )
        .unwrap();
        assert_eq!(out[0].content.as_deref(), Some("two\n"));
    }

    #[test]
    fn deletes_a_file() {
        let out = plan_against(
            "*** Begin Patch\n*** Delete File: old.txt\n*** End Patch",
            &[("old.txt", "a\nb\n")],
        )
        .unwrap();
        assert_eq!(out[0].kind, "deleted");
        assert_eq!(out[0].content, None);
        assert_eq!(out[0].removed, 2);
    }

    #[test]
    fn deleting_a_missing_file_is_refused() {
        let error = plan_against(
            "*** Begin Patch\n*** Delete File: nope.txt\n*** End Patch",
            &[],
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");
    }

    #[test]
    fn several_files_in_one_patch_are_all_planned() {
        let out = plan_against(
            "*** Begin Patch\n\
             *** Add File: b.txt\n+beta\n\
             *** Update File: a.txt\n-alpha\n+alpha2\n\
             *** Delete File: c.txt\n\
             *** End Patch",
            &[("a.txt", "alpha\n"), ("c.txt", "gamma\n")],
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        let kinds: Vec<&str> = out.iter().map(|file| file.kind).collect();
        assert_eq!(kinds, vec!["added", "updated", "deleted"]);
    }

    #[test]
    fn two_hunks_for_one_file_are_applied_in_order() {
        let out = plan_against(
            "*** Begin Patch\n\
             *** Update File: a.txt\n-one\n+ONE\n\
             *** Update File: a.txt\n-three\n+THREE\n\
             *** End Patch",
            &[("a.txt", "one\ntwo\nthree\n")],
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[1].content.as_deref(),
            Some("ONE\ntwo\nTHREE\n"),
            "the second hunk must see the first one's result"
        );
    }

    #[test]
    fn a_broken_patch_reports_what_is_wrong() {
        assert!(parse("no markers here").is_err());
        assert!(parse("*** Begin Patch\n*** End Patch").is_err());
        assert!(parse("*** Begin Patch\n*** Update File:\n-x\n+y\n*** End Patch").is_err());
        assert!(parse("*** Begin Patch\n*** Nope File: a\n*** End Patch").is_err());
        // Unclosed — the shape a truncated stream leaves behind.
        let error = parse("*** Begin Patch\n*** Add File: a.txt\n+half").unwrap_err();
        assert!(error.to_string().contains("cut off"), "{error}");
    }

    #[test]
    fn prose_and_fences_around_the_patch_are_ignored() {
        let out = plan_against(
            "Here you go:\n```diff\n*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch\n```\nDone.",
            &[],
        )
        .unwrap();
        assert_eq!(out[0].content.as_deref(), Some("x\n"));
    }

    #[test]
    fn report_lists_every_file() {
        let out = plan_against(
            "*** Begin Patch\n*** Add File: a.txt\n+x\n*** End Patch",
            &[],
        )
        .unwrap();
        let report = format_report(&out);
        assert!(report.contains("added a.txt (+1 -0)"), "{report}");
    }

    #[test]
    fn empty_context_lines_are_real_blank_lines() {
        let out = plan_against(
            r#"*** Begin Patch
*** Update File: a.txt
 a
 
-b
+B
*** End Patch"#,
            &[("a.txt", "a\n\nb\n")],
        )
        .unwrap();
        assert_eq!(out[0].content.as_deref(), Some("a\n\nB\n"));
    }
}
