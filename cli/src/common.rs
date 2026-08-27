//! Harness core: repo-root resolution, the run/dry-run helper, shell quoting. Every subcommand
//! shells out to an existing Python/bash tool through `run`; nothing heavy is reimplemented here.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo root: the binary ships at cli/target/{release,debug}/aupai, so the root is four levels up
/// (aupai -> release -> target -> cli -> ROOT). Fall back to the caller's cwd if that doesn't check
/// out (e.g. the binary was moved), keyed on pyproject.toml.
pub fn repo_root() -> PathBuf {
    if let Ok(exe) = env::current_exe() {
        // ancestors(): 0=exe, 1=release, 2=target, 3=cli, 4=ROOT
        if let Some(root) = exe.ancestors().nth(4) {
            if root.join("pyproject.toml").is_file() {
                return root.to_path_buf();
            }
        }
    }
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    // Walk up from cwd to the first dir holding pyproject.toml, else cwd itself.
    for anc in cwd.ancestors() {
        if anc.join("pyproject.toml").is_file() {
            return anc.to_path_buf();
        }
    }
    cwd
}

/// One shell-out step: the program and its args. Rendered verbatim under --dry-run.
pub struct Step {
    pub prog: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

impl Step {
    pub fn new(prog: &str, args: Vec<String>) -> Self {
        Step {
            prog: prog.into(),
            args,
            env: Vec::new(),
        }
    }

    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    /// `uv run python <script> [args]`.
    pub fn uv_py(script: &str, args: &[String]) -> Self {
        let mut full = vec!["run".into(), "python".into(), script.into()];
        full.extend_from_slice(args);
        Step::new("uv", full)
    }

    /// `bash <script> [args]`.
    pub fn bash(script: &str, args: &[String]) -> Self {
        let mut full = vec![script.to_string()];
        full.extend_from_slice(args);
        Step::new("bash", full)
    }

    /// The command as a copy-pasteable shell string (for --dry-run and banners).
    pub fn render(&self, root: &Path) -> String {
        let mut s = format!("cd {} &&", shell_quote(&root.display().to_string()));
        for (k, v) in &self.env {
            s.push_str(&format!(" {}={}", k, shell_quote(v)));
        }
        s.push(' ');
        s.push_str(&self.prog);
        for a in &self.args {
            s.push(' ');
            s.push_str(&shell_quote(a));
        }
        s
    }
}

/// Run a step from the repo root with inherited stdio, or print it under --dry-run. Returns the exit
/// code (0 when dry).
pub fn run(root: &Path, dry: bool, step: &Step) -> i32 {
    if dry {
        println!("{}", step.render(root));
        return 0;
    }
    let mut cmd = Command::new(&step.prog);
    cmd.args(&step.args).current_dir(root);
    for (k, v) in &step.env {
        cmd.env(k, v);
    }
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("aupai: {}: {e}", step.prog);
            127
        }
    }
}

/// Single-quote a string for shell display if it contains any special character.
pub fn shell_quote(s: &str) -> String {
    if s.is_empty()
        || s.chars().any(|c| {
            matches!(
                c,
                ' ' | '\t'
                    | '\n'
                    | '"'
                    | '\''
                    | '\\'
                    | '$'
                    | '`'
                    | '&'
                    | '|'
                    | ';'
                    | '<'
                    | '>'
                    | '*'
                    | '?'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '('
                    | ')'
                    | '#'
                    | '~'
                    | '!'
                    | '^'
            )
        })
    {
        format!("'{}'", s.replace('\'', "'\\''"))
    } else {
        s.to_string()
    }
}
