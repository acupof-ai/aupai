//! Harness core: repo-root resolution, the run/dry-run helper, shell quoting. Every subcommand
//! shells out to an existing Python/bash tool through `run`; nothing heavy is reimplemented here.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

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
///
/// A `py` step defers picking the Python launcher until run/render, so the same Step runs through
/// `uv run python` on the mac (a `.venv` is present) and plain `python3` on the pod (no venv).
pub struct Step {
    pub prog: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub py: bool,
}

impl Step {
    pub fn new(prog: &str, args: Vec<String>) -> Self {
        Step {
            prog: prog.into(),
            args,
            env: Vec::new(),
            py: false,
        }
    }

    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    /// A Python script step: `<launcher> <script> [args]`, launcher resolved per box at run time.
    /// (`prog` holds the script; the launcher is prepended in `resolve`.)
    pub fn uv_py(script: &str, args: &[String]) -> Self {
        Step {
            prog: script.into(),
            args: args.to_vec(),
            env: Vec::new(),
            py: true,
        }
    }

    /// `bash <script> [args]`.
    pub fn bash(script: &str, args: &[String]) -> Self {
        let mut full = vec![script.to_string()];
        full.extend_from_slice(args);
        Step::new("bash", full)
    }

    /// Materialize (program, args) for this box: py steps get the resolved launcher prepended.
    fn resolve(&self, root: &Path) -> (String, Vec<String>) {
        if self.py {
            let (prog, mut full) = py_runner(root);
            full.push(self.prog.clone());
            full.extend(self.args.iter().cloned());
            (prog, full)
        } else {
            (self.prog.clone(), self.args.clone())
        }
    }

    /// The command as a copy-pasteable shell string (for --dry-run and banners).
    pub fn render(&self, root: &Path) -> String {
        let (prog, args) = self.resolve(root);
        let mut s = format!("cd {} &&", shell_quote(&root.display().to_string()));
        for (k, v) in &self.env {
            s.push_str(&format!(" {}={}", k, shell_quote(v)));
        }
        s.push(' ');
        s.push_str(&prog);
        for a in &args {
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
    let (prog, args) = step.resolve(root);
    let mut cmd = Command::new(&prog);
    cmd.args(&args).current_dir(root);
    for (k, v) in &step.env {
        cmd.env(k, v);
    }
    match cmd.status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("aupai: {prog}: {e}");
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

/// Python launcher, resolved per box so one `aupai <cmd>` works everywhere: a repo `.venv` (the mac
/// dev box, created by uv) runs through `uv run python`; otherwise (the pod: system python3, no
/// venv) it falls back to plain `python3`. No config, no flags — just a filesystem check.
pub fn py_runner(root: &Path) -> (String, Vec<String>) {
    if root.join(".venv").is_dir() {
        ("uv".into(), vec!["run".into(), "python".into()])
    } else {
        ("python3".into(), Vec::new())
    }
}

/// Cores available to this process (physical/affinity-aware), for auto-parallelism defaults.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Env that makes the CPU-bound Python tools (tokenization in pretokenize / data_overview) use every
/// core without the user naming a count. HF `tokenizers` honors RAYON_NUM_THREADS + TOKENIZERS_
/// PARALLELISM; the math libs honor OMP/MKL. Attached only to the CPU stages, never to GPU launches.
pub fn cpu_env() -> Vec<(String, String)> {
    let n = cpu_count().to_string();
    vec![
        ("RAYON_NUM_THREADS".into(), n.clone()),
        ("TOKENIZERS_PARALLELISM".into(), "true".into()),
        ("OMP_NUM_THREADS".into(), n.clone()),
        ("MKL_NUM_THREADS".into(), n),
    ]
}

/// Seconds since the Unix epoch (UTC).
pub fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Format an epoch-seconds timestamp as `YYYY-MM-DDTHH:MM:SSZ` (UTC), no chrono dependency.
/// Civil date from days via Howard Hinnant's algorithm.
pub fn fmt_utc(secs: i64) -> String {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Human-readable byte size (e.g. 412.3 MB).
pub fn fmt_bytes(n: u64) -> String {
    let n = n as f64;
    for (div, suf) in [(1e9, "GB"), (1e6, "MB"), (1e3, "KB")] {
        if n >= div {
            return format!("{:.1} {}", n / div, suf);
        }
    }
    format!("{n} B")
}
