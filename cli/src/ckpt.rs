//! `aupai ckpt` — enumerate, rank, and prune checkpoints. Reading a .pt in Rust isn't feasible
//! (pickle), so `list`/`best` shell out once to `scripts/ckpt_info.py` (cheap mmap metadata read);
//! `clean` is pure filesystem and never opens a checkpoint. Destructive `clean` is dry-run by default
//! and only deletes `.stepN`/`.epN` intermediates, never a final `ckpt_<name>.pt`.

use std::fs;
use std::path::{Path, PathBuf};

use clap::Subcommand;
use serde_json::Value;

use crate::common::{fmt_bytes, py_runner, shell_quote};

/// The best base recorded in EXPERIMENTS.md / runs/experiments.jsonl. Used as the fallback when the
/// jsonl scan finds nothing, and named in `ckpt best`'s "check EXPERIMENTS.md" note.
pub const KNOWN_BEST: &str = "ckpt_k4_11b_lr05.pt";

#[derive(Subcommand)]
pub enum Cmd {
    /// List ckpt_*.pt (+ .stepN/.epN) with size, step, arch (reads cfg via scripts/ckpt_info.py)
    List {
        /// Machine-readable JSONL
        #[arg(long)]
        json: bool,
    },
    /// Report the current best base (source: runs/experiments.jsonl / EXPERIMENTS.md)
    Best {
        #[arg(long)]
        json: bool,
    },
    /// Prune intermediate .stepN/.epN checkpoints (dry-run unless --force); never touches finals
    Clean {
        /// Keep the newest N intermediates per base (default 1: the highest step)
        #[arg(long, default_value_t = 1)]
        keep: usize,
        /// Actually delete (default: print what would be deleted and stop)
        #[arg(long, alias = "yes")]
        force: bool,
    },
}

/// One checkpoint file on disk and whatever cfg/step we could read from it.
struct Info {
    name: String,
    size: u64,
    is_final: bool,
    step: Option<i64>,
    params: Option<i64>,
    layers: Option<i64>,
    d: Option<i64>,
    attn_res: Option<bool>,
    date: Option<String>,
    error: Option<String>,
}

/// A final checkpoint is `ckpt_<name>.pt`; an intermediate is `ckpt_<name>.pt.step<N>` / `.pt.ep<N>`.
fn is_final(name: &str) -> bool {
    name.starts_with("ckpt_") && name.ends_with(".pt")
}
fn is_intermediate(name: &str) -> bool {
    name.starts_with("ckpt_") && (name.contains(".pt.step") || name.contains(".pt.ep"))
}

/// All checkpoint files in the repo root (finals + intermediates), sorted by name.
fn scan(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(root) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if is_final(&name) || is_intermediate(&name) {
                out.push(e.path());
            }
        }
    }
    out.sort();
    out
}

/// Shell out once to ckpt_info.py for cfg/step/params (JSONL, one object per path). On any failure
/// the files are still listed from the filesystem, just without the cfg fields.
fn read_infos(root: &Path, paths: &[PathBuf]) -> Vec<Info> {
    let mut meta: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    if !paths.is_empty() {
        let (prog, mut argv) = py_runner(root);
        argv.push("scripts/ckpt_info.py".into());
        for p in paths {
            argv.push(p.file_name().unwrap().to_string_lossy().to_string());
        }
        if let Ok(out) = std::process::Command::new(&prog)
            .args(&argv)
            .current_dir(root)
            .output()
        {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
                        meta.insert(p.to_string(), v);
                    }
                }
            }
        }
    }
    paths
        .iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            let v = meta.get(&name);
            let g = |k: &str| v.and_then(|v| v.get(k)).cloned().unwrap_or(Value::Null);
            Info {
                name: name.clone(),
                size,
                is_final: is_final(&name),
                step: g("step").as_i64(),
                params: g("params").as_i64(),
                layers: g("layers").as_i64(),
                d: g("d").as_i64(),
                attn_res: g("attn_res").as_bool(),
                date: g("date").as_str().map(str::to_string),
                error: g("error").as_str().map(str::to_string),
            }
        })
        .collect()
}

fn fmt_params(n: Option<i64>) -> String {
    match n {
        Some(n) => format!("{:.1}M", n as f64 / 1e6),
        None => "?".into(),
    }
}

fn arch_str(i: &Info) -> String {
    if let Some(e) = &i.error {
        return format!("(unreadable: {e})");
    }
    let mut parts = Vec::new();
    if let (Some(l), Some(d)) = (i.layers, i.d) {
        parts.push(format!("L{l} d{d}"));
    }
    if i.attn_res == Some(true) {
        parts.push("attn_res".into());
    }
    parts.join(" ")
}

pub fn run(root: &Path, cmd: Cmd) -> i32 {
    match cmd {
        Cmd::List { json } => list(root, json),
        Cmd::Best { json } => best(root, json),
        Cmd::Clean { keep, force } => clean(root, keep, force),
    }
}

fn list(root: &Path, json: bool) -> i32 {
    let paths = scan(root);
    let infos = read_infos(root, &paths);
    if json {
        for i in &infos {
            let obj = serde_json::json!({
                "name": i.name,
                "kind": if i.is_final { "final" } else { "intermediate" },
                "size_bytes": i.size,
                "step": i.step,
                "params": i.params,
                "layers": i.layers,
                "d": i.d,
                "attn_res": i.attn_res,
                "date": i.date,
                "error": i.error,
            });
            println!("{obj}");
        }
        return 0;
    }
    if infos.is_empty() {
        println!("no checkpoints (ckpt_*.pt) in {}", root.display());
        return 0;
    }
    let total: u64 = infos.iter().map(|i| i.size).sum();
    println!(
        "aupai checkpoints — {} files, {} on disk ({})\n",
        infos.len(),
        fmt_bytes(total),
        root.display()
    );
    println!(
        "{:<34}{:>9}{:>8}{:>9}  {:<16}{:>12}",
        "name", "size", "step", "params", "arch", "date"
    );
    for i in &infos {
        let tag = if i.is_final { "" } else { "  ·" };
        println!(
            "{:<34}{:>9}{:>8}{:>9}  {:<16}{:>12}{}",
            i.name,
            fmt_bytes(i.size),
            i.step.map(|s| s.to_string()).unwrap_or_else(|| "—".into()),
            fmt_params(i.params),
            arch_str(i),
            i.date.clone().unwrap_or_default(),
            tag,
        );
    }
    println!("\n(· = intermediate .stepN/.epN — prune with `aupai ckpt clean`)");
    0
}

/// Best base: scan runs/experiments.jsonl for the highest math-hard result whose checkpoint exists.
/// The eval records store the number in a free-text `result` field, so we extract the "math-hard X%"
/// token. When nothing parses, fall back to the known best and say so — never fabricate a metric.
fn best(root: &Path, json: bool) -> i32 {
    let jsonl = root.join("runs").join("experiments.jsonl");
    let mut best: Option<(String, f64, String)> = None; // (name, math-hard %, date)
    if let Ok(text) = fs::read_to_string(&jsonl) {
        for line in text.lines() {
            let v: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let result = v.get("result").and_then(|x| x.as_str()).unwrap_or("");
            let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("");
            let started = v.get("started").and_then(|x| x.as_str()).unwrap_or("");
            if let Some(pct) = parse_math_hard(result) {
                if best.as_ref().map(|b| pct > b.1).unwrap_or(true) {
                    best = Some((name.to_string(), pct, started.to_string()));
                }
            }
        }
    }
    // The best *base* is a pretrain checkpoint; the highest math-hard often belongs to an RL run
    // whose ckpt is derived from it. Report the metric leader, but resolve the on-disk base honestly.
    let known = root.join(KNOWN_BEST);
    if json {
        let obj = serde_json::json!({
            "known_best_base": KNOWN_BEST,
            "known_best_exists": known.exists(),
            "metric_leader": best.as_ref().map(|b| serde_json::json!({
                "run": b.0, "math_hard_pct": b.1, "started": b.2,
            })),
            "source": "runs/experiments.jsonl",
        });
        println!("{obj}");
        return 0;
    }
    println!("aupai best base");
    println!(
        "  known best base : {} {}",
        KNOWN_BEST,
        if known.exists() {
            "(present)"
        } else {
            "(MISSING on disk)"
        }
    );
    match &best {
        Some((run, pct, when)) => println!(
            "  metric leader   : run '{run}' math-hard {pct:.1}% (started {when}) — from runs/experiments.jsonl"
        ),
        None => println!("  metric leader   : none parseable in runs/experiments.jsonl"),
    }
    println!("  note: the pretrain base of record is {KNOWN_BEST} (EXPERIMENTS.md); RL/SFT ckpts derive from it.");
    0
}

/// Pull a "math-hard X%" (or "math-hard X.Y%") percentage out of a free-text result string.
fn parse_math_hard(s: &str) -> Option<f64> {
    let lower = s.to_lowercase();
    let idx = lower.find("math-hard")?;
    let after = &lower[idx + "math-hard".len()..];
    // first number followed (possibly after spaces/colon) by % — e.g. "math-hard 4.1% (42/1032)"
    let mut num = String::new();
    let mut seen_digit = false;
    for c in after.chars() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            seen_digit = true;
        } else if c == '%' && seen_digit {
            return num.parse().ok();
        } else if seen_digit && c != ' ' && c != ':' {
            // number ended without a %, keep scanning for the next candidate
            num.clear();
            seen_digit = false;
        } else if !seen_digit && (c == ' ' || c == ':') {
            continue;
        } else if !seen_digit {
            // skip leading non-numeric noise
            continue;
        }
    }
    None
}

/// Delete intermediate `.stepN`/`.epN` checkpoints, keeping the newest `keep` per base. Dry-run
/// unless `force`. Never touches a final `ckpt_<name>.pt`.
fn clean(root: &Path, keep: usize, force: bool) -> i32 {
    let paths = scan(root);
    // Group intermediates by their base (ckpt_<name>.pt), ordered by step/ep number descending.
    let mut groups: std::collections::BTreeMap<String, Vec<(u64, PathBuf, u64)>> =
        std::collections::BTreeMap::new(); // base -> [(order, path, size)]
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !is_intermediate(&name) {
            continue;
        }
        let (base, ord) = split_intermediate(&name);
        let size = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        groups.entry(base).or_default().push((ord, p.clone(), size));
    }
    if groups.is_empty() {
        println!("aupai ckpt clean: no intermediate (.stepN/.epN) checkpoints to prune");
        return 0;
    }
    let mut to_delete: Vec<(PathBuf, u64)> = Vec::new();
    for (_base, mut items) in groups {
        items.sort_by(|a, b| b.0.cmp(&a.0)); // highest step first
        for (_, p, sz) in items.into_iter().skip(keep) {
            to_delete.push((p, sz));
        }
    }
    if to_delete.is_empty() {
        println!(
            "aupai ckpt clean: nothing to prune (keeping newest {keep} per base; all within limit)"
        );
        return 0;
    }
    let total: u64 = to_delete.iter().map(|(_, s)| *s).sum();
    let verb = if force { "deleting" } else { "would delete" };
    println!(
        "aupai ckpt clean: {verb} {} intermediate checkpoint(s), {} — keeping newest {keep} per base, all finals untouched\n",
        to_delete.len(),
        fmt_bytes(total)
    );
    for (p, sz) in &to_delete {
        println!(
            "  {}  ({})",
            p.file_name().unwrap().to_string_lossy(),
            fmt_bytes(*sz)
        );
    }
    if !force {
        println!("\nNothing deleted. Re-run with --force (or --yes) to remove these.");
        return 0;
    }
    let mut failed = 0;
    for (p, _) in &to_delete {
        if let Err(e) = fs::remove_file(p) {
            eprintln!(
                "aupai ckpt clean: failed to remove {}: {e}",
                shell_quote(&p.display().to_string())
            );
            failed += 1;
        }
    }
    if failed > 0 {
        eprintln!("aupai ckpt clean: {failed} file(s) could not be removed");
        return 1;
    }
    println!(
        "\nDeleted {} file(s), freed {}.",
        to_delete.len(),
        fmt_bytes(total)
    );
    0
}

/// `ckpt_foo.pt.step2000` -> ("ckpt_foo.pt", 2000); `.ep3` -> ("ckpt_foo.pt", 3) (ep sorts below
/// step by staying small — epochs are few and we only need a stable per-base ordering).
fn split_intermediate(name: &str) -> (String, u64) {
    for marker in [".pt.step", ".pt.ep"] {
        if let Some(pos) = name.find(marker) {
            let base = format!("{}.pt", &name[..pos]);
            let num: u64 = name[pos + marker.len()..].parse().unwrap_or(0);
            return (base, num);
        }
    }
    (name.to_string(), 0)
}
