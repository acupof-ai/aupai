//! `aupai status` — one at-a-glance dashboard of the whole system. Read-only, never blocks, exits 0.
//! Aggregates: data readiness (mix domains with token caches / corpus counts), checkpoints (count,
//! best base, disk), pipelines (each runs/*.pipeline.json and its furthest-completed stage), and the
//! most-recently-touched runs/*.log tail (so an agent sees "pretrain at step N/M" without the name).
//! Each section is short and skippable — a scan surface, not a wall of text.

use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use crate::ckpt::KNOWN_BEST;
use crate::common::{fmt_bytes, fmt_utc};

pub fn run(root: &Path, json_out: bool) -> i32 {
    let data = data_readiness(root);
    let ckpts = checkpoints(root);
    let pipes = pipelines(root);
    let active = active_run(root);

    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "data": data.1,
                "checkpoints": ckpts.1,
                "pipelines": pipes.1,
                "active_run": active.1,
            }))
            .unwrap_or_default()
        );
        return 0;
    }
    println!("aupai status — {}\n", root.display());
    println!("DATA");
    println!("{}", indent(&data.0));
    println!("\nCHECKPOINTS");
    println!("{}", indent(&ckpts.0));
    println!("\nPIPELINES");
    println!("{}", indent(&pipes.0));
    println!("\nACTIVE RUN");
    println!("{}", indent(&active.0));
    0
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|l| format!("  {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn fmt_tokens(n: u64) -> String {
    let n = n as f64;
    for (div, suf) in [(1e9, "B"), (1e6, "M"), (1e3, "K")] {
        if n >= div {
            return format!("{:.2}{}", n / div, suf);
        }
    }
    format!("{n}")
}

/// Which mix domains have data, and total tokens vs the mix target. Cheap: token caches are read by
/// file size (int32, 4 bytes/token) and corpus counts from the .counts.json sidecar the Python tool
/// already maintains — no tokenization here.
fn data_readiness(root: &Path) -> (String, Value) {
    let mix_path = root.join("data").join("mix.json");
    let mix: Value = match fs::read_to_string(&mix_path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
    {
        Some(v) => v,
        None => {
            return (
                "no data/mix.json — flat-corpus mode".into(),
                json!({"mix": false}),
            )
        }
    };
    let target = mix
        .get("total_tokens")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let domains = match mix.get("domains").and_then(|v| v.as_object()) {
        Some(d) => d,
        None => return ("mix.json has no domains".into(), json!({"mix": true})),
    };

    // token caches: <cache_dir>/tokens_<domain>.pt, size/4 = token count. cache_dir = AUPAI_CACHE_DIR
    // or /data00 (matches train.py's TOKEN_CACHE dir).
    let cache_dir = std::env::var("AUPAI_CACHE_DIR").unwrap_or_else(|_| "/data00".into());
    // corpus sidecar: data/corpus/.counts.json — {"<domain>/<file>": [size, mtime, tokens]}
    let sidecar: Value = fs::read_to_string(root.join("data").join("corpus").join(".counts.json"))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(json!({}));

    let mut lines = Vec::new();
    let mut total: u64 = 0;
    let mut jrows = Vec::new();
    let mut ready = 0;
    for (dom, _) in domains {
        let cache = Path::new(&cache_dir).join(format!("tokens_{dom}.pt"));
        let (toks, src) = if let Ok(m) = fs::metadata(&cache) {
            (m.len() / 4, "cache")
        } else {
            // sum sidecar entries whose key starts with "<dom>/"
            let mut sum: u64 = 0;
            let mut hit = false;
            if let Some(obj) = sidecar.as_object() {
                for (k, v) in obj {
                    if k.starts_with(&format!("{dom}/")) {
                        if let Some(n) = v.get(2).and_then(|x| x.as_u64()) {
                            sum += n;
                            hit = true;
                        }
                    }
                }
            }
            (sum, if hit { "corpus" } else { "none" })
        };
        total += toks;
        if src != "none" {
            ready += 1;
        }
        let mark = if src == "none" { "MISSING" } else { src };
        lines.push(format!("{dom:<6} {:>9}  {mark}", fmt_tokens(toks)));
        jrows.push(json!({"domain": dom, "tokens": toks, "source": src}));
    }
    let head = format!(
        "{ready}/{} domains ready · {} tokens (mix target {:.1}B)",
        domains.len(),
        fmt_tokens(total),
        target / 1e9
    );
    let body = format!("{head}\n{}", lines.join("\n"));
    (
        body,
        json!({"mix": true, "ready": ready, "total_domains": domains.len(),
               "total_tokens": total, "target_tokens": target, "domains": jrows}),
    )
}

/// Checkpoint summary: reuse the same filesystem scan `ckpt list` uses (finals + intermediates),
/// report count and disk without opening any .pt (status must stay instant).
fn checkpoints(root: &Path) -> (String, Value) {
    let mut finals = 0;
    let mut inters = 0;
    let mut total: u64 = 0;
    if let Ok(rd) = fs::read_dir(root) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let is_final = name.starts_with("ckpt_") && name.ends_with(".pt");
            let is_inter =
                name.starts_with("ckpt_") && (name.contains(".pt.step") || name.contains(".pt.ep"));
            if is_final || is_inter {
                total += fs::metadata(e.path()).map(|m| m.len()).unwrap_or(0);
                if is_final {
                    finals += 1;
                } else {
                    inters += 1;
                }
            }
        }
    }
    let best_exists = root.join(KNOWN_BEST).exists();
    let body = format!(
        "{finals} final + {inters} intermediate · {} on disk\nbest base: {KNOWN_BEST} {}",
        fmt_bytes(total),
        if best_exists {
            "(present)"
        } else {
            "(MISSING)"
        }
    );
    (
        body,
        json!({"finals": finals, "intermediates": inters, "disk_bytes": total,
               "best_base": KNOWN_BEST, "best_exists": best_exists}),
    )
}

/// Every runs/*.pipeline.json and its furthest-completed stage (reuses the pipeline state layout).
fn pipelines(root: &Path) -> (String, Value) {
    let dir = root.join("runs");
    let mut rows = Vec::new();
    let mut jrows = Vec::new();
    if let Ok(rd) = fs::read_dir(&dir) {
        let mut names: Vec<String> = rd
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().to_string();
                n.strip_suffix(".pipeline.json").map(|s| s.to_string())
            })
            .collect();
        names.sort();
        for name in names {
            let state: Value = fs::read_to_string(dir.join(format!("{name}.pipeline.json")))
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .unwrap_or(json!({}));
            let stages = state.get("stages").and_then(|v| v.as_object());
            let (furthest, failed) = furthest_stage(stages);
            let label = if let Some(f) = &failed {
                format!("FAILED at {f}")
            } else {
                furthest.clone().unwrap_or_else(|| "pending".into())
            };
            rows.push(format!("{name:<20} {label}"));
            jrows.push(json!({"name": name, "furthest": furthest, "failed": failed}));
        }
    }
    let body = if rows.is_empty() {
        "no pipelines (runs/*.pipeline.json)".into()
    } else {
        rows.join("\n")
    };
    (body, json!(jrows))
}

/// The furthest stage marked done (in canonical order), and any stage marked failed.
fn furthest_stage(
    stages: Option<&serde_json::Map<String, Value>>,
) -> (Option<String>, Option<String>) {
    const ORDER: &[&str] = &[
        "tokenizer",
        "pretokenize",
        "data",
        "pretrain",
        "eval",
        "sft",
        "rl",
    ];
    let mut furthest = None;
    let mut failed = None;
    if let Some(obj) = stages {
        for stage in ORDER {
            if let Some(rec) = obj.get(*stage) {
                match rec.get("status").and_then(|v| v.as_str()) {
                    Some("done") => furthest = Some((*stage).to_string()),
                    Some("failed") => failed = Some((*stage).to_string()),
                    _ => {}
                }
            }
        }
    }
    (furthest, failed)
}

/// The tail of the most-recently-modified runs/*.log — the last progress line, so an agent sees the
/// active run's state without knowing its name.
fn active_run(root: &Path) -> (String, Value) {
    let dir = root.join("runs");
    let mut newest: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    if let Ok(rd) = fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("log") {
                continue;
            }
            if let Ok(mt) = e.metadata().and_then(|m| m.modified()) {
                if newest.as_ref().map(|(t, _)| mt > *t).unwrap_or(true) {
                    newest = Some((mt, p));
                }
            }
        }
    }
    let (mtime, path) = match newest {
        Some(v) => v,
        None => return ("no runs/*.log".into(), json!(null)),
    };
    let text = fs::read_to_string(&path).unwrap_or_default();
    let last = text
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .to_string();
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let when = mtime
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| fmt_utc(d.as_secs() as i64))
        .unwrap_or_default();
    let body = format!("{name} ({when})\n{last}");
    (
        body,
        json!({"run": name, "modified": when, "last_line": last}),
    )
}
