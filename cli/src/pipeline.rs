//! `aupai pipeline` — chain the training stages, each shelling out to its existing tool, with a small
//! persistent stage-state file so a run can be resumed, inspected, or forced.
//!
//! Stages in order: tokenizer -> pretokenize -> data -> pretrain -> eval -> sft -> rl. Select a
//! contiguous slice with --from/--to, or an explicit subset with --stages. State lives in
//! `runs/<name>.pipeline.json`: per stage a status (pending/running/done/failed), start/end
//! timestamps, and the artifact it produces. --resume skips stages already done (or whose artifact is
//! already on disk); --status prints the table and runs nothing; --force reruns everything. --dry-run
//! prints the ordered plan AND what a resume would skip, and touches no files.

use std::fs;
use std::path::{Path, PathBuf};

use clap::Args as ClapArgs;
use serde_json::{json, Value};

use crate::common::{cpu_env, fmt_utc, now_secs, run as run_step, Step};
use crate::recipe;
use crate::train_step;

/// Stage order. `optional` stages (sft, rl) are excluded from a default --from/--to slice unless
/// named explicitly or reached by an explicit --to.
const STAGES: &[&str] = &[
    "tokenizer",
    "pretokenize",
    "data",
    "pretrain",
    "eval",
    "sft",
    "rl",
];
/// Default slice when neither --from/--to/--stages is given: the core pretraining path.
const DEFAULT_LAST: &str = "eval";

#[derive(ClapArgs)]
pub struct Args {
    /// First stage to run (default: tokenizer)
    #[arg(long)]
    from: Option<String>,
    /// Last stage to run (default: eval)
    #[arg(long)]
    to: Option<String>,
    /// Explicit comma-separated stage subset (overrides --from/--to)
    #[arg(long)]
    stages: Option<String>,
    /// Run name -> runs/<name>.log, ckpt_<name>.pt, the eval target, and runs/<name>.pipeline.json
    #[arg(long, default_value = "pretrain")]
    name: String,
    /// Pretrain recipe profile: 'best' (verified ckpt_k4_11b_lr05 recipe) or 'base' (bare train.py)
    #[arg(long, default_value = "best")]
    profile: String,
    /// GPUs for pretrain/eval/rl
    #[arg(long, default_value_t = 8)]
    ngpu: u8,
    /// Log pretraining to trackio
    #[arg(long)]
    track: bool,
    /// SFT: packed data (.pt) the sft stage trains on
    #[arg(long)]
    sft_pt: Option<String>,
    /// RL: prompt data the rl stage trains on
    #[arg(long)]
    rl_data: Option<String>,
    /// Print the stage-state table and exit (optionally naming the run: --status <name>)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    status: Option<String>,
    /// Resume: skip stages already done or whose artifact exists (optionally --resume <name>)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,
    /// Rerun every selected stage, ignoring saved state
    #[arg(long)]
    force: bool,
    /// Machine-readable output (for --status)
    #[arg(long)]
    json: bool,
    /// Extra flags forwarded to the pretrain stage (train.py) after `--`, e.g. `-- --grad_ckpt`
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
    passthrough: Vec<String>,
}

/// The run name: the value given to --status/--resume if any, else --name.
fn run_name(a: &Args) -> String {
    for opt in [&a.status, &a.resume] {
        if let Some(v) = opt {
            if !v.is_empty() {
                return v.clone();
            }
        }
    }
    a.name.clone()
}

fn idx(stage: &str) -> Option<usize> {
    STAGES.iter().position(|s| *s == stage)
}

/// Resolve the ordered stage list from the flags.
fn resolve(a: &Args) -> Result<Vec<String>, String> {
    if let Some(explicit) = &a.stages {
        let mut out = Vec::new();
        for s in explicit.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if idx(s).is_none() {
                return Err(format!(
                    "unknown stage '{s}' (valid: {})",
                    STAGES.join(", ")
                ));
            }
            out.push(s.to_string());
        }
        return Ok(out);
    }
    let from = a.from.as_deref().unwrap_or("tokenizer");
    let to = a.to.as_deref().unwrap_or(DEFAULT_LAST);
    let (fi, ti) = match (idx(from), idx(to)) {
        (Some(f), Some(t)) => (f, t),
        _ => {
            return Err(format!(
                "unknown --from/--to (valid: {})",
                STAGES.join(", ")
            ))
        }
    };
    if fi > ti {
        return Err(format!("--from '{from}' comes after --to '{to}'"));
    }
    Ok(STAGES[fi..=ti].iter().map(|s| s.to_string()).collect())
}

/// Relative artifact path a stage produces, when it produces one checkable on disk. `pretokenize`
/// caches live off-repo (AUPAI_CACHE_DIR / /data00), and `data` is validation-only, so both are None
/// and resume falls back to their saved status.
fn artifact_for(stage: &str, name: &str) -> Option<String> {
    match stage {
        "tokenizer" => Some("data/tokenizer.json".into()),
        "pretrain" => Some(format!("ckpt_{name}.pt")),
        "eval" => Some(format!("data/eval/hard_ckpt_{name}.pt.jsonl")),
        "sft" => Some(format!("ckpt_sft_{name}.pt")),
        "rl" => Some(format!("ckpt_rl_{name}.pt")),
        _ => None,
    }
}

/// The command(s) for one stage. Most stages are a single Step; `data` is two (overview + gate).
/// CPU-bound stages get `cpu_env()` so tokenization uses every core with no core-count flag.
fn steps_for(stage: &str, a: &Args) -> Result<Vec<Step>, String> {
    let name = run_name(a);
    let ckpt = format!("ckpt_{name}.pt");
    Ok(match stage {
        "tokenizer" => vec![Step::uv_py("scripts/build_tokenizer.py", &[]).with_env(cpu_env())],
        "pretokenize" => vec![Step::uv_py("scripts/pretokenize.py", &[]).with_env(cpu_env())],
        "data" => vec![
            Step::uv_py("scripts/data_overview.py", &[]).with_env(cpu_env()),
            Step::uv_py("scripts/check_mix.py", &[]),
        ],
        "pretrain" => {
            let (recipe_args, _) = recipe::build(&name, &a.profile, &a.passthrough);
            // recipe::build already prepends --name; train_step only appends --track/env.
            vec![train_step(a.track, None, recipe_args)]
        }
        "eval" => vec![Step::bash(
            "scripts/eval_hard.sh",
            &[ckpt, a.ngpu.to_string()],
        )],
        "sft" => {
            let sft_pt = a
                .sft_pt
                .clone()
                .ok_or("sft stage needs --sft-pt <packed .pt>")?;
            let sft_name = format!("sft_{name}");
            vec![Step::bash("scripts/run_sft.sh", &[sft_name, ckpt, sft_pt])]
        }
        "rl" => {
            let sft_ckpt = format!("ckpt_sft_{name}.pt");
            let data = a
                .rl_data
                .clone()
                .unwrap_or_else(|| "data/rl/rl_band.jsonl".into());
            let devs: Vec<String> = (0..a.ngpu).map(|i| i.to_string()).collect();
            vec![Step::new(
                "torchrun",
                vec![
                    format!("--nproc_per_node={}", a.ngpu),
                    "algorithms/rlvr.py".into(),
                    "--resume".into(),
                    sft_ckpt,
                    "--data".into(),
                    data,
                    "--out".into(),
                    format!("ckpt_rl_{name}.pt"),
                ],
            )
            .with_env(vec![("CUDA_VISIBLE_DEVICES".into(), devs.join(","))])]
        }
        _ => return Err(format!("unhandled stage '{stage}'")),
    })
}

// ---- state file ----

fn state_path(root: &Path, name: &str) -> PathBuf {
    root.join("runs").join(format!("{name}.pipeline.json"))
}

/// Load the state file as a map stage -> record, plus keep the top-level object for rewriting.
fn load_state(root: &Path, name: &str) -> Value {
    let p = state_path(root, name);
    fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({"name": name, "stages": {}}))
}

fn stage_status<'a>(state: &'a Value, stage: &str) -> Option<&'a str> {
    state.get("stages")?.get(stage)?.get("status")?.as_str()
}

/// Write the state file (atomic-ish: temp then rename). Never called under --dry-run.
fn save_state(root: &Path, name: &str, state: &Value) {
    let p = state_path(root, name);
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    let tmp = p.with_extension("json.tmp");
    if fs::write(
        &tmp,
        serde_json::to_string_pretty(state).unwrap_or_default(),
    )
    .is_ok()
    {
        let _ = fs::rename(&tmp, &p);
    }
}

/// Record a stage transition into the state object.
fn mark(
    state: &mut Value,
    stage: &str,
    status: &str,
    artifact: &Option<String>,
    stamp_start: bool,
) {
    let stages = state
        .get_mut("stages")
        .and_then(|s| s.as_object_mut())
        .expect("stages object");
    let entry = stages.entry(stage.to_string()).or_insert_with(|| json!({}));
    let now = fmt_utc(now_secs());
    entry["status"] = json!(status);
    if stamp_start {
        entry["start"] = json!(now);
    }
    if status == "done" || status == "failed" {
        entry["end"] = json!(now);
    }
    if let Some(art) = artifact {
        entry["artifact"] = json!(art);
    }
}

/// True when the resume logic should skip this stage: it is marked done, or its artifact is on disk.
fn should_skip(root: &Path, state: &Value, stage: &str, name: &str) -> Option<&'static str> {
    if stage_status(state, stage) == Some("done") {
        return Some("state=done");
    }
    if let Some(art) = artifact_for(stage, name) {
        if root.join(&art).exists() {
            return Some("artifact exists");
        }
    }
    None
}

// ---- status view ----

fn show_status(root: &Path, name: &str, json_out: bool) -> i32 {
    let p = state_path(root, name);
    if !p.exists() {
        if json_out {
            println!("{}", json!({"name": name, "exists": false, "stages": []}));
        } else {
            println!("aupai pipeline: no state for '{name}' (no runs/{name}.pipeline.json) — nothing has run yet.");
        }
        return 0;
    }
    let state = load_state(root, name);
    let stages_obj = state.get("stages").and_then(|s| s.as_object());
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&state).unwrap_or_default()
        );
        return 0;
    }
    println!("aupai pipeline '{name}' — {}\n", p.display());
    println!(
        "{:<12}{:<10}{:<40}{}",
        "stage", "status", "artifact", "when"
    );
    for stage in STAGES {
        if let Some(rec) = stages_obj.and_then(|o| o.get(*stage)) {
            let st = rec
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("pending");
            let art = rec.get("artifact").and_then(|v| v.as_str()).unwrap_or("");
            let when = rec
                .get("end")
                .or_else(|| rec.get("start"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            println!("{:<12}{:<10}{:<40}{}", stage, st, art, when);
        }
    }
    0
}

pub fn run(root: &Path, dry: bool, a: Args) -> i32 {
    let name = run_name(&a);

    // --status: print and exit, run nothing.
    if a.status.is_some() {
        return show_status(root, &name, a.json);
    }

    let stages = match resolve(&a) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("aupai pipeline: {e}");
            return 2;
        }
    };
    if stages.is_empty() {
        eprintln!("aupai pipeline: no stages selected");
        return 2;
    }

    let resuming = a.resume.is_some() && !a.force;
    let mut state = load_state(root, &name);

    println!(
        "aupai pipeline '{}' — {} stage(s): {}{}\n",
        name,
        stages.len(),
        stages.join(" -> "),
        if a.force {
            "  [--force: rerun all]"
        } else if resuming {
            "  [--resume]"
        } else {
            ""
        }
    );

    for (n, stage) in stages.iter().enumerate() {
        let artifact = artifact_for(stage, &name);

        // Resume / dry-run skip decision.
        if !a.force {
            if let Some(reason) = should_skip(root, &state, stage, &name) {
                if resuming || dry {
                    println!("[{}/{}] {} — SKIP ({reason})", n + 1, stages.len(), stage);
                    if resuming && !dry {
                        // ensure state reflects the skip as done (artifact-exists case)
                        mark(&mut state, stage, "done", &artifact, false);
                        save_state(root, &name, &state);
                    }
                    continue;
                }
            }
        }

        let steps = match steps_for(stage, &a) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("aupai pipeline: {e}");
                return 2;
            }
        };

        // pretrain: show the resolved-config block so the recipe is visible before launch.
        if stage == "pretrain" {
            let (_, shown) = recipe::build(&name, &a.profile, &a.passthrough);
            recipe::print_block(&a.profile, &shown);
        }

        if !dry {
            mark(&mut state, stage, "running", &artifact, true);
            save_state(root, &name, &state);
        }

        for step in &steps {
            println!("========================================================================");
            println!("[{}/{}] {}", n + 1, stages.len(), stage);
            println!("  {}", step.render(root));
            println!("========================================================================");
            if dry {
                continue;
            }
            let code = run_step(root, false, step);
            if code != 0 {
                mark(&mut state, stage, "failed", &artifact, false);
                save_state(root, &name, &state);
                eprintln!("\naupai pipeline: stage '{stage}' failed (exit {code}) — stopping. Resume with `aupai pipeline --resume {name}`.");
                return code;
            }
        }
        if !dry {
            mark(&mut state, stage, "done", &artifact, false);
            save_state(root, &name, &state);
        }
    }

    println!(
        "\naupai pipeline: {}",
        if dry {
            "plan complete (dry-run) — no files touched".to_string()
        } else {
            format!("all stages passed — state in runs/{name}.pipeline.json")
        }
    );
    0
}
