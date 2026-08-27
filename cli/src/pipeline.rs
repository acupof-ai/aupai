//! `aupai pipeline` — chain the training stages, each shelling out to its existing tool.
//!
//! Stages in order: tokenizer -> pretokenize -> data -> pretrain -> eval -> sft -> rl. Select a
//! contiguous slice with --from/--to, or an explicit subset with --stages. --dry-run prints the full
//! ordered plan (every underlying command) so `aupai pipeline --dry-run ...` is the single source of
//! truth for what a run executes. Stops on the first failing stage (set -e semantics).

use std::path::Path;

use clap::Args as ClapArgs;

use crate::common::{run as run_step, Step};
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
    /// Run name -> runs/<name>.log, ckpt_<name>.pt, and the eval target
    #[arg(long, default_value = "pretrain")]
    name: String,
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
    /// Extra flags forwarded to the pretrain stage (train.py), e.g. --fp8 --attn_res
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    passthrough: Vec<String>,
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

/// The command(s) for one stage. Most stages are a single Step; `data` is two (overview + gate).
fn steps_for(stage: &str, a: &Args) -> Result<Vec<Step>, String> {
    let ckpt = format!("ckpt_{}.pt", a.name);
    Ok(match stage {
        "tokenizer" => vec![Step::uv_py("scripts/build_tokenizer.py", &[])],
        "pretokenize" => vec![Step::uv_py("scripts/pretokenize.py", &[])],
        "data" => vec![
            Step::uv_py("scripts/data_overview.py", &[]),
            Step::uv_py("scripts/check_mix.py", &[]),
        ],
        "pretrain" => {
            let mut args = vec!["--name".into(), a.name.clone()];
            args.extend(a.passthrough.clone());
            vec![train_step(a.track, None, args)]
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
            let name = format!("sft_{}", a.name);
            vec![Step::bash("scripts/run_sft.sh", &[name, ckpt, sft_pt])]
        }
        "rl" => {
            let sft_ckpt = format!("ckpt_sft_{}.pt", a.name);
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
                    format!("ckpt_rl_{}.pt", a.name),
                ],
            )
            .with_env(vec![("CUDA_VISIBLE_DEVICES".into(), devs.join(","))])]
        }
        _ => return Err(format!("unhandled stage '{stage}'")),
    })
}

pub fn run(root: &Path, dry: bool, a: Args) -> i32 {
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

    println!(
        "aupai pipeline — {} stage(s): {}\n",
        stages.len(),
        stages.join(" -> ")
    );

    for (n, stage) in stages.iter().enumerate() {
        let steps = match steps_for(stage, &a) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("aupai pipeline: {e}");
                return 2;
            }
        };
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
                eprintln!("\naupai pipeline: stage '{stage}' failed (exit {code}) — stopping.");
                return code;
            }
        }
    }
    println!(
        "\naupai pipeline: {} — DONE",
        if dry {
            "plan complete (dry-run)"
        } else {
            "all stages passed"
        }
    );
    0
}
