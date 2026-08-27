//! aupai — a thin harness over the project's Python/bash training tools.
//!
//! Every subcommand shells out to an existing script through `common::run`, inherits stdio, and
//! propagates the exit code. Nothing heavy is reimplemented in Rust: the harness sequences the tools
//! (notably `pipeline`, which chains a whole training run) and renders the plan under --dry-run.

mod common;
mod pipeline;

use std::path::Path;

use clap::{Parser, Subcommand};

use common::{run, Step};

#[derive(Parser)]
#[command(
    name = "aupai",
    about = "Unified harness for the aupai project tools",
    arg_required_else_help = true
)]
struct Cli {
    /// Print the exact command(s) instead of running
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    // ---- data / tokenization ----
    /// Corpus overview: per-domain tokens vs the mix target (scripts/data_overview.py)
    Data {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Validate data/mix.json: rows/phase, epoch caps, step count (scripts/check_mix.py)
    Mix {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Train data/tokenizer.json — ByteLevel BPE + the 4 chat specials (scripts/build_tokenizer.py)
    Tokenizer {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Warm every mix domain's token cache before training (scripts/pretokenize.py)
    Pretokenize {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    // ---- training ----
    /// DDP pretraining (run_ddp.sh; flags pass through to train.py). --track enables trackio logging
    Train {
        /// Log to trackio (sets TRACKIO_PROJECT env + passes --track to train.py)
        #[arg(long)]
        track: bool,
        /// trackio project name (implies --track; default: aupai)
        #[arg(long)]
        track_project: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Full training pipeline: tokenizer -> pretokenize -> data -> pretrain -> eval [-> sft -> rl]
    Pipeline(pipeline::Args),
    /// SFT experiment end to end (scripts/run_sft.sh)
    Sft {
        /// Experiment name (checkpoint becomes ckpt_<name>.pt)
        name: String,
        /// Checkpoint to resume from
        resume_ckpt: String,
        /// Packed SFT data (.pt)
        sft_pt: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// RLVR / GSPO trainer (torchrun algorithms/rlvr.py; needs --resume <sft ckpt>)
    Rl {
        /// GPUs (default 8)
        #[arg(long, default_value_t = 8)]
        ngpu: u8,
        /// master port
        #[arg(long, default_value_t = 29662)]
        port: u16,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    // ---- SFT/RL data prep ----
    /// Pack instruction-tuning data into a .pt (scripts/prepare_sft.py)
    PrepSft {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Pack math SFT data into a .pt (prepare_sft_math.py)
    PrepSftMath {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Prepare RLVR prompt data (algorithms/prepare_rlvr.py)
    PrepRl {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    // ---- eval / analysis ----
    /// Sharded math-hard eval — the metric of record (scripts/eval_hard.sh)
    Eval {
        /// Checkpoint path
        ckpt: String,
        /// Number of GPUs (default: script default)
        ngpu: Option<u8>,
    },
    /// Sharded math-500 eval (scripts/eval_math.sh; saturated — never conclude from it alone)
    EvalMath { ckpt: String, ngpu: Option<u8> },
    /// Probe per-instance solve rate and keep the 20-80% band (scripts/probe_band.sh)
    Band { ckpt: String, ngpu: Option<u8> },
    /// Select the RL difficulty band from a rates file (scripts/select_band.py)
    SelectBand {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Plot training curves from runs/*.log -> plots/<name>.png (scripts/plot_curves.py)
    Plot {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Open a training dashboard: trackio's local web UI, or plot the run's curves (--plot)
    Dashboard {
        /// Run name (runs/<name>.log, plots/<name>.png) — used by --plot
        name: Option<String>,
        /// trackio project to show (default: the TRACKIO_PROJECT env, else all)
        #[arg(long)]
        project: Option<String>,
        /// Plot the run's curves to a PNG instead of launching the trackio UI
        #[arg(long)]
        plot: bool,
    },
    /// FP8 NaN probe (scripts/nan_probe.py; env like COMPILE/BS/MUON/STEPS pass through)
    NanProbe {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Diff two checkpoints (scripts/ckpt_diff.py)
    CkptDiff {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    // ---- inference ----
    /// Local one-shot / REPL inference (infer_local.py)
    Infer {
        /// Prompt text (omit for interactive REPL)
        prompt: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Chat with a checkpoint (chat.py)
    Chat {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Serve a checkpoint over HTTP (serve.py; --port default 8080)
    Serve {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },

    // ---- ops ----
    /// Experiment log (scripts/exp.py)
    Exp {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Open arch.html in the browser
    Arch,
    /// List all commands
    List,
}

/// (invocation, description) for `aupai list`, grouped by the section comment above it.
const COMMANDS: &[(&str, &str)] = &[
    (
        "data [args]",
        "Corpus overview vs the mix target (scripts/data_overview.py)",
    ),
    (
        "mix [args]",
        "Validate data/mix.json (scripts/check_mix.py)",
    ),
    (
        "tokenizer [--force --sample-tokens N]",
        "Train data/tokenizer.json + the 4 chat specials (scripts/build_tokenizer.py)",
    ),
    (
        "pretokenize [--domains a,b]",
        "Warm every mix domain's token cache (scripts/pretokenize.py)",
    ),
    (
        "train [--track] [train.py flags]",
        "DDP pretraining (run_ddp.sh -> train.py)",
    ),
    (
        "pipeline [--from --to --stages] [flags]",
        "Full run: tokenizer->pretokenize->data->pretrain->eval[->sft->rl]",
    ),
    (
        "sft <name> <resume_ckpt> <sft_pt> [args]",
        "SFT experiment end to end (scripts/run_sft.sh)",
    ),
    (
        "rl [--ngpu --port] --resume <ckpt> [args]",
        "RLVR/GSPO trainer (torchrun algorithms/rlvr.py)",
    ),
    (
        "prep-sft [args]",
        "Pack instruction SFT data (scripts/prepare_sft.py)",
    ),
    (
        "prep-sft-math [args]",
        "Pack math SFT data (prepare_sft_math.py)",
    ),
    (
        "prep-rl [args]",
        "Prepare RLVR prompt data (algorithms/prepare_rlvr.py)",
    ),
    (
        "eval <ckpt> [ngpu]",
        "Math-hard eval, the metric of record (scripts/eval_hard.sh)",
    ),
    (
        "eval-math <ckpt> [ngpu]",
        "Math-500 eval (scripts/eval_math.sh)",
    ),
    (
        "band <ckpt> [ngpu]",
        "Probe the 20-80% RL difficulty band (scripts/probe_band.sh)",
    ),
    (
        "select-band <out> [args]",
        "Pick the RL band from a rates file (scripts/select_band.py)",
    ),
    (
        "plot [logs...]",
        "Plot training curves -> plots/<name>.png (scripts/plot_curves.py)",
    ),
    (
        "dashboard [name] [--project N] [--plot]",
        "Launch trackio's local UI, or plot the run's curves (--plot)",
    ),
    ("nan-probe [args]", "FP8 NaN probe (scripts/nan_probe.py)"),
    (
        "ckpt-diff [args]",
        "Diff two checkpoints (scripts/ckpt_diff.py)",
    ),
    ("infer [prompt] [args]", "Local inference (infer_local.py)"),
    ("chat [args]", "Chat with a checkpoint (chat.py)"),
    ("serve [args]", "Serve a checkpoint over HTTP (serve.py)"),
    ("exp [args]", "Experiment log (scripts/exp.py)"),
    ("arch", "Open arch.html in the browser"),
    ("list", "Show this table"),
];

fn main() {
    let cli = Cli::parse();
    let root = common::repo_root();
    let dry = cli.dry_run;

    let code = match cli.cmd {
        Cmd::Data { args } => run(&root, dry, &Step::uv_py("scripts/data_overview.py", &args)),
        Cmd::Mix { args } => run(&root, dry, &Step::uv_py("scripts/check_mix.py", &args)),
        Cmd::Tokenizer { args } => run(
            &root,
            dry,
            &Step::uv_py("scripts/build_tokenizer.py", &args),
        ),
        Cmd::Pretokenize { args } => run(&root, dry, &Step::uv_py("scripts/pretokenize.py", &args)),

        Cmd::Train {
            track,
            track_project,
            args,
        } => run(&root, dry, &train_step(track, track_project, args)),
        Cmd::Pipeline(args) => pipeline::run(&root, dry, args),
        Cmd::Sft {
            name,
            resume_ckpt,
            sft_pt,
            args,
        } => {
            let mut full = vec![name, resume_ckpt, sft_pt];
            full.extend(args);
            run(&root, dry, &Step::bash("scripts/run_sft.sh", &full))
        }
        Cmd::Rl { ngpu, port, args } => run(&root, dry, &rl_step(ngpu, port, &args)),

        Cmd::PrepSft { args } => run(&root, dry, &Step::uv_py("scripts/prepare_sft.py", &args)),
        Cmd::PrepSftMath { args } => run(&root, dry, &Step::uv_py("prepare_sft_math.py", &args)),
        Cmd::PrepRl { args } => run(
            &root,
            dry,
            &Step::uv_py("algorithms/prepare_rlvr.py", &args),
        ),

        Cmd::Eval { ckpt, ngpu } => run(&root, dry, &eval_step("scripts/eval_hard.sh", ckpt, ngpu)),
        Cmd::EvalMath { ckpt, ngpu } => {
            run(&root, dry, &eval_step("scripts/eval_math.sh", ckpt, ngpu))
        }
        Cmd::Band { ckpt, ngpu } => {
            run(&root, dry, &eval_step("scripts/probe_band.sh", ckpt, ngpu))
        }
        Cmd::SelectBand { args } => run(&root, dry, &Step::uv_py("scripts/select_band.py", &args)),
        Cmd::Plot { args } => run(&root, dry, &Step::uv_py("scripts/plot_curves.py", &args)),
        Cmd::Dashboard {
            name,
            project,
            plot,
        } => dashboard(&root, dry, name, project, plot),
        Cmd::NanProbe { args } => run(&root, dry, &Step::uv_py("scripts/nan_probe.py", &args)),
        Cmd::CkptDiff { args } => run(&root, dry, &Step::uv_py("scripts/ckpt_diff.py", &args)),

        Cmd::Infer { prompt, args } => {
            let mut full = Vec::new();
            if let Some(p) = prompt {
                full.push(p);
            }
            full.extend(args);
            run(&root, dry, &Step::uv_py("infer_local.py", &full))
        }
        Cmd::Chat { args } => run(&root, dry, &Step::uv_py("chat.py", &args)),
        Cmd::Serve { args } => run(&root, dry, &Step::uv_py("serve.py", &args)),

        Cmd::Exp { args } => run(&root, dry, &Step::uv_py("scripts/exp.py", &args)),
        Cmd::Arch => {
            let opener = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            run(
                &root,
                dry,
                &Step::new(opener, vec![root.join("arch.html").display().to_string()]),
            )
        }
        Cmd::List => {
            list();
            0
        }
    };
    std::process::exit(code);
}

/// `bash run_ddp.sh [--track] <flags>`, with TRACKIO_PROJECT env when enabled.
pub fn train_step(track: bool, project: Option<String>, mut args: Vec<String>) -> Step {
    let on = track || project.is_some();
    let mut env = Vec::new();
    if on {
        if !args.iter().any(|a| a == "--track") {
            args.push("--track".into());
        }
        env.push((
            "TRACKIO_PROJECT".into(),
            project.unwrap_or_else(|| "aupai".into()),
        ));
    }
    Step::bash("run_ddp.sh", &args).with_env(env)
}

/// `CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=N --master_port=P algorithms/rlvr.py <args>`.
fn rl_step(ngpu: u8, port: u16, args: &[String]) -> Step {
    let mut full = vec![
        format!("--nproc_per_node={ngpu}"),
        format!("--master_port={port}"),
        "algorithms/rlvr.py".into(),
    ];
    full.extend_from_slice(args);
    let devs: Vec<String> = (0..ngpu).map(|i| i.to_string()).collect();
    Step::new("torchrun", full).with_env(vec![("CUDA_VISIBLE_DEVICES".into(), devs.join(","))])
}

/// `bash <script> <ckpt> [ngpu]`.
fn eval_step(script: &str, ckpt: String, ngpu: Option<u8>) -> Step {
    let mut args = vec![ckpt];
    if let Some(n) = ngpu {
        args.push(n.to_string());
    }
    Step::bash(script, &args)
}

/// Launch trackio's local web UI, or fall back to plotting the run's curves locally (--plot).
fn dashboard(
    root: &Path,
    dry: bool,
    name: Option<String>,
    project: Option<String>,
    plot: bool,
) -> i32 {
    if !plot {
        // trackio is local: `uv run python -m trackio show [--project <name>]` starts its Gradio UI.
        let mut args = vec![
            "run".into(),
            "python".into(),
            "-m".into(),
            "trackio".into(),
            "show".into(),
        ];
        let proj = project.or_else(|| std::env::var("TRACKIO_PROJECT").ok());
        if let Some(p) = proj {
            args.push("--project".into());
            args.push(p);
        }
        return run(root, dry, &Step::new("uv", args));
    }
    // --plot: plot the run's curves, then open the PNG.
    let name = name.unwrap_or_else(|| "pretrain".into());
    let log = format!("runs/{name}.log");
    let code = run(root, dry, &Step::uv_py("scripts/plot_curves.py", &[log]));
    if code != 0 && !dry {
        return code;
    }
    let png = root.join("plots").join(format!("{name}.png"));
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    run(
        root,
        dry,
        &Step::new(opener, vec![png.display().to_string()]),
    )
}

fn list() {
    println!("aupai <command> [args] — thin harness over the project's Python/bash tools");
    println!(
        "(every command shells out to an existing script; --dry-run prints the exact command)\n"
    );
    for (name, desc) in COMMANDS {
        println!("  {:<42} {}", name, desc);
    }
    println!("\npipeline stages: tokenizer, pretokenize, data, pretrain, eval, sft, rl");
    println!(
        "  aupai pipeline --dry-run --name mybase --fp8   # the whole training flow at a glance"
    );
}
