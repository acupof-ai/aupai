//! aupai — a thin harness over the project's Python/bash training tools.
//!
//! Every subcommand shells out to an existing script through `common::run`, inherits stdio, and
//! propagates the exit code. Nothing heavy is reimplemented in Rust: the harness sequences the tools
//! (notably `pipeline`, which chains a whole training run and persists per-stage state), renders the
//! plan under --dry-run, resolves the best-known recipe as the default, and surfaces system state
//! (`status`, `ckpt`). Trailing backend flags are collected only after a literal `--` (clap
//! `last = true`), so a stray `--dry-run`/`--name` can never be swallowed into a real launch.

mod ckpt;
mod common;
mod pipeline;
mod recipe;
mod status;

use std::path::Path;

use clap::{Parser, Subcommand};

use common::{cpu_env, run, Step};

#[derive(Parser)]
#[command(
    name = "aupai",
    about = "Unified harness for the aupai project tools",
    arg_required_else_help = true
)]
struct Cli {
    /// Print the exact command(s) instead of running (place BEFORE the subcommand)
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    // ---- data / tokenization ----
    /// Corpus overview: per-domain tokens vs the mix target (scripts/data_overview.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/data_overview.py; run `uv run python scripts/data_overview.py --help` for its flags."
    )]
    Data {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Validate data/mix.json: rows/phase, epoch caps, step count (scripts/check_mix.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/check_mix.py; run `uv run python scripts/check_mix.py --help` for its flags."
    )]
    Mix {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Train data/tokenizer.json — ByteLevel BPE + the 4 chat specials (scripts/build_tokenizer.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/build_tokenizer.py; run `uv run python scripts/build_tokenizer.py --help` for its flags."
    )]
    Tokenizer {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Warm every mix domain's token cache before training (scripts/pretokenize.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/pretokenize.py; run `uv run python scripts/pretokenize.py --help` for its flags. Uses all CPU cores automatically."
    )]
    Pretokenize {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },

    // ---- training ----
    /// DDP pretraining (run_ddp.sh -> train.py). Defaults to the verified-best recipe; prints it.
    #[command(
        after_help = "Zero-input runs the verified 'best' recipe (profile=best). Extra backend flags go AFTER `--`, e.g. `aupai train --name k5 -- --grad_ckpt --max_steps 500`; they override the recipe. Run `uv run python train.py --help` for train.py's flags."
    )]
    Train {
        /// Run name -> runs/<name>.log, ckpt_<name>.pt
        #[arg(long, default_value = "pretrain")]
        name: String,
        /// Recipe profile: 'best' (verified ckpt_k4_11b_lr05 recipe) or 'base' (bare train.py)
        #[arg(long, default_value = "best")]
        profile: String,
        /// Log to trackio (sets TRACKIO_PROJECT env + passes --track to train.py)
        #[arg(long)]
        track: bool,
        /// trackio project name (implies --track; default: aupai)
        #[arg(long)]
        track_project: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Full training pipeline: tokenizer -> pretokenize -> data -> pretrain -> eval [-> sft -> rl]
    Pipeline(pipeline::Args),
    /// SFT experiment end to end (scripts/run_sft.sh)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to sft_math.py (via scripts/run_sft.sh); run `uv run python sft_math.py --help` for its flags."
    )]
    Sft {
        /// Experiment name (checkpoint becomes ckpt_<name>.pt)
        name: String,
        /// Checkpoint to resume from
        resume_ckpt: String,
        /// Packed SFT data (.pt)
        sft_pt: String,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// RLVR / GSPO trainer (torchrun algorithms/rlvr.py; needs `-- --resume <sft ckpt>`)
    #[command(
        after_help = "Backend flags go AFTER `--`, e.g. `aupai rl --ngpu 8 -- --resume ckpt_sft_k4.pt`. Run `uv run python algorithms/rlvr.py --help` for its flags."
    )]
    Rl {
        /// GPUs (default 8)
        #[arg(long, default_value_t = 8)]
        ngpu: u8,
        /// master port
        #[arg(long, default_value_t = 29662)]
        port: u16,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },

    // ---- SFT/RL data prep ----
    /// Pack instruction-tuning data into a .pt (scripts/prepare_sft.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/prepare_sft.py; run `uv run python scripts/prepare_sft.py --help` for its flags."
    )]
    PrepSft {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Pack math SFT data into a .pt (prepare_sft_math.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to prepare_sft_math.py; run `uv run python prepare_sft_math.py --help` for its flags (e.g. --sources a.jsonl,b.jsonl --out out.pt)."
    )]
    PrepSftMath {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Prepare RLVR prompt data (algorithms/prepare_rlvr.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to algorithms/prepare_rlvr.py; run `uv run python algorithms/prepare_rlvr.py --help` for its flags."
    )]
    PrepRl {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },

    // ---- eval / analysis ----
    /// Sharded math-hard eval — the metric of record (scripts/eval_hard.sh)
    #[command(
        after_help = "Shells out to scripts/eval_hard.sh <ckpt> [ngpu]. Pre-flight checks the checkpoint exists."
    )]
    Eval {
        /// Checkpoint path
        ckpt: String,
        /// Number of GPUs (default: script default)
        #[arg(long)]
        ngpu: Option<u8>,
    },
    /// Sharded math-500 eval (scripts/eval_math.sh; saturated — never conclude from it alone)
    #[command(after_help = "Shells out to scripts/eval_math.sh <ckpt> [ngpu].")]
    EvalMath {
        ckpt: String,
        #[arg(long)]
        ngpu: Option<u8>,
    },
    /// Probe per-instance solve rate and keep the 20-80% band (scripts/probe_band.sh)
    #[command(after_help = "Shells out to scripts/probe_band.sh <ckpt> [ngpu].")]
    Band {
        ckpt: String,
        #[arg(long)]
        ngpu: Option<u8>,
    },
    /// Select the RL difficulty band from a rates file (scripts/select_band.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/select_band.py; run `uv run python scripts/select_band.py --help` for its flags."
    )]
    SelectBand {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Plot training curves from runs/*.log -> plots/<name>.png (scripts/plot_curves.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/plot_curves.py; run `uv run python scripts/plot_curves.py --help` for its flags."
    )]
    Plot {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
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
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/nan_probe.py; run `uv run python scripts/nan_probe.py --help` for its flags."
    )]
    NanProbe {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Diff two checkpoints (scripts/ckpt_diff.py)
    #[command(
        after_help = "Extra ARGS (after `--`) are checkpoint paths passed verbatim to scripts/ckpt_diff.py. Pre-flight checks each exists. e.g. `aupai ckpt-diff -- ckpt_a.pt ckpt_b.pt`."
    )]
    CkptDiff {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },

    // ---- checkpoints ----
    /// Manage checkpoints: list / best / clean (see `aupai ckpt --help`)
    Ckpt {
        #[command(subcommand)]
        cmd: ckpt::Cmd,
    },

    // ---- inference ----
    /// Local one-shot / REPL inference (infer_local.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to infer_local.py; run `uv run python infer_local.py --help` for its flags (e.g. --ckpt X). Pre-flight checks --ckpt if given."
    )]
    Infer {
        /// Prompt text (omit for interactive REPL)
        prompt: Option<String>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Chat with a checkpoint (chat.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to chat.py; run `uv run python chat.py --help` for its flags."
    )]
    Chat {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },
    /// Serve a checkpoint over HTTP (serve.py; --port default 8080)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to serve.py; run `uv run python serve.py --help` for its flags."
    )]
    Serve {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
        args: Vec<String>,
    },

    // ---- ops ----
    /// System dashboard: data readiness, checkpoints, pipelines, active run (read-only)
    Status {
        /// Machine-readable JSON
        #[arg(long)]
        json: bool,
    },
    /// Experiment log (scripts/exp.py)
    #[command(
        after_help = "Extra ARGS (after `--`) pass verbatim to scripts/exp.py; run `uv run python scripts/exp.py --help` for its flags."
    )]
    Exp {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, last = true)]
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
        "data [-- args]",
        "Corpus overview vs the mix target (scripts/data_overview.py)",
    ),
    (
        "mix [-- args]",
        "Validate data/mix.json (scripts/check_mix.py)",
    ),
    (
        "tokenizer [-- --force ...]",
        "Train data/tokenizer.json + the 4 chat specials (scripts/build_tokenizer.py)",
    ),
    (
        "pretokenize [-- --domains a,b]",
        "Warm every mix domain's token cache, all cores (scripts/pretokenize.py)",
    ),
    (
        "train [--name --profile --track] [-- train.py flags]",
        "DDP pretraining, best recipe by default (run_ddp.sh -> train.py)",
    ),
    (
        "pipeline [--from --to --stages] [--status --resume --force]",
        "Full run with per-stage state: tokenizer->...->eval[->sft->rl]",
    ),
    (
        "sft <name> <resume_ckpt> <sft_pt> [-- args]",
        "SFT experiment end to end (scripts/run_sft.sh)",
    ),
    (
        "rl [--ngpu --port] [-- --resume <ckpt>]",
        "RLVR/GSPO trainer (torchrun algorithms/rlvr.py)",
    ),
    (
        "prep-sft [-- args]",
        "Pack instruction SFT data (scripts/prepare_sft.py)",
    ),
    (
        "prep-sft-math [-- args]",
        "Pack math SFT data (prepare_sft_math.py)",
    ),
    (
        "prep-rl [-- args]",
        "Prepare RLVR prompt data (algorithms/prepare_rlvr.py)",
    ),
    (
        "eval <ckpt> [--ngpu N]",
        "Math-hard eval, the metric of record (scripts/eval_hard.sh)",
    ),
    (
        "eval-math <ckpt> [--ngpu N]",
        "Math-500 eval (scripts/eval_math.sh)",
    ),
    (
        "band <ckpt> [--ngpu N]",
        "Probe the 20-80% RL difficulty band (scripts/probe_band.sh)",
    ),
    (
        "select-band [-- <out> args]",
        "Pick the RL band from a rates file (scripts/select_band.py)",
    ),
    (
        "plot [-- logs...]",
        "Plot training curves -> plots/<name>.png (scripts/plot_curves.py)",
    ),
    (
        "dashboard [name] [--project N] [--plot]",
        "Launch trackio's local UI, or plot the run's curves (--plot)",
    ),
    (
        "nan-probe [-- args]",
        "FP8 NaN probe (scripts/nan_probe.py)",
    ),
    (
        "ckpt-diff [-- a.pt b.pt]",
        "Diff two checkpoints (scripts/ckpt_diff.py)",
    ),
    (
        "ckpt list|best|clean [--json]",
        "List / rank / prune checkpoints (scripts/ckpt_info.py)",
    ),
    (
        "infer [prompt] [-- args]",
        "Local inference (infer_local.py)",
    ),
    ("chat [-- args]", "Chat with a checkpoint (chat.py)"),
    ("serve [-- args]", "Serve a checkpoint over HTTP (serve.py)"),
    (
        "status [--json]",
        "System dashboard: data, checkpoints, pipelines, active run",
    ),
    ("exp [-- args]", "Experiment log (scripts/exp.py)"),
    ("arch", "Open arch.html in the browser"),
    ("list", "Show this table"),
];

fn main() {
    let cli = Cli::parse();
    let root = common::repo_root();
    let dry = cli.dry_run;

    let code = match cli.cmd {
        Cmd::Data { args } => run(
            &root,
            dry,
            &Step::uv_py("scripts/data_overview.py", &args).with_env(cpu_env()),
        ),
        Cmd::Mix { args } => run(&root, dry, &Step::uv_py("scripts/check_mix.py", &args)),
        Cmd::Tokenizer { args } => run(
            &root,
            dry,
            &Step::uv_py("scripts/build_tokenizer.py", &args).with_env(cpu_env()),
        ),
        Cmd::Pretokenize { args } => run(
            &root,
            dry,
            &Step::uv_py("scripts/pretokenize.py", &args).with_env(cpu_env()),
        ),

        Cmd::Train {
            name,
            profile,
            track,
            track_project,
            args,
        } => {
            let (recipe_args, shown) = recipe::build(&name, &profile, &args);
            recipe::print_block(&profile, &shown);
            println!();
            run(&root, dry, &train_step(track, track_project, recipe_args))
        }
        Cmd::Pipeline(args) => pipeline::run(&root, dry, args),
        Cmd::Sft {
            name,
            resume_ckpt,
            sft_pt,
            args,
        } => {
            if let Some(rc) = preflight(
                &root,
                dry,
                "sft",
                &[(Kind::Ckpt, &resume_ckpt), (Kind::Data, &sft_pt)],
            ) {
                return_exit(rc);
            }
            let mut full = vec![name, resume_ckpt, sft_pt];
            full.extend(args);
            run(&root, dry, &Step::bash("scripts/run_sft.sh", &full))
        }
        Cmd::Rl { ngpu, port, args } => {
            if let Some(path) = flag_value(&args, "--resume") {
                if let Some(rc) = preflight(&root, dry, "rl", &[(Kind::Ckpt, path)]) {
                    return_exit(rc);
                }
            }
            run(&root, dry, &rl_step(ngpu, port, &args))
        }

        Cmd::PrepSft { args } => run(&root, dry, &Step::uv_py("scripts/prepare_sft.py", &args)),
        Cmd::PrepSftMath { args } => {
            // --sources is a comma-separated list of input jsonl files; check each if present.
            if let Some(srcs) = flag_value(&args, "--sources") {
                let checks: Vec<(Kind, &str)> = srcs.split(',').map(|p| (Kind::Data, p)).collect();
                if let Some(rc) = preflight(&root, dry, "prep-sft-math", &checks) {
                    return_exit(rc);
                }
            }
            run(&root, dry, &Step::uv_py("prepare_sft_math.py", &args))
        }
        Cmd::PrepRl { args } => run(
            &root,
            dry,
            &Step::uv_py("algorithms/prepare_rlvr.py", &args),
        ),

        Cmd::Eval { ckpt, ngpu } => {
            if let Some(rc) = preflight(&root, dry, "eval", &[(Kind::Ckpt, &ckpt)]) {
                return_exit(rc);
            }
            run(&root, dry, &eval_step("scripts/eval_hard.sh", ckpt, ngpu))
        }
        Cmd::EvalMath { ckpt, ngpu } => {
            if let Some(rc) = preflight(&root, dry, "eval-math", &[(Kind::Ckpt, &ckpt)]) {
                return_exit(rc);
            }
            run(&root, dry, &eval_step("scripts/eval_math.sh", ckpt, ngpu))
        }
        Cmd::Band { ckpt, ngpu } => {
            if let Some(rc) = preflight(&root, dry, "band", &[(Kind::Ckpt, &ckpt)]) {
                return_exit(rc);
            }
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
        Cmd::CkptDiff { args } => {
            // all args are checkpoint paths.
            let checks: Vec<(Kind, &str)> = args.iter().map(|p| (Kind::Ckpt, p.as_str())).collect();
            if let Some(rc) = preflight(&root, dry, "ckpt-diff", &checks) {
                return_exit(rc);
            }
            run(&root, dry, &Step::uv_py("scripts/ckpt_diff.py", &args))
        }

        Cmd::Ckpt { cmd } => ckpt::run(&root, cmd),

        Cmd::Infer { prompt, args } => {
            if let Some(path) = flag_value(&args, "--ckpt") {
                if let Some(rc) = preflight(&root, dry, "infer", &[(Kind::Ckpt, path)]) {
                    return_exit(rc);
                }
            }
            let mut full = Vec::new();
            if let Some(p) = prompt {
                full.push(p);
            }
            full.extend(args);
            run(&root, dry, &Step::uv_py("infer_local.py", &full))
        }
        Cmd::Chat { args } => run(&root, dry, &Step::uv_py("chat.py", &args)),
        Cmd::Serve { args } => run(&root, dry, &Step::uv_py("serve.py", &args)),

        Cmd::Status { json } => status::run(&root, json),
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

/// Whether a required file arg exists on disk. Two message flavors: checkpoint vs data file.
#[derive(Clone, Copy)]
enum Kind {
    Ckpt,
    Data,
}

/// Pre-flight: fail fast if a required checkpoint / data file doesn't exist, BEFORE spawning a tool
/// (an agent shouldn't wait minutes for a python traceback). Skipped entirely under --dry-run so a
/// dry-run never touches disk state and always prints + exits 0. Returns Some(2) on the first miss.
fn preflight(root: &Path, dry: bool, cmd: &str, files: &[(Kind, &str)]) -> Option<i32> {
    if dry {
        return None;
    }
    for (kind, path) in files {
        if path.is_empty() {
            continue;
        }
        if !root.join(path).exists() && !Path::new(path).exists() {
            let what = match kind {
                Kind::Ckpt => "checkpoint not found",
                Kind::Data => "data file not found",
            };
            eprintln!("aupai {cmd}: {what}: {path}");
            return Some(2);
        }
    }
    None
}

/// The value following `flag` in a passthrough arg list (e.g. `--resume ckpt.pt` -> "ckpt.pt").
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).map(|s| s.as_str())
}

/// Exit immediately with `code`. (Helper so match arms that pre-flight-fail can bail cleanly.)
fn return_exit(code: i32) -> ! {
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
        "(every command shells out to an existing script; --dry-run prints the exact command)"
    );
    println!("(backend flags go AFTER `--`; the best recipe is the default and printed before it runs)\n");
    for (name, desc) in COMMANDS {
        println!("  {:<48} {}", name, desc);
    }
    println!("\npipeline stages: tokenizer, pretokenize, data, pretrain, eval, sft, rl");
    println!(
        "  aupai --dry-run pipeline --name mybase          # the whole training flow at a glance"
    );
    println!("  aupai --dry-run train --name k5 -- --grad_ckpt  # resolved recipe + backend flags");
    println!(
        "  aupai status                                    # data / ckpts / pipelines / active run"
    );
}
