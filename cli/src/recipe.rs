//! The verified-best pretrain recipe, made the default so zero-input `aupai train --name X` runs a
//! correct, fully-visible configuration. Source of truth: the `ckpt_k4_11b_lr05` run — the only run
//! to hit 51.6% math-500 — whose cfg line in `runs/k4_11b_lr05.log` reads
//! `fp8 True attn_res True/4 ... warmup 150 ... lr_scale 0.5` (see EXPERIMENTS.md, 2026-08-26).
//! Any flag the user passes through overrides the recipe default (train.py argparse takes the last).

/// One recipe flag: a store_true switch (`value: None`) or a valued flag (`--warmup 150`).
pub struct Flag {
    pub name: &'static str,
    pub value: Option<&'static str>,
}

/// The `best` profile = the ckpt_k4_11b_lr05 recipe.
pub const BEST: &[Flag] = &[
    Flag {
        name: "--fp8",
        value: None,
    },
    Flag {
        name: "--attn_res",
        value: None,
    },
    Flag {
        name: "--attn_res_blocks",
        value: Some("4"),
    },
    Flag {
        name: "--warmup",
        value: Some("150"),
    },
    Flag {
        name: "--lr_scale",
        value: Some("0.5"),
    },
];

/// Provenance shown under the resolved-config block so the recipe is never a magic constant.
pub const PROVENANCE: &str =
    "profile 'best' = ckpt_k4_11b_lr05 recipe (runs/k4_11b_lr05.log cfg line; EXPERIMENTS.md 2026-08-26)";

/// The recipe flags for a profile name. `base` is the bare train.py default (nothing added).
fn flags_for(profile: &str) -> &'static [Flag] {
    match profile {
        "base" | "bare" | "none" => &[],
        _ => BEST, // "best" and anything else default to the verified recipe
    }
}

/// Does the user's passthrough already set `flag`?
fn user_has(user: &[String], flag: &str) -> bool {
    user.iter().any(|a| a == flag)
}

/// One line of the resolved-config display: `flag value  (source)`.
pub struct Resolved {
    pub flag: String,
    pub value: String,
    pub source: String,
}

/// Build the effective train.py argv (recipe defaults the user didn't override, then the user's own
/// flags) plus the resolved-config lines to print. `--name` is always first and always shown.
pub fn build(name: &str, profile: &str, user: &[String]) -> (Vec<String>, Vec<Resolved>) {
    let mut argv: Vec<String> = vec!["--name".into(), name.into()];
    let mut shown = vec![Resolved {
        flag: "--name".into(),
        value: name.into(),
        source: "arg".into(),
    }];
    for f in flags_for(profile) {
        if user_has(user, f.name) {
            // user overrides this recipe default — the user's own copy is appended below
            shown.push(Resolved {
                flag: f.name.into(),
                value: f.value.unwrap_or("(set)").into(),
                source: "overridden by you".into(),
            });
            continue;
        }
        argv.push(f.name.into());
        if let Some(v) = f.value {
            argv.push(v.into());
        }
        shown.push(Resolved {
            flag: f.name.into(),
            value: f.value.unwrap_or("(on)").into(),
            source: format!("default: {profile}"),
        });
    }
    // The user's passthrough goes last so train.py's argparse takes it over any recipe default.
    for a in user {
        argv.push(a.clone());
    }
    // Surface user flags that aren't part of the recipe so the block shows the full picture.
    let recipe_names: Vec<&str> = flags_for(profile).iter().map(|f| f.name).collect();
    let mut i = 0;
    while i < user.len() {
        let a = &user[i];
        if a.starts_with("--") && !recipe_names.contains(&a.as_str()) {
            let val = user.get(i + 1).filter(|n| !n.starts_with("--"));
            shown.push(Resolved {
                flag: a.clone(),
                value: val.cloned().unwrap_or_else(|| "(on)".into()),
                source: "user".into(),
            });
        }
        i += 1;
    }
    (argv, shown)
}

/// Print the resolved-config block (both under --dry-run and before a real launch).
pub fn print_block(profile: &str, shown: &[Resolved]) {
    println!("resolved config (profile={profile}):");
    for r in shown {
        println!("  {:<20}{:<10}{}", r.flag, r.value, r.source);
    }
    println!("  ({PROVENANCE})");
}
