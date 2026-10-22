//! Where the client finds its assets, models and saves.

/// Root directory for assets, models and saves.
///
/// Resolution order — this is what makes a *shipped* build work, because
/// `env!("CARGO_MANIFEST_DIR")` is baked in at compile time and does not exist
/// on a player's machine:
///
/// 1. `$IRL_HOME`, if set (explicit override, also used by tests);
/// 2. the directory containing the executable, when it looks like a portable
///    install (i.e. it has `assets/speech/manifest.json`);
/// 3. the Cargo workspace root baked in at compile time (development runs);
/// 4. the executable directory, else the current directory.
pub fn home_root() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("IRL_HOME") {
        return std::path::PathBuf::from(dir);
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));
    if let Some(dir) = &exe_dir {
        if dir.join("assets/speech/manifest.json").is_file() {
            return dir.clone();
        }
    }
    let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf);
    if let Some(dir) = &dev {
        if dir.join("assets/speech/manifest.json").is_file() {
            return dir.clone();
        }
    }
    exe_dir.unwrap_or_else(|| std::path::PathBuf::from("."))
}

pub fn workspace_cache() -> std::path::PathBuf {
    home_root().join(".cache/speech")
}

pub fn workspace_manifest() -> std::path::PathBuf {
    home_root().join("assets/speech/manifest.json")
}

/// The first `.gguf` under `<home>/.cache/models`, so a portable bundle's
/// executable finds its model with no arguments.
pub fn bundled_model() -> Option<String> {
    let dir = home_root().join(".cache/models");
    let mut models: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    models.sort();
    models
        .into_iter()
        .next()
        .map(|p| p.to_string_lossy().into_owned())
}

/// Default save slot used by the graphical client's quick-save keys.
///
/// Lives under [`home_root`] so a portable install keeps saves next to itself
/// and the text client can share a save produced by the same simulation.
pub fn default_save_path() -> String {
    home_root()
        .join("irl_save.json")
        .to_string_lossy()
        .into_owned()
}

/// Human-readable report of where the executable looks for its files, printed
/// by `bevy-game --doctor`. Useful for support and to verify that a portable
/// bundle resolves its own assets regardless of the working directory.
pub fn doctor_report() -> String {
    let home = home_root();
    let manifest = workspace_manifest();
    let cache = workspace_cache();
    let model = bundled_model();
    let found = |ok: bool| if ok { "found" } else { "MISSING" };
    format!(
        "IRL asset resolution\n\
         home           : {}\n\
         speech manifest: {} [{}]\n\
         speech cache   : {} [{}]\n\
         bundled model  : {} [{}]\n\
         save file      : {}\n\
         (set IRL_HOME to override the home directory)",
        home.display(),
        manifest.display(),
        found(manifest.is_file()),
        cache.display(),
        found(cache.is_dir()),
        model.as_deref().unwrap_or("<none>"),
        found(model.is_some()),
        default_save_path(),
    )
}
