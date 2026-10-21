//! Diagnostic: show exactly what `apply_chat_template` produces for the model.

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default()).unwrap();

    let tmpl = model.chat_template(None).unwrap();
    let raw = tmpl.to_str().unwrap_or("<non-utf8>");
    println!("template_len={}", raw.len());
    println!(
        "template_has_enable_thinking={}",
        raw.contains("enable_thinking")
    );
    println!("template_has_think_literal={}", raw.contains("think"));
    println!("template_starts={:?}", &raw[..raw.len().min(60)]);

    let sys = LlamaChatMessage::new(
        "system".into(),
        "You are Marta Voss, a guarded grocer.".into(),
    )
    .unwrap();
    let user = LlamaChatMessage::new("user".into(), "Show me your ledger.".into()).unwrap();

    for add_ass in [true, false] {
        let rendered = model.apply_chat_template(&tmpl, &[sys.clone(), user.clone()], add_ass);
        match rendered {
            Ok(p) => {
                let tail: String = p
                    .chars()
                    .rev()
                    .take(80)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                println!(
                    "add_ass={add_ass} len={} think_count={} tail_escaped={:?}",
                    p.len(),
                    p.matches("think").count(),
                    tail
                );
            }
            Err(e) => println!("add_ass={add_ass} ERROR {e:?}"),
        }
    }
}
