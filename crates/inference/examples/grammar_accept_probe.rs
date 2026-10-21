//! Feed a known-valid JSON string to a grammar sampler token by token and
//! report exactly where acceptance fails. This isolates grammar behaviour from
//! model sampling.

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".cache/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
    let backend = LlamaBackend::init().unwrap();
    let model = LlamaModel::load_from_file(&backend, &path, &LlamaModelParams::default()).unwrap();
    let grammar = llama_cpp_2::json_schema_to_grammar(rpg_inference::ACTION_SCHEMA).unwrap();

    let text = r#"{"action":"idle"}"#;
    let tokens = model.vocab().tokenize(text.as_bytes(), false, true);
    println!("text={text:?} tokens={tokens:?}");

    println!("-- standalone grammar sampler --");
    let mut standalone = LlamaSampler::grammar(&model, &grammar, "root").unwrap();
    for t in &tokens {
        let r = standalone.try_accept(*t);
        println!("  token={t:?} -> {r:?}");
        if r.is_err() {
            break;
        }
    }

    println!("-- chain [grammar, temp, top_p, dist] --");
    let g = LlamaSampler::grammar(&model, &grammar, "root").unwrap();
    let mut chain = LlamaSampler::chain_simple([
        g,
        LlamaSampler::temp(0.2),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::dist(7),
    ]);
    for t in &tokens {
        let piece = model.vocab().token_to_piece(*t, false, None);
        let r = chain.try_accept(*t);
        println!(
            "  token={t:?} piece={:?} -> {r:?}",
            String::from_utf8_lossy(&piece)
        );
        if r.is_err() {
            break;
        }
    }
}
