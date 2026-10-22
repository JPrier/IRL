// IRL browser host: thin platform I/O for the WebAssembly game.
//
// Everything here is plumbing the Rust runtime asks for — browser speech,
// IndexedDB, file download, OAuth PKCE — and it reports back through the
// exported `irl_host_event`. No gameplay rules, validation or world state
// live in JavaScript: the authoritative simulation is the Rust `SimCore`.
//
// Credentials: this file never stores, logs or forwards the OpenRouter key.
// The OAuth helper hands a freshly minted key straight to the caller.

const DB_NAME = "irl";
const DB_STORE = "saves";

let wasm = null; // the wasm-bindgen module, set by attach()

// An OAuth callback's authorization code is taken out of the address bar
// immediately, before anything else loads, and kept only in memory.
const oauthCallback = (() => {
  const params = new URLSearchParams(location.search);
  const code = params.get("code");
  if (!code) return null;
  const state = params.get("state");
  params.delete("code");
  params.delete("state");
  const rest = params.toString();
  history.replaceState(null, "", location.pathname + (rest ? "?" + rest : "") + location.hash);
  return { code, state };
})();
let recognition = null;
let recognitionConv = null;
let recognitionFinal = "";
let recognitionDone = false;

function emit(kind, conv, line, text) {
  if (wasm) wasm.irl_host_event(kind, conv ?? 0, line ?? 0, text ?? "");
}

function speechRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function openDb() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("IndexedDB is not available"));
      return;
    }
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error("IndexedDB open failed"));
    req.onblocked = () => reject(new Error("IndexedDB is blocked by another tab"));
  });
}

function b64url(bytes) {
  let s = "";
  for (const b of new Uint8Array(bytes)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

window.irlHost = {
  attach(module) {
    wasm = module;
  },

  fatal(message) {
    const el = document.getElementById("fatal");
    if (el) {
      el.querySelector(".msg").textContent = message;
      el.hidden = false;
    }
  },

  // --- speech ---------------------------------------------------------

  speechSupport() {
    return {
      recognition: !!speechRecognitionCtor(),
      synthesis: "speechSynthesis" in window && "SpeechSynthesisUtterance" in window,
    };
  },

  recognitionStart(conv) {
    const Ctor = speechRecognitionCtor();
    if (!Ctor) throw new Error("speech recognition is not supported by this browser");
    if (recognition) {
      try { recognition.abort(); } catch (_) { /* already stopped */ }
    }
    const r = new Ctor();
    r.lang = navigator.language || "en-US";
    r.interimResults = false;
    r.continuous = true;
    r.maxAlternatives = 1;
    recognition = r;
    recognitionConv = conv;
    recognitionFinal = "";
    recognitionDone = false;
    const finish = (kind, text) => {
      if (recognitionDone || recognition !== r) return;
      recognitionDone = true;
      emit(kind, conv, 0, text);
    };
    r.onresult = (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        if (ev.results[i].isFinal) recognitionFinal += ev.results[i][0].transcript + " ";
      }
    };
    r.onerror = (ev) => finish("recognition_error", ev.error || "error");
    r.onnomatch = () => finish("recognition_nomatch", "");
    r.onend = () => {
      const text = recognitionFinal.trim();
      finish(text ? "recognition_result" : "recognition_nomatch", text);
    };
    document.body.classList.add("recording");
    r.start();
  },

  recognitionStop(conv) {
    document.body.classList.remove("recording");
    if (recognition && recognitionConv === conv) recognition.stop();
  },

  recognitionAbort() {
    document.body.classList.remove("recording");
    if (recognition) {
      recognitionDone = true;
      try { recognition.abort(); } catch (_) { /* already stopped */ }
    }
  },

  speak(conv, line, speaker, text) {
    const u = new SpeechSynthesisUtterance(text);
    const voices = speechSynthesis.getVoices().filter((v) => v.lang && v.lang.startsWith("en"));
    if (voices.length) {
      // A stable voice per speaker, so the cast stays distinguishable.
      let h = 0;
      for (const c of speaker) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      u.voice = voices[h % voices.length];
    }
    u.onstart = () => emit("tts_start", conv, line, "");
    u.onend = () => emit("tts_end", conv, line, "");
    u.onerror = (ev) => emit("tts_error", conv, line, ev.error || "error");
    speechSynthesis.speak(u);
  },

  stopSpeaking() {
    if ("speechSynthesis" in window) speechSynthesis.cancel();
  },

  // --- storage --------------------------------------------------------

  async storagePut(slot, json) {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(json, slot);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error("save failed"));
      tx.onabort = () => reject(tx.error || new Error("save aborted (storage full?)"));
    });
    db.close();
  },

  async storageGet(slot) {
    const db = await openDb();
    const value = await new Promise((resolve, reject) => {
      const req = db.transaction(DB_STORE, "readonly").objectStore(DB_STORE).get(slot);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error || new Error("load failed"));
    });
    db.close();
    return typeof value === "string" ? value : null;
  },

  downloadText(filename, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  },

  // --- OpenRouter OAuth (PKCE, S256) ----------------------------------

  async oauthStart() {
    const verifier = b64url(crypto.getRandomValues(new Uint8Array(48)));
    const challenge = b64url(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    );
    const state = b64url(crypto.getRandomValues(new Uint8Array(16)));
    // The verifier must survive the redirect; it is not a credential on its
    // own and is deleted as soon as the callback is handled.
    sessionStorage.setItem("irl.oauth", JSON.stringify({ verifier, state, t: Date.now() }));
    const callback = location.origin + location.pathname;
    const url = new URL("https://openrouter.ai/auth");
    url.searchParams.set("callback_url", callback);
    url.searchParams.set("code_challenge", challenge);
    url.searchParams.set("code_challenge_method", "S256");
    url.searchParams.set("state", state);
    url.searchParams.set("key_label", "IRL browser test");
    location.assign(url.toString());
  },

  // Returns a key if this page load is an OAuth callback, else null.
  async oauthFinish() {
    if (!oauthCallback) return null;
    const { code, state: returnedState } = oauthCallback;
    const saved = JSON.parse(sessionStorage.getItem("irl.oauth") || "null");
    sessionStorage.removeItem("irl.oauth");
    if (!saved || saved.state !== returnedState) {
      throw new Error("OpenRouter sign-in failed: state mismatch (start again)");
    }
    if (Date.now() - saved.t > 10 * 60 * 1000) {
      throw new Error("OpenRouter sign-in expired (codes last 10 minutes)");
    }
    const r = await fetch("https://openrouter.ai/api/v1/auth/keys", {
      method: "POST",
      mode: "cors",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, code_verifier: saved.verifier, code_challenge_method: "S256" }),
    });
    if (!r.ok) throw new Error(`OpenRouter sign-in failed (HTTP ${r.status})`);
    const body = await r.json();
    if (!body || typeof body.key !== "string") throw new Error("OpenRouter sign-in returned no key");
    return body.key;
  },
};
