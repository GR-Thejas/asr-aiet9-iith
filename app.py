"""Speak & Transcribe: Streamlit front end for the Group 6 fine-tuned wav2vec2 model.

Run with:  streamlit run app.py
"""
import hashlib
import html
import io
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

import asr_core as core

APP_DIR = Path(__file__).parent


def setting(name: str, default=None):
    """Read a deployment setting from the environment, then from Streamlit secrets."""
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml (local runs)
        return default


DEFAULT_MODEL_DIR = setting("ASR_MODEL_PATH", str(APP_DIR / "model"))
FEEDBACK_FILE = Path(setting("ASR_FEEDBACK_FILE", str(APP_DIR / "feedback" / "feedback.jsonl")))
# Hugging Face Hub repo holding the fine-tuned weights, e.g. "your-name/wav2vec2-group6-ckpt8920".
# Used on Streamlit Community Cloud, where the 360 MB checkpoint can't live in the GitHub repo.
HF_MODEL_ID = (setting("ASR_HF_MODEL") or "").strip()
# Hide the "model folder" box on a public deployment so visitors can't browse server paths.
LOCK_MODEL = str(setting("ASR_LOCK_MODEL", "")).lower() in ("1", "true", "yes")
SAMPLES_DIR = APP_DIR / "samples"
MAX_SECONDS = 180
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}
ISSUE_TYPES = [
    "Typo / misspelled word",
    "Wrong word (misheard)",
    "Missing words",
    "Extra words",
    "Meaning changed / misleading",
    "Name or uncommon word",
    "Words merged or split",
]

# Validation results from the notebook's training log (LibriSpeech dev-clean).
TRAINING_LOG = pd.DataFrame(
    {
        "Step": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8920],
        # Training loss at 8920 is the last logged value (step 8900).
        "Training loss": [0.0758, 0.0566, 0.0518, 0.0400, 0.0401, 0.0297, 0.0299, 0.0259, 0.0267],
        "Validation loss": [0.0664, 0.0671, 0.0669, 0.0688, 0.0672, 0.0700, 0.0689, 0.0715, 0.0715],
        "Validation WER": [0.03976, 0.03963, 0.03976, 0.03849, 0.03755, 0.03713, 0.03713,
                           0.03619, 0.03619],
    }
)

# Runs in the viewer's browser: the server can't see the visitor's microphone, and
# st.audio_input only reports a generic error when none is usable. The script must not
# contain HTML tags in its strings, or Streamlit's sanitizer drops the whole <script>.
MIC_CHECK_HTML = """
<div class="mic-check" style="padding:.6rem .9rem;border-radius:.5rem;font-size:.92rem;
     background:rgba(128,128,128,.10)">Checking for a microphone…</div>
<script>
(() => {
  const COLORS = {ok: 'rgba(34,197,94,.15)', warn: 'rgba(245,158,11,.18)', bad: 'rgba(239,68,68,.15)',
                  info: 'rgba(128,128,128,.10)'};
  const FALLBACK = ' Use Upload or Sample clip instead.';
  const show = (tone, title, detail) => document.querySelectorAll('.mic-check').forEach(el => {
    const strong = document.createElement('strong');
    strong.textContent = title;
    el.replaceChildren(strong, document.createTextNode(' ' + detail));
    el.style.background = COLORS[tone];
  });
  const REASONS = {
    NotFoundError: ['warn', '⚠️ No microphone found.', 'Connect a microphone or headset; this message updates by itself.'],
    NotReadableError: ['warn', '⚠️ The microphone cannot be opened.',
      'It may be a disconnected Bluetooth headset, or another app (Teams, Zoom…) is using it.'],
    NotAllowedError: ['bad', '🚫 Microphone access is blocked for this page.',
      'Allow it from the lock icon in the address bar, then reload.'],
  };
  const check = async () => {
    const md = navigator.mediaDevices;
    if (!window.isSecureContext || !md || !md.getUserMedia) {
      show('bad', '🔒 Recording is not available on this page.',
           'Browsers only allow the microphone on localhost or https:// addresses.' + FALLBACK);
      return;
    }
    const devices = await md.enumerateDevices();
    if (!devices.some(d => d.kind === 'audioinput')) {
      const [tone, title, detail] = REASONS.NotFoundError;
      show(tone, title, detail + FALLBACK);
      return;
    }
    let state = 'prompt';
    try { state = (await navigator.permissions.query({name: 'microphone'})).state; } catch (e) {}
    if (state === 'prompt') {
      show('info', '🎙️ A microphone input was found.',
           'Your browser will ask for permission when you press record.');
      return;
    }
    try {  // permission already decided, so this opens the device without a prompt
      const stream = await md.getUserMedia({audio: true});
      stream.getTracks().forEach(t => t.stop());
      show('ok', '✅ Microphone ready.', 'Press the microphone below to record.');
    } catch (err) {
      const [tone, title, detail] = REASONS[err.name] ||
        ['warn', '⚠️ The microphone is not working.', '(' + err.name + ')'];
      show(tone, title, detail + FALLBACK);
    }
  };
  check();
  navigator.mediaDevices?.addEventListener?.('devicechange', check);
})();
</script>
"""

st.set_page_config(page_title="Speak & Transcribe", page_icon="🎙️", layout="centered")

st.markdown(
    """
    <style>
    .transcript {font-size: 1.35rem; line-height: 1.6; padding: 1rem 1.25rem;
        border-radius: 0.75rem; border: 1px solid rgba(128,128,128,.25);
        background: rgba(79,70,229,.06); letter-spacing: .02em; word-wrap: break-word;}
    .diff {font-size: 1.1rem; line-height: 2.1;}
    .diff span {padding: 0.1rem 0.3rem; border-radius: 0.3rem; margin-right: 0.15rem;}
    .w-bad  {background: rgba(239,68,68,.18); text-decoration: line-through;}
    .w-fix  {background: rgba(34,197,94,.22);}
    .w-miss {background: rgba(59,130,246,.20); border-bottom: 2px dashed rgba(59,130,246,.8);}
    .legend span {font-size: .85rem; margin-right: .75rem; padding: .1rem .35rem; border-radius: .3rem;}
    .legend .w-bad {text-decoration: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# max_entries bounds memory: Community Cloud gives about 2.7 GB of RAM to the whole app.
@st.cache_resource(max_entries=2, show_spinner="Loading the speech model (first time only)…")
def get_model(source: str):
    return core.load_model(source)


def model_label(source: str) -> str:
    if source == core.BASE_MODEL_ID:
        return "wav2vec2-base-960h (base, not fine-tuned)"
    return Path(source).name


def diff_html(ops) -> str:
    esc = html.escape
    parts = []
    for op, ref, hyp in ops:
        if op == "ok":
            parts.append(f"<span>{esc(ref)}</span>")
        elif op == "sub":
            parts.append(f"<span class='w-bad'>{esc(hyp)}</span><span class='w-fix'>{esc(ref)}</span>")
        elif op == "del":
            parts.append(f"<span class='w-miss'>{esc(ref)}</span>")
        else:
            parts.append(f"<span class='w-bad'>{esc(hyp)}</span>")
    return "<div class='diff'>" + " ".join(parts) + "</div>"


def pct(x: float) -> str:
    return f"{max(0.0, x) * 100:.1f}%"


# Drawn before the sidebar loads the model, so the page isn't blank on first start.
st.title("🎙️ Speak & Transcribe")
st.write(
    "Say something, and our fine-tuned **wav2vec2** speech model will write down what it heard. "
    "Then tell us what it got wrong. Every correction updates the model's live accuracy score."
)

# ---------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.header("⚙️ Model")
    if LOCK_MODEL:
        folder = DEFAULT_MODEL_DIR
    else:
        folder = st.text_input(
            "Fine-tuned model folder",
            DEFAULT_MODEL_DIR,
            help="Folder holding config.json + model.safetensors, or the training output "
            "folder with checkpoint-* subfolders (e.g. a copy of "
            "'My Drive/wav2vec2_librispeech_finetuned').",
        )
    candidates = ([HF_MODEL_ID] if HF_MODEL_ID else []) + [
        str(c) for c in core.find_checkpoints(folder)]
    source = st.selectbox(
        "Weights to use", candidates + [core.BASE_MODEL_ID], format_func=model_label
    )
    if not candidates:
        st.warning(
            "No fine-tuned checkpoint found in that folder, so the base model is used. "
            "Copy `checkpoint-8920` from Google Drive into it (see the About tab)."
        )
    processor, model, device = get_model(source)
    st.caption(f"Running on **{device.upper()}** · {model_label(source)}")
    st.divider()
    st.caption(
        "🔒 Recordings are processed in memory and never saved. "
        "Only the transcript text and your feedback are stored."
    )

# ------------------------------------------------------------------- main ----

tab_try, tab_stats, tab_about = st.tabs(["🎤 Try it", "📊 Accuracy dashboard", "ℹ️ About the model"])

with tab_try:
    # Demo clips are kept out of the public repo, so the option only shows when they exist.
    samples = sorted(SAMPLES_DIR.glob("*.wav"))
    modes = ["🎙️ Record with microphone", "📁 Upload an audio or video file"]
    if samples:
        modes.append("🎧 Use a sample clip")
    mode = st.radio("Input", modes, horizontal=True, label_visibility="collapsed")
    if mode.startswith("🎙️"):
        mic_html = MIC_CHECK_HTML if samples else MIC_CHECK_HTML.replace(
            "Use Upload or Sample clip instead.", "Use Upload instead.")
        st.html(mic_html, unsafe_allow_javascript=True)
        clip = st.audio_input("Press the microphone, speak clearly in English, then press stop.")
        alternatives = ("Choose **Upload** or **Sample clip** above. Neither needs a microphone or speakers."
                        if samples else "Choose **Upload** above. It doesn't need a microphone or speakers.")
        st.caption("If the recorder says *\"An error has occurred, please try again\"*, the browser "
                   "could not open a microphone (none connected, or another app is using it). "
                   + alternatives)
    elif mode.startswith("📁"):
        clip = st.file_uploader("Audio or video file (WAV, FLAC, OGG, MP3, M4A, MP4, MOV, WEBM or MKV)",
                                type=["wav", "flac", "ogg", "mp3", "m4a", "aac",
                                      "mp4", "mov", "webm", "mkv"])
        if clip is not None:
            # WhatsApp voice notes and similar are .mp4 files with no picture: play those as audio.
            if Path(clip.name).suffix.lower() in VIDEO_EXTS and core.has_video(clip.getvalue()):
                st.video(clip)
                st.caption("Only the audio track is transcribed.")
            else:
                st.audio(clip)
    else:
        pick = st.selectbox("Sample clip", samples,
                            format_func=lambda p: p.stem.split("_", 1)[-1].replace("_", " ").capitalize())
        script = pick.with_suffix(".txt")
        if script.exists():
            st.markdown(f"**What this clip says:** _{script.read_text(encoding='utf-8-sig').strip()}_")
        clip = io.BytesIO(pick.read_bytes())
        st.audio(clip.getvalue(), format="audio/wav")

    if clip is None:
        st.info("Tip: speak a full sentence of 3–15 seconds in a quiet room for the best results.")
    else:
        raw = clip.getvalue()
        audio_id = hashlib.sha1(raw + source.encode()).hexdigest()[:16]

        if st.session_state.get("audio_id") != audio_id:
            try:
                audio = core.load_audio(raw)
            except Exception as exc:
                st.error(f"Could not read the audio in that file: {exc}")
                st.stop()
            duration = len(audio) / core.TARGET_SR
            if duration > MAX_SECONDS:
                st.error(f"That clip is {duration:.0f} s long. Please keep it under {MAX_SECONDS} s.")
                st.stop()
            with st.spinner("Listening to your recording…"):
                result = core.transcribe(audio, processor, model, device)
            result["duration"] = duration
            result["quiet"] = float(abs(audio).max()) < 0.01 if len(audio) else True
            st.session_state.audio_id = audio_id
            st.session_state.result = result

        result = st.session_state.result
        text = result["text"]

        st.subheader("📝 What the model heard")
        shown = html.escape(text) if text else "<em>(no speech recognised)</em>"
        st.markdown(f"<div class='transcript'>{shown}</div>", unsafe_allow_html=True)
        if result["quiet"]:
            st.warning("The recording is very quiet. Check your microphone or move closer to it.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Clip length", f"{result['duration']:.1f} s")
        c2.metric("Processing time", f"{result['seconds']:.2f} s")
        conf = result["confidence"]
        c3.metric("Model confidence", "n/a" if conf is None else f"{conf * 100:.0f}%",
                  help="Average probability the model gave the letters it chose. "
                  "This is the model's own estimate, not measured accuracy.")

        st.divider()
        scored = st.session_state.setdefault("scored", {})

        if audio_id not in scored:
            st.subheader("💬 How did it do?")
            with st.form(f"feedback_{audio_id}"):
                corrected = st.text_area(
                    "What did you actually say? Fix any wrong, missing or extra words.",
                    value=text,
                    height=110,
                    help="Capital letters and punctuation are ignored when scoring. "
                    "Write numbers as words (e.g. 'twenty five'), because the model only outputs letters.",
                )
                issues = st.multiselect("What went wrong? (optional)", ISSUE_TYPES)
                comment = st.text_input("Anything else you'd like to tell us? (optional)")
                b1, b2 = st.columns(2)
                is_perfect = b1.form_submit_button("👍 It's exactly right", width="stretch")
                is_fix = b2.form_submit_button("✏️ Submit my correction", type="primary",
                                               width="stretch")

            if is_perfect or is_fix:
                reference = text if is_perfect else corrected
                s = core.score(text, reference)
                if s is None:
                    st.warning("Please type what you said before submitting.")
                else:
                    core.save_feedback(FEEDBACK_FILE, {
                        "model": model_label(source),
                        "duration_s": round(result["duration"], 2),
                        "confidence": None if conf is None else round(conf, 4),
                        "transcript": text,
                        "corrected": reference,
                        "issues": [] if is_perfect else issues,
                        "comment": comment.strip(),
                        **{k: v for k, v in s.items() if k != "ops"},
                    })
                    scored[audio_id] = s
                    st.toast("Thanks! Your feedback was saved.", icon="✅")
                    st.rerun()

        if audio_id in scored:
            s = scored[audio_id]
            st.subheader("✅ Result for this recording")
            m1, m2, m3 = st.columns(3)
            m1.metric("Word accuracy", pct(1 - s["wer"]))
            m2.metric("Character accuracy", pct(1 - s["cer"]))
            m3.metric("Word errors", f"{s['word_errors']} of {s['ref_words']}")
            if s["word_errors"] == 0:
                st.success("Perfect transcript. The model got every word right. 🎉")
            else:
                st.caption(f"{s['substitutions']} wrong · {s['deletions']} missing · "
                           f"{s['insertions']} extra")
                st.markdown(
                    "<div class='legend'><span class='w-bad'>model wrote</span>"
                    "<span class='w-fix'>you said</span>"
                    "<span class='w-miss'>model missed</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(diff_html(s["ops"]), unsafe_allow_html=True)
            st.info("Record or upload another clip above to keep testing the model.")

with tab_stats:
    rows = core.load_feedback(FEEDBACK_FILE)
    if not rows:
        st.info("No feedback yet. Try the model in the **Try it** tab and submit a correction "
                "to start building the accuracy score.")
    else:
        df = pd.DataFrame(rows)
        models = sorted(df["model"].unique())
        current = model_label(source)
        chosen = st.selectbox("Show results for", ["All models"] + models,
                              index=(models.index(current) + 1) if current in models else 0)
        if chosen != "All models":
            df = df[df["model"] == chosen]

        if df.empty:
            st.info("No feedback for this model yet.")
        else:
            wer = df["word_errors"].sum() / df["ref_words"].sum()
            cer = df["char_errors"].sum() / df["ref_chars"].sum()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Recordings rated", len(df))
            k2.metric("Word accuracy", pct(1 - wer), help=f"Overall WER = {wer:.2%}")
            k3.metric("Character accuracy", pct(1 - cer), help=f"Overall CER = {cer:.2%}")
            k4.metric("Perfect transcripts", pct((df["word_errors"] == 0).mean()))
            st.caption(
                f"For comparison, the notebook measured **{pct(1 - TRAINING_LOG['Validation WER'].iloc[-1])}** "
                "word accuracy on LibriSpeech read speech. Real users, accents and microphones are usually harder."
            )

            st.markdown("##### Accuracy as feedback comes in")
            if len(df) < 2:
                st.caption("The trend chart appears once at least two recordings have been rated.")
            else:
                trend = pd.DataFrame({
                    "Overall word accuracy (%)":
                        (1 - df["word_errors"].cumsum() / df["ref_words"].cumsum()).clip(lower=0).values * 100,
                    "This recording (%)": (1 - df["wer"]).clip(lower=0).values * 100,
                }, index=pd.RangeIndex(1, len(df) + 1, name="Recording #"))
                st.line_chart(trend, x_label="Recording #", y_label="Word accuracy (%)")

            left, right = st.columns(2)
            with left:
                st.markdown("##### Reported problems")
                issue_counts = Counter(i for lst in df["issues"] for i in lst)
                if issue_counts:
                    st.bar_chart(pd.Series(issue_counts, name="Reports").sort_values(), horizontal=True)
                else:
                    st.caption("No problems reported yet.")
            with right:
                st.markdown("##### Most confused words")
                pairs = Counter(f"{r} → {h}" for lst in df["confusions"] for r, h in lst)
                if pairs:
                    st.dataframe(
                        pd.DataFrame(pairs.most_common(10), columns=["You said → model wrote", "Times"]),
                        hide_index=True, width="stretch",
                    )
                else:
                    st.caption("No misheard words yet.")

            st.markdown("##### Recent feedback")
            recent = df.iloc[::-1].head(50).assign(
                **{"Word accuracy": lambda d: (1 - d["wer"]).clip(lower=0).map("{:.0%}".format)}
            )[["timestamp", "model", "transcript", "corrected", "Word accuracy", "comment"]]
            st.dataframe(recent.rename(columns={
                "timestamp": "Time (UTC)", "model": "Model", "transcript": "Model output",
                "corrected": "User correction", "comment": "Comment"}),
                hide_index=True, width="stretch")
            st.download_button("⬇️ Download all feedback (CSV)",
                               df.drop(columns=["confusions"]).to_csv(index=False).encode("utf-8"),
                               "asr_feedback.csv", "text/csv")

with tab_about:
    st.markdown(
        """
        **Model:** `facebook/wav2vec2-base-960h` fine-tuned by Group 6 on LibriSpeech
        *train-clean-100* (28,539 clips), validated on *dev-clean* (2,703 clips).
        It turns raw 16 kHz audio into uppercase English letters with CTC decoding.
        It doesn't add punctuation or digits.

        **Training status:** ✅ completed all **8,920 steps** (5 epochs). This app uses
        **checkpoint-8920**, the final weights, which have the best validation WER (3.62%).
        The top-level model the Trainer exported is a copy of checkpoint-1000, the step with
        the lowest validation loss but a higher WER (3.98%).
        """
    )
    st.dataframe(TRAINING_LOG.style.format({"Validation WER": "{:.2%}", "Training loss": "{:.4f}",
                                            "Validation loss": "{:.4f}"}),
                 hide_index=True, width="stretch")
    st.markdown(
        """
        **Using the fine-tuned weights:** put `checkpoint-8920` (its `config.json`,
        `model.safetensors` and processor files) into this app's `model/` folder, or set
        `ASR_HF_MODEL` to a Hugging Face repo holding them. If a checkpoint has no tokenizer
        files, the app uses the base model's, which has the same vocabulary.

        **How accuracy is measured:** your correction is treated as the ground truth. The app
        aligns it word-by-word with the model output and counts wrong, missing and extra words
        (Word Error Rate). Word accuracy = 1 − WER, pooled over all rated recordings.
        """
    )
