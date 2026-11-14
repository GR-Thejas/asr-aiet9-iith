"""Model loading, transcription, scoring and feedback storage for the ASR demo.

Kept free of Streamlit calls so it can be tested or reused on its own.
"""
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

os.environ.setdefault("USE_TF", "0")  # PyTorch only; skip slow TensorFlow import if installed
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

BASE_MODEL_ID = "facebook/wav2vec2-base-960h"
TARGET_SR = 16_000
CHUNK_SECONDS = 30  # wav2vec2 memory grows with length, so long clips are split
MIN_SAMPLES = TARGET_SR // 10

_write_lock = threading.Lock()


# ---------------------------------------------------------------- model ----

def find_checkpoints(folder: str) -> list[Path]:
    """Return loadable model folders: the folder itself and/or its checkpoint-* children."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    found = [root] if (root / "config.json").exists() else []

    def step(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    ckpts = [c for c in root.glob("checkpoint-*") if (c / "config.json").exists()]
    return found + sorted(ckpts, key=step, reverse=True)


def pick_device() -> str:
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).item()  # fails on GPUs this torch build can't run
            return "cuda"
        except Exception:
            pass
    return "cpu"


def load_model(source: str):
    """Load the CTC model and its processor from a local folder or a Hub id.

    Trainer checkpoints only contain the feature extractor (the notebook's final
    processor.save_pretrained never ran), so the tokenizer falls back to the base
    model's, which is the vocabulary the fine-tuning used.
    """
    try:
        processor = Wav2Vec2Processor.from_pretrained(source)
    except Exception:
        processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_ID)
    device = pick_device()
    model = Wav2Vec2ForCTC.from_pretrained(source).eval().to(device)
    return processor, model, device


# ---------------------------------------------------------------- audio ----

def load_audio(raw: bytes) -> np.ndarray:
    """Decode audio or video bytes to mono float32 at 16 kHz.

    WAV/FLAC/OGG (and MP3 on recent libsndfile) are read with soundfile. Anything it
    can't read, such as MP4, M4A, MOV, WEBM or MKV, is decoded with PyAV, whose wheel
    bundles FFmpeg, so no system packages are needed on Streamlit Community Cloud.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        return _decode_with_av(raw)
    audio = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
    return audio


def _decode_with_av(raw: bytes) -> np.ndarray:
    """Extract the first audio track of any FFmpeg-readable file as mono 16 kHz float32."""
    import av  # imported lazily: only needed for formats soundfile can't read

    with av.open(io.BytesIO(raw)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("this file has no audio track")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush samples buffered in the resampler
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def transcribe(audio: np.ndarray, processor, model, device: str) -> dict:
    """Greedy CTC decoding, chunked; confidence is the mean max-prob of non-blank frames."""
    start = time.perf_counter()
    texts, frame_probs = [], []
    step = CHUNK_SECONDS * TARGET_SR
    blank_id = processor.tokenizer.pad_token_id
    for offset in range(0, len(audio), step):
        chunk = audio[offset:offset + step]
        if len(chunk) < MIN_SAMPLES:
            continue
        inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits[0]
        probs = logits.softmax(-1)
        best, ids = probs.max(-1)
        frame_probs.append(best[ids != blank_id].cpu().numpy())
        texts.append(processor.decode(ids.cpu()))
    probs_all = np.concatenate(frame_probs) if frame_probs else np.array([])
    return {
        "text": " ".join(t for t in texts if t).strip(),
        "confidence": float(probs_all.mean()) if probs_all.size else None,
        "seconds": time.perf_counter() - start,
    }


# -------------------------------------------------------------- scoring ----

def normalize(text: str) -> str:
    """Map text onto the model's vocabulary (A-Z, apostrophe, space)."""
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def align(ref: list, hyp: list) -> list[tuple]:
    """Levenshtein alignment as (op, ref_token, hyp_token), op in ok/sub/del/ins."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(hypothesis: str, reference: str) -> dict | None:
    """Compare the model output (hypothesis) with the user's correction (reference)."""
    hyp, ref = normalize(hypothesis), normalize(reference)
    if not ref:
        return None
    ops = align(ref.split(), hyp.split())
    counts = {op: sum(1 for o in ops if o[0] == op) for op in ("ok", "sub", "del", "ins")}
    ref_words = len(ref.split())
    word_errors = counts["sub"] + counts["del"] + counts["ins"]
    char_errors = edit_distance(ref, hyp)
    return {
        "ops": ops,
        "ref_words": ref_words,
        "word_errors": word_errors,
        "substitutions": counts["sub"],
        "deletions": counts["del"],
        "insertions": counts["ins"],
        "wer": word_errors / ref_words,
        "ref_chars": len(ref),
        "char_errors": char_errors,
        "cer": char_errors / len(ref),
        "confusions": [[r, h] for op, r, h in ops if op == "sub"],
    }


# ------------------------------------------------------------- feedback ----

def save_feedback(path: Path, record: dict) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip a partially written line rather than break the dashboard
    return rows
