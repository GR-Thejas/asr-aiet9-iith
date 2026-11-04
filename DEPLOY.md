# Deploying to Streamlit Community Cloud

The result is a public HTTPS link such as `https://speak-and-transcribe.streamlit.app`.
It's HTTPS, so visitors' microphones work.

You need a GitHub account, a Hugging Face account (both free) and the fine-tuned weights in
`model/checkpoint-8920/`. See `model/PUT_CHECKPOINT_HERE.txt`.

## 1. Put the fine-tuned weights on Hugging Face

`model.safetensors` is about 360 MB, which is over GitHub's 100 MB file limit, so the app
downloads it from the Hugging Face Hub instead. Upload the weights, config and processor files only.
Skip `optimizer.pt` (≈700 MB, only used for resuming training).

```powershell
pip install -U huggingface_hub
hf auth login                       # paste a *write* token from https://huggingface.co/settings/tokens
hf upload YOUR_HF_NAME/wav2vec2-group6-ckpt8920 model\checkpoint-8920 . --include "*.json" "model.safetensors"
```

## 2. Push this folder to GitHub

Create an **empty** repository on github.com (for example `speak-and-transcribe`), then run this
from this folder (the one with `app.py`):

```powershell
git init -b main
git add .
git commit -m "Speak & Transcribe ASR app"
git remote add origin https://github.com/YOUR_GITHUB_NAME/speak-and-transcribe.git
git push -u origin main
```

`.gitignore` keeps `.venv/`, model weights, the voice clips in `samples/`, collected feedback and secrets out of the repo.

## 3. Create the app on Streamlit

1. Go to <https://share.streamlit.io>, sign in with GitHub, and click **Create app** → **Deploy a public app from GitHub**.
2. **Repository:** `YOUR_GITHUB_NAME/speak-and-transcribe` · **Branch:** `main` · **Main file path:** `app.py`
3. **App URL:** choose your subdomain, e.g. `speak-and-transcribe` → `speak-and-transcribe.streamlit.app`
4. **Advanced settings:**
   - **Python version:** 3.12 (3.11 and 3.13 also work. `requirements.txt` pins a CPU PyTorch wheel for each.)
   - **Secrets:** paste the content of `.streamlit/secrets.toml.example`, with your Hugging Face repo name filled in.
5. Click **Deploy**. The first build takes about 5 minutes. The first visit then downloads the model (about 1 minute).

Check the sidebar. It should say `Running on CPU · wav2vec2-group6-ckpt8920`.

## Good to know

- **Custom domain:** Community Cloud only serves `*.streamlit.app` addresses. You choose the
  subdomain, but you can't point your own domain (e.g. `asr.example.com`) at it. For your own
  domain, run the app on a machine you control and expose it with a
  [Cloudflare named tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/)
  on that domain.
- **Feedback is not permanent there:** `feedback/feedback.jsonl` lives on the cloud container's
  disk. It is wiped when the app restarts, redeploys or wakes from sleep (apps sleep after about
  12 hours without visitors). Download the CSV from the dashboard regularly, or move storage to an
  external database.
- **Resources:** about 2.7 GB RAM and CPU only. The model needs about 0.5 GB, and a 30 s clip
  transcribes in a few seconds.
- **Updating:** `git push` to `main` and the app redeploys automatically.
- **Private model repo:** add `HF_TOKEN = "hf_..."` (a read token) to the app's secrets.
