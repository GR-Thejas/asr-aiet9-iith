# Speak & Transcribe: ASR demo for the Group 6 wav2vec2 model

A web page where anyone can give the speech model some audio, read its transcript, correct
mistakes, and watch the model's measured accuracy build up from that feedback.

---

## Quick start

**You need:** Python 3.10 or newer, tested on 3.11 and 3.13 ([python.org](https://www.python.org/downloads/); on Windows tick
*"Add python.exe to PATH"*), an internet connection the first time, and about 2 GB of free disk space.

| System                  | What to do                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------ |
| **Windows**       | Unzip the folder and double-click **`run_app.bat`**                          |
| **macOS / Linux** | Unzip, open a terminal in the folder and run `chmod +x run_app.sh && ./run_app.sh` |

The first run creates a private `.venv` folder and installs the packages (5–15 minutes). On first
start the app also downloads the base model (about 360 MB) if no fine-tuned checkpoint is present.
Later runs start in seconds.

When the window says *"You can now view your Streamlit app"*, open **http://localhost:8501** in
Chrome or Edge. To stop the app, close the window or press **Ctrl+C**.

### From a VS Code / PowerShell terminal

Run the commands **inside the app folder** (the one that contains `app.py`). PowerShell also needs
`.\` in front of a script in the current folder:

```powershell
cd path\to\ASR_Speak_and_Transcribe
.\run_app.bat
```

You don't need to activate any venv first, because the launcher uses its own `.venv`. If you
activate one by hand, use the app's own `.venv` (`.\.venv\Scripts\Activate.ps1`) and then run
`python -m streamlit run app.py`. In VS Code, pick that same interpreter with
*Python: Select Interpreter*.

Open the page in **Chrome or Edge**, not VS Code's built-in Simple Browser, which can't access the microphone.

### Manual alternative

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

---

## Using the fine-tuned model

Training completed all 8920 steps. The weights are in Google Drive under
`My Drive/wav2vec2_librispeech_finetuned/`:

- `checkpoint-8920`: final weights, best validation WER (3.62%). **Use this one.**
- `checkpoint-1000` (also exported as the top-level model): lowest validation loss, WER 3.98%.

Copy `checkpoint-8920` into the `model/` folder, so `model/checkpoint-8920/config.json` exists.
Also copy the top-level `vocab.json`, `tokenizer_config.json` and `processor_config.json` into it.
Then restart the app. On Streamlit Cloud, the weights come from Hugging Face instead (see `DEPLOY.md`). The sidebar shows which weights are loaded and lets you switch between them.
Without a checkpoint, the app uses the base `facebook/wav2vec2-base-960h` model.

---

## No microphone or speakers?

The page has three ways to give it audio:

| Option                                                                             | Needs a mic? | Needs speakers? |
| ---------------------------------------------------------------------------------- | ------------ | --------------- |
| 🎙️ **Record with microphone**                                              | Yes          | No              |
| 📁 **Upload an audio file** (WAV/FLAC/OGG/MP3)                               | No           | No              |
| 🎧 **Use a sample clip**: demo clips with their script shown (only when `samples/` has clips; not in the GitHub repo) | No           | No              |

On a machine with no microphone, use **Upload** or **Sample clip**. Speakers are never needed:
the audio player is only there if you want to listen.

If **Record** shows *"This app would like to use your microphone"* or opens a help page,
it means the browser has no microphone or permission was refused. Allow it from the 🔒 icon in
the address bar, or use the other two options.

---

## Letting other people use it

Browsers allow the microphone only on `localhost` or `https://` pages.

- **Other computers on the same network (upload and samples only):**
  `run_app.bat --server.address 0.0.0.0`. Others open `http://<this-computer's-IP>:8501`.
  Recording is blocked there because the page is plain HTTP. Windows may ask to allow Python
  through the firewall.
- **Public HTTPS link, so other people's microphones work:** keep the app running and in a second window run
  `cloudflared tunnel --url http://localhost:8501`
  ([download cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)).
  Share the `https://….trycloudflare.com` link it prints. Visitors record with *their own* microphones,
  so this computer needs none.

---

## What the page does

1. **Try it:** record, upload or pick a sample. The model returns its transcript, the processing
   time and its own confidence.
2. **Feedback:** fix the transcript to what was actually said, optionally tag the problem
   (typo, misheard word, missing or extra words, misleading meaning…) and add a comment.
   Then click **👍 It's exactly right** or **✏️ Submit my correction**.
3. **Result:** word and character accuracy for that clip, plus a colour-coded diff showing what the
   model wrote, what was said, and which words it missed.
4. **Accuracy dashboard:** overall word accuracy from everyone's feedback, a trend chart, the
   problems reported, the most-confused words, recent feedback and a CSV download.

Audio is processed in memory and never saved. Feedback text is stored in `feedback/feedback.jsonl`.

**How accuracy is calculated:** the user's correction is the reference. Both texts are uppercased
and punctuation is stripped, because the model only outputs A–Z and apostrophes. They are then
aligned word by word. WER = (wrong + missing + extra words) / reference words, and word accuracy = 1 − WER,
pooled over all rated clips.

---

## Troubleshooting

| Problem                                                                            | Fix                                                                                                                      |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `Python was not found`                                                           | Install Python 3.10 or newer and tick *Add to PATH*                                                                    |
| Install failed part-way                                                            | Delete the `.venv` folder and run the launcher again                                                                   |
| `Could not import module 'Wav2Vec2ForCTC'` / `torchvision::nms does not exist` | A mismatched global `torchvision`. Use the launcher, which makes a clean `.venv`                                     |
| `streamlit` / `run_app.bat` *is not recognized*                              | You're in the wrong folder or the wrong venv is active. `cd` into the folder with `app.py` and run `.\run_app.bat` |
| Port 8501 already in use                                                           | `run_app.bat --server.port 8502`                                                                                       |
| Numbers missing from the transcript                                                | Expected: the model outputs only letters, so type numbers as words when correcting                                       |

## Files

| File                             | Purpose                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `run_app.bat` / `run_app.sh` | One-click setup and launch                                                                |
| `app.py`                       | Streamlit page (input → transcript → feedback → dashboard)                             |
| `asr_core.py`                  | Model loading, audio decoding/resampling, CTC decoding, WER/CER scoring, feedback storage |
| `samples/`                     | Demo clips (`.wav`) and their scripts (`.txt`) for use without a microphone. Git-ignored, so not deployed |
| `model/`                       | Put the fine-tuned checkpoint here                                                        |
| `.streamlit/config.toml`       | Upload size limit and theme                                                               |
