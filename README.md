# Psychosocial Risk Dashboard

**CHAP – Closed Loop AI Prediction Platform** · Georgia Tech VIP: AutoRobotics in Construction

Wearable sensors on construction workers → AI prediction of stress / frustration / mental
demand → decision engine → spoken feedback to the worker → the worker's response shows up
again in the sensor data (closed loop).

```text
Sensors → AI Stress Prediction → Decision Engine → Audio Feedback → Worker feedback → Sensor data
```

This is the **single shared repository** for all VIP sub-teams. Each team works on a
feature branch and merges into `main` through a pull request (see [Team workflow](#team-workflow)).

---

## What it does today

| Feature | Files | Status |
|---|---|---|
| **Live wristband poller** – downloads Empatica EmbracePlus data (EDA, skin temperature, pulse rate) from AWS S3 every 30 s and runs the ML models | `avro_stream_poller_with_ml.py`, `biomarker_runtime.py` | Working |
| **ML models** – Frustration and Mental Demand (Low / Medium / High) from EDA + temperature + minute index | `models/`, `train_biomarker_models.py` | Working (see [Model notes](#model-notes)) |
| **Supervisor dashboard** – historical sessions + a live worker, auto-refresh every 5 s | `serve_dashboard.py`, `live_dashboard_backend.py`, `psychosocial_dashboard*.html` | Working |
| **Polar H10 panel** – connects to a Polar H10 chest strap from the browser and shows the EXG 200 ms RMS signal | `sensor/`, `assets/polar-panel.js` | Working (Chrome / Edge) |
| **Voice coach** – speaks feedback to the worker: label announcements, or escalating stress / break / heat / heart-rate alerts | `speak_biomarker_labels.py` | Working |

---

## Setup

Python 3.11–3.13 recommended.

```bash
git clone https://github.com/andrewlangie/psychosocial-risk-dashboard.git
cd psychosocial-risk-dashboard
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn==1.8.0 joblib flask boto3 requests pyttsx3
```

- `scikit-learn==1.8.0` is the version the saved models in `models/` load with.
- Linux only: `pyttsx3` also needs `sudo apt-get install espeak-ng`.
- macOS: if `python3` refuses to install packages ("externally-managed-environment"), you
  are outside the virtual environment — run `source .venv/bin/activate` first.

### AWS credentials (live wristband data only)

The poller reads credentials **from environment variables only**. Never write keys into a
file in this repo.

```bash
export AWS_ACCESS_KEY_ID=...        # ask the team lead
export AWS_SECRET_ACCESS_KEY=...
```

Everything except the live poller (dashboard history, voice-coach simulation, Polar panel)
works without AWS.

---

## Running

### 1. Live wristband poller (terminal 1, needs AWS)

```bash
python avro_stream_poller_with_ml.py
```

Serves the latest sample + predictions at `http://127.0.0.1:7000/latest`.

### 2. Dashboard (terminal 2)

```bash
python serve_dashboard.py
```

Open `http://127.0.0.1:8000/psychosocial_dashboard.html`. Without the poller it still shows
the historical sessions; with it, a **Live worker** appears in the Frustration and Mental
Demand lanes.

### 3. Polar H10 panel

In the dashboard (Chrome or Edge), click **Connect Polar H10** and allow Bluetooth.
Wear the strap (it sleeps without skin contact). For firmware 4+, pair it in your OS first.
Only one app can connect to the strap at a time — close the Polar Beat app first.

The panel shows only the EXG 200 ms RMS signal, using the filter/RMS code from
[polar-h10-visualizer](https://github.com/yhzhao343/polar-h10-visualizer) (Yuhui Zhao, ISC;
commit `771559b`): 4th-order Butterworth high-pass at 25 Hz, 130 Hz sampling, 26-sample RMS
window. Source is in `sensor/`; the built bundle is `assets/polar-panel.js`.

> The `package.json` used to build `assets/polar-panel.js` is not in the repo yet, so the
> bundle cannot be rebuilt after editing `sensor/`. Ask the Polar panel author to add it.

### 4. Voice coach (terminal 3)

```bash
# Try it without any hardware:
python speak_biomarker_labels.py --simulate --dry-run --speed 0   # instant, text only
python speak_biomarker_labels.py --simulate                       # 70-min shift, spoken

# Live, with the poller running:
python speak_biomarker_labels.py            # announces Frustration / Mental Demand changes
python speak_biomarker_labels.py --coach    # escalating coaching messages
```

| Option | Meaning |
|---|---|
| `--coach` | Coaching mode: check-in, micro-break (20 min), required break (45 min), heat, racing heart, focus, recovery, hydration |
| `--simulate` | Scripted 70-minute shift, no wristband needed (implies `--coach`) |
| `--speed N` | Seconds per simulated minute (default 1.5; `0` = as fast as possible) |
| `--dry-run` | Print messages instead of speaking |
| `--lang en\|es` | English or Spanish messages |
| `--worker-name NAME` | Personalise messages ("…, Carlos") |
| `--vary` | Rotate through alternative phrasings |
| `--endpoint URL`, `--interval S`, `--debug` | Poller URL, polling interval, print raw payloads |

Coach messages, thresholds and cooldowns are in `COACH_MESSAGES` / `COACH_CONFIG` in
`speak_biomarker_labels.py`. Every coach alert is logged to `voice_alert_log.csv`
(git-ignored).

---

## Repository layout

```text
backend/                        Python
  avro_stream_poller_with_ml.py   Live S3 poller + ML prediction API (port 7000)
  biomarker_runtime.py            Loads the models, builds features, predicts per minute
  train_biomarker_models.py       Retrains the models: --dataset <folder|zip> --outdir models
  serve_dashboard.py              Dashboard web server + /api/... routes (port 8000)
  live_dashboard_backend.py       Merges the live worker into the dashboard data
  build_dashboard.py              Original generator for the historical dashboard data/page
  speak_biomarker_labels.py       Voice coach
frontend/                       The only folder the server publishes
  psychosocial_dashboard.html     Dashboard (current)   psychosocial_dashboard_v2.html (variant)
  sensor/  assets/                Polar H10 panel (source / built bundle)
models/                         frustration_model.joblib, mental_demand_model.joblib, model_metadata.json
dashboard_data.json             Historical example sessions (16 MB)
biomarker_data/                 Downloaded wristband CSVs (participant data)
serve_dashboard.py  speak_biomarker_labels.py  avro_stream_poller_with_ml.py
train_biomarker_models.py       Launchers: the old commands still work from the project root
```

---

## Model notes

- Inputs: **EDA, skin temperature, minutes since session start**. Pulse rate is collected
  and shown but is **not** a model input.
- Frustration needs 2 minute-level samples and shows **Pending** until then; Mental Demand needs 1.
- Cross-validated by participant (GroupKFold): Frustration macro-F1 ≈ 0.61, Mental Demand
  macro-F1 ≈ 0.52. Details in `models/model_metadata.json`.
- Trained on lab sessions, not yet validated on construction workers. Treat predictions as
  prototype indicators. The dashboard's stress / fatigue / cognitive-load scores are
  heuristic proxies, not trained models.

---

## Team workflow

1. **Never push to `main` directly.** `main` is protected; every change goes through a pull request with one approval.
2. Start from the latest `main`:
   ```bash
   git checkout main && git pull
   git checkout -b feature/<team>-<feature>      # e.g. feature/polar-fatigue-posture
   ```
3. Commit in small steps with clear messages, then push and open a PR into `main`:
   ```bash
   git push -u origin feature/<team>-<feature>
   ```
4. In the PR: what it adds, how to run/test it, anything reviewers must decide.
5. After merging, delete the branch and `git pull` on `main` before starting the next one.

**Branch prefixes:** `feature/` new work · `fix/` bug fixes · `docs/` documentation only.

---

## Data & security

- **No secrets in Git.** AWS keys come from environment variables. Before committing, check
  `git diff --staged` for keys, tokens or `.env` files. If a key is ever committed, tell the
  team lead immediately so it can be rotated — deleting the file does not remove it from history.
- **Participant data.** `biomarker_data/` and `dashboard_data.json` contain wristband data from
  study participants. Keep this repository **private**, and check with the team lead / study
  protocol before adding more participant data.
- Logs (`voice_alert_log.csv`) and virtual environments (`.venv/`) are git-ignored.

---

## Roadmap

| Item | Owner | Branch |
|---|---|---|
| Restructure into `backend/` (Python) and `frontend/` (pages); server only publishes `frontend/` | Polar / fatigue team | done (`restructure-backend-frontend`) |
| Polar H10 fatigue (HR, RMSSD, SDNN) and posture (accelerometer angles) voice feedback | Polar / fatigue team | `feature/polar-fatigue-posture` |
| Microsoft Edge neural TTS voices | Dean | `dean_tts` (work in progress) |
| Cause-aware stress / frustration decision engine (Level 1–4), worker phone page | Stress / frustration team | to be ported |
| Dashboard panel for fatigue and posture | — | planned |

---

## Credits

Georgia Tech VIP – AutoRobotics in Construction. Polar H10 EXG filtering adapted from
[polar-h10-visualizer](https://github.com/yhzhao343/polar-h10-visualizer) (ISC license).
