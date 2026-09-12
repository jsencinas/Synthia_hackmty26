# Altur Challenge: HackMTY 2026

Recorded phone calls between a caller and a bank's AI customer-service agent, in Mexican Spanish.
In some calls the caller is a real person. In others the caller is an autonomous AI: speech recognition, a language model and a synthetic voice, dialing the same number.

Your task: given a call, decide whether the caller is human or synthetic.

## Files

| Path | Contents |
| --- | --- |
| `manifest.csv` | One row per call: `anon_id`, `label` (`human` or `synthetic`), `split` (`train` or `val`), `duration_s`. |
| `audio/<anon_id>.wav` | Stereo, 8 kHz, 16-bit PCM. Channel 0 is the caller (the one you classify). Channel 1 is the agent. |
| `turns/<anon_id>.json` | Speech segments per channel, `{"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}`, seconds from the start of the file. Derived automatically from the audio; use them as a starting point. |

Audio is distributed as `altur-challenge-audio.zip` (see Releases). Unzip it in the repo root so the files land in `audio/`.

The production pipeline treats `manifest.csv`, `audio/`, and `turns/` as
immutable inputs. Models do not train from checked-in feature tables. Timing
boundaries and both feature sets are derived directly from each WAV by the same
code used during serving.

## The conversation

Every call follows the same customer-service flow, whoever is calling. The agent asks callers to repeat information back,
sometimes asks about things that do not exist, and there are moments where it interrupts, falls silent, or talks over the caller.
Both sides are given to you for a reason: channel 1 tells you what the caller was reacting to.

## Splits

`train` and `val` are speaker-disjoint: no caller appears in both. Judging uses a hidden set of calls from callers and voices that appear in neither split.

## Reproducible workflow

Install runtime and training dependencies:

```bash
python -m pip install -r requirements.txt
```

Train production artifacts:

```bash
python -m src.train
```

This command selects `split=train` before feature extraction. Cross-validation,
calibration, fusion fitting, and final fitting use training rows only. It never
extracts VAL features or reports VAL metrics. Artifacts are written atomically
to `models/` with train-only provenance and a source-data fingerprint.

After the model and all decisions are frozen, run validation explicitly:

```bash
python -m src.evaluate
```

Evaluation loads existing artifacts, processes only `split=val`, prints metrics,
and does not write models, thresholds, calibration, or feature caches. Repeated
VAL inspection should not be used for model selection.

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Evaluation

Your system exposes `POST /detect`. It receives a stereo WAV clip (8 kHz, base64-encoded, channel 0 = caller, channel 1 = agent) and returns:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` is required. `confidence` is optional and used to break ties and reward calibration.

`confidence` is the probability of the returned decision, so it is always in
the range 0.5–1.0. The internal synthetic-class probability is not exposed.

Run the API with:

```bash
uvicorn api.app:app
```

The endpoint accepts only the uploaded stereo 8 kHz WAV. It does not accept a
local turns path; turn extraction always happens directly from that WAV.

## Terms

Human callers volunteered, were told the call was recorded for an AI test, and used invented personal data. Do not try to identify anyone.
This dataset is provided for HackMTY 2026 only; do not redistribute.
