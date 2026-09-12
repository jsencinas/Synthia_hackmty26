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

## The conversation

Every call follows the same customer-service flow, whoever is calling. The agent asks callers to repeat information back,
sometimes asks about things that do not exist, and there are moments where it interrupts, falls silent, or talks over the caller.
Both sides are given to you for a reason: channel 1 tells you what the caller was reacting to.

## Splits

`train` and `val` are speaker-disjoint: no caller appears in both. Judging uses a hidden set of calls from callers and voices that appear in neither split.

## Evaluation

Your system exposes `POST /detect`. It receives a stereo WAV clip (8 kHz, base64-encoded, channel 0 = caller, channel 1 = agent) and returns:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` is required. `confidence` is optional and used to break ties and reward calibration.

## Terms

Human callers volunteered, were told the call was recorded for an AI test, and used invented personal data. Do not try to identify anyone.
This dataset is provided for HackMTY 2026 only; do not redistribute.
