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

Your system exposes `POST /detect`. The judge sends one call per request, as JSON:

```json
{
  "call_id": "call_0181ce113ebe",
  "audio_base64": "<base64 of the complete WAV file, same format as audio/*.wav>",
  "sample_rate": 8000,
  "channels": 2
}
```

Channel 0 is the caller, channel 1 is the agent. Decode `audio_base64` and you have the exact bytes of a file from `audio/`.

Respond with HTTP 200 and:

```json
{"is_synthetic": true, "confidence": 0.87}
```

`is_synthetic` (boolean) is required. `confidence` (0 to 1, your certainty in the `is_synthetic` value you returned) is optional; when every answer carries one we also report AUC and calibration, and it breaks ties.

Rules the judge applies:

- 30 seconds per call. A timeout, a non-200 status, or a body without a boolean `is_synthetic` counts as a wrong answer.
- Calls are 1 to 4 minutes long; the JSON body is up to about 5 MB.
- The hidden set has callers and voices that are in neither `train` nor `val`. Main metric is balanced accuracy.
- Your endpoint has to stay reachable during your judging slot; we call it live from your station.

### Test your endpoint before judging

`scripts/check_endpoint.py` is the judge's client. Point it at your URL and it sends dataset calls, validates the response shape, and prints accuracy and latency:

```bash
python scripts/check_endpoint.py --url http://localhost:8000/detect --split val --n 20
```

`scripts/example_server.py` is a minimal server that implements the contract with a placeholder decision, so you can see the plumbing work end to end:

```bash
python scripts/example_server.py --port 8000
```

Both scripts need only the Python standard library and the unzipped `audio/` folder.

## Terms

Human callers volunteered, were told the call was recorded for an AI test, and used invented personal data. Do not try to identify anyone.
This dataset is provided for HackMTY 2026 only; do not redistribute.