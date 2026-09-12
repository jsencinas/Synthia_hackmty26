from __future__ import annotations

from pathlib import Path


class AudioPipeline:
    """
    Future audio pipeline boundary.

    v0.1 deliberately does not fabricate turns from audio. Later versions can
    implement VAD, diarization, channel separation and/or Whisper here without
    changing the model/risk/API contracts.
    """

    def analyze(self, wav_path: Path) -> dict:
        raise NotImplementedError(
            "Audio turn extraction is intentionally not implemented in v0.1."
        )
