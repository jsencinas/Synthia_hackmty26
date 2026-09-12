"""Human-vs-synthetic caller detector.

- ``config``: paths and API settings
- ``data``: manifest splits, audio files, leak checks
- ``features``: speaker turns from WAV, then timing and voice features
- ``models``: trained artifact I/O, temperature calibration, fusion stacker
- ``timing``: step 1, turn-timing head
- ``voice``: step 2, caller-voice head
- ``stt``: step 3, transcript head (ElevenLabs)
- ``detect``: run the three steps and fuse
- ``train``: fit models on the train split only
- ``evaluate``: score frozen artifacts on the val split
"""
