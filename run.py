import base64, json, urllib.request

path = "audio/call_185e958a253b.wav"
payload = {
    "call_id": "call_185e958a253b",
    "audio_base64": base64.b64encode(open(path, "rb").read()).decode(),
    "sample_rate": 8000,
    "channels": 2,
}
req = urllib.request.Request(
    "https://governmental-hook-fill-instances.trycloudflare.com/detect",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req)))