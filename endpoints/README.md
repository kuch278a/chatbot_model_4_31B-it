# src/endpoints

End-to-end API endpoints that chain multiple services into a single request/response cycle.

---

## talk.py — Voice conversation endpoint

```
POST /tesfansh-api/api/v1/chat/talk
Content-Type: multipart/form-data
```

### Form fields

| Field | Required | Default | Description |
|---|---|---|---|
| `audio` | yes | — | Audio file. WAV preferred; any ffmpeg-decodable format works (WebM, MP4, OGG, …) |
| `session_id` | no | `default_session` | Conversation session ID for history tracking |
| `lang` | no | `auto` | `auto` \| `am-ET` \| `en-US` |

### Response

Returns an **MP3 audio stream** (`audio/mpeg`) — the spoken version of the model's reply.

Three headers are also set for clients that want the text without a second request:

| Header | Content |
|---|---|
| `X-Transcript` | What the STT model heard |
| `X-Response` | LLM reply text (first 500 chars) |
| `X-Detected-Language` | `am-ET` or `en-US` |

### Internal pipeline

```
audio bytes
    │
    ▼
 STT (auto mode)
    ├─ Ethio-ASR  (badrex/Ethio-ASR-amharic)   ← tried first
    └─ Whisper    (distil-large-v3.5)           ← fallback if empty
    │
    ▼
 transcript text
    │
    ▼
 LLM  (Gemma 4 31B — ConversationService.chat)
    │   queued behind shared _llm_lock
    │   session history preserved in SQLite
    │
    ▼
 response text
    │
    ▼
 TTS  (Microsoft Edge TTS — am-ET-MekdesNeural default)
    │
    ▼
 MP3 audio response
```

### Quick test with curl

```bash
curl -X POST http://SERVER_IP:PORT/tesfansh-api/api/v1/chat/talk \
  -F "audio=@your_recording.wav" \
  -F "session_id=test-session" \
  --output reply.mp3
```
