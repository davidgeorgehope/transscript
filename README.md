# Transscript

Simple web app: upload audio/video → get (1) a cleaned, document-style version and (2) the raw transcript.

## Setup

1. Create `.env`:
   - `cp .env.example .env`
   - Set `XAI_API_KEY=...`
2. (Optional, for large files) Install `ffmpeg` (macOS: `brew install ffmpeg`)
3. Install deps: `npm install`
4. Run: `npm start`
5. Open `http://localhost:3000`

## Configuration

- `XAI_API_KEY` (required)
- `PORT` (default: `3000`)
- `UPLOAD_MAX_MB` (default: `512`)
- `XAI_TRANSCRIPTION_MODEL` (default: `grok-voice-transcribe-2.0`)
- `XAI_SUMMARY_MODEL` (default: `grok-4.7`)
- `XAI_REASONING_EFFORT` (default: `high`)
- `OPENAI_CHUNK_SECONDS` (default: `1200`)
- `OPENAI_TRANSCODE_BITRATE` (default: `64k`)
- `OPENAI_TRANSCODE_SAMPLE_RATE` (default: `16000`)

## Notes

- Whisper accepts ~25MB per transcription request; larger uploads are downsampled+chunked (requires `ffmpeg` on your machine).
- To see the exact model IDs available to your API key: `npm run models`
