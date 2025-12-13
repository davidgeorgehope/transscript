# Transscript

Simple web app: upload audio/video → get (1) a cleaned, document-style version and (2) the raw transcript.

## Setup

1. Create `.env`:
   - `cp .env.example .env`
   - Set `OPENAI_API_KEY=...`
2. (Optional, for large files) Install `ffmpeg` (macOS: `brew install ffmpeg`)
3. Install deps: `npm install`
4. Run: `npm start`
5. Open `http://localhost:3000`

## Configuration

- `OPENAI_API_KEY` (required)
- `PORT` (default: `3000`)
- `UPLOAD_MAX_MB` (default: `512`)
- `OPENAI_TRANSCRIPTION_MODEL` (default: `whisper-1`)
- `OPENAI_SUMMARY_MODEL` (default: prefers `gpt-5.2`, falls back if unavailable)
- `OPENAI_REASONING_EFFORT` (default: `high`)
- `OPENAI_CHUNK_SECONDS` (default: `1200`)
- `OPENAI_TRANSCODE_BITRATE` (default: `64k`)
- `OPENAI_TRANSCODE_SAMPLE_RATE` (default: `16000`)

## Notes

- Whisper accepts ~25MB per transcription request; larger uploads are downsampled+chunked (requires `ffmpeg` on your machine).
- To see the exact model IDs available to your API key: `npm run models`
