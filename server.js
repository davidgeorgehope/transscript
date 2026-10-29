import "dotenv/config";

import express from "express";
import multer from "multer";
import OpenAI from "openai";

const XAI_BASE_URL = "https://api.x.ai/v1";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WHISPER_MAX_BYTES = 25_000_000;
const DEFAULT_UPLOAD_MAX_MB = 512;
const DEFAULT_CHUNK_SECONDS = 20 * 60;
const DEFAULT_TRANSCODE_BITRATE = "64k";
const DEFAULT_TRANSCODE_SAMPLE_RATE = 16000;

const DEFAULT_SUMMARY_MODEL_CANDIDATES = ["grok-4.7"];

const DEFAULT_REASONING_EFFORT = "high";
const MODEL_LIST_CACHE_TTL_MS = 10 * 60 * 1000;

let cachedModels = null;
let cachedModelsAtMs = 0;

let cachedFfmpegAvailable = null;

function positiveInt(value, fallback) {
  const n = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function httpError(status, message) {
  const err = new Error(message);
  err.httpStatus = status;
  return err;
}

function normalizeReasoningEffort(value) {
  const effort = String(value ?? "")
    .trim()
    .toLowerCase();
  return effort === "low" || effort === "medium" || effort === "high" ? effort : "";
}

function uniqueStrings(values) {
  const out = [];
  const seen = new Set();
  for (const v of values) {
    const s = typeof v === "string" ? v.trim() : "";
    if (!s) continue;
    if (seen.has(s)) continue;
    seen.add(s);
    out.push(s);
  }
  return out;
}

async function listModelsCached(openai) {
  const now = Date.now();
  if (cachedModels && now - cachedModelsAtMs < MODEL_LIST_CACHE_TTL_MS) {
    return cachedModels;
  }

  const resp = await openai.models.list();
  const models = Array.isArray(resp?.data) ? resp.data : [];
  cachedModels = models;
  cachedModelsAtMs = now;
  return models;
}

function pickBestModelId(models, wantedIdOrPrefix) {
  const wanted = String(wantedIdOrPrefix ?? "").trim();
  if (!wanted) return "";

  const exact = models.find((m) => m?.id === wanted);
  if (exact?.id) return exact.id;

  const matches = models
    .filter((m) => typeof m?.id === "string")
    .filter((m) => m.id.startsWith(`${wanted}-`) || m.id.includes(wanted))
    .map((m) => ({
      id: m.id,
      created: typeof m.created === "number" ? m.created : 0
    }))
    .sort((a, b) => b.created - a.created);

  return matches[0]?.id ?? "";
}

function extractResponseText(response) {
  const direct = response?.output_text;
  if (typeof direct === "string") return direct.trim();

  const output = response?.output;
  if (!Array.isArray(output)) return "";

  let text = "";
  for (const item of output) {
    if (item?.type !== "message") continue;
    const content = item?.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (part?.type === "output_text" && typeof part.text === "string") text += part.text;
      if (part?.type === "text" && typeof part.text === "string") text += part.text;
    }
  }

  return text.trim();
}

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderrTail = "";

    child.on("error", reject);
    child.stderr.on("data", (chunk) => {
      stderrTail += chunk.toString("utf8");
      if (stderrTail.length > 16_384) stderrTail = stderrTail.slice(-16_384);
    });

    child.on("close", (code) => {
      if (code === 0) return resolve({ stderr: stderrTail });
      const err = new Error(`${command} failed (exit ${code}). ${stderrTail.trim()}`);
      err.exitCode = code;
      err.stderr = stderrTail;
      return reject(err);
    });
  });
}

async function isFfmpegAvailable() {
  if (cachedFfmpegAvailable !== null) return cachedFfmpegAvailable;
  try {
    await runProcess("ffmpeg", ["-version"]);
    cachedFfmpegAvailable = true;
  } catch {
    cachedFfmpegAvailable = false;
  }
  return cachedFfmpegAvailable;
}

async function transcodeToMp3(inputPath, outputPath, { bitrate, sampleRate }) {
  await runProcess("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    inputPath,
    "-map",
    "0:a:0",
    "-vn",
    "-ac",
    "1",
    "-ar",
    String(sampleRate),
    "-b:a",
    bitrate,
    outputPath
  ]);
}

async function segmentMp3(inputPath, outputPatternPath, chunkSeconds) {
  await runProcess("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    inputPath,
    "-f",
    "segment",
    "-segment_time",
    String(chunkSeconds),
    "-reset_timestamps",
    "1",
    "-c",
    "copy",
    outputPatternPath
  ]);
}

async function transcribeFilePath({ openai, transcriptionModel, filePath, fileName, mimeType }) {
  // xAI speech-to-text (POST /v1/stt, multipart: model + file)
  const buf = await fsp.readFile(filePath);
  const form = new FormData();
  form.append("model", transcriptionModel);
  form.append("file", new Blob([buf], { type: mimeType || "application/octet-stream" }), fileName);
  const resp = await fetch(`${XAI_BASE_URL}/stt`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.XAI_API_KEY}` },
    body: form
  });
  if (!resp.ok) throw new Error(`xAI STT ${resp.status}: ${(await resp.text()).slice(0, 300)}`);
  const transcription = await resp.json();
  return transcription?.text?.trim?.() ? transcription.text : "";
}

async function safeRm(targetPath) {
  try {
    await fsp.rm(targetPath, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

async function transcribeWithChunkingIfNeeded({
  openai,
  transcriptionModel,
  uploadedPath,
  originalName,
  mimeType
}) {
  const stat = await fsp.stat(uploadedPath);
  if (stat.size <= WHISPER_MAX_BYTES) {
    return await transcribeFilePath({
      openai,
      transcriptionModel,
      filePath: uploadedPath,
      fileName: originalName ?? path.basename(uploadedPath),
      mimeType
    });
  }

  const ok = await isFfmpegAvailable();
  if (!ok) {
    const sizeMb = (stat.size / 1024 / 1024).toFixed(1);
    throw httpError(
      400,
      `File is ${sizeMb}MB. OpenAI Whisper accepts up to 25MB per request, so this app needs ffmpeg to downsample+chunk larger files. Install ffmpeg (macOS: brew install ffmpeg) or upload a smaller/trimmed file.`
    );
  }

  const chunkSeconds = positiveInt(process.env.OPENAI_CHUNK_SECONDS, DEFAULT_CHUNK_SECONDS);
  const bitrate = String(process.env.OPENAI_TRANSCODE_BITRATE ?? DEFAULT_TRANSCODE_BITRATE);
  const sampleRate = positiveInt(process.env.OPENAI_TRANSCODE_SAMPLE_RATE, DEFAULT_TRANSCODE_SAMPLE_RATE);

  const jobDir = path.join(os.tmpdir(), "transscript-jobs", crypto.randomUUID());
  await fsp.mkdir(jobDir, { recursive: true });

  try {
    const mp3Path = path.join(jobDir, "audio.mp3");
    await transcodeToMp3(uploadedPath, mp3Path, { bitrate, sampleRate });

    const mp3Stat = await fsp.stat(mp3Path);
    if (mp3Stat.size <= WHISPER_MAX_BYTES) {
      return await transcribeFilePath({
        openai,
        transcriptionModel,
        filePath: mp3Path,
        fileName: "audio.mp3",
        mimeType: "audio/mpeg"
      });
    }

    const partsDir = path.join(jobDir, "parts");
    await fsp.mkdir(partsDir, { recursive: true });

    const pattern = path.join(partsDir, "chunk-%03d.mp3");
    await segmentMp3(mp3Path, pattern, chunkSeconds);

    const partNames = (await fsp.readdir(partsDir))
      .filter((n) => n.endsWith(".mp3"))
      .sort((a, b) => a.localeCompare(b, "en"));

    if (!partNames.length) throw new Error("Chunking produced no audio parts.");

    const chunks = [];
    for (const name of partNames) {
      const partPath = path.join(partsDir, name);
      const partStat = await fsp.stat(partPath);
      if (partStat.size > WHISPER_MAX_BYTES) {
        throw new Error(
          `Chunk ${name} is ${(partStat.size / 1024 / 1024).toFixed(1)}MB which exceeds the 25MB upload limit. Reduce OPENAI_CHUNK_SECONDS or OPENAI_TRANSCODE_BITRATE.`
        );
      }

      const text = await transcribeFilePath({
        openai,
        transcriptionModel,
        filePath: partPath,
        fileName: name,
        mimeType: "audio/mpeg"
      });
      if (text.trim()) chunks.push(text.trim());
    }

    return chunks.join("\n\n");
  } finally {
    await safeRm(jobDir);
  }
}

async function generateDocumentText({ openai, model, transcriptText, reasoningEffort }) {
  const prompt = [
    "Rewrite the transcript into a clean, readable document.",
    "Use clear headings and paragraphs.",
    "Keep it faithful to the source: do not invent facts, names, decisions, or action items.",
    "If something is unclear, say so.",
    "Prefer Markdown output with sections like: Title, Executive Summary, Key Points, Decisions, Action Items, Open Questions.",
    "",
    "Transcript:",
    "",
    transcriptText
  ].join("\n");

  const effort = normalizeReasoningEffort(reasoningEffort);
  const temperature = 0.2;

  const messages = [
    {
      role: "system",
      content:
        "Rewrite the transcript into a clean, readable document. Use clear headings and paragraphs. Keep it faithful to the source: do not invent facts, names, decisions, or action items. If something is unclear, say so. Prefer Markdown output with sections like: Title, Executive Summary, Key Points, Decisions, Action Items, Open Questions."
    },
    {
      role: "user",
      content: `Transcript:\n\n${transcriptText}`
    }
  ];

  const base = { model, temperature, messages };
  const attempts = effort ? [{ ...base, reasoning_effort: effort }, base] : [base];

  let lastErr;
  for (const params of attempts) {
    try {
      const completion = await openai.chat.completions.create(params);
      return completion?.choices?.[0]?.message?.content?.trim?.() ?? "";
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr ?? new Error("Failed to generate document.");
}

const app = express();
app.disable("x-powered-by");

const uploadDir = path.join(os.tmpdir(), "transscript-uploads");
fs.mkdirSync(uploadDir, { recursive: true });
const uploadMaxMb = positiveInt(process.env.UPLOAD_MAX_MB, DEFAULT_UPLOAD_MAX_MB);

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, uploadDir),
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname ?? "");
      cb(null, `${crypto.randomUUID()}${ext}`);
    }
  }),
  limits: { fileSize: uploadMaxMb * 1024 * 1024 }
});

app.get("/health", (_req, res) => res.json({ ok: true }));

app.use(express.static(path.join(__dirname, "public")));

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  const uploadedPath = req.file?.path;
  try {
    if (!process.env.XAI_API_KEY) {
      return res.status(500).json({
        error: "Missing XAI_API_KEY. Put it in your .env or environment."
      });
    }

    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded (field name: audio)" });
    }

    const mime = req.file.mimetype ?? "";
    if (!mime.startsWith("audio/") && !mime.startsWith("video/")) {
      return res.status(400).json({ error: `Unsupported file type: ${mime}` });
    }

    const transcriptionModel = process.env.XAI_TRANSCRIPTION_MODEL ?? "grok-voice-transcribe-2.0";
    const requestedSummaryModel = process.env.XAI_SUMMARY_MODEL?.trim?.() ?? "";
    const reasoningEffort = normalizeReasoningEffort(
      process.env.XAI_REASONING_EFFORT ?? DEFAULT_REASONING_EFFORT
    );
    const openai = new OpenAI({ apiKey: process.env.XAI_API_KEY, baseURL: XAI_BASE_URL });

    const transcriptText = await transcribeWithChunkingIfNeeded({
      openai,
      transcriptionModel,
      uploadedPath: req.file.path,
      originalName: req.file.originalname,
      mimeType: mime
    });

    if (!transcriptText) {
      return res.status(502).json({ error: "Transcription returned empty text." });
    }

    let documentText = "";
    let documentError = "";
    let documentModel = "";

    const availableModels = requestedSummaryModel
      ? []
      : await listModelsCached(openai).catch(() => []);

    const resolvedModelCandidates = requestedSummaryModel
      ? []
      : DEFAULT_SUMMARY_MODEL_CANDIDATES.map((m) => pickBestModelId(availableModels, m));

    const summaryModelCandidates = requestedSummaryModel
      ? [requestedSummaryModel]
      : uniqueStrings([...resolvedModelCandidates, ...DEFAULT_SUMMARY_MODEL_CANDIDATES]);

    try {
      for (const candidate of summaryModelCandidates) {
        try {
          documentText = await generateDocumentText({
            openai,
            model: candidate,
            transcriptText,
            reasoningEffort
          });
          documentModel = candidate;
          break;
        } catch (err) {
          documentError = err?.message ?? "Failed to generate document.";
          if (requestedSummaryModel) break;

          const status = Number(err?.status ?? err?.code) || 0;
          const msg = String(err?.message ?? "").toLowerCase();
          const maybeModelIssue =
            status === 404 ||
            (status === 400 && msg.includes("model") && (msg.includes("not found") || msg.includes("exist")));

          if (!maybeModelIssue) break;
        }
      }
    } catch (err) {
      documentError = err?.message ?? "Failed to generate document.";
    }

    return res.json({
      transcript: transcriptText,
      document: documentText,
      ...(documentModel ? { documentModel } : {}),
      ...(documentError ? { documentError } : {})
    });
  } catch (err) {
    const status = Number(err?.httpStatus ?? err?.status ?? err?.code) || 500;
    const message = err?.message ?? "Request failed.";
    return res.status(status).json({ error: message });
  } finally {
    if (uploadedPath) await safeRm(uploadedPath);
  }
});

app.use((err, _req, res, _next) => {
  if (err?.name === "MulterError") {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({
        error: `Upload too large. Increase UPLOAD_MAX_MB (currently ${uploadMaxMb}MB).`
      });
    }
    return res.status(400).json({ error: err.message });
  }
  console.error(err);
  return res.status(500).json({ error: "Unexpected server error" });
});

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  console.log(`Transscript listening on http://localhost:${port}`);
});
