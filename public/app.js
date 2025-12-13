const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#audioFile");
const submitBtn = document.querySelector("#submitBtn");
const statusEl = document.querySelector("#status");

const documentOut = document.querySelector("#documentOut");
const transcriptOut = document.querySelector("#transcriptOut");
const copyDocBtn = document.querySelector("#copyDocBtn");
const copyTranscriptBtn = document.querySelector("#copyTranscriptBtn");

function setStatus(text, kind = "info") {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

async function copyToClipboard(text) {
  await navigator.clipboard.writeText(text);
}

copyDocBtn.addEventListener("click", async () => {
  await copyToClipboard(documentOut.textContent ?? "");
  setStatus("Copied document.", "success");
});

copyTranscriptBtn.addEventListener("click", async () => {
  await copyToClipboard(transcriptOut.textContent ?? "");
  setStatus("Copied transcript.", "success");
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) return;

  submitBtn.disabled = true;
  copyDocBtn.disabled = true;
  copyTranscriptBtn.disabled = true;

  documentOut.textContent = "";
  transcriptOut.textContent = "";
  setStatus("Processing…");

  const body = new FormData();
  body.append("audio", file, file.name);

  try {
    const resp = await fetch("/api/transcribe", { method: "POST", body });
    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      documentOut.textContent = "";
      transcriptOut.textContent = "";
      setStatus(data.error ?? `Request failed (${resp.status})`, "error");
      return;
    }

    documentOut.textContent = data.document ?? "";
    transcriptOut.textContent = data.transcript ?? "";

    if (data.documentError && !(data.document ?? "").trim()) {
      documentOut.textContent = `Document generation failed:\n${data.documentError}`;
    }

    copyDocBtn.disabled = !(data.document ?? "").trim();
    copyTranscriptBtn.disabled = !(data.transcript ?? "").trim();
    const modelSuffix = data.documentModel ? ` (model: ${data.documentModel})` : "";
    if (data.documentError) {
      setStatus(`Transcript ready; document had an error${modelSuffix}.`, "error");
    } else {
      setStatus(`Done${modelSuffix}.`, "success");
    }
  } catch (err) {
    setStatus(err?.message ?? "Network error", "error");
  } finally {
    submitBtn.disabled = false;
  }
});
