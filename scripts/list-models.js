import "dotenv/config";

import OpenAI from "openai";

if (!process.env.XAI_API_KEY) {
  console.error("Missing XAI_API_KEY (set it in .env or your environment).");
  process.exit(1);
}

const openai = new OpenAI({ apiKey: process.env.XAI_API_KEY, baseURL: "https://api.x.ai/v1" });

const models = await openai.models.list();
const rows = (models?.data ?? [])
  .map((m) => ({
    id: m.id,
    created: typeof m.created === "number" ? m.created : 0,
    owned_by: m.owned_by ?? ""
  }))
  .sort((a, b) => b.created - a.created);

for (const row of rows) {
  const created = row.created ? new Date(row.created * 1000).toISOString() : "";
  console.log([row.id, created, row.owned_by].filter(Boolean).join("\t"));
}

