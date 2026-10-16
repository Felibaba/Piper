// Thin Node.js front-end for Piper's own HTTP server (model stays loaded there).
// No npm dependencies. Needs Node 18+.
//
//   POST /tts   body: plain text  ->  audio/wav
//   GET  /health
//
// Env: PIPER_URL (default http://127.0.0.1:5000), PORT (default 3000)

import http from "node:http";

const PIPER_URL = process.env.PIPER_URL || "http://127.0.0.1:5000";
const PORT = Number(process.env.PORT || 3000);
const MAX_CHARS = 1000;
const MAX_BODY = 16 * 1024;

let queue = Promise.resolve(); // one synthesis at a time to protect RAM

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (c) => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new Error("too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function synthesize(text) {
  const r = await fetch(PIPER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!r.ok) throw new Error(`piper returned ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

http
  .createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "Content-Type": "text/plain" }).end("ok");
      return;
    }
    if (req.method !== "POST" || !req.url.startsWith("/tts")) {
      res.writeHead(404).end();
      return;
    }
    try {
      const text = (await readBody(req)).slice(0, MAX_CHARS).trim();
      if (!text) {
        res.writeHead(400, { "Content-Type": "text/plain" }).end("empty text");
        return;
      }
      const job = queue.then(() => synthesize(text));
      queue = job.catch(() => {}); // keep the queue alive after failures
      const wav = await job;
      res.writeHead(200, {
        "Content-Type": "audio/wav",
        "Content-Length": wav.length,
      });
      res.end(wav);
    } catch (err) {
      console.error(err.message);
      res.writeHead(500, { "Content-Type": "text/plain" }).end("synthesis failed");
    }
  })
  .listen(PORT, () => console.log(`TTS wrapper on :${PORT} -> ${PIPER_URL}`));
