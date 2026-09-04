// ============================================================
// Demo HTTP server — plan §26 REST contract
//   POST /api/v1/sms/content/validate
//   POST /api/v1/sms/content/benchmark   (demo-only, DEMO_ENDPOINTS=1)
//   GET  /api/v1/config                  (demo-only, DEMO_ENDPOINTS=1)
//   GET  /healthz                        liveness probe
//   GET  /                               demo UI
// Zero dependencies (node:http). Never mutates request content.
//
// Hardening notes (review A1-A4):
//   A1 the whole handler runs inside try/catch and the path is parsed from
//      req.url directly — a hostile/absent Host header can no longer throw an
//      unhandled ERR_INVALID_URL and kill the process.
//   A2 the body is collected as Buffers and decoded ONCE, so a chunk boundary
//      falling inside a multi-byte UTF-8 sequence can no longer corrupt
//      Vietnamese text into U+FFFD.
//   A3 the body limit is counted in BYTES, enforced while streaming, answers
//      413 and destroys the socket instead of letting the upload continue.
//   A4 the two demo-only endpoints are OFF unless DEMO_ENDPOINTS=1.
// ============================================================
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from './engine.mjs';
import { MessageMode } from './core.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(HERE, '..', 'public');
const PORT = Number(process.env.PORT ?? 3000);
// A3: an SMS is at most a few hundred characters; engine cost grows with the
// token count, so the request body is capped far below the old 1 MB.
export const MAX_BODY_BYTES = Number(process.env.MAX_BODY_BYTES ?? 16_384);
// A4: benchmark/config expose corpus-scale CPU work and internal thresholds.
export const DEMO_ENDPOINTS = process.env.DEMO_ENDPOINTS === '1';

const engine = createDefaultEngine();

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
  }
}

function json(res, status, body, { close = false } = {}) {
  const data = JSON.stringify(body);
  const headers = {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(data),
  };
  if (close) headers.connection = 'close';
  res.writeHead(status, headers);
  res.end(data);
}

/** A3: answer, flush, THEN drop the connection so the upload cannot continue. */
function jsonAndClose(req, res, status, body) {
  json(res, status, body, { close: true });
  res.on('finish', () => req.destroy());
}

/**
 * Display serialization (review B3): confidence is rounded HERE, for the UI.
 * The engine keeps the raw calibrated value so no threshold ever compares
 * against a number that was rounded up for display.
 */
export function serializeIssue(i) {
  return {
    ruleId: i.ruleId,
    severity: i.severity,
    start: i.start,
    end: i.end,
    value: i.value,
    message: i.message,
    suggestions: i.suggestions, // stable [] — plan §26
    confidence: i.confidence == null ? null : Math.round(i.confidence * 100) / 100,
  };
}

/**
 * A1: parse the path WITHOUT `new URL(...)`, which needs a valid Host header
 * and throws ERR_INVALID_URL on garbage. Only the path is ever used here.
 */
export function requestPath(rawUrl) {
  const withoutQuery = String(rawUrl ?? '/').split('?')[0].split('#')[0];
  try {
    return decodeURIComponent(withoutQuery);
  } catch {
    return withoutQuery; // malformed %-escape: keep the raw form, still safe
  }
}

/**
 * A2/A3: collect Buffers and decode once at the end. `data += chunk` would
 * call toString('utf8') per chunk and mangle any multi-byte character split
 * across a TCP boundary — nearly every Vietnamese character is multi-byte.
 */
export function readBody(req, maxBytes = MAX_BODY_BYTES) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let settled = false;
    const fail = (err) => {
      if (settled) return;
      settled = true;
      // A3: stop reading immediately. The socket is destroyed by the caller
      // AFTER the 413 has been flushed (destroying here races the response
      // off the wire and the client sees nothing).
      req.pause?.();
      reject(err);
    };
    req.on('data', (chunk) => {
      if (settled) return;
      size += chunk.length;
      if (size > maxBytes) {
        fail(new HttpError(413, `payload too large (max ${maxBytes} bytes)`));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks).toString('utf8'));
    });
    req.on('error', fail);
  });
}

async function handle(req, res) {
  const pathname = requestPath(req.url);

  if (req.method === 'GET' && (pathname === '/' || pathname === '/index.html')) {
    try {
      const html = readFileSync(path.join(PUBLIC, 'index.html'));
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(html);
    } catch {
      json(res, 500, { error: 'UI not found' });
    }
    return;
  }

  if (req.method === 'GET' && pathname === '/healthz') {
    json(res, 200, {
      status: 'ok',
      profile: engine.profile,
      degraded: engine.degradedSummary(),
    });
    return;
  }

  if (req.method === 'POST' && pathname === '/api/v1/sms/content/validate') {
    let body;
    try {
      body = JSON.parse((await readBody(req)) || '{}');
    } catch (err) {
      if (err?.status === 413) {
        jsonAndClose(req, res, 413, { error: err.message });
        return;
      }
      json(res, 400, { error: err?.message ?? 'bad request' });
      return;
    }
    // plan §27: invalid request => 400
    if (typeof body.content !== 'string') {
      json(res, 400, { error: '"content" (string) is required' });
      return;
    }
    if (!Object.values(MessageMode).includes(body.messageMode)) {
      json(res, 400, { error: '"messageMode" must be ACCENTED | NON_ACCENTED' });
      return;
    }
    try {
      // plan §7: client CANNOT send confidence thresholds — server config
      // is the single source of truth. Any options payload is ignored.
      const ctx = new ValidationContext(
        body.content,
        body.messageMode,
        body.brandname ?? null,
        body.customerId ?? null,
      );
      const t0 = performance.now();
      const result = engine.validate(ctx);
      const tookMs = performance.now() - t0;
      json(res, 200, {
        valid: result.valid,
        hasErrors: result.hasErrors,
        hasWarnings: result.hasWarnings,
        issues: result.issues.map(serializeIssue),
        // Review §7: an unaccented SMS validated in ACCENTED mode produces one
        // warning per token. The flag lets the UI show a single message-level
        // notice instead of flooding; issues are unchanged.
        summary: result.summary,
        tookMs: Math.round(tookMs * 100) / 100,
      });
    } catch (err) {
      if (err?.name === 'ValidationEngineError') {
        json(res, 500, { error: err.message });
        return;
      }
      json(res, 400, { error: err?.message ?? 'bad request' });
    }
    return;
  }

  if (req.method === 'POST' && pathname === '/api/v1/sms/content/benchmark') {
    if (!DEMO_ENDPOINTS) {
      json(res, 404, { error: 'not found' });
      return;
    }
    // Loaded lazily: the corpus is demo-only weight the server should not
    // carry when DEMO_ENDPOINTS is off.
    const { runBenchmark } = await import('../benchmark/run-benchmark.mjs');
    json(res, 200, runBenchmark(engine));
    return;
  }

  if (req.method === 'GET' && pathname === '/api/v1/config') {
    if (!DEMO_ENDPOINTS) {
      json(res, 404, { error: 'not found' });
      return;
    }
    const snap = engine.configService.snapshot();
    json(res, 200, {
      linguistic: snap.data.linguistic,
      rules: snap.data.rules,
      characters: snap.data.characters,
      spelling: snap.data.spelling,
      tuningSource: engine.configService.tuningSource ?? null,
    });
    return;
  }

  json(res, 404, { error: 'not found' });
}

export const server = createServer((req, res) => {
  // A1: nothing above this line may throw into the 'request' event.
  handle(req, res).catch((err) => {
    console.error('[server] unhandled request error:', err);
    if (res.headersSent) {
      res.destroy();
      return;
    }
    json(res, err?.status ?? 500, { error: err?.status ? err.message : 'internal error' });
  });
});

// Last-resort guards: a bad socket must never take the process down.
server.on('clientError', (err, socket) => {
  if (socket.writable) socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n');
  else socket.destroy();
});

// The module exports `server` so tests can bind an ephemeral port themselves
// (test/test_server_hardening.mjs); importing it must not start listening.
if (process.env.NODE_ENV !== 'test') {
  server.listen(PORT, () => {
    const p = engine.profile;
    console.log(`SMS Validation demo  ->  http://localhost:${PORT}`);
    console.log('POST /api/v1/sms/content/validate');
    console.log(`data profile: ${p.name}  (LM ${p.lm.artifact}`
      + ` U=${p.lm.counts?.U} B=${p.lm.counts?.B} T=${p.lm.counts?.T},`
      + ` lexicon ${p.lexicon?.artifactPath} ${p.lexicon?.entries} entries)`);
    console.log(`RSS after load: ${Math.round(process.memoryUsage().rss / 1e6)} MB`
      + `  ·  switch with ENGINE_PROFILE=full|lite`);
    console.log(`demo endpoints (/benchmark, /config): ${DEMO_ENDPOINTS ? 'ON' : 'OFF (set DEMO_ENDPOINTS=1)'}`);
    console.log(`max body: ${MAX_BODY_BYTES} bytes`);
  });
}
