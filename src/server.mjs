// ============================================================
// Demo HTTP server — plan §26 REST contract
//   POST /api/v1/sms/content/validate
//   POST /api/v1/sms/content/benchmark   (demo-only helper)
//   GET  /api/v1/config                  (demo-only: show thresholds)
//   GET  /                               demo UI
// Zero dependencies (node:http). Never mutates request content.
// ============================================================
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from './engine.mjs';
import { MessageMode } from './core.mjs';
import { runBenchmark } from '../benchmark/run-benchmark.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(HERE, '..', 'public');
const PORT = Number(process.env.PORT ?? 3000);

const engine = createDefaultEngine();

function json(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(data),
  });
  res.end(data);
}

function serializeIssue(i) {
  return {
    ruleId: i.ruleId,
    severity: i.severity,
    start: i.start,
    end: i.end,
    value: i.value,
    message: i.message,
    suggestions: i.suggestions, // stable [] — plan §26
    confidence: i.confidence,
  };
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 1_000_000) reject(new Error('payload too large'));
    });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
    try {
      const html = readFileSync(path.join(PUBLIC, 'index.html'));
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      res.end(html);
    } catch {
      json(res, 500, { error: 'UI not found' });
    }
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/v1/sms/content/validate') {
    try {
      const body = JSON.parse((await readBody(req)) || '{}');
      // plan §27: invalid request => 400
      if (typeof body.content !== 'string') {
        json(res, 400, { error: '"content" (string) is required' });
        return;
      }
      if (!Object.values(MessageMode).includes(body.messageMode)) {
        json(res, 400, { error: '"messageMode" must be ACCENTED | NON_ACCENTED' });
        return;
      }
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

  if (req.method === 'POST' && url.pathname === '/api/v1/sms/content/benchmark') {
    const report = runBenchmark(engine);
    json(res, 200, report);
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/v1/config') {
    const snap = engine.configService.snapshot();
    json(res, 200, {
      linguistic: snap.data.linguistic,
      rules: snap.data.rules,
      characters: snap.data.characters,
    });
    return;
  }

  json(res, 404, { error: 'not found' });
});

server.listen(PORT, () => {
  console.log(`SMS Validation demo  ->  http://localhost:${PORT}`);
  console.log('POST /api/v1/sms/content/validate');
  console.log('POST /api/v1/sms/content/benchmark');
});
