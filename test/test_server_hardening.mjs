// ============================================================
// HTTP layer regression tests — review A1-A4.
//
// A1 a hostile/absent Host header must not kill the process
// A2 a body split mid-UTF-8-character must decode intact (Vietnamese)
// A3 an oversized body gets 413 and the socket is destroyed
// A4 the demo endpoints are OFF unless DEMO_ENDPOINTS=1
// ============================================================
process.env.NODE_ENV = 'test';

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';

const { server, readBody, requestPath, serializeIssue, MAX_BODY_BYTES, DEMO_ENDPOINTS } =
  await import('../src/server.mjs');

let port;
before(async () => {
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = server.address().port;
});
after(() => new Promise((resolve) => server.close(resolve)));

/** Sends raw bytes and returns the whole response text. */
function rawRequest(chunks, { delayMs = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const socket = net.connect(port, '127.0.0.1');
    let out = '';
    socket.setEncoding('utf8');
    socket.on('data', (d) => { out += d; });
    socket.on('error', (err) => {
      // A3 destroys the socket; whatever arrived first is still the answer
      if (out) resolve(out); else reject(err);
    });
    socket.on('close', () => resolve(out));
    socket.on('connect', async () => {
      for (const chunk of chunks) {
        if (socket.destroyed) break;
        socket.write(chunk);
        if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
      }
    });
    setTimeout(() => socket.destroy(), 30_000).unref();
  });
}

function statusOf(response) {
  return Number(response.split(' ')[1]);
}

function bodyOf(response) {
  const i = response.indexOf('\r\n\r\n');
  return i < 0 ? '' : response.slice(i + 4);
}

// ---------- A1 ----------
test('A1: an invalid Host header answers instead of crashing the process', async () => {
  const res = await rawRequest([
    'GET /api/v1/config HTTP/1.1\r\nHost: bad host\r\nConnection: close\r\n\r\n',
  ]);
  assert.ok(res.startsWith('HTTP/1.1'), `expected an HTTP response, got: ${JSON.stringify(res.slice(0, 80))}`);
  assert.ok([404, 400, 200].includes(statusOf(res)), res.split('\r\n')[0]);
  // the process is still serving
  const after_ = await rawRequest([
    'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n',
  ]);
  assert.equal(statusOf(after_), 200);
});

test('A1: requestPath never throws on hostile input', () => {
  assert.equal(requestPath('/api/v1/config?x=1#frag'), '/api/v1/config');
  assert.equal(requestPath('/%E1%BA%A1'), '/ạ');
  assert.equal(requestPath('/%%%'), '/%%%'); // malformed escape, no throw
  assert.equal(requestPath(undefined), '/');
});

// ---------- A2 ----------
test('A2: a body split inside a multi-byte character decodes intact', async () => {
  const payload = Buffer.from(JSON.stringify({
    content: 'Kính chào quý khách', messageMode: 'ACCENTED',
  }), 'utf8');
  // cut in the middle of the "í" (byte 2 of a 2-byte sequence)
  const cut = payload.indexOf(Buffer.from('í', 'utf8')) + 1;
  const head = `POST /api/v1/sms/content/validate HTTP/1.1\r\nHost: localhost\r\n`
    + `Content-Type: application/json\r\nContent-Length: ${payload.length}\r\n`
    + 'Connection: close\r\n\r\n';
  const res = await rawRequest(
    [head, payload.subarray(0, cut), payload.subarray(cut)], { delayMs: 20 },
  );
  assert.equal(statusOf(res), 200, res.split('\r\n')[0]);
  const parsed = JSON.parse(bodyOf(res));
  // U+FFFD would show up as INVALID_CHARACTER / a shifted offset
  assert.ok(!JSON.stringify(parsed).includes('�'), 'response contains a replacement character');
  assert.ok(!parsed.issues.some((i) => i.ruleId === 'INVALID_CHARACTER'),
    `unexpected INVALID_CHARACTER: ${JSON.stringify(parsed.issues)}`);
});

test('A2: readBody concatenates Buffers before decoding', async () => {
  const bytes = Buffer.from('{"content":"Kính"}', 'utf8');
  const at = bytes.indexOf(Buffer.from('í', 'utf8')) + 1;
  const fake = new (await import('node:events')).EventEmitter();
  const promise = readBody(fake);
  fake.emit('data', bytes.subarray(0, at));
  fake.emit('data', bytes.subarray(at));
  fake.emit('end');
  assert.equal(await promise, '{"content":"Kính"}');
});

// ---------- A3 ----------
test('A3: an oversized body is rejected with 413', async () => {
  const content = 'a'.repeat(MAX_BODY_BYTES + 4096);
  const payload = Buffer.from(JSON.stringify({ content, messageMode: 'ACCENTED' }), 'utf8');
  const head = `POST /api/v1/sms/content/validate HTTP/1.1\r\nHost: localhost\r\n`
    + `Content-Type: application/json\r\nContent-Length: ${payload.length}\r\n`
    + 'Connection: close\r\n\r\n';
  const res = await rawRequest([head, payload]);
  assert.equal(statusOf(res), 413, res.split('\r\n')[0]);
});

test('A3: readBody rejects and stops reading past the byte limit', async () => {
  const fake = new (await import('node:events')).EventEmitter();
  let paused = false;
  fake.pause = () => { paused = true; };
  const promise = readBody(fake, 8);
  fake.emit('data', Buffer.alloc(9));
  await assert.rejects(promise, (err) => err.status === 413 && /payload too large/.test(err.message));
  assert.equal(paused, true, 'the request stream must be paused, not drained');
  // late chunks after the rejection must not resolve the promise again
  fake.emit('data', Buffer.alloc(4));
  fake.emit('end');
});

// ---------- A4 ----------
test('A4: demo endpoints are off by default', async () => {
  assert.equal(DEMO_ENDPOINTS, false, 'test env must not set DEMO_ENDPOINTS');
  const cfg = await rawRequest([
    'GET /api/v1/config HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n',
  ]);
  assert.equal(statusOf(cfg), 404);
  const bench = await rawRequest([
    'POST /api/v1/sms/content/benchmark HTTP/1.1\r\nHost: localhost\r\n'
    + 'Content-Length: 0\r\nConnection: close\r\n\r\n',
  ]);
  assert.equal(statusOf(bench), 404);
});

// ---------- B3 ----------
test('B3: confidence is rounded for display only, at the serialization edge', () => {
  const wire = serializeIssue({
    ruleId: 'POSSIBLE_SPELLING_ERROR', severity: 'WARNING', start: 0, end: 3,
    value: 'abc', message: 'x', suggestions: [], confidence: 0.9551,
  });
  assert.equal(wire.confidence, 0.96);
  assert.equal(serializeIssue({
    ruleId: 'MULTIPLE_WHITESPACE', severity: 'WARNING', start: 0, end: 1,
    value: ' ', message: 'x', suggestions: [], confidence: null,
  }).confidence, null);
});

// ---------- contract ----------
test('validate still answers 400 on a bad request shape', async () => {
  const payload = Buffer.from(JSON.stringify({ content: 'hi' }), 'utf8');
  const res = await rawRequest([
    `POST /api/v1/sms/content/validate HTTP/1.1\r\nHost: localhost\r\n`
    + `Content-Length: ${payload.length}\r\nConnection: close\r\n\r\n`,
    payload,
  ]);
  assert.equal(statusOf(res), 400);
});

test('the response carries the message-level unaccented summary (review §7)', async () => {
  const payload = Buffer.from(JSON.stringify({
    content: 'Ma OTP cua quy khach la 123456, hieu luc 5 phut.', messageMode: 'ACCENTED',
  }), 'utf8');
  const res = await rawRequest([
    `POST /api/v1/sms/content/validate HTTP/1.1\r\nHost: localhost\r\n`
    + `Content-Length: ${payload.length}\r\nConnection: close\r\n\r\n`,
    payload,
  ]);
  assert.equal(statusOf(res), 200);
  const parsed = JSON.parse(bodyOf(res));
  assert.equal(parsed.summary.unaccentedContent, true);
  assert.ok(parsed.summary.linguisticIssueCount > 0);
});
