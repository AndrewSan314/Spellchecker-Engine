#!/usr/bin/env node
// ============================================================
// Quick local checker — validate SMS text without starting the server.
//
//   npm run check -- "Ma OTP cua quy khach la 123456"
//   npm run check                      # interactive, one message per line
//   echo "..." | npm run check
//   npm run check -- --mode NON_ACCENTED "Ma OTP..."
//   npm run check -- --json "..."      # machine-readable
//
// ENGINE_PROFILE=lite (the npm script's default) loads the SMS-domain
// artifacts: ~0.7 s to start instead of ~6 s.
// ============================================================
import readline from 'node:readline';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';

const argv = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const value = argv[i + 1];
  argv.splice(i, 2);
  return value;
};
const has = (name) => {
  const i = argv.indexOf(`--${name}`);
  if (i < 0) return false;
  argv.splice(i, 1);
  return true;
};

const asJson = has('json');
const quiet = has('quiet');
const mode = (flag('mode', 'ACCENTED') ?? 'ACCENTED').toUpperCase();
const brand = flag('brand', 'VT_TENDOO');

const t0 = Date.now();
const engine = createDefaultEngine();
if (!quiet && !asJson) {
  const p = engine.profile;
  console.error(`[engine] profile=${p.name} lm=${p.lm.artifact} `
    + `lexicon=${p.lexicon?.artifactPath} ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

const COLORS = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = (text, code) => (COLORS ? `[${code}m${text}[0m` : text);

function underline(text, issues) {
  // one caret row per issue span, so offsets are visible at a glance
  const lines = [];
  for (const issue of issues) {
    const pad = ' '.repeat([...text.slice(0, issue.start)].length);
    const bar = '^'.repeat(Math.max(1, [...issue.value].length));
    lines.push(`  ${pad}${paint(bar, 33)} ${issue.ruleId}`);
  }
  return lines;
}

function report(text) {
  const t = performance.now();
  const result = engine.validate(new ValidationContext(text, mode, brand));
  const ms = performance.now() - t;

  if (asJson) {
    console.log(JSON.stringify({
      valid: result.valid, summary: result.summary, tookMs: +ms.toFixed(2),
      issues: result.issues.map((i) => ({
        ruleId: i.ruleId, severity: i.severity, start: i.start, end: i.end,
        value: i.value, message: i.message, suggestions: [...i.suggestions],
        confidence: i.confidence == null ? null : Math.round(i.confidence * 100) / 100,
      })),
    }));
    return;
  }

  const verdict = result.hasErrors ? paint('INVALID', 31)
    : result.hasWarnings ? paint('WARNINGS', 33) : paint('OK', 32);
  console.log(`\n${verdict}  ${result.issues.length} issue(s)  ${ms.toFixed(1)} ms`);
  console.log(`  ${text}`);
  for (const line of underline(text, result.issues)) console.log(line);
  if (result.summary?.unaccentedContent) {
    console.log(paint('  ⚠ nội dung có vẻ được gõ KHÔNG DẤU trong chế độ ACCENTED '
      + `(${result.summary.linguisticIssueCount} cảnh báo gộp lại thành một)`, 33));
  }
  for (const i of result.issues) {
    const conf = i.confidence == null ? '' : ` conf=${(Math.round(i.confidence * 100) / 100).toFixed(2)}`;
    console.log(`  [${i.start},${i.end}) ${paint(i.ruleId, 36)}${conf}`);
    console.log(`      ${i.message}`);
  }
}

const inline = argv.filter((a) => !a.startsWith('--')).join(' ').trim();
if (inline) {
  report(inline);
} else if (!process.stdin.isTTY) {
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of rl) if (line.trim()) report(line);
} else {
  console.log('Nhập nội dung SMS, mỗi dòng một tin (Ctrl+D để thoát):');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
  rl.prompt();
  rl.on('line', (line) => {
    if (line.trim()) report(line);
    rl.prompt();
  });
  await new Promise((resolve) => rl.on('close', resolve));
}
