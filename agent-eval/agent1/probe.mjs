// QA probe (agent1) — dry-run planned corpus rows against the engine.
// Read-only with respect to src/, test/, benchmark/, public/, corpus*.json.
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';
import { readFileSync } from 'node:fs';

const engine = createDefaultEngine();

function asciiEscape(s) {
  let out = '';
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp > 126 || ch === '\\') {
      out += ch === '\\' ? '\\\\' : '\\u' + cp.toString(16).toUpperCase().padStart(4, '0');
    } else out += ch;
  }
  return out;
}

const rows = JSON.parse(readFileSync(new URL('./probe-rows.json', import.meta.url), 'utf8')).rows;

for (const r of rows) {
  const res = engine.validate(new ValidationContext(r.text, r.mode, r.brand ?? null));
  const brief = res.issues.map((i) =>
    `${i.ruleId}[${i.start},${i.end})"${asciiEscape(i.value)}"${i.severity[0]}`);
  console.log(`${r.id} n=${res.issues.length} :: ${brief.length ? brief.join(' | ') : '(none)'}`);
}
