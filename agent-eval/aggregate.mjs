// Aggregate the 3 independent QA agents' results into an average report.
// Usage: node agent-eval/aggregate.mjs
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const AGENTS = [
  { id: 'agent1', label: 'Deterministic rules', dir: path.join(HERE, 'agent1') },
  { id: 'agent2', label: 'Missing diacritic', dir: path.join(HERE, 'agent2') },
  { id: 'agent3', label: 'Adversarial precision', dir: path.join(HERE, 'agent3') },
];

function pct(x) { return (x * 100).toFixed(1) + '%'; }
function avg(list) { return list.length ? list.reduce((a, b) => a + b, 0) / list.length : null; }

const reports = [];
for (const a of AGENTS) {
  const file = path.join(a.dir, 'result.json');
  if (!existsSync(file)) {
    reports.push({ ...a, missing: true });
    continue;
  }
  const raw = JSON.parse(readFileSync(file, 'utf8'));
  // normalize across agent schemas
  let r;
  if (raw.summary) {
    // agent3 schema
    const s = raw.summary;
    r = {
      rowCount: s.totalRows,
      expectedCount: s.plantedExpectations,
      caughtCount: s.plantedMatched,
      recall: s.plantedExpectations ? s.plantedMatched / s.plantedExpectations : null,
      falsePositiveCount: s.falsePositiveCount,
      falsePositives: raw.falsePositives ?? [],
      precision: (s.plantedMatched + s.falsePositiveCount) > 0
        ? s.plantedMatched / (s.plantedMatched + s.falsePositiveCount)
        : null,
    };
  } else {
    r = raw;
  }
  reports.push({ ...a, r });
}

console.log('════════ 3-AGENT INDEPENDENT EVALUATION ════════\n');
for (const rep of reports) {
  if (rep.missing) { console.log(`[${rep.id}] ${rep.label}: result.json NOT FOUND`); continue; }
  const r = rep.r;
  console.log(`[${rep.id}] ${rep.label}`);
  console.log(`   rows=${r.rowCount ?? '?'} expected=${r.expectedCount ?? '?'} caught=${r.caughtCount ?? '?'}`
    + ` recall=${pct(r.recall)} FP=${r.falsePositiveCount ?? (r.falsePositives?.length ?? '?')}`
    + ` precision=${pct(r.precision)}`);
  if (r.unitTests) {
    console.log(`   unit tests: ${JSON.stringify(r.unitTests)}`);
  }
  if (r.officialBenchmark) {
    console.log(`   official bench: ${typeof r.officialBenchmark === 'string' ? r.officialBenchmark : JSON.stringify(r.officialBenchmark)}`);
  }
  console.log('');
}

const valid = reports.filter((x) => !x.missing && typeof x.r?.recall === 'number');
if (valid.length) {
  const recalls = valid.map((x) => x.r.recall);
  const precisions = valid.map((x) => x.r.precision);
  const fps = valid.map((x) => x.r.falsePositiveCount ?? x.r.falsePositives?.length ?? 0);
  console.log('──────── AVERAGES ────────');
  console.log(`agents reporting : ${valid.length}/${AGENTS.length}`);
  console.log(`avg recall       : ${pct(avg(recalls))}  (${recalls.map(pct).join(', ')})`);
  console.log(`avg precision    : ${pct(avg(precisions))}  (${precisions.map(pct).join(', ')})`);
  console.log(`avg FP count     : ${avg(fps).toFixed(1)}  (${fps.join(', ')})`);
  console.log(`total FP         : ${fps.reduce((a, b) => a + b, 0)}`);
}

// findings digest
console.log('\n──────── FINDINGS ────────');
for (const rep of reports) {
  const f = path.join(rep.dir, 'findings.json');
  if (!existsSync(f)) continue;
  let list = [];
  try { list = JSON.parse(readFileSync(f, 'utf8')); } catch { continue; }
  if (!Array.isArray(list) || list.length === 0) continue;
  console.log(`[${rep.id}] ${list.length} finding(s):`);
  for (const item of list.slice(0, 12)) {
    if (typeof item === 'string') {
      console.log(`   - ${item.slice(0, 160)}`);
      continue;
    }
    const kind = item.kind ?? item.type ?? '';
    const title = item.title ?? item.summary ?? item.description ?? '';
    const repro = item.text ?? item.repro ?? '';
    const line = [kind && `[${kind}]`, title, repro && `repro: ${String(repro).slice(0, 110)}`]
      .filter(Boolean).join(' ');
    console.log(`   - ${line.slice(0, 220)}`);
  }
}
