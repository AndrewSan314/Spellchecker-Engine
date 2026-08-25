// Task 9 — tuning determinism guarantees:
//   1. identical config overrides always produce identical engine output
//      (config churn via reload() cannot change decisions behind the same
//      snapshot values);
//   2. reload({}) restores default behavior exactly;
//   3. the recorded tuning artifact (when present) is schema-sane: every
//      trial carries finite metrics and a boolean feasibility flag, so the
//      search result is reproducible/auditable.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';

const SENTENCES = [
  'Dat ban toi nay giam 10%',
  'Quy khach vui long kiem tra lai thong tin',
  'Kinh chao quy khách, mua ngay tai shop',
];
const OVERRIDES = { linguistic: { spellingCheapTopK: 2, spellingMinConfidence: 0.65 } };

function issuesJson(engine, text) {
  const res = engine.validate(new ValidationContext(text, 'ACCENTED', 'VT_TENDOO'));
  return JSON.stringify(res.issues.map((i) => [i.ruleId, i.start, i.end, i.value,
    i.suggestions, i.confidence]));
}

test('same overrides -> byte-identical issue sets across reloads', () => {
  const engine = createDefaultEngine();
  const first = SENTENCES.map((s) => {
    engine.configService.reload(OVERRIDES);
    return issuesJson(engine, s);
  });
  const second = SENTENCES.map((s) => {
    engine.configService.reload(OVERRIDES);
    return issuesJson(engine, s);
  });
  assert.deepEqual(second, first);
});

test('reload({}) restores exact default behavior', () => {
  const engine = createDefaultEngine();
  const before = SENTENCES.map((s) => {
    engine.configService.reload({});
    return issuesJson(engine, s);
  });
  // perturb, then restore
  for (const s of SENTENCES) {
    engine.configService.reload(OVERRIDES);
    void issuesJson(engine, s);
  }
  const after = SENTENCES.map((s) => {
    engine.configService.reload({});
    return issuesJson(engine, s);
  });
  assert.deepEqual(after, before);
});

test('tuning artifact is schema-sane when present', () => {
  const p = path.resolve('dataset_artifacts/evaluation/tuning-run.json');
  if (!existsSync(p)) return; // tuner not yet run in this workspace
  const r = JSON.parse(readFileSync(p, 'utf8'));
  assert.equal(r.objective, 'crossLane F0.5 on VSEC dev');
  assert.ok(Array.isArray(r.trials) && r.trials.length >= 10,
    'coordinate+grid search must record its trials');
  for (const t of r.trials) {
    assert.equal(typeof t.ok, 'boolean');
    for (const k of ['f05', 'precision', 'recall']) {
      assert.equal(Number.isFinite(t[k]), true, `trial ${k} must be finite`);
    }
    assert.ok(t.overrides && typeof t.overrides === 'object');
  }
  // winner must equal the best feasible trial (deterministic selection)
  const feasible = r.trials.filter((t) => t.ok);
  const bestF05 = Math.max(...feasible.map((t) => t.f05));
  assert.ok(Math.abs(r.winner.metrics.f05 - bestF05) < 1e-9,
    'winner f05 must match the best recorded feasible trial');
});
