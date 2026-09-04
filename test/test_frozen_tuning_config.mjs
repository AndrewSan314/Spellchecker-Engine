// ============================================================
// Review B9 — "the frozen config" and "the config being served" must be the
// same thing. config/spelling-tuning.json declared itself frozen while
// nothing under src/ read it, so the runtime quietly served different
// attention thresholds than the calibration record.
//
// These tests fail the moment the two drift again.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  ValidationConfigService, loadFrozenTuning, TUNING_FILE, DEFAULT_SNAPSHOT,
} from '../src/config.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const TUNING_JSON = JSON.parse(readFileSync(path.join(ROOT, 'config', 'spelling-tuning.json'), 'utf8'));

test('the runtime loads config/spelling-tuning.json', () => {
  const service = new ValidationConfigService();
  assert.ok(service.tuningSource, 'ValidationConfigService must record its tuning source');
  assert.equal(service.tuningSource.file, 'spelling-tuning.json');
  assert.equal(service.tuningSource.frozen, true);
});

test('every frozen parameter is actually served (or listed as a serving override)', () => {
  const snap = new ValidationConfigService().snapshot();
  const overrides = TUNING_JSON.servingOverrides?.overrides ?? {};
  for (const [key, frozenValue] of Object.entries(TUNING_JSON.frozenParams)) {
    const served = snap.get(`linguistic.${key}`) ?? snap.get(`spelling.${key}`);
    if (key in overrides) {
      assert.deepEqual(served, overrides[key].value,
        `${key}: serving override declared but not served`);
      assert.ok(overrides[key].reason, `${key}: a serving override needs a reason`);
      assert.deepEqual(overrides[key].frozenValue, frozenValue,
        `${key}: servingOverrides.frozenValue must quote the frozen value`);
      continue;
    }
    assert.deepEqual(served, frozenValue,
      `${key}: frozen ${JSON.stringify(frozenValue)} but served ${JSON.stringify(served)}`);
  }
});

test('the calibrated confidence temperatures are served', () => {
  const snap = new ValidationConfigService().snapshot();
  for (const [key, value] of Object.entries(TUNING_JSON.confidenceCalibration.constants)) {
    assert.equal(snap.get(`linguistic.${key}`), value, key);
  }
});

test('the attention thresholds exist in DEFAULT_SNAPSHOT (no silent 0.80 fallback)', () => {
  for (const key of ['attentionMinProbability', 'attentionMinCandidateWindows', 'attentionMaxOriginalWindows']) {
    assert.notEqual(DEFAULT_SNAPSHOT.spelling[key], undefined, `spelling.${key}`);
    assert.notEqual(DEFAULT_SNAPSHOT.linguistic[key], undefined, `linguistic.${key}`);
  }
});

test('an unknown frozen key fails loudly instead of being ignored', () => {
  const dir = mkdtempSync(path.join(tmpdir(), 'tuning-'));
  const file = path.join(dir, 'bad.json');
  writeFileSync(file, JSON.stringify({
    version: 99, frozen: true, frozenParams: { thisKeyDoesNotExist: 1 },
  }));
  assert.throws(() => loadFrozenTuning(file), /unknown key "thisKeyDoesNotExist"/);
});

test('a missing tuning file falls back to the defaults', () => {
  assert.equal(loadFrozenTuning(path.join(tmpdir(), 'definitely-missing-tuning.json')), null);
  const service = new ValidationConfigService(null, { tuningFile: null });
  assert.equal(service.tuningSource, null);
  assert.equal(service.snapshot().get('linguistic.wrongDiacriticMode'), 'ACTIVE');
});

test('reload() keeps the frozen file as the base layer', () => {
  const service = new ValidationConfigService();
  const snap = service.reload({ linguistic: { mode: 'SHADOW' } });
  assert.equal(snap.get('linguistic.mode'), 'SHADOW');
  assert.equal(snap.get('linguistic.realWordTypoMinProbability'),
    TUNING_JSON.frozenParams.realWordTypoMinProbability);
});

test('TUNING_FILE points at the repo config by default', () => {
  assert.equal(path.basename(TUNING_FILE), 'spelling-tuning.json');
});
