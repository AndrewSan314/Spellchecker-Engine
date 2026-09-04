// ============================================================
// Contracts for the in-repo SMS dataset and the "lite" serving profile.
//
//   - the generator is deterministic (same seed => same bytes)
//   - NO template group appears in two splits (the leakage the review flagged
//     on VSEC is designed out here, not apologised for later)
//   - the LM training corpus contains ONLY train-split text
//   - labels really describe the difference between text and clean
//   - ENGINE_PROFILE picks artifacts, falls back loudly, never silently
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { splitOf, telexTypo } from '../tools/build_sms_dataset.mjs';
import { resolveArtifact, profileName, PROFILES, resetProfileWarnings } from '../src/profile.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const DS = path.join(ROOT, 'dataset_sms');

const readJsonl = (file) => readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const SPLITS = ['train', 'dev', 'test'];
const HAS_DATASET = SPLITS.every((s) => existsSync(path.join(DS, `sms-${s}.jsonl`)));
const skipNoDataset = HAS_DATASET ? false : 'dataset_sms not built (node tools/build_sms_dataset.mjs)';

test('split assignment is a pure function of the group name', { skip: skipNoDataset }, () => {
  assert.equal(splitOf('promo-discount'), splitOf('promo-discount'));
  const counts = { train: 0, dev: 0, test: 0 };
  for (const g of Array.from({ length: 200 }, (_, i) => `g${i}`)) counts[splitOf(g)] += 1;
  assert.ok(counts.train > counts.dev && counts.train > counts.test, JSON.stringify(counts));
  assert.ok(counts.dev > 0 && counts.test > 0, JSON.stringify(counts));
});

test('no template group appears in more than one split', { skip: skipNoDataset }, () => {
  const seen = new Map();
  for (const split of SPLITS) {
    for (const row of readJsonl(path.join(DS, `sms-${split}.jsonl`))) {
      const prev = seen.get(row.group);
      assert.ok(prev === undefined || prev === split,
        `group "${row.group}" appears in both ${prev} and ${split}`);
      seen.set(row.group, split);
      assert.equal(row.split, split, `${row.id} carries the wrong split label`);
    }
  }
  assert.ok(seen.size >= 10, `expected many groups, got ${seen.size}`);
});

test('the LM training corpus contains only train-split text', { skip: skipNoDataset }, () => {
  const corpus = new Set(readFileSync(path.join(DS, 'sms-clean-train.txt'), 'utf8')
    .split('\n').map((l) => l.trim()).filter(Boolean));
  for (const split of ['dev', 'test']) {
    for (const row of readJsonl(path.join(DS, `sms-${split}.jsonl`))) {
      assert.ok(!corpus.has(row.clean.trim()),
        `${split} message leaked into the training corpus: ${row.id}`);
    }
  }
  // Every kept train row's clean text must be in the corpus. The corpus can
  // be LARGER: a render whose error profile produced no edit is dropped from
  // the labelled rows but its clean text is still legitimate training text.
  const train = readJsonl(path.join(DS, 'sms-train.jsonl'));
  for (const row of train) {
    assert.ok(corpus.has(row.clean.trim()), `train row missing from the corpus: ${row.id}`);
  }
  assert.ok(corpus.size >= 100, `training corpus is suspiciously small: ${corpus.size}`);
});

test('the dataset on disk matches its manifest (deterministic build)', { skip: skipNoDataset }, () => {
  const manifest = JSON.parse(readFileSync(path.join(DS, 'manifest.json'), 'utf8'));
  for (const [name, expected] of Object.entries(manifest.files)) {
    const file = name.startsWith('benchmark/') ? path.join(ROOT, name) : path.join(DS, name);
    const actual = createHash('sha256').update(readFileSync(file, 'utf8')).digest('hex');
    assert.equal(actual, expected,
      `${name} does not match manifest.json — rebuild with: node tools/build_sms_dataset.mjs`);
  }
});

test('every label describes a real difference between text and clean', { skip: skipNoDataset }, () => {
  for (const split of SPLITS) {
    for (const row of readJsonl(path.join(DS, `sms-${split}.jsonl`))) {
      if (row.correction_pairs.length === 0) {
        if (row.profile === 'CLEAN') assert.equal(row.text, row.clean, row.id);
        continue;
      }
      assert.notEqual(row.text, row.clean, `${row.id} claims errors but text === clean`);
      for (const pair of row.correction_pairs) {
        assert.equal(row.text.slice(pair.start, pair.end), pair.error,
          `${row.id}: offset does not point at the error surface`);
        assert.notEqual(pair.error, pair.correction, `${row.id}: identity "correction"`);
      }
      for (const expected of row.expect) {
        assert.ok(expected.ruleId, `${row.id}: label without a ruleId`);
        assert.equal(row.text.slice(expected.positionStart, expected.positionEnd), expected.value);
      }
    }
  }
});

test('slot values (URL, money, phone, date, code) are never damaged', { skip: skipNoDataset }, () => {
  const PROTECTED = /(https?:\/\/\S+|\d[\d.,]*d\b|\b0\d{9}\b|\b\d{2}\/\d{2}\/\d{4}\b)/g;
  for (const split of SPLITS) {
    for (const row of readJsonl(path.join(DS, `sms-${split}.jsonl`))) {
      const inClean = row.clean.match(PROTECTED) ?? [];
      for (const value of inClean) {
        assert.ok(row.text.includes(value),
          `${row.id}: protected value "${value}" was altered by error injection`);
      }
    }
  }
});

test('telex injection produces the shape the engine recognises', () => {
  assert.equal(telexTypo('hàng'), 'hangf');
  assert.equal(telexTypo('quý'), 'quys');
  assert.equal(telexTypo('hưởng'), 'hươngr');
  assert.equal(telexTypo('sẽ'), 'sex');
  // quality marks (ư, ơ, ă, â, ê, ô) stay; only the TONE becomes a letter
  assert.equal(telexTypo('lực'), 'lưcj');
  assert.equal(telexTypo('cam'), null); // no tone, nothing to leave behind
});

// ---------- profile resolution ----------
test('ENGINE_PROFILE selects both artifacts together', () => {
  assert.equal(profileName({}), 'full');
  assert.equal(profileName({ ENGINE_PROFILE: 'lite' }), 'lite');
  assert.throws(() => profileName({ ENGINE_PROFILE: 'nope' }), /unknown ENGINE_PROFILE/);

  const fullLm = resolveArtifact('lm', {});
  assert.equal(path.basename(fullLm.path), PROFILES.full.lm);
  assert.equal(path.basename(resolveArtifact('lexicon', {}).path), PROFILES.full.lexicon);

  const liteLm = resolveArtifact('lm', { ENGINE_PROFILE: 'lite' });
  const liteLex = resolveArtifact('lexicon', { ENGINE_PROFILE: 'lite' });
  // both artifacts must resolve to the SAME profile — a mixed pair would mean
  // the ranker can propose words the LM cannot score
  assert.equal(liteLm.profile, liteLex.profile);
});

test('an explicit artifact override wins', () => {
  const r = resolveArtifact('lm', { ENGINE_PROFILE: 'lite', LM_ARTIFACT: '/tmp/custom.tsv' });
  assert.equal(r.path, '/tmp/custom.tsv');
  assert.equal(r.profile, 'custom');
});

test('a missing lite artifact falls back to full, loudly, once', () => {
  resetProfileWarnings();
  const warnings = [];
  const original = console.warn;
  console.warn = (msg) => warnings.push(msg);
  try {
    // 'lite' resolves against src/data; if the artifact exists this is a no-op
    // assertion, if it does not we must see exactly one warning.
    const first = resolveArtifact('lm', { ENGINE_PROFILE: 'lite' });
    resolveArtifact('lm', { ENGINE_PROFILE: 'lite' });
    if (first.profile === 'full') {
      assert.equal(warnings.length, 1, 'fallback must warn exactly once');
      assert.match(warnings[0], /build_sms_profile/);
    } else {
      assert.equal(warnings.length, 0);
    }
  } finally {
    console.warn = original;
  }
});
