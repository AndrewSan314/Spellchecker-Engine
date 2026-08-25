// Task 3 — schema, leakage-safe split and oracle@K rules for the attention
// ranking dataset. Pure-function tests; the generated .tmp artifact checks
// activate once the extractor has run.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ATTENTION_SPLIT_SALT,
  K_CHOICES,
  assertAllowedTrainingSources,
  assertNoGroupOverlap,
  assertRankingRowShape,
  calculateOracleMetrics,
  normalizedSentenceHash,
  oracleAtK,
  rowHitsAtK,
  selectShortlistK,
  splitBucketForGroup,
  splitForGroup,
} from '../src/attention-ranking-schema.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const TMP = path.join(ROOT, '.tmp');

function correctionRow(over = {}) {
  return {
    recordType: 'ranking-row',
    id: over.id ?? 'r1',
    groupId: over.groupId ?? 'g1',
    source: over.source ?? 'vsec-train',
    context: {},
    original: 'hang',
    candidates: over.candidates ?? [
      { surface: 'hàng' }, { surface: 'hanh' },
    ],
    classicalFeatures: [],
    labelIndex: over.labelIndex ?? 1,
    lane: over.lane ?? 'UNKNOWN_TYPO',
    ...over,
  };
}

test('split buckets are deterministic, in range, and roughly balanced', () => {
  const a = splitForGroup('some-group-id');
  assert.equal(a, splitForGroup('some-group-id'));
  // salt sensitivity
  assert.equal(splitBucketForGroup('some-group-id', ATTENTION_SPLIT_SALT),
    splitBucketForGroup('some-group-id'));
  for (let i = 0; i < 200; i++) {
    const b = splitBucketForGroup(`g${i}`);
    assert.ok(b >= 0 && b <= 99);
  }
  const counts = { train: 0, calibration: 0, 'internal-test': 0 };
  for (let i = 0; i < 3000; i++) counts[splitForGroup(`grp-${i}`)]++;
  assert.ok(counts.train / 3000 > 0.72 && counts.train / 3000 < 0.88,
    `train share ${counts.train / 3000}`);
  assert.ok(counts.calibration / 3000 > 0.05 && counts.calibration / 3000 < 0.15);
  assert.ok(counts['internal-test'] / 3000 > 0.05
    && counts['internal-test'] / 3000 < 0.15);
});

test('group overlap across splits is detected', () => {
  assert.equal(assertNoGroupOverlap({
    train: ['a', 'b'], calibration: ['c'], 'internal-test': ['d'],
  }), true);
  assert.throws(() => assertNoGroupOverlap({
    train: [{ groupId: 'x' }], calibration: ['x'],
  }), /group overlap/);
});

test('ranking row shape contract', () => {
  assert.equal(assertRankingRowShape(correctionRow()), true);
  // option 0 is KEEP by construction: candidates never contain it
  assert.throws(() => assertRankingRowShape(
    correctionRow({ candidates: [{ surface: 'KEEP_ORIGINAL' }] })),
  /KEEP_ORIGINAL/);
  // KEEP lanes must label 0
  assert.throws(() => assertRankingRowShape(correctionRow({
    lane: 'HARD_NEGATIVE_KEEP', labelIndex: 1,
  })), /label index 0/);
  // correction rows need a candidate
  assert.throws(() => assertRankingRowShape(correctionRow({ candidates: [] })),
    /at least one candidate/);
  // labelIndex bounds
  assert.throws(() => assertRankingRowShape(correctionRow({ labelIndex: 5 })),
    /out of range/);
  // closed enums
  assert.throws(() => assertRankingRowShape(correctionRow({ lane: 'MADE_UP' })),
    /unknown lane/);
  assert.throws(() => assertRankingRowShape(correctionRow({ source: 'vsec-dev' })),
    /unknown source/);
});

test('oracle@K arithmetic, three-metric schema and the freeze rule', () => {
  const rows = [
    correctionRow({ id: 'a', labelIndex: 1 }), // gold at candidate idx 0
    correctionRow({ id: 'b', labelIndex: 5 }), // gold at candidate idx 4: miss@4, hit@6/@8
    correctionRow({ id: 'c', labelIndex: 0 }), // KEEP rows carry no signal
    correctionRow({ id: 'd', labelIndex: 6 }), // gold at candidate idx 5: hit@6+
  ];
  assert.equal(rowHitsAtK(rows[0], 4), true);
  assert.equal(rowHitsAtK(rows[1], 4), false); // gold at candidate index 4
  assert.equal(rowHitsAtK(rows[1], 6), true);
  assert.equal(rowHitsAtK(rows[2], 4), null);

  // gold positions: a->0, b->4, d->5 (candidate indexes)
  const at4 = oracleAtK(rows, 4);   // hits: a -> 1/3
  const at6 = oracleAtK(rows, 6);   // + d -> 3/3
  const at8 = oracleAtK(rows, 8);   // 3/3
  assert.ok(Math.abs(at4 - 1 / 3) < 1e-9);
  assert.equal(at6, 1);
  assert.equal(at8, 1);

  // Test calculateOracleMetrics and invariants
  const validMetrics = calculateOracleMetrics({
    allAttempts: 1000,
    widePoolHits: 800,
    shortlistHitsAt4: 650,
    shortlistHitsAt6: 720,
    shortlistHitsAt8: 760,
  });
  assert.equal(validMetrics.widePoolOracle, 0.8);
  assert.equal(validMetrics.shortlistRetention[8], 0.95);
  assert.equal(validMetrics.absoluteOracle[8], 0.76);
  assert.ok(validMetrics.absoluteOracle[8] <= validMetrics.widePoolOracle);

  // Invariant violations must throw
  assert.throws(() => calculateOracleMetrics({
    allAttempts: 1000,
    widePoolHits: 700,
    shortlistHitsAt4: 600,
    shortlistHitsAt6: 650,
    shortlistHitsAt8: 750, // 750 > 700!
  }), /Invariant violation/);

  // freeze rule: smallest K within 0.005 of retention@8 with retention >= 90%
  const sel = selectShortlistK(validMetrics);
  assert.equal(sel.k, 8);
  assert.match(sel.reason, /K=8/);
  assert.deepEqual(K_CHOICES, [4, 6, 8]);
});

test('training-source path guard rejects held-out markers before any read', () => {
  const bad = [
    'dataset_artifacts/vsec/vsec-dev.jsonl',
    'dataset_artifacts/vsec/vsec-test.jsonl',
    'benchmark/corpus-viwiki-spelling.json',
    'dataset_artifacts/evaluation/final-recall-heldout.json',
    'data/clean-dev.txt',
  ];
  for (const p of bad) {
    assert.throws(() => assertAllowedTrainingSources([p]), /forbidden/i, p);
  }
  assert.throws(
    () => assertAllowedTrainingSources(['src/data/corpus-train.txt']),
    /not an allowed supervised input/,
    'corpus-train is NOT an extraction source here (it belongs to Task 4)',
  );
  assert.doesNotThrow(() => assertAllowedTrainingSources([
    'dataset_artifacts/vsec/vsec-train.jsonl',
    'dataset_artifacts/clean-source/clean-train.txt',
  ]));
});

test('normalized sentence hash is NFC/lowercase stable', () => {
  assert.equal(normalizedSentenceHash('Hòa Bình!'),
    normalizedSentenceHash('Hòa Bình'.normalize('NFD').concat('!')));
  assert.notEqual(normalizedSentenceHash('a'), normalizedSentenceHash('b'));
});

// ---------------------------------------------------------------------------
// Generated-artifact validation — active once tools/extract_attention_
// ranking_rows.mjs has produced the .tmp outputs.
// ---------------------------------------------------------------------------

function readJsonl(p) {
  if (!existsSync(p)) return null;
  return readFileSync(p, 'utf8').split(/\r?\n/).filter(Boolean)
    .map((l) => JSON.parse(l));
}
function loadManifest() {
  const p = path.join(TMP, 'attention-ranking-split-manifest.json');
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
}

test('generated splits satisfy the plan contract (post-extraction)', { skip: !loadManifest() }, () => {
  const manifest = loadManifest();
  assert.equal(manifest.schema, 'attention-ranking-split-manifest-v1');
  assert.ok(manifest.hashes?.vsecTrain && manifest.hashes?.cleanTrain);
  for (const k of ['widePoolOracle', 'oracleAt4', 'oracleAt6', 'oracleAt8']) {
    assert.ok(typeof manifest.oracle[k] === 'number');
  }
  assert.ok(K_CHOICES.includes(manifest.frozenK));

  const groupsPerSplit = {};
  let rowsTotal = 0;
  for (const split of ['train', 'calibration', 'internal-test']) {
    const records = readJsonl(path.join(TMP, `attention-ranking-${split}.jsonl`));
    assert.ok(records && records.length > 0, `${split} non-empty`);
    const header = records[0];
    assert.equal(header.recordType, 'header', `${split} header first`);
    const rows = records.slice(1).filter((r) => r.recordType === 'ranking-row');
    assert.ok(rows.length > 0, `${split} has ranking rows`);
    rowsTotal += rows.length;
    groupsPerSplit[split] = [...new Set(rows.map((r) => r.groupId))];
    for (const row of rows) assertRankingRowShape(row);
  }
  assertNoGroupOverlap(groupsPerSplit);

  // deny list covers calibration + internal-test group hashes
  const denied = new Set([
    ...manifest.denyList.calibration,
    ...manifest.denyList.internalTest,
  ]);
  for (const g of groupsPerSplit.calibration) {
    assert.ok(denied.has(normalizedSentenceHash(g)) || denied.has(g),
      'calibration groups denied for pretraining');
  }

  // shortlist config frozen before Task 4
  const cfg = JSON.parse(readFileSync(
    path.join(TMP, 'attention-shortlist-config.json'), 'utf8'));
  assert.equal(cfg.schema, 'attention-shortlist-config-v1');
  assert.equal(cfg.k, manifest.frozenK);
  assert.ok(cfg.selectionReason);
  assert.ok(cfg.hashes?.generatorConfig);

  assert.ok(rowsTotal > 100, `enough rows mined (${rowsTotal})`);
});
