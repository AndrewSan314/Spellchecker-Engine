// Task 1 (tiny-attention-spelling-reranker-FIXED plan) — evaluation
// provenance contract. Every evaluation JSONL starts with a header record
// (recordType:'header'); decision lines carry recordType:'decision'.
// validateEvaluationHeader must reject a mismatch in EACH protected axis:
// linguistic rules hash, config hash, classical reranker hash, LM hash,
// tokenizer/vocab/model hashes (when present), feature-contract version,
// and split.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  EVALUATION_HEADER_VERSION,
  createEvaluationHeader,
  validateEvaluationHeader,
  tagDecisionRecord,
  writeEvaluationJsonl,
  readEvaluationJsonl,
  sha256OfJson,
} from '../src/evaluation-provenance.mjs';

const H = {
  linguisticRulesDotMjs: 'a'.repeat(64),
  configDotMjs: 'b'.repeat(64),
  recallRerankerModelJson: 'c'.repeat(64),
  lmNgramsTsv: 'd'.repeat(64),
};

function makeHeader(overrides = {}) {
  return createEvaluationHeader({
    schema: 'shadow-decisions-v2',
    split: 'dev',
    hashes: H,
    config: {
      configHash: sha256OfJson({ a: 1 }),
      featureContract: 'recall-pairwise-v1',
    },
    createdBy: 'test',
    ...overrides,
  });
}

test('header carries recordType, version, split, hashes, config, creator', () => {
  const h = makeHeader();
  assert.equal(h.recordType, 'header');
  assert.equal(h.headerVersion, EVALUATION_HEADER_VERSION);
  assert.equal(h.schema, 'shadow-decisions-v2');
  assert.equal(h.split, 'dev');
  assert.deepEqual(h.hashes, H);
  assert.equal(h.config.featureContract, 'recall-pairwise-v1');
  assert.equal(h.createdBy, 'test');
  assert.ok(!Number.isNaN(Date.parse(h.createdAt)));
});

test('createEvaluationHeader rejects missing schema/split/createdBy', () => {
  assert.throws(() => createEvaluationHeader({ split: 'dev', createdBy: 'x' }));
  assert.throws(() => createEvaluationHeader({ schema: 's', createdBy: 'x' }));
  assert.throws(() => createEvaluationHeader({ schema: 's', split: 'dev' }));
});

test('validate accepts a fully matching header', () => {
  assert.equal(validateEvaluationHeader(makeHeader(), {
    schema: 'shadow-decisions-v2',
    split: 'dev',
    hashes: H,
    config: { configHash: sha256OfJson({ a: 1 }), featureContract: 'recall-pairwise-v1' },
  }), true);
});

test('rejects linguistic-rules hash mismatch', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), {
    hashes: { linguisticRulesDotMjs: '0'.repeat(64) },
  }), /hash mismatch for linguisticRulesDotMjs/);
});

test('rejects config hash mismatch', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), {
    config: { configHash: 'different' },
  }), /config hash mismatch/);
});

test('rejects classical reranker hash mismatch', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), {
    hashes: { recallRerankerModelJson: '1'.repeat(64) },
  }), /hash mismatch for recallRerankerModelJson/);
});

test('rejects LM hash mismatch', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), {
    hashes: { lmNgramsTsv: '2'.repeat(64) },
  }), /hash mismatch for lmNgramsTsv/);
});

test('tokenizer/vocab/model hashes validated only when present', () => {
  // absent from expected -> not checked, header without them passes
  assert.equal(validateEvaluationHeader(makeHeader(), { hashes: {} }), true);
  // present in expected -> enforced, including absence in the header
  for (const key of ['attentionTokenizer', 'attentionVocab', 'attentionModel']) {
    assert.throws(
      () => validateEvaluationHeader(makeHeader({ hashes: {} }), {
        hashes: { [key]: '3'.repeat(64) },
      }),
      new RegExp(`missing hash "${key}"`),
      `${key} absence must be rejected when expected`,
    );
    const withKey = createEvaluationHeader({
      schema: 's', split: 'dev', createdBy: 't',
      hashes: { [key]: '3'.repeat(64) },
    });
    assert.throws(() => validateEvaluationHeader(withKey, {
      hashes: { [key]: '4'.repeat(64) },
    }), new RegExp(`hash mismatch for ${key}:`));
  }
});

test('rejects feature-contract version mismatch', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), {
    config: { featureContract: 'recall-pairwise-v2' },
  }), /feature-contract version mismatch/);
});

test('rejects split and schema mismatches', () => {
  assert.throws(() => validateEvaluationHeader(makeHeader(), { split: 'train' }),
    /split mismatch/);
  assert.throws(() => validateEvaluationHeader(makeHeader(), { schema: 'other-v1' }),
    /schema mismatch/);
});

test('rejects non-header records and unknown header versions', () => {
  assert.throws(() => validateEvaluationHeader(null, {}),
    /not an evaluation JSONL header/);
  assert.throws(() => validateEvaluationHeader({ recordType: 'decision' }, {}),
    /not an evaluation JSONL header/);
  assert.throws(() => validateEvaluationHeader({
    ...makeHeader(), headerVersion: 99,
  }, {}), /header version/);
});

test('decision records are tagged recordType:"decision" without clobbering', () => {
  assert.deepEqual(tagDecisionRecord({ id: 1 }),
    { id: 1, recordType: 'decision' });
  assert.deepEqual(tagDecisionRecord({ id: 2, recordType: 'decision' }),
    { id: 2, recordType: 'decision' });
  assert.throws(() => tagDecisionRecord({ id: 3, recordType: 'header' }),
    /refusing to relabel/);
});

test('JSONL round trip: first line is the header, then decision lines', () => {
  const out = path.join(tmpdir(), `evalprov-test-${process.pid}.jsonl`);
  try {
    const header = makeHeader();
    const records = [{ id: 'r1', stage: 'correct' }, { id: 'r2', stage: 'candidate-miss' }];
    writeEvaluationJsonl(out, header, records);

    const rawLines = readFileSync(out, 'utf8').split(/\r?\n/).filter(Boolean);
    assert.equal(rawLines.length, 3);
    assert.equal(JSON.parse(rawLines[0]).recordType, 'header');
    for (const line of rawLines.slice(1)) {
      assert.equal(JSON.parse(line).recordType, 'decision');
    }

    const back = readEvaluationJsonl(out);
    assert.equal(back.header.schema, 'shadow-decisions-v2');
    assert.deepEqual(back.records.map((r) => r.id), ['r1', 'r2']);

    // stale/header-less dumps are refused at read time
    const stale = path.join(tmpdir(), `evalprov-stale-${process.pid}.jsonl`);
    writeFileSync(stale, `${JSON.stringify(records[0])}\n`, 'utf8');
    assert.throws(() => readEvaluationJsonl(stale),
      /not an evaluation JSONL header|stale/);
    rmSync(stale, { force: true });
  } finally {
    rmSync(out, { force: true });
  }
});
