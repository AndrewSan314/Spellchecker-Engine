// Task 2 (spelling-engine-optimization plan): LM artifact loading must fail
// fast and observably. Silent raw-corpus fallback is only allowed for a
// MISSING artifact; malformed/unreadable artifacts must throw
// LanguageModelArtifactError carrying path + line + reason.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import {
  NGramLanguageModel,
  LanguageModelArtifactError,
} from '../src/language.mjs';

const FIXTURE_DIR = path.resolve('.tmp/lm-loader-fixtures');
const RAW_CORPUS = path.resolve('src/data/corpus-train.txt');

function fixture(name, content) {
  mkdirSync(FIXTURE_DIR, { recursive: true });
  const file = path.join(FIXTURE_DIR, name);
  writeFileSync(file, content, 'utf8');
  return file;
}

const VALID_TSV = [
  '#tokens=8',
  'U\tkính\t2',
  'U\tchào\t2',
  'U\tquý\t2',
  'U\tkhách\t2',
  'B\tkính chào\t2',
  'B\tchào quý\t1',
  'B\tquý khách\t2',
  'T\tkính chào quý\t1',
  '',
].join('\n');

test('missing artifact falls back to documented raw-corpus loader', () => {
  const missing = path.join(FIXTURE_DIR, 'does-not-exist.tsv');
  const model = NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: missing });
  assert.equal(model.loadDiagnostics.backend, 'raw-corpus-fallback');
  // raw corpus really loaded (vocab sanity)
  assert.ok(model.totalTokens > 1000);
});

test('allowFallback:false turns missing artifact into a hard error', () => {
  const missing = path.join(FIXTURE_DIR, 'still-missing.tsv');
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: missing, allowFallback: false }),
    (err) => err instanceof LanguageModelArtifactError
      && err.artifactPath === missing,
  );
});

test('malformed artifact throws with path+line+reason and NEVER falls back', () => {
  const bad = fixture('malformed.tsv', VALID_TSV + 'X\tthis-kind-is-invalid\t1\n');
  let caught = null;
  try {
    NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: bad });
  } catch (err) {
    caught = err;
  }
  assert.ok(caught instanceof LanguageModelArtifactError, 'expected LanguageModelArtifactError');
  assert.equal(caught.artifactPath, bad);
  assert.match(String(caught.reason ?? ''), /kind/);
  assert.match(caught.message, new RegExp(bad.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
});

test('directory as artifact (EISDIR) is an artifact error, not a silent fallback', () => {
  mkdirSync(path.join(FIXTURE_DIR, 'dir-as-artifact'), { recursive: true });
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, {
      prebuiltPath: path.join(FIXTURE_DIR, 'dir-as-artifact'),
    }),
    (err) => err instanceof LanguageModelArtifactError && !/raw-corpus/.test(err.message),
  );
});

test('valid artifact loads exact U/B/T counts with diagnostics + hash', () => {
  const good = fixture('valid.tsv', VALID_TSV);
  const model = NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: good });
  assert.equal(model.loadDiagnostics.backend, 'tsv');
  assert.equal(model.unigram.size, 4);
  assert.equal(model.bigram.size, 3);
  assert.equal(model.trigram.size, 1);
  assert.equal(model.totalTokens, 8);
  assert.match(model.loadDiagnostics.sha256, /^[0-9a-f]{64}$/);
});

test('section ordering violated (trigram before unigram) is rejected', () => {
  const bad = fixture('bad-order.tsv', [
    '#tokens=3',
    'T\ta b c\t1',
    'U\tword\t1',
    'U\tother\t1',
    '',
  ].join('\n'));
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: bad }),
    (err) => err instanceof LanguageModelArtifactError && /order/i.test(String(err.reason)),
  );
});

test('duplicate key inside a section is rejected even though Map would hide it', () => {
  const bad = fixture('dup-key.tsv', [
    '#tokens=5',
    'U\tdupe\t1',
    'U\tdupe\t2',
    'U\tother\t2',
    'B\tdupe other\t1',
    'T\ta b c\t1',
    '',
  ].join('\n'));
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: bad }),
    (err) => err instanceof LanguageModelArtifactError && /duplicate/i.test(String(err.reason)),
  );
});

test('missing #tokens header is rejected', () => {
  const bad = fixture('no-header.tsv', 'U\tword\t1\n');
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: bad }),
    (err) => err instanceof LanguageModelArtifactError && /#tokens/.test(err.message),
  );
});

test('non-integer or non-positive counts are rejected with line info', () => {
  const bad = fixture('bad-count.tsv', [
    '#tokens=2',
    'U\tword\tx',
    '',
  ].join('\n'));
  assert.throws(
    () => NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: bad }),
    (err) => err instanceof LanguageModelArtifactError && err.line === 2 && /count/i.test(String(err.reason)),
  );
});

test('totalTokens comes from the #tokens header, not from record lines', () => {
  // 4 records but a weighted token count of 1000000: unigram probabilities
  // must be computed against 1,000,000 (the corpus total), never against the
  // number of n-gram rows.
  const good = fixture('header-tokens.tsv', [
    '#tokens=1000000',
    'U\tkính\t600000',
    'U\tchào\t300000',
    'U\tquý\t70000',
    'U\tkhách\t30000',
    'B\tkính chào\t250000',
    'B\tchào quý\t120000',
    'B\tquý khách\t180000',
    'T\tkính chào quý\t90000',
    '',
  ].join('\n'));
  const model = NGramLanguageModel.load(RAW_CORPUS, { prebuiltPath: good });
  assert.equal(model.totalTokens, 1000000);
  // sanity: P(kính) uses the header total (with the model's +0.5 smoothing),
  // never a record-count denominator
  const expectedP = (600000 + 0.5) / (1000000 + 0.5 * 4);
  assert.ok(Math.abs(model.pUni('kính') - expectedP) < 1e-12,
    `pUni(kính)=${model.pUni('kính')} expected ${expectedP}`);
});
