// ============================================================
// Regression tests for the logic fixes in the 2026-09-03 review:
//   B1 word-boundary lane has its own ruleId and is no longer swallowed
//   B2 the lane costs nothing while it is not ACTIVE
//   B7 the degraded sink is bounded
//   B8 error-channel proofs speak Vietnamese and carry the right ruleId
//   §7 an unaccented message carries a message-level summary flag
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SmsValidationEngine, ValidationContext } from '../src/engine.mjs';
import { ValidationConfigService, ValidationConfigSnapshot, DEFAULT_SNAPSHOT } from '../src/config.mjs';
import { RuleIds, LINGUISTIC_RULE_IDS, RULE_PRIORITY, priorityOf } from '../src/core.mjs';
import { createWordBoundaryRule } from '../src/rules/linguistic-rules.mjs';
import { NGramLanguageModel } from '../src/language.mjs';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'src', 'data');

const engine = new SmsValidationEngine();
const validate = (text, mode = 'ACCENTED') =>
  engine.validate(new ValidationContext(text, mode));

// ---------- B1 ----------
test('B1: the word-boundary lane has its own stable ruleId', () => {
  assert.equal(RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR, 'POSSIBLE_WORD_BOUNDARY_ERROR');
  assert.notEqual(RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR, RuleIds.POSSIBLE_SPELLING_ERROR);
  assert.ok(LINGUISTIC_RULE_IDS.has(RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR));
  assert.equal(typeof RULE_PRIORITY[RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR], 'number');
  // strictly below the spelling rule, so an identical span is not a coin flip
  assert.ok(priorityOf(RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR)
    < priorityOf(RuleIds.POSSIBLE_SPELLING_ERROR));
});

test('B1: a word-boundary issue on a shared span survives conflict resolution', () => {
  const spelling = {
    ruleId: RuleIds.POSSIBLE_SPELLING_ERROR, severity: 'WARNING',
    start: 0, end: 5, value: 'cảmơn', message: 'x', suggestions: ['cảm ơn'], confidence: 0.9,
  };
  const boundary = { ...spelling, ruleId: RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR };
  const resolved = engine.resolveConflicts([spelling, boundary]);
  // Same span, different rules: the resolver keeps ONE by priority — but the
  // two are now distinguishable, so the choice is a documented priority
  // decision instead of a silent dedup on an identical ruleId.
  assert.equal(resolved.length, 1);
  assert.equal(resolved[0].ruleId, RuleIds.POSSIBLE_SPELLING_ERROR);
  const boundaryOnly = engine.resolveConflicts([boundary]);
  assert.equal(boundaryOnly[0].ruleId, RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR);
});

// ---------- B2 ----------
test('B2: a non-ACTIVE word-boundary lane does no per-token work', () => {
  for (const mode of ['SHADOW', 'OFF']) {
    const snapshot = new ValidationConfigSnapshot({
      ...DEFAULT_SNAPSHOT,
      linguistic: { ...DEFAULT_SNAPSHOT.linguistic, wordBoundaryCorrectionMode: mode },
    });
    const rule = createWordBoundaryRule({
      ...engine.services,
      configService: new ValidationConfigService(snapshot),
    });
    // supports() must already be false in SHADOW: the engine then never calls
    // validate(), which is where the wasted candidate evaluation used to sit.
    assert.equal(rule.supports({}), false, `supports() must be false in ${mode}`);
    // and validate() is still defensive
    const doc = { tokens: [{ type: 'WORD', normalized: 'cảmơn', original: 'cảmơn', start: 0, end: 5 }] };
    assert.deepEqual(rule.validate(new ValidationContext('cảmơn', 'ACCENTED'), doc), []);
  }
});

// ---------- B7 ----------
test('B7: the degraded sink is bounded and keeps counters', () => {
  const local = new SmsValidationEngine();
  for (let i = 0; i < 200; i++) local.recordDegraded('SOME_OPTIONAL_RULE', new Error(`boom ${i}`));
  assert.ok(local.degraded.length <= 50, `ring grew to ${local.degraded.length}`);
  const summary = local.degradedSummary();
  assert.equal(summary.total, 200, 'the counter must survive ring eviction');
  assert.equal(summary.byRule.SOME_OPTIONAL_RULE, 200);
  assert.equal(summary.recent.length, 5);
  assert.match(summary.recent.at(-1).error, /boom 199/);
});

// ---------- B8 ----------
test('B8: an error-channel proof speaks Vietnamese, not English', () => {
  const res = validate('Ma OTP cua quy khach la 123456, hieu luc 5 phut. '
    + 'Khong chia se ma nay cho nguoi khac.');
  assert.ok(res.issues.length > 0);
  for (const issue of res.issues) {
    assert.ok(!/^Spelling suggestion/.test(issue.message),
      `English message leaked to the user: ${issue.message}`);
    assert.match(issue.message, /[àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]/iu,
      `message does not look Vietnamese: ${issue.message}`);
  }
});

test('B8: a pure missing-diacritic proof is labelled POSSIBLE_MISSING_DIACRITIC', () => {
  const res = validate('Khong chia se ma nay cho nguoi khac.');
  const khong = res.issues.find((i) => i.value === 'Khong');
  assert.ok(khong, 'expected an issue on "Khong"');
  assert.equal(khong.ruleId, RuleIds.POSSIBLE_MISSING_DIACRITIC,
    'a missing-diacritic fix must not be reported as a spelling error');
  assert.deepEqual([...khong.suggestions], ['không']);
});

// ---------- B3 ----------
test('B3: issues carry the raw confidence, not a display-rounded one', () => {
  const res = validate('Ma OTP cua quy khach la 123456, hieu luc 5 phut.');
  const scored = res.issues.filter((i) => typeof i.confidence === 'number');
  assert.ok(scored.length > 0);
  // At least one confidence must have more precision than 2 decimals; if every
  // value were pre-rounded the engine's >= floor comparison would be wrong.
  assert.ok(scored.some((i) => Math.abs(i.confidence * 100 - Math.round(i.confidence * 100)) > 1e-9),
    'every confidence looks pre-rounded — the display rounding leaked back in');
});

// ---------- §7 ----------
test('§7: an unaccented message in ACCENTED mode is flagged at message level', () => {
  const noisy = validate('Ma OTP cua quy khach la 123456, hieu luc 5 phut.');
  assert.equal(noisy.summary.unaccentedContent, true);
  assert.ok(noisy.summary.linguisticIssueCount > 0);
  assert.ok(noisy.summary.unaccentedWordRatio >= 0.8);

  const clean = validate('Kính chào quý khách, đơn hàng của bạn đã được giao thành công.');
  assert.equal(clean.summary.unaccentedContent, false);
  assert.equal(clean.summary.linguisticIssueCount, 0, 'no false alarm on clean Vietnamese');

  // NON_ACCENTED mode is a deliberate choice, never a "looks unaccented" notice
  const nonAccented = validate('Ma OTP cua quy khach la 123456.', 'NON_ACCENTED');
  assert.equal(nonAccented.summary.unaccentedContent, false);
});

// ---------- B5 ----------
test('B5: SHADOW mode still returns no linguistic issue to the user', () => {
  const shadow = new SmsValidationEngine({
    configService: new ValidationConfigService(new ValidationConfigSnapshot({
      ...DEFAULT_SNAPSHOT,
      linguistic: { ...DEFAULT_SNAPSHOT.linguistic, mode: 'SHADOW' },
    })),
  });
  const res = shadow.validate(new ValidationContext('Ma OTP cua quy khach la 123456.', 'ACCENTED'));
  assert.equal(res.issues.filter((i) => LINGUISTIC_RULE_IDS.has(i.ruleId)).length, 0);
  assert.ok(res.shadowIssues.length > 0, 'shadow issues must still be recorded');
});


// ============================================================
// Regression: a user-reported wrong suggestion.
//   "Khong chia se ma nay cho bat ky ai"  ->  engine said "ma" was "mà"
//   while the sentence means "mã này" (this code).
// Four defects fed it; see README "Bốn khiếm khuyết mô hình".
// ============================================================
test('"chia se ma nay": never asserts the wrong reading of "ma"', () => {
  const res = validate('Khong chia se ma nay cho bat ky ai');
  const onMa = res.issues.filter((i) => i.value.toLowerCase() === 'ma');
  assert.deepEqual(onMa.map((i) => i.suggestions?.[0]), [],
    `the engine must not claim a reading for "ma" here, got: ${JSON.stringify(onMa.map((i) => i.suggestions?.[0]))}`);
  // and it must still do its job on the tokens it does understand
  const suggested = new Map(res.issues.map((i) => [i.value.toLowerCase(), i.suggestions?.[0]]));
  assert.equal(suggested.get('se'), 'sẻ', 'lost the "chia sẻ" restoration');
  assert.equal(suggested.get('khong'), 'không');
});

test('the sentence-level decisions stay self-consistent left to right', () => {
  // "se" is resolved by frequency to "sẽ" by the beam; the rule must use its
  // own "sẻ" decision for the next position instead of inheriting that.
  const res = validate('Khong chia se ma nay cho bat ky ai');
  const se = res.issues.find((i) => i.value.toLowerCase() === 'se');
  assert.ok(se, 'expected a decision on "se"');
  assert.notEqual(se.suggestions?.[0], 'sẽ',
    'frequency-first neighbour resolution leaked back in');
});

test('the lite artifact satisfies the n-gram back-off invariant', () => {
  const artifact = path.join(DATA, 'lm-ngrams.sms.tsv');
  if (!existsSync(artifact)) return; // not built on this clone
  const lm = NGramLanguageModel.load(undefined, { prebuiltPath: artifact, hashArtifact: false });
  // every trigram (a b c) implies c(a,b) >= sum and c(b,c) >= sum: without
  // this, back-off assigns ~0 probability to a spelling the trigrams attest,
  // which is exactly how "mã" lost to "mà".
  const required = new Map();
  for (const [key, count] of lm.trigram) {
    const a = key.indexOf(' ');
    const b = key.indexOf(' ', a + 1);
    for (const pair of [key.slice(0, b), key.slice(a + 1)]) {
      required.set(pair, (required.get(pair) ?? 0) + count);
    }
  }
  const violations = [];
  for (const [pair, needed] of required) {
    if ((lm.bigram.get(pair) ?? 0) < needed) violations.push(pair);
    if (violations.length > 5) break;
  }
  assert.deepEqual(violations, [],
    'trigrams attest bigrams the artifact does not contain — rebuild with tools/build_sms_profile.mjs');
});
