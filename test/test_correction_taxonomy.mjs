// Recall-improvement plan Task 1: correction-relation taxonomy.
//
// The recall wall (execution evidence §0) is lane eligibility/prefiltering,
// so every labeled correction must be classifiable by its LINGUISTIC
// relation before any emission behavior changes:
//   IDENTITY | UNACCENTED_SAME_KEY | ACCENTED_SAME_KEY |
//   DIFFERENT_KEY_SINGLE_TOKEN | SPLIT | MERGE
// (+ DIFFERENT_KEY_MULTI_TOKEN residual, documented extension).
// The classifier normalizes NFC + case and strips only LEADING/TRAILING
// punctuation (the existing benchmark normalization policy); internal
// punctuation and Telex typing are NOT undone here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyCorrectionRelation } from '../src/correction-taxonomy.mjs';

const RELATIONS = [
  'IDENTITY', 'UNACCENTED_SAME_KEY', 'ACCENTED_SAME_KEY',
  'DIFFERENT_KEY_SINGLE_TOKEN', 'SPLIT', 'MERGE',
  'DIFFERENT_KEY_MULTI_TOKEN',
];

test('plan-mandated cases classify exactly as specified', () => {
  assert.equal(classifyCorrectionRelation('khach', 'khách'), 'UNACCENTED_SAME_KEY');
  assert.equal(classifyCorrectionRelation('quỳ', 'quý'), 'ACCENTED_SAME_KEY');
  assert.equal(classifyCorrectionRelation('đế', 'đến'), 'DIFFERENT_KEY_SINGLE_TOKEN');
  assert.equal(classifyCorrectionRelation('cảmơn', 'cảm ơn'), 'SPLIT');
  assert.equal(classifyCorrectionRelation('cảm ơn', 'cảmơn'), 'MERGE');
  assert.equal(classifyCorrectionRelation('abc', 'abc'), 'IDENTITY');
});

test('NFC/NFD composition and case folding do not change the relation', () => {
  // NFD-decomposed "khách" vs composed target — same key, input accented
  const nfdAccented = 'khách'.normalize('NFD');
  assert.equal(classifyCorrectionRelation(nfdAccented, 'khach'), 'ACCENTED_SAME_KEY');
  // uppercase input folds down
  assert.equal(classifyCorrectionRelation('KHACH', 'khách'), 'UNACCENTED_SAME_KEY');
  // tone-spelled differently but same stripped key stays same-key
  assert.equal(classifyCorrectionRelation('HOÁ', 'hóa'), 'ACCENTED_SAME_KEY');
});

test('leading/trailing punctuation follows the benchmark policy only', () => {
  assert.equal(classifyCorrectionRelation('(khach', 'khách.'), 'UNACCENTED_SAME_KEY');
  assert.equal(classifyCorrectionRelation('"quỳ"', 'quý!'), 'ACCENTED_SAME_KEY');
  // INTERNAL whitespace decides split/merge; internal punctuation stays
  assert.equal(classifyCorrectionRelation('cảmơn.', 'cảm ơn'), 'SPLIT');
  // multiple spaces still merge
  assert.equal(classifyCorrectionRelation('cảm  ơn', 'cảmơn'), 'MERGE');
});

test('same key routes by the INPUT accent state', () => {
  // unaccented input -> missing-diacritic lane ownership
  assert.equal(classifyCorrectionRelation('long', 'lòng'), 'UNACCENTED_SAME_KEY');
  // accented-but-wrong-tone input -> wrong-diacritic spelling lane ownership
  assert.equal(classifyCorrectionRelation('hưỏng', 'hướng'), 'ACCENTED_SAME_KEY');
  // đ participates in the key ("dat" family)
  assert.equal(classifyCorrectionRelation('ddat', 'đặt'.normalize('NFC')), 'DIFFERENT_KEY_SINGLE_TOKEN');
});

test('multi-token residuals keep DIFFERENT_KEY_SINGLE_TOKEN precise', () => {
  // both sides multi-token and different keys: documented extension value,
  // so the single-token bucket stays pure for lane routing (plan Task 5)
  assert.equal(
    classifyCorrectionRelation('khoan cách', 'khoản cắt'),
    'DIFFERENT_KEY_MULTI_TOKEN');
  // both multi-token but SAME joined key: relation follows the INPUT's
  // accent state, exactly like the single-token same-key cases
  assert.equal(
    classifyCorrectionRelation('hoá sau', 'hóa sầu'),
    'ACCENTED_SAME_KEY');
  assert.equal(
    classifyCorrectionRelation('hoa sau', 'hóa sầu'),
    'UNACCENTED_SAME_KEY');
  assert.ok(RELATIONS.includes('DIFFERENT_KEY_MULTI_TOKEN'));
});

test('degenerate inputs are deterministic and never throw', () => {
  assert.equal(classifyCorrectionRelation('', ''), 'IDENTITY');
  assert.equal(classifyCorrectionRelation('   ', '   '), 'IDENTITY');
  assert.equal(classifyCorrectionRelation('khach', ''), 'DIFFERENT_KEY_SINGLE_TOKEN');
  assert.equal(classifyCorrectionRelation('', 'khách'), 'DIFFERENT_KEY_SINGLE_TOKEN');
});
