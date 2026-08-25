// Recall-improvement plan Task 2 — semantic linguistic correction matching.
//
// VSEC labels every correction POSSIBLE_SPELLING_ERROR, but the product
// intentionally emits unaccented same-key fixes as
// POSSIBLE_MISSING_DIACRITIC. Strict rule-ID matching stays the contract
// metric; `linguisticCorrectionMatches` is the PRIMARY product-recall view:
// a correction is credited wherever a LINGUISTIC rule emitted it, as long
// as value AND suggestion match. Deterministic rules never count.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuleIds } from '../src/core.mjs';
import {
  linguisticCorrectionMatches,
} from '../src/correction-taxonomy.mjs';
import { benchmarkIssueMatches } from '../benchmark/run-benchmark.mjs';

test('khach -> khách is credited when PMD emits khách', () => {
  const issue = {
    ruleId: RuleIds.POSSIBLE_MISSING_DIACRITIC,
    value: 'khach',
    suggestions: ['khách'],
  };
  assert.equal(linguisticCorrectionMatches(issue,
    { value: 'khach', suggestions: ['khách'] }), true);
});

test('quỳ -> quý is credited when spelling emits quý', () => {
  const issue = {
    ruleId: RuleIds.POSSIBLE_SPELLING_ERROR,
    value: 'quỳ',
    suggestions: ['quý'],
  };
  assert.equal(linguisticCorrectionMatches(issue,
    { value: 'quỳ', suggestions: ['quý'] }), true);
});

test('wrong value or wrong suggestion is NOT credited', () => {
  const pmd = { ruleId: RuleIds.POSSIBLE_MISSING_DIACRITIC, value: 'khach', suggestions: ['khách'] };
  assert.equal(linguisticCorrectionMatches(pmd,
    { value: 'hoa', suggestions: ['khách'] }), false);
  assert.equal(linguisticCorrectionMatches(pmd,
    { value: 'khach', suggestions: ['khash'] }), false);
  // suggestion list without any matching entry
  assert.equal(linguisticCorrectionMatches(pmd,
    { value: 'khach', suggestions: [] }), false);
});

test('deterministic issues never count as linguistic corrections', () => {
  for (const ruleId of [
    RuleIds.MULTIPLE_WHITESPACE,
    RuleIds.ABBREVIATION_DETECTED,
    RuleIds.REPEATED_PUNCTUATION,
    RuleIds.LEADING_WHITESPACE,
  ]) {
    assert.equal(linguisticCorrectionMatches(
      { ruleId, value: 'khach', suggestions: ['khách'] },
      { value: 'khach', suggestions: ['khách'] }), false);
  }
});

test('strict rule-ID matching semantics are unchanged (documented here)', () => {
  // benchmarkIssueMatches requires ruleId equality; a PMD emission must NOT
  // satisfy a strict POSSIBLE_SPELLING_ERROR expectation, while the semantic
  // matcher DOES credit it. This pins the two views side by side.
  const pmdIssue = { ruleId: RuleIds.POSSIBLE_MISSING_DIACRITIC, value: 'khach', suggestions: ['khách'] };
  const strictExpect = {
    ruleId: RuleIds.POSSIBLE_SPELLING_ERROR,
    value: 'khach',
    suggestion: 'khách',
  };
  assert.equal(benchmarkIssueMatches(pmdIssue, strictExpect), false);
  assert.equal(linguisticCorrectionMatches(pmdIssue,
    { value: 'khach', suggestions: ['khách'] }), true);
});

test('matching normalizes NFC and case and tolerates both expect shapes', () => {
  const issue = { ruleId: RuleIds.POSSIBLE_MISSING_DIACRITIC, value: 'KHACH', suggestions: ['Khách'.normalize('NFD')] };
  assert.equal(linguisticCorrectionMatches(issue,
    { value: 'khach', suggestions: ['khách'] }), true);
  // benchmark-style singular `suggestion`
  assert.equal(linguisticCorrectionMatches(
    { ruleId: RuleIds.POSSIBLE_SPELLING_ERROR, value: 'quỳ', suggestions: ['quý'] },
    { value: 'Quỳ', suggestion: 'QUÝ' }), true);
});
