import assert from 'node:assert/strict';
import test from 'node:test';
import { benchmarkIssueMatches, matchBenchmarkRow, normalizeBenchmarkSurface } from '../benchmark/run-benchmark.mjs';

test('rule-scoped full labels count spelling extras but ignore formatting extras', () => {
  const row = {
    expect: [{ ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'sanh', suggestion: 'xanh' }],
    fullyLabeledRuleIds: ['POSSIBLE_SPELLING_ERROR'],
  };
  const issues = [
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'sanh', suggestions: ['xanh'], start: 0, end: 4 },
    { ruleId: 'MULTIPLE_WHITESPACE', value: '  ', suggestions: [], start: 4, end: 6 },
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'giam', suggestions: ['giảm'], start: 7, end: 11 },
  ];
  const result = matchBenchmarkRow(row, issues);
  assert.deepEqual(result.extraIssues, ['POSSIBLE_SPELLING_ERROR']);
  assert.equal(result.missed.length, 0);
});

test('legacy fullyLabeled true still scopes extras to every rule', () => {
  const result = matchBenchmarkRow({
    fullyLabeled: true,
    expect: [{ ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'sanh', suggestion: 'xanh' }],
  }, [
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'sanh', suggestions: ['xanh'] },
    { ruleId: 'MULTIPLE_WHITESPACE', value: '  ', suggestions: [] },
  ]);
  assert.deepEqual(result.extraIssues, ['MULTIPLE_WHITESPACE']);
});

test('suggestion matching is NFC/case-insensitive and surface-safe', () => {
  assert.equal(normalizeBenchmarkSurface('"Gắn'), 'gắn');
  assert.equal(normalizeBenchmarkSurface('gắn,'), 'gắn');
  assert.equal(benchmarkIssueMatches(
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'lam', suggestions: ['"Gắn'] },
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'LAM', suggestion: 'gắn,' },
  ), true);
  assert.equal(benchmarkIssueMatches(
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'lam', suggestions: ['gắn'] },
    { ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'lam', suggestion: 'gan' },
  ), false);
});
