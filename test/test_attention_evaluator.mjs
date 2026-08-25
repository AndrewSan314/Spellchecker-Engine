import { test } from 'node:test';
import assert from 'node:assert/strict';
import { matchIssuesOneToOne, evaluateOnMessages } from '../tools/evaluate_attention_messages.mjs';

const issue = (over = {}) => ({
  ruleId: 'POSSIBLE_SPELLING_ERROR', value: 'ba', start: 2, end: 4,
  suggestions: ['bá'], ...over,
});
const label = (over = {}) => ({
  value: 'ba', start: 2, end: 4, suggestions: ['bá'], ...over,
});

test('evaluator matches exact span, value, and suggestion', () => {
  assert.deepEqual(matchIssuesOneToOne([issue()], [label()]).tp, 1);
  assert.deepEqual(matchIssuesOneToOne(
    [issue({ start: 3, end: 5 })], [label()]),
  { tp: 0, fp: 1, fn: 1, usedLabels: new Set(), matchedIssues: new Set() });
});

test('wrong suggestion and duplicate issue are not true positives', () => {
  const wrong = matchIssuesOneToOne([issue({ suggestions: ['bà'] })], [label()]);
  assert.equal(wrong.tp, 0);
  assert.equal(wrong.fp, 1);
  assert.equal(wrong.fn, 1);
  const duplicate = matchIssuesOneToOne([issue(), issue()], [label()]);
  assert.equal(duplicate.tp, 1);
  assert.equal(duplicate.fp, 1);
  assert.equal(duplicate.fn, 0);
});

test('extra issue elsewhere is counted as FP in full message metrics', () => {
  const engine = { validate: () => ({ issues: [issue(), issue({ value: 'co', start: 8, end: 10 })] }) };
  const result = evaluateOnMessages(engine, [{
    recordType: 'message-row', text: 'ba ... co', labels: [label()],
  }]);
  assert.equal(result.tp, 1);
  assert.equal(result.fp, 1);
  assert.equal(result.fn, 0);
});
