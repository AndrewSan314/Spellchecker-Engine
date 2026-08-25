// Task 8 Step 4 — evaluation split guards: tuning may only ever touch
// dev/train; held-out splits open exclusively behind --final.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assertSplitAllowed } from '../tools/run_spelling_eval.mjs';

test('dev and train are always allowed tuning splits', () => {
  assert.deepEqual(assertSplitAllowed('dev'), { allowed: true, final: false });
  assert.deepEqual(assertSplitAllowed('train'), { allowed: true, final: false });
  // --final on a tuning split is redundant but not an error
  assert.deepEqual(assertSplitAllowed('dev', { final: true }),
    { allowed: true, final: false });
});

test('held-out splits refuse without --final', () => {
  for (const split of ['test', 'external-test']) {
    assert.throws(
      () => assertSplitAllowed(split),
      (err) => /REFUSED/.test(err.message) && new RegExp(split).test(err.message),
      `${split} must refuse without --final`,
    );
  }
});

test('held-out splits open only behind explicit --final', () => {
  for (const split of ['test', 'external-test']) {
    assert.deepEqual(assertSplitAllowed(split, { final: true }),
      { allowed: true, final: true });
  }
});

test('unknown splits are rejected outright', () => {
  assert.throws(() => assertSplitAllowed('benchmark'), /unknown split/i);
});
