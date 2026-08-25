// Task 1 (spelling-engine-optimization plan): Viwiki converter must retain
// normalized suggestions end-to-end so the external benchmark measures
// CORRECTION quality, not detection only.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  convertDocumentFixture,
  IDENTITY_SUGGESTION_POLICY,
} from '../tools/convert_viwiki_spelling.mjs';

test('converter preserves all normalized suggestions', () => {
  const row = convertDocumentFixture({
    text: 'Kính chao quý khách.',
    mistakes: [{ start_offset: 5, text: 'chao', suggest: ['chào'] }],
  });
  assert.deepEqual(row.expect[0].suggestions, ['chào']);
  assert.equal(row.expect[0].suggestion, 'chào');
});

test('multiple mistakes in one sentence become sorted expect entries', () => {
  const row = convertDocumentFixture({
    text: 'Kính chao quý khach hang dau.',
    mistakes: [
      { start_offset: 5, text: 'chao', suggest: ['chào'] },
      { start_offset: 13, text: 'khach', suggest: ['khách'] },
    ],
  });
  assert.equal(row.expect.length, 2);
  assert.deepEqual(row.expect.map((e) => e.value), ['chao', 'khach']);
  assert.deepEqual(row.expect[0].suggestions, ['chào']);
  assert.deepEqual(row.expect[1].suggestions, ['khách']);
});

test('multiple suggestions for one mistake are all retained in order', () => {
  const row = convertDocumentFixture({
    text: 'Kính chao quý khách.',
    mistakes: [{ start_offset: 5, text: 'chao', suggest: ['chào', 'chạo', 'cháo'] }],
  });
  assert.deepEqual(row.expect[0].suggestions, ['chào', 'chạo', 'cháo']);
  assert.equal(row.expect[0].suggestion, 'chào');
});

test(`identity suggestions are dropped per explicit policy (${IDENTITY_SUGGESTION_POLICY})`, () => {
  const row = convertDocumentFixture({
    text: 'Dòng sông ràng qua rặng dương.',
    mistakes: [{ start_offset: 10, text: 'ràng', suggest: ['ràng'] }],
  });
  // detection label survives; correction signal is empty because the only
  // suggestion is identical to the mistake itself
  assert.equal(row.expect.length, 1);
  assert.equal(row.expect[0].value, 'ràng');
  assert.deepEqual(row.expect[0].suggestions, []);
  assert.equal(row.expect[0].suggestion, undefined);
});

test('offset preservation after sentence trimming', () => {
  const row = convertDocumentFixture({
    text: '     Kính chao quý khách.',
    mistakes: [{ start_offset: 10, text: 'chao', suggest: ['chào'] }],
  });
  // sentence "Kính chao quý khách." starts after 5 leading spaces;
  // 'chao' sits at absolute 10 => relative 5 inside the trimmed sentence
  assert.equal(row.text.startsWith('Kính chao'), true);
  assert.deepEqual(row.expect[0].span, [5, 9]);
  assert.equal(row.text.slice(5, 9), 'chao');
});

test('mixed identity + real suggestions keep only the real ones', () => {
  const row = convertDocumentFixture({
    text: 'Ban co the xem them tai day.',
    mistakes: [{ start_offset: 0, text: 'Ban', suggest: ['Bán', 'Ban'] }],
  });
  assert.deepEqual(row.expect[0].suggestions, ['Bán']);
});
