// ============================================================
// Core + preprocessing tests — plan §30
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  stripVietnameseDiacritics, hasVietnameseAccent, isVietnameseAccentedChar,
} from '../src/normalizer.mjs';
import { tokenize } from '../src/tokenizer.mjs';
import { detectProtectedRanges } from '../src/protected-ranges.mjs';

// ---------- plan §7.1 strip tests ----------
test('stripVietnameseDiacritics: minimal required cases', () => {
  assert.equal(stripVietnameseDiacritics('quý'), 'quy');
  assert.equal(stripVietnameseDiacritics('khách'), 'khach');
  assert.equal(stripVietnameseDiacritics('đăng'), 'dang');
  assert.equal(stripVietnameseDiacritics('Đà'), 'Da');
  assert.equal(stripVietnameseDiacritics('Viettel'), 'Viettel');
  assert.equal(stripVietnameseDiacritics('SMS'), 'SMS');
});

test('strip keeps đ mapping explicit (đ -> d)', () => {
  assert.equal(stripVietnameseDiacritics('đ'), 'd');
  assert.equal(stripVietnameseDiacritics('Đ'), 'D');
  assert.equal(stripVietnameseDiacritics('Đặt hàng'), 'Dat hang');
});

test('accent detection', () => {
  assert.equal(hasVietnameseAccent('Xin chào'), true);
  assert.equal(hasVietnameseAccent('Đang ky'), true); // Đ counts
  assert.equal(hasVietnameseAccent('Chuc mung'), false);
  assert.equal(hasVietnameseAccent('OTP SMS API'), false);
  assert.equal(isVietnameseAccentedChar('ý'), true);
  assert.equal(isVietnameseAccentedChar('z'), false);
});

// ---------- tokenizer offsets (UTF-16 invariant) ----------
test('tokenizer preserves UTF-16 offsets on original text', () => {
  const text = 'Xin chào quý khách 😊 10.000đ';
  for (const t of tokenize(text)) {
    assert.equal(text.substring(t.start, t.end), t.original,
      `token offset mismatch for ${t.original}`);
  }
});

test('tokenizer types', () => {
  const tokens = tokenize('Xin chào 10!');
  const types = tokens.map((t) => `${t.original}:${t.type}`).join(',');
  assert.match(types, /Xin:WORD/);
  assert.match(types, /chào:WORD/);
  assert.match(types, /10:NUMBER/);
  assert.match(types, /!:PUNCTUATION/);
});

// ---------- protected ranges — plan §8.3..§8.9 ----------
const protectedTests = [
  ['https://example.com', 'URL'],
  ['http://a.vn/x?id=10', 'URL'],
  ['www.viettel.vn', 'URL'],
  ['example.com', 'URL'],
  ['abc.com/path?a=1', 'URL'],
  ['abc@example.com', 'EMAIL'],
  ['a.b+c@company.vn', 'EMAIL'],
  ['0912345678', 'PHONE'],
  ['+84912345678', 'PHONE'],
  ['84 912 345 678', 'PHONE'],
  ['12/10/2026', 'DATE'],
  ['23:59', 'TIME'],
  ['08:30', 'TIME'],
  ['10.000', 'NUMBER'],
  ['1,000,000', 'NUMBER'],
  ['10,5', 'NUMBER'],
  ['DH123456', 'PRODUCT_CODE'],
  ['VT001', 'PRODUCT_CODE'],
  ['ABC-2026', 'PRODUCT_CODE'],
  ['OTP123', 'PRODUCT_CODE'],
  ['{{customer_name}}', 'PLACEHOLDER'],
  ['${otp}', 'PLACEHOLDER'],
  ['{amount}', 'PLACEHOLDER'],
];
for (const [text, type] of protectedTests) {
  test(`protected range: "${text}" -> ${type}`, () => {
    const ranges = detectProtectedRanges(text);
    assert.ok(ranges.length >= 1, `no range detected in ${text}`);
    const r = ranges.find((x) => x.contains(0, text.length))
      ?? ranges[0];
    assert.equal(r.type, type, `types were: ${ranges.map((x) => x.type).join(',')}`);
  });
}

test('"Xin chào.Quý khách" must NOT be protected as URL (plan §8.3 negative)', () => {
  const ranges = detectProtectedRanges('Xin chào.Quý khách');
  assert.equal(ranges.length, 0);
});

test('"12-10-2026" is a DATE, not a phone', () => {
  const ranges = detectProtectedRanges('12-10-2026');
  assert.ok(ranges.some((r) => r.type === 'DATE'));
  assert.ok(!ranges.some((r) => r.type === 'PHONE'));
});

test('"A.B.C" is not swallowed as a domain/code', () => {
  const ranges = detectProtectedRanges('A.B.C la vi du');
  // no URL-style protection over the initials
  assert.ok(!ranges.some((r) => r.type === 'URL'));
});
