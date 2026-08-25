// ============================================================
// Rule-level tests — plan §11 (each rule: shouldWarn / shouldNotWarn /
// offsets) + plan §32 Case A..G + §30.2 global offset invariant.
// ============================================================
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { RuleIds } from '../src/core.mjs';

const engine = createDefaultEngine();

function issues(text, mode = 'ACCENTED', brandname = 'VT_TENDOO') {
  return engine.validate(new ValidationContext(text, mode, brandname)).issues;
}

function ids(list) {
  return list.map((i) => i.ruleId);
}

// ---------- offset invariant over everything we test (plan §30.2) ----------
function assertOffsetInvariant(text) {
  const result = engine.validate(new ValidationContext(text, 'ACCENTED', 'VT_TENDOO'));
  for (const issue of result.issues) {
    assert.equal(
      text.substring(issue.start, issue.end),
      issue.value,
      `offset invariant violated: ${issue.ruleId} [${issue.start},${issue.end})`,
    );
  }
  // also in NON_ACCENTED mode
  const r2 = engine.validate(new ValidationContext(text, 'NON_ACCENTED', 'VT_TENDOO'));
  for (const issue of r2.issues) {
    assert.equal(text.substring(issue.start, issue.end), issue.value);
  }
}

test('global offset invariant on edge-case corpus (plan §30.4)', () => {
  const edges = [
    '', ' ', '  ', '\tXin chào', 'Xin chào ', 'Xin  chào',
    'Xin chào !', 'Xin chào.Quý khách',
    'https://example.com', 'example.com', 'abc@example.com',
    '10.000', '1,5', '12/10/2026', '23:59',
    'A.B.C', 'ABC-2026', 'DH123456',
    'Khuyến mãi!!!!!', '...', '....',
    'Xin chào\u200Bquý khách', 'Xin\u00A0chào',
    'Chuc mung quý khach', 'Kính chao quý khách',
    'Quy khach vui long nhap thong tin', 'Quy KH vui long kiem tra SDT',
    '😊 Chúc mừng 🎉', 'Truy cap https://a.vn/x?id=10 ngay.',
    'Goi 0912345678 luc 23:59 ngay 12/10/2026',
    'Ma don DH123456 cua ban',
  ];
  for (const text of edges) assertOffsetInvariant(text);
});

// ---------- LEADING / TRAILING ----------
test('LEADING_WHITESPACE warns and spans the run', () => {
  for (const text of [' Xin chào', '   Xin chào', '\tXin chào']) {
    const hits = issues(text).filter((i) => i.ruleId === RuleIds.LEADING_WHITESPACE);
    assert.equal(hits.length, 1, text);
    assert.equal(hits[0].start, 0);
  }
  assert.ok(!ids(issues('Xin chào')).includes(RuleIds.LEADING_WHITESPACE));
  assert.ok(!ids(issues('')).includes(RuleIds.LEADING_WHITESPACE));
});

test('TRAILING_WHITESPACE warns; all-space text yields leading only', () => {
  const hits = issues('Xin chào ').filter((i) => i.ruleId === RuleIds.TRAILING_WHITESPACE);
  assert.equal(hits.length, 1);
  const allSpace = issues('   ');
  assert.ok(ids(allSpace).includes(RuleIds.LEADING_WHITESPACE));
  assert.ok(!ids(allSpace).includes(RuleIds.TRAILING_WHITESPACE));
});

// ---------- MULTIPLE_WHITESPACE ----------
test('MULTIPLE_WHITESPACE: body runs warn, single spaces do not', () => {
  assert.ok(ids(issues('Xin chào  quý khách')).includes(RuleIds.MULTIPLE_WHITESPACE));
  assert.ok(ids(issues('Xin chào     quý khách')).includes(RuleIds.MULTIPLE_WHITESPACE));
  assert.ok(!ids(issues('Xin chào quý khách')).includes(RuleIds.MULTIPLE_WHITESPACE));
});

// ---------- WHITESPACE_BEFORE_PUNCTUATION ----------
test('WHITESPACE_BEFORE_PUNCTUATION value is the whitespace only', () => {
  const hits = issues('Xin chào !').filter((i) => i.ruleId === RuleIds.WHITESPACE_BEFORE_PUNCTUATION);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].value, ' ');
  assert.equal(hits[0].start, 8); assert.equal(hits[0].end, 9);
  assert.ok(!ids(issues('Xin chào!')).includes(RuleIds.WHITESPACE_BEFORE_PUNCTUATION));
});

// ---------- REPEATED_PUNCTUATION ----------
test('REPEATED_PUNCTUATION: !!! and ????? warn; ... allowed; .... warns', () => {
  assert.ok(ids(issues('Khuyến mãi!!!!!')).includes(RuleIds.REPEATED_PUNCTUATION));
  assert.ok(ids(issues('Đồng ý?????')).includes(RuleIds.REPEATED_PUNCTUATION));
  assert.ok(ids(issues('Hello....')).includes(RuleIds.REPEATED_PUNCTUATION));
  assert.ok(!ids(issues('Xin chào!')).includes(RuleIds.REPEATED_PUNCTUATION));
  assert.ok(!ids(issues('Đợi tôi...')).includes(RuleIds.REPEATED_PUNCTUATION));
  assert.ok(!ids(issues('https://a.vn/...')).includes(RuleIds.REPEATED_PUNCTUATION),
    'URL span must suppress repeated punct');
});

// ---------- MISSING_WHITESPACE_AFTER_PUNCTUATION ----------
test('MISSING_WHITESPACE_AFTER_PUNCTUATION positives', () => {
  for (const text of ['Xin chào.Quý khách', 'A,B', 'Xin chào!Quý khách']) {
    assert.ok(ids(issues(text)).includes(RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION), text);
  }
});

test('MISSING_WHITESPACE_AFTER_PUNCTUATION negatives (protected + exceptions)', () => {
  for (const text of [
    'https://example.com', 'example.com', '10.000', '1,5',
    '12/10/2026', '23:59', 'A.B.C', '...',
    'So tien 10.000 dong.', 'Hen 23:59 ngay 12/10/2026 nhe.',
    'abc@example.com gui qua email.',
    'Ma DH123456 va OTP123 da duoc tao.',
  ]) {
    assert.ok(
      !ids(issues(text, 'NON_ACCENTED')).includes(RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION),
      `unexpected MWAP: ${text}`,
    );
  }
});

// ---------- ZERO_WIDTH / NBSP ----------
test('ZERO_WIDTH_CHARACTER detects U+200B with codepoint message', () => {
  const hits = issues('Xin chào\u200Bquý khách')
    .filter((i) => i.ruleId === RuleIds.ZERO_WIDTH_CHARACTER);
  assert.equal(hits.length, 1);
  assert.match(hits[0].message, /U\+200B/);
});

test('NON_BREAKING_SPACE detects U+00A0 without duplicate MULTIPLE_WHITESPACE', () => {
  const list = issues('Xin\u00A0chào');
  assert.ok(ids(list).includes(RuleIds.NON_BREAKING_SPACE));
  assert.ok(!ids(list).includes(RuleIds.MULTIPLE_WHITESPACE));
});

// ---------- Case A: accent char in NON_ACCENTED mode (plan §32) ----------
test('Case A: NON_ACCENTED + "quý" -> ERROR grouped by token', () => {
  const result = engine.validate(
    new ValidationContext('Chuc mung sinh nhat quý khach', 'NON_ACCENTED', 'VT_TENDOO'));
  const hits = result.issues.filter((i) => i.ruleId === RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].value, 'quý');
  assert.equal(hits[0].severity, 'ERROR');
  assert.equal(result.hasErrors, true);
  assert.equal(result.valid, false);
});

test('Case A negative: clean unaccented text has no accent-mode error', () => {
  for (const text of ['Chuc mung quy khach', 'SMS OTP API', 'https://example.com',
    'Dang ky ngay hom nay de nhan uu dai']) {
    assert.ok(!ids(issues(text, 'NON_ACCENTED'))
      .includes(RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE), text);
  }
});

// ---------- Case B: missing diacritic with high confidence ----------
test('Case B: "Kính chao quý khách" suggests chào', () => {
  const hits = issues('Kính chao quý khách')
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
  const chao = hits.find((h) => h.value === 'chao');
  assert.ok(chao, 'expected PMD on "chao"');
  assert.deepEqual(chao.suggestions, ['chào']);
  assert.equal(chao.severity, 'WARNING');
  assert.ok(chao.confidence >= 0.85);
});

// ---------- Case G (mandatory): ambiguity suppresses ----------
test('Case G: isolated ambiguous "long" emits NO linguistic issue', () => {
  const list = issues('long');
  assert.ok(!ids(list).includes(RuleIds.POSSIBLE_MISSING_DIACRITIC));
  assert.ok(!ids(list).includes(RuleIds.POSSIBLE_SPELLING_ERROR));
});

test('ambiguous "ban" alone emits nothing (bạn/bán/bàn)', () => {
  const list = issues('ban');
  assert.equal(list.filter((i) => i.ruleId.startsWith('POSSIBLE')).length, 0);
});

test('context disambiguates: "Vui long kiem tra" -> lòng fires', () => {
  const hits = issues('Vui long kiem tra lai thong tin')
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC && i.value === 'long');
  assert.equal(hits.length, 1);
  assert.equal(hits[0].suggestions[0], 'lòng');
});

// ---------- Case C/D: abbreviation scoped to brandname ----------
test('Case C: VT_TENDOO sees KH and SDT abbreviations', () => {
  const hits = issues('Quy KH vui long kiem tra SDT')
    .filter((i) => i.ruleId === RuleIds.ABBREVIATION_DETECTED);
  assert.deepEqual(hits.map((h) => h.value).sort(), ['KH', 'SDT']);
});

test('Case D: ABC_BANK does not inherit VT_TENDOO abbreviation rule', () => {
  const hits = issues('Quy KH vui long kiem tra SDT', 'ACCENTED', 'ABC_BANK')
    .filter((i) => i.ruleId === RuleIds.ABBREVIATION_DETECTED);
  assert.equal(hits.length, 0);
});

test('abbreviation is whole-token only ("KHACH" != KH)', () => {
  assert.ok(!ids(issues('Quy KHACH vui long'))
    .includes(RuleIds.ABBREVIATION_DETECTED));
});

// ---------- whitelist ----------
test('whitelisted technical terms never trigger spelling warnings', () => {
  for (const text of ['SMS OTP API tu Viettel', 'Voucher app online Tendoo mobile SMS OTP']) {
    const list = issues(text);
    assert.ok(!ids(list).includes(RuleIds.POSSIBLE_SPELLING_ERROR), text);
  }
});

// ---------- no mutation of input ----------
test('engine never mutates the original content string', () => {
  const original = '  Kính chao quý khách !  ';
  const copy = original;
  engine.validate(new ValidationContext(original, 'ACCENTED', 'VT_TENDOO'));
  assert.equal(original, copy);
});

// ============================================================
// Regression guards — session "cải tiến engine": trigram collocation
// gate, neighbour-window qua protected range, same-key typo siblings,
// LM code-cleanup (GIAM50K), red-team precision rows (agent3).
// Mỗi test khóa một root cause đã fix.
// ============================================================

test('spelling rule emits after prevWord fix (hangf caught)', () => {
  const hits = issues('Don hangf cua ban da giao')
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR);
  assert.ok(hits.length >= 1);
});

test('neighbour window: PMD sees context across protected TIME range', () => {
  const hits = issues('He thong se bao tri tu 23:00 den 05:00')
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC
      && i.value === 'den');
  assert.equal(hits.length, 1);
  assert.equal(hits[0].suggestions[0], 'đến');
});

test('trigram collocation veto: legit "mua ngay tại" stays silent', () => {
  for (const text of [
    'Mua ngay tại cua hang de nhan uu dai',
    'Đặt đơn ngay tại https://m.tendoo.vn/deal?id=1 hoặc gọi 0987654321',
  ]) {
    const hits = issues(text)
      .filter((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC
        && i.value === 'ngay');
    assert.equal(hits.length, 0, text);
  }
});

test('same-key siblings: wrong-tone quỳ corrected to quý (spelling owns accented typos)', () => {
  const hits = issues('Kính chao quỳ khách')
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
      && i.value.toLowerCase() === 'quỳ');
  assert.equal(hits.length, 1);
  assert.ok(hits[0].suggestions.includes('quý'));
});

test('LM: code cleanup keeps "mã giam" phantom below real "mã giảm"', () => {
  const phantom = engine.languageModel.bigram.get('mã giam') ?? 0;
  const real = engine.languageModel.bigram.get('mã giảm') ?? 1;
  // GIAM50K-style promo codes must never out-evidence the real collocation
  assert.ok(phantom < real, `phantom=${phantom} real=${real}`);
});

test('red-team rows stay silent (agent3 adversarial corpus)', () => {
  const cleanRows = [
    ['Mua ngay iPhone 15 Pro Max giá tốt tại Tendoo Store, bảo hành chính hãng 12 tháng, freeship mọi đơn hàng!', 'ACCENTED'],
    ['Chúc mừng sinh nhật Trần Thị Hoa! Nhận ngay voucher 200.000đ dành cho bạn tại tendoo.vn/sinh-nhat. Happy birthday 🎂', 'ACCENTED'],
    ['Giảm đến 70% toàn bộ sản phẩm... Mua ngay tại cskh@tendoo.vn hoặc link facebook.com/tendoo.vn! Số lượng có hạn.', 'ACCENTED'],
    ['[Tendoo] Chỉ còn 3 giờ! Giảm đến 50% toàn bộ iPhone, Samsung; freeship đơn từ 99.000đ. Đặt đơn ngay tại https://m.tendoo.vn/deal?utm=ny&id=8888 hoặc gọi 0987 654 321. Hẹn gặp lại!', 'ACCENTED'],
    ['Don hang DH552010 cua ban da gui thanh cong. Thanh toan 1.250.000d khi nhan hang. Hen gap lai ban vao dip Tet 2026!', 'NON_ACCENTED'],
  ];
  for (const [text, mode] of cleanRows) {
    assert.equal(issues(text, mode).length, 0, text.slice(0, 50));
  }
});

test('sentence-initial ambiguous words still warn when context decides', () => {
  const expect = [
    ['Ban co the doi lich hen mien phi', 'Ban'],
    ['Luu y khong cung cap mat khau cho bat ky ai', 'Luu'],
    ['Dat ban toi nay giam 10%', 'Dat'],
  ];
  for (const [text, word] of expect) {
    const hits = issues(text)
      .filter((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC
        && i.value === word);
    assert.equal(hits.length, 1, text);
  }
});

test('plain-form gate escape hatch fires on giam at clause end', () => {
  const hits = issues('Dat ban toi nay giam 10%')
    .filter((i) => i.value === 'giam'
      && i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
  assert.equal(hits.length, 1);
});

// ============================================================
// P0 plan batch — benchmark FP accounting, NFD bypass, URL range,
// runtime config, SHADOW semantics, no client thresholds, fail-closed.
// ============================================================
import { ValidationEngineError } from '../src/core.mjs';

test('NFD bypass: decomposed accents caught like precomposed (plan §3)', () => {
  const hits = issues('Chuc mung quy\u0301 khach', 'NON_ACCENTED')
    .filter((i) => i.ruleId === RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE
      && i.value === 'quy\u0301');
  assert.equal(hits.length, 1);
  // offsets still reference the ORIGINAL text
  assert.equal(hits[0].start, 10);
  assert.equal(hits[0].end, 14);
});

test('URL range trims trailing sentence punctuation (plan §4)', () => {
  const withComma = issues('https://example.com,Quy khach');
  assert.ok(!ids(withComma).includes('PROTECTED'), 'comma must be outside URL');
  const mws = withComma.filter((i) =>
    i.ruleId === RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION && i.value === ',');
  assert.equal(mws.length, 1, 'missing-space after comma must warn');

  const bang = issues('https://a.vn!Nhan qua');
  const mb = bang.filter((i) =>
    i.ruleId === RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION && i.value === '!');
  assert.equal(mb.length, 1);

  // plain URL and path-ellipsis stay fully protected (punctuation-wise)
  const plainUrl = issues('Truy cap https://example.com ngay')
    .filter((i) => i.ruleId === RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION
      || i.ruleId === RuleIds.REPEATED_PUNCTUATION);
  assert.equal(plainUrl.length, 0);
  const ellipsis = issues('link https://a.vn/... ok')
    .filter((i) => i.ruleId === RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION
      || i.ruleId === RuleIds.REPEATED_PUNCTUATION);
  assert.equal(ellipsis.length, 0);
});

test('config reload changes behavior without restart (plan §5)', () => {
  const eng2 = createDefaultEngine();
  const ctx = new ValidationContext('Xin chào\u200Bquý khách', 'ACCENTED', null);
  const zw = (r) => r.issues.find((i) => i.ruleId === RuleIds.ZERO_WIDTH_CHARACTER);
  assert.equal(zw(eng2.validate(ctx)).severity, 'WARNING');
  eng2.configService.reload({ rules: { zeroWidthSeverity: 'ERROR' } });
  assert.equal(zw(eng2.validate(ctx)).severity, 'ERROR');

  // beamWidth is read per-decode now: width=1 must still decode paths
  eng2.configService.reload({ linguistic: { beamWidth: 1 } });
  const r = eng2.validate(new ValidationContext('Vui long kiem tra', 'ACCENTED', null));
  assert.ok(r.issues.some((i) => i.value === 'long'));
});

test('SHADOW mode: linguistic issues collected, never returned (plan §6)', () => {
  const eng3 = createDefaultEngine();
  eng3.configService.reload({ linguistic: { mode: 'SHADOW' } });
  const res = eng3.validate(new ValidationContext('Kính chao quý khách', 'ACCENTED', null));
  assert.equal(res.issues.length, 0);
  assert.ok(res.shadowIssues.some((i) =>
    i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC && i.value === 'chao'));
  // back to ACTIVE: same input warns again
  eng3.configService.reload({ linguistic: { mode: 'ACTIVE' } });
  assert.ok(eng3.validate(new ValidationContext('Kính chao quý khách', 'ACCENTED', null))
    .issues.some((i) => i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC));
});

test('client thresholds are ignored — server config is authoritative (plan §7)', () => {
  const eng4 = createDefaultEngine();
  const sneaky = new ValidationContext('Kính chao quý khách', 'ACCENTED', null,
    null, { minConfidence: 0, minMargin: 0 });
  // even with options object present, gates use snapshot thresholds only;
  // output must be identical to a clean context call.
  const a = JSON.stringify(eng4.validate(sneaky).issues.map((i) => [i.value, i.confidence]));
  const b = JSON.stringify(eng4.validate(
    new ValidationContext('Kính chao quý khách', 'ACCENTED', null)).issues
    .map((i) => [i.value, i.confidence]));
  assert.equal(a, b);
});

test('CRITICAL rule crash fails CLOSED; OPTIONAL degrades gracefully (plan §8)', () => {
  const eng5 = createDefaultEngine();
  // force an OPTIONAL rule to throw via a poisoned service
  const boom = () => { throw new Error('boom'); };
  eng5.services.languageModel.scoreCandidateOverSurfaces = boom;
  const soft = eng5.validate(new ValidationContext('Vui long kiem tra', 'ACCENTED', null));
  assert.ok(Array.isArray(soft.issues));               // no throw, degraded
  assert.ok(eng5.degraded.length >= 1);

  // force a CRITICAL rule to throw
  const eng6 = createDefaultEngine();
  eng6.rules.find((r) => r.id() === RuleIds.LEADING_WHITESPACE)
    .validate = () => { throw new Error('critical-boom'); };
  assert.throws(() => eng6.validate(
    new ValidationContext(' Xin chào', 'ACCENTED', null)),
  ValidationEngineError);
});
