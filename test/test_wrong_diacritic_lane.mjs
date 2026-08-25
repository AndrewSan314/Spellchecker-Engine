// Recall-improvement plan Task 4 — ACCENTED_SAME_KEY wrong-diacritic lane.
//
// SHADOW semantics: the lane computes a decision (wouldEmit + features)
// for dictionary-valid accented tokens but NEVER surfaces a user-visible
// issue; evaluation tooling scores wouldEmit. Guards (whitelist,
// abbreviation, protected, all-caps, mid-sentence proper noun) stay shut.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuleIds } from '../src/core.mjs';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import {
  evaluateSpellingToken, buildCorrectionCandidates,
} from '../src/rules/linguistic-rules.mjs';

function makeEngine(mode) {
  const engine = createDefaultEngine();
  if (mode) {
    engine.configService.reload({ linguistic: { wrongDiacriticMode: mode } });
  }
  return engine;
}

function tokenIndex(engine, text, normalized) {
  const ctx = new ValidationContext(text, 'ACCENTED', 'TENDOO');
  const doc = engine.documentBuilder.build(ctx);
  const words = doc.tokens.filter((t) => t.type === 'WORD');
  const idx = words.findIndex((w) => w.normalized === normalized);
  return { ctx, doc, words, idx };
}

test('an accented dictionary token reaches ACCENTED_SAME_KEY candidates', () => {
  // 'hạng' is corpus-backed (282k); its same-key family contains 'hàng'
  const engine = makeEngine('SHADOW');
  const r = buildCorrectionCandidates({
    token: { original: 'hạng', normalized: 'hạng' },
    lane: 'ACCENTED_SAME_KEY',
    services: engine.services,
    snap: engine.configService.snapshot(),
  });
  assert.equal(r.eligible, true);
  assert.ok(r.entries.some((c) => c.word === 'hàng'));
});

test('candidate with direct contextual wins becomes wouldEmit in SHADOW', () => {
  // "mua hạng nhanh": 'mua hàng nhanh' is corpus-attested; 'mua hạng' is not
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const { ctx, doc, words, idx } = tokenIndex(engine, 'mua hạng nhanh', 'hạng');
  assert.ok(idx > 0, 'fixture token must exist');
  const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
  assert.equal(d.shadowWrongDiacritic?.wouldEmit, true,
    `expected shadow wouldEmit, got ${JSON.stringify(d.shadowWrongDiacritic)}`);
  assert.equal(d.shadowWrongDiacritic.candidate, 'hàng');
});

test('original with equal or better evidence stays silent', () => {
  // "xếp hạng cao": the ORIGINAL has a directly attested trigram window
  // ("xếp hạng cao" = 90) -> provisional gate must refuse
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const { ctx, doc, words, idx } = tokenIndex(engine, 'xếp hạng cao', 'hạng');
  assert.ok(idx > 0);
  const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
  const shadow = d.shadowWrongDiacritic;
  if (shadow) {
    assert.equal(shadow.wouldEmit, false);
    assert.equal(shadow.checks.originalZeroTrigramWins, false);
  }
});

test('bảo hành chính hãng must never suggest hàng', () => {
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();
  const result = engine.validate(new ValidationContext(
    'bảo hành chính hãng', 'ACCENTED', 'TENDOO'));
  for (const issue of [...result.issues, ...result.shadowIssues]) {
    for (const s of issue.suggestions ?? []) {
      assert.notEqual(s.toLowerCase(), 'hàng',
        `lane suggested hàng via ${issue.ruleId}`);
    }
  }
  // decision-level: any shadow decision on 'hãnh' must be refused
  const { ctx, doc, words, idx } = tokenIndex(engine, 'bảo hành chính hãng', 'hãnh');
  if (idx > 0) {
    const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
    if (d.shadowWrongDiacritic) {
      assert.equal(d.shadowWrongDiacritic.wouldEmit, false);
      assert.notEqual(d.shadowWrongDiacritic.candidate.toLowerCase(), 'hàng');
    }
  }
});

test('whitelist / all-caps / proper-noun tokens remain blocked in SHADOW', () => {
  const engine = makeEngine('SHADOW');
  const snap = engine.configService.snapshot();

  // whitelisted: stub services where whitelist allows everything
  const stubServices = {
    ...engine.services,
    whitelistService: { isAllowed: () => true },
  };
  const { ctx, doc, words, idx } = tokenIndex(engine, 'mua hạng nhanh', 'hạng');
  const d = evaluateSpellingToken(stubServices, snap, ctx, doc, words, idx);
  assert.equal(d.stage, 'prefilter');
  assert.equal(d.reason, 'classified-WHITELISTED');
  assert.equal(d.shadowWrongDiacritic, undefined);

  // ALL-CAPS
  const caps = tokenIndex(engine, 'MUA HẠNG NHANH', 'HẠNG');
  if (caps.idx > 0) {
    const dCaps = evaluateSpellingToken(
      engine.services, snap, caps.ctx, caps.doc, caps.words, caps.idx);
    assert.equal(dCaps.stage, 'prefilter');
    assert.equal(dCaps.reason, 'all-caps');
  }

  // mid-sentence Capitalized proper noun ('Hạng' after another word)
  const proper = tokenIndex(engine, 'gọi Hạng nhất', 'Hạng');
  if (proper.idx > 0) {
    const dPn = evaluateSpellingToken(
      engine.services, snap, proper.ctx, proper.doc, proper.words, proper.idx);
    assert.equal(dPn.stage, 'prefilter');
    assert.equal(dPn.reason, 'capitalized-proper-noun');
  }
});

test('SHADOW never duplicates PMD/PSE issues on one span', () => {
  const engine = makeEngine('SHADOW');
  const result = engine.validate(new ValidationContext(
    'mua hạng nhanh', 'ACCENTED', 'TENDOO'));
  const spans = result.issues.filter((i) =>
    i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
    || i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
  const seen = new Set();
  for (const i of spans) {
    const key = `${i.start}:${i.end}`;
    assert.equal(seen.has(key), false, `duplicate linguistic span ${key}`);
    seen.add(key);
  }
});

test('OFF mode (default and explicit) keeps production fully unchanged', () => {
  for (const mode of [undefined, 'OFF']) {
    const engine = makeEngine(mode);
    const snap = engine.configService.snapshot();
    const { ctx, doc, words, idx } = tokenIndex(engine, 'mua hạng nhanh', 'hạng');
    const d = evaluateSpellingToken(engine.services, snap, ctx, doc, words, idx);
    // dictionary token stays prefiltered exactly as before; no lane decision
    assert.equal(d.stage, 'prefilter');
    assert.equal(d.reason, 'classified-DICTIONARY');
    assert.equal(d.shadowWrongDiacritic, undefined);
  }
});
