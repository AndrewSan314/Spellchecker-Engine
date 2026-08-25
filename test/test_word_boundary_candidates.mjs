// Recall-improvement plan Task 8 — bounded word-boundary candidates.
// One token can suggest two corpus-backed words ("cảmơn" -> "cảm ơn");
// two adjacent words can suggest one corpus-backed word ("như ng" ->
// "nhưng"); spans are exact; URL/code/placeholder ranges never participate;
// counts are bounded and deterministic; SHADOW never emits.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import {
  evaluateWordBoundaryCandidates,
} from '../src/word-boundary-candidates.mjs';

const ENGINE = createDefaultEngine();

function decide(text, mode = 'SHADOW') {
  ENGINE.configService.reload({
    linguistic: { wordBoundaryCorrectionMode: mode },
  });
  const snap = ENGINE.configService.snapshot();
  const ctx = new ValidationContext(text, 'ACCENTED', 'TENDOO');
  const doc = ENGINE.documentBuilder.build(ctx);
  const words = doc.tokens.filter((t) => t.type === 'WORD');
  return { snap, ctx, doc, words };
}

test('one token suggests two corpus-backed words (cảmơn -> cảm ơn)', () => {
  const { snap, ctx, doc, words } = decide('Cảmơn nhiều');
  assert.equal(words[0].normalized, 'cảmơn');
  const d = evaluateWordBoundaryCandidates({
    services: ENGINE.services, snap, languageModel:
      ENGINE.services.languageModel, ctx, doc, words, idx: 0,
  });
  assert.ok(d.splits.some((s) => s.suggestion === 'cảm ơn'),
    `expected split suggestion "cảm ơn", got ${JSON.stringify(d.splits)}`);
});

test('two adjacent words suggest one corpus-backed word (như ng -> nhưng)', () => {
  const { snap, ctx, doc, words } = decide('như ng người ta vẫn đi');
  const idx = words.findIndex((w) => w.normalized === 'như');
  assert.ok(idx >= 0);
  const d = evaluateWordBoundaryCandidates({
    services: ENGINE.services, snap, languageModel:
      ENGINE.services.languageModel, ctx, doc, words, idx,
  });
  assert.ok(d.merges.some((m) => m.suggestion === 'nhưng'),
    `expected merge "nhưng", got ${JSON.stringify(d.merges)}`);
});

test('issue span covers the exact original substring incl. the space', () => {
  const text = 'Tôi như ng biết';
  const { snap, ctx, doc, words } = decide(text);
  const idx = words.findIndex((w) => w.normalized === 'như');
  const d = evaluateWordBoundaryCandidates({
    services: ENGINE.services, snap, languageModel:
      ENGINE.services.languageModel, ctx, doc, words, idx,
  });
  const m = d.merges.find((x) => x.suggestion === 'nhưng');
  assert.ok(m, 'merge candidate missing');
  assert.equal(text.slice(m.start, m.end), 'như ng',
    'span must include the inter-token whitespace verbatim');
});

test('URL / code / placeholder ranges never split or merge', () => {
  // a URL-ish OTHER run plus an abbreviation token
  for (const text of ['vào https x.com ngay', 'gọi VT_TENDOO ho tro']) {
    const { snap, ctx, doc, words } = decide(text);
    for (let i = 0; i < words.length; i++) {
      const d = evaluateWordBoundaryCandidates({
        services: ENGINE.services, snap, languageModel:
          ENGINE.services.languageModel, ctx, doc, words, idx: i,
      });
      const touched = [...d.splits, ...d.merges]
        .filter((x) => x.wouldEmit)
        .some((x) => /https|\.com|VT_TENDOO/i.test(x.value));
      assert.equal(touched, false, `boundary lane touched ${text}`);
    }
  }
});

test('candidate counts are bounded and order deterministic', () => {
  const { snap, ctx, doc, words } = decide('không');
  const run = () => evaluateWordBoundaryCandidates({
    services: ENGINE.services, snap, languageModel:
      ENGINE.services.languageModel, ctx, doc, words, idx: 0,
  }).splits.map((s) => s.suggestion);
  const a = run();
  const b = run();
  assert.deepEqual(a, b);
  assert.ok(a.length <= 4, `splits bound exceeded: ${a.length}`);
});

test('SHADOW never emits user-visible issues (rule returns empty)', async () => {
  const { createWordBoundaryRule } = await import(
    '../src/rules/linguistic-rules.mjs');
  const rule = createWordBoundaryRule(ENGINE.services);
  const ctx = new ValidationContext('Cảmơn nhiều', 'ACCENTED', 'TENDOO');
  const doc = ENGINE.documentBuilder.build(ctx);
  const issues = rule.validate(ctx, doc);
  assert.equal(issues.length, 0,
    'SHADOW word-boundary rule must surface nothing');
});
