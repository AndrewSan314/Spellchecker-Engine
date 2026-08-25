// Debug tool: trace PMD/PSE internals for one sentence.
// Usage: node tools/debug-rule.mjs "<text>" [ACCENTED|NON_ACCENTED]
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { accentKey } from '../src/normalizer.mjs';
import { softmax } from '../src/language.mjs';
import { classifyToken } from '../src/rules/linguistic-rules.mjs';

const text = process.argv[2] ?? 'Vuj long kiem tra';
const mode = process.argv[3] ?? 'ACCENTED';
const engine = createDefaultEngine();
const services = engine.services;
const snap = engine.configService.snapshot();
const ctx = new ValidationContext(text, mode, 'VT_TENDOO');
const doc = engine.documentBuilder.build(ctx);

console.log('=== ' + JSON.stringify(text) + ' mode=' + mode);

for (const t of doc.tokens.filter((x) => x.type === 'WORD')) {
  const cls = classifyToken(t, ctx, doc, services);
  const stripped = accentKey(t.normalized);
  const accentCands = services.accentIndex.candidates(stripped);
  console.log(`  '${t.original}' norm='${t.normalized}' stripped='${stripped}' cls=${cls}`
    + ` accentCands=[${accentCands.map((c) => c.word).join(',')}]`);
  if (cls !== 'UNKNOWN') continue;
  const maxDist = t.normalized.length <= 7 ? 1 : 2;
  const typoCands = engine.typoCandidateProvider.candidates(t.normalized, maxDist);
  console.log(`    typoCands(maxDist=${maxDist}): [${typoCands.map((c) => `${c.word}/d${c.dist}/f${c.freq}`).join(', ')}]`);
}

// beam paths
const { buildPositions } = await import('../src/rules/linguistic-rules.mjs').then(() => ({}));
// rebuild positions like PMD does
const eligible = [];
for (let ti = 0; ti < doc.tokens.length; ti++) {
  const t = doc.tokens[ti];
  if (t.type !== 'WORD') continue;
  const cls = classifyToken(t, ctx, doc, services);
  if (cls !== 'UNKNOWN' && cls !== 'UNACCENTED_VALID') continue;
  const nearestL = (() => { for (let j = ti - 1; j >= 0; j--) if (doc.tokens[j].type !== 'WHITESPACE') return doc.tokens[j]; return null; })();
  const nearestR = (() => { for (let j = ti + 1; j < doc.tokens.length; j++) if (doc.tokens[j].type !== 'WHITESPACE') return doc.tokens[j]; return null; })();
  if (nearestL?.type !== 'WORD' && nearestR?.type !== 'WORD') continue;
  const stripped = accentKey(t.normalized);
  if (stripped !== t.normalized) continue;
  if (t.normalized.length < snap.get('linguistic.minTokenLength')) continue;
  const cands = services.accentIndex.candidates(stripped).filter((c) => c.word.toLowerCase() !== t.normalized);
  if (cands.length === 0) continue;
  eligible.push({ token: t, cands });
}
const ambByIndex = new Map(eligible.map((a) => [a.token.start, a]));
const positions = [];
for (const t of doc.tokens) {
  if (t.type !== 'WORD') continue;
  const amb = ambByIndex.get(t.start);
  if (amb) {
    positions.push({ token: t, candidates: [
      ...amb.cands.map((c) => ({ word: c.word.toLowerCase(), isOriginal: false, freq: c.freq })),
      { word: t.normalized, isOriginal: true, freq: 1 },
    ]});
  } else {
    let w = t.normalized;
    if (!engine.languageModel.knows(w)) {
      const s = services.accentIndex.candidates(accentKey(w));
      if (s.length > 0) w = s[0].word.toLowerCase();
    }
    positions.push({ token: t, candidates: [{ word: w, isOriginal: true, freq: 1 }] });
  }
}
const paths = engine.beamDecoder.decode(positions);
console.log('top1: ' + paths[0].words.join(' '));
const prior = snap.get('linguistic.originalPriorBonusMissingAccent');
positions.forEach((pos, idx) => {
  if (pos.candidates.length < 2) return;
  const prev = idx > 0 ? paths[0].words[idx - 1] : null;
  const next = idx + 1 < paths[0].words.length ? paths[0].words[idx + 1] : null;
  const scores = pos.candidates.map((c) => engine.languageModel.scoreCandidate(c.word, prev, next) + (c.isOriginal ? prior : 0));
  const probs = softmax(scores);
  console.log(`  [${idx}] '${pos.token.original}' chosen=${paths[0].words[idx]} probs=[${pos.candidates.map((c, k) => c.word + ':' + probs[k].toFixed(3)).join(', ')}]`);
});
