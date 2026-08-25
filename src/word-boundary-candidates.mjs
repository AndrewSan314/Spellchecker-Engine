// ============================================================
// Word-boundary candidates — recall-improvement plan Task 8.
// BOUNDED recovery of split ("cảmơn" -> "cảm ơn") and merge
// ("như ng" -> "nhưng") errors. Split only at internal character
// boundaries producing two CORPUS-BACKED words; merge only an adjacent
// pair separated by ONE ordinary space whose concatenation is a
// corpus-backed word. No arbitrary sequence-to-sequence rewriting.
// Pure analysis + provisional decisions; the rule owns emission policy.
// NOTE: deliberately imports NOTHING from linguistic-rules.mjs — callers
// inject classifyToken to avoid an ESM import cycle.
// ============================================================

export const WORD_BOUNDARY_MAX_SPLITS = 4;
export const WORD_BOUNDARY_MAX_MERGES = 2;

const BLOCKED_CLASSES = new Set([
  'PROTECTED', 'WHITELISTED', 'CODE_LIKE', 'ABBREVIATION',
]);

/** letters-only guard: URLs, codes, placeholders never participate */
function isPlainWord(original) {
  return /^\p{L}+$/u.test(original);
}

/**
 * Split candidates for one WORD token: every internal boundary whose two
 * halves are corpus-backed (realFrequency > 0), ranked by combined log
 * frequency then lexicographically — bounded, deterministic.
 */
export function generateSplitCandidates({ token, services }) {
  const norm = token.normalized;
  const out = [];
  if (!isPlainWord(token.original)) return out;
  for (let i = 1; i < norm.length - 1; i++) {
    const left = norm.slice(0, i);
    const right = norm.slice(i);
    // single-character halves are junk in corpus terms (bare letters carry
    // large news-corpus frequencies) and never constitute real syllables
    if (left.length < 2 || right.length < 2) continue;
    const freqLeft = services.lexicon.realFrequency(left);
    if (freqLeft <= 0) continue;
    const freqRight = services.lexicon.realFrequency(right);
    if (freqRight <= 0) continue;
    out.push({
      left, right, freqLeft, freqRight,
      score: Math.log10(freqLeft + 1) + Math.log10(freqRight + 1),
    });
  }
  out.sort((a, b) => b.score - a.score || a.left.localeCompare(b.left));
  return out.slice(0, WORD_BOUNDARY_MAX_SPLITS);
}

/**
 * Merge candidate for words[idx] + words[idx+1]: allowed only when they are
 * separated by exactly one ordinary space and the concatenation is a
 * corpus-backed word. The reported span covers BOTH tokens INCLUDING the
 * original whitespace characters verbatim.
 */
export function generateMergeCandidates(
  { words, idx, text, services },
) {
  const out = [];
  if (idx + 1 >= words.length) return out;
  const t1 = words[idx];
  const t2 = words[idx + 1];
  // ordinary single-space separation only (tokenizer keeps exact offsets)
  if (text.slice(t1.end, t2.start) !== ' ') return out;
  if (!isPlainWord(t1.original) || !isPlainWord(t2.original)) return out;
  const merged = `${t1.normalized}${t2.normalized}`;
  const freq = services.lexicon.realFrequency(merged);
  if (freq <= 0) return out;
  out.push({
    merged, freq,
    score: Math.log10(freq + 1),
    start: t1.start, end: t2.end,
    value: text.slice(t1.start, t2.end),
  });
  return out.slice(0, WORD_BOUNDARY_MAX_MERGES);
}

const bigramCount = (lm, a, b) => (a != null && b != null
  ? (lm.bigram.get(`${a} ${b}`) ?? 0) : 0);

/**
 * Full boundary decision for word index `idx` under the current snapshot:
 *   splits  — solid-written compound suggestions ("cảmơn" -> "cảm ơn");
 *   merges  — accidental-split repairs ("như ng" -> "nhưng").
 * Each item carries an exact span [start,end), the ORIGINAL substring as
 * `value`, a ranked `suggestion` and a provisional `wouldEmit` gate:
 * both/all parts must clear `wordBoundaryMinFrequency` AND the repaired
 * sequence needs DIRECT context attestation (its own bigram for splits,
 * neighbour bigrams for merges) so unigram frequency alone never fires.
 */
export function evaluateWordBoundaryCandidates(
  { services, snap, languageModel, ctx, doc, words, idx, classify = null,
    text = null },
) {
  if (snap.get('linguistic.wordBoundaryCorrectionMode') === 'OFF') {
    return { splits: [], merges: [] };
  }
  const originalText = text ?? ctx?.content ?? '';
  const t = words[idx];
  if (!t) return { splits: [], merges: [] };
  const blocked = (token) => classify != null
    && BLOCKED_CLASSES.has(classify(token, ctx, doc, services));
  if (blocked(t)) return { splits: [], merges: [] };
  const minFreq = snap.get('linguistic.wordBoundaryMinFrequency') ?? 2000;
  // Never split a well-attested dictionary word ("nhưng", "trương"): a
  // genuine solid-written error ("cảmơn") has NO corpus frequency of its
  // own. This is the boundary-lane version of "never judge a real word
  // from candidate probability alone".
  const solidFreq = services.lexicon.realFrequency(t.normalized);
  const splitAllowed = solidFreq < minFreq;
  const prevW = idx > 0 ? words[idx - 1].normalized.toLowerCase() : null;
  const nextW = idx + 1 < words.length
    ? words[idx + 1].normalized.toLowerCase() : null;
  const lm = languageModel;

  const splits = (splitAllowed
    ? generateSplitCandidates({ token: t, services }) : [])
    .map((c) => {
      const suggestion = `${c.left} ${c.right}`;
      // direct evidence that the SPLIT form occurs in context: the pair's
      // own bigram plus one anchored neighbour side
      const ownBigram = bigramCount(lm, c.left, c.right) > 0;
      const ctxAttested = bigramCount(lm, prevW, c.left) > 0
        || bigramCount(lm, c.right, nextW) > 0;
      return {
        start: t.start, end: t.end, value: t.original,
        suggestion,
        score: Math.round((c.score + (ownBigram ? 0.5 : 0)
          + (ctxAttested ? 0.25 : 0)) * 1000) / 1000,
        wouldEmit: c.freqLeft >= minFreq && c.freqRight >= minFreq
          && ownBigram && (ctxAttested || prevW == null && nextW == null),
      };
    });

  const merges = generateMergeCandidates({
    words, idx, text: originalText, services,
  })
    .filter((c) => !(idx + 1 < words.length && blocked(words[idx + 1])))
    .map((c) => {
      const w = c.merged.toLowerCase();
      const ctxAttested = bigramCount(lm, prevW, w) > 0
        || bigramCount(lm, w, nextW) > 0;
      return {
        start: c.start, end: c.end, value: c.value,
        suggestion: c.merged,
        score: Math.round((c.score + (ctxAttested ? 0.5 : 0)) * 1000) / 1000,
        wouldEmit: c.freq >= minFreq && ctxAttested,
      };
    });

  return { splits, merges };
}
