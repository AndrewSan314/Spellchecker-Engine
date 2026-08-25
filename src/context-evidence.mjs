// ============================================================
// Context evidence — recall-improvement plan Task 4.
// Pairwise candidate-vs-original features over FIXED observed windows.
// Every ratio is log1p(candidateCount) - log1p(originalCount): finite at
// zero, no conditional division, never a centered trigram divided by a
// non-contiguous bigram. Pure analysis data — no emission decisions here.
// ============================================================
import { accentKey } from './normalizer.mjs';
import { applyTelexHints, damerauOsaDistance } from './language.mjs';

const log1p = Math.log1p;

/**
 * Extract the `recall-pairwise-v1` evidence block for one candidate against
 * the original surface at word-sequence position `idx`.
 *
 * @param {object} p
 * @param {{unigram:Map,bigram:Map,trigramJoint:Function}} p.languageModel
 * @param {string[]} p.words  normalized WORD surfaces (observed/beam-fixed)
 * @param {number} p.idx      position of the original token in `words`
 * @param {string} p.candidateWord
 * @param {string} p.originalWord
 */
export function extractContextEvidence(
  { languageModel, words, idx, candidateWord, originalWord },
) {
  const lm = languageModel;
  const cand = String(candidateWord).toLowerCase();
  const orig = String(originalWord).toLowerCase();
  const p1 = idx > 0 ? words[idx - 1] : null;
  const p2 = idx > 1 ? words[idx - 2] : null;
  const n1 = idx + 1 < words.length ? words[idx + 1] : null;
  const n2 = idx + 2 < words.length ? words[idx + 2] : null;

  const uCnt = (w) => (w != null ? (lm.unigram.get(w) ?? 0) : 0);
  const bCnt = (a, b) => (a != null && b != null
    ? (lm.bigram.get(`${a} ${b}`) ?? 0) : 0);
  const tCnt = (a, b, c) => (a != null && b != null && c != null
    ? lm.trigramJoint(a, b, c) : 0);
  const ratio = (c, o) => log1p(c) - log1p(o);

  // the five direct context windows (null sides contribute zero on both)
  const leftBiC = bCnt(p1, cand);
  const leftBiO = bCnt(p1, orig);
  const rightBiC = bCnt(cand, n1);
  const rightBiO = bCnt(orig, n1);
  const triC = tCnt(p1, cand, n1);
  const triO = tCnt(p1, orig, n1);
  const fwdC = tCnt(cand, n1, n2);
  const fwdO = tCnt(orig, n1, n2);
  const backC = tCnt(p2, p1, cand);
  const backO = tCnt(p2, p1, orig);

  return {
    unigramLogRatio: ratio(uCnt(cand), uCnt(orig)),
    leftBigramLogRatio: ratio(leftBiC, leftBiO),
    rightBigramLogRatio: ratio(rightBiC, rightBiO),
    centeredTrigramLogRatio: ratio(triC, triO),
    forwardTrigramLogRatio: ratio(fwdC, fwdO),
    backwardTrigramLogRatio: ratio(backC, backO),
    candidateAttestedWindows: (leftBiC > 0 ? 1 : 0)
      + (rightBiC > 0 ? 1 : 0)
      + (triC > 0 ? 1 : 0)
      + (fwdC > 0 ? 1 : 0)
      + (backC > 0 ? 1 : 0),
    originalAttestedWindows: (leftBiO > 0 ? 1 : 0)
      + (rightBiO > 0 ? 1 : 0)
      + (triO > 0 ? 1 : 0)
      + (fwdO > 0 ? 1 : 0)
      + (backO > 0 ? 1 : 0),
    sameAccentKey: accentKey(cand) === accentKey(orig),
    editDistance: damerauOsaDistance(
      applyTelexHints(cand), applyTelexHints(orig)),
  };
}

/**
 * Provisional pairwise evidence score (plan Task 4 Step 5). A fixed linear
 * combination of the six log-ratio features; "pairwise margin positive"
 * means this score is > 0. Replaced by the Task 6 trained calibrator.
 */
export const PROVISIONAL_EVIDENCE_WEIGHTS = Object.freeze({
  unigramLogRatio: 0.25,
  leftBigramLogRatio: 0.5,
  rightBigramLogRatio: 0.5,
  centeredTrigramLogRatio: 1.0,
  forwardTrigramLogRatio: 1.0,
  backwardTrigramLogRatio: 1.0,
});

export function evidenceScore(features) {
  let s = 0;
  for (const [k, w] of Object.entries(PROVISIONAL_EVIDENCE_WEIGHTS)) {
    s += w * (features[k] ?? 0);
  }
  return s;
}
