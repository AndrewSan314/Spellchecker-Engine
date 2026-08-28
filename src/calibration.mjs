// ============================================================
// Confidence calibration — plan Phase 1 (unified, non-saturating scale)
//
// PROBLEM this replaces. Both linguistic rules shipped a number in
// ValidationIssue.confidence, but they meant different things and neither
// was calibrated:
//
//   POSSIBLE_MISSING_DIACRITIC  conf = p1 / (p1 + p2)   (pairwise dominance)
//   POSSIBLE_SPELLING_ERROR     conf = p1               (raw softmax)
//
// p1/(p1+p2) is algebraically sigmoid(s1 - s2) on the RAW score scale. Those
// scores are log-domain LM sums whose winner-vs-runner-up gap routinely
// exceeds 20 nats, so the sigmoid saturates: conf === 1.0 EXACTLY whenever
// the runner-up is negligible. Measured consequence — no threshold below 1.0
// discriminates anything (plainFormStrongMargin 0.9 -> 0.98 moved PMD
// precision 0.525 -> 0.530), and all four false alarms on the real marketing
// corpus fired at conf 1.0.
//
// FIX. Divide the log-odds margin by a temperature T fit offline, then map
// through a sigmoid. Same transform, same monotone ordering of decisions,
// but the output spreads across (0.5, 1) instead of pinning at 1.0 — so the
// gate thresholds regain resolution. Both rules call this, so a threshold
// finally means the same thing in both.
//
// T is a pure scale on the score margin; T=1 reproduces the legacy PMD
// number bit-for-bit, which is what makes this safe to land before the
// constant is fit.
// ============================================================

/** Default temperature. 1 == legacy behaviour (sigmoid of the raw margin). */
export const DEFAULT_CONFIDENCE_TEMPERATURE = 1;

/**
 * Calibrated confidence from a RAW score margin (log domain, nats).
 *
 * @param {number} marginNats  s1 - s2, the winner's lead in score units.
 *   Pass Infinity when there is no runner-up.
 * @param {number} temperature  divisor; larger = less confident.
 * @returns {number} in (0.5, 1] for a non-negative margin.
 */
export function calibratedConfidence(marginNats, temperature = DEFAULT_CONFIDENCE_TEMPERATURE) {
  const T = Number.isFinite(temperature) && temperature > 0
    ? temperature : DEFAULT_CONFIDENCE_TEMPERATURE;
  if (!Number.isFinite(marginNats)) return marginNats > 0 ? 1 : 0;
  const z = marginNats / T;
  // numerically stable logistic
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/**
 * Recover the raw score margin from a softmax probability vector. Softmax is
 * shift-invariant, so log(p1) - log(p2) === s1 - s2 exactly; going through
 * the probabilities loses precision only once both underflow to 0, which the
 * ordered-scores overload below avoids.
 *
 * @param {number} p1 winner probability
 * @param {number} p2 runner-up probability (0 or undefined => no runner-up)
 */
export function marginFromProbabilities(p1, p2) {
  if (!(p2 > 0)) return Infinity;
  if (!(p1 > 0)) return -Infinity;
  return Math.log(p1) - Math.log(p2);
}

/**
 * Temperature-scaling fit by 1-D search on negative log-likelihood.
 * Deterministic (fixed grid + bisection refine), no dependencies.
 *
 * @param {Array<{margin:number, correct:boolean}>} samples
 *   margin = raw s1-s2 at the decision; correct = winner was the right call.
 * @returns {{temperature:number, nll:number, n:number}}
 */
export function fitTemperature(samples) {
  const usable = samples.filter((s) => Number.isFinite(s.margin));
  if (usable.length === 0) {
    return { temperature: DEFAULT_CONFIDENCE_TEMPERATURE, nll: NaN, n: 0 };
  }
  const nllAt = (T) => {
    let sum = 0;
    for (const s of usable) {
      const p = Math.min(1 - 1e-12, Math.max(1e-12, calibratedConfidence(s.margin, T)));
      sum -= s.correct ? Math.log(p) : Math.log(1 - p);
    }
    return sum / usable.length;
  };
  // coarse geometric grid, then bisection refine around the winner
  let bestT = DEFAULT_CONFIDENCE_TEMPERATURE;
  let bestNll = Infinity;
  for (let e = -2; e <= 3.0001; e += 0.125) {
    const T = 10 ** e;
    const v = nllAt(T);
    if (v < bestNll) { bestNll = v; bestT = T; }
  }
  let lo = bestT / 2;
  let hi = bestT * 2;
  for (let it = 0; it < 60; it++) {
    const m1 = lo + (hi - lo) / 3;
    const m2 = hi - (hi - lo) / 3;
    if (nllAt(m1) <= nllAt(m2)) hi = m2; else lo = m1;
  }
  const T = (lo + hi) / 2;
  const nll = nllAt(T);
  return nll <= bestNll
    ? { temperature: T, nll, n: usable.length }
    : { temperature: bestT, nll: bestNll, n: usable.length };
}

/**
 * Expected calibration error, equal-width bins over the confidence range.
 * @param {Array<{confidence:number, correct:boolean}>} samples
 */
export function expectedCalibrationError(samples, bins = 10) {
  if (samples.length === 0) return NaN;
  const acc = Array.from({ length: bins }, () => ({ n: 0, conf: 0, hit: 0 }));
  for (const s of samples) {
    const c = Math.min(1, Math.max(0, s.confidence));
    const b = Math.min(bins - 1, Math.floor(c * bins));
    acc[b].n++; acc[b].conf += c; acc[b].hit += s.correct ? 1 : 0;
  }
  let ece = 0;
  for (const b of acc) {
    if (!b.n) continue;
    ece += (b.n / samples.length) * Math.abs(b.conf / b.n - b.hit / b.n);
  }
  return ece;
}
