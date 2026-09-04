// ============================================================
// Correction-relation taxonomy — recall-improvement plan Task 1.
// Classifies the linguistic relation between a labeled error surface and
// its correction target so recall failures can be attributed to the lane
// that OWNS them (PMD owns unaccented same-key, wrong-diacritic lane owns
// accented same-key, real-word lane owns different-key single tokens).
// Pure analysis data: never mutates text, never decides emission.
// ============================================================
import { accentKey, hasVietnameseAccent } from './normalizer.mjs';

/**
 * Existing benchmark normalization policy (tools/run_spelling_eval.mjs):
 * lowercase + strip LEADING/TRAILING non-letter/non-digit runs. Internal
 * punctuation is preserved — "cảmơn." keeps its internal structure.
 */
function benchmarkSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

/**
 * Classify how a correction target relates to its error surface.
 *
 * Relation follows the INPUT's accent state for same-key pairs:
 *   - input unaccented  -> UNACCENTED_SAME_KEY  (missing-diacritic lane)
 *   - input accented    -> ACCENTED_SAME_KEY    (wrong-diacritic spelling lane)
 *
 * Token-count differences dominate same-key detection:
 *   - input 1 token -> target n tokens : SPLIT
 *   - input n tokens -> target 1 token : MERGE
 *
 * Residual multi-token pairs with differing keys classify as
 * DIFFERENT_KEY_MULTI_TOKEN (documented extension) so that
 * DIFFERENT_KEY_SINGLE_TOKEN stays precise for real-word lane routing.
 *
 * @param {string} input  labeled error surface ("quỳ")
 * @param {string} target correction suggestion ("quý")
 * @returns {'IDENTITY'|'UNACCENTED_SAME_KEY'|'ACCENTED_SAME_KEY'
 *   |'DIFFERENT_KEY_SINGLE_TOKEN'|'DIFFERENT_KEY_MULTI_TOKEN'
 *   |'SPLIT'|'MERGE'}
 */
export function classifyCorrectionRelation(input, target) {
  const inNorm = benchmarkSurface(input).normalize('NFC');
  const tgtNorm = benchmarkSurface(target).normalize('NFC');
  if (inNorm === tgtNorm) return 'IDENTITY';
  const inToks = inNorm ? inNorm.split(/\s+/) : [];
  const tgtToks = tgtNorm ? tgtNorm.split(/\s+/) : [];
  if (inToks.length > 1 && tgtToks.length === 1) return 'MERGE';
  if (inToks.length === 1 && tgtToks.length > 1) return 'SPLIT';

  const joinedIn = inToks.join('');
  const joinedTarget = tgtToks.join('');
  // A side emptied by normalization cannot share a key unless both did
  // (already returned IDENTITY above).
  if (!joinedIn || !joinedTarget) {
    return 'DIFFERENT_KEY_SINGLE_TOKEN';
  }
  if (accentKey(joinedIn) === accentKey(joinedTarget)) {
    return hasVietnameseAccent(joinedIn)
      ? 'ACCENTED_SAME_KEY'
      : 'UNACCENTED_SAME_KEY';
  }
  return inToks.length === 1 && tgtToks.length === 1
    ? 'DIFFERENT_KEY_SINGLE_TOKEN'
    : 'DIFFERENT_KEY_MULTI_TOKEN';
}

// ============================================================
// Task 2 (recall-improvement plan): semantic product-recall matching.
// VSEC labels every correction POSSIBLE_SPELLING_ERROR while the product
// intentionally emits unaccented same-key fixes as
// POSSIBLE_MISSING_DIACRITIC — strict rule-ID recall stays the contract
// metric, but the PRIMARY PRODUCT-RECALL view credits a correction
// wherever a LINGUISTIC rule emitted it with matching value + suggestion.
// Deterministic rules (whitespace/punct/abbreviation/…) never count.
// ============================================================
const LINGUISTIC_RULE_IDS = new Set([
  'POSSIBLE_MISSING_DIACRITIC',
  'POSSIBLE_SPELLING_ERROR',
  // review B1: the word-boundary lane now has its own id; it is still a
  // linguistic correction for the product-recall view.
  'POSSIBLE_WORD_BOUNDARY_ERROR',
]);

function normalizedValueMatches(issue, expected) {
  return benchmarkSurface(issue?.value).normalize('NFC')
    === benchmarkSurface(expected?.value).normalize('NFC');
}

function normalizedSuggestionMatches(issue, expected) {
  // supports both expect shapes: vsec adapter `suggestions:[…]` and
  // benchmark corpus singular `suggestion`
  const wanted = expected?.suggestion !== undefined
    ? benchmarkSurface(expected.suggestion).normalize('NFC')
    : benchmarkSurface(expected?.suggestions?.[0]).normalize('NFC');
  return (issue?.suggestions ?? [])
    .some((s) => benchmarkSurface(s).normalize('NFC') === wanted);
}

function positionedValueMatches(issue, expected) {
  if (!Number.isInteger(expected?.positionStart) || !Number.isInteger(expected?.positionEnd)) {
    return true;
  }
  return Number.isInteger(issue?.start) && Number.isInteger(issue?.end)
    && issue.start >= expected.positionStart && issue.end <= expected.positionEnd;
}

/**
 * Semantic linguistic match: ruleId must be a LINGUISTIC correction rule,
 * then value AND suggestion must match under benchmark normalization.
 * Strict contract matching (`benchmarkIssueMatches`) is untouched.
 *
 * @param {{ruleId:string, value:string, suggestions?:string[]}} issue
 * @param {{value:string, suggestions?:string[], suggestion?:string}} expected
 */
export function linguisticCorrectionMatches(issue, expected) {
  if (!issue || !expected) return false;
  if (!LINGUISTIC_RULE_IDS.has(issue.ruleId)) return false;
  return positionedValueMatches(issue, expected)
    && normalizedValueMatches(issue, expected)
    && normalizedSuggestionMatches(issue, expected);
}
