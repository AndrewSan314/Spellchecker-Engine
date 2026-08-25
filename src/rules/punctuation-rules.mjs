// ============================================================
// Punctuation rules — plan §11.6 / §11.7
//   REPEATED_PUNCTUATION: runs of ! ? , ; : (>=2) and '.' (allow exactly "...")
//   MISSING_WHITESPACE_AFTER_PUNCTUATION: with exceptions
//     (protected ranges, ellipsis continuation, initialisms like A.B.C)
// ============================================================
import { RuleIds, Severity, ValidationIssue } from '../core.mjs';

const REPEAT_PUNCT = new Set(['!', '?', ',', ';', ':']);
const AFTER_PUNCT = new Set(['.', ',', '!', '?', ':', ';']);
const MAX_DOTS_ALLOWED = 3; // "..." is allowed by default (plan §11.6)

/** Find runs of the same punctuation char. Returns [{char,start,len}] */
export function repeatedPunctRuns(text) {
  const runs = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (!REPEAT_PUNCT.has(ch) && ch !== '.') { i++; continue; }
    let j = i + 1;
    while (j < text.length && text[j] === ch) j++;
    const len = j - i;
    if (ch === '.') {
      if (len > MAX_DOTS_ALLOWED) runs.push({ char: ch, start: i, len });
    } else if (len >= 2) {
      runs.push({ char: ch, start: i, len });
    }
    i = j;
  }
  return runs;
}

export function createRepeatedPunctuationRule() {
  return {
    id: () => RuleIds.REPEATED_PUNCTUATION,
    priority: () => 300,
    supports: () => true,
    validate: (_ctx, doc) => {
      const text = doc.originalText;
      const issues = [];
      for (const run of repeatedPunctRuns(text)) {
        if (doc.inProtectedRange(run.start, run.start + run.len)) continue;
        issues.push(new ValidationIssue(
          RuleIds.REPEATED_PUNCTUATION, Severity.WARNING,
          run.start, run.start + run.len,
          text.substring(run.start, run.start + run.len),
          `Phát hiện dấu "${run.char}" lặp lại ${run.len} lần liên tiếp.`,
          [], null,
        ));
      }
      return issues;
    },
  };
}

/**
 * True when punct at index i sits in an initialism chain like "A.B.C":
 * both neighbours are single uppercase-letter tokens.
 * Applies to '.' only — "A,B" is a genuine missing-space error.
 */
function isInitialismNeighbourhood(text, i) {
  if (text[i] !== '.') return false;
  const prev = i > 0 ? text[i - 1] : '';
  const next = i + 1 < text.length ? text[i + 1] : '';
  if (!/[A-Z]/.test(prev) || !/[A-Z]/.test(next)) return false;
  // neighbour letters must themselves be single-letter tokens
  const prevIsTokenStart = i - 2 < 0 || /[\s\d]/.test(text[i - 2]) || /[.,;:!?)]/.test(text[i - 2]);
  const nextEndsToken = i + 2 >= text.length || /[\s\d]/.test(text[i + 2]) || /[.,;:!?(]/.test(text[i + 2]);
  return prevIsTokenStart && nextEndsToken;
}

export function detectMissingWhitespaceAfterPunctuation(text, doc) {
  const hits = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (!AFTER_PUNCT.has(ch)) continue;

    // repeated-run continuation: dot of "...", char of "!!!" — handled by other rule
    const prevSame = i > 0 && text[i - 1] === ch && (ch !== '.' || true);
    if (prevSame) continue;

    const next = i + 1 < text.length ? text[i + 1] : '';
    if (next === '') continue;
    if (/\s/.test(next)) continue;              // whitespace present -> OK
    if (next === ch) continue;                  // run continues ("!!", "...")
    if (AFTER_PUNCT.has(next) && ch === '.' ) {
      // e.g. ".," odd mixes — leave to future INFO rule, don't double-flag
      continue;
    }

    if (isInitialismNeighbourhood(text, i)) continue;   // A.B.C exception
    if (doc.inProtectedRange(i, i + 1)) continue;       // URL/number/date/code

    hits.push({ index: i });
  }
  return hits;
}

export function createMissingWhitespaceAfterPunctuationRule() {
  return {
    id: () => RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION,
    priority: () => 300,
    supports: () => true,
    validate: (_ctx, doc) => {
      const text = doc.originalText;
      return detectMissingWhitespaceAfterPunctuation(text, doc).map(({ index }) =>
        new ValidationIssue(
          RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION, Severity.WARNING,
          index, index + 1,
          text[index],
          `Thiếu khoảng trắng sau dấu "${text[index]}".`,
          [`${text[index]} `], null,
        ));
    },
  };
}
