// ============================================================
// Whitespace rules — plan §11.2 .. §11.5
//   LEADING_WHITESPACE / TRAILING_WHITESPACE / MULTIPLE_WHITESPACE /
//   WHITESPACE_BEFORE_PUNCTUATION
// All WARNING severity, deterministic, offset-exact.
// ============================================================
import { RuleIds, Severity } from '../core.mjs';

const PUNCT_BEFORE = new Set(['.', ',', '!', '?', ':', ';']);
const ASCII_SPACES_RE = / {2,}/g;

function wsIssue(ruleId, start, end, text) {
  return {
    make: () => new (issueCtor())(ruleId, Severity.WARNING, start, end,
      text.substring(start, end),
      messageFor(ruleId),
      [' '],
      null),
    start, end, ruleId,
  };
}

// small helper to avoid circular import noise
import { ValidationIssue } from '../core.mjs';
function issueCtor() { return ValidationIssue; }
function messageFor(ruleId) {
  switch (ruleId) {
    case RuleIds.LEADING_WHITESPACE: return 'Nội dung bắt đầu bằng khoảng trắng.';
    case RuleIds.TRAILING_WHITESPACE: return 'Nội dung kết thúc bằng khoảng trắng.';
    case RuleIds.MULTIPLE_WHITESPACE: return 'Phát hiện nhiều khoảng trắng liên tiếp.';
    case RuleIds.WHITESPACE_BEFORE_PUNCTUATION: return 'Phát hiện khoảng trắng trước dấu câu.';
    default: return 'Lỗi khoảng trắng.';
  }
}

/** Leading whitespace run starting at offset 0 */
export function detectLeadingWhitespace(text) {
  const m = /^\s+/.exec(text);
  return m ? { start: 0, end: m[0].length } : null;
}

/** Trailing whitespace run ending at text.length, not touching offset 0 */
export function detectTrailingWhitespace(text) {
  const m = /\s+$/.exec(text);
  if (!m) return null;
  const start = text.length - m[0].length;
  if (start === 0) return null; // all-whitespace => leading rule already covers
  return { start, end: text.length };
}

export function createLeadingWhitespaceRule() {
  return {
    id: () => RuleIds.LEADING_WHITESPACE,
    priority: () => 400,
    supports: () => true,
    validate: (_ctx, doc) => {
      const r = detectLeadingWhitespace(doc.originalText);
      if (!r) return [];
      return [new ValidationIssue(
        RuleIds.LEADING_WHITESPACE, Severity.WARNING, r.start, r.end,
        doc.originalText.substring(r.start, r.end),
        'Nội dung bắt đầu bằng khoảng trắng.', [], null,
      )];
    },
  };
}

export function createTrailingWhitespaceRule() {
  return {
    id: () => RuleIds.TRAILING_WHITESPACE,
    priority: () => 400,
    supports: () => true,
    validate: (_ctx, doc) => {
      const r = detectTrailingWhitespace(doc.originalText);
      if (!r) return [];
      return [new ValidationIssue(
        RuleIds.TRAILING_WHITESPACE, Severity.WARNING, r.start, r.end,
        doc.originalText.substring(r.start, r.end),
        'Nội dung kết thúc bằng khoảng trắng.', [], null,
      )];
    },
  };
}

/** Two or more consecutive ASCII spaces in the body (plan §11.4 regex ` {2,}`) */
export function createMultipleWhitespaceRule() {
  return {
    id: () => RuleIds.MULTIPLE_WHITESPACE,
    priority: () => 400,
    supports: () => true,
    validate: (_ctx, doc) => {
      const text = doc.originalText;
      const lead = detectLeadingWhitespace(text);
      const trail = detectTrailingWhitespace(text);
      const issues = [];
      ASCII_SPACES_RE.lastIndex = 0;
      let m;
      while ((m = ASCII_SPACES_RE.exec(text)) !== null) {
        const s = m.index, e = s + m[0].length;
        // body-only: leave edge runs to their dedicated rules
        if (lead && s < lead.end) continue;
        if (trail && e > trail.start) continue;
        issues.push(new ValidationIssue(
          RuleIds.MULTIPLE_WHITESPACE, Severity.WARNING, s, e,
          text.substring(s, e),
          'Phát hiện nhiều khoảng trắng liên tiếp.', [' '], null,
        ));
      }
      return issues;
    },
  };
}

/** Whitespace run immediately before . , ! ? : ; (plan §11.5) */
export function createWhitespaceBeforePunctuationRule() {
  return {
    id: () => RuleIds.WHITESPACE_BEFORE_PUNCTUATION,
    priority: () => 400,
    supports: () => true,
    validate: (_ctx, doc) => {
      const text = doc.originalText;
      const issues = [];
      for (let i = 0; i < text.length; i++) {
        if (!PUNCT_BEFORE.has(text[i])) continue;
        // walk back over contiguous whitespace
        let j = i - 1;
        while (j >= 0 && /\s/.test(text[j])) j--;
        const runStart = j + 1;
        if (runStart === i) continue;        // no whitespace before punct
        if (runStart === 0) continue;        // leading-whitespace rule owns this span
        if (doc.inProtectedRange(runStart, i)) continue; // e.g. inside URL/money
        issues.push(new ValidationIssue(
          RuleIds.WHITESPACE_BEFORE_PUNCTUATION, Severity.WARNING, runStart, i,
          text.substring(runStart, i),
          'Phát hiện khoảng trắng trước dấu câu.', [], null,
        ));
      }
      return issues;
    },
  };
}
