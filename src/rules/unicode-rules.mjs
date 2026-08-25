// ============================================================
// Unicode / character rules — plan §11.8 .. §11.10
//   ZERO_WIDTH_CHARACTER  (U+200B/200C/200D/FEFF)
//   NON_BREAKING_SPACE    (U+00A0)
//   INVALID_CHARACTER     (character-policy based, emoji default ALLOW)
// These rules are NOT suppressed by protected ranges (plan §24.2).
// ============================================================
import { RuleIds, Severity, ValidationIssue } from '../core.mjs';

const ZERO_WIDTH = new Map([
  ['\u200B', 'U+200B ZERO WIDTH SPACE'],
  ['\u200C', 'U+200C ZERO WIDTH NON-JOINER'],
  ['\u200D', 'U+200D ZERO WIDTH JOINER'],
  ['\uFEFF', 'U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)'],
]);

export function findZeroWidthChars(text) {
  const hits = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ZERO_WIDTH.has(ch)) hits.push({ index: i, ch, name: ZERO_WIDTH.get(ch) });
  }
  return hits;
}

export function createZeroWidthCharacterRule(configService) {
  return {
    id: () => RuleIds.ZERO_WIDTH_CHARACTER,
    priority: () => 1000,
    supports: () => true,
    validate: (_ctx, doc) => {
      // plan §5: read severity at validate-time so reload takes effect
      const severity = configService?.snapshot()?.get?.('rules.zeroWidthSeverity')
        ?? Severity.WARNING;
      return findZeroWidthChars(doc.originalText).map(({ index, name }) =>
        new ValidationIssue(
          RuleIds.ZERO_WIDTH_CHARACTER, severity, index, index + 1,
          doc.originalText[index],
          `Phát hiện ký tự ẩn ${name}.`,
          [], null,
        ));
    },
  };
}

export function findNonBreakingSpaces(text) {
  const hits = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\u00A0') hits.push({ index: i });
  }
  return hits;
}

export function createNonBreakingSpaceRule() {
  return {
    id: () => RuleIds.NON_BREAKING_SPACE,
    priority: () => 550,
    supports: () => true,
    validate: (_ctx, doc) =>
      findNonBreakingSpaces(doc.originalText).map(({ index }) =>
        new ValidationIssue(
          RuleIds.NON_BREAKING_SPACE, Severity.WARNING, index, index + 1,
          doc.originalText[index],
          'Phát hiện ký tự khoảng trắng không ngắt (U+00A0).',
          [' '], null,
        )),
  };
}

/**
 * Character policy — plan §11.10.
 * Default emoji-policy ALLOW (business undecided) => only invisible control
 * characters are invalid. Policy is a config extension point.
 */
const CONTROL_ALLOWED = new Set(['\t', '\n', '\r']);

export function createInvalidCharacterRule(configService) {
  return {
    id: () => RuleIds.INVALID_CHARACTER,
    priority: () => 950,
    supports: () => true,
    validate: (_ctx, doc) => {
      // plan §5: read policy at validate-time so reload takes effect
      const policy = configService?.snapshot()?.get?.('characters.emojiPolicy')
        ?? 'ALLOW';
      if (policy === 'ALLOW') {
        // even in ALLOW mode, raw control chars are never acceptable
        const issues = [];
        const text = doc.originalText;
        for (let i = 0; i < text.length; i++) {
          const cp = text.codePointAt(i);
          const isControl = (cp < 0x20 && !CONTROL_ALLOWED.has(text[i]))
            || cp === 0x7F
            || (cp >= 0x80 && cp <= 0x9F);
          if (isControl) {
            issues.push(new ValidationIssue(
              RuleIds.INVALID_CHARACTER, Severity.ERROR, i, i + 1,
              text[i],
              `Phát hiện ký tự điều khiển không hợp lệ U+${cp.toString(16).toUpperCase().padStart(4, '0')}.`,
              [], null,
            ));
          }
        }
        return issues;
      }
      // stricter policies (INFO/ERROR for emoji & symbols): extension point,
      // not enabled by default per plan §34 "do not invent high-impact behavior".
      return [];
    },
  };
}
