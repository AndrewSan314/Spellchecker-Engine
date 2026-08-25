// ============================================================
// ACCENT_CHARACTER_IN_NON_ACCENT_MODE — plan §11.1
// supports: messageMode == NON_ACCENTED. Severity ERROR.
// Output granularity: group per WORD token (UI sees "quý" not "ý").
// ============================================================
import { RuleIds, Severity, ValidationIssue, MessageMode } from '../core.mjs';
import { isVietnameseAccentedChar } from '../normalizer.mjs';

export function findAccentedTokens(doc) {
  // plan "Fix Unicode NFD bypass": detect accents on the NFC-ANALYSIS form
  // so decomposed input ("quy" + U+0301) is caught identically to precomposed
  // "quý". Output start/end/value still reference the ORIGINAL text.
  return doc.tokens.filter(
    (t) => t.type === 'WORD'
      && [...t.original.normalize('NFC')].some((ch) => isVietnameseAccentedChar(ch)),
  );
}

export function createAccentCharacterInNonAccentModeRule() {
  return {
    id: () => RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE,
    priority: () => 900,
    supports: (ctx) => ctx.messageMode === MessageMode.NON_ACCENTED,
    validate: (_ctx, doc) =>
      findAccentedTokens(doc).map((t) =>
        new ValidationIssue(
          RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE, Severity.ERROR,
          t.start, t.end, t.original,
          `Chế độ KHÔNG DẤU nhưng phát hiện ký tự tiếng Việt có dấu trong "${t.original}".`,
          [], null,
        )),
  };
}
