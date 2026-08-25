// ============================================================
// ABBREVIATION_DETECTED — plan §11.11
// supports: abbreviationService.isRuleEnabledFor(context.brandname)
// Seed: VT_TENDOO => enabled, others => disabled.
// Whole-token exact match only (never substring).
// ============================================================
import { RuleIds, Severity, ValidationIssue } from '../core.mjs';

export function createAbbreviationRule(abbreviationService) {
  return {
    id: () => RuleIds.ABBREVIATION_DETECTED,
    priority: () => 800,
    supports: (ctx) => abbreviationService.isRuleEnabledFor(ctx.brandname),
    validate: (_ctx, doc) => {
      const issues = [];
      for (const t of doc.tokens) {
        if (t.type !== 'WORD') continue;
        if (doc.inProtectedRange(t.start, t.end)) continue;
        if (!abbreviationService.isAbbreviation(t.original)) continue;
        issues.push(new ValidationIssue(
          RuleIds.ABBREVIATION_DETECTED, Severity.WARNING,
          t.start, t.end, t.original,
          `Phát hiện từ viết tắt "${t.original}", nên viết đầy đủ trong SMS thương hiệu.`,
          [], null,
        ));
      }
      return issues;
    },
  };
}
