// ============================================================
// ValidationDocumentBuilder — plan §6 / §7
// Builds the immutable analysis document. originalText is the
// ONLY source of output offsets and values.
// ============================================================
import { tokenize } from './tokenizer.mjs';
import { detectProtectedRanges, ProtectedRange } from './protected-ranges.mjs';
import { normalizeNfc } from './normalizer.mjs';

export class ValidationDocument {
  constructor(originalText, tokens, protectedRanges) {
    this.originalText = originalText;                 // source of truth
    this.analysisText = normalizeNfc(originalText);   // lookup only — never for offsets
    this.tokens = Object.freeze(tokens.map((t) => Object.freeze({ ...t })));
    this.protectedRanges = Object.freeze([...protectedRanges]);
    Object.freeze(this);
  }

  /** does [s,e) overlap any protected range? */
  inProtectedRange(s, e) {
    return this.protectedRanges.some((r) => r.overlaps(s, e));
  }

  /** is [s,e) fully inside one protected range? */
  insideSingleProtectedRange(s, e) {
    return this.protectedRanges.some((r) => r.contains(s, e));
  }

  /** token containing offset (or null) */
  tokenAt(index) {
    return this.tokens.find((t) => index >= t.start && index < t.end) ?? null;
  }
}

/** plan §7 build order: no trim, no lowercase of original. */
export function buildValidationDocument(context) {
  const original = context.content;
  return new ValidationDocument(
    original,
    tokenize(original),
    detectProtectedRanges(original),
  );
}

export { ProtectedRange };
