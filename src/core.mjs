// ============================================================
// Core contracts — mirrors plan §5 (Domain Contracts)
// Offsets: UTF-16 code units, start inclusive / end exclusive.
// Original text is NEVER mutated; offsets always refer to it.
// ============================================================

/** @enum {string} plan §5.1 */
export const Severity = Object.freeze({
  ERROR: 'ERROR',
  WARNING: 'WARNING',
  INFO: 'INFO',
});

/** @enum {string} plan §5.2 */
export const MessageMode = Object.freeze({
  ACCENTED: 'ACCENTED',
  NON_ACCENTED: 'NON_ACCENTED',
});

/** @enum {string} plan §6.1 */
export const TokenType = Object.freeze({
  WORD: 'WORD',
  NUMBER: 'NUMBER',
  PUNCTUATION: 'PUNCTUATION',
  WHITESPACE: 'WHITESPACE',
  OTHER: 'OTHER',
});

/** @enum {string} plan §6.2 */
export const ProtectedRangeType = Object.freeze({
  URL: 'URL',
  EMAIL: 'EMAIL',
  PHONE: 'PHONE',
  DATE: 'DATE',
  TIME: 'TIME',
  NUMBER: 'NUMBER',
  MONEY: 'MONEY',
  TRANSACTION_CODE: 'TRANSACTION_CODE',
  PRODUCT_CODE: 'PRODUCT_CODE',
  PLACEHOLDER: 'PLACEHOLDER',
});

/**
 * plan §5.3 ValidationContext
 */
export class ValidationContext {
  /**
   * @param {string} content
   * @param {MessageMode} messageMode
   * @param {string|null} brandname
   * @param {string|null} customerId
   * @param {object} [options] demo-only runtime overrides (thresholds)
   */
  constructor(content, messageMode, brandname = null, customerId = null, options = {}) {
    if (typeof content !== 'string') throw new TypeError('content must be a string');
    if (!Object.values(MessageMode).includes(messageMode)) {
      throw new TypeError(`messageMode must be ACCENTED|NON_ACCENTED, got: ${messageMode}`);
    }
    this.content = content;
    this.messageMode = messageMode;
    this.brandname = brandname;
    this.customerId = customerId;
    this.options = options;
    Object.freeze(this);
  }
}

/**
 * plan §5.4 ValidationIssue
 * Invariant: context.content.substring(start, end) === value
 */
export class ValidationIssue {
  /**
   * @param {string} ruleId stable id from RuleIds
   * @param {Severity} severity
   * @param {number} start inclusive UTF-16 index on original text
   * @param {number} end exclusive UTF-16 index on original text
   * @param {string} value exact original slice [start,end)
   * @param {string} message human message (Vietnamese)
   * @param {string[]} suggestions display-only, never auto-applied
   * @param {number|null} confidence null for deterministic rules
   */
  constructor(ruleId, severity, start, end, value, message, suggestions = [], confidence = null) {
    if (typeof ruleId !== 'string' || !ruleId) throw new TypeError('ruleId required');
    if (!Object.values(Severity).includes(severity)) throw new TypeError('bad severity');
    if (!Number.isInteger(start) || start < 0) throw new TypeError('start < 0');
    if (!Number.isInteger(end) || end < start) throw new TypeError('end < start');
    if (typeof value !== 'string') throw new TypeError('value must be string');
    if (typeof message !== 'string' || !message) throw new TypeError('message required');

    this.ruleId = ruleId;
    this.severity = severity;
    this.start = start;
    this.end = end;
    this.value = value;
    this.message = message;
    this.suggestions = Object.freeze([...(suggestions ?? [])]);
    this.confidence = confidence;
    Object.freeze(this);
  }

  /** span overlap helper (inclusive/exclusive semantics) */
  overlaps(s, e) {
    return s < this.end && e > this.start;
  }
}

/**
 * plan §5.5 ValidationResult
 * valid = !hasErrors ; hasErrors/hasWarnings derived from issues.
 * shadowIssues: linguistic issues collected in SHADOW mode — never surfaced
 * to users (plan "Implement SHADOW đúng nghĩa").
 */
export class ValidationResult {
  constructor(valid, hasErrors, hasWarnings, issues, shadowIssues = [], summary = null) {
    this.valid = valid;
    this.hasErrors = hasErrors;
    this.hasWarnings = hasWarnings;
    this.issues = Object.freeze([...issues]);
    this.shadowIssues = Object.freeze([...shadowIssues]);
    // Message-level observations for the UI (never a substitute for issues):
    // { unaccentedContent: boolean, linguisticIssueCount: number, wordCount: number }
    this.summary = summary ? Object.freeze({ ...summary }) : null;
    Object.freeze(this);
  }
}

/**
 * plan "Rule critical crash không được fail-open".
 * Thrown when a CRITICAL rule fails — the caller must NOT report
 * `valid=true` on a validation that never fully ran.
 */
export class ValidationEngineError extends Error {
  constructor(ruleId, cause) {
    super(`CRITICAL rule failed: ${ruleId}${cause ? ` — ${cause.message ?? cause}` : ''}`);
    this.name = 'ValidationEngineError';
    this.ruleId = ruleId;
    this.cause = cause;
  }
}

/**
 * plan §10 RuleIds — STABLE contract with frontend. Never rename.
 */
export const RuleIds = Object.freeze({
  ACCENT_CHARACTER_IN_NON_ACCENT_MODE: 'ACCENT_CHARACTER_IN_NON_ACCENT_MODE',
  POSSIBLE_MISSING_DIACRITIC: 'POSSIBLE_MISSING_DIACRITIC',
  ABBREVIATION_DETECTED: 'ABBREVIATION_DETECTED',
  LEADING_WHITESPACE: 'LEADING_WHITESPACE',
  TRAILING_WHITESPACE: 'TRAILING_WHITESPACE',
  MULTIPLE_WHITESPACE: 'MULTIPLE_WHITESPACE',
  WHITESPACE_BEFORE_PUNCTUATION: 'WHITESPACE_BEFORE_PUNCTUATION',
  REPEATED_PUNCTUATION: 'REPEATED_PUNCTUATION',
  MISSING_WHITESPACE_AFTER_PUNCTUATION: 'MISSING_WHITESPACE_AFTER_PUNCTUATION',
  INVALID_CHARACTER: 'INVALID_CHARACTER',
  ZERO_WIDTH_CHARACTER: 'ZERO_WIDTH_CHARACTER',
  NON_BREAKING_SPACE: 'NON_BREAKING_SPACE',
  POSSIBLE_SPELLING_ERROR: 'POSSIBLE_SPELLING_ERROR',
  // Review B1: the split/merge lane used to reuse POSSIBLE_SPELLING_ERROR,
  // so IssueConflictResolver silently dropped whichever of the two lanes
  // registered second on an identical span, and per-lane precision/recall
  // could not be measured. It now has its own stable id.
  POSSIBLE_WORD_BOUNDARY_ERROR: 'POSSIBLE_WORD_BOUNDARY_ERROR',
});

/**
 * plan §3.1 SmsValidationRule interface (JS duck-typing).
 * Rules must be deterministic given (context, configSnapshot),
 * must not mutate inputs, must not do IO, must not throw for normal user input.
 * @typedef {Object} SmsValidationRule
 * @property {() => string} id
 * @property {() => number} priority
 * @property {(ctx: ValidationContext) => boolean} supports
 * @property {(ctx: ValidationContext, doc: ValidationDocument) => ValidationIssue[]} validate
 */

/**
 * plan §25.1 conflict priority table (specific > generic).
 */
export const RULE_PRIORITY = Object.freeze({
  [RuleIds.ZERO_WIDTH_CHARACTER]: 1000,
  [RuleIds.INVALID_CHARACTER]: 950,
  [RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE]: 900,
  [RuleIds.ABBREVIATION_DETECTED]: 800,
  [RuleIds.POSSIBLE_MISSING_DIACRITIC]: 700,
  [RuleIds.POSSIBLE_SPELLING_ERROR]: 600,
  [RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR]: 580,
  [RuleIds.NON_BREAKING_SPACE]: 550,
  [RuleIds.LEADING_WHITESPACE]: 400,
  [RuleIds.TRAILING_WHITESPACE]: 400,
  [RuleIds.MULTIPLE_WHITESPACE]: 400,
  [RuleIds.WHITESPACE_BEFORE_PUNCTUATION]: 400,
  [RuleIds.REPEATED_PUNCTUATION]: 300,
  [RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION]: 300,
});

export function priorityOf(ruleId) {
  return RULE_PRIORITY[ruleId] ?? 100;
}

/** fail-closed tiers (plan §8): deterministic business rules are critical */
export const CRITICAL_RULE_IDS = Object.freeze(new Set([
  RuleIds.ACCENT_CHARACTER_IN_NON_ACCENT_MODE,
  RuleIds.ABBREVIATION_DETECTED,
  RuleIds.LEADING_WHITESPACE,
  RuleIds.TRAILING_WHITESPACE,
  RuleIds.MULTIPLE_WHITESPACE,
  RuleIds.WHITESPACE_BEFORE_PUNCTUATION,
  RuleIds.REPEATED_PUNCTUATION,
  RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION,
  RuleIds.INVALID_CHARACTER,
]));

/** SHADOW-mode candidates: run, record metrics, NEVER return to users */
export const LINGUISTIC_RULE_IDS = Object.freeze(new Set([
  RuleIds.POSSIBLE_MISSING_DIACRITIC,
  RuleIds.POSSIBLE_SPELLING_ERROR,
  RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR,
]));
