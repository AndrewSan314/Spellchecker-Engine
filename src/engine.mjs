// ============================================================
// SmsValidationEngine — plan §9 / §24 / §25
// Pipeline: document -> rules -> suppression -> conflict -> sort -> result
// ============================================================
import {
  ValidationContext, ValidationResult, Severity, RuleIds, priorityOf, MessageMode,
  ValidationEngineError, CRITICAL_RULE_IDS, LINGUISTIC_RULE_IDS,
} from './core.mjs';
import { accentKey } from './normalizer.mjs';
import { buildValidationDocument } from './document-builder.mjs';
import { ValidationConfigService } from './config.mjs';
import {
  LexiconService, WhitelistService, AbbreviationService, AccentIndex,
} from './lexical.mjs';
import { RecallReranker } from './recall-reranker.mjs';
import { ErrorChannel } from './error-channel.mjs';
import { extractContextEvidence } from './context-evidence.mjs';
import { resolveArtifact, profileName } from './profile.mjs';
import { loadAttentionReranker } from './attention-reranker.mjs';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DATA_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)), 'data');

/** review B7 — most recent OPTIONAL-rule failures kept for diagnostics */
const MAX_DEGRADED_ENTRIES = 50;

/** provenance of the lexicon the last loadMergedLexicon() call produced */
let lastLexiconArtifact = null;

/**
 * Built corpus lexicon (tools/build_lm.py) merged with the curated domain
 * seed. On a key conflict the CORPUS frequency wins (it is the only real
 * evidence) and the entry is marked src:'both'; see put() below.
 */
function loadMergedLexicon() {
  // ENGINE_PROFILE decides WHICH built lexicon (see src/profile.mjs): the
  // full 51k-word corpus lexicon, or the SMS-domain one that mirrors the
  // pruned LM. The curated seed below is always merged on top.
  const artifact = resolveArtifact('lexicon');
  let built;
  try {
    built = readFileSync(artifact.path, 'utf8');
  } catch {
    return LexiconService.load();
  }
  // dictionary governance: entries banned from the lexicon ENTIRELY (built
  // AND curated) — they leave accent-index & spelling competition and are
  // re-classified as UNKNOWN, where the same-key sibling typo path applies.
  let stop = new Set();
  try {
    stop = new Set(readFileSync(path.join(DATA_DIR, 'stoplex.txt'), 'utf8')
      .split(/\r?\n/).map((l) => l.trim().toLowerCase()).filter(Boolean));
  } catch { /* optional file */ }
  const merged = new Map();
  // plan §10: corpus-backed counts beat hand-assigned seed numbers.
  // src: 'corpus' | 'curated' | 'both' — gates use realFrequency(), which
  // only trusts a frequency that came from the corpus ('corpus' or 'both').
  //
  // Review B6: the previous version could never produce 'both' (the loop
  // order is corpus-then-curated, so the corpus branch never saw a curated
  // predecessor) and its curated branch was an empty comment, while the
  // header claimed "curated entries win on key conflicts". Both directions
  // are now handled symmetrically and load order no longer matters:
  //   corpus frequency ALWAYS wins over a curated seed number,
  //   a key present in both sources is marked 'both'.
  const put = (word, freqStr, src) => {
    const freq = Number.parseInt(freqStr, 10) || 1;
    const key = word.toLowerCase();
    const prev = merged.get(key);
    if (!prev) {
      merged.set(key, { word, freq, type: 'A', src });
    } else if (src === prev.src || prev.src === 'both') {
      if (freq > prev.freq) merged.set(key, { ...prev, freq });
    } else if (src === 'corpus') {
      // curated seen first: adopt the real corpus count
      merged.set(key, { ...prev, freq, src: 'both' });
    } else {
      // curated seen second: keep the corpus count, record the overlap
      merged.set(key, { ...prev, src: 'both' });
    }
  };
  for (const line of built.split(/\r?\n/)) {
    if (!line || line.startsWith('#')) continue;
    const [w, f] = line.split('\t');
    if (w && !stop.has(w.toLowerCase())) put(w, f, 'corpus');
  }
  for (const line of readFileSync(path.join(DATA_DIR, 'lexicon.txt'), 'utf8')
    .split(/\r?\n/)) {
    if (!line || line.startsWith('#')) continue;
    const [w, f] = line.split('\t');
    if (w && !stop.has(w.toLowerCase())) put(w, f, 'curated');
  }
  // LexiconService is frozen, so the artifact provenance is recorded here and
  // read back through SmsValidationEngine.profileInfo().
  lastLexiconArtifact = {
    profile: artifact.profile,
    artifactPath: path.basename(artifact.path),
    entries: merged.size,
  };
  return LexiconService.fromEntries(merged.entries());
}
import {
  NGramLanguageModel, BeamSearchDecoder, SymSpellCandidateProvider,
} from './language.mjs';
import {
  createLeadingWhitespaceRule, createTrailingWhitespaceRule,
  createMultipleWhitespaceRule, createWhitespaceBeforePunctuationRule,
} from './rules/whitespace-rules.mjs';
import {
  createRepeatedPunctuationRule, createMissingWhitespaceAfterPunctuationRule,
} from './rules/punctuation-rules.mjs';
import {
  createZeroWidthCharacterRule, createNonBreakingSpaceRule,
  createInvalidCharacterRule,
} from './rules/unicode-rules.mjs';
import { createAccentCharacterInNonAccentModeRule } from './rules/accent-mode-rule.mjs';
import { createAbbreviationRule } from './rules/abbreviation-rule.mjs';
import {
  createPossibleMissingDiacriticRule, createPossibleSpellingErrorRule,
  createWordBoundaryRule,
} from './rules/linguistic-rules.mjs';

// plan §24.2 — rules suppressed when the issue span sits in a protected range
const SUPPRESSIBLE_IN_PROTECTED = new Set([
  RuleIds.MISSING_WHITESPACE_AFTER_PUNCTUATION,
  RuleIds.REPEATED_PUNCTUATION,
  RuleIds.WHITESPACE_BEFORE_PUNCTUATION,
  RuleIds.MULTIPLE_WHITESPACE,
  RuleIds.POSSIBLE_MISSING_DIACRITIC,
  RuleIds.POSSIBLE_SPELLING_ERROR,
  RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR,
  RuleIds.ABBREVIATION_DETECTED,
]);
// ZERO_WIDTH / INVALID / NON_BREAKING / ACCENT_MODE are NEVER suppressed here.

export class SmsValidationEngine {
  constructor({
    configService = new ValidationConfigService(),
    lexicon = loadMergedLexicon(),
    whitelist = WhitelistService.load(),
    abbreviations = AbbreviationService.load(),
    languageModel = NGramLanguageModel.load(),
    // Task 6 (recall-improvement plan): ONE immutable instance injected
    // here — the artifact is never read per token. Missing artifact loads a
    // disabled stub; present-but-malformed fails fast at construction.
    recallReranker = RecallReranker.loadDefault(),
    // Task 9 (attention plan): Optional attention reranker instance.
    attentionReranker = null,
  } = {}) {
    this.configService = configService;
    this.lexicon = lexicon;
    this.whitelistService = whitelist;
    this.abbreviationService = abbreviations;
    this.languageModel = languageModel;
    this.recallReranker = recallReranker;
    this.accentIndex = AccentIndex.build(lexicon);

    if (attentionReranker) {
      this.attentionReranker = attentionReranker;
    } else {
      const snap0 = configService.snapshot();
      const attMode = snap0.get('spelling.attentionMode')
        ?? snap0.get('linguistic.attentionMode') ?? 'OFF';
      if (attMode !== 'OFF') {
        try {
          const binP = path.join(DATA_DIR, 'attention-reranker.int8.bin');
          const metaP = path.join(DATA_DIR, 'attention-reranker.json');
          if (existsSync(binP) && existsSync(metaP)) {
            this.attentionReranker = loadAttentionReranker({ binPath: binP, metaPath: metaP });
          } else {
            this.attentionReranker = null;
          }
        } catch {
          this.attentionReranker = null;
        }
      } else {
        this.attentionReranker = null;
      }
    }
    // Task 3: scoring weights + smoothing masses come from the config
    // snapshot so calibration (Task 9) never touches model code.
    {
      const snap0 = configService.snapshot();
      languageModel.setScoreConfig({
        weights: snap0.get('linguistic.scoreWeights'),
        rightJointSmoothing: snap0.get('linguistic.rightJointSmoothing'),
        forwardJointSmoothing: snap0.get('linguistic.forwardJointSmoothing'),
      });
    }
    // Context-aware surface enumeration for OOV unaccented words (see
    // NGramLanguageModel.resolveInContext) — wired before the resolver so
    // both mechanisms share the same accent index.
    languageModel.setSurfaceProvider((key) => this.accentIndex.candidates(key));
    // Consistent context resolution: unaccented tokens (e.g. "long", "ban", "kiem")
    // are mapped to their most plausible accented surface ("lòng", "bạn", "kiểm")
    // before n-gram lookup, matching the accented training corpus distribution.
    languageModel.setResolver((w) => {
      if (typeof w !== 'string') return w;
      const stripped = accentKey(w);
      const surfaces = this.accentIndex.candidates(stripped);
      if (surfaces.length > 0 && (stripped === w || !languageModel.knows(w))) {
        return surfaces[0].word.toLowerCase();
      }
      return w;
    });
    this.beamDecoder = new BeamSearchDecoder(languageModel);
    this.typoCandidateProvider = new SymSpellCandidateProvider(lexicon);
    this.errorChannel = ErrorChannel.load(path.join(DATA_DIR, 'error-channel.json'));

    this.services = {
      configService,
      lexicon,
      whitelistService: whitelist,
      abbreviationService: abbreviations,
      accentIndex: this.accentIndex,
      languageModel,
      beamDecoder: this.beamDecoder,
      typoCandidateProvider: this.typoCandidateProvider,
      recallReranker: this.recallReranker,
      attentionReranker: this.attentionReranker,
      errorChannel: this.errorChannel,
      // Offline calibration hook: null in production, set by
      // tools/fit_confidence_temperature.mjs to capture raw score margins.
      confidenceSink: null,
    };

    // Which data artifacts this instance is actually serving (ENGINE_PROFILE).
    this.profile = {
      name: profileName(),
      lm: {
        profile: languageModel.loadDiagnostics?.profile ?? null,
        artifact: languageModel.loadDiagnostics?.artifactPath
          ? path.basename(languageModel.loadDiagnostics.artifactPath) : null,
        counts: languageModel.loadDiagnostics?.counts ?? null,
      },
      lexicon: lastLexiconArtifact,
    };

    this.documentBuilder = { build: buildValidationDocument };
    // plan §8 degraded-metric sink for OPTIONAL rule failures.
    // Review B7: the engine is a process-lifetime singleton, so an unbounded
    // array here is a slow leak (one entry per failure, forever). It is now a
    // bounded ring of the most recent failures plus monotonic counters.
    this.degraded = [];
    this.degradedCounts = new Map();
    this.degradedTotal = 0;
    this.rules = [
      createZeroWidthCharacterRule(configService),
      createInvalidCharacterRule(configService),
      createAccentCharacterInNonAccentModeRule(),
      createAbbreviationRule(abbreviations),
      createPossibleMissingDiacriticRule(this.services),
      createPossibleSpellingErrorRule(this.services),
      // Task 8: word-boundary lane (SHADOW default — emits nothing unless
      // wordBoundaryCorrectionMode is ACTIVE)
      createWordBoundaryRule(this.services),
      createNonBreakingSpaceRule(),
      createLeadingWhitespaceRule(),
      createTrailingWhitespaceRule(),
      createMultipleWhitespaceRule(),
      createWhitespaceBeforePunctuationRule(),
      createRepeatedPunctuationRule(),
      createMissingWhitespaceAfterPunctuationRule(),
    ];
  }

  /** plan §9 engine flow — rules are order-independent; output is sorted. */
  validate(context) {
    const doc = this.documentBuilder.build(context);
    // ONE immutable snapshot per validation (plan §13): every gate below reads
    // the same configuration, even if a reload lands mid-request.
    const snap = this.configService.snapshot();
    const shadowMode = snap.get('linguistic.mode') === 'SHADOW';

    const raw = [];
    const shadowRaw = [];
    for (const rule of this.rules) {
      if (!rule.supports(context)) continue;
      try {
        const found = rule.validate(context, doc);
        // plan "Implement SHADOW đúng nghĩa": linguistic issues in SHADOW
        // mode are recorded but NEVER returned to the user.
        if (shadowMode && LINGUISTIC_RULE_IDS.has(rule.id())) {
          shadowRaw.push(...found);
        } else {
          raw.push(...found);
        }
      } catch (err) {
        // plan §8: CRITICAL rules fail CLOSED (throw); OPTIONAL rules are
        // skipped with a degraded marker.
        if (CRITICAL_RULE_IDS.has(rule.id())) {
          throw new ValidationEngineError(rule.id(), err);
        }
        this.recordDegraded(rule.id(), err);
      }
    }

    // Review B5: this used to end in `.concat(shadowMode ? [] : [])`, which
    // is a no-op in both branches and read as if SHADOW were handled here.
    // SHADOW routing happens in the loop above.
    const unsuppressed = this.applySuppression(doc, raw);
    const resolved = this.resolveConflicts(unsuppressed);
    // Review B3: this compares the RAW calibrated confidence. The rules used
    // to round to 2 decimals before emitting, so a true 0.9550 was promoted
    // to 0.96 and cleared this floor; rounding is now display-only (server).
    // Review B4: both magic numbers are named config values now.
    const confidenceFloor = snap.get('linguistic.verifiedConfidenceFloor') ?? 0.96;
    const pairCountVeto = snap.get('linguistic.errorChannelPairCountVeto');
    let words;
    const verified = resolved.filter((issue) => {
      if (!LINGUISTIC_RULE_IDS.has(issue.ruleId)) return true;
      // Review B1: the word-boundary lane has its own calibrated gates and
      // no error-channel pair to prove (it rewrites token boundaries, not
      // spellings); the pair verifier below does not apply to it.
      if (issue.ruleId === RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR) return true;
      const target = issue.suggestions?.[0];
      const baseProven = !this.lexicon.contains(issue.value)
        || (this.errorChannel.hasPair(issue.value, target)
          && (pairCountVeto == null
            || this.errorChannel.pairCount(issue.value, target) !== pairCountVeto))
        || issue.confidence >= confidenceFloor;
      if (baseProven || issue.ruleId !== RuleIds.POSSIBLE_MISSING_DIACRITIC) return baseProven;

      const nearProtected = doc.protectedRanges?.some((range) =>
        Math.abs(range.start - issue.end) <= 2 || Math.abs(range.end - issue.start) <= 2);
      if (!nearProtected || !target) return false;
      words ??= doc.tokens.filter((t) => t.type === 'WORD');
      const idx = words.findIndex((word) => word.start === issue.start && word.end === issue.end);
      if (idx < 0) return false;
      const ev = extractContextEvidence({
        languageModel: this.languageModel, words: words.map((word) => word.normalized), idx,
        candidateWord: target, originalWord: words[idx].normalized,
      });
      return ev.candidateAttestedWindows > ev.originalAttestedWindows;
    });
    // SHADOW issues bypass suppression/conflict bookkeeping — they are
    // observability data only; keep them sorted for stable diffs.
    const shadowIssues = shadowMode
      ? shadowRaw.sort((a, b) => a.start - b.start || a.end - b.end)
      : [];

    const sorted = verified.sort((a, b) =>
      a.start - b.start || a.end - b.end || a.ruleId.localeCompare(b.ruleId));

    const hasErrors = sorted.some((i) => i.severity === Severity.ERROR);
    const hasWarnings = sorted.some((i) => i.severity === Severity.WARNING);
    return new ValidationResult(!hasErrors, hasErrors, hasWarnings, sorted, shadowIssues,
      this.buildSummary(context, doc, sorted, snap));
  }

  /**
   * Review §7: an unaccented SMS checked in ACCENTED mode makes the engine
   * flag nearly every token (11-12 issues on a 15-word message), which reads
   * as noise in the UI. The issues are NOT suppressed — per-token metrics and
   * benchmarks must stay comparable — but the result carries a message-level
   * observation so a client can render ONE notice instead of a wall.
   */
  buildSummary(context, doc, issues, snap) {
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    const ratioThreshold = snap.get('linguistic.unaccentedRatioThreshold') ?? 0.8;
    const minWords = snap.get('linguistic.unaccentedMinWords') ?? 5;
    let unaccented = 0;
    for (const w of words) if (accentKey(w.normalized) === w.normalized.toLowerCase()) unaccented += 1;
    const ratio = words.length ? unaccented / words.length : 0;
    return {
      wordCount: words.length,
      linguisticIssueCount: issues.filter((i) => LINGUISTIC_RULE_IDS.has(i.ruleId)).length,
      unaccentedWordRatio: Math.round(ratio * 1000) / 1000,
      // true => "this message looks like it was typed without diacritics",
      // i.e. show one message-level notice rather than one warning per token.
      unaccentedContent: context.messageMode === MessageMode.ACCENTED
        && words.length >= minWords && ratio >= ratioThreshold,
    };
  }

  /** plan §8 — bounded degraded sink (review B7) */
  recordDegraded(ruleId, err) {
    this.degradedTotal += 1;
    this.degradedCounts.set(ruleId, (this.degradedCounts.get(ruleId) ?? 0) + 1);
    this.degraded.push({
      ruleId, error: err?.message ?? String(err), at: new Date().toISOString(),
    });
    if (this.degraded.length > MAX_DEGRADED_ENTRIES) {
      this.degraded.splice(0, this.degraded.length - MAX_DEGRADED_ENTRIES);
    }
  }

  /** aggregate view for /healthz — counters survive the ring eviction */
  degradedSummary() {
    return {
      total: this.degradedTotal,
      byRule: Object.fromEntries(this.degradedCounts),
      recent: this.degraded.slice(-5),
    };
  }

  /** plan §24 IssueSuppressor */
  applySuppression(doc, issues) {
    return issues.filter((issue) => {
      if (SUPPRESSIBLE_IN_PROTECTED.has(issue.ruleId)
        && doc.inProtectedRange(issue.start, issue.end)) {
        return false;
      }
      return true;
    });
  }

  /**
   * plan §25 IssueConflictResolver
   * - drop exact duplicates (same rule + same span)
   * - same span, different rule: keep highest priority (specific > generic)
   *   with a small specificity tiebreak inside equal priority
   * - different spans / semantics are NEVER merged (§25.3)
   */
  resolveConflicts(issues) {
    const bySpan = new Map(); // "start:end" -> issues[]
    for (const issue of issues) {
      const k = `${issue.start}:${issue.end}`;
      if (!bySpan.has(k)) bySpan.set(k, []);
      bySpan.get(k).push(issue);
    }
    const out = [];
    for (const group of bySpan.values()) {
      const dedup = new Map();
      for (const issue of group) {
        if (!dedup.has(issue.ruleId)) dedup.set(issue.ruleId, issue);
      }
      const list = [...dedup.values()];
      list.sort((a, b) =>
        priorityOf(b.ruleId) - priorityOf(a.ruleId)
        || SPECIFICITY_TIEBREAK(b.ruleId) - SPECIFICITY_TIEBREAK(a.ruleId));
      out.push(list[0]);
    }
    return out;
  }
}

/** inside an equal priority tier, prefer the more specific rule */
function SPECIFICITY_TIEBREAK(ruleId) {
  switch (ruleId) {
    case RuleIds.WHITESPACE_BEFORE_PUNCTUATION: return 3;
    case RuleIds.NON_BREAKING_SPACE: return 3;
    case RuleIds.ABBREVIATION_DETECTED: return 3;
    case RuleIds.POSSIBLE_MISSING_DIACRITIC: return 2;
    case RuleIds.POSSIBLE_SPELLING_ERROR: return 1;
    case RuleIds.POSSIBLE_WORD_BOUNDARY_ERROR: return 1;
    default: return 0;
  }
}

/** singleton for server/demo use */
export function createDefaultEngine(options) {
  return new SmsValidationEngine(options);
}

export { ValidationContext };
