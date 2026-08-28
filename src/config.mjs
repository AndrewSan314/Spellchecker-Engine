// ============================================================
// ValidationConfigService — plan §13 / §38
// Immutable snapshot built once at startup (and reloadable);
// RULES MUST NEVER QUERY DB/FILES PER TOKEN.
// ============================================================
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');

const DEFAULT_SNAPSHOT = Object.freeze({
  enabled: true,
  rules: {
    deterministicEnabled: true,
    missingDiacriticEnabled: true,   // demo enables; production starts OFF (§38)
    spellingEnabled: true,
    zeroWidthSeverity: 'WARNING',
    invalidCharacterSeverity: 'ERROR',
  },
  characters: {
    // ALLOW | INFO | ERROR — default ALLOW because business undecided (§11.10)
    emojiPolicy: 'ALLOW',
  },
  spelling: {
    attentionMode: 'OFF',                 // OFF | SHADOW | EXPERIMENTAL_ACTIVE
  },
  linguistic: {
    mode: 'ACTIVE',                       // OFF | SHADOW | ACTIVE (§39)
    attentionMode: 'OFF',                 // OFF | SHADOW | EXPERIMENTAL_ACTIVE
    maxAccentCandidatesPerToken: 12,      // §16.3
    beamWidth: 5,                         // §18.2
    minTokenLength: 2,                    // §16.4
    missingDiacriticMinConfidence: 0.7,   // §19 tuned on POC bench (seed was 0.85)
    missingDiacriticMinMargin: 0.20,
    // Phase 1 (calibration plan) — temperature for the unified confidence
    // scale sigmoid((s1-s2)/T). Both linguistic rules share the semantics;
    // T=1 reproduces the legacy pre-calibration numbers exactly, so raising
    // T is the only lever that changes emitted confidence. Fitted offline by
    // tools/fit_confidence_temperature.mjs against VSEC dev.
    missingDiacriticConfidenceTemperature: 11.7759,
    spellingConfidenceTemperature: 18.9688,
    spellingMinConfidence: 0.70,
    spellingMinMargin: 0.15,
    spellingMinFrequency: 1000,
    spellingMinTokenLength: 2, // runtime verifier below keeps short-token emissions proof-gated
    maxSpellingSuggestions: 3,            // §22
    originalPriorBonusMissingAccent: 0.8, // §18.3 preserve-original bias (tuned on POC bench)
    originalPriorBonusSpelling: 0.8,      // token already failed dictionary gate
    // plain-form gate: a bare token that is itself a high-frequency dictionary
    // word ("ngay", "hoa") needs direct evidence before being flipped:
    // trigram collocation > bigram joint > strong conf+margin escape hatch.
    plainFormGateMinFrequency: 5000,
    plainFormStrongConfidence: 0.95,      // T3 escape hatch (thin context)
    plainFormStrongMargin: 0.90,
    // cheap-ranker lexical frequency term (plan Task 5 Step 4): log10-scaled
    // lexicon frequency added to PMD local candidate scores so nonsense twins
    // cannot outrank the correct word merely through lattice dilution
    lexicalFreqWeight: 0.6,
    // Real-word wrong-tone correction (VIWIKI-SP class: "quyết đinh",
    // "ảnh hưỏng", "cơ sơ"). Requires DECISIVE trigram evidence — an
    // attested collocation for the candidate AND zero for the original in
    // the same context window — otherwise stays silent (precision-first).
    // OFF by default: with bigram-era signal it produced real-word FPs
    // ("bảo hành chính hãng" -> "hàng"); re-enable only after Task 9/10
    // calibration proves a precision-safe win on VSEC dev.
    wrongToneEnabled: false,
    // Task 4 (recall-improvement plan) — explicit mode for the accented
    // same-key WRONG-DIACRITIC spelling lane. Supersedes the ambiguous
    // boolean above when present:
    //   OFF    (default) lane fully closed — legacy boolean governs nothing;
    //   SHADOW           compute + report wouldEmit, NEVER surface issues;
    //   ACTIVE           lane may emit (post-calibration only, Task 9).
    // Legacy mapping: absent key + wrongToneEnabled=true behaves ACTIVE.
    // ACTIVE since the wrong-diacritic breakthrough: ACCENTED_SAME_KEY is
    // 53% of real dev errors and this lane lifts its recall 0.145 -> 0.500.
    wrongDiacriticMode: 'ACTIVE',
    // Symmetric counterpart of wrongDiacriticMode for UNACCENTED_VALID
    // tokens ("quang" -> "quảng"). Same same-key lane, same trigram/bigram
    // proof; separate switch so the two can be measured independently.
    unaccentedRealWordMode: 'OFF',
    // Task 5 (recall-improvement plan) — DIFFERENT_KEY_REAL_WORD lane for
    // dictionary-valid real-word typos ("đế" -> "đến", sentence-initial
    // "Các" -> "Cách"). Tighter eligibility than the wrong-diacritic lane:
    // corpus-backed candidate + DIRECT bigram/trigram evidence + positive
    // pairwise margin. ACTIVE is reserved for the Task 9 calibration freeze
    // and currently behaves exactly like SHADOW.
    realWordTypoMode: 'ACTIVE',
    // Task 9 (recall-improvement plan) — calibrated REAL-WORD lane gates.
    // The lane emits ONLY when every provisional hard check passes AND the
    // trained pairwise probability clears the frozen threshold. Defaults are
    // the dev-frozen accepted values (see config/spelling-tuning.json);
    // realWordTypoMode governs whether the lane can emit at all.
    // 0.97 (was 0.95): with the wrong-diacritic lane ACTIVE the two lanes
    // compose, so this one is tightened to hold overall precision >= 0.72.
    realWordTypoMinProbability: 0.97,
    realWordTypoMinCandidateWindows: 1,
    realWordTypoMaxOriginalWindows: 3,
    realWordTypoMinMargin: 0.0,
    // Phase 1: this lane reports a TRAINED pairwise probability, so its
    // unified-scale margin is that probability's log-odds. Kept separate
    // from spellingConfidenceTemperature because the two axes have different
    // natural units (trained logits vs raw LM score gaps); falls back to the
    // spelling temperature when null.
    realWordTypoConfidenceTemperature: 6.0299,
    // Wrong-diacritic lane calibration keys (lane stays SHADOW pre-freeze;
    // recorded so Task 9 trials and the frozen file stay schema-complete).
    wrongDiacriticMinProbability: null,
    wrongDiacriticMinCandidateWindows: 1,
    wrongDiacriticMaxOriginalWindows: 0,
    wrongDiacriticMinMargin: 0.0,
    // Task 8 (recall-improvement plan) — bounded word-boundary lane
    // (split "cảmơn" -> "cảm ơn"; merge "như ng" -> "nhưng"). SHADOW
    // computes decisions without emitting; ACTIVE emits post-calibration.
    wordBoundaryCorrectionMode: 'SHADOW',
    // both/all parts of a boundary repair must be corpus-backed above this
    wordBoundaryMinFrequency: 2000,
    typoAlpha: 1.0, typoBeta: 0.3, typoGamma: 1.2, // §22 seed weights
    // Task 5 cascade sizing: wide cheap generation, narrow expensive rerank.
    spellingCandidateKeys: 20,     // stripped keys per lookup (was 6; dev
                                   // oracle shows correct keys cluster at
                                   // rank 14-29 among equal-distance rivals)
    spellingSurfacesPerKey: 4,     // retained surfaces per key (raw kept)
    spellingPoolMax: 80,           // hard bound on the cheap pool
    spellingCheapTopK: 3,          // candidates entering trigram reranking
    // Task 7 (recall-improvement plan): diversity-preserving shortlist size
    // — union of {best-distance, best-context-attested, best-same-key, cheap
    // fill}, capped so the expensive stage never sees more than 4 rivals.
    spellingShortlistSize: 4,
    // cheap-ranker weights (edit dist / log-freq / family / attestation)
    cheapDistWeight: 1.0,
    cheapFreqWeight: 0.5,
    cheapFamilyBonus: 0.75,
    cheapAttestBonus: 0.25,
    // Task 3/9 — candidate-scoring weights (see src/language.mjs SCORE_WEIGHTS)
    // and joint-smoothing pseudo-counts; tuned on dev only (Task 9).
    scoreWeights: { A: 0.2, B: 1.0, C: 1.0, D: 1.5, E: 1.0, F: 1.0 },
    rightJointSmoothing: 20,
    forwardJointSmoothing: 10,
  },
});

export class ValidationConfigSnapshot {
  constructor(data) {
    this.data = deepFreeze(structuredClone(data));
    Object.freeze(this);
  }
  get(path_) {
    return path_.split('.').reduce((o, k) => o?.[k], this.data);
  }
}

function deepFreeze(obj) {
  if (obj && typeof obj === 'object') {
    for (const v of Object.values(obj)) deepFreeze(v);
    Object.freeze(obj);
  }
  return obj;
}

export class ValidationConfigService {
  constructor(snapshot = new ValidationConfigSnapshot(DEFAULT_SNAPSHOT)) {
    this._snapshot = snapshot;
  }
  snapshot() {
    return this._snapshot;
  }
  /** plan §13: reload without restart (admin endpoint / scheduled job) */
  reload(overrides = {}) {
    const merged = mergeDeep(DEFAULT_SNAPSHOT, overrides);
    this._snapshot = new ValidationConfigSnapshot(merged);
    return this._snapshot;
  }
}

function mergeDeep(base, patch) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(patch ?? {})) {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = mergeDeep(out[k] ?? {}, v);
    } else if (v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}
