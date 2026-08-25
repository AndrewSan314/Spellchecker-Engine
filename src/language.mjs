// ============================================================
// Language layer — plan §17 / §18 / §20 / §21 / §22
//   NGramLanguageModel   local bigram LM with interpolation backoff
//   BeamSearchDecoder    candidate-lattice decoding (no cartesian product)
//   TypoCandidateProvider  SymSpell-style deletion index
//   VietnameseEditScorer weighted edit distance + telex typing hints
// Immutable after load; safe to share across requests.
// ============================================================
import { readFileSync, accessSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { accentKey } from './normalizer.mjs';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');

/**
 * Task 2: raised instead of silently degrading when an LM artifact exists but
 * cannot be parsed. Carries artifactPath / line / reason for diagnostics.
 */
export class LanguageModelArtifactError extends Error {
  constructor(message, { path: artifactPath, line, reason } = {}) {
    super(message);
    this.name = 'LanguageModelArtifactError';
    this.artifactPath = artifactPath ?? null;
    this.line = line ?? null;
    this.reason = reason ?? null;
  }
}

const KIND_RANK = Object.freeze({ U: 1, B: 2, T: 3 });

// plan §17.6 seed weights (POC only — tuned via config, Task 9 calibrates):
//   A unigram prior · B left bigram P(w|prev) · C right bigram JOINT
//   D centered chain log P(w|prev)+log P(next|prev,w) · E left trigram
//   F forward trigram joint
export const SCORE_WEIGHTS = Object.freeze({
  A: 0.2, B: 1.0, C: 1.0, D: 1.5, E: 1.0, F: 1.0,
});
/** pseudo-count smoothing masses for joint-style (candidate-independent) terms */
const JOINT_SMOOTHING_DEFAULTS = Object.freeze({ pair: 20, triple: 10 });
// plan §18.3 — prior keeping the ORIGINAL token alive in every lattice
export const ORIGINAL_PRIOR_BONUS = 2.2;

function log(x) { return Math.log(Math.max(x, Number.EPSILON)); }

/**
 * Bigram LM trained from a plain-text corpus (one sentence per line).
 * P(w)      = (c+0.5)/(N+0.5V)
 * P(w|prev) = (c(prev,w)+ALPHA*P(w)) / (c(prev)+ALPHA)   when prev known
 *           = P(w)                                       otherwise
 */
export class NGramLanguageModel {
  constructor(unigram, bigram, totalTokens, vocabSize, trigram = new Map()) {
    this.unigram = unigram;     // Map word -> count (frozen)
    this.bigram = bigram;       // Map "w1 w2" -> count (frozen)
    // RAW-surface trigram counts ("w1 w2 w3", lowercase, NO resolver applied).
    // Used for collocation evidence (e.g. "mua ngay tại") where the exact
    // surface form is the point — resolved lookups would erase it.
    this.trigram = trigram;
    this.totalTokens = totalTokens;
    this.vocabSize = vocabSize;
    // Optional resolver: map an out-of-vocab surface (e.g. unaccented "lam")
    // to its most plausible vocab form ("làm") BEFORE bigram lookup, so that
    // known-vs-unknown neighbours are scored consistently.
    this._resolver = null;
    this._scoreCfg = {
      weights: SCORE_WEIGHTS,
      rightJointSmoothing: JOINT_SMOOTHING_DEFAULTS.pair,
      forwardJointSmoothing: JOINT_SMOOTHING_DEFAULTS.triple,
    };
  }

  /**
   * Task 3/9: scoring weights and joint-smoothing masses come from the config
   * snapshot (engine wires this at construction); defaults keep the LM usable
   * standalone.
   */
  setScoreConfig(cfg) {
    this._scoreCfg = {
      weights: Object.freeze({ ...SCORE_WEIGHTS, ...(cfg?.weights ?? {}) }),
      rightJointSmoothing: cfg?.rightJointSmoothing
        ?? JOINT_SMOOTHING_DEFAULTS.pair,
      forwardJointSmoothing: cfg?.forwardJointSmoothing
        ?? JOINT_SMOOTHING_DEFAULTS.triple,
    };
  }

  /** engine wires this after the accent index is built */
  setResolver(fn) {
    this._resolver = fn;
  }

  /** engine wires this so resolveInContext can enumerate accent surfaces */
  setSurfaceProvider(fn) {
    this._surfaceProvider = fn;
  }

  resolve(w) {
    return this._resolver ? this._resolver(w) : w;
  }

  /**
   * Context-aware surface pick. EVERY word with accent twins is re-judged
   * by bigram compatibility with its RAW neighbours — including vocab-valid
   * raw forms, because corpus frequency says nothing about which twin this
   * particular sentence needs ("y" the letter vs "ý" in "Lưu ý...").
   * Attested bigrams decide; frequency order is only the fallback.
   */
  resolveInContext(w, prevRaw, nextRaw) {
    if (typeof w !== 'string') return w;
    const surfs = this._surfaceProvider
      ? this._surfaceProvider(accentKey(w))
      : [];
    if (!surfs || surfs.length === 0) return w;
    if (prevRaw == null && nextRaw == null) {
      // no context to judge with: keep a vocab-valid raw form, else top twin
      return this.knows(w) ? w : surfs[0].word.toLowerCase();
    }
    let best = surfs[0].word.toLowerCase();
    let bestScore = -Infinity;
    for (const { word } of surfs) {
      const s = word.toLowerCase();
      let sc = 0;
      if (prevRaw != null && this.unigram.has(this.resolve(prevRaw))) {
        sc += Math.log(Math.max(this._pBiRaw(s, this.resolve(prevRaw)), 1e-12));
      }
      if (nextRaw != null && this.unigram.has(this.resolve(nextRaw))) {
        sc += Math.log(Math.max(this._pBiRaw(this.resolve(nextRaw), s), 1e-12));
      }
      if (sc > bestScore) { bestScore = sc; best = s; }
    }
    return best;
  }

  /**
   * Load path (Task 2 — fail fast, observable):
   *   1. lm-ngrams.tsv  (built by tools/build_lm.py) when present — MUST parse;
   *      any malformed/unreadable artifact throws LanguageModelArtifactError.
   *   2. legacy raw-text corpus fallback ONLY for a missing artifact
   *      (ENOENT) and only when allowFallback is not explicitly false.
   *
   * Options:
   *   prebuiltPath   override artifact location (dependency injection/tests)
   *   allowFallback  default true; false makes missing artifact a hard error
   *   hashArtifact   default true; computes sha256 into loadDiagnostics
   */
  static load(file = path.join(DATA_DIR, 'corpus-train.txt'), opts = {}) {
    const prebuiltPath = opts.prebuiltPath ?? path.join(DATA_DIR, 'lm-ngrams.tsv');
    const allowFallback = opts.allowFallback !== false;
    const t0 = Date.now();
    if (!file.endsWith('corpus-train.txt')) {
      const model = NGramLanguageModel._loadRawText(file);
      model.loadDiagnostics = { backend: 'raw-corpus', artifactPath: file };
      return model;
    }
    let st;
    try {
      st = statSync(prebuiltPath);
    } catch (err) {
      if (allowFallback && err?.code === 'ENOENT') {
        const model = NGramLanguageModel._loadRawText(file);
        model.loadDiagnostics = {
          backend: 'raw-corpus-fallback',
          artifactPath: prebuiltPath,
          reason: 'ENOENT',
        };
        return model;
      }
      throw new LanguageModelArtifactError(
        `LM artifact inaccessible: ${prebuiltPath} (${err?.code ?? String(err)})`,
        { path: prebuiltPath, reason: err?.code ?? String(err) },
      );
    }
    if (!st.isFile()) {
      throw new LanguageModelArtifactError(
        `LM artifact is not a regular file: ${prebuiltPath}`,
        { path: prebuiltPath, reason: 'EISDIR' },
      );
    }
    // Artifact exists: it MUST parse. Never fall back on parse failure.
    const model = NGramLanguageModel._loadPrebuiltValidated(prebuiltPath, {
      hashArtifact: opts.hashArtifact !== false,
    });
    model.loadDiagnostics = {
      ...model.loadDiagnostics,
      backend: 'tsv',
      artifactPath: prebuiltPath,
      bytes: st.size,
      loadMs: Date.now() - t0,
    };
    return model;
  }

  static _loadRawText(rawPath) {
    const raw = readFileSync(rawPath, 'utf8');
    const unigram = new Map();
    const bigram = new Map();
    const trigram = new Map();
    let total = 0;
    for (const lineRaw of raw.split(/\r?\n/)) {
      const line = lineRaw.trim();
      if (!line || line.startsWith('#')) continue;
      // §17.3 corpus cleanup: drop mixed letter/digit clusters (promo codes
      // like "GIAM50K", "DH123") BEFORE counting — their letter runs would
      // otherwise pollute the model with phantom words ("mã giam").
      const cleaned = line.replace(
        /[0-9\p{L}][0-9\p{L}.-]*[0-9\p{L}]/gu,
        (m) => (/\d/.test(m) ? ' ' : m),
      );
      // tokenize for counting: words only, lowercase lookup form
      const words = cleaned.normalize('NFC').toLowerCase().match(/[\p{L}\p{M}]+/gu) ?? [];
      let prev = null;
      let prev2 = null;
      for (const w of words) {
        unigram.set(w, (unigram.get(w) ?? 0) + 1);
        total++;
        if (prev !== null) {
          const k = `${prev} ${w}`;
          bigram.set(k, (bigram.get(k) ?? 0) + 1);
        }
        if (prev2 !== null) {
          const k3 = `${prev2} ${prev} ${w}`;
          trigram.set(k3, (trigram.get(k3) ?? 0) + 1);
        }
        prev2 = prev;
        prev = w;
      }
    }
    return new NGramLanguageModel(unigram, bigram, total, unigram.size, trigram);
  }

  /** pre-counted artifact from tools/build_lm.py (see file header) */
  static _loadPrebuilt(artPath) {
    return NGramLanguageModel._loadPrebuiltValidated(artPath, { hashArtifact: false });
  }

  /**
   * Task 2: strict validated loader. Validates header, kind, tab structure,
   * positive-integer counts, section ordering and unique keys (detected by
   * comparing per-section record lines against final map sizes — O(1) memory,
   * equivalent to exact duplicate detection). Any violation raises
   * LanguageModelArtifactError with artifact path + line + reason.
   */
  static _loadPrebuiltValidated(artPath, { hashArtifact = true } = {}) {
    const raw = readFileSync(artPath, 'utf8');
    const sha256 = hashArtifact ? createHash('sha256').update(raw).digest('hex') : null;
    const unigram = new Map();
    const bigram = new Map();
    const trigram = new Map();
    let totalFromHeader = null;
    let lastKindRank = 0;
    const sectionLines = { U: 0, B: 0, T: 0 };
    const sectionLastLine = { U: 0, B: 0, T: 0 };
    const fail = (lineNo, reason) => {
      throw new LanguageModelArtifactError(
        `Invalid LM artifact ${artPath} at line ${lineNo}: ${reason}`,
        { path: artPath, line: lineNo, reason },
      );
    };
    const lines = raw.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line) continue;
      const lineNo = i + 1;
      const kind = line[0];
      if (kind === '#') {
        const m = /^#tokens=(\d+)$/.exec(line);
        // free-form comment lines (e.g. "#SMS-LM v1") are allowed; a
        // #tokens=<int> line must appear before the first record
        if (m) {
          if (totalFromHeader !== null) fail(lineNo, 'duplicate #tokens header');
          totalFromHeader = Number(m[1]);
        }
        continue;
      }
      if (totalFromHeader === null) fail(lineNo, 'missing #tokens=<int> header before first record');
      const rank = KIND_RANK[kind];
      if (!rank) fail(lineNo, `invalid record kind "${kind}"`);
      if (rank < lastKindRank) {
        fail(lineNo, `section order violation: "${kind}" appears after higher-order sections`);
      }
      const i1 = line.indexOf('\t');
      if (i1 < 0 || i1 !== 1) fail(lineNo, 'expected "<kind>\\t<key>\\t<count>"');
      const i2 = line.indexOf('\t', i1 + 1);
      if (i2 < 0) fail(lineNo, 'missing count column');
      if (line.indexOf('\t', i2 + 1) !== -1) fail(lineNo, 'extra tab separator');
      const key = line.slice(i1 + 1, i2);
      if (!key) fail(lineNo, 'empty key');
      const count = Number(line.slice(i2 + 1));
      if (!Number.isInteger(count) || count <= 0) fail(lineNo, 'count must be a positive integer');
      if (kind === 'U') unigram.set(key, count);
      else if (kind === 'B') bigram.set(key, count);
      else trigram.set(key, count);
      sectionLines[kind] += 1;
      sectionLastLine[kind] = lineNo;
      lastKindRank = rank;
    }
    if (totalFromHeader === null) fail(lines.length + 1, 'missing #tokens=<int> header');
    for (const kind of ['U', 'B', 'T']) {
      const mapSize = kind === 'U' ? unigram.size : kind === 'B' ? bigram.size : trigram.size;
      if (sectionLines[kind] !== mapSize) {
        throw new LanguageModelArtifactError(
          `Invalid LM artifact ${artPath}: duplicate key in ${kind} section `
          + `(${sectionLines[kind]} records but ${mapSize} distinct keys)`,
          { path: artPath, line: sectionLastLine[kind], reason: `duplicate key in ${kind} section` },
        );
      }
    }
    // totalTokens is the WEIGHTED token count declared by the #tokens header
    // (the corpus total used by pUni) — never the number of n-gram rows.
    const model = new NGramLanguageModel(
      unigram, bigram, totalFromHeader, unigram.size, trigram);
    model.loadDiagnostics = {
      counts: { U: unigram.size, B: bigram.size, T: trigram.size },
      headerTokens: totalFromHeader,
      ...(sha256 ? { sha256 } : {}),
    };
    return model;
  }

  pUni(w) {
    const c = this.unigram.get(w) ?? 0;
    return (c + 0.5) / (this.totalTokens + 0.5 * this.vocabSize);
  }

  /** is the word present in the trained vocabulary? (context-anchor check) */
  knows(w) {
    return this.unigram.has(w);
  }

  /** raw bigram joint count with both sides resolved through the resolver */
  bigramJoint(w1, w2) {
    return this.bigram.get(`${this.resolve(w1)} ${this.resolve(w2)}`) ?? 0;
  }

  /**
   * RAW-surface trigram joint count (no resolver): exact collocation
   * evidence like "mua ngay tại". Inputs are lowercased only.
   */
  trigramJoint(w1, w2, w3) {
    const k = `${String(w1).toLowerCase()} ${String(w2).toLowerCase()} `
      + `${String(w3).toLowerCase()}`;
    return this.trigram.get(k) ?? 0;
  }

  _pBiRaw(w, prev) {
    if (prev == null || !this.unigram.has(prev)) return this.pUni(w);
    const joint = this.bigram.get(`${prev} ${w}`) ?? 0;
    const prevCount = this.unigram.get(prev);
    const ALPHA = 2.0; // interpolation mass toward unigram
    return (joint + ALPHA * this.pUni(w)) / (prevCount + ALPHA);
  }

  pBi(w, prev) {
    return this._pBiRaw(w, prev != null ? this.resolve(prev) : null);
  }

  _pTriRaw(w, prev1, prev2) {
    if (prev1 == null && prev2 == null) return this.pUni(w);
    if (prev2 == null) return this._pBiRaw(w, prev1);
    const tri = this.trigram.get(`${prev2} ${prev1} ${w}`) ?? 0;
    const biDenom = this.bigram.get(`${prev2} ${prev1}`) ?? 0;
    const BETA = 1.0; // interpolation mass toward bigram
    return (tri + BETA * this._pBiRaw(w, prev1)) / (biDenom + BETA);
  }

  pTri(w, prev1, prev2) {
    const r1 = prev1 != null ? this.resolve(prev1) : null;
    const r2 = prev2 != null ? this.resolve(prev2) : null;
    return this._pTriRaw(w, r1, r2);
  }

  /** plan §17.6 bidirectional local score for candidate `w` between prev/next */
  scoreCandidate(w, prev, next) {
    const { A, B, C } = SCORE_WEIGHTS;
    const resolvedPrev = prev != null ? this.resolve(prev) : null;
    const resolvedNext = next != null ? this.resolve(next) : null;
    return this._scoreWith(w, resolvedPrev, resolvedNext);
  }

  /**
   * Same scoring but the caller supplies ALREADY context-resolved neighbours
   * (see resolveInContext). Prevents double-resolution from mangling the
   * bigram evidence during candidate re-ranking.
   */
  scoreCandidateResolved(w, prevResolved, nextResolved) {
    return this._scoreWith(w, prevResolved, nextResolved);
  }

  /**
   * Smoothed co-occurrence probability of an adjacent pair. The normalizer
   * (totalTokens + S) never depends on the candidate being ranked, so rare
   * candidates cannot win through tiny-denominator conditional backoff —
   * the Task 3 fix for the "đật beats đặt" inversion.
   */
  jointPairProb(a, b) {
    const joint = this.bigram.get(`${a} ${b}`) ?? 0;
    const S = this._scoreCfg.rightJointSmoothing;
    return (joint + S * this.pUni(a) * this.pUni(b))
      / (this.totalTokens + S);
  }

  /** smoothed co-occurrence probability of an adjacent triple */
  jointTripleProb(a, b, c) {
    const joint = this.trigram.get(`${a} ${b} ${c}`) ?? 0;
    const S = this._scoreCfg.forwardJointSmoothing;
    return (joint + S * this.pUni(a) * this.pUni(b) * this.pUni(c))
      / (this.totalTokens + S);
  }

  /**
   * Centered trigram evidence with the VALID factorization (plan Task 3):
   *   log P(candidate | prev) + log P(next | prev, candidate)
   * reusing _pBiRaw/_pTriRaw. The old form divided count(prev,cand,next) by
   * count(prev,next) — a non-contiguous neighbour bigram that is not the
   * conditioning context. Beam-first short-circuit: when the beam's chosen
   * (p,n) pair attests the collocation, sibling surfaces are not scanned.
   * Returns null when no vocab-known (prev,next) pair exists.
   */
  centeredEvidence(w, prevSurfaces, nextSurfaces) {
    const PL = prevSurfaces ?? [];
    const NL = nextSurfaces ?? [];
    let best = null;
    for (let pi = 0; pi < PL.length; pi++) {
      const p = PL[pi];
      if (!this.unigram.has(p)) continue;
      for (let ni = 0; ni < NL.length; ni++) {
        const n = NL[ni];
        if (!this.unigram.has(n)) continue;
        const v = log(this._pBiRaw(w, p)) + log(this._pTriRaw(n, w, p));
        if (best === null || v > best) best = v;
        if (pi === 0 && ni === 0
          && (this.trigram.get(`${p} ${w} ${n}`) ?? 0) > 0) {
          return v; // beam-first attested collocation decides
        }
      }
    }
    return best;
  }

  /**
   * Left trigram evidence log P(w | p2, p1) — conditions only on fixed left
   * context. Beam-first short-circuit: when the beam's chosen (p2,p1) pair
   * attests the trigram, sibling surfaces are not probed.
   */
  leftTrigramEvidence(w, prev2Surfaces, prevSurfaces) {
    const P2 = prev2Surfaces ?? [];
    const P1 = prevSurfaces ?? [];
    let best = null;
    for (let i = 0; i < P2.length; i++) {
      const p2 = P2[i];
      if (!this.unigram.has(p2)) continue;
      for (let j = 0; j < P1.length; j++) {
        const p1 = P1[j];
        if (!this.unigram.has(p1)) continue;
        const v = log(this._pTriRaw(w, p1, p2));
        if (best === null || v > best) best = v;
        if (i === 0 && j === 0
          && (this.trigram.get(`${p2} ${p1} ${w}`) ?? 0) > 0) {
          return v;
        }
      }
    }
    return best;
  }

  /** smoothed forward triple joint log jointProb(w, n1, n2), beam-first */
  forwardTripleJoint(w, nextSurfaces, next2Surfaces) {
    const N1 = nextSurfaces ?? [];
    const N2 = next2Surfaces ?? [];
    let best = null;
    for (let i = 0; i < N1.length; i++) {
      const n1 = N1[i];
      if (!this.unigram.has(n1)) continue;
      for (let j = 0; j < N2.length; j++) {
        const n2 = N2[j];
        if (!this.unigram.has(n2)) continue;
        const v = log(this.jointTripleProb(w, n1, n2));
        if (best === null || v > best) best = v;
        if (i === 0 && j === 0
          && (this.trigram.get(`${w} ${n1} ${n2}`) ?? 0) > 0) {
          return v;
        }
      }
    }
    return best;
  }

  /**
   * Candidate scoring over neighbour SURFACE sets (Task 3 formulation).
   * Every term either conditions ONLY on fixed observed context or is a
   * joint-style estimate whose normalizer is candidate-independent:
   *   A·log P(w) + B·log P(w|prev-anchored) + C·log jointPair(w,next)
   * + D·centeredChain(prev,next) + E·log P(w|p2,p1) + F·log jointTriple(w,n1,n2)
   * Surface lists are ordered beam-choice-first; left bigram keeps the
   * beam-first attestation semantics (_anchoredBi); every surface-pair loop
   * short-circuits on an attested beam-first pair (Task 6 probe reduction).
   */
  scoreCandidateOverSurfaces(w, prevSurfaces, nextSurfaces, prev2Surfaces = [], next2Surfaces = []) {
    const W = this._scoreCfg.weights;
    let s = W.A * log(this.pUni(w));
    const bL = this._anchoredBi(w, prevSurfaces, 'left');
    if (bL !== null) s += W.B * bL;

    const PL = prevSurfaces ?? [];
    const NL = nextSurfaces ?? [];
    if (NL.length) {
      // RIGHT side: smoothed pair JOINT (candidate-independent normalizer).
      let bestR = null;
      for (const n of NL) {
        const v = log(this.jointPairProb(w, n));
        if (bestR === null || v > bestR) bestR = v;
      }
      s += W.C * bestR;
    }

    if (PL.length && NL.length) {
      const centered = this.centeredEvidence(w, PL, NL);
      if (centered !== null) s += W.D * centered;
    }

    if (prev2Surfaces?.length && PL.length) {
      const lt = this.leftTrigramEvidence(w, prev2Surfaces, PL);
      if (lt !== null) s += W.E * lt;
    }

    if (NL.length && next2Surfaces?.length) {
      const ft = this.forwardTripleJoint(w, NL, next2Surfaces);
      if (ft !== null) s += W.F * ft;
    }

    return s;
  }

  /**
   * log P(candidate | left neighbour surface), beam-first attestation
   * semantics: an ATTESTED pair with the beam's decision returns immediately;
   * otherwise the best backoff across surfaces. Only the LEFT direction is a
   * valid candidate ranking term — the right side uses jointPairProb.
   */
  _anchoredBi(w, surfs, side) {
    void side;
    if (!surfs || surfs.length === 0) return null;
    let bestFallback = null;
    for (const sv of surfs) {
      const joint = this.bigram.get(`${sv} ${w}`) ?? 0;
      const val = log(this._pBiRaw(w, sv));
      if (joint > 0) return val;            // attested pair decides
      if (bestFallback === null || val > bestFallback) bestFallback = val;
    }
    return bestFallback;
  }

  /** best raw joint count between candidate and any surface of a neighbour */
  bestJointOverSurfaces(candidateLower, surfaces, side /* 'left'|'right' */) {
    let best = 0;
    for (const sv of surfaces ?? []) {
      const k = side === 'left'
        ? `${sv} ${candidateLower}`
        : `${candidateLower} ${sv}`;
      const v = this.bigram.get(k) ?? 0;
      if (v > best) best = v;
    }
    return best;
  }

  _scoreWith(w, resolvedPrev, resolvedNext) {
    const W = this._scoreCfg.weights;
    let s = W.A * log(this.pUni(w));
    s += W.B * log(resolvedPrev != null ? this._pBiRaw(w, resolvedPrev) : 1);
    // right side: joint-style pair probability (candidate-independent
    // normalizer), matching scoreCandidateOverSurfaces
    s += W.C * (resolvedNext != null
      ? Math.log(Math.max(this.jointPairProb(w, resolvedNext), Number.EPSILON))
      : 0);
    return s;
  }
}

/**
 * BeamSearchDecoder over the token/candidate lattice with 2nd-order Markov (Trigram) decoding.
 * Every position keeps its original token as an always-present candidate.
 */
export class BeamSearchDecoder {
  constructor(languageModel, beamWidth = 5, originalPriorBonus = 2.2) {
    this.lm = languageModel;
    this.beamWidth = beamWidth;
    this.originalPriorBonus = originalPriorBonus;
  }

  /**
   * @param positions Array<{tokenIndex:number, candidates:Array<{word:string,isOriginal:boolean,freq:number}>}>
   *   one entry per WORD position that participates in decoding.
   * @param {{beamWidth?:number, originalPrior?:number}} [opts] per-decode
   *   overrides so config reloads apply WITHOUT rebuilding the engine.
   * @returns top paths sorted desc score: [{score, words:string[]}]
   */
  decode(positions, opts = {}) {
    const width = opts.beamWidth ?? this.beamWidth;
    const prior = opts.originalPrior ?? this.originalPriorBonus;
    if (positions.length === 0) return [];
    // 2nd-order Markov state: {prev2, prev1, score, words}
    let beam = [{ prev2: null, prev1: null, score: 0, words: [] }];
    for (const pos of positions) {
      /** @type {typeof beam} */
      const next = [];
      for (const st of beam) {
        for (const cand of pos.candidates) {
          let s = st.score + log(this.lm.pTri(cand.word, st.prev1, st.prev2));
          // small frequency prior so ties prefer common words
          s += 0.05 * Math.log10((cand.freq ?? 1) + 1);
          if (cand.isOriginal && pos.candidates.length > 1) {
            s += prior; // preserve-original prior bonus
          }
          next.push({
            prev2: st.prev1,
            prev1: cand.word,
            score: s,
            words: [...st.words, cand.word]
          });
        }
      }
      next.sort((a, b) => b.score - a.score);
      beam = next.slice(0, width);
    }
    return beam.map(({ score, words }) => ({ score, words }));
  }
}

/** softmax over scores with temperature */
export function softmax(scores, temperature = 1) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - max) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * TypoCandidateProvider — SymSpell-style deletion index — plan §20/§21.
 * Keys are accent-stripped lowercase forms; lookup generates deletes of the
 * input up to the requested depth and unions matches.
 */
export class SymSpellCandidateProvider {
  /**
   * @param lexicon LexiconService
   * @param maxIndexDepth CAP on delete depth indexed
   * @param opts.surfacesPerKey retained surfaces per stripped key
   * @param opts.lengthAware Task 7: index each key with delete depth
   *        maxEditDistanceFor(key.length) (capped by maxIndexDepth) — lookups
   *        never request a depth beyond the same policy for the INPUT, and a
   *        distance-≤k match always shares a variant at depth ≤k on both
   *        sides, so results are unchanged while short words stop carrying
   *        depth-2 variants they can never be matched with.
   */
  constructor(lexicon, maxIndexDepth = 2,
    { surfacesPerKey = 4, lengthAware = true } = {}) {
    this.lexicon = lexicon;
    this.surfacesPerKey = surfacesPerKey;
    this.keyFreq = new Map(); // strippedKey -> max frequency across surfaces
    this.deletes = new Map(); // deleteVariant -> Set<strippedWord>
    this._buildMs = null;
    // O(1) surface lookup (plan "Optimize SymSpell lookup"): precomputed
    // per-key surface lists — accented first, then descending frequency —
    // so neither candidates() nor display resolution ever scans the lexicon.
    this.surfacesByKey = new Map();
    const tBuild0 = Date.now();
    let edgeCount = 0;
    for (const { word, freq } of lexicon.allWords()) {
      const key = accentKey(word);
      const prev = this.keyFreq.get(key) ?? 0;
      if (freq > prev) this.keyFreq.set(key, freq);
      // Task 7: dynamic index depth — short words never need depth-2
      // variants because lookups cap depth at maxEditDistanceFor(input len).
      const depth = lengthAware
        ? Math.min(maxIndexDepth, maxEditDistanceFor(key.length))
        : maxIndexDepth;
      for (const variant of generateDeletes(key, depth)) {
        if (!this.deletes.has(variant)) this.deletes.set(variant, new Set());
        this.deletes.get(variant).add(key);
        edgeCount++;
      }
      if (!this.surfacesByKey.has(key)) this.surfacesByKey.set(key, []);
      this.surfacesByKey.get(key).push({ word, freq });
    }
    this._indexStats = {
      keys: this.surfacesByKey.size,
      variants: this.deletes.size,
      edges: edgeCount,
      buildMs: Date.now() - tBuild0,
    };
    for (const [key, list] of this.surfacesByKey) {
      // Task 5 retention policy: a real UNACCENTED surface must survive even
      // when three accented siblings are more frequent (dropping it cost
      // oracle coverage in VSEC dev). Selection forces it in; presentation
      // order stays accented-first/frequency for stable ranking.
      const accentedList = list.filter((s) => s.word.toLowerCase() !== key);
      const unaccentedList = list.filter((s) => s.word.toLowerCase() === key);
      accentedList.sort((a, b) => b.freq - a.freq || a.word.localeCompare(b.word));
      const kept = [
        ...accentedList.slice(0, Math.max(3, this.surfacesPerKey - 1)),
        ...unaccentedList,
      ];
      kept.sort((a, b) => (hasAccent(b.word) ? 1 : 0) - (hasAccent(a.word) ? 1 : 0)
        || b.freq - a.freq || a.word.localeCompare(b.word));
      this.surfacesByKey.set(key, Object.freeze(kept.slice(0, this.surfacesPerKey)));
    }
    Object.freeze(this);
  }

  /** Task 7: read-only index diagnostics for the profiler/ops dashboards */
  indexStats() {
    let maxBucket = 0;
    for (const keys of this.deletes.values()) {
      if (keys.size > maxBucket) maxBucket = keys.size;
    }
    return Object.freeze({
      ...this._indexStats,
      maxBucket,
    });
  }

  /** retained surfaces of one stripped key (Task 5 policy — see constructor) */
  surfacesFor(key) {
    return this.surfacesByKey.get(key) ?? [];
  }

  /**
   * Task 5: WIDE cheap candidate pool — generation split from ranking.
   * Up to `keyCap` stripped keys (default 12, was 6), each contributing its
   * retained surfaces (raw/unaccented kept — see retention policy), with
   * edit distance and frequency metadata. NO trigram/context scoring here;
   * the caller cheap-ranks before expensive context reranking.
   */
  generatePool(normalizedToken, maxEditDistance, opts = {}) {
    const keyCap = opts.keyCap ?? 12;
    const surfacesPerKey = opts.surfacesPerKey ?? this.surfacesPerKey;
    // Task 4: lookup keys = raw stripped key PLUS Telex-decoded stripped key
    // ("dawnf" -> "dan", "ddat" -> "dat"). The raw key keeps behaviour for
    // ordinary typos identical; union only widens recall.
    const rawKey = accentKey(normalizedToken);
    const telexKey = accentKey(telexLookupSurface(normalizedToken));
    const inputKeys = telexKey !== rawKey ? [rawKey, telexKey] : [rawKey];
    const seen = new Set();
    for (const inputKey of inputKeys) {
      for (const variant of generateDeletes(inputKey, maxEditDistance)) {
        for (const w of this.deletes.get(variant) ?? []) {
          // exclude ONLY the token's own stripped key (unchanged semantics):
          // for Telex inputs the decoded key legitimately matches the target
          // family ("ddat" -> "dat" -> đặt/đất) and must be returned.
          if (w !== rawKey) seen.add(w);
        }
      }
    }
    // rank by TRUE distance (best across lookup keys) then real frequency
    const matchedKeys = [...seen]
      .map((w) => ({
        key: w,
        dist: Math.min(...inputKeys.map((k) => damerauOsaDistance(k, w))),
      }))
      .filter(({ dist }) => dist <= maxEditDistance)
      .sort((a, b) => a.dist - b.dist
        || (this.keyFreq.get(b.key) ?? 0) - (this.keyFreq.get(a.key) ?? 0))
      .slice(0, keyCap);

    const entries = [];
    for (const { key, dist } of matchedKeys) {
      for (const s of this.surfacesFor(key).slice(0, surfacesPerKey)) {
        entries.push({ word: s.word, stripped: key, dist, freq: s.freq });
      }
    }
    return {
      inputKeys,
      matchedKeys: matchedKeys.map((m) => m.key),
      entries,
      poolBound: keyCap * surfacesPerKey,
    };
  }

  candidates(normalizedToken, maxEditDistance, limit = 12) {
    const pool = this.generatePool(normalizedToken, maxEditDistance);
    return pool.entries.slice(0, limit);
  }
}

function hasAccent(s) {
  for (const ch of s) {
    const d = ch.normalize('NFD');
    if (d.length > 1 && /\p{M}/u.test(d.slice(1))) return true;
    if (ch === 'đ' || ch === 'Đ') return true;
  }
  return false;
}

/** all strings reachable by deleting up to `depth` chars (includes identity) */
export function generateDeletes(word, depth) {
  const out = new Set([word]);
  let frontier = new Set([word]);
  for (let d = 0; d < depth; d++) {
    const nxt = new Set();
    for (const w of frontier) {
      for (let i = 0; i < w.length; i++) nxt.add(w.slice(0, i) + w.slice(i + 1));
    }
    for (const w of nxt) out.add(w);
    frontier = nxt;
  }
  return out;
}

/**
 * Weighted Damerau-Levenshtein (OSA) — plan §21.
 * Telex digraph hints are normalized on BOTH sides before scoring so that
 * e.g. "dd" vs "đ" cost ~0.3 instead of 2 edits.
 */
const TELEX = [
  ['dd', 'đ'], ['aw', 'ă'], ['aa', 'â'], ['ee', 'ê'],
  ['oo', 'ô'], ['ow', 'ơ'], ['uw', 'ư'], ['w', 'ư'],
];

export function applyTelexHints(s) {
  let out = s.toLowerCase();
  for (const [from, to] of TELEX) out = out.split(from).join(to);
  return out;
}

/**
 * Task 4: Telex-decoded BASE form used to derive lookup keys BEFORE
 * accentKey()/delete generation — the edit scorer already understands Telex,
 * but until now a Telex-typed input ("dawnf", "ddat") could never reach it
 * because the deletion index only contains accent-stripped vocab keys.
 * Digraphs decode through the same TELEX table; tone letters (s/f/r/x/j)
 * immediately after a vowel are dropped (they encode tone, which accentKey
 * strips anyway). Non-Telex words are unchanged. Original tokens are never
 * mutated — this is analysis data for lookup only.
 */
export function telexLookupSurface(input) {
  let out = String(input).toLowerCase().normalize('NFC');
  // vowel digraphs + dd (the TELEX table minus bare-w, which would corrupt
  // "www." / ".vn" style tokens)
  for (const [from, to] of TELEX.slice(0, -1)) {
    out = out.split(from).join(to);
  }
  // standalone "w" (not adjacent to another ASCII letter) still decodes to ư,
  // so "www" survives while lone-telex usage keeps working
  out = out.replace(/(^|[^a-z])w(?![a-z])/gu, '$1ư');
  // tone letters encode tone, which accentKey strips anyway:
  //  - immediately after the vowel cluster ("dafn", "hoir")
  //  - at syllable/token end ("danf", "dawnf")
  out = out.replace(/([aeiouyăâđêôơư])([sfrxj]+)/gu, '$1');
  out = out.replace(/[sfrxj]+$/u, '');
  return out;
}

export function damerauOsaDistance(a, b) {
  const m = a.length, n = b.length;
  const INF = m + n + 1;
  let prev2 = null;
  let prev = new Array(n + 1).fill(0).map((_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        v = Math.min(v, prev2[j - 2] + 0.8); // transposition slightly cheaper
      }
      cur.push(v);
    }
    prev2 = prev;
    prev = cur;
  }
  void INF;
  return prev[n];
}

/** dynamic edit distance policy — plan §20.2 */
export function maxEditDistanceFor(len) {
  if (len <= 7) return 1;
  return 2;
}
