// ============================================================
// Linguistic rules — plan §14 (Lexical Gate) / §16.4 / §19 / §22
//   POSSIBLE_MISSING_DIACRITIC  (accent candidate pipeline + beam + gates)
//   POSSIBLE_SPELLING_ERROR     (SymSpell-style candidates + weighted edit + gates)
//
// Precision-first: if not confident => NO issue (§2.3, §19, Case G).
// Never auto-correct; suggestions are display-only.
// ============================================================
import {
  RuleIds, Severity, ValidationIssue, MessageMode,
} from '../core.mjs';
import { accentKey, hasVietnameseAccent } from '../normalizer.mjs';
import {
  softmax, applyTelexHints, damerauOsaDistance, maxEditDistanceFor,
} from '../language.mjs';
import { ORIGINAL_PRIOR_BONUS } from '../language.mjs';
import { extractContextEvidence, evidenceScore } from '../context-evidence.mjs';
import {
  RECALL_RANK_SENTINEL, ranksForCandidates, buildRecallPairwiseFeatures,
} from '../recall-reranker.mjs';
import { evaluateWordBoundaryCandidates } from '../word-boundary-candidates.mjs';
import { unitsFromDocument, encodeContextUnits, encodeOptionSurface, normalizeForModel } from '../attention-tokenizer.mjs';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'data');
let cachedAttentionVocab = null;
function getAttentionVocab() {
  if (cachedAttentionVocab) return cachedAttentionVocab;
  try {
    const p = path.join(DATA_DIR, 'attention-vocab.json');
    if (existsSync(p)) {
      const d = JSON.parse(readFileSync(p, 'utf8'));
      cachedAttentionVocab = new Map(Object.entries(d.wordToId || {}));
      return cachedAttentionVocab;
    }
  } catch {}
  return new Map();
}

/**
 * Lexical Gate — plan §14 fixed order. Returns classification for one token.
 */
export function classifyToken(token, ctx, doc, services) {
  const { whitelistService, abbreviationService, lexicon, accentIndex } = services;
  // 1. fully inside protected range => skip linguistic checks
  if (doc.insideSingleProtectedRange(token.start, token.end)) return 'PROTECTED';
  // 2/3. exact/normalized whitelist or allowed brand/customer term
  if (whitelistService.isAllowed(token.original, ctx)) return 'WHITELISTED';
  if (whitelistService.isAllowed(token.normalized, ctx)) return 'WHITELISTED';
  // 4. product/transaction code-like pattern (all caps w/ digits etc.)
  if (/^[A-Z0-9-]+$/.test(token.original) && /\d/.test(token.original)) return 'CODE_LIKE';
  // 5. abbreviation rule owns this token
  if (abbreviationService.isRuleEnabledFor(ctx.brandname)
    && abbreviationService.isAbbreviation(token.original)) return 'ABBREVIATION';

  const stripped = accentKey(token.normalized);
  const unaccentedForm = stripped === token.normalized;

  // 6. valid dictionary word — plan §10: only CORPUS-BACKED entries count.
  // Curated-only placeholders (fake uniform freqs) are not language
  // evidence; they fall through to the typo/sibling paths instead.
  if (lexicon.contains(token.normalized)
    && lexicon.realFrequency(token.normalized) > 0) {
    if (ctx.messageMode === MessageMode.NON_ACCENTED) return 'DICTIONARY';
    // ACCENTED mode: a bare-unaccented form whose accented siblings exist
    // (thanh/thành, mua/mùa...) must be judged by the contextual accent
    // pipeline, not silently accepted as dictionary-valid.
    if (unaccentedForm
      && accentIndex.candidates(stripped)
        .some((c) => c.word.toLowerCase() !== token.normalized)) {
      return 'UNACCENTED_VALID';
    }
    return 'DICTIONARY';
  }
  // 6b. plan §14 step 7: unaccented token WITH accent candidates belongs to the
  //     missing-diacritic pipeline — it is legitimate Vietnamese without dấu,
  //     never a spelling error (even in NON_ACCENTED mode).
  if (unaccentedForm && accentIndex.candidates(stripped).length > 0) {
    return 'UNACCENTED_VALID';
  }
  return 'UNKNOWN';
}

/**
 * DECISIVE trigram evidence for a real-word tone flip (plan §12):
 * at least one context window where the CANDIDATE collocation is attested
 * (>0) while the ORIGINAL's is absent (0). Three windows:
 *   centered (p1, w, n1) · forward (w, n1, n2) · backward (p2, p1, w).
 * Windows operate on the WORD sequence (punctuation-skipped) — identical
 * to how the LM builder counted, so counts are directly comparable.
 */
function decisiveTrigramFor(lm, seq, i, orig, cand) {
  const p1 = i > 0 ? seq[i - 1].normalized : null;
  const p2 = i > 1 ? seq[i - 2].normalized : null;
  const n1 = i + 1 < seq.length ? seq[i + 1].normalized : null;
  const n2 = i + 2 < seq.length ? seq[i + 2].normalized : null;
  if (p1 != null && n1 != null
    && lm.trigramJoint(p1, cand, n1) > 0
    && lm.trigramJoint(p1, orig, n1) === 0) return true;
  if (n1 != null && n2 != null
    && lm.trigramJoint(cand, n1, n2) > 0
    && lm.trigramJoint(orig, n1, n2) === 0) return true;
  if (p2 != null && p1 != null
    && lm.trigramJoint(p2, p1, cand) > 0
    && lm.trigramJoint(p2, p1, orig) === 0) return true;
  return false;
}

/**
 * Plausible surfaces of a raw context word for evidence lookups: the raw
 * form itself PLUS all its accent twins. Even a vocab-valid raw form
 * ("ban", "quy") may be the unaccented spelling of another word in this
 * very sentence, so expansion is unconditional — the max-evidence
 * aggregation decides which surface actually supports each candidate.
 */
function neighbourSurfaces(languageModel, accentIndex, rawW) {
  if (rawW == null) return [];
  const set = new Set([rawW]);
  for (const c of accentIndex.candidates(accentKey(rawW))) {
    set.add(c.word.toLowerCase());
  }
  return [...set];
}

function buildPositions(doc, eligibleList, services) {
  // every WORD position participates so the beam sees full sentence context
  const ambByIndex = new Map(eligibleList.map((a) => [a.token.start, a]));
  const wordIdx = doc.tokens
    .map((t, i) => ({ t, i }))
    .filter(({ t }) => t.type === 'WORD');
  const positions = [];
  for (let wi = 0; wi < wordIdx.length; wi++) {
    const { t } = wordIdx[wi];
    const amb = ambByIndex.get(t.start);
    if (amb) {
      positions.push({
        token: t,
        candidates: [
          ...amb.cands.map((c) => ({
            word: c.word.toLowerCase(), isOriginal: false, freq: c.freq,
          })),
          { word: t.normalized, isOriginal: true, freq: 1 },
        ],
      });
    } else {
      // Context word: resolve to the evidence-best surface using the token's
      // RAW neighbours (context-aware, not blind top-frequency), so that an
      // ambiguous neighbour ("y", "quy") contributes its informative twin.
      let w = t.normalized;
      {
        const prevTok = wi > 0 ? wordIdx[wi - 1].t : null;
        const nextTok = wi + 1 < wordIdx.length ? wordIdx[wi + 1].t : null;
        w = services.languageModel.resolveInContext(
          w,
          prevTok?.normalized.toLowerCase() ?? null,
          nextTok?.normalized.toLowerCase() ?? null,
        );
      }
      positions.push({
        token: t,
        candidates: [{ word: w, isOriginal: true, freq: 1 }],
      });
    }
  }
  return positions;
}

export function createPossibleMissingDiacriticRule(services) {
  const { configService, languageModel, beamDecoder } = services;
  return {
    id: () => RuleIds.POSSIBLE_MISSING_DIACRITIC,
    priority: () => 700,
    supports: (ctx) => ctx.messageMode === MessageMode.ACCENTED
      && configService.snapshot().get('rules.missingDiacriticEnabled') !== false
      && configService.snapshot().get('linguistic.mode') !== 'OFF',

    validate: (ctx, doc) => {
      const snap = configService.snapshot();
      const eligible = [];
      for (let ti = 0; ti < doc.tokens.length; ti++) {
        const t = doc.tokens[ti];
        if (t.type !== 'WORD') continue;
        // gate step 7 feeds BOTH dictionary-valid-unaccented and unknown tokens
        // into the accent pipeline; everything else is owned by other rules
        const cls = classifyToken(t, ctx, doc, services);
        // Wrong-TONE lane (EXPERIMENTAL, off by default): bigram evidence
        // systematically favours the common twin even when the rare one is
        // semantically right ("1kg tặng" -> tăng, "/hộp" -> hợp, "tin
        // dùng" -> đúng). Real-word tone correction needs signal beyond
        // n-grams — plan §2.3 keeps this OFF until such signal exists.
        // Rare-surface typos still caught when the surface itself is NOT in
        // the lexicon (spelling rule's same-key sibling path).
        const wrongToneEnabled = snap.get('linguistic.wrongToneEnabled')
          === true;
        if (cls === 'DICTIONARY') {
          if (!wrongToneEnabled) continue;
          const top = services.accentIndex
            .candidates(accentKey(t.normalized))[0];
          if (!top || top.word.toLowerCase() === t.normalized) continue;
        } else if (cls !== 'UNKNOWN' && cls !== 'UNACCENTED_VALID') {
          continue;
        }
        // mid-sentence Capitalized token is a proper-noun / product-name
        // signal ("Pro", "Hoa", "Thị") — never a diacritic-error candidate.
        // Sentence-initial capitals stay eligible ("Dat ban...", "Luu y...").
        {
          let wordPos = -1;
          for (let k = 0, n = 0; k < doc.tokens.length; k++) {
            if (doc.tokens[k].type === 'WORD') {
              if (doc.tokens[k].start === t.start) { wordPos = n; break; }
              n++;
            }
          }
          const rest = t.original.slice(1);
          const capitalized = /^\p{Lu}/u.test(t.original)
            && rest === rest.toLowerCase() && rest.length > 0;
          if (wordPos > 0 && capitalized) continue;
        }
        // §19 contextual gate: an isolated token gives the n-gram model no
        // context to rank with => precision-first says DO NOT WARN (Case G).
        // Neighbours are sought past protected/non-word runs (times, numbers,
        // URLs...) within a bounded window, but never across sentence breaks,
        // so "tu 23:00 den 05:00" still sees "tu" as usable left context.
        const nearestWord = (dir) => {
          let barriers = 0;
          for (let j = ti + dir; j >= 0 && j < doc.tokens.length; j += dir) {
            const tk = doc.tokens[j];
            if (tk.type === 'WORD') return tk;
            if (tk.type === 'WHITESPACE') continue;
            if (tk.type === 'PUNCTUATION' && /[.!?;]/.test(tk.original)) return null;
            if (++barriers >= 4) return null;
          }
          return null;
        };
        const hasNeighborWord =
          nearestWord(-1) != null || nearestWord(1) != null;
        if (!hasNeighborWord) continue;
        // whitelist-adjacency guard: a lone unaccented token SANDWICHED
        // between two whitelisted/code tokens is technical phrasing
        // ("SMS OTP API tu Viettel"), not a missing-diacritic candidate.
        {
          const nL = nearestWord(-1);
          const nR = nearestWord(1);
          const techy = (tk) => tk != null
            && (services.whitelistService.isAllowed(tk.original, ctx)
              || services.whitelistService.isAllowed(tk.normalized, ctx)
              || classifyToken(tk, ctx, doc, services) === 'CODE_LIKE');
          if (nL != null && nR != null && techy(nL) && techy(nR)) continue;
        }
        const stripped = accentKey(t.normalized);
        // Tone-mark choice IS an ambiguity class: an accented token whose
        // stripped key owns several surfaces ("quỳ" among quý/quy/quỵ)
        // competes in the same contextual pipeline as unaccented tokens.
        // Wrong-tone real-word typos ("chao quỳ khách") are a top VSEC
        // error category; original-prior + confidence gates protect
        // correctly-accented usage.
        if (t.normalized.length < snap.get('linguistic.minTokenLength')) continue;
        if (t.original === t.original.toUpperCase() && /[A-Z]/.test(t.original)) continue;
        const cands = services.accentIndex.candidates(stripped)
          .filter((c) => c.word.toLowerCase() !== t.normalized);
        if (cands.length === 0) continue;
        eligible.push({ token: t, cands });
      }
      if (eligible.length === 0) return [];

      const prior = snap.get('linguistic.originalPriorBonusMissingAccent')
        ?? ORIGINAL_PRIOR_BONUS;
      // plan §7: thresholds come ONLY from server-side config snapshot —
      // client-supplied options can no longer lower confidence gates.
      const minConf = snap.get('linguistic.missingDiacriticMinConfidence');
      const minMargin = snap.get('linguistic.missingDiacriticMinMargin');
      const freqWeight = snap.get('linguistic.lexicalFreqWeight') ?? 0;

      const positions = buildPositions(doc, eligible, services);
      // plan §5: decoder is stateless w.r.t. config — per-decode overrides
      const paths = beamDecoder.decode(positions, {
        beamWidth: snap.get('linguistic.beamWidth'),
        originalPrior: prior,
      });
      if (paths.length === 0) return [];
      const top1 = paths[0];

      const issues = [];
      const wordSeq = doc.tokens.filter((t) => t.type === 'WORD');
      const wordIndexOfStart = new Map(wordSeq.map((t, i) => [t.start, i]));
      const gateFreq = snap.get('linguistic.plainFormGateMinFrequency') ?? 5000;
      const beamOpts = {
        beamWidth: snap.get('linguistic.beamWidth'),
        originalPrior: prior,
      };
      let posIdx = 0;
      for (const pos of positions) {
        // plan "Fix missing-diacritic logic": the beam only provides CONTEXT.
        // EVERY ambiguous position is locally re-ranked against that context
        // and must pass the confidence gates on its own merit — emission no
        // longer depends on whether the beam happened to flip this token.
        if (pos.candidates.length > 1) {
          // Scoring evidence = TWO grounded entities per neighbour: beam's
          // global choice + raw form. Family-max marginalization diluted
          // pairwise confidence and cost latency (p95 x4); the strict
          // strong-neighbour literal rule lives in the GATE below.
          const surfWithBeam = (j) => {
            const list = [];
            const beamW = top1.words[j]?.toLowerCase();
            const raw = wordSeq[j].normalized.toLowerCase();
            if (beamW != null) list.push(beamW);
            if (!list.includes(raw)) list.push(raw);
            return list;
          };
          const prevSurfs = posIdx > 0 ? surfWithBeam(posIdx - 1) : [];
          const nextSurfs = posIdx + 1 < wordSeq.length
            ? surfWithBeam(posIdx + 1) : [];
          const prev2Surfs = posIdx > 1 ? surfWithBeam(posIdx - 2) : [];
          const next2Surfs = posIdx + 2 < wordSeq.length
            ? surfWithBeam(posIdx + 2) : [];
          const scores = pos.candidates.map((c) =>
            languageModel.scoreCandidateOverSurfaces(c.word, prevSurfs, nextSurfs, prev2Surfs, next2Surfs)
            + (c.isOriginal ? prior : 0)
            // cheap-ranker lexical frequency (plan Task 5 Step 4): the LM
            // alone cannot separate a correct word from 11 lattice twins
            // when neighbours are OOV; real-word frequency can.
            + freqWeight * Math.log10(1 + (c.isOriginal ? 1 : c.freq)));
          const probs = softmax(scores);
          const order = probs.map((p, i) => ({ p, i }))
            .sort((a, b) => b.p - a.p);
          const best = order[0];
          const bestCand = pos.candidates[best.i];
          const secondP = order[1]?.p ?? 0;
          // plan §13 interim: raw softmax is candidate-count-sensitive
          // (12 twins dilute p1 below thresholds). Pairwise dominance
          // p1/(p1+p2) is count-invariant until a proper calibrator lands.
          const conf = secondP > 0 ? best.p / (best.p + secondP) : best.p;
          const margin = best.p - secondP;

          // ---- precision gates -------------------------------------------
          // G1 plain-form gate: when the bare form itself is a strong
          // dictionary word ("ngay", "hoa"), flipping it needs DIRECT
          // evidence. Evidence ladder, strongest first:
          //   T1 centered-trigram collocation ("mua ngay tại" attested,
          //      "mua ngày tại" not) => keep original / flip, decisively;
          //   T2 direct bigram joint with a known neighbour on some side;
          //   T3 overwhelming conf+margin escape hatch (thin context);
          //   T4 name-chain guard for unknown TitleCase neighbours.
          let suppressedByGate = false;
          const plainFreq = services.lexicon.frequency(pos.token.normalized);
          const strongConf = snap.get('linguistic.plainFormStrongConfidence') ?? 0.95;
          const strongMargin = snap.get('linguistic.plainFormStrongMargin') ?? 0.9;
          const wIdxG = wordIndexOfStart.get(pos.token.start);
          const prevRawTok = wIdxG > 0 ? wordSeq[wIdxG - 1] : null;
          const nextRawTok = wIdxG + 1 < wordSeq.length ? wordSeq[wIdxG + 1] : null;
          const nextRawTok2 = wIdxG + 2 < wordSeq.length ? wordSeq[wIdxG + 2] : null;
          // Trigram evidence, two windows (plan §12):
          //   centered (prev, tok, next) and forward (tok, next, next2).
          // Forward window rescues sentence-initial tokens ("Mua ngay tại"
          // attests 'mua' even without left context).
          const evPairs = [];
          const triUsable = prevRawTok != null && nextRawTok != null;
          if (triUsable) {
            evPairs.push({
              o: languageModel.trigramJoint(prevRawTok.normalized,
                pos.token.normalized, nextRawTok.normalized),
              c: languageModel.trigramJoint(prevRawTok.normalized,
                bestCand.word, nextRawTok.normalized),
            });
          }
          const fwdUsable = nextRawTok != null && nextRawTok2 != null;
          if (fwdUsable) {
            evPairs.push({
              o: languageModel.trigramJoint(pos.token.normalized,
                nextRawTok.normalized, nextRawTok2.normalized),
              c: languageModel.trigramJoint(bestCand.word,
                nextRawTok.normalized, nextRawTok2.normalized),
            });
          }
          const triOrig = Math.max(0, ...evPairs.map((e) => e.o));
          const triCandAllZero = evPairs.every((e) => e.c === 0);
          const triCand = evPairs.length && !triCandAllZero ? 1 : 0;
          if (plainFreq >= gateFreq) {
            if (evPairs.length > 0 && triOrig > 0 && triCandAllZero) {
              suppressedByGate = true;         // T1: original collocation attested
            } else if (!(evPairs.length > 0 && triCand > 0
              && evPairs.every((e) => e.o === 0))) {
              const supportSurfaces = (j) => {
                const raw = wordSeq[j].normalized.toLowerCase();
                const list = [raw];
                // Strong neighbours (corpus-backed) contribute their LITERAL
                // form only — trusting a strong word's beam flip ("ngay" ->
                // "ngày") lets an unrelated twin family vote ("mưa ngày").
                // Weak/OOV neighbours are exactly the ones needing family
                // expansion ("co" -> có) to expose real collocations.
                if (services.lexicon.realFrequency(raw) < gateFreq) {
                  for (const s of neighbourSurfaces(languageModel,
                    services.accentIndex, raw)) {
                    if (!list.includes(s)) list.push(s);
                  }
                }
                return list;
              };
              const prevSurfsG = wIdxG > 0
                ? supportSurfaces(wIdxG - 1) : [];
              const nextSurfsG = wIdxG + 1 < wordSeq.length
                ? supportSurfaces(wIdxG + 1) : [];
              const prevJoint = languageModel.bestJointOverSurfaces(
                bestCand.word, prevSurfsG, 'left');
              const nextJoint = languageModel.bestJointOverSurfaces(
                bestCand.word, nextSurfsG, 'right');
              // relative evidence: the ORIGINAL's own collocational support
              // on the same sides ("thanh toán" attests keeping 'thanh')
              const origPrevJoint = languageModel.bestJointOverSurfaces(
                pos.token.normalized, prevSurfsG, 'left');
              const origNextJoint = languageModel.bestJointOverSurfaces(
                pos.token.normalized, nextSurfsG, 'right');
              const candSupport = prevJoint + nextJoint;
              const origSupport = origPrevJoint + origNextJoint;
              const prevKnown = prevSurfsG.length > 0;
              const nextKnown = nextSurfsG.length > 0;
              if (prevKnown || nextKnown) {
                // T2: flip needs support STRICTLY stronger than the
                // original's own collocation ("thanh toán" keeps 'thanh'
                // against a big unrelated "kỳ thành" joint).
                // Thin context (zero evidence BOTH ways) must NOT default to
                // silence for every head: a WEAK/unknown head ("Dat" ở đầu
                // câu) still warns through the normal confidence gates, while
                // a STRONG real-word head ("ngay" @150k) demands the
                // overwhelming escape hatch — precision-first where it matters.
                const thinContext = candSupport === 0 && origSupport === 0;
                if (!thinContext) {
                  suppressedByGate = candSupport <= origSupport;
                } else {
                  const strongEscape = conf >= strongConf && margin >= strongMargin;
                  const headIsStrongRealWord = plainFreq >= gateFreq;
                  const normalGatesPass = conf >= minConf && margin >= minMargin;
                  suppressedByGate = headIsStrongRealWord
                    ? !strongEscape
                    : !normalGatesPass;
                }
              } else {
                // T4 name-chain guard: "Trần Thị Hoa"
                suppressedByGate = Boolean(prevRawTok
                  && /^[A-Z]/.test(prevRawTok.original)
                  && /^[A-Z]/.test(pos.token.original));
              }
            }
          }

          if (process.env.SMS_VAL_DEBUG) {
            console.error(`[pmd] '${pos.token.original}' -> ${bestCand.word}`
              + ` conf=${conf.toFixed(3)} margin=${margin.toFixed(3)}`
              + ` probs=[${pos.candidates.map((c, k) => c.word + ':' + probs[k].toFixed(3)).join(', ')}]`
              + ` neq=${bestCand.word !== pos.token.normalized}`
              + ` accented=${hasVietnameseAccent(bestCand.word)}`
              + ` minConf=${minConf} minMargin=${minMargin}`
              + ` tri=[${triOrig}/${triCand}]`
              + ` plainGateSuppressed=${suppressedByGate}`
              + ` emit=${!suppressedByGate && bestCand.word !== pos.token.normalized && hasVietnameseAccent(bestCand.word) && conf >= minConf && margin >= minMargin}`);
          }

          if (!suppressedByGate
            && bestCand.word !== pos.token.normalized
            && hasVietnameseAccent(bestCand.word)
            // Rule-boundary: PMD owns MISSING-diacritic only. A token that
            // already carries an accent ("hưỏng") is a wrong-tone TYPO and
            // belongs to POSSIBLE_SPELLING_ERROR's same-key sibling path.
            && !hasVietnameseAccent(pos.token.normalized)
            && conf >= minConf
            && margin >= minMargin) {
            issues.push(new ValidationIssue(
              RuleIds.POSSIBLE_MISSING_DIACRITIC, Severity.WARNING,
              pos.token.start, pos.token.end, pos.token.original,
              `Từ "${pos.token.original}" có thể đang thiếu dấu (gợi ý: "${bestCand.word}").`,
              [bestCand.word],
              Math.round(conf * 100) / 100,
            ));
          }
        }
        posIdx++;
      }
      return issues;
    },
  };
}

/**
 * Task 3 (recall-improvement plan): PURE candidate construction, extracted
 * from emission policy. A lane builds candidates WITHOUT deciding whether
 * to warn; protected/whitelisted/abbreviation tokens are ineligible before
 * any construction. Production currently consumes only UNKNOWN_TYPO — the
 * real-word lanes exist for the shadow modes of Tasks 4-5.
 *
 * Result (data-only):
 *   { lane, eligible, reason,
 *     entries: [{ word, stripped, dist, freq, sameAccentKey }],
 *     original }
 */
export const CORRECTION_LANES = Object.freeze([
  'UNKNOWN_TYPO',
  'UNACCENTED_SAME_KEY',      // missing-diacritic lane (PMD-owned)
  'ACCENTED_SAME_KEY',        // wrong-diacritic spelling lane (Task 4)
  'DIFFERENT_KEY_REAL_WORD',  // real-word typo lane (Task 5)
]);

// Task 5: SHADOW real-word lane bounds — bounded deterministic coverage of
// the wide pool (at most 12 pairwise probes per token, 3 recorded hard
// negatives for the Task 6 trainer); sizing revisited by Task 7/9.
const REAL_WORD_SHADOW_EVAL_LIMIT = 12;
const REAL_WORD_SHADOW_RIVAL_LIMIT = 3;

/**
 * Task 7: diversity-preserving shortlist — the union, deduplicated in a
 * deterministic order, of:
 *   1. the best edit-distance candidate;
 *   2. the best direct-context-attested candidate;
 *   3. the best same-accent-key candidate (when one exists);
 *   4. candidates in descending cheap score (fills remaining slots).
 * Selection uses PRODUCTION FEATURES ONLY — no label/target access — and is
 * capped at `size` non-original candidates so the expensive stage bound
 * cannot grow beyond four.
 * @param {Array<{word:string,dist:number,freq:number,stripped?:string}>} entries
 * @param {{attestOf:Map<string,boolean>, famKey:string, size:number}} p
 */
export function selectDiverseShortlist(entries, { attestOf, famKey, size = 8,
  cheapOrder = null, recallScoreOf = null }) {
  const keyOf = (c) => `${c.word.toLowerCase()}|${c.stripped ?? ''}`;
  const seen = new Set();
  const out = [];
  const add = (c) => {
    if (!c || out.length >= size) return;
    const k = keyOf(c);
    if (seen.has(k)) return;
    seen.add(k);
    out.push(c);
  };

  const cheapMap = new Map();
  if (cheapOrder) {
    cheapOrder.forEach((c, idx) => cheapMap.set(c.word.toLowerCase(), idx));
  }
  const cheapRank = (c) => cheapMap.get(c.word.toLowerCase()) ?? 999;
  const recallScore = (c) => {
    if (typeof recallScoreOf !== 'function') return null;
    const value = recallScoreOf(c);
    return Number.isFinite(value) ? value : null;
  };

  // 1. Top 2 by cheap overall score (head)
  if (cheapOrder) {
    if (cheapOrder[0]) add(cheapOrder[0]);
    if (cheapOrder[1]) add(cheapOrder[1]);
  }

  // 2. Best same-accent-key candidates (up to 3 for family diversity)
  const sameKeys = entries.filter((c) => c.stripped === famKey)
    .sort((a, b) => cheapRank(a) - cheapRank(b) || (b.freq ?? 0) - (a.freq ?? 0) || a.dist - b.dist);
  if (sameKeys[0]) add(sameKeys[0]);
  if (sameKeys[1]) add(sameKeys[1]);
  if (sameKeys[2] && size >= 8) add(sameKeys[2]);

  // 3. Best edit-distance candidates (typos)
  const byDistFreq = [...entries].sort((a, b) => a.dist - b.dist
    || cheapRank(a) - cheapRank(b) || (b.freq ?? 0) - (a.freq ?? 0));
  if (byDistFreq[0]) add(byDistFreq[0]);
  if (byDistFreq[1] && size >= 6) add(byDistFreq[1]);

  // 4. Best context-attested candidates
  const attested = entries.filter(
    (c) => attestOf?.get?.(c.word.toLowerCase()) === true,
  ).sort((a, b) => cheapRank(a) - cheapRank(b) || (b.freq ?? 0) - (a.freq ?? 0));
  if (attested[0]) add(attested[0]);
  if (attested[1] && size >= 8) add(attested[1]);

  // 5. Best production recall-reranker signal. This is deliberately a
  // callback: callers may reuse the already-loaded immutable model, while
  // tests and extraction can supply a deterministic feature score. Labels
  // are never visible to this selector.
  const recallRanked = typeof recallScoreOf === 'function'
    ? [...entries].map((c, index) => ({ c, index, score: recallScore(c) }))
      .filter((x) => x.score != null)
      .sort((a, b) => b.score - a.score
        || cheapRank(a.c) - cheapRank(b.c)
        || a.index - b.index)
    : [];
  if (recallRanked[0]) add(recallRanked[0].c);

  // 6. Highest frequency candidate
  const byFreq = [...entries].sort((a, b) => (b.freq ?? 0) - (a.freq ?? 0) || cheapRank(a) - cheapRank(b));
  if (byFreq[0]) add(byFreq[0]);

  // 7. Fill remainder with cheap ranking / distance order
  for (const c of (cheapOrder ?? byDistFreq)) {
    if (out.length >= size) break;
    add(c);
  }

  // If all diversity heads consumed the bound, let a materially stronger
  // recall score replace only a non-head filler. This preserves the existing
  // distance/family/context guarantees while rescuing candidates that are
  // cheap-rank outliers. No gold/label information enters this decision.
  if (recallRanked[0] && !seen.has(keyOf(recallRanked[0].c)) && out.length >= size) {
    const headKeys = new Set([
      cheapOrder?.[0], cheapOrder?.[1], sameKeys[0], byDistFreq[0], attested[0],
    ].filter(Boolean).map(keyOf));
    const candidate = recallRanked[0].c;
    const weakest = [...out].map((c, index) => ({ c, index }))
      .filter((x) => !headKeys.has(keyOf(x.c)))
      .sort((a, b) => cheapRank(b.c) - cheapRank(a.c)
        || (a.c.freq ?? 0) - (b.c.freq ?? 0)
        || a.index - b.index)[0];
    if (weakest) {
      const oldScore = recallScore(weakest.c);
      if (oldScore == null || recallScore(candidate) > oldScore) {
        out[weakest.index] = candidate;
      }
    }
  }
  return out;
}

export function buildCorrectionCandidates(
  { token, lane, services, snap, ctx = null, doc = null },
) {
  const original = token.normalized;
  const result = { lane, eligible: false, reason: null, entries: [], original };
  if (!CORRECTION_LANES.includes(lane)) {
    result.reason = 'unknown-lane';
    return result;
  }
  if (ctx && doc) {
    const cls = classifyToken(token, ctx, doc, services);
    if (cls === 'PROTECTED' || cls === 'WHITELISTED'
      || cls === 'CODE_LIKE' || cls === 'ABBREVIATION') {
      result.reason = `classified-${cls}`;
      return result;
    }
  }
  const famKey = accentKey(original);
  const withSameKeyFlag = (e) => ({
    ...e,
    sameAccentKey: e.stripped === famKey,
  });

  if (lane === 'UNKNOWN_TYPO') {
    // exact production behavior: wide SymSpell pool + same-key sibling merge
    const maxDist = maxEditDistanceFor(original.length);
    const pool = services.typoCandidateProvider.generatePool(original, maxDist, {
      keyCap: snap.get('linguistic.spellingCandidateKeys') ?? 12,
      surfacesPerKey: snap.get('linguistic.spellingSurfacesPerKey') ?? 4,
    });
    let cands = pool.entries.slice(0, snap.get('linguistic.spellingPoolMax') ?? 48);
    if (hasVietnameseAccent(original)) {
      const seen = new Set(cands.map((c) => c.word.toLowerCase()));
      for (const c of services.accentIndex.candidates(famKey)) {
        const lw = c.word.toLowerCase();
        if (lw === original || seen.has(lw)) continue;
        const dist = damerauOsaDistance(
          applyTelexHints(original),
          applyTelexHints(c.word),
        );
        if (dist <= 2) {
          seen.add(lw);
          cands.push({
            word: c.word,
            stripped: famKey,
            dist,
            freq: c.freq,
          });
        }
      }
    }
    result.eligible = true;
    result.reason = 'ok';
    result.entries = cands.map(withSameKeyFlag);
    return result;
  }

  if (lane === 'UNACCENTED_SAME_KEY' || lane === 'ACCENTED_SAME_KEY') {
    // every OTHER surface of this stripped key — dictionary validity of the
    // input is irrelevant to construction (that is an emission-policy matter)
    const telexTok = applyTelexHints(original);
    result.eligible = true;
    result.reason = 'ok';
    result.entries = services.accentIndex.candidates(famKey)
      .filter((c) => c.word.toLowerCase() !== original)
      .map((c) => withSameKeyFlag({
        word: c.word,
        stripped: famKey,
        dist: damerauOsaDistance(telexTok, applyTelexHints(c.word)),
        freq: c.freq,
      }));
    return result;
  }

  // DIFFERENT_KEY_REAL_WORD: SymSpell alternatives for a valid dictionary
  // word. Same-key surfaces stay in the pool but are FLAGGED (not filtered)
  // so audits can slice lanes without duplicating generation logic.
  {
    const maxDist = maxEditDistanceFor(original.length);
    const pool = services.typoCandidateProvider.generatePool(original, maxDist, {
      keyCap: snap.get('linguistic.spellingCandidateKeys') ?? 12,
      surfacesPerKey: snap.get('linguistic.spellingSurfacesPerKey') ?? 4,
    });
    result.eligible = true;
    result.reason = 'ok';
    result.entries = pool.entries
      .slice(0, snap.get('linguistic.spellingPoolMax') ?? 48)
      .map(withSameKeyFlag);
  }
  return result;
}

/**
 * Task 4: explicit wrong-diacritic lane mode. Absent key falls back to the
 * legacy boolean mapping (wrongToneEnabled=true -> ACTIVE) so pre-existing
 * config keeps its exact semantics.
 */
export function wrongDiacriticModeOf(snap) {
  const mode = snap.get('linguistic.wrongDiacriticMode');
  if (mode != null) return mode;
  return snap.get('linguistic.wrongToneEnabled') === true
    ? 'ACTIVE' : 'OFF';
}

/**
 * Task 4/5 Step — conservative PROVISIONAL pairwise gate. Used only to
 * collect SHADOW features and establish a ceiling until the Task 6
 * calibrator replaces it; never activates a lane by itself. wouldEmit
 * requires ALL of:
 *   - candidate accent-key relation matches the lane's ownership;
 *   - candidate has >=1 directly attested bigram/trigram window;
 *   - original has ZERO directly attested trigram wins in those windows;
 *   - pairwise evidence margin is positive;
 *   - candidate frequency passes the existing floor;
 *   - (DIFFERENT_KEY_REAL_WORD only) candidate is corpus-backed.
 */
function provisionalPairwiseDecision(
  { lane, services, snap, languageModel, words, idx, candWord, candFreq,
    original, cheapRank = 0, poolRank = 0 },
) {
  // accept both plain surfaces and WORD-token objects
  const seq = words.map((w) => (typeof w === 'string' ? w : w.normalized));
  const features = extractContextEvidence({
    languageModel, words: seq, idx,
    candidateWord: candWord, originalWord: original,
  });
  const trigramWins = (features.centeredTrigramLogRatio > 0 ? 1 : 0)
    + (features.forwardTrigramLogRatio > 0 ? 1 : 0)
    + (features.backwardTrigramLogRatio > 0 ? 1 : 0);
  const score = evidenceScore(features);
  // Task 6: pairwise model probability rides along on every shadow decision
  // (SHADOW measurement only — it never changes emission here). Assembled
  // from the SAME evidence block to avoid duplicate LM probes; a failure is
  // recorded as null rather than blocking the gate chain.
  const rr = services.recallReranker;
  let modelProbability = null;
  if (rr && rr.enabled) {
    try {
      const cf = services.lexicon.frequency(candWord) || 0;
      const ofq = services.lexicon.frequency(original) || 0;
      modelProbability = rr.score({
        editDistance: features.editDistance,
        sameAccentKey: features.sameAccentKey ? 1 : 0,
        candidateMinusOriginalLogFrequency:
          Math.log(1 + cf) - Math.log(1 + ofq),
        leftBigramLogRatio: features.leftBigramLogRatio,
        rightBigramLogRatio: features.rightBigramLogRatio,
        centeredTrigramLogRatio: features.centeredTrigramLogRatio,
        forwardTrigramLogRatio: features.forwardTrigramLogRatio,
        backwardTrigramLogRatio: features.backwardTrigramLogRatio,
        candidateAttestedWindows: features.candidateAttestedWindows,
        originalAttestedWindows: features.originalAttestedWindows,
        tokenLength: original.length,
        originalIsDictionary: services.lexicon.contains(original) ? 1 : 0,
        candidateCheapRank: cheapRank || RECALL_RANK_SENTINEL,
        candidatePoolRank: poolRank || RECALL_RANK_SENTINEL,
      });
    } catch {
      modelProbability = null;
    }
  }
  // Lane routing (Task 5): ACCENTED_SAME_KEY demands a same-key twin. The
  // real-word lane rejects same-key twins that differ ONLY by tone mark
  // ("quỳ" -> "quý", "hành" -> "hàng" — wrong-diacritic territory,
  // historically FP-prone) while allowing same-key pairs that also change
  // vowel quality ("đen" -> "đến": e -> ê), reachable by no other lane
  // because the token itself is dictionary-valid.
  const toneOnlyFlip = (a, b) => {
    // the five Vietnamese tone marks in NFD; vowel-quality marks (circumflex
    // U+0302, horn, breve) are deliberately KEPT so "đen"/"đến" differ
    const TONE_MARKS = /[̣̀́̉̃]/gu;
    return a.normalize('NFD').toLowerCase().replace(TONE_MARKS, '')
      === b.normalize('NFD').toLowerCase().replace(TONE_MARKS, '');
  };
  const checks = {
    accentKeyRelation: lane === 'ACCENTED_SAME_KEY'
      ? features.sameAccentKey
      : (!features.sameAccentKey || !toneOnlyFlip(candWord, original)),
    candidateAttestedWindow: features.candidateAttestedWindows >= 1,
    // Task 4 gate (wrong-diacritic only): a tone twin needs a CLEAN SWEEP —
    // the original owns zero directly attested trigram wins. The real-word
    // lane (Task 5) deliberately OMITS this: grammatical originals ("Các",
    // "hàng") legitimately attest inside trigrams everywhere, so demanding
    // zero would blind the lane exactly where it matters; their pull is
    // already priced into marginPositive instead.
    ...(lane === 'ACCENTED_SAME_KEY'
      ? { originalZeroTrigramWins: trigramWins === 0 } : {}),
    marginPositive: Number.isFinite(score) && score > 0,
    frequencyFloor: (candFreq ?? 0)
      >= (snap.get('linguistic.spellingMinFrequency') ?? 1000),
  };
  if (lane === 'DIFFERENT_KEY_REAL_WORD') {
    checks.corpusBacked = services.lexicon.contains(candWord)
      && services.lexicon.realFrequency(candWord) > 0;
  }
  return {
    lane,
    wouldEmit: Object.values(checks).every(Boolean),
    checks,
    candidate: candWord,
    evidenceScore: Math.round(score * 1000) / 1000,
    modelProbability,
    features,
  };
}

/**
 * Task 8: single-source-of-truth spelling decision for ONE token.
 * Used BOTH by createPossibleSpellingErrorRule and by the split-safe
 * evaluation workflow (tools/run_spelling_eval.mjs) so diagnostic stages can
 * never drift from production behavior. Returns why the token was skipped
 * (stage 'prefilter'), why no candidates survived ('no-candidates'), or the
 * full ranking + gate decision ('decided').
 */
/**
 * Task 5/9 shared real-word lane machinery. Builds the bounded candidate
 * set from the wide pool, applies the pairwise gate to each survivor and
 * picks the best-evidence winner of the MINIMAL surface-distance tier.
 * Used by BOTH the cascade SHADOW path and the ACTIVE light path so the two
 * modes can never drift apart.
 *
 * Quick-attested precheck: candidates with no left-bigram, right-bigram nor
 * centered-trigram attestation can essentially never clear
 * candidateAttestedWindows >= 1, so they are skipped after 3 map lookups
 * instead of a full 11-probe feature extraction (forward/backward-only
 * window winners are sacrificed — measured cost of the p95 budget).
 */
export function computeRealWordLaneDecision(
  { builtEntries, services, snap, languageModel, words, idx, original },
) {
  const seq = words.map((w) => (typeof w === 'string' ? w : w.normalized));
  const seenRw = new Set([original]);
  const rankMap = ranksForCandidates(builtEntries);
  const p1 = idx > 0 ? seq[idx - 1].toLowerCase() : null;
  const n1 = idx + 1 < seq.length ? seq[idx + 1].toLowerCase() : null;
  const lm = languageModel;
  const quickAttested = (w) => (p1 !== null
    && (lm.bigram.get(`${p1} ${w}`) ?? 0) > 0)
    || (n1 !== null && (lm.bigram.get(`${w} ${n1}`) ?? 0) > 0)
    || (p1 !== null && n1 !== null && lm.trigramJoint(p1, w, n1) > 0);
  const pool = [...builtEntries]
    .filter((c) => c.dist <= maxEditDistanceFor(original.length))
    .sort((a, b) => a.dist - b.dist
      || (b.freq ?? 0) - (a.freq ?? 0) || a.word.localeCompare(b.word))
    .filter((c) => {
      const lw = c.word.toLowerCase();
      if (seenRw.has(lw)) return false;
      seenRw.add(lw);
      return true;
    })
    .filter((c) => quickAttested(c.word.toLowerCase()))
    .slice(0, REAL_WORD_SHADOW_EVAL_LIMIT);
  // deterministic evidence-desc order; stable sort keeps the pool's
  // (dist,freq,word) order for ties so the whole selection is reproducible
  const evaluated = pool.map((c) => provisionalPairwiseDecision({
    lane: 'DIFFERENT_KEY_REAL_WORD', services, snap, languageModel,
    words: seq, idx, candWord: c.word,
    candFreq: services.lexicon.frequency(c.word) || c.freq || 1,
    original,
    cheapRank: rankMap.get(c.word.toLowerCase())?.cheapRank ?? 0,
    poolRank: rankMap.get(c.word.toLowerCase())?.poolRank ?? 0,
  })).sort((a, b) => b.evidenceScore - a.evidenceScore
    || b.modelProbability - a.modelProbability);
  if (evaluated.length === 0) {
    return { winner: null, rivals: [], evaluatedCount: 0 };
  }
  // Winner comes from the MINIMAL edit-distance tier whenever one of its
  // candidates passes the gate: a real-word typo is the smallest plausible
  // corruption of the intended surface ("đến" beats the higher-frequency
  // but farther "để" for "Đen rồi mà"). Distance here is the SURFACE
  // Telex-aware feature distance, not the stripped-key generation distance —
  // key distance collapses prefix families ("de" is dist-1 from "den") and
  // cannot separate "đến" from "để".
  let winner;
  const emitIdx = evaluated.findIndex((x) => x.wouldEmit);
  if (emitIdx >= 0) {
    const minDistRw = Math.min(
      ...evaluated.map((x) => x.features.editDistance));
    winner = evaluated
      .find((x) => x.wouldEmit && x.features.editDistance === minDistRw)
      ?? evaluated[emitIdx];
  } else {
    // nobody passes: report the strongest decision anyway so SHADOW
    // accounting sees WHY the gate declined
    winner = evaluated[0];
  }
  const rivals = evaluated
    .filter((x) => x !== winner)
    .slice(0, REAL_WORD_SHADOW_RIVAL_LIMIT)
    .map(({ checks, candidate, evidenceScore, features, wouldEmit }) => ({
      candidate, evidenceScore, wouldEmit, checks, features,
    }));
  return { winner, rivals, evaluatedCount: evaluated.length };
}

/** Task 9 ACTIVE light path — dictionary-valid tokens NEVER traverse the
 * legacy unknown-token cascade. Guards are inherited from the caller; this
 * function owns pool construction, the pairwise decision and calibrated
 * emission. */
function evaluateRealWordToken(
  { services, snap, languageModel, ctx, doc, words, idx },
) {
  const t = words[idx];
  const built = buildCorrectionCandidates({
    token: t, lane: 'UNKNOWN_TYPO', services, snap, ctx, doc,
  });
  const { winner, rivals, evaluatedCount } = computeRealWordLaneDecision({
    builtEntries: built.entries ?? [],
    services, snap, languageModel, words, idx, original: t.normalized,
  });
  if (winner) winner.rivals = rivals;

  const minProb = snap.get('linguistic.realWordTypoMinProbability') ?? 1;
  const minWin = snap.get('linguistic.realWordTypoMinCandidateWindows') ?? 1;
  const maxOrig = snap.get('linguistic.realWordTypoMaxOriginalWindows') ?? 3;
  const minMargin = snap.get('linguistic.realWordTypoMinMargin') ?? 0;
  const feats = winner?.features ?? {};
  const pass = Boolean(winner?.wouldEmit)
    && (winner?.modelProbability ?? 0) >= minProb
    && (feats.candidateAttestedWindows ?? 0) >= minWin
    && (feats.originalAttestedWindows ?? 99) <= maxOrig
    && (winner?.evidenceScore ?? -Infinity) >= minMargin;

  // Task 1 (attention plan): explicit decision diagnostics shared by the
  // classical and attention paths. Terminal-stage attribution must derive
  // from THIS shape — never from empty ranked/cheapKept arrays.
  let rejectionReason = null;
  if (!pass) {
    if (!(built.entries ?? []).length) rejectionReason = 'no-candidates';
    else if (!winner) rejectionReason = 'no-viable-candidate';
    else if (!winner.wouldEmit) rejectionReason = 'lane-evidence-declined';
    else if ((winner.modelProbability ?? 0) < minProb) rejectionReason = 'below-min-probability';
    else if ((feats.candidateAttestedWindows ?? 0) < minWin) rejectionReason = 'candidate-windows-insufficient';
    else if ((feats.originalAttestedWindows ?? 99) > maxOrig) rejectionReason = 'original-windows-exceeded';
    else rejectionReason = 'margin-below-minimum';
  }
  const decision = {
    generatedWords: (built.entries ?? []).map((c) => String(c.word).toLowerCase()),
    consideredWords: evaluatedCount,
    selectedWord: winner ? String(winner.candidate).toLowerCase() : null,
    emitted: pass,
    rejectionReason,
    decisionOwner: 'classical-real-word-lane',
  };

  const gates = pass
    ? { realWordLaneEmit: true }
    : { realWordLaneDeclined: true };
  const best = pass
    ? {
      word: winner.candidate,
      p: winner.modelProbability ?? 0,
      freq: services.lexicon.frequency(winner.candidate) || 1,
      isOriginal: false,
    }
    : null;
  const maxSuggestions = snap.get('linguistic.maxSpellingSuggestions');
  const suggestions = pass
    ? [winner.candidate]
    : [];
  void maxSuggestions;
  return {
    stage: 'decided',
    emit: pass,
    gates,
    decision,
    poolWideCount: (built.entries ?? []).length,
    wideRankedWords: (built.entries ?? []).map((c) => c.word),
    cheapKept: [],
    ranked: [],
    best,
    second: null,
    suggestions,
    isRealWordTone: false,
    reason: undefined,
    shadowWrongDiacritic: null,
    shadowRealWordTypo: winner,
  };
}

export function evaluateSpellingToken(services, snap, ctx, doc, words, idx) {
  const { languageModel, typoCandidateProvider } = services;
  const t = words[idx];

  let cls = classifyToken(t, ctx, doc, services);
  // Task 4: wrong-diacritic lane mode (OFF | SHADOW | ACTIVE). In SHADOW the
  // lane computes a decision for dictionary-valid accented tokens but NEVER
  // emits; in ACTIVE it behaves like the legacy real-word-tone path.
  const wdMode = wrongDiacriticModeOf(snap);
  const legacyTone = snap.get('linguistic.wrongToneEnabled') === true;
  let isRealWordTone = false;
  if (wdMode === 'OFF') {
    // legacy exact behavior: DICTIONARY token that is NOT the frequency-top
    // surface of its stripped key may be a real word used wrongly
    if (legacyTone && cls === 'DICTIONARY') {
      const top = services.accentIndex
        .candidates(accentKey(t.normalized))[0];
      if (top && top.word.toLowerCase() !== t.normalized) {
        isRealWordTone = true;
      }
    }
  } else if (cls === 'DICTIONARY' && hasVietnameseAccent(t.normalized)) {
    const top = services.accentIndex
      .candidates(accentKey(t.normalized))[0];
    if (top && top.word.toLowerCase() !== t.normalized) {
      isRealWordTone = true;
    }
  }
  // Task 5/9: DIFFERENT_KEY_REAL_WORD lane. SHADOW computes + reports the
  // decision without emitting; ACTIVE (post-calibration only) may emit when
  // the trained pairwise probability clears the frozen threshold.
  const rwMode = snap.get('linguistic.realWordTypoMode') ?? 'OFF';
  let isRealWordTypo = false;
  if (rwMode !== 'OFF' && cls === 'DICTIONARY') {
    isRealWordTypo = true;
  }
  if (cls !== 'UNKNOWN' && !isRealWordTone && !isRealWordTypo) {
    return { stage: 'prefilter', reason: `classified-${cls}` };
  }
  if (t.normalized.length < snap.get('linguistic.spellingMinTokenLength')) {
    return { stage: 'prefilter', reason: 'min-token-length' };
  }
  if (t.original === t.original.toUpperCase() && /[A-Z]/.test(t.original)) {
    return { stage: 'prefilter', reason: 'all-caps' };
  }
  // mid-sentence Capitalized token ("Pro", "Hoa") is almost surely a
  // proper noun / product name — never a spelling-error candidate
  if (idx > 0 && /^[A-Z][a-zà-ỹ]/.test(t.original)) {
    return { stage: 'prefilter', reason: 'capitalized-proper-noun' };
  }
  const prevWord = idx > 0 ? words[idx - 1].normalized : null;
  const nextWord = idx + 1 < words.length ? words[idx + 1].normalized : null;
  // plan §22 "context score đủ": require at least one usable neighbour —
  // known to the LM, or itself a legitimate unaccented Vietnamese token
  // (accent candidates exist). Without any anchor the ranking is
  // unreliable and precision-first says DO NOT WARN.
  const anchorable = (w) => w != null
    && (languageModel.knows(w)
      || services.accentIndex.candidates(accentKey(w)).length > 0);
  if (!anchorable(prevWord) && !anchorable(nextWord)) {
    return { stage: 'prefilter', reason: 'no-context-anchor' };
  }

  // Task 9: ACTIVE dictionary tokens take the dedicated LIGHT path — they
  // never traverse the legacy unknown-token cascade (latency + precision).
  if (isRealWordTypo && rwMode === 'ACTIVE') {
    return evaluateRealWordToken(
      { services, snap, languageModel, ctx, doc, words, idx });
  }

  // Task 3: generation goes through the shared pure builder (UNKNOWN_TYPO
  // lane) so candidate construction has ONE source of truth; the decision
  // cascade below is unchanged.
  const built = buildCorrectionCandidates({
    token: t, lane: 'UNKNOWN_TYPO', services, snap, ctx, doc,
  });
  let cands = built.entries;
  const pool = { entries: cands };
  // Wrong-tone typos ("quỳ khách" for "quý khách"): same-key sibling merge
  // now lives INSIDE the builder's UNKNOWN_TYPO lane.
  if (cands.length === 0) {
    return {
      stage: 'no-candidates', reason: 'empty-pool', pool,
      decision: {
        generatedWords: [], consideredWords: 0, selectedWord: null,
        emitted: false, rejectionReason: 'no-candidates',
        decisionOwner: 'classical-unknown-lane',
      },
    };
  }

  const alpha = snap.get('linguistic.typoAlpha');
  const beta = snap.get('linguistic.typoBeta');
  const gamma = snap.get('linguistic.typoGamma');
  const origPrior = snap.get('linguistic.originalPriorBonusSpelling');
  const minConf = snap.get('linguistic.spellingMinConfidence');
  const minMargin = snap.get('linguistic.spellingMinMargin');
  const minFreq = snap.get('linguistic.spellingMinFrequency');

  // anchored family context (same machinery as PMD §12) so sibling
  // ranking sees attested collocations regardless of raw form
  const prevFam = idx > 0
    ? neighbourSurfaces(languageModel, services.accentIndex,
      words[idx - 1].normalized.toLowerCase()) : [];
  const nextFam = idx + 1 < words.length
    ? neighbourSurfaces(languageModel, services.accentIndex,
      words[idx + 1].normalized.toLowerCase()) : [];

  // Task 5 Step 4 — CHEAP first-stage ranker (no trigram probes):
  // Telex-aware edit distance + lexicon frequency + direct
  // accent-family relation + at most ONE bigram attestation check.
  const cheapTopK = snap.get('linguistic.spellingCheapTopK') ?? 3;
  const cw = {
    dist: snap.get('linguistic.cheapDistWeight') ?? 1.0,
    freq: snap.get('linguistic.cheapFreqWeight') ?? 0.5,
    family: snap.get('linguistic.cheapFamilyBonus') ?? 0.75,
    attest: snap.get('linguistic.cheapAttestBonus') ?? 0.25,
  };
  const telexTok = applyTelexHints(t.normalized);
  const famKey = accentKey(t.normalized);
  const attestSide = nextFam.length >= prevFam.length
    ? { surfs: nextFam, side: 'right' } : { surfs: prevFam, side: 'left' };
  // Task 7: probe attestation ONCE per candidate (previously re-probed per
  // sort comparison); the map feeds both cheap scoring and shortlist head 2.
  const attestOf = new Map();
  for (const c of cands) {
    attestOf.set(c.word.toLowerCase(),
      attestSide.surfs.length > 0
        && languageModel.bestJointOverSurfaces(
          c.word.toLowerCase(), attestSide.surfs, attestSide.side) > 0);
  }
  const cheapScore = (c) => {
    let sc = -cw.dist * damerauOsaDistance(telexTok, applyTelexHints(c.stripped));
    sc += cw.freq * Math.log10(1 + (c.freq || 1));
    if (c.stripped === famKey) sc += cw.family;
    if (attestOf.get(c.word.toLowerCase())) sc += cw.attest;
    return sc;
  };
  const wideRanked = [...cands]
    .sort((a, b) => cheapScore(b) - cheapScore(a)
      || a.word.localeCompare(b.word));
  const shortlistRankMap = ranksForCandidates(cands);
  const recallScoreOf = services.recallReranker?.enabled ? (candidate) => {
    try {
      const rank = shortlistRankMap.get(candidate.word.toLowerCase()) ?? {};
      return services.recallReranker.score(buildRecallPairwiseFeatures({
        languageModel,
        services,
        words,
        idx,
        candidateWord: candidate.word,
        originalWord: t.normalized,
        cheapRank: rank.cheapRank,
        poolRank: rank.poolRank,
      }));
    } catch {
      return null;
    }
  } : null;
  // Task 7 Step 2: diversity-preserving shortlist replaces the pure cheap
  // top-K cut so a context-attested winner below the frequency cut survives
  // to the expensive stage. Bound stays at four non-original candidates.
  const shortlistSize = Math.max(1,
    snap.get('linguistic.spellingShortlistSize')
    ?? Math.max(1, cheapTopK));
  cands = selectDiverseShortlist(wideRanked,
    {
      attestOf,
      famKey,
      size: shortlistSize,
      cheapOrder: wideRanked,
      recallScoreOf,
    });

  let attentionCands = cands;
  const scored = cands.map((c) => {
    const freq = services.lexicon.frequency(c.word) || 1;
    const cost = damerauOsaDistance(
      applyTelexHints(t.normalized),
      applyTelexHints(c.stripped),
    );
    const ctxScore = languageModel.scoreCandidateOverSurfaces(
      c.word.toLowerCase(), prevFam, nextFam);
    return {
      word: c.word,
      dist: cost,
      freq,
      final: -alpha * cost + beta * Math.log(freq + 1) + gamma * ctxScore,
    };
  }).filter((c) => c.freq >= minFreq);

  // original kept as candidate with smaller prior (already failed dict gate)
  const origScore = gamma
    * languageModel.scoreCandidateOverSurfaces(
      t.normalized, prevFam, nextFam)
    + origPrior;
  const all = [...scored.map((c) => ({ ...c, isOriginal: false })),
    { word: t.normalized, final: origScore, isOriginal: true, freq: 1 }];
  const probsRaw = softmax(all.map((a) => a.final));
  const ranked = all.map((a, i) => ({ ...a, p: probsRaw[i] }))
    .sort((x, y) => y.final - x.final);
  const best = ranked[0];
  const second = ranked[1];

  const gates = {};
  let shadowWrongDiacritic = null;
  let shadowRealWordTypo = null;
  let emit = Boolean(best && !best.isOriginal);
  // Task 4: the SHADOW lane owns its own pairwise gate — it must report a
  // decision EVEN WHEN the legacy softmax gates would decline, so headroom
  // measurement is not capped by the unknown-token thresholds.
  const shadowLaneCandidate = (emit && isRealWordTone && wdMode === 'SHADOW')
    ? best : null;
  const wdRankMap = shadowLaneCandidate
    ? ranksForCandidates(built.entries) : null;
  if (shadowLaneCandidate) {
    const lwBest = shadowLaneCandidate.word.toLowerCase();
    shadowWrongDiacritic = provisionalPairwiseDecision({
      lane: 'ACCENTED_SAME_KEY', services, snap, languageModel, words, idx,
      candWord: shadowLaneCandidate.word,
      candFreq: shadowLaneCandidate.freq,
      original: t.normalized,
      cheapRank: Math.max(1,
        wideRanked.findIndex((w) => w.word.toLowerCase() === lwBest) + 1),
      poolRank: wdRankMap.get(lwBest)?.poolRank ?? 0,
    });
    emit = false;
    gates.shadowSuppressed = true;
  } else if (!emit) {
    gates.noNonOriginalWinner = true;
  } else if (second && (best.p - second.p) < minMargin) {
    emit = false; gates.marginFail = true;
  } else if (best.p < minConf) {
    emit = false; gates.confidenceFail = true;
  } else if (best.freq < minFreq) {
    emit = false; gates.frequencyFail = true;
  }
  // real-word tone flips REQUIRE decisive symmetric trigram proof:
  // candidate attested in some window AND original attested in NONE.
  // One-sided proof let unrelated twins win ("mến"/"lúc" FPs).
  if (emit && isRealWordTone) {
    const p1i = idx > 0 ? words[idx - 1].normalized : null;
    const p2i = idx > 1 ? words[idx - 2].normalized : null;
    const n1i = idx + 1 < words.length ? words[idx + 1].normalized : null;
    const n2i = idx + 2 < words.length ? words[idx + 2].normalized : null;
    const wins = [];
    if (p1i && n1i) wins.push([p1i, n1i]);
    if (n1i && n2i) wins.push([n1i, n2i]);
    if (p2i && p1i) wins.push([null, p1i, p2i]);
    let candAttested = false;
    let origAttested = false;
    for (const w of wins) {
      const o = w.length === 3
        ? languageModel.trigramJoint(w[2], w[1], t.normalized)
        : languageModel.trigramJoint(w[0], t.normalized, w[1]);
      const c = w.length === 3
        ? languageModel.trigramJoint(w[2], w[1], best.word)
        : languageModel.trigramJoint(w[0], best.word, w[1]);
      if (c > 0) candAttested = true;
      if (o > 0) origAttested = true;
    }
    if (!candAttested || origAttested) {
      emit = false;
      gates.realWordProofFail = true;
    }
  }
  // Task 5: dictionary-valid tokens pulled INTO the cascade by the real-word
  // lane NEVER emit from the legacy unknown-token gates — SHADOW measures
  // them solely via the lane's own pairwise decision (below); ACTIVE decides
  // emission from that same calibrated decision. Without this suppression
  // every dictionary token bypasses the classified-DICTIONARY prefilter and
  // floods users with spelling warnings.
  if (emit && isRealWordTypo && rwMode !== 'OFF') {
    emit = false;
    gates.realWordLegacySuppressed = true;
  }

  // Task 5/9: DIFFERENT_KEY_REAL_WORD shadow decision via the SHARED lane
  // machinery — identical to the ACTIVE light path by construction.
  if (isRealWordTypo && rwMode !== 'OFF') {
    const { winner, rivals } = computeRealWordLaneDecision({
      builtEntries: built.entries ?? [],
      services, snap, languageModel, words, idx, original: t.normalized,
    });
    if (winner) {
      winner.rivals = rivals;
      shadowRealWordTypo = winner;
    }
  }

  const maxSuggestions = snap.get('linguistic.maxSpellingSuggestions');
  let suggestions = ranked.filter((r) => !r.isOriginal)
    .slice(0, maxSuggestions)
    .map((r) => r.word);

  const attMode = snap.get('spelling.attentionMode')
    ?? snap.get('linguistic.attentionMode') ?? 'OFF';
  let shadowAttention = null;

  if (attMode !== 'OFF' && services.attentionReranker && doc && cands.length > 0) {
    try {
      const t0 = performance.now();
      const reranker = services.attentionReranker;
      reranker.evaluatedCount = (reranker.evaluatedCount || 0) + 1;
      const vocab = getAttentionVocab();
      const units = unitsFromDocument(doc);
      const targetUnitIdx = units.findIndex((u) => u.start === t.start && u.end === t.end);
      if (targetUnitIdx >= 0) {
        const enc = encodeContextUnits(units, targetUnitIdx, vocab);
        const ctxIds = [...enc.ids];
        const ctxMask = [...enc.mask];
        const ctxCharHashes = enc.charHashes.map((hashes) => [...hashes]);
        while (ctxIds.length < 32) {
          ctxIds.push(0);
          ctxMask.push(0);
          ctxCharHashes.push([]);
        }

        const k = reranker.k || 8;
        attentionCands = selectDiverseShortlist(built.entries, {
          attestOf, famKey, size: k, cheapOrder: wideRanked, recallScoreOf,
        });
        const originalOption = encodeOptionSurface(t.normalized, vocab);
        const optIds = [originalOption.id];
        const optMask = [1];
        const optionCharHashes = [originalOption.charHashes];
        const classicalFeats = [new Array(15).fill(0)];
        classicalFeats[0][1] = 1.0;
        classicalFeats[0][10] = t.normalized.length;
        classicalFeats[0][11] = services.lexicon.contains(t.normalized) ? 1 : 0;
        classicalFeats[0][14] = classicalFeats[0][11];

        const rankMap = ranksForCandidates(attentionCands);
        for (let ci = 0; ci < Math.min(attentionCands.length, k); ci++) {
          const c = attentionCands[ci];
          const option = encodeOptionSurface(c.word, vocab);
          optIds.push(option.id);
          optionCharHashes.push(option.charHashes);
          optMask.push(1);

          const fVec = new Array(15).fill(0);
          const cf = services.lexicon.frequency(c.word) || 0;
          const ofq = services.lexicon.frequency(t.normalized) || 0;
          const feats = extractContextEvidence({
            languageModel: services.languageModel,
            words: words.map((w) => (typeof w === 'string' ? w : w.normalized)),
            idx,
            candidateWord: c.word,
            originalWord: t.normalized,
          });
          fVec[0] = feats.editDistance;
          fVec[1] = feats.sameAccentKey ? 1 : 0;
          fVec[2] = Math.log(1 + cf) - Math.log(1 + ofq);
          fVec[3] = feats.leftBigramLogRatio;
          fVec[4] = feats.rightBigramLogRatio;
          fVec[5] = feats.centeredTrigramLogRatio;
          fVec[6] = feats.forwardTrigramLogRatio;
          fVec[7] = feats.backwardTrigramLogRatio;
          fVec[8] = feats.candidateAttestedWindows;
          fVec[9] = feats.originalAttestedWindows;
          fVec[10] = t.normalized.length;
          fVec[11] = services.lexicon.contains(t.normalized) ? 1 : 0;
          fVec[12] = rankMap.get(c.word.toLowerCase())?.cheapRank ?? (ci + 1);
          fVec[13] = rankMap.get(c.word.toLowerCase())?.poolRank ?? (ci + 1);
          fVec[14] = services.lexicon.contains(c.word) ? 1 : 0;
          classicalFeats.push(fVec);
        }

        while (optIds.length < k + 1) {
          optIds.push(0);
          optMask.push(0);
          classicalFeats.push(new Array(15).fill(0));
          optionCharHashes.push([]);
        }

        const attRes = reranker.scoreOptions({
          wordIds: ctxIds,
          mask: ctxMask,
          targetPosition: enc.targetPosition,
          optionWordIds: optIds,
          optionMask: optMask,
          optionCharHashes,
          charHashes: ctxCharHashes,
          classicalFeatures: classicalFeats,
        });

        const elapsed = performance.now() - t0;
        const selIdx = attRes.selectedIndex;
        const selWord = selIdx === 0 ? t.normalized : (attentionCands[selIdx - 1]?.word ?? null);
        const classWord = (best && !best.isOriginal) ? best.word : t.normalized;

        let candWin = 0;
        let origWin = 0;
        let chosenWord = null;
        if (selIdx > 0 && attentionCands[selIdx - 1]) {
          const chosenCand = attentionCands[selIdx - 1];
          chosenWord = chosenCand.word;
          const feats = extractContextEvidence({
            languageModel: services.languageModel,
            words: words.map((w) => (typeof w === 'string' ? w : w.normalized)),
            idx,
            candidateWord: chosenCand.word,
            originalWord: t.normalized,
          });
          candWin = feats.candidateAttestedWindows ?? 0;
          origWin = feats.originalAttestedWindows ?? 0;
        }

        shadowAttention = {
          evaluated: true,
          mode: attMode,
          selectedOptionIndex: selIdx,
          selectedWord: selWord,
          chosenCandidateWord: chosenWord,
          probabilities: attRes.probabilities,
          logits: attRes.logits,
          confidence: attRes.probabilities[selIdx],
          agreementWithClassical: selWord === classWord,
          candidateAttestedWindows: candWin,
          originalAttestedWindows: origWin,
          tokenStart: t.start,
          tokenEnd: t.end,
          tokenOriginal: t.original,
          tokenNormalized: t.normalized,
          latencyMs: elapsed,
        };

        if (attMode === 'EXPERIMENTAL_ACTIVE') {
          const attMinProb = snap.get('spelling.attentionMinProbability')
            ?? snap.get('linguistic.attentionMinProbability') ?? 0.80;
          const attMinCandWin = snap.get('spelling.attentionMinCandidateWindows')
            ?? snap.get('linguistic.attentionMinCandidateWindows') ?? 1;
          const attMaxOrigWin = snap.get('spelling.attentionMaxOriginalWindows')
            ?? snap.get('linguistic.attentionMaxOriginalWindows') ?? 3;

          if (selIdx > 0) {
            const chosenCand = attentionCands[selIdx - 1];
            if (shadowAttention.confidence >= attMinProb && candWin >= attMinCandWin && origWin <= attMaxOrigWin) {
              emit = true;
              best = {
                word: chosenCand.word,
                p: shadowAttention.confidence,
                freq: services.lexicon.frequency(chosenCand.word) || 1,
                isOriginal: false,
              };
              suggestions = [
                chosenCand.word,
                ...suggestions.filter((s) => s.toLowerCase() !== chosenCand.word.toLowerCase()),
              ].slice(0, maxSuggestions);
              gates.attentionEmit = true;
            } else {
              emit = false;
              gates.attentionDeclined = true;
            }
          } else {
            emit = false;
            gates.attentionKeepOriginal = true;
          }
        }
      }
    } catch (err) {
      shadowAttention = { evaluated: false, error: err?.message ?? String(err) };
    }
  }

  const rejectionReason = emit ? null
    : (gates.attentionDeclined ? 'attention-threshold'
      : gates.attentionKeepOriginal ? 'attention-kept-original'
        : gates.confidenceFail ? 'confidence-fail'
          : gates.marginFail ? 'margin-fail'
            : gates.noNonOriginalWinner ? 'keep-original'
              : 'classical-gate-rejected');
  return {
    stage: 'decided',
    emit,
    gates,
    decision: {
      generatedWords: wideRanked.map((c) => String(c.word).toLowerCase()),
      consideredWords: cands.length,
      selectedWord: best?.isOriginal ? t.normalized.toLowerCase()
        : best?.word?.toLowerCase() ?? null,
      emitted: emit,
      rejectionReason,
      decisionOwner: attMode === 'EXPERIMENTAL_ACTIVE' && gates.attentionEmit
        ? 'attention-gated' : 'classical-unknown-lane',
    },
    poolWideCount: wideRanked.length,
    wideRankedWords: wideRanked.map((c) => c.word),
    cheapKept: cands.map((c) => c.word),
    ranked,
    best: best ?? null,
    second: second ?? null,
    suggestions,
    isRealWordTone,
    shadowWrongDiacritic,
    shadowRealWordTypo,
    shadowAttention,
  };
}

export function createPossibleSpellingErrorRule(services) {
  const { configService } = services;
  return {
    id: () => RuleIds.POSSIBLE_SPELLING_ERROR,
    priority: () => 600,
    supports: (ctx) => configService.snapshot().get('rules.spellingEnabled') !== false
      && configService.snapshot().get('linguistic.mode') !== 'OFF',

    validate: (ctx, doc) => {
      const snap = configService.snapshot();
      const issues = [];
      const words = doc.tokens.filter((t) => t.type === 'WORD');

      for (let idx = 0; idx < words.length; idx++) {
        const d = evaluateSpellingToken(services, snap, ctx, doc, words, idx);
        if (d.stage !== 'decided' || !d.emit) continue;
        const t = words[idx];
        issues.push(new ValidationIssue(
          RuleIds.POSSIBLE_SPELLING_ERROR, Severity.WARNING,
          t.start, t.end, t.original,
          `Từ "${t.original}" có thể là lỗi chính tả (gợi ý: "${d.best.word}").`,
          d.suggestions,
          Math.round(d.best.p * 100) / 100,
        ));
      }
      return issues;
    },
  };
}

/**
 * Task 8 — bounded word-boundary correction rule. SHADOW (default) computes
 * decisions for evaluation tooling but surfaces NOTHING; ACTIVE emits at
 * most ONE issue per originating token (highest-scoring wouldEmit item).
 */
export function createWordBoundaryRule(services) {
  const { configService } = services;
  return {
    id: () => RuleIds.POSSIBLE_SPELLING_ERROR,
    priority: () => 600,
    supports: () => configService.snapshot()
      .get('linguistic.wordBoundaryCorrectionMode') !== 'OFF',

    validate: (ctx, doc) => {
      const snap = configService.snapshot();
      if (snap.get('linguistic.wordBoundaryCorrectionMode') === 'OFF') {
        return [];
      }
      const issues = [];
      const words = doc.tokens.filter((t) => t.type === 'WORD');
      for (let idx = 0; idx < words.length; idx++) {
        const { splits, merges } = evaluateWordBoundaryCandidates({
          services, snap, languageModel: services.languageModel,
          ctx, doc, words, idx, classify: classifyToken,
        });
        const candidates = [...splits, ...merges]
          .filter((c) => c.wouldEmit)
          .sort((a, b) => b.score - a.score
            || a.suggestion.localeCompare(b.suggestion));
        const win = candidates[0];
        if (!win) continue;
        // SHADOW computes and reports upstream but NEVER emits
        if (snap.get('linguistic.wordBoundaryCorrectionMode') !== 'ACTIVE') {
          continue;
        }
        issues.push(new ValidationIssue(
          RuleIds.POSSIBLE_SPELLING_ERROR, Severity.WARNING,
          win.start, win.end, win.value,
          `"${win.value}" có thể cần tách/gộp từ (gợi ý: "${win.suggestion}").`,
          [win.suggestion],
          Math.round(win.score * 100) / 100,
        ));
      }
      return issues;
    },
  };
}
