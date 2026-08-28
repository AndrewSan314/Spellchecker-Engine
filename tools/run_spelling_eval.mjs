// Task 8 — split-safe spelling evaluation workflow
// (spelling-engine-optimization plan).
//
//   node tools/run_spelling_eval.mjs --split dev            # tuning metric
//   node tools/run_spelling_eval.mjs --split test --final   # ONE post-freeze run
//
// Guards: default split is dev; test/external-test require --final; the
// report records code/config/artifact hashes so a run can be attributed to
// an exact source state. For every labeled correction the report classifies
// the failure stage using THE PRODUCTION DECISION FUNCTION
// (evaluateSpellingToken) — no mirrored logic:
//    prefilter / candidate-miss / cheap-ranker-loss / trigram-ranker-loss /
//    gate-rejected / wrong-suggestion / unsupported-multi-token
import { readFileSync, writeFileSync, renameSync, mkdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { RuleIds } from '../src/core.mjs';
import { accentKey } from '../src/normalizer.mjs';
import { adaptVsecSplit } from './spelling_benchmark_adapter_vsec.mjs';
import { evaluateSpellingToken, classifyToken } from '../src/rules/linguistic-rules.mjs';
import { evaluateWordBoundaryCandidates } from '../src/word-boundary-candidates.mjs';
import { classifyCorrectionRelation, linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';
import {
  createEvaluationHeader, writeEvaluationJsonl, sha256OfJson as computeConfigFingerprint,
} from '../src/evaluation-provenance.mjs';
import { RECALL_FEATURE_CONTRACT } from '../src/recall-reranker.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const TUNING_SPLITS = new Set(['dev', 'train']);
export const HELD_OUT_SPLITS = new Set(['test', 'external-test']);

/**
 * Task 8 Step 2 guard. Throws when a held-out split is requested without an
 * explicit --final flag; returns { allowed:true, final:boolean } otherwise.
 */
export function assertSplitAllowed(split, { final = false } = {}) {
  if (!HELD_OUT_SPLITS.has(split)) {
    if (!TUNING_SPLITS.has(split)) {
      throw new Error(`unknown split "${split}" (use dev|train|test|external-test)`);
    }
    return { allowed: true, final: false };
  }
  if (!final) {
    throw new Error(`REFUSED: split "${split}" is held out for the single `
      + 'post-freeze evaluation. Re-run with --final after configuration '
      + 'freeze, or use --split dev for tuning.');
  }
  return { allowed: true, final: true };
}

function parseArgs(argv) {
  const flags = {
    split: 'dev', final: false, out: null, wrongDiacriticMode: null,
    realWordTypoMode: null, dumpShadow: null,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--split') flags.split = argv[++i];
    else if (argv[i] === '--final') flags.final = true;
    else if (argv[i] === '--out') flags.out = argv[++i];
    else if (argv[i] === '--wrong-diacritic-mode') flags.wrongDiacriticMode = argv[++i];
    else if (argv[i] === '--real-word-typo-mode') flags.realWordTypoMode = argv[++i];
    else if (argv[i] === '--dump-shadow-decisions') flags.dumpShadow = argv[++i];
  }
  return flags;
}

const LANE_MODES = new Set(['OFF', 'SHADOW', 'ACTIVE']);

export function sha256File(p) {
  try {
    return createHash('sha256').update(readFileSync(p)).digest('hex');
  } catch {
    return null;
  }
}

/** atomic JSON write + sidecar SHA-256 manifest (shared with analyzers) */
export function writeArtifact(outPath, payload) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  const body = Buffer.from(JSON.stringify(payload, null, 2), 'utf8');
  const tmp = `${outPath}.tmp-${process.pid}`;
  writeFileSync(tmp, body);
  renameSync(tmp, outPath);
  const sha256 = createHash('sha256').update(body).digest('hex');
  writeFileSync(`${outPath}.manifest.json`, JSON.stringify({
    artifact: path.basename(outPath), bytes: body.length, sha256,
  }, null, 2));
  return sha256;
}

/** benchmark-compatible surface normalization */
function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

export function expectedPositionMatches(issue, expected) {
  if (!Number.isInteger(expected?.positionStart) || !Number.isInteger(expected?.positionEnd)) {
    return true;
  }
  return issue?.start >= expected.positionStart && issue?.end <= expected.positionEnd;
}

export function tokenIndexForExpected(words, expected) {
  if (Number.isInteger(expected?.positionStart) && Number.isInteger(expected?.positionEnd)) {
    return words.findIndex((word) => word.start >= expected.positionStart
      && word.end <= expected.positionEnd);
  }
  const errLower = normSurface(expected?.value).toLowerCase();
  const direct = words.findIndex((word) => word.normalized.toLowerCase() === errLower);
  return direct >= 0
    ? direct
    : words.findIndex((word) => accentKey(word.normalized) === accentKey(errLower));
}

/**
 * Task 1 (recall-improvement plan): single-source terminal-stage
 * attribution over one evaluateSpellingToken decision. Shared by the
 * evaluator and tools/analyze_recall_failures.mjs so both reports follow
 * the production cascade identically and can never drift apart.
 * @returns one of prefilter | candidate-miss | cheap-ranker-loss |
 *   gate-rejected | wrong-suggestion | suppressed-downstream
 */
export function stageFromDecision(d, targetLower) {
  // Task 1 (attention plan): prefer the explicit decision diagnostic shared
  // by the classical real-word light path and the attention path. Empty
  // ranked/cheapKept arrays on the light path carry NO stage information.
  const diag = d?.decision;
  if (diag && Array.isArray(diag.generatedWords)) {
    const target = String(targetLower ?? '').toLowerCase();
    if (!diag.generatedWords.includes(target)) {
      return 'candidate-miss';            // lost at generation
    }
    if (diag.emitted) {
      const selected = String(diag.selectedWord ?? '').toLowerCase();
      return selected === target
        ? 'correct'                       // decided correctly before any
                                          // downstream suppression logic
        : 'wrong-suggestion';             // emitted, but for another option
    }
    return 'gate-rejected';               // confidence/margin/windows gates
  }
  if (d.stage === 'prefilter') return 'prefilter';
  if (d.stage === 'no-candidates') return 'candidate-miss';
  const inWide = d.wideRankedWords.some((w) => w.toLowerCase() === targetLower);
  if (!inWide) {
    return 'candidate-miss';            // lost at generation
  }
  if (!d.ranked.some((r) => !r.isOriginal
    && r.word.toLowerCase() === targetLower)) {
    // generated but absent from the post-scoring candidate list:
    // either the cheap ranker cut it, or the minFreq gate dropped it
    return d.cheapKept.some((w) => w.toLowerCase() === targetLower)
      ? 'gate-rejected'                 // scored then freq-gated away
      : 'cheap-ranker-loss';
  }
  const bestIsTarget = Boolean(d.best && !d.best.isOriginal
    && d.best.word.toLowerCase() === targetLower);
  if (!d.emit) {
    // confidence/margin/frequency/real-word-proof gates (or original won)
    return 'gate-rejected';
  }
  if (!bestIsTarget) return 'wrong-suggestion'; // emitted, but for another word
  // decided to emit yet not caught -> removed by issue suppression /
  // conflict resolution downstream of the rule
  return 'suppressed-downstream';
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const access = assertSplitAllowed(flags.split, { final: flags.final });
  if (access.final) {
    console.error(`WARNING: --final evaluation on "${flags.split}". This `
      + 'counts as the single post-freeze external measurement.');
  }

  const engine = createDefaultEngine();
  // Task 4/5: lane-mode overrides. Both ride in ONE reload() — reload merges
  // onto the DEFAULT snapshot, so sequential reloads would silently drop the
  // earlier flag.
  const laneOverrides = {};
  if (flags.wrongDiacriticMode != null) {
    if (!LANE_MODES.has(flags.wrongDiacriticMode)) {
      throw new Error('invalid --wrong-diacritic-mode (use OFF|SHADOW|ACTIVE)');
    }
    laneOverrides.wrongDiacriticMode = flags.wrongDiacriticMode;
  }
  if (flags.realWordTypoMode != null) {
    if (!LANE_MODES.has(flags.realWordTypoMode)) {
      throw new Error('invalid --real-word-typo-mode (use OFF|SHADOW|ACTIVE)');
    }
    laneOverrides.realWordTypoMode = flags.realWordTypoMode;
  }
  if (Object.keys(laneOverrides).length > 0) {
    // reload BEFORE the snapshot so the whole run sees the lane modes
    engine.configService.reload({ linguistic: laneOverrides });
  }
  const snap = engine.configService.snapshot();
  const adapted = adaptVsecSplit(flags.split);

  // provenance: code/config/artifact hashes (plan Task 8 Step 2)
  const lmPath = path.join(ROOT, 'src', 'data', 'lm-ngrams.tsv');
  const provenance = {
    split: flags.split,
    final: access.final,
    node: process.version,
    rows: adapted.meta.rowCount,
    expectedIssues: adapted.meta.expectedIssueCount,
    hashes: {
      languageDotMjs: sha256File(path.join(ROOT, 'src', 'language.mjs')),
      linguisticRulesDotMjs: sha256File(
        path.join(ROOT, 'src', 'rules', 'linguistic-rules.mjs')),
      configDotMjs: sha256File(path.join(ROOT, 'src', 'config.mjs')),
      lmArtifact: sha256File(lmPath),
      lmBytes: (() => {
        try { return statSync(lmPath).size; } catch { return null; }
      })(),
    },
    configSnapshotKeys: {
      wrongDiacriticMode: snap.get('linguistic.wrongDiacriticMode'),
      wrongDiacriticModeOverride: flags.wrongDiacriticMode,
      realWordTypoMode: snap.get('linguistic.realWordTypoMode'),
      realWordTypoModeOverride: flags.realWordTypoMode,
      spellingCandidateKeys: snap.get('linguistic.spellingCandidateKeys'),
      spellingSurfacesPerKey: snap.get('linguistic.spellingSurfacesPerKey'),
      spellingCheapTopK: snap.get('linguistic.spellingCheapTopK'),
      spellingMinConfidence: snap.get('linguistic.spellingMinConfidence'),
      spellingMinMargin: snap.get('linguistic.spellingMinMargin'),
      spellingMinFrequency: snap.get('linguistic.spellingMinFrequency'),
      typoAlpha: snap.get('linguistic.typoAlpha'),
      typoBeta: snap.get('linguistic.typoBeta'),
      typoGamma: snap.get('linguistic.typoGamma'),
      originalPriorBonusSpelling: snap.get('linguistic.originalPriorBonusSpelling'),
    },
  };

  const stageCounts = {
    correct: 0,
    'correct-other-lane': 0,
    prefilter: 0,
    'candidate-miss': 0,
    'cheap-ranker-loss': 0,
    'trigram-ranker-loss': 0,
    'gate-rejected': 0,
    'wrong-suggestion': 0,
    'suppressed-downstream': 0,
    'unsupported-multi-token': 0,
    'label-token-not-found': 0,
  };
  const examples = [];
  // Task 1 (recall-improvement plan): relation-aware observability.
  // Every label increments EXACTLY ONE stage bucket and one relation-stage
  // cell, so totals reconcile to the label count by construction.
  const relationCounts = {};
  const relationStages = {};
  const prefilterReasons = {};
  // Every label increments EXACTLY ONE stage bucket via countStage, so
  // stage totals and per-relation stage cells always reconcile to the
  // label count by construction.
  const countStage = (rel, stage, reason = null) => {
    stageCounts[stage] += 1;
    relationStages[rel] ??= {};
    relationStages[rel][stage] = (relationStages[rel][stage] ?? 0) + 1;
    if (stage === 'prefilter' && reason != null) {
      prefilterReasons[reason] = (prefilterReasons[reason] ?? 0) + 1;
    }
  };
  let tp = 0;         // strict: spelling-rule issue matched value+suggestion
  let fpStrict = 0;   // strict: spelling-rule emissions not in labels
  let tpAny = 0;      // cross-lane: any linguistic issue matched value+suggestion
  let valueTp = 0;    // cross-lane: any linguistic issue matched the value
  let fp = 0;         // cross-lane: emitted linguistic issue not in labels
  let tpSemantic = 0; // Task 2: semantic linguistic credit (shared matcher)
  let fpSemantic = 0; // Task 2: linguistic emissions matching NO label semantically
  // Task 4: SHADOW wrong-diacritic headroom (uncaught labels only — caught
  // labels short-circuit before the lane decision runs).
  const shadowWD = {
    mode: flags.wrongDiacriticMode ?? snap.get('linguistic.wrongDiacriticMode'),
    decisions: 0, wouldEmit: 0, tp: 0, fp: 0,
    checkFailures: {},
    reachableNoPmdLine: 0,
  };
  // Task 5 Step 5: SHADOW real-word headroom (uncaught labels only — caught
  // labels short-circuit before the lane decision runs).
  const shadowRW = {
    mode: flags.realWordTypoMode ?? snap.get('linguistic.realWordTypoMode'),
    decisions: 0, wouldEmit: 0, tp: 0, fp: 0,
    checkFailures: {},
    reachableNoPmdLine: 0,
  };
  let labels = 0;
  const shadowDecisionRecords = [];
  // Task 8: SHADOW word-boundary headroom
  const shadowWB = {
    mode: snap.get('linguistic.wordBoundaryCorrectionMode'),
    decisions: 0, wouldEmit: 0, tp: 0, fp: 0,
  };

  for (const row of adapted.rows) {
    if (row.expect.length === 0) continue;
    const result = engine.validate(new ValidationContext(
      row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO'));
    const spellingIssues = result.issues.filter((i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR);
    // Cross-lane view: a correction is a correction wherever the engine
    // attributes it — PMD owns same-key unaccented tone fixes by design, so
    // strict spelling-only scoring would cap measurable recall at the size
    // of the typo-only slice. Both views are reported; tuning uses the
    // semantic one (Task 2). The semantic matcher is THE shared production
    // helper (linguisticCorrectionMatches) so this view can never drift.
    const linguisticIssues = result.issues.filter((i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
      || i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
    for (const issue of spellingIssues) {
      if (!row.expect.some((e) => issue.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
        && linguisticCorrectionMatches(issue, e))) fpStrict++;
    }
    for (const issue of linguisticIssues) {
      if (row.expect.some((e) => expectedPositionMatches(issue, e)
        && normSurface(issue.value) === normSurface(e.value))) valueTp++;
      else fp++;
      const semanticHit = row.expect.some((e) => linguisticCorrectionMatches(issue, e));
      if (!semanticHit) fpSemantic++;
    }

    const doc = engine.documentBuilder.build(new ValidationContext(
      row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO'));
    const words = doc.tokens.filter((t) => t.type === 'WORD');

    for (const exp of row.expect) {
      labels++;
      const targetRaw = String(exp.suggestions?.[0] ?? '');
      const rel = classifyCorrectionRelation(exp.value, targetRaw);
      relationCounts[rel] = (relationCounts[rel] ?? 0) + 1;
      if (/\s/.test(targetRaw.trim()) || targetRaw.trim() === '') {
        countStage(rel, 'unsupported-multi-token');
        continue;
      }
      const targetLower = targetRaw.toLowerCase();
      const byKey = tokenIndexForExpected(words, exp);
      if (byKey < 0) { countStage(rel, 'label-token-not-found'); continue; }

      // was the label caught by the ENGINE? Strict view = spelling-rule
      // issue only (official benchmark semantics); cross-lane view credits
      // any linguistic issue matching value + suggestion.
      const caughtStrict = spellingIssues.some((i) => linguisticCorrectionMatches(i, exp));
      const caughtAny = linguisticIssues.some((i) => linguisticCorrectionMatches(i, exp));
      // semantic credit via the shared matcher, evaluated INDEPENDENTLY of
      // the legacy predicates so view drift is detectable rather than hidden
      if (linguisticIssues.some((i) => linguisticCorrectionMatches(i, exp))) {
        tpSemantic++;
      }
      if (caughtAny) {
        countStage(rel, caughtStrict ? 'correct' : 'correct-other-lane');
        if (caughtStrict) tp++;
        tpAny++;
        continue;
      }

      const d = evaluateSpellingToken(engine.services, snap,
        new ValidationContext(row.text, row.mode ?? 'ACCENTED',
          row.brand ?? row.brandname ?? 'TENDOO'),
        doc, words, byKey);
      // follow the production cascade: wide generation -> cheap top-K ->
      // expensive scoring+gates -> emission
      const stage = stageFromDecision(d, targetLower);
      countStage(rel, stage,
        stage === 'prefilter' ? (d.reason ?? 'unknown') : null);

      // Task 4 Step 6: shadow headroom bookkeeping
      if (d.shadowWrongDiacritic) {
        const s = d.shadowWrongDiacritic;
        shadowWD.decisions++;
        if (s.wouldEmit) {
          shadowWD.wouldEmit++;
          if (normSurface(s.candidate) === normSurface(targetRaw)) shadowWD.tp++;
          else shadowWD.fp++;
          if (relationStages[rel]?.prefilter && rel === 'ACCENTED_SAME_KEY') {
            shadowWD.reachableNoPmdLine++;
          }
        } else {
          for (const [k, v] of Object.entries(s.checks)) {
            if (!v) shadowWD.checkFailures[k] = (shadowWD.checkFailures[k] ?? 0) + 1;
          }
        }
      }
      // Task 5 Step 6: real-word shadow bookkeeping. This lane reaches BOTH
      // different-key labels and dictionary-valid same-key labels ("đen" ->
      // "đến") that no other lane can touch, so reachability counts both
      // relations when the label sat in a production prefilter.
      if (d.shadowRealWordTypo) {
        const s = d.shadowRealWordTypo;
        shadowRW.decisions++;
        if (s.wouldEmit) {
          shadowRW.wouldEmit++;
          if (normSurface(s.candidate) === normSurface(targetRaw)) shadowRW.tp++;
          else shadowRW.fp++;
          if (relationStages[rel]?.prefilter
            && (rel === 'DIFFERENT_KEY_SINGLE_TOKEN'
              || rel === 'ACCENTED_SAME_KEY')) {
            shadowRW.reachableNoPmdLine++;
          }
        } else {
          for (const [k, v] of Object.entries(s.checks)) {
            if (!v) shadowRW.checkFailures[k] = (shadowRW.checkFailures[k] ?? 0) + 1;
          }
        }
      }
      // Task 6 Step 7: per-decision records for offline threshold replay
      // (Task 9 calibration sweeps these WITHOUT re-running the engine).
      if (d.shadowWrongDiacritic) {
        shadowDecisionRecords.push({
          lane: 'ACCENTED_SAME_KEY', rel,
          id: row.id, value: exp.value, target: targetRaw,
          candidate: d.shadowWrongDiacritic.candidate,
          wouldEmit: d.shadowWrongDiacritic.wouldEmit,
          evidenceScore: d.shadowWrongDiacritic.evidenceScore,
          modelProbability: d.shadowWrongDiacritic.modelProbability,
          checks: d.shadowWrongDiacritic.checks,
          features: d.shadowWrongDiacritic.features,
        });
      }
      if (d.shadowRealWordTypo) {
        shadowDecisionRecords.push({
          lane: 'DIFFERENT_KEY_REAL_WORD', rel,
          id: row.id, value: exp.value, target: targetRaw,
          candidate: d.shadowRealWordTypo.candidate,
          wouldEmit: d.shadowRealWordTypo.wouldEmit,
          evidenceScore: d.shadowRealWordTypo.evidenceScore,
          modelProbability: d.shadowRealWordTypo.modelProbability,
          checks: d.shadowRealWordTypo.checks,
          features: d.shadowRealWordTypo.features,
        });
      }
      // Task 8: word-boundary decisions attach to the label's token span
      const wb = evaluateWordBoundaryCandidates({
        services: engine.services, snap,
        languageModel: engine.services.languageModel,
        ctx: new ValidationContext(row.text, row.mode ?? 'ACCENTED',
          row.brand ?? row.brandname ?? 'TENDOO'),
        doc, words, idx: byKey, classify: classifyToken,
      });
      for (const item of [...wb.splits, ...wb.merges]) {
        shadowDecisionRecords.push({
          lane: 'WORD_BOUNDARY', rel,
          id: row.id, value: exp.value, target: targetRaw,
          candidate: item.suggestion,
          wouldEmit: item.wouldEmit,
          evidenceScore: item.score,
          modelProbability: null,
          checks: { wouldEmit: item.wouldEmit },
          features: {},
        });
        shadowWB.decisions++;
        if (item.wouldEmit) {
          shadowWB.wouldEmit++;
          const hit = normSurface(item.suggestion) === normSurface(targetRaw);
          if (hit) shadowWB.tp++; else shadowWB.fp++;
        }
      }
      if (examples.length < 60) {
        examples.push({
          id: row.id, value: exp.value, target: targetRaw,
          stage,
          best: d.best ? `${d.best.isOriginal ? '<orig>' : ''}${d.best.word}` : null,
          gates: d.gates ?? null,
          conf: d.best ? Math.round((d.best.p ?? 0) * 100) / 100 : null,
        });
      }
    }
  }

  const precision = tp / (tp + fpStrict || 1);
  const recall = tp / (labels || 1);
  const precisionAny = tpAny / (tpAny + fp || 1);
  const recallAny = tpAny / (labels || 1);
  const precisionSem = tpSemantic / (tpSemantic + fpSemantic || 1);
  const recallSem = tpSemantic / (labels || 1);
  const beta = 0.5;
  const f05Of = (p, r) => (1 + beta * beta) * p * r
    / (beta * beta * p + r || 1);
  // Task 2 Step 4: the three recall views, side by side.
  //   strictRuleId        — backwards-compatible contract score;
  //   semanticLinguistic  — PRIMARY product correction score (shared matcher);
  //   valueOnlyDiagnostic — diagnostic only, NEVER an acceptance metric.
  const views = {
    strictRuleId: {
      tpWithSuggestion: tp,
      fp: fpStrict,
      precision: Math.round(precision * 10000) / 10000,
      recall: Math.round(recall * 10000) / 10000,
      f05: Math.round(f05Of(precision, recall) * 10000) / 10000,
    },
    semanticLinguistic: {
      tpWithSuggestion: tpSemantic,
      fp: fpSemantic,
      precision: Math.round(precisionSem * 10000) / 10000,
      recall: Math.round(recallSem * 10000) / 10000,
      f05: Math.round(f05Of(precisionSem, recallSem) * 10000) / 10000,
    },
    valueOnlyDiagnostic: {
      tpValueOnly: valueTp,
      fp,
      precision: Math.round((valueTp / (valueTp + fp || 1)) * 10000) / 10000,
      recall: Math.round((valueTp / (labels || 1)) * 10000) / 10000,
      diagnosticOnly: true,
    },
  };
  const report = {
    ...provenance,
    metrics: {
      labels,
      views,
      // legacy aliases kept for existing consumers of this artifact
      strict: views.strictRuleId,
      crossLane: {
        tpWithSuggestion: tpAny,
        tpValueOnly: valueTp,
        fp,
        precision: views.semanticLinguistic.precision,
        recall: views.semanticLinguistic.recall,
        f05: views.semanticLinguistic.f05,
        precisionValueOnly: views.valueOnlyDiagnostic.precision,
      },
    },
    stages: stageCounts,
    // Task 1 (recall-improvement plan): lane observability blocks. Every
    // label lands in exactly one relation and one stage cell.
    relations: relationCounts,
    relationStages,
    prefilterReasons,
    shadowWrongDiacritic: shadowWD,
    shadowRealWordTypo: shadowRW,
    shadowWordBoundary: shadowWB,
    examples,
  };
  console.log(JSON.stringify(report, null, 2));
  if (flags.out) {
    const sha = writeArtifact(path.resolve(flags.out), report);
    console.error(`artifact written: ${flags.out} (sha256 ${sha.slice(0, 16)}…)`);
  }
  if (flags.dumpShadow) {
    // Task 1 (attention plan): dumps are self-describing. The first line
    // binds the file to exact source/artifact hashes, the effective config
    // fingerprint and the feature contract; consumers refuse stale files.
    const dumpHeader = createEvaluationHeader({
      schema: 'shadow-decisions-v2',
      split: flags.split,
      hashes: {
        linguisticRulesDotMjs: provenance.hashes.linguisticRulesDotMjs,
        configDotMjs: provenance.hashes.configDotMjs,
        lmNgramsTsv: provenance.hashes.lmArtifact,
        recallRerankerModelJson: sha256File(
          path.join(ROOT, 'src', 'data', 'recall-reranker.json')),
      },
      config: {
        configHash: computeConfigFingerprint(snap.get('linguistic')),
        featureContract: RECALL_FEATURE_CONTRACT,
      },
      createdBy: 'tools/run_spelling_eval.mjs',
    });
    mkdirSync(path.dirname(path.resolve(flags.dumpShadow)), { recursive: true });
    const { count } = writeEvaluationJsonl(
      path.resolve(flags.dumpShadow), dumpHeader, shadowDecisionRecords);
    console.error(`shadow decision records: ${count}`
      + ` -> ${flags.dumpShadow}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}
