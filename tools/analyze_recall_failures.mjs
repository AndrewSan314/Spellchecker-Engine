// Task 1 (recall-improvement plan) Step 5 — recall-failure analyzer.
//
//   node tools/analyze_recall_failures.mjs --split dev --out <path>
//
// Replaces the temporary PMD diagnostic. Every labeled correction is
// attributed by LINGUISTIC RELATION and terminal production stage using
// THE SAME PRODUCTION DECISION FUNCTIONS as the evaluator
// (engine.validate + classifyToken + evaluateSpellingToken +
// stageFromDecision) — no mirrored logic, no behavior change.
// Held-out splits stay behind --final via the shared split guard.
import path from 'node:path';
import { statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { RuleIds } from '../src/core.mjs';
import { accentKey } from '../src/normalizer.mjs';
import { adaptVsecSplit } from './spelling_benchmark_adapter_vsec.mjs';
import {
  classifyToken, evaluateSpellingToken,
} from '../src/rules/linguistic-rules.mjs';
import { classifyCorrectionRelation } from '../src/correction-taxonomy.mjs';
import {
  assertSplitAllowed, sha256File, writeArtifact, stageFromDecision,
} from './run_spelling_eval.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const MAX_EXAMPLES_PER_BUCKET = 20;

function parseArgs(argv) {
  const flags = { split: 'dev', final: false, out: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--split') flags.split = argv[++i];
    else if (argv[i] === '--final') flags.final = true;
    else if (argv[i] === '--out') flags.out = argv[++i];
  }
  return flags;
}

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function bump(obj, key) {
  obj[key] = (obj[key] ?? 0) + 1;
}

function rankKey(rank) {
  if (rank == null) return 'absent';
  return rank <= 3 ? `rank${rank}` : 'rank4plus';
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  assertSplitAllowed(flags.split, { final: flags.final });

  const engine = createDefaultEngine();
  const snap = engine.configService.snapshot();
  const adapted = adaptVsecSplit(flags.split);
  const lmPath = path.join(ROOT, 'src', 'data', 'lm-ngrams.tsv');

  const relationCounts = {};
  const relationStages = {};
  const stageCounts = {};
  const prefilterReasons = {};
  const pmdDecisions = {};
  const poolRanks = {
    wide: {},       // SymSpell+family generation pool
    cheap: {},      // post-cheap-ranker top-K
    expensive: {},  // post-scoring ranked list (non-original entries)
  };
  const examplesByRelationStage = {};
  const examplesByPmdDecision = {};
  let labels = 0;

  const countStage = (rel, stage, reason = null) => {
    bump(stageCounts, stage);
    relationStages[rel] ??= {};
    bump(relationStages[rel], stage);
    if (stage === 'prefilter' && reason != null) {
      bump(prefilterReasons, reason);
    }
  };
  const addExample = (store, bucket, entry) => {
    store[bucket] ??= [];
    if (store[bucket].length < MAX_EXAMPLES_PER_BUCKET) store[bucket].push(entry);
  };

  for (const row of adapted.rows) {
    if (row.expect.length === 0) continue;
    const vctx = () => new ValidationContext(
      row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO');
    const result = engine.validate(vctx());
    const spellingIssues = result.issues.filter(
      (i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR);
    const linguisticIssues = result.issues.filter((i) =>
      i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
      || i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);

    const doc = engine.documentBuilder.build(vctx());
    const words = doc.tokens.filter((t) => t.type === 'WORD');

    for (const exp of row.expect) {
      labels++;
      const targetRaw = String(exp.suggestions?.[0] ?? '');
      const rel = classifyCorrectionRelation(exp.value, targetRaw);
      bump(relationCounts, rel);

      if (/\s/.test(targetRaw.trim()) || targetRaw.trim() === '') {
        countStage(rel, 'unsupported-multi-token');
        continue;
      }
      const targetLower = targetRaw.toLowerCase();
      const errLower = normSurface(exp.value).toLowerCase();
      const tokIdx = words.findIndex((w) => w.normalized.toLowerCase() === errLower);
      const byKey = tokIdx >= 0
        ? tokIdx
        : words.findIndex((w) => accentKey(w.normalized) === accentKey(errLower));
      if (byKey < 0) { countStage(rel, 'label-token-not-found'); continue; }
      const t = words[byKey];

      // caught views (same semantics as the evaluator)
      const valueMatch = (i) => normSurface(i.value) === normSurface(exp.value);
      const suggMatch = (i) => (i.suggestions ?? [])
        .some((s) => normSurface(s) === normSurface(targetRaw));
      const caughtStrict = spellingIssues.some((i) => valueMatch(i) && suggMatch(i));
      const caughtAny = linguisticIssues.some((i) => valueMatch(i) && suggMatch(i));
      if (caughtAny) {
        countStage(rel, caughtStrict ? 'correct' : 'correct-other-lane');
        continue;
      }

      // ---- PMD decision attribution (production functions only) --------
      const cls = classifyToken(t, vctx(), doc, engine.services);
      const wrongToneEnabled = snap.get('linguistic.wrongToneEnabled') === true;
      const inPmdPipeline = cls === 'UNKNOWN' || cls === 'UNACCENTED_VALID'
        || (cls === 'DICTIONARY' && wrongToneEnabled);
      const pmdIssue = linguisticIssues.find((i) =>
        i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC && valueMatch(i));
      let pmdDecision;
      if (pmdIssue) {
        pmdDecision = suggMatch(pmdIssue)
          ? 'pmd-emitted-target'
          : `pmd-emitted-other:${normSurface(pmdIssue.suggestions?.[0] ?? '?')}`;
      } else if (!inPmdPipeline) {
        pmdDecision = `not-in-pipeline-classified-${cls}`;
      } else {
        const rivals = engine.services.accentIndex
          .candidates(accentKey(t.normalized))
          .filter((c) => c.word.toLowerCase() !== t.normalized);
        pmdDecision = rivals.length === 0
          ? 'no-competing-surface'
          : 'gate-declined';
      }
      bump(pmdDecisions, pmdDecision);

      // ---- spelling-lane cascade + pool ranks ---------------------------
      const d = evaluateSpellingToken(engine.services, snap, vctx(),
        doc, words, byKey);
      const stage = d.stage === 'prefilter'
        ? 'prefilter'
        : (d.stage === 'no-candidates'
          ? 'candidate-miss'
          : stageFromDecision(d, targetLower));

      countStage(rel, stage,
        stage === 'prefilter' ? (d.reason ?? 'unknown') : null);

      if (d.stage === 'decided') {
        bump(poolRanks.wide, rankKey(
          d.wideRankedWords.findIndex((w) => w.toLowerCase() === targetLower) >= 0
            ? d.wideRankedWords.findIndex((w) => w.toLowerCase() === targetLower) + 1
            : null));
        bump(poolRanks.cheap, rankKey(
          d.cheapKept.findIndex((w) => w.toLowerCase() === targetLower) >= 0
            ? d.cheapKept.findIndex((w) => w.toLowerCase() === targetLower) + 1
            : null));
        const rIdx = d.ranked.findIndex((r) => !r.isOriginal
          && r.word.toLowerCase() === targetLower);
        bump(poolRanks.expensive, rankKey(rIdx >= 0 ? rIdx + 1 : null));
      }

      const example = {
        id: row.id, value: exp.value, target: targetRaw,
        stage, reason: d.reason ?? null,
        pmdDecision, classified: cls,
        best: d.best ? `${d.best.isOriginal ? '<orig>' : ''}${d.best.word}` : null,
      };
      addExample(examplesByRelationStage, `${rel}|${stage}`, example);
      addExample(examplesByPmdDecision, pmdDecision, example);
    }
  }

  // reconciliation invariant: every label lands in EXACTLY one stage cell
  const staged = Object.values(stageCounts).reduce((a, b) => a + b, 0);
  if (staged !== labels) {
    throw new Error(`reconciliation FAILED: ${staged} staged vs ${labels} labels`);
  }

  const report = {
    split: flags.split,
    node: process.version,
    rows: adapted.meta.rowCount,
    expectedIssues: adapted.meta.expectedIssueCount,
    labels,
    relations: relationCounts,
    relationStages,
    stages: stageCounts,
    prefilterReasons,
    pmdDecisions,
    poolRanks,
    examples: {
      byRelationStage: examplesByRelationStage,
      byPmdDecision: examplesByPmdDecision,
    },
    hashes: {
      languageDotMjs: sha256File(path.join(ROOT, 'src', 'language.mjs')),
      linguisticRulesDotMjs: sha256File(path.join(ROOT, 'src', 'rules', 'linguistic-rules.mjs')),
      correctionTaxonomyDotMjs: sha256File(path.join(ROOT, 'src', 'correction-taxonomy.mjs')),
      runSpellingEvalDotMjs: sha256File(path.join(HERE, 'run_spelling_eval.mjs')),
      configDotMjs: sha256File(path.join(ROOT, 'src', 'config.mjs')),
      lmArtifact: sha256File(lmPath),
      lmBytes: (() => { try { return statSync(lmPath).size; } catch { return null; } })(),
    },
    configSnapshotKeys: {
      wrongToneEnabled: snap.get('linguistic.wrongToneEnabled'),
      spellingCandidateKeys: snap.get('linguistic.spellingCandidateKeys'),
      spellingSurfacesPerKey: snap.get('linguistic.spellingSurfacesPerKey'),
      spellingCheapTopK: snap.get('linguistic.spellingCheapTopK'),
      spellingMinConfidence: snap.get('linguistic.spellingMinConfidence'),
      spellingMinMargin: snap.get('linguistic.spellingMinMargin'),
      spellingMinFrequency: snap.get('linguistic.spellingMinFrequency'),
    },
  };

  console.log(JSON.stringify({
    labels: report.labels,
    relations: report.relations,
    stages: report.stages,
    prefilterReasons: report.prefilterReasons,
    pmdDecisions: report.pmdDecisions,
    poolRanks: report.poolRanks,
  }, null, 2));
  if (flags.out) {
    const sha = writeArtifact(path.resolve(flags.out), report);
    console.error(`artifact written: ${flags.out} (sha256 ${sha.slice(0, 16)}…)`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}
