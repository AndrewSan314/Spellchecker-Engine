// ============================================================
// Task 10 — Evaluate Attention Engine on Message Datasets
// (Internal-test / Dev).
//
// Computes baseline (OFF) vs evaluated mode (EXPERIMENTAL_ACTIVE / SHADOW),
// exact precision, recall, F0.5, deltas, transitions (new TP, removed TP),
// clean FP, candidate/shortlist misses, decision counters, and stage counters.
// ============================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';
import { evaluateSpellingToken } from '../src/rules/linguistic-rules.mjs';

export function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

export function sha256File(p) {
  try {
    if (!p || !existsSync(p)) return null;
    return createHash('sha256').update(readFileSync(p)).digest('hex');
  } catch {
    return null;
  }
}

/** Match emitted spelling issues to labels exactly once each. */
export function matchIssuesOneToOne(issues, labels) {
  const usedLabels = new Set();
  const matchedIssues = new Set();
  for (let issIdx = 0; issIdx < issues.length; issIdx++) {
    const iss = issues[issIdx];
    for (let labIdx = 0; labIdx < labels.length; labIdx++) {
      if (usedLabels.has(labIdx)) continue;
      const lab = labels[labIdx] ?? {};
      const valueMatch = normSurface(iss.value) === normSurface(lab.value);
      if (!valueMatch) continue;
      const hasSpan = Number.isInteger(lab.start) && Number.isInteger(lab.end);
      const spanMatch = hasSpan
        ? iss.start === lab.start && iss.end === lab.end
        : true;
      if (!spanMatch) continue;
      const wanted = Array.isArray(lab.suggestions) ? lab.suggestions : [];
      const suggestionMatch = wanted.length === 0
        || wanted.some((s) => (iss.suggestions || [])
          .some((isug) => normSurface(isug) === normSurface(s)));
      if (!suggestionMatch) continue;
      usedLabels.add(labIdx);
      matchedIssues.add(issIdx);
      break;
    }
  }
  return {
    tp: matchedIssues.size,
    fp: issues.length - matchedIssues.size,
    fn: labels.length - usedLabels.size,
    usedLabels,
    matchedIssues,
  };
}

export function evaluateOnMessages(engine, messages) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let totalIssues = 0;

  for (const m of messages) {
    if (m.recordType !== 'message-row' || typeof m.text !== 'string') continue;
    const ctx = new ValidationContext(m.text, MessageMode.ACCENTED, m.brand || 'TEST');
    const res = engine.validate(ctx);
    const issues = res.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_SPELLING_ERROR' || iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );
    totalIssues += issues.length;

    const labels = m.labels || [];
    const matched = matchIssuesOneToOne(issues, labels);
    tp += matched.tp;
    fp += matched.fp;
    fn += matched.fn;
  }

  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
  const f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)) || 0.0;

  return { tp, fp, fn, totalIssues, precision, recall, f05 };
}

export function evaluateComprehensive(evalEngine, baseEngine, messages, overrides) {
  let baseTP = 0, baseFP = 0, baseFN = 0, baseTotalIssues = 0, baseCleanFP = 0;
  let evalTP = 0, evalFP = 0, evalFN = 0, evalTotalIssues = 0, evalCleanFP = 0;
  let newTP = 0, removedClassicalTP = 0;

  let totalTokens = 0;
  let prefilterCount = 0;
  let evaluatedCount = 0;
  let emittedCount = 0;
  let declinedCount = 0;

  let keepDeclineCount = 0;
  let thresholdDeclineCount = 0;
  let wrongCandidateCount = 0;
  let attentionEmitCount = 0;
  let candidateMissCount = 0;
  let shortlistMissCount = 0;

  const evalSnap = evalEngine.configService.snapshot();

  for (const m of messages) {
    if (m.recordType !== 'message-row' || typeof m.text !== 'string') continue;
    const labels = m.labels || [];
    const isClean = labels.length === 0;

    // 1. Baseline validation
    const baseCtx = new ValidationContext(m.text, MessageMode.ACCENTED, m.brand || 'TEST');
    const baseRes = baseEngine.validate(baseCtx);
    const baseIssues = baseRes.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_SPELLING_ERROR' || iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );
    baseTotalIssues += baseIssues.length;
    const baseMatched = matchIssuesOneToOne(baseIssues, labels);
    baseTP += baseMatched.tp;
    baseFP += baseMatched.fp;
    baseFN += baseMatched.fn;
    if (isClean) baseCleanFP += baseIssues.length;

    // 2. Evaluated mode validation
    const evalCtx = new ValidationContext(m.text, MessageMode.ACCENTED, m.brand || 'TEST');
    const evalRes = evalEngine.validate(evalCtx);
    const evalIssues = evalRes.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_SPELLING_ERROR' || iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );
    evalTotalIssues += evalIssues.length;
    const evalMatched = matchIssuesOneToOne(evalIssues, labels);
    evalTP += evalMatched.tp;
    evalFP += evalMatched.fp;
    evalFN += evalMatched.fn;
    if (isClean) evalCleanFP += evalIssues.length;

    // 3. Transitions between baseline and eval
    for (let labIdx = 0; labIdx < labels.length; labIdx++) {
      const inBase = baseMatched.usedLabels.has(labIdx);
      const inEval = evalMatched.usedLabels.has(labIdx);
      if (!inBase && inEval) newTP++;
      if (inBase && !inEval) removedClassicalTP++;
    }

    // 4. Token-level decision instrumentation
    const doc = evalEngine.documentBuilder.build(evalCtx);
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    totalTokens += words.length;

    for (let idx = 0; idx < words.length; idx++) {
      const d = evaluateSpellingToken(evalEngine.services, evalSnap, evalCtx, doc, words, idx);
      if (d.stage === 'prefilter') {
        prefilterCount++;
      } else {
        evaluatedCount++;
        if (d.emit) {
          emittedCount++;
        } else {
          declinedCount++;
        }

        const shadow = d.shadowAttention;
        if (shadow && shadow.evaluated) {
          if (shadow.selectedOptionIndex === 0) {
            keepDeclineCount++;
          } else if (!d.emit) {
            thresholdDeclineCount++;
          } else {
            attentionEmitCount++;
          }

          // Check against gold labels if this word had a target
          const wordTok = words[idx];
          const goldLab = labels.find((l) => normSurface(l.value) === normSurface(wordTok.original)
            && (!Number.isInteger(l.start) || l.start === wordTok.start));
          if (goldLab && Array.isArray(goldLab.suggestions) && goldLab.suggestions.length > 0) {
            const goldTarget = normSurface(goldLab.suggestions[0]);
            const inPool = (d.decision?.generatedWords || []).some((w) => normSurface(w) === goldTarget);
            if (!inPool) {
              candidateMissCount++;
            } else {
              const inShortlist = (d.cheapKept || []).some((w) => normSurface(w) === goldTarget);
              if (!inShortlist) {
                shortlistMissCount++;
              } else if (shadow.selectedWord && normSurface(shadow.selectedWord) !== goldTarget) {
                wrongCandidateCount++;
              }
            }
          }
        }
      }
    }
  }

  const basePrecision = (baseTP + baseFP) > 0 ? baseTP / (baseTP + baseFP) : 0.0;
  const baseRecall = (baseTP + baseFN) > 0 ? baseTP / (baseTP + baseFN) : 0.0;
  const baseF05 = ((1.25 * basePrecision * baseRecall) / (0.25 * basePrecision + baseRecall)) || 0.0;

  const evalPrecision = (evalTP + evalFP) > 0 ? evalTP / (evalTP + evalFP) : 0.0;
  const evalRecall = (evalTP + evalFN) > 0 ? evalTP / (evalTP + evalFN) : 0.0;
  const evalF05 = ((1.25 * evalPrecision * evalRecall) / (0.25 * evalPrecision + evalRecall)) || 0.0;

  return {
    baseline: {
      tp: baseTP,
      fp: baseFP,
      fn: baseFN,
      totalIssues: baseTotalIssues,
      cleanFp: baseCleanFP,
      precision: basePrecision,
      recall: baseRecall,
      f05: baseF05,
    },
    evaluated: {
      tp: evalTP,
      fp: evalFP,
      fn: evalFN,
      totalIssues: evalTotalIssues,
      cleanFp: evalCleanFP,
      precision: evalPrecision,
      recall: evalRecall,
      f05: evalF05,
    },
    deltas: {
      precisionDelta: evalPrecision - basePrecision,
      recallDelta: evalRecall - baseRecall,
      f05Delta: evalF05 - baseF05,
      fpDelta: evalFP - baseFP,
      tpDelta: evalTP - baseTP,
      fnDelta: evalFN - baseFN,
    },
    transitions: {
      newTP,
      removedClassicalTP,
      netTP: newTP - removedClassicalTP,
    },
    oracleBreakdown: {
      candidateMisses: candidateMissCount,
      shortlistMisses: shortlistMissCount,
    },
    decisionBreakdown: {
      keepDeclines: keepDeclineCount,
      thresholdDeclines: thresholdDeclineCount,
      wrongCandidates: wrongCandidateCount,
      attentionEmits: attentionEmitCount,
    },
    stageCounters: {
      totalTokens,
      prefiltered: prefilterCount,
      evaluated: evaluatedCount,
      emitted: emittedCount,
      declined: declinedCount,
    },
  };
}

async function main() {
  const args = process.argv.slice(2);
  let messagesPath = '.tmp/attention-messages-internal-test.jsonl';
  let configPath = 'config/spelling-tuning.json';
  let mode = 'EXPERIMENTAL_ACTIVE';
  let outputReport = '.tmp/attention-internal-test-report.json';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--messages') messagesPath = args[++i];
    if (args[i] === '--config') configPath = args[++i];
    if (args[i] === '--mode') mode = args[++i];
    if (args[i] === '--output-report') outputReport = args[++i];
  }

  const rawMessages = readFileSync(messagesPath);
  const parsed = rawMessages.toString('utf8').split(/\r?\n/).filter(Boolean)
    .map((line) => JSON.parse(line));
  const sourceHeader = parsed[0]?.recordType === 'header' ? parsed[0] : null;
  const messages = parsed.filter((m) => m.recordType === 'message-row');

  // Load tuning config
  let tuning = {};
  if (existsSync(configPath)) {
    tuning = JSON.parse(readFileSync(configPath, 'utf8'));
  }
  const overrides = {
    attentionMode: mode,
    attentionMinProbability: 0.5,
    attentionMinCandidateWindows: 2,
    attentionMaxOriginalWindows: 1,
    ...(tuning.winner?.overrides || {}),
    ...(tuning.frozenParams || {}),
    attentionMode: mode,
  };

  // 1. Setup baseline OFF engine
  const baseConfigService = new ValidationConfigService();
  baseConfigService.reload({ spelling: { attentionMode: 'OFF' } });
  const baseEngine = new SmsValidationEngine({ configService: baseConfigService });

  // 2. Setup evaluated mode engine
  const evalConfigService = new ValidationConfigService();
  evalConfigService.reload({
    spelling: overrides,
  });
  const evalEngine = new SmsValidationEngine({ configService: evalConfigService });

  // 3. Compute comprehensive evaluation
  const results = evaluateComprehensive(evalEngine, baseEngine, messages, overrides);

  // 4. Exact hashes
  const modelBinPath = 'src/data/attention-reranker.int8.bin';
  const modelMetaPath = 'src/data/attention-reranker.json';
  const vocabPath = '.tmp/attention-vocab.json';
  const tokenizerPath = 'src/attention-tokenizer.mjs';

  const hashes = {
    ...(sourceHeader?.hashes ?? {}),
    dataset: createHash('sha256').update(rawMessages).digest('hex'),
    messages: createHash('sha256').update(rawMessages).digest('hex'),
    model: sha256File(modelBinPath),
    modelMeta: sha256File(modelMetaPath),
    vocab: sha256File(vocabPath),
    tokenizer: sha256File(tokenizerPath),
    config: createHash('sha256').update(JSON.stringify(overrides)).digest('hex'),
  };

  const report = {
    schema: 'attention-internal-test-report-v2',
    createdAt: new Date().toISOString(),
    messagesPath,
    messagesCount: messages.length,
    provenance: {
      hashes,
      sourceHeader,
    },
    mode,
    config: overrides,
    baseline: results.baseline,
    evaluated: results.evaluated,
    attention: results.evaluated,
    deltas: results.deltas,
    precisionDelta: results.deltas.precisionDelta,
    recallDelta: results.deltas.recallDelta,
    f05Delta: results.deltas.f05Delta,
    fpDelta: results.deltas.fpDelta,
    transitions: results.transitions,
    oracleBreakdown: results.oracleBreakdown,
    decisionBreakdown: results.decisionBreakdown,
    stageCounters: results.stageCounters,
    improved: results.deltas.f05Delta >= 0 && results.deltas.precisionDelta >= -0.02,
  };

  if (outputReport) {
    writeFileSync(outputReport, JSON.stringify(report, null, 2), 'utf8');
  }

  console.log(JSON.stringify({
    ok: true,
    baseline: report.baseline,
    evaluated: report.evaluated,
    deltas: report.deltas,
    transitions: report.transitions,
    stageCounters: report.stageCounters,
    oracleBreakdown: report.oracleBreakdown,
    decisionBreakdown: report.decisionBreakdown,
    reportPath: outputReport,
  }, null, 2));
}

if (process.argv[1] && process.argv[1].endsWith('evaluate_attention_messages.mjs')) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
