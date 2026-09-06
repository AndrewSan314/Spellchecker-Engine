// SMS-REAL-200 evaluation for the shipped Tiny Attention Reranker.
// Benchmark/instrumentation only: no training, artifact generation, candidate
// changes, threshold search, or production configuration changes.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { SmsValidationEngine, ValidationContext } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { MessageMode } from '../src/core.mjs';
import {
  classifyToken,
  evaluateSpellingToken,
} from '../src/rules/linguistic-rules.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_DATASET = path.resolve(ROOT, '..', 'SMS real 200', 'corpus-sms-real-200.json');
const DEFAULT_OUT = path.join(ROOT, 'benchmark', 'sms-real-200-attention');
const ARMS = ['OFF', 'SHADOW', 'ACTIVE'];
const ATTENTION_MODES = Object.freeze({
  OFF: 'OFF',
  SHADOW: 'SHADOW',
  ACTIVE: 'EXPERIMENTAL_ACTIVE',
});
const EXPECTED_CATEGORIES = Object.freeze({
  CLEAN: 40,
  ALL_UNACCENTED: 40,
  SOME_UNACCENTED: 30,
  WRONG_DIACRITIC: 25,
  TYPO_TELEX: 25,
  PROTECTED_ENTITY: 20,
  CODE_SWITCH: 10,
  ADVERSARIAL_MINIMAL_PAIR: 10,
});
const LEXICAL_RULES = new Set([
  'POSSIBLE_SPELLING_ERROR',
  'POSSIBLE_MISSING_DIACRITIC',
]);

function argValue(args, name, fallback) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] != null ? args[i + 1] : fallback;
}

function hasArg(args, name) {
  return args.includes(name);
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function sha256Buffer(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function sha256File(file) {
  return sha256Buffer(fs.readFileSync(file));
}

function round(value, digits = 6) {
  if (value == null || !Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function normalizeSurface(value) {
  return String(value ?? '')
    .normalize('NFC')
    .toLocaleLowerCase('vi-VN')
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function overlaps(aStart, aEnd, bStart, bEnd) {
  return aStart < bEnd && bStart < aEnd;
}

function serializeIssue(issue) {
  return {
    ruleId: issue.ruleId,
    severity: issue.severity,
    start: issue.start,
    end: issue.end,
    value: issue.value,
    suggestions: [...(issue.suggestions ?? [])],
    confidence: issue.confidence ?? null,
  };
}

function issueSignature(issue) {
  return JSON.stringify({
    ruleId: issue.ruleId,
    severity: issue.severity,
    start: issue.start,
    end: issue.end,
    value: issue.value,
    suggestions: issue.suggestions ?? [],
  });
}

function matchFinalIssues(issues, labels) {
  const usedIssues = new Set();
  const labelMatches = labels.map(() => null);
  const candidatesFor = (label) => issues.map((issue, index) => {
    const spanExact = issue.start === label.positionStart && issue.end === label.positionEnd;
    const valueMatch = normalizeSurface(issue.value) === normalizeSurface(label.value);
    const spanOverlap = overlaps(issue.start, issue.end, label.positionStart, label.positionEnd);
    return {
      issue, index, spanExact, valueMatch, spanOverlap,
      suggestionMatch: (issue.suggestions ?? []).some((s) =>
        normalizeSurface(s) === normalizeSurface(label.suggestion)),
    };
  }).filter((x) => x.valueMatch && (x.spanExact || x.spanOverlap) && !usedIssues.has(x.index))
    .sort((a, b) => Number(b.spanExact) - Number(a.spanExact)
      || Number(b.suggestionMatch) - Number(a.suggestionMatch)
      || a.index - b.index);

  for (let labelIndex = 0; labelIndex < labels.length; labelIndex++) {
    const hit = candidatesFor(labels[labelIndex])[0];
    if (!hit) continue;
    usedIssues.add(hit.index);
    labelMatches[labelIndex] = {
      issueIndex: hit.index,
      detected: true,
      corrected: hit.suggestionMatch,
    };
  }
  return {
    labelMatches,
    matchedIssueIndexes: usedIssues,
    tp: usedIssues.size,
    fp: issues.length - usedIssues.size,
    fn: labels.length - usedIssues.size,
    correct: labelMatches.filter((x) => x?.corrected).length,
  };
}

function findToken(trace, label) {
  const exact = trace.find((token) => token.start === label.positionStart
    && token.end === label.positionEnd);
  if (exact) return exact;
  return trace.filter((token) => overlaps(token.start, token.end,
    label.positionStart, label.positionEnd)
    && normalizeSurface(token.token) === normalizeSurface(label.value))
    .sort((a, b) => (b.end - b.start) - (a.end - a.start))[0] ?? null;
}

function percentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return round(sorted[index], 4);
}

function median(values) {
  return percentile(values, 50);
}

function csvCell(value) {
  if (value == null) return '';
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(headers, rows) {
  return [headers, ...rows].map((row) => row.map(csvCell).join(',')).join('\n') + '\n';
}

function validateDataset(rows) {
  if (!Array.isArray(rows) || rows.length !== 200) {
    throw new Error(`SMS-REAL-200 mismatch: expected 200 messages, got ${rows?.length}`);
  }
  const counts = Object.fromEntries(Object.keys(EXPECTED_CATEGORIES).map((category) => [
    category, rows.filter((row) => row.category === category).length,
  ]));
  if (JSON.stringify(counts) !== JSON.stringify(EXPECTED_CATEGORIES)) {
    throw new Error(`SMS-REAL-200 category mismatch: ${JSON.stringify(counts)}`);
  }
  for (const row of rows) {
    if (typeof row.id !== 'string' || typeof row.content !== 'string'
      || !Array.isArray(row.expect) || !Array.isArray(row.protected_spans)) {
      throw new Error(`Invalid SMS-REAL-200 row: ${row?.id}`);
    }
  }
  return counts;
}

function effectiveConfig(configService) {
  const snap = configService.snapshot();
  const keys = [
    'attentionMode', 'attentionMinProbability', 'attentionMinCandidateWindows',
    'attentionMaxOriginalWindows', 'spellingShortlistSize', 'spellingCandidateKeys',
    'spellingPoolMax', 'realWordTypoMode', 'wrongDiacriticMode',
    'wordBoundaryCorrectionMode', 'verifiedConfidenceFloor',
  ];
  return Object.fromEntries(keys.map((key) => [
    key, snap.get(`spelling.${key}`) ?? snap.get(`linguistic.${key}`),
  ]));
}

function createAttentionSnapshot() {
  const service = new ValidationConfigService();
  service.reload({ spelling: { attentionMode: ATTENTION_MODES.ACTIVE } });
  return service.snapshot();
}

function installCapture(engine) {
  const reranker = engine.attentionReranker;
  const calls = [];
  if (!reranker) return { calls, disable() {}, enable() {} };
  const original = reranker.scoreOptions;
  const wrapped = function wrappedScoreOptions(args) {
    const start = performance.now();
    const result = original.call(this, args);
    calls.push({
      latencyMs: performance.now() - start,
      selectedIndex: result.selectedIndex,
      probabilities: [...(result.probabilities ?? [])],
      logits: [...(result.logits ?? [])],
    });
    return result;
  };
  reranker.scoreOptions = wrapped;
  return {
    calls,
    disable() { reranker.scoreOptions = original; },
    enable() { reranker.scoreOptions = wrapped; },
  };
}

function traceToken({ word, classification, diagnostic, actualCall, ordinal }) {
  const attention = diagnostic.shadowAttention;
  const generatedWords = diagnostic.decision?.generatedWords ?? [];
  const shortlist = diagnostic.cheapKept ?? [];
  return {
    token: word.original,
    start: word.start,
    end: word.end,
    class: classification,
    stage: diagnostic.stage ?? null,
    prefilterReason: diagnostic.reason ?? null,
    generatedCount: generatedWords.length,
    generatedWords,
    shortlistCount: shortlist.length,
    shortlist,
    selected: attention?.selectedWord ?? diagnostic.decision?.selectedWord ?? null,
    selectedProbability: attention?.confidence ?? null,
    keepProbability: attention?.probabilities?.[0] ?? null,
    confidence: attention?.confidence ?? null,
    logits: attention?.logits ?? null,
    probabilities: attention?.probabilities ?? null,
    attentionEvaluated: Boolean(attention?.evaluated),
    attentionLatencyMs: actualCall?.latencyMs ?? attention?.latencyMs ?? null,
    attentionCallOrdinal: attention?.evaluated ? ordinal : null,
    attentionModeDecision: diagnostic.decision?.rejectionReason ?? null,
    decisionOwner: diagnostic.decision?.decisionOwner ?? null,
    emitDecision: diagnostic.emit ?? false,
    gates: diagnostic.gates ?? {},
    shadowAttention: attention ? {
      lane: attention.lane,
      lanes: attention.lanes,
      tokenNormalized: attention.tokenNormalized,
      generatedCount: attention.generatedCount,
      shortlistSize: attention.shortlistSize,
      selectedOptionIndex: attention.selectedOptionIndex,
    } : null,
  };
}

function runArm(rows, arm) {
  const configService = new ValidationConfigService();
  configService.reload({ spelling: { attentionMode: ATTENTION_MODES[arm] } });
  const rssBeforeLoad = process.memoryUsage().rss;
  globalThis.gc?.();
  const startLoad = performance.now();
  const engine = new SmsValidationEngine({ configService });
  const coldStartMs = performance.now() - startLoad;
  const rssAfterLoad = process.memoryUsage().rss;
  const capture = installCapture(engine);
  const activeSnapshot = arm === 'OFF' ? null : createAttentionSnapshot();
  const warmupCount = Math.min(3, rows.length);

  for (let i = 0; i < warmupCount; i++) {
    const row = rows[i];
    engine.validate(new ValidationContext(row.content, row.messageMode, row.domain ?? 'SMS-REAL-200'));
  }
  capture.calls.length = 0;
  globalThis.gc?.();

  const messages = [];
  for (const row of rows) {
    const context = new ValidationContext(row.content, row.messageMode, row.domain ?? 'SMS-REAL-200');
    const callStart = capture.calls.length;
    const t0 = performance.now();
    const result = engine.validate(context);
    const engineMs = performance.now() - t0;
    const calls = capture.calls.slice(callStart);

    capture.disable();
    const doc = engine.documentBuilder.build(context);
    const words = doc.tokens.filter((token) => token.type === 'WORD');
    const trace = [];
    let attentionOrdinal = 0;
    for (let index = 0; index < words.length; index++) {
      const servingDiagnostic = evaluateSpellingToken(
        engine.services,
        configService.snapshot(),
        context,
        doc,
        words,
        index,
      );
      // SHADOW must be attributed to the exact serving path. Its attention
      // candidate names are recovered from ACTIVE only after SHADOW actually
      // evaluated this token; this does not alter the timed serving run.
      const diagnostic = arm === 'SHADOW' && servingDiagnostic.shadowAttention?.evaluated
        ? evaluateSpellingToken(
          engine.services, activeSnapshot, context, doc, words, index,
        )
        : servingDiagnostic;
      const classification = classifyToken(words[index], context, doc, engine.services);
      const actualCall = diagnostic.shadowAttention?.evaluated
        ? calls[attentionOrdinal++] ?? null : null;
      trace.push(traceToken({
        word: words[index], classification, diagnostic, actualCall,
        ordinal: actualCall ? attentionOrdinal - 1 : null,
      }));
    }
    capture.enable();

    messages.push({
      id: row.id,
      category: row.category,
      domain: row.domain,
      content: row.content,
      clean: row.clean,
      expected: row.expect,
      protectedSpans: row.protected_spans,
      finalIssues: result.issues.map(serializeIssue),
      shadowIssues: result.shadowIssues.map(serializeIssue),
      engineMs: round(engineMs, 6),
      attentionCalls: calls.length,
      attentionTotalMs: round(calls.reduce((sum, call) => sum + call.latencyMs, 0), 6),
      attentionCallMismatch: calls.length - attentionOrdinal,
      tokenCount: words.length,
      tokenTrace: trace,
    });
  }

  const engineLatencies = messages.map((message) => message.engineMs);
  const attentionPerMessage = messages.map((message) => message.attentionCalls);
  const attentionMsPerMessage = messages.map((message) => message.attentionTotalMs);
  const shortlistCandidatesPerMessage = messages.map((message) => message.tokenTrace
    .filter((token) => token.attentionEvaluated)
    .reduce((sum, token) => sum + token.shortlistCount, 0));
  const totalAttentionCalls = attentionPerMessage.reduce((sum, value) => sum + value, 0);
  const totalAttentionMs = attentionMsPerMessage.reduce((sum, value) => sum + value, 0);
  const totalShortlistCandidates = shortlistCandidatesPerMessage
    .reduce((sum, value) => sum + value, 0);

  return {
    arm,
    attentionMode: ATTENTION_MODES[arm],
    effectiveConfig: effectiveConfig(configService),
    runtime: {
      coldStartMs: round(coldStartMs, 4),
      rssBeforeLoadBytes: rssBeforeLoad,
      rssAfterLoadBytes: rssAfterLoad,
      engine: {
        p50Ms: median(engineLatencies),
        p95Ms: percentile(engineLatencies, 95),
        p99Ms: percentile(engineLatencies, 99),
      },
      attention: {
        callsPerMessage: {
          mean: round(totalAttentionCalls / messages.length, 6),
          median: median(attentionPerMessage),
          p95: percentile(attentionPerMessage, 95),
          max: Math.max(...attentionPerMessage),
        },
        totalMsPerMessage: {
          mean: round(totalAttentionMs / messages.length, 6),
          median: median(attentionMsPerMessage),
          p95: percentile(attentionMsPerMessage, 95),
          max: round(Math.max(...attentionMsPerMessage), 6),
        },
        msPerCall: totalAttentionCalls ? round(totalAttentionMs / totalAttentionCalls, 6) : 0,
        candidateTokensEvaluatedPerMessage: {
          mean: round(totalShortlistCandidates / messages.length, 6),
          median: median(shortlistCandidatesPerMessage),
          p95: percentile(shortlistCandidatesPerMessage, 95),
          max: Math.max(...shortlistCandidatesPerMessage),
        },
        totalCalls: totalAttentionCalls,
        totalMs: round(totalAttentionMs, 6),
        traceCallMismatches: messages.reduce((sum, message) =>
          sum + Math.abs(message.attentionCallMismatch), 0),
      },
      environment: {
        node: process.version,
        platform: process.platform,
        release: os.release(),
        cpu: os.cpus()[0]?.model ?? 'unknown',
        engineProfile: process.env.ENGINE_PROFILE ?? 'default',
        warmupCount,
      },
    },
    messages,
  };
}

function attachEvaluation(armData) {
  return armData.messages.map((message) => {
    const lexicalIssues = message.finalIssues.filter((issue) => LEXICAL_RULES.has(issue.ruleId));
    const detection = matchFinalIssues(lexicalIssues, message.expected);
    const labelEvaluations = message.expected.map((label, index) => {
      const match = detection.labelMatches[index];
      const issue = match ? lexicalIssues[match.issueIndex] : null;
      return {
        label,
        detected: Boolean(match),
        corrected: Boolean(match?.corrected),
        issue: issue ?? null,
      };
    });
    const cleanFalsePositiveIssues = message.category === 'CLEAN' ? lexicalIssues.length : 0;
    const protectedViolationSpans = new Set();
    for (let index = 0; index < message.protectedSpans.length; index++) {
      const span = message.protectedSpans[index];
      if (message.finalIssues.some((issue) => overlaps(issue.start, issue.end, span.start, span.end))) {
        protectedViolationSpans.add(index);
      }
    }
    return {
      ...message,
      lexicalIssues,
      detection,
      labelEvaluations,
      cleanFalsePositiveIssues,
      protectedViolationCount: protectedViolationSpans.size,
    };
  });
}

function classifyAttribution(message, labelIndex, arm) {
  const evaluation = message.labelEvaluations[labelIndex];
  const label = evaluation.label;
  const token = findToken(message.tokenTrace, label);
  const finalEmitted = evaluation.detected;
  const finalCorrect = evaluation.corrected;
  const base = {
    id: message.id,
    category: message.category,
    input: message.content,
    error: label.value,
    expected: label.suggestion,
    start: label.positionStart,
    end: label.positionEnd,
    tokenClass: token?.class ?? null,
    prefilterReason: null,
    candidatePoolSize: null,
    goldInPool: null,
    goldPoolRank: null,
    shortlistSize: null,
    goldInShortlist: null,
    goldShortlistRank: null,
    selectedWord: null,
    selectedProbability: null,
    keepProbability: null,
    finalEmitted,
    finalCorrect,
    failureStage: null,
  };
  if (arm === 'OFF') {
    base.prefilterReason = 'attention-mode-off';
    base.failureStage = finalCorrect ? 'CORRECT_EMIT' : 'PREFILTERED';
    return base;
  }
  if (!token) {
    base.prefilterReason = 'no-word-token';
    base.failureStage = 'PREFILTERED';
    return base;
  }
  base.prefilterReason = token.prefilterReason ?? token.attentionModeDecision ?? null;
  base.candidatePoolSize = token.generatedCount;
  const gold = normalizeSurface(label.suggestion);
  const poolIndex = token.generatedWords.findIndex((word) => normalizeSurface(word) === gold);
  const shortlistIndex = token.shortlist.findIndex((word) => normalizeSurface(word) === gold);
  base.goldInPool = poolIndex >= 0;
  base.goldPoolRank = poolIndex >= 0 ? poolIndex + 1 : null;
  base.shortlistSize = token.shortlistCount;
  base.goldInShortlist = shortlistIndex >= 0;
  base.goldShortlistRank = shortlistIndex >= 0 ? shortlistIndex + 1 : null;
  base.selectedWord = token.selected;
  base.selectedProbability = token.selectedProbability;
  base.keepProbability = token.keepProbability;

  if (token.stage === 'prefilter') {
    base.failureStage = finalCorrect ? 'CORRECT_EMIT' : 'PREFILTERED';
  } else if (!base.goldInPool || token.generatedCount === 0) {
    base.failureStage = 'CANDIDATE_MISS';
  } else if (!base.goldInShortlist) {
    base.failureStage = 'SHORTLIST_MISS';
  } else if (!token.selected || normalizeSurface(token.selected) === normalizeSurface(token.token)) {
    base.failureStage = 'MODEL_KEEP_ERROR';
  } else if (normalizeSurface(token.selected) !== gold) {
    base.failureStage = 'MODEL_WRONG_CANDIDATE';
  } else if (finalCorrect) {
    base.failureStage = 'CORRECT_EMIT';
  } else {
    base.failureStage = 'CORRECT_RANK_BUT_GATE_REJECT';
  }
  return base;
}

function metricsFor(armData, evaluations) {
  const allLabels = evaluations.flatMap((message) => message.expected);
  const tp = evaluations.reduce((sum, message) => sum + message.detection.tp, 0);
  const fp = evaluations.reduce((sum, message) => sum + message.detection.fp, 0);
  const fn = evaluations.reduce((sum, message) => sum + message.detection.fn, 0);
  const corrected = evaluations.reduce((sum, message) => sum + message.detection.correct, 0);
  const cleanMessages = evaluations.filter((message) => message.category === 'CLEAN');
  const protectedSpans = evaluations.reduce((sum, message) => sum + message.protectedSpans.length, 0);
  const protectedViolations = evaluations.reduce((sum, message) =>
    sum + message.protectedViolationCount, 0);
  const lexicalIssues = evaluations.reduce((sum, message) => sum + message.lexicalIssues.length, 0);
  const stageRows = evaluations.flatMap((message) => message.labelEvaluations
    .map((_, index) => classifyAttribution(message, index, armData.arm)));
  const stageCounts = Object.fromEntries([
    'PREFILTERED', 'CANDIDATE_MISS', 'SHORTLIST_MISS', 'MODEL_KEEP_ERROR',
    'MODEL_WRONG_CANDIDATE', 'CORRECT_RANK_BUT_GATE_REJECT', 'CORRECT_EMIT',
  ].map((stage) => [stage, stageRows.filter((row) => row.failureStage === stage).length]));
  const evaluatedRows = stageRows.filter((row) => row.failureStage !== 'PREFILTERED'
    && row.failureStage !== 'CORRECT_EMIT');
  const poolRows = stageRows.filter((row) => row.goldInPool != null);
  const shortlistRows = stageRows.filter((row) => row.goldInShortlist != null);
  const goldInShortlistRows = stageRows.filter((row) => row.goldInShortlist === true);
  const top1Correct = goldInShortlistRows.filter((row) =>
    normalizeSurface(row.selectedWord) === normalizeSurface(row.expected)).length;
  const keepErrors = goldInShortlistRows.filter((row) =>
    !row.selectedWord || normalizeSurface(row.selectedWord) === normalizeSurface(row.error)).length;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  const cleanFpMessages = cleanMessages.filter((message) => message.lexicalIssues.length > 0).length;
  return {
    arm: armData.arm,
    messages: evaluations.length,
    gtErrors: allLabels.length,
    detectedErrors: tp,
    correctSuggestionCount: corrected,
    tp,
    fp,
    fn,
    precision: round(precision),
    recall: round(recall),
    f1: round(f1),
    correctionAccuracyAmongDetected: round(tp ? corrected / tp : 0),
    endToEndCorrectionRecall: round(allLabels.length ? corrected / allLabels.length : 0),
    cleanMessages: cleanMessages.length,
    cleanFpMessages,
    cleanFpMessageRate: round(cleanMessages.length ? cleanFpMessages / cleanMessages.length : 0),
    cleanFpIssues: evaluations.reduce((sum, message) => sum + message.cleanFalsePositiveIssues, 0),
    protectedSpans,
    protectedSpanViolations: protectedViolations,
    protectedSpanViolationRate: round(protectedSpans ? protectedViolations / protectedSpans : 0),
    lexicalIssueCount: lexicalIssues,
    stageCounts,
    oracle: armData.arm === 'OFF' ? {
      candidatePoolRecall: null,
      shortlistAt8Recall: null,
      transformerTop1GivenGoldAt8: null,
      transformerKeepErrorRate: null,
    } : {
      candidatePoolRecall: round(poolRows.length ? poolRows.filter((row) => row.goldInPool).length / poolRows.length : 0),
      shortlistAt8Recall: round(shortlistRows.length
        ? shortlistRows.filter((row) => row.goldInShortlist).length / shortlistRows.length : 0),
      transformerTop1GivenGoldAt8: round(goldInShortlistRows.length
        ? top1Correct / goldInShortlistRows.length : 0),
      transformerKeepErrorRate: round(goldInShortlistRows.length
        ? keepErrors / goldInShortlistRows.length : 0),
      candidatePoolHits: poolRows.filter((row) => row.goldInPool).length,
      shortlistHits: shortlistRows.filter((row) => row.goldInShortlist).length,
      goldInShortlist: goldInShortlistRows.length,
      top1Correct,
      keepErrors,
      attentionEvaluatedErrors: evaluatedRows.length,
    },
    attributionRows: stageRows,
  };
}

function categoryBreakdown(arm, evaluations) {
  const categories = Object.keys(EXPECTED_CATEGORIES);
  return categories.map((category) => {
    const subset = evaluations.filter((message) => message.category === category);
    const m = metricsFor({ arm }, subset);
    return {
      arm,
      category,
      gtErrors: m.gtErrors,
      tp: m.tp,
      fp: m.fp,
      fn: m.fn,
      precision: m.precision,
      recall: m.recall,
      f1: m.f1,
      corrected: m.correctSuggestionCount,
      e2eRecall: m.endToEndCorrectionRecall,
    };
  });
}

function outputJsonl(file, rows) {
  fs.writeFileSync(file, rows.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
}

function gitValue(args) {
  try {
    return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

function buildAttributionCsv(rows) {
  const headers = [
    'id', 'category', 'input', 'error', 'expected', 'start', 'end', 'token_class',
    'prefilter_reason', 'candidate_pool_size', 'gold_in_pool', 'gold_pool_rank',
    'shortlist_size', 'gold_in_shortlist', 'gold_shortlist_rank', 'selected_word',
    'selected_probability', 'keep_probability', 'final_emitted', 'final_correct',
    'failure_stage',
  ];
  return toCsv(headers, rows.map((row) => [
    row.id, row.category, row.input, row.error, row.expected, row.start, row.end,
    row.tokenClass, row.prefilterReason, row.candidatePoolSize, row.goldInPool,
    row.goldPoolRank, row.shortlistSize, row.goldInShortlist, row.goldShortlistRank,
    row.selectedWord, row.selectedProbability, row.keepProbability, row.finalEmitted,
    row.finalCorrect, row.failureStage,
  ]));
}

function formatPercent(value) {
  return value == null ? 'N/A' : `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 4) {
  return value == null ? 'N/A' : Number(value).toFixed(digits);
}

function compareRows(metrics, arms) {
  const off = metrics.OFF;
  const shadow = metrics.SHADOW;
  const active = metrics.ACTIVE;
  const runtime = arms;
  const delta = (a, b) => a == null || b == null ? null : round(b - a);
  return [
    ['Precision', off.precision, shadow.precision, active.precision, delta(off.precision, active.precision)],
    ['Recall', off.recall, shadow.recall, active.recall, delta(off.recall, active.recall)],
    ['F1', off.f1, shadow.f1, active.f1, delta(off.f1, active.f1)],
    ['E2E correction recall', off.endToEndCorrectionRecall, shadow.endToEndCorrectionRecall,
      active.endToEndCorrectionRecall, delta(off.endToEndCorrectionRecall, active.endToEndCorrectionRecall)],
    ['Clean FP messages', off.cleanFpMessages, shadow.cleanFpMessages, active.cleanFpMessages,
      delta(off.cleanFpMessages, active.cleanFpMessages)],
    ['Protected violations', off.protectedSpanViolations, shadow.protectedSpanViolations,
      active.protectedSpanViolations, delta(off.protectedSpanViolations, active.protectedSpanViolations)],
    ['Candidate pool oracle', null, shadow.oracle.candidatePoolRecall, active.oracle.candidatePoolRecall,
      delta(null, active.oracle.candidatePoolRecall)],
    ['Shortlist@8 oracle', null, shadow.oracle.shortlistAt8Recall, active.oracle.shortlistAt8Recall,
      delta(null, active.oracle.shortlistAt8Recall)],
    ['Transformer top1 given gold@8', null, shadow.oracle.transformerTop1GivenGoldAt8,
      active.oracle.transformerTop1GivenGoldAt8, delta(null, active.oracle.transformerTop1GivenGoldAt8)],
    ['p50 latency', runtime.OFF.runtime.engine.p50Ms,
      runtime.SHADOW.runtime.engine.p50Ms, runtime.ACTIVE.runtime.engine.p50Ms,
      delta(runtime.OFF.runtime.engine.p50Ms, runtime.ACTIVE.runtime.engine.p50Ms)],
    ['p95 latency', runtime.OFF.runtime.engine.p95Ms,
      runtime.SHADOW.runtime.engine.p95Ms, runtime.ACTIVE.runtime.engine.p95Ms,
      delta(runtime.OFF.runtime.engine.p95Ms, runtime.ACTIVE.runtime.engine.p95Ms)],
    ['p99 latency', runtime.OFF.runtime.engine.p99Ms,
      runtime.SHADOW.runtime.engine.p99Ms, runtime.ACTIVE.runtime.engine.p99Ms,
      delta(runtime.OFF.runtime.engine.p99Ms, runtime.ACTIVE.runtime.engine.p99Ms)],
    ['attention calls/msg', 0, runtime.SHADOW.runtime.attention.callsPerMessage.mean,
      runtime.ACTIVE.runtime.attention.callsPerMessage.mean,
      runtime.ACTIVE.runtime.attention.callsPerMessage.mean],
  ];
}

function buildSummaryCsv(rows) {
  return toCsv(['Metric', 'OFF', 'SHADOW', 'ACTIVE', 'ACTIVE delta vs OFF'], rows);
}

function chooseBottleneck(activeMetrics) {
  const counts = activeMetrics.stageCounts;
  const candidates = [
    ['prefilter', counts.PREFILTERED],
    ['candidate generation', counts.CANDIDATE_MISS],
    ['K=8 shortlist', counts.SHORTLIST_MISS],
    ['Transformer ranking', counts.MODEL_WRONG_CANDIDATE],
    ['KEEP behavior', counts.MODEL_KEEP_ERROR],
    ['threshold gate', counts.CORRECT_RANK_BUT_GATE_REJECT],
  ];
  return candidates.sort((a, b) => b[1] - a[1])[0];
}

function buildReport({ runInfo, metrics, arms, categoryRows, attributionRows, parity }) {
  const off = metrics.OFF;
  const active = metrics.ACTIVE;
  const shadow = metrics.SHADOW;
  const bottleneck = chooseBottleneck(active);
  const categoryDelta = categoryRows.filter((row) => row.arm === 'ACTIVE')
    .map((row) => {
      const offRow = categoryRows.find((x) => x.arm === 'OFF' && x.category === row.category);
      return { category: row.category, delta: round(row.recall - offRow.recall) };
    }).sort((a, b) => b.delta - a.delta);
  const bestCategory = categoryDelta[0];
  const attention = arms.ACTIVE.runtime.attention;
  const activeRecallDelta = round(active.recall - off.recall);
  const activeE2eDelta = round(active.endToEndCorrectionRecall - off.endToEndCorrectionRecall);
  const precisionDelta = round(active.precision - off.precision);
  const qualityStrong = activeRecallDelta >= 0.03 && activeE2eDelta >= 0.03
    && active.precision >= 0.985 && active.protectedSpanViolations === 0;
  const regression = active.protectedSpanViolations > off.protectedSpanViolations
    || active.precision < off.precision - 0.01
    || active.endToEndCorrectionRecall < off.endToEndCorrectionRecall - 0.01;
  const attentionQuality = regression ? 'REGRESSION'
    : qualityStrong ? 'STRONG'
      : (activeRecallDelta >= 0.01 || activeE2eDelta >= 0.01 ? 'MODERATE' : 'WEAK');
  const runtimeDelta = round(arms.ACTIVE.runtime.engine.p95Ms - arms.OFF.runtime.engine.p95Ms);
  const perCall = attention.msPerCall;
  const callMean = attention.callsPerMessage.mean;
  const nextStep = bottleneck[0] === 'candidate generation'
    ? 'improve candidate generation and measure pool oracle recall again'
    : bottleneck[0] === 'K=8 shortlist'
      ? 'improve shortlist diversity/ranking before changing the Transformer'
      : bottleneck[0] === 'Transformer ranking' || bottleneck[0] === 'KEEP behavior'
        ? 'retrain the Transformer with lane-specific contextual hard negatives'
        : bottleneck[0] === 'threshold gate'
          ? 'calibrate the gate on a separate dev/calibration set'
          : bottleneck[0] === 'prefilter'
            ? 'audit attention eligibility/prefilter routing on the missed categories'
            : 'profile and reduce attention invocation/runtime overhead';
  const shouldOptimize = qualityStrong && active.protectedSpanViolations === 0 ? 'YES' : 'NO';

  const categoryTable = categoryRows.map((row) =>
    `| ${row.category} | ${row.arm} | ${row.gtErrors} | ${row.tp} | ${row.fp} | ${row.fn} | ${formatPercent(row.precision)} | ${formatPercent(row.recall)} | ${formatPercent(row.f1)} | ${row.corrected} | ${formatPercent(row.e2eRecall)} |`).join('\n');
  const stageTable = Object.entries(active.stageCounts)
    .map(([stage, count]) => `| ${stage} | ${count} | ${formatPercent(active.gtErrors ? count / active.gtErrors : 0)} |`).join('\n');
  const compareTable = compareRows(metrics, arms)
    .map((row) => `| ${row[0]} | ${row[1] == null ? 'N/A' : row[1]} | ${row[2] == null ? 'N/A' : row[2]} | ${row[3] == null ? 'N/A' : row[3]} | ${row[4] == null ? 'N/A' : row[4]} |`).join('\n');

  return `# Tiny Attention Transformer — SMS-REAL-200

Run commit: ${runInfo.source.commit}; dataset records: **${runInfo.dataset.messages}**; Node: ${runInfo.environment.node}.

This is a direct test-set evaluation. No training, fine-tuning, artifact regeneration, vocabulary/LM/dictionary changes, candidate-generation changes, shortlist changes, or threshold tuning were performed. ACTIVE overrides only the serving mode to EXPERIMENTAL_ACTIVE; frozen threshold values remain unchanged.

## Current artifact

| Field | Value |
| --- | --- |
| Architecture | ${runInfo.model.architecture.arch} |
| Blocks | ${runInfo.model.architecture.blocks} |
| Hidden dim | ${runInfo.model.architecture.hiddenDim} |
| Heads / head dim | ${runInfo.model.architecture.numHeads} / ${runInfo.model.architecture.headDim} |
| FFN dim | ${runInfo.model.architecture.ffnDim} |
| Max context | ${runInfo.model.architecture.maxTokens} |
| Embedding vocab | ${runInfo.model.architecture.vocabSize} |
| Attention word-vocab entries | ${runInfo.model.architecture.wordVocabEntries} |
| Shortlist K | ${runInfo.model.architecture.k} |
| Binary size | ${runInfo.model.artifacts.binarySize} bytes |
| Binary SHA-256 | ${runInfo.model.artifacts.binarySha256} |
| Metadata SHA-256 | ${runInfo.model.artifacts.metadataSha256} |

## Compare table

| Metric | OFF | SHADOW | ACTIVE | ACTIVE delta vs OFF |
| --- | ---: | ---: | ---: | ---: |
${compareTable}

## Category breakdown

| Category | Arm | GT errors | TP | FP | FN | Precision | Recall | F1 | Corrected | E2E recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${categoryTable}

## Required answers

### 1. Does Tiny Attention improve SMS-REAL-200?

ACTIVE recall is ${formatPercent(active.recall)} vs OFF ${formatPercent(off.recall)} (${activeRecallDelta >= 0 ? '+' : ''}${formatPercent(Math.abs(activeRecallDelta))} absolute delta); F1 is ${formatPercent(active.f1)} vs ${formatPercent(off.f1)}; end-to-end correction recall is ${formatPercent(active.endToEndCorrectionRecall)} vs ${formatPercent(off.endToEndCorrectionRecall)} (${activeE2eDelta >= 0 ? '+' : ''}${formatPercent(Math.abs(activeE2eDelta))}). Precision changes from ${formatPercent(off.precision)} to ${formatPercent(active.precision)} (${precisionDelta >= 0 ? '+' : ''}${formatPercent(Math.abs(precisionDelta))}). Verdict: **${attentionQuality}**.

### 2. Which category drives recall change?

The largest ACTIVE-vs-OFF recall delta is **${bestCategory.category}** at ${bestCategory.delta >= 0 ? '+' : ''}${formatPercent(Math.abs(bestCategory.delta))}. Full per-category numbers are in category-breakdown.csv.

### 3. Precision or clean-FP impact

ACTIVE has ${active.fp} total lexical FP vs OFF ${off.fp}; precision is ${formatPercent(active.precision)} vs ${formatPercent(off.precision)}. CLEAN FP is ${active.cleanFpMessages}/${active.cleanMessages} messages (${formatPercent(active.cleanFpMessageRate)}) and ${active.cleanFpIssues} issues, versus OFF ${off.cleanFpMessages}/${off.cleanMessages} and ${off.cleanFpIssues} issues.

### 4. Protected entities

ACTIVE protected-span violations: **${active.protectedSpanViolations}/${active.protectedSpans} (${formatPercent(active.protectedSpanViolationRate)})**; OFF: ${off.protectedSpanViolations}/${off.protectedSpans}. Target 0 is ${active.protectedSpanViolations === 0 ? 'met' : 'not met'}.

### 5. Primary pipeline bottleneck

ACTIVE attribution counts:

| Stage | Errors | Share of GT errors |
| --- | ---: | ---: |
${stageTable}

The largest observed bucket is **${bottleneck[0]}** (${bottleneck[1]} errors). This is the primary diagnosis from the trace, not an assumption about attention quality.

### 6. Why is attention latency high?

ACTIVE measures ${formatNumber(callMean, 3)} attention calls/message, ${formatNumber(attention.totalMsPerMessage.mean, 3)} ms attention time/message, and ${formatNumber(perCall, 3)} ms/call. Engine p95 delta vs OFF is ${formatNumber(runtimeDelta, 3)} ms. The dominant factor is **${callMean > 5 ? 'call count/message' : 'per-call compute'}**; see latency-report.json for p50/p95/p99 and call distributions.

### 7. Highest-ROI next step

**${nextStep}.** Do not use SMS-REAL-200 to choose a new threshold; it remains the test set.

## SHADOW parity

shadow_output_mismatches = ${parity.mismatches} across all final issue fields required by the task (ruleId, span, value, suggestions, severity). ${parity.mismatches === 0 ? 'SHADOW preserves OFF output.' : 'Mismatches are listed in summary.json.'}

## Decision metrics

| Metric | SHADOW | ACTIVE |
| --- | ---: | ---: |
| Candidate pool oracle recall | ${formatPercent(shadow.oracle.candidatePoolRecall)} | ${formatPercent(active.oracle.candidatePoolRecall)} |
| Shortlist@8 oracle recall | ${formatPercent(shadow.oracle.shortlistAt8Recall)} | ${formatPercent(active.oracle.shortlistAt8Recall)} |
| Transformer top-1 given gold@8 | ${formatPercent(shadow.oracle.transformerTop1GivenGoldAt8)} | ${formatPercent(active.oracle.transformerTop1GivenGoldAt8)} |
| Transformer KEEP error rate | ${formatPercent(shadow.oracle.transformerKeepErrorRate)} | ${formatPercent(active.oracle.transformerKeepErrorRate)} |

## Runtime

| Arm | Engine p50 | Engine p95 | Engine p99 | Cold start | RSS after load | Calls/message | Attention ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${ARMS.map((arm) => `| ${arm} | ${formatNumber(arms[arm].runtime.engine.p50Ms, 3)} ms | ${formatNumber(arms[arm].runtime.engine.p95Ms, 3)} ms | ${formatNumber(arms[arm].runtime.engine.p99Ms, 3)} ms | ${formatNumber(arms[arm].runtime.coldStartMs, 3)} ms | ${Math.round(arms[arm].runtime.rssAfterLoadBytes / 1024 / 1024)} MiB | ${formatNumber(arms[arm].runtime.attention.callsPerMessage.mean, 3)} | ${formatNumber(arms[arm].runtime.attention.msPerCall, 3)} ms |`).join('\n')}

## Minimal-pair analysis

See adversarial-analysis.md for the required per-case INPUT, EXPECTED, generated pool, K=8 shortlist, KEEP/candidate logits/probabilities, selected word, final emission, stage, and verdict.

## FINAL VERDICT

Attention quality:
${attentionQuality}

Primary bottleneck:
${bottleneck[0]}

Current model worth optimizing further:
${shouldOptimize}

Reason:
${qualityStrong ? 'ACTIVE clears the task quality/safety thresholds and the remaining issue is runtime or the measured bottleneck.' : 'The SMS-REAL-200 evidence does not justify treating the current attention model as a strong production improvement.'}

Highest-ROI next task:
${nextStep}

Do NOT do yet:
Train/fine-tune or tune thresholds on SMS-REAL-200; use a separate dev/calibration set for any future calibration work.
`;
}

function buildAdversarialReport(evaluationsByArm) {
  const rows = evaluationsByArm.ACTIVE.filter((message) =>
    message.category === 'ADVERSARIAL_MINIMAL_PAIR');
  const sections = rows.map((message) => {
    const expected = message.expected[0];
    const renderArm = (arm) => {
      const m = evaluationsByArm[arm].find((item) => item.id === message.id);
      const attr = classifyAttribution(m, 0, arm);
      const token = findToken(m.tokenTrace, expected);
      const options = token?.shortlist?.map((word, index) => ({
        word,
        logit: token.logits?.[index + 1] ?? null,
        probability: token.probabilities?.[index + 1] ?? null,
      })) ?? [];
      if (token?.attentionEvaluated) options.unshift({
        word: token.token,
        logit: token.logits?.[0] ?? null,
        probability: token.probabilities?.[0] ?? null,
      });
      const verdict = attr.finalCorrect ? 'CORRECT' : attr.failureStage;
      return `### ${arm}\n\n- Generated pool: ${token?.generatedWords?.join(', ') || 'N/A'}\n- Shortlist K=8: ${token?.shortlist?.join(', ') || 'N/A'}\n- Selected: ${token?.selected || 'N/A'}; final emitted: ${attr.finalEmitted}; final correct: ${attr.finalCorrect}\n- Stage: ${attr.failureStage}; verdict: **${verdict}**\n- KEEP/candidate logits/probs: ${options.length ? options.map((x) => `${x.word} (logit=${x.logit == null ? 'N/A' : x.logit.toFixed(6)}, p=${x.probability == null ? 'N/A' : x.probability.toFixed(6)})`).join(' | ') : 'N/A'}\n`;
    };
    return `## ${message.id}\n\n- INPUT: ${message.content}\n- EXPECTED: ${expected.value} -> ${expected.suggestion}\n\n${ARMS.map(renderArm).join('\n')}`;
  });
  return `# Adversarial minimal-pair analysis\n\nThese are direct test-set observations. No candidate, model, or threshold tuning was performed.\n\n${sections.join('\n')}`;
}

function buildParity(offMessages, shadowMessages) {
  const mismatches = [];
  const shadowById = new Map(shadowMessages.map((message) => [message.id, message]));
  for (const off of offMessages) {
    const shadow = shadowById.get(off.id);
    const a = off.finalIssues.map(issueSignature);
    const b = shadow?.finalIssues.map(issueSignature) ?? [];
    if (JSON.stringify(a) !== JSON.stringify(b)) {
      mismatches.push({ id: off.id, off: off.finalIssues, shadow: shadow?.finalIssues ?? [] });
    }
  }
  return { mismatches: mismatches.length, cases: mismatches };
}

function buildRunInfo({ datasetPath, rows, categoryCounts, commit, arms }) {
  const metadataPath = path.join(ROOT, 'src', 'data', 'attention-reranker.json');
  const binaryPath = path.join(ROOT, 'src', 'data', 'attention-reranker.int8.bin');
  const actualVocabPath = path.join(ROOT, 'src', 'data', 'attention-vocab.json');
  const metadata = readJson(metadataPath);
  const vocab = readJson(actualVocabPath);
  const configService = new ValidationConfigService();
  const effective = {};
  for (const arm of ARMS) {
    configService.reload({ spelling: { attentionMode: ATTENTION_MODES[arm] } });
    effective[arm] = effectiveConfig(configService);
  }
  return {
    schema: 'sms-real-200-attention-benchmark-v1',
    createdAt: new Date().toISOString(),
    source: {
      branch: gitValue(['branch', '--show-current']),
      commit,
      originMain: gitValue(['ls-remote', 'origin', 'refs/heads/main'])?.split(/\s+/)[0] ?? null,
      workingTreeStatus: gitValue(['status', '--short']),
      sourceCodeUnderTest: 'src/ and config/ at recorded commit; benchmark script is instrumentation only',
    },
    dataset: {
      path: path.relative(ROOT, datasetPath).replaceAll('\\', '/'),
      messages: rows.length,
      categories: categoryCounts,
      sha256: sha256File(datasetPath),
      usedAs: 'test-only; no training, fine-tuning, or threshold tuning',
    },
    model: {
      architecture: {
        arch: metadata.arch,
        blocks: metadata.config.blocks,
        hiddenDim: metadata.config.hidden_dim,
        numHeads: metadata.config.num_heads,
        headDim: metadata.config.head_dim,
        ffnDim: metadata.config.ffn_dim,
        maxTokens: metadata.config.max_tokens,
        vocabSize: metadata.config.vocab_size,
        wordVocabEntries: Object.keys(vocab.wordToId ?? {}).length,
        k: metadata.k,
        preLn: metadata.config.pre_ln,
      },
      artifacts: {
        binaryPath: path.relative(ROOT, binaryPath).replaceAll('\\', '/'),
        metadataPath: path.relative(ROOT, metadataPath).replaceAll('\\', '/'),
        vocabPath: path.relative(ROOT, actualVocabPath).replaceAll('\\', '/'),
        binarySize: fs.statSync(binaryPath).size,
        binarySha256: sha256File(binaryPath),
        metadataSha256: sha256File(metadataPath),
        vocabSha256: sha256File(actualVocabPath),
      },
    },
    frozenConfig: {
      tuningFile: 'config/spelling-tuning.json',
      tuningFileSha256: sha256File(path.join(ROOT, 'config', 'spelling-tuning.json')),
      effectiveByArm: effective,
      attentionModeOverrideOnly: true,
    },
    environment: {
      node: process.version,
      platform: process.platform,
      release: os.release(),
      cpu: os.cpus()[0]?.model ?? 'unknown',
      engineProfile: process.env.ENGINE_PROFILE ?? 'default',
      warmupCount: 3,
    },
    arms: Object.fromEntries(ARMS.map((arm) => [arm, {
      attentionMode: ATTENTION_MODES[arm],
      runtime: arms[arm].runtime,
    }])),
  };
}

function runWorker(args) {
  const datasetPath = path.resolve(argValue(args, '--dataset', DEFAULT_DATASET));
  const outputPath = path.resolve(argValue(args, '--worker-output', ''));
  const arm = argValue(args, '--arm', null);
  if (!outputPath || !ARMS.includes(arm)) throw new Error('worker requires --arm and --worker-output');
  const rows = readJson(datasetPath);
  validateDataset(rows);
  const result = runArm(rows, arm);
  fs.writeFileSync(outputPath, JSON.stringify(result), 'utf8');
}

function spawnArmWorker(datasetPath, arm) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sms-real-200-attention-'));
  const outputPath = path.join(tempDir, `${arm}.json`);
  const scriptPath = fileURLToPath(import.meta.url);
  const child = spawnSync(process.execPath, [
    '--expose-gc', scriptPath, '--worker', '--arm', arm,
    '--dataset', datasetPath, '--worker-output', outputPath,
  ], {
    cwd: ROOT,
    encoding: 'utf8',
    env: { ...process.env },
    maxBuffer: 100 * 1024 * 1024,
  });
  try {
    if (child.status !== 0) {
      throw new Error(`worker ${arm} failed (${child.status}): ${child.stderr || child.stdout}`);
    }
    return readJson(outputPath);
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

function writeOutputs({ outDir, runInfo, armData, evaluations, metrics, categoryRows, parity }) {
  fs.mkdirSync(outDir, { recursive: true });
  for (const arm of ARMS) {
    outputJsonl(path.join(outDir, `${arm.toLowerCase()}-results.jsonl`), evaluations[arm]);
  }
  const allTraceRows = ARMS.flatMap((arm) => evaluations[arm].map((message) => ({
    arm,
    id: message.id,
    tokenCount: message.tokenCount,
    attentionCalls: message.attentionCalls,
    attentionTotalMs: message.attentionTotalMs,
    evaluatedTokens: message.tokenTrace.filter((token) => token.attentionEvaluated),
  })));
  outputJsonl(path.join(outDir, 'attention-token-trace.jsonl'), allTraceRows);
  fs.writeFileSync(path.join(outDir, 'failure-attribution.csv'), buildAttributionCsv(metrics.ACTIVE.attributionRows), 'utf8');
  const summaryRows = compareRows(metrics, armData);
  fs.writeFileSync(path.join(outDir, 'summary.csv'), buildSummaryCsv(summaryRows), 'utf8');
  fs.writeFileSync(path.join(outDir, 'category-breakdown.csv'), toCsv(
    ['arm', 'category', 'gt_errors', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'corrected', 'e2e_recall'],
    categoryRows.map((row) => [row.arm, row.category, row.gtErrors, row.tp, row.fp, row.fn,
      row.precision, row.recall, row.f1, row.corrected, row.e2eRecall]),
  ), 'utf8');
  fs.writeFileSync(path.join(outDir, 'latency-report.json'), JSON.stringify(
    Object.fromEntries(ARMS.map((arm) => [arm, armData[arm].runtime])), null, 2) + '\n', 'utf8');
  fs.writeFileSync(path.join(outDir, 'summary.json'), JSON.stringify({
    schema: 'sms-real-200-attention-summary-v1',
    runInfo,
    metrics,
    shadowOutputMismatches: parity.mismatches,
    parityCases: parity.cases,
  }, null, 2) + '\n', 'utf8');
  fs.writeFileSync(path.join(outDir, 'run-info.json'), JSON.stringify(runInfo, null, 2) + '\n', 'utf8');
  fs.writeFileSync(path.join(outDir, 'adversarial-analysis.md'), buildAdversarialReport(evaluations), 'utf8');
  fs.writeFileSync(path.join(outDir, 'benchmark-report.md'), buildReport({
    runInfo, metrics, arms: armData, categoryRows, attributionRows: metrics.ACTIVE.attributionRows,
    parity,
  }), 'utf8');
}

async function main() {
  const args = process.argv.slice(2);
  const datasetPath = path.resolve(argValue(args, '--dataset', DEFAULT_DATASET));
  const outDir = path.resolve(argValue(args, '--out', DEFAULT_OUT));
  const rows = readJson(datasetPath);
  const categoryCounts = validateDataset(rows);
  const commit = gitValue(['rev-parse', 'HEAD']);
  const branch = gitValue(['branch', '--show-current']);
  const originMain = gitValue(['ls-remote', 'origin', 'refs/heads/main'])?.split(/\s+/)[0] ?? null;
  if (branch !== 'main' || !commit || commit !== originMain) {
    throw new Error(`STOP: expected current main at origin/main; branch=${branch}, HEAD=${commit}, origin/main=${originMain}`);
  }

  const armData = {};
  for (const arm of ARMS) armData[arm] = spawnArmWorker(datasetPath, arm);
  const evaluations = Object.fromEntries(ARMS.map((arm) => [arm, attachEvaluation(armData[arm])]));
  const metrics = Object.fromEntries(ARMS.map((arm) => [
    arm, metricsFor(armData[arm], evaluations[arm]),
  ]));
  const categoryRows = ARMS.flatMap((arm) => categoryBreakdown(arm, evaluations[arm]));
  const parity = buildParity(evaluations.OFF, evaluations.SHADOW);
  const runInfo = buildRunInfo({ datasetPath, rows, categoryCounts, commit, arms: armData });
  writeOutputs({ outDir, runInfo, armData, evaluations, metrics, categoryRows, parity });
  console.log(JSON.stringify({
    ok: true,
    commit,
    datasetRecords: rows.length,
    output: path.relative(ROOT, outDir).replaceAll('\\', '/'),
    shadowOutputMismatches: parity.mismatches,
    metrics: Object.fromEntries(ARMS.map((arm) => [arm, {
      precision: metrics[arm].precision,
      recall: metrics[arm].recall,
      f1: metrics[arm].f1,
      endToEndCorrectionRecall: metrics[arm].endToEndCorrectionRecall,
    }])),
  }, null, 2));
}

if (hasArg(process.argv.slice(2), '--worker')) {
  runWorker(process.argv.slice(2));
} else {
  await main();
}
