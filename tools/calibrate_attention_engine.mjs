// ============================================================
// Task 10 (tiny-attention-spelling-reranker-FIXED plan) — Full-Engine
// Attention Calibration on Calibration Messages using Shadow-Cache Replay.
//
// Constraints honored:
//   - Operates on .tmp/attention-messages-calibration.jsonl ONLY;
//   - Never reads internal-test or dev;
//   - Runs 1 OFF pass + 1 SHADOW pass to build decision cache;
//   - Replays 450 grid points in-memory without calling engine.validate();
//   - Runs 1 full-engine verification pass for winner (total calls <= messages * 3);
//   - Strict 4 criteria for winner selection (P >= base, R >= base, F0.5 >= base, FP <= base);
//   - Fallback to SHADOW if no trial passes;
//   - Writes report to .tmp/attention-engine-calibration-report.json.
// ============================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';
import { evaluateSpellingToken } from '../src/rules/linguistic-rules.mjs';
import { matchIssuesOneToOne } from './evaluate_attention_messages.mjs';
import { validateEvaluationHeader } from '../src/evaluation-provenance.mjs';

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

export function evaluateEngineOnMessages(engine, messages) {
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

export function replayShadowCache(shadowCache, { minProb, minCandWin, maxOrigWin }) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let totalIssues = 0;

  for (const entry of shadowCache) {
    const issues = [...entry.otherIssues];
    for (const tok of entry.tokenDecisions) {
      if (tok.shadowEvaluated) {
        if (tok.selectedOptionIndex > 0 && tok.chosenCandidateWord) {
          if (tok.confidence >= minProb && tok.candidateAttestedWindows >= minCandWin && tok.originalAttestedWindows <= maxOrigWin) {
            issues.push({
              ruleId: 'POSSIBLE_SPELLING_ERROR',
              start: tok.start,
              end: tok.end,
              value: tok.value,
              suggestions: tok.attentionSuggestions,
            });
          }
        }
        // tok.selectedOptionIndex === 0 is KEEP_ORIGINAL -> no issue emitted
      } else if (tok.classicalEmit) {
        issues.push({
          ruleId: 'POSSIBLE_SPELLING_ERROR',
          start: tok.start,
          end: tok.end,
          value: tok.value,
          suggestions: tok.classicalSuggestions,
        });
      }
    }
    totalIssues += issues.length;
    const matched = matchIssuesOneToOne(issues, entry.labels);
    tp += matched.tp;
    fp += matched.fp;
    fn += matched.fn;
  }

  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
  const f05 = ((1.25 * precision * recall) / (0.25 * precision + recall)) || 0.0;

  return { tp, fp, fn, totalIssues, precision, recall, f05 };
}

export async function calibrateAttentionEngine({
  messagesPath,
  probabilitySteps = null,
  minCandidateWindowsList = [0, 1, 2],
  maxOriginalWindowsList = [1, 2, 3],
  outputReportPath = null,
  outputConfigPath = null,
}) {
  const lines = readFileSync(messagesPath, 'utf8').split(/\r?\n/).filter(Boolean);
  if (!lines.length) throw new Error(`${messagesPath}: empty calibration dump`);
  const header = JSON.parse(lines[0]);
  validateEvaluationHeader(header, {
    schema: 'attention-messages-v1',
    split: 'calibration',
  });
  const messages = lines.slice(1).map((line) => JSON.parse(line))
    .filter((m) => m.recordType === 'message-row');
  const messagesHash = createHash('sha256')
    .update(readFileSync(messagesPath)).digest('hex');

  let engineValidationCount = 0;

  // 1. Instantiate engine
  const configService = new ValidationConfigService();
  configService.reload({
    spelling: {
      attentionMode: 'SHADOW',
    },
  });
  const engine = new SmsValidationEngine({ configService });

  // Baseline metrics with attention OFF (Pass 1)
  configService.reload({ spelling: { attentionMode: 'OFF' } });
  const baseline = evaluateEngineOnMessages(engine, messages);
  engineValidationCount += messages.length;

  // Pass 2: SHADOW mode pass to build the shadow decision replay cache
  configService.reload({ spelling: { attentionMode: 'SHADOW' } });
  const shadowSnap = configService.snapshot();
  const shadowCache = [];

  for (const m of messages) {
    const ctx = new ValidationContext(m.text, MessageMode.ACCENTED, m.brand || 'TEST');
    const res = engine.validate(ctx);
    engineValidationCount++;

    const otherIssues = res.issues.filter(
      (iss) => iss.ruleId === 'POSSIBLE_MISSING_DIACRITIC',
    );

    const doc = engine.documentBuilder.build(ctx);
    const words = doc.tokens.filter((t) => t.type === 'WORD');
    const tokenDecisions = [];

    for (let idx = 0; idx < words.length; idx++) {
      const d = evaluateSpellingToken(engine.services, shadowSnap, ctx, doc, words, idx);
      const w = words[idx];
      const classicalEmit = (d.stage === 'decided' && d.emit);
      const shadow = d.shadowAttention;
      const chosenWord = shadow?.chosenCandidateWord ?? null;
      let attentionSuggestions = [];
      if (chosenWord) {
        attentionSuggestions = [
          chosenWord,
          ...(d.suggestions || []).filter((s) => s.toLowerCase() !== chosenWord.toLowerCase()),
        ].slice(0, 3);
      }
      tokenDecisions.push({
        start: w.start,
        end: w.end,
        value: w.original,
        normalized: w.normalized,
        classicalEmit,
        classicalSuggestions: d.suggestions || [],
        shadowEvaluated: Boolean(shadow?.evaluated),
        selectedOptionIndex: shadow?.selectedOptionIndex ?? 0,
        confidence: shadow?.confidence ?? 0,
        candidateAttestedWindows: shadow?.candidateAttestedWindows ?? 0,
        originalAttestedWindows: shadow?.originalAttestedWindows ?? 0,
        chosenCandidateWord: chosenWord,
        attentionSuggestions,
      });
    }

    shadowCache.push({
      text: m.text,
      labels: m.labels || [],
      otherIssues,
      tokenDecisions,
    });
  }

  // 3. Replay 450 grid points entirely in memory (0 extra engine validations!)
  const pSteps = probabilitySteps || Array.from({ length: 50 }, (_, i) => Math.round((0.50 + i * 0.01) * 100) / 100);

  const trials = [];
  let bestF05 = -Infinity;
  let winner = null;

  for (const minProb of pSteps) {
    for (const minCandWin of minCandidateWindowsList) {
      for (const maxOrigWin of maxOriginalWindowsList) {
        const metrics = replayShadowCache(shadowCache, {
          minProb,
          minCandWin,
          maxOrigWin,
        });

        // Strict 4-condition gate: precision, recall, f05 >= baseline, fp <= baseline
        const passesGates =
          metrics.precision >= baseline.precision
          && metrics.recall >= baseline.recall
          && metrics.f05 >= baseline.f05
          && metrics.fp <= baseline.fp;

        const trial = {
          params: {
            attentionMode: 'EXPERIMENTAL_ACTIVE',
            attentionMinProbability: minProb,
            attentionMinCandidateWindows: minCandWin,
            attentionMaxOriginalWindows: maxOrigWin,
          },
          metrics,
          precisionDelta: metrics.precision - baseline.precision,
          recallDelta: metrics.recall - baseline.recall,
          f05Delta: metrics.f05 - baseline.f05,
          fpDelta: metrics.fp - baseline.fp,
          passesGates,
        };
        trials.push(trial);

        if (passesGates && metrics.f05 > bestF05) {
          bestF05 = metrics.f05;
          winner = {
            ...trial.params,
            ...metrics,
            precisionDelta: trial.precisionDelta,
            recallDelta: trial.recallDelta,
            f05Delta: trial.f05Delta,
            fpDelta: trial.fpDelta,
          };
        }
      }
    }
  }

  // 4. Pass 3: Full engine verification for winner (if winner found)
  let verifiedFullEngine = null;
  if (winner) {
    configService.reload({
      spelling: {
        attentionMode: 'EXPERIMENTAL_ACTIVE',
        attentionMinProbability: winner.attentionMinProbability,
        attentionMinCandidateWindows: winner.attentionMinCandidateWindows,
        attentionMaxOriginalWindows: winner.attentionMaxOriginalWindows,
      },
    });
    verifiedFullEngine = evaluateEngineOnMessages(engine, messages);
    engineValidationCount += messages.length;

    // Verify replay winner metrics equal full-engine winner metrics
    assert.equal(verifiedFullEngine.tp, winner.tp, 'Replay TP must match full-engine TP');
    assert.equal(verifiedFullEngine.fp, winner.fp, 'Replay FP must match full-engine FP');
    assert.equal(verifiedFullEngine.fn, winner.fn, 'Replay FN must match full-engine FN');
  }

  // Verify instrumentation constraint: validation count proportional to messages * 3
  const maxAllowedValidations = messages.length * 3 + 10;
  assert.ok(
    engineValidationCount <= maxAllowedValidations,
    `Validation count (${engineValidationCount}) exceeded bound (${maxAllowedValidations})`,
  );

  const report = {
    schema: 'attention-engine-calibration-report-v2',
    createdAt: new Date().toISOString(),
    messagesPath,
    messagesCount: messages.length,
    provenance: {
      hashes: { ...header.hashes, messages: messagesHash },
      sourceHeader: header,
    },
    baseline,
    winner,
    verifiedFullEngine,
    engineValidationCount,
    status: winner ? 'CALIBRATED' : 'SHADOW',
    reason: winner ? null : 'No trial met strict baseline precision/recall/F0.5 and FP constraints',
    totalTrials: trials.length,
    passingTrials: trials.filter((t) => t.passesGates).length,
  };

  if (outputReportPath) {
    writeFileSync(outputReportPath, JSON.stringify(report, null, 2), 'utf8');
  }

  if (winner && outputConfigPath && existsSync(outputConfigPath)) {
    const curConfig = JSON.parse(readFileSync(outputConfigPath, 'utf8'));
    curConfig.winner = curConfig.winner || {};
    curConfig.winner.overrides = {
      ...curConfig.winner.overrides,
      attentionMode: 'SHADOW',
      attentionMinProbability: winner.attentionMinProbability,
      attentionMinCandidateWindows: winner.attentionMinCandidateWindows,
      attentionMaxOriginalWindows: winner.attentionMaxOriginalWindows,
    };
    curConfig.frozenParams = {
      ...curConfig.frozenParams,
      attentionMode: 'SHADOW',
      attentionMinProbability: winner.attentionMinProbability,
      attentionMinCandidateWindows: winner.attentionMinCandidateWindows,
      attentionMaxOriginalWindows: winner.attentionMaxOriginalWindows,
    };
    writeFileSync(outputConfigPath, JSON.stringify(curConfig, null, 2), 'utf8');
  }

  return report;
}

async function main() {
  const args = process.argv.slice(2);
  let messagesPath = '.tmp/attention-messages-calibration.jsonl';
  let outputReportPath = '.tmp/attention-engine-calibration-report.json';
  let outputConfigPath = 'config/spelling-tuning.json';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--messages') messagesPath = args[++i];
    if (args[i] === '--output-report') outputReportPath = args[++i];
    if (args[i] === '--output-config') outputConfigPath = args[++i];
  }

  console.log(`Calibrating full-engine attention on ${messagesPath} using shadow-cache replay...`);
  const report = await calibrateAttentionEngine({
    messagesPath,
    outputReportPath,
    outputConfigPath,
  });
  console.log(JSON.stringify({
    ok: true,
    engineValidationCount: report.engineValidationCount,
    baseline: report.baseline,
    winner: report.winner,
    status: report.status,
    reason: report.reason,
    reportPath: outputReportPath,
  }, null, 2));
}

if (process.argv[1] && process.argv[1].endsWith('calibrate_attention_engine.mjs')) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
