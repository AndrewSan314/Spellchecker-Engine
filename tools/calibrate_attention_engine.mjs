// ============================================================
// Task 10 (tiny-attention-spelling-reranker-FIXED plan) — Full-Engine
// Attention Calibration on Calibration Messages.
//
// Constraints honored:
//   - Operates on .tmp/attention-messages-calibration.jsonl ONLY;
//   - Never reads internal-test or dev;
//   - Grid sweeps probability threshold (0.50..0.99) and windows;
//   - Selects operating point maximizing F0.5 subject to:
//       precision >= baseline.precision
//       fp_delta <= 0 (no new false positives)
//   - Writes report to .tmp/attention-engine-calibration-report.json;
//   - Updates config/spelling-tuning.json.
// ============================================================
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SmsValidationEngine } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { ValidationContext, MessageMode } from '../src/core.mjs';

function normSurface(v) {
  return String(v ?? '').toLowerCase()
    .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function evaluateEngineOnMessages(engine, messages) {
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
    const usedLabels = new Set();
    const matchedIssues = new Set();

    // 1-to-1 greedy match: exact span + suggestion match first, then normalized surface + suggestion match
    for (let issIdx = 0; issIdx < issues.length; issIdx++) {
      const iss = issues[issIdx];
      const issVal = normSurface(iss.value);
      for (let labIdx = 0; labIdx < labels.length; labIdx++) {
        if (usedLabels.has(labIdx)) continue;
        const lab = labels[labIdx];
        const labVal = normSurface(lab.value);

        const spanMatch = (lab.start != null && iss.start === lab.start) || issVal === labVal;
        if (!spanMatch) continue;

        const sugMatch = (lab.suggestions && lab.suggestions.length > 0)
          ? lab.suggestions.some((s) => (iss.suggestions || []).some((isug) => normSurface(isug) === normSurface(s)))
          : true;

        if (sugMatch) {
          usedLabels.add(labIdx);
          matchedIssues.add(issIdx);
          break;
        }
      }
    }

    const msgTP = matchedIssues.size;
    const msgFP = issues.length - msgTP;
    const msgFN = labels.length - usedLabels.size;

    tp += msgTP;
    fp += msgFP;
    fn += msgFN;
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
  const messages = lines.map((l) => JSON.parse(l)).filter((m) => m.recordType === 'message-row');

  // 2. Instantiate engine ONCE with attention reranker loaded
  const configService = new ValidationConfigService();
  configService.reload({
    spelling: {
      attentionMode: 'SHADOW',
    },
  });
  const engine = new SmsValidationEngine({ configService });

  // Baseline metrics with attention OFF
  configService.reload({ spelling: { attentionMode: 'OFF' } });
  const baseline = evaluateEngineOnMessages(engine, messages);

  // 3. Grid sweep using configService.reload() (reusing the same engine in-memory)
  const pSteps = probabilitySteps || Array.from({ length: 50 }, (_, i) => Math.round((0.50 + i * 0.01) * 100) / 100);

  const trials = [];
  let bestF05 = -Infinity;
  let winner = null;

  for (const minProb of pSteps) {
    for (const minCandWin of minCandidateWindowsList) {
      for (const maxOrigWin of maxOriginalWindowsList) {
        configService.reload({
          spelling: {
            attentionMode: 'EXPERIMENTAL_ACTIVE',
            attentionMinProbability: minProb,
            attentionMinCandidateWindows: minCandWin,
            attentionMaxOriginalWindows: maxOrigWin,
          },
        });
        const metrics = evaluateEngineOnMessages(engine, messages);

        // Strict 4-condition gate: precision, recall, f05 >= baseline, fp <= baseline
        const passesGates =
          metrics.precision >= baseline.precision - 1e-6
          && metrics.recall >= baseline.recall - 1e-6
          && metrics.f05 >= baseline.f05 - 1e-6
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

  if (!winner) {
    // If no trial beat baseline subject to all 4 constraints, fallback to SHADOW
    winner = {
      attentionMode: 'SHADOW',
      attentionMinProbability: 0.95,
      attentionMinCandidateWindows: 1,
      attentionMaxOriginalWindows: 3,
      ...baseline,
      precisionDelta: 0,
      recallDelta: 0,
      f05Delta: 0,
      fpDelta: 0,
      status: 'FALLBACK_SHADOW',
      reason: 'No trial met all 4 mandatory criteria (precision >= base, recall >= base, F0.5 >= base, FP <= base)',
    };
  }

  const report = {
    schema: 'attention-engine-calibration-report-v1',
    createdAt: new Date().toISOString(),
    messagesPath,
    messagesCount: messages.length,
    baseline,
    winner,
    totalTrials: trials.length,
    passingTrials: trials.filter((t) => t.passesGates).length,
  };

  if (outputReportPath) {
    writeFileSync(outputReportPath, JSON.stringify(report, null, 2), 'utf8');
  }

  if (outputConfigPath && existsSync(outputConfigPath)) {
    const curConfig = JSON.parse(readFileSync(outputConfigPath, 'utf8'));
    curConfig.winner = curConfig.winner || {};
    curConfig.winner.overrides = {
      ...curConfig.winner.overrides,
      attentionMode: winner.attentionMode,
      attentionMinProbability: winner.attentionMinProbability,
      attentionMinCandidateWindows: winner.attentionMinCandidateWindows,
      attentionMaxOriginalWindows: winner.attentionMaxOriginalWindows,
    };
    curConfig.frozenParams = {
      ...curConfig.frozenParams,
      attentionMode: winner.attentionMode,
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

  console.log(`Calibrating full-engine attention on ${messagesPath}...`);
  const report = await calibrateAttentionEngine({
    messagesPath,
    outputReportPath,
    outputConfigPath,
  });
  console.log(JSON.stringify({
    ok: true,
    baseline: report.baseline,
    winner: report.winner,
    reportPath: outputReportPath,
  }, null, 2));
}

if (process.argv[1] && process.argv[1].endsWith('calibrate_attention_engine.mjs')) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
