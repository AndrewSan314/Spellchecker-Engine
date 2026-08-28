#!/usr/bin/env node
// ============================================================
// Phase 1 — fit the confidence temperature T for the unified scale
// sigmoid((s1 - s2) / T), separately per linguistic rule.
//
// WHY the temperature is fit on a SERVING-distribution split and not on the
// ranker's training rows: the training file carries 81.8% correct-me tokens
// while real traffic carries 3.96%. Fitting on the training prior reproduces
// the same over-confidence this phase exists to remove. VSEC dev is scored
// token-by-token through the real engine, so the margins seen here are the
// margins production sees.
//
// A raw margin is captured at each decision point through services
// .confidenceSink, BEFORE the precision gates, so the fit sees the full
// distribution instead of the surviving tail. Each sample is labelled by the
// VSEC gold pairs: correct = the proposed change matches gold, or the
// decision was to keep an unchanged token that gold also leaves alone.
//
// Usage:
//   node tools/fit_confidence_temperature.mjs [--split dev] [--write]
// --write updates config/spelling-tuning.json; default is report-only.
// ============================================================
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { SmsValidationEngine, ValidationContext } from '../src/engine.mjs';
import { ValidationConfigService } from '../src/config.mjs';
import { MessageMode } from '../src/core.mjs';
import {
  fitTemperature, calibratedConfidence, expectedCalibrationError,
} from '../src/calibration.mjs';

const argv = process.argv.slice(2);
const split = (argv[argv.indexOf('--split') + 1] ?? 'dev').replace(/^--.*/, 'dev');
const write = argv.includes('--write');

const rows = readFileSync(
  path.join('dataset_artifacts', 'vsec', `vsec-${split}.jsonl`), 'utf8',
).split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l));

const cfg = new ValidationConfigService();
cfg.reload({});
const engine = new SmsValidationEngine({ configService: cfg });

const samples = new Map();   // rule -> [{margin, correct}]
let captured = 0;
for (const row of rows) {
  const want = new Map((row.correction_pairs ?? [])
    .map((p) => [String(p.error).toLowerCase(), String(p.correction).toLowerCase()]));
  engine.services.confidenceSink = (ev) => {
    if (!Number.isFinite(ev.rawMargin)) return;
    const orig = String(ev.original).toLowerCase();
    const cand = String(ev.candidate ?? '').toLowerCase();
    const goldFix = want.get(orig);
    // A decision is "correct" when it agrees with gold: propose exactly the
    // gold correction for a wrong token, or keep a token gold never touches.
    const correct = ev.changed
      ? (goldFix !== undefined && cand === goldFix)
      : (goldFix === undefined);
    const key = ev.rule;
    if (!samples.has(key)) samples.set(key, []);
    samples.get(key).push({ margin: ev.rawMargin, correct });
    captured++;
  };
  try {
    engine.validate(new ValidationContext(row.text, MessageMode.ACCENTED, null));
  } finally {
    engine.services.confidenceSink = null;
  }
}

const CONFIG_KEY = {
  POSSIBLE_MISSING_DIACRITIC: 'missingDiacriticConfidenceTemperature',
  POSSIBLE_SPELLING_ERROR: 'spellingConfidenceTemperature',
  DIFFERENT_KEY_REAL_WORD: 'realWordTypoConfidenceTemperature',
};

console.log(`split=${split}  rows=${rows.length}  decisions captured=${captured}\n`);
const fitted = {};
for (const [rule, list] of [...samples].sort()) {
  const pos = list.filter((s) => s.correct).length;
  const { temperature, nll, n } = fitTemperature(list);
  const eceBefore = expectedCalibrationError(
    list.map((s) => ({ confidence: calibratedConfidence(s.margin, 1), correct: s.correct })));
  const eceAfter = expectedCalibrationError(
    list.map((s) => ({ confidence: calibratedConfidence(s.margin, temperature), correct: s.correct })));
  const sat1 = list.filter((s) => calibratedConfidence(s.margin, 1) >= 0.995).length;
  const satT = list.filter((s) => calibratedConfidence(s.margin, temperature) >= 0.995).length;
  fitted[CONFIG_KEY[rule] ?? rule] = Number(temperature.toFixed(4));
  console.log(`${rule}`);
  console.log(`  n=${n}  correct=${pos} (${(100 * pos / n).toFixed(1)}%)`);
  console.log(`  fitted T = ${temperature.toFixed(4)}   NLL=${nll.toFixed(4)}`);
  console.log(`  ECE  T=1 -> ${eceBefore.toFixed(4)}   T=fit -> ${eceAfter.toFixed(4)}`
    + `   (${(100 * (1 - eceAfter / eceBefore)).toFixed(1)}% reduction)`);
  console.log(`  saturated (conf>=0.995): T=1 ${sat1}/${n}`
    + ` (${(100 * sat1 / n).toFixed(1)}%) -> T=fit ${satT}/${n} (${(100 * satT / n).toFixed(1)}%)\n`);
}

console.log('fitted constants:', JSON.stringify(fitted, null, 2));
if (write) {
  const p = path.join('config', 'spelling-tuning.json');
  const doc = JSON.parse(readFileSync(p, 'utf8'));
  doc.confidenceCalibration = {
    method: 'temperature-scaling on sigmoid((s1-s2)/T)',
    fittedOn: `vsec-${split} through the real engine (serving prior)`,
    fittedAt: new Date().toISOString().slice(0, 10),
    constants: fitted,
  };
  writeFileSync(p, `${JSON.stringify(doc, null, 2)}\n`);
  console.log(`\nwritten to ${p}`);
}
