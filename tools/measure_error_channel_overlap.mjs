#!/usr/bin/env node
// ============================================================
// Review C1 — how much of the reported spelling score is a LOOKUP?
//
// src/data/error-channel.json is built from vsec-train.jsonl, and on the
// serving path `pairProven` is the main condition that lets an issue be
// emitted (src/rules/linguistic-rules.mjs + the verifier in src/engine.mjs).
// If the (error -> correction) pairs of dev/test are largely the SAME pairs
// as train, then dev/test precision & recall are an optimistic upper bound:
// the evaluation is partly measuring a memorized table, not generalization.
//
// The split itself is honest (grouped by message, no sentence leaks) — what
// leaks is the ERROR DISTRIBUTION. This script measures exactly that, so the
// number in the README is reproducible instead of anecdotal.
//
// Usage:
//   node tools/measure_error_channel_overlap.mjs
//   node tools/measure_error_channel_overlap.mjs --json > overlap.json
// ============================================================
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const VSEC = path.join(ROOT, 'dataset_artifacts', 'vsec');

/**
 * @returns {{pairs: Set<string>, occurrences: string[], lines: number}}
 *   `pairs` are DISTINCT "error\tcorrection" pairs; `occurrences` keeps one
 *   entry per LABEL, which is the view that answers "what share of the labels
 *   we score on can be answered straight from the table?".
 */
function readPairs(file) {
  const pairs = new Set();
  const occurrences = [];
  let lines = 0;
  for (const line of readFileSync(file, 'utf8').split(/\r?\n/)) {
    if (!line.trim()) continue;
    lines += 1;
    const row = JSON.parse(line);
    for (const p of row.correction_pairs ?? []) {
      const error = String(p.error ?? '').toLowerCase();
      const correction = String(p.correction ?? '').toLowerCase();
      if (!error || !correction) continue;
      pairs.add(`${error}\t${correction}`);
      occurrences.push(`${error}\t${correction}`);
    }
  }
  return { pairs, occurrences, lines };
}

function overlap(evalSplit, trainPairs) {
  let distinct = 0;
  for (const pair of evalSplit.pairs) if (trainPairs.has(pair)) distinct += 1;
  let labels = 0;
  for (const pair of evalSplit.occurrences) if (trainPairs.has(pair)) labels += 1;
  const pct = (n, d) => (d ? Math.round((n / d) * 1000) / 10 : 0);
  return {
    pairs: evalSplit.pairs.size,
    alsoInTrain: distinct,
    pctAlsoInTrain: pct(distinct, evalSplit.pairs.size),
    labels: evalSplit.occurrences.length,
    labelsAlsoInTrain: labels,
    pctLabelsAlsoInTrain: pct(labels, evalSplit.occurrences.length),
  };
}

export function measureOverlap({ vsecDir = VSEC } = {}) {
  const files = {
    train: path.join(vsecDir, 'vsec-train.jsonl'),
    dev: path.join(vsecDir, 'vsec-dev.jsonl'),
    test: path.join(vsecDir, 'vsec-test.jsonl'),
  };
  for (const [split, file] of Object.entries(files)) {
    if (!existsSync(file)) throw new Error(`missing ${split} split: ${file}`);
  }
  const train = readPairs(files.train);
  const dev = readPairs(files.dev);
  const test = readPairs(files.test);

  // What the serving artifact actually knows.
  const channelPath = path.join(ROOT, 'src', 'data', 'error-channel.json');
  let channel = null;
  if (existsSync(channelPath)) {
    const raw = JSON.parse(readFileSync(channelPath, 'utf8'));
    const channelPairs = new Set();
    for (const [error, targets] of Object.entries(raw.pairProof ?? {})) {
      for (const t of targets) channelPairs.add(`${error.toLowerCase()}\t${String(t).toLowerCase()}`);
    }
    channel = {
      source: raw.source ?? null,
      pairProofPairs: channelPairs.size,
      directEntries: Object.keys(raw.direct ?? {}).length,
      dev: overlap(dev, channelPairs),
      test: overlap(test, channelPairs),
    };
  }

  return {
    generatedAt: new Date().toISOString(),
    messages: { train: train.lines, dev: dev.lines, test: test.lines },
    uniquePairs: { train: train.pairs.size, dev: dev.pairs.size, test: test.pairs.size },
    vsTrain: { dev: overlap(dev, train.pairs), test: overlap(test, train.pairs) },
    vsServingErrorChannel: channel,
    interpretation: 'A high pctAlsoInTrain means the evaluation splits reuse the '
      + 'training error distribution, so pairProven can answer them from the table. '
      + 'Treat dev/test P/R as an OPTIMISTIC upper bound for real SMS traffic.',
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const report = measureOverlap();
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    const { vsTrain, vsServingErrorChannel: ch, uniquePairs, messages } = report;
    console.log('VSEC error-distribution overlap (review C1)');
    console.log(`  messages     train=${messages.train} dev=${messages.dev} test=${messages.test}`);
    console.log(`  unique pairs train=${uniquePairs.train} dev=${uniquePairs.dev} test=${uniquePairs.test}`);
    for (const split of ['dev', 'test']) {
      const o = vsTrain[split];
      console.log(`  ${split.padEnd(4)} vs train           : distinct ${o.alsoInTrain}/${o.pairs} = ${o.pctAlsoInTrain}%`
        + `  |  labels ${o.labelsAlsoInTrain}/${o.labels} = ${o.pctLabelsAlsoInTrain}%`);
    }
    if (ch) {
      console.log(`  serving error-channel: ${ch.pairProofPairs} pairProof pairs, ${ch.directEntries} direct`);
      for (const split of ['dev', 'test']) {
        const o = ch[split];
        console.log(`  ${split.padEnd(4)} vs error-channel   : distinct ${o.alsoInTrain}/${o.pairs} = ${o.pctAlsoInTrain}%`
          + `  |  labels ${o.labelsAlsoInTrain}/${o.labels} = ${o.pctLabelsAlsoInTrain}%`);
      }
    }
    console.log(`\n  ${report.interpretation}`);
  }
}
