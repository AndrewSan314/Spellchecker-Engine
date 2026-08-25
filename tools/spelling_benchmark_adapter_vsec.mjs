// VSEC benchmark adapter. Reads deterministic real VSEC JSONL splits; it does
// not synthesize benchmark rows from corpus-train.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const BENCHMARK_DIR = path.join(ROOT, 'benchmark');
const VSEC_SPLIT_DIR = process.env.VSEC_SPLIT_DIR
  ? path.resolve(process.env.VSEC_SPLIT_DIR)
  : path.join(ROOT, 'dataset_artifacts', 'vsec');

function readJsonl(filePath) {
  return readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function surface(value) {
  return String(value ?? '').replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function numberSet(values) {
  return new Set((values ?? []).map((value) => Number(value)).filter(Number.isInteger));
}

/**
 * Gather every annotated error position. correction_pairs is not assumed to
 * have one item per error: ambiguous rows are completed from positions and
 * syllable_annotations.
 */
export function collectVsecExpectations(row) {
  const annotations = Array.isArray(row.syllable_annotations) ? row.syllable_annotations : [];
  const byPosition = new Map();
  const positions = numberSet(row.error_positions);
  for (const annotation of annotations) {
    const position = Number(annotation?.position);
    if (!Number.isInteger(position)) continue;
    byPosition.set(position, annotation);
    if (annotation.is_correct === false) positions.add(position);
  }
  const pairsByPosition = new Map();
  for (const pair of Array.isArray(row.correction_pairs) ? row.correction_pairs : []) {
    const position = Number(pair?.position);
    if (!Number.isInteger(position)) continue;
    positions.add(position);
    const list = pairsByPosition.get(position) ?? [];
    list.push(pair);
    pairsByPosition.set(position, list);
  }

  const expect = [];
  for (const position of [...positions].sort((a, b) => a - b)) {
    const annotation = byPosition.get(position);
    const pairs = pairsByPosition.get(position) ?? [];
    const value = surface(pairs.find((pair) => pair?.error)?.error ?? annotation?.syllable);
    if (!value) continue;
    const suggestions = [];
    for (const pair of pairs) if (pair?.correction) suggestions.push(String(pair.correction));
    for (const correction of Array.isArray(annotation?.corrections) ? annotation.corrections : []) {
      if (correction) suggestions.push(String(correction));
    }
    const uniqueSuggestions = [...new Set(suggestions)];
    const expected = {
      ruleId: 'POSSIBLE_SPELLING_ERROR',
      value,
      position,
      suggestions: uniqueSuggestions,
    };
    if (uniqueSuggestions.length > 0) expected.suggestion = uniqueSuggestions[0];
    expect.push(expected);
  }
  return expect;
}

export function vsecRowToBenchmark(row, index, split = 'test') {
  return {
    id: `VSEC_${split.toUpperCase()}_${String(index + 1).padStart(5, '0')}`,
    category: 'spelling-vsec',
    source: 'VSEC',
    split,
    text: row.text,
    mode: 'ACCENTED',
    groundTruth: row.corrected_text,
    expect: collectVsecExpectations(row),
    fullyLabeledRuleIds: ['POSSIBLE_SPELLING_ERROR'],
    allowExtra: false,
  };
}

export function adaptVsecSplit(split = 'test', splitDir = VSEC_SPLIT_DIR) {
  const inputPath = path.join(splitDir, `vsec-${split}.jsonl`);
  if (!existsSync(inputPath)) throw new Error(`VSEC split not found: ${inputPath}. Run tools/split_spelling_datasets.py first.`);
  const sourceRows = readJsonl(inputPath);
  const rows = sourceRows.map((row, index) => vsecRowToBenchmark(row, index, split));
  return {
    meta: {
      description: `VSEC spelling correction benchmark (${split} split)`,
      source: 'VSEC2021/VSEC raw parquet via deterministic grouped split',
      split,
      sourceArtifact: path.relative(ROOT, inputPath).replaceAll(path.sep, '/'),
      rowCount: rows.length,
      expectedIssueCount: rows.reduce((count, row) => count + row.expect.length, 0),
      fullyLabeledRuleIds: ['POSSIBLE_SPELLING_ERROR'],
      generatedFromCorpusTrain: false,
    },
    rows,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const split = process.argv[2] ?? 'test';
  const outPath = process.argv[3] ? path.resolve(process.argv[3]) : path.join(BENCHMARK_DIR, `corpus-vsec-${split}.json`);
  const output = adaptVsecSplit(split);
  writeFileSync(outPath, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Saved ${output.rows.length} real VSEC ${split} rows to ${outPath}`);
  console.log(`Expected labels: ${output.meta.expectedIssueCount}`);
}
