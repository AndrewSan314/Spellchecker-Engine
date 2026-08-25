// ============================================================
// Vietnamese corpus pipeline with leakage-safe source splits.
// Viwiki-Spelling annotated documents are external test data and are never
// loaded here. Synthetic evaluation is generated only from clean-test rows.
// ============================================================
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeNfc } from '../src/normalizer.mjs';
import { generateSyntheticBenchmark } from './diacritic_generator.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const DATA_DIR = path.join(ROOT, 'src', 'data');
const BENCHMARK_DIR = path.join(ROOT, 'benchmark');
const SOURCE_SPLIT_DIR = path.join(ROOT, 'dataset_artifacts', 'clean-source');
export const SOURCE_SEED = 20260824;

export function cleanSentence(rawLine) {
  if (!rawLine || typeof rawLine !== 'string') return null;
  let text = normalizeNfc(rawLine.trim())
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/\s+/gu, ' ')
    .trim();
  if (text.length < 10 || text.length > 300 || !/[\p{L}]/u.test(text)) return null;
  if ((text.match(/[\p{L}\p{M}]+/gu) ?? []).length < 3) return null;
  return text;
}

function normalized(text) {
  return normalizeNfc(String(text)).toLocaleLowerCase('vi-VN').replace(/\s+/gu, ' ').trim();
}

function digest(seed, text) {
  return createHash('sha256').update(`${seed}:${normalized(text)}`).digest('hex');
}

/** Deterministic whole-row split used only as fallback when Python artifacts are absent. */
export function splitCleanSources(sentences, seed = SOURCE_SEED) {
  const unique = [...new Map(sentences.map((text) => [normalized(text), text])).values()];
  const sorted = unique.map((text) => ({ text, hash: digest(seed, text) })).sort((a, b) => a.hash.localeCompare(b.hash));
  const targets = { train: sorted.length * 0.8, dev: sorted.length * 0.1, test: sorted.length * 0.1 };
  const counts = { train: 0, dev: 0, test: 0 };
  const out = { train: [], dev: [], test: [] };
  for (const entry of sorted) {
    const split = ['train', 'dev', 'test'].sort((a, b) => (targets[b] - counts[b]) - (targets[a] - counts[a]))[0];
    out[split].push(entry.text);
    counts[split]++;
  }
  return out;
}

function readLines(filePath) {
  if (!existsSync(filePath)) return [];
  return readFileSync(filePath, 'utf8').split(/\r?\n/).map(cleanSentence).filter(Boolean);
}

function writeSourceArtifacts(splits, seed = SOURCE_SEED) {
  mkdirSync(SOURCE_SPLIT_DIR, { recursive: true });
  for (const split of ['train', 'dev', 'test']) {
    writeFileSync(path.join(SOURCE_SPLIT_DIR, `clean-${split}.txt`), `${splits[split].join('\n')}\n`, 'utf8');
  }
  writeFileSync(path.join(SOURCE_SPLIT_DIR, 'clean-manifest.json'), JSON.stringify({
    format: 'clean-source-split-v1',
    seed,
    source: 'domain seed files only; Viwiki annotated documents excluded',
    rows: Object.values(splits).reduce((n, rows) => n + rows.length, 0),
    splits: Object.fromEntries(Object.entries(splits).map(([name, rows]) => [name, rows.length])),
    artifacts: { train: 'clean-train.txt', dev: 'clean-dev.txt', test: 'clean-test.txt' },
  }, null, 2) + '\n', 'utf8');
}

export function loadCleanSourceSplits(domainFiles, seed = SOURCE_SEED) {
  const files = ['train', 'dev', 'test'].map((split) => path.join(SOURCE_SPLIT_DIR, `clean-${split}.txt`));
  if (files.every(existsSync)) {
    return {
      train: readLines(files[0]),
      dev: readLines(files[1]),
      test: readLines(files[2]),
      source: 'dataset_artifacts/clean-source',
    };
  }
  const raw = domainFiles.flatMap(readLines);
  const splits = splitCleanSources(raw, seed);
  writeSourceArtifacts(splits, seed);
  return { ...splits, source: 'deterministic in-memory split persisted to dataset_artifacts/clean-source' };
}

export class VietnameseCorpusPipeline {
  constructor(options = {}) {
    this.minWordFreq = options.minWordFreq ?? 2;
    this.maxLexiconSize = options.maxLexiconSize ?? 50000;
  }

  process(sentences) {
    const unigramCounts = new Map();
    const bigramCounts = new Map();
    const trigramCounts = new Map();
    let totalTokens = 0;
    for (const sentence of sentences) {
      const words = normalized(sentence).match(/[\p{L}\p{M}]+/gu) ?? [];
      let previous = [];
      for (const word of words) {
        unigramCounts.set(word, (unigramCounts.get(word) ?? 0) + 1);
        totalTokens++;
        if (previous.length >= 1) {
          const bigram = `${previous[previous.length - 1]} ${word}`;
          bigramCounts.set(bigram, (bigramCounts.get(bigram) ?? 0) + 1);
        }
        if (previous.length >= 2) {
          const trigram = `${previous[previous.length - 2]} ${previous[previous.length - 1]} ${word}`;
          trigramCounts.set(trigram, (trigramCounts.get(trigram) ?? 0) + 1);
        }
        previous = [...previous.slice(-1), word];
      }
    }
    const lexiconEntries = [...unigramCounts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .filter(([, count]) => count >= this.minWordFreq)
      .slice(0, this.maxLexiconSize)
      .map(([word, freq]) => ({ word, freq: Math.min(990000, Math.max(1000, freq * 500)), type: freq > 1000 ? 'F' : 'A' }));
    return {
      totalSentences: sentences.length,
      totalTokens,
      uniqueWords: unigramCounts.size,
      uniqueBigrams: bigramCounts.size,
      uniqueTrigrams: trigramCounts.size,
      lexiconEntries,
      unigramCounts,
      bigramCounts,
      trigramCounts,
    };
  }

  exportToProject(sentences, stats, { syntheticSources = [], syntheticSeed = SOURCE_SEED, sourceSplit = 'test' } = {}) {
    mkdirSync(DATA_DIR, { recursive: true });
    mkdirSync(BENCHMARK_DIR, { recursive: true });
    writeFileSync(path.join(DATA_DIR, 'corpus-train.txt'), `${sentences.join('\n')}\n`, 'utf8');
    const lexLines = ['# Standardized Vietnamese Lexicon — generated from leakage-safe train split'];
    for (const entry of stats.lexiconEntries) lexLines.push(`${entry.word}\t${entry.freq}\t${entry.type}`);
    writeFileSync(path.join(DATA_DIR, 'lexicon.txt'), `${lexLines.join('\n')}\n`, 'utf8');
    const syntheticBenchmark = generateSyntheticBenchmark(syntheticSources, 200, { seed: syntheticSeed, sourceSplit });
    syntheticBenchmark.meta.trainArtifact = 'src/data/corpus-train.txt';
    syntheticBenchmark.meta.groundTruthExcludedFromTrain = true;
    writeFileSync(path.join(BENCHMARK_DIR, 'corpus-synthetic-diacritics.json'), JSON.stringify(syntheticBenchmark, null, 2), 'utf8');
    console.log(`\n✅ Pipeline export: ${sentences.length} train sentences, ${stats.lexiconEntries.length} lexicon entries, ${syntheticBenchmark.rows.length} held-out synthetic rows.`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const domainFiles = [
    path.join(DATA_DIR, 'corpus_banking_fintech.txt'),
    path.join(DATA_DIR, 'corpus_retail_ecommerce.txt'),
    path.join(DATA_DIR, 'corpus_telco_utilities_services.txt'),
  ];
  const source = loadCleanSourceSplits(domainFiles);
  const pipeline = new VietnameseCorpusPipeline({ minWordFreq: 1, maxLexiconSize: 20000 });
  const stats = pipeline.process(source.train);
  console.log(`Source split (${source.source}): train=${source.train.length} dev=${source.dev.length} test=${source.test.length}`);
  pipeline.exportToProject(source.train, stats, { syntheticSources: source.test, sourceSplit: 'test' });
}
