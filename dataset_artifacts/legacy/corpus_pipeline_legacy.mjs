// ============================================================
// Vietnamese Corpus & NLP Data Pipeline — plan & roadmap
// Ingests, cleans, deduplicates Vietnamese text from open corpora
// (binhvq/news-corpus, Vietnamese Wikipedia, SMS marketing),
// generates clean Lexicon, N-gram Language Model, and Benchmarks.
// ============================================================
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeNfc, dictKey, accentKey } from '../src/normalizer.mjs';
import { generateSyntheticBenchmark } from './diacritic_generator.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(HERE, '..', 'src', 'data');
const BENCHMARK_DIR = path.join(HERE, '..', 'benchmark');

/**
 * Text cleaner and normalizer for Vietnamese sentences.
 */
export function cleanSentence(rawLine) {
  if (!rawLine || typeof rawLine !== 'string') return null;
  let text = rawLine.trim();

  // Strip Markdown / HTML tags
  text = text.replace(/<[^>]+>/g, ' ');
  text = text.replace(/&nbsp;/gi, ' ');
  text = text.replace(/&amp;/gi, '&');
  text = text.replace(/&quot;/gi, '"');
  text = text.replace(/&lt;/gi, '<');
  text = text.replace(/&gt;/gi, '>');

  // Normalize Unicode NFC
  text = normalizeNfc(text);

  // Collapse multiple whitespaces
  text = text.replace(/\s+/gu, ' ').trim();

  // Filter out noise or non-Vietnamese/garbage lines
  if (text.length < 10 || text.length > 300) return null;
  if (!/[\p{L}]/u.test(text)) return null;

  // Check basic sentence criteria (at least 3 words)
  const words = text.match(/[\p{L}\p{M}]+/gu) ?? [];
  if (words.length < 3) return null;

  return text;
}

/**
 * Pipeline processor to build Lexicon & Language Model from training sentences.
 */
export class VietnameseCorpusPipeline {
  constructor(options = {}) {
    this.minWordFreq = options.minWordFreq ?? 2;
    this.minBigramCount = options.minBigramCount ?? 2;
    this.maxLexiconSize = options.maxLexiconSize ?? 50000;
  }

  /**
   * Process a list of clean sentences into lexicon and n-gram statistics.
   */
  process(sentences) {
    const unigramCounts = new Map();
    const bigramCounts = new Map();
    const trigramCounts = new Map();
    let totalTokens = 0;

    for (const sentence of sentences) {
      const words = sentence.normalize('NFC').toLowerCase().match(/[\p{L}\p{M}]+/gu) ?? [];
      let prev2 = null;
      let prev1 = null;

      for (const w of words) {
        unigramCounts.set(w, (unigramCounts.get(w) ?? 0) + 1);
        totalTokens++;

        if (prev1 !== null) {
          const biKey = `${prev1} ${w}`;
          bigramCounts.set(biKey, (bigramCounts.get(biKey) ?? 0) + 1);

          if (prev2 !== null) {
            const triKey = `${prev2} ${prev1} ${w}`;
            trigramCounts.set(triKey, (trigramCounts.get(triKey) ?? 0) + 1);
          }
        }

        prev2 = prev1;
        prev1 = w;
      }
    }

    // Build pruned Lexicon
    const sortedWords = [...unigramCounts.entries()]
      .sort((a, b) => b[1] - a[1]);

    const lexiconEntries = sortedWords
      .filter(([w, count]) => count >= this.minWordFreq)
      .slice(0, this.maxLexiconSize)
      .map(([word, freq]) => ({
        word,
        freq: Math.min(990000, Math.max(1000, freq * 500)),
        type: freq > 1000 ? 'F' : 'A',
      }));

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

  /**
   * Export processed lexicon and training corpus into project data directory.
   */
  exportToProject(sentences, stats) {
    // 1. Export corpus-train.txt
    writeFileSync(path.join(DATA_DIR, 'corpus-train.txt'), sentences.join('\n'), 'utf8');

    // 2. Export standardized lexicon.txt
    const lexLines = ['# Standardized Vietnamese Lexicon — generated from Corpus Pipeline'];
    for (const entry of stats.lexiconEntries) {
      lexLines.push(`${entry.word}\t${entry.freq}\t${entry.type}`);
    }
    writeFileSync(path.join(DATA_DIR, 'lexicon.txt'), lexLines.join('\n'), 'utf8');

    // 3. Generate multi-pattern synthetic missing diacritic benchmark
    const syntheticBenchmark = generateSyntheticBenchmark(sentences, 200);
    writeFileSync(
      path.join(BENCHMARK_DIR, 'corpus-synthetic-diacritics.json'),
      JSON.stringify(syntheticBenchmark, null, 2),
      'utf8'
    );

    console.log(`\n✅ Pipeline Export Successful:`);
    console.log(`- Corpus sentences: ${sentences.length} saved to src/data/corpus-train.txt`);
    console.log(`- Lexicon entries: ${stats.lexiconEntries.length} saved to src/data/lexicon.txt`);
    console.log(`- Synthetic Benchmark: ${syntheticBenchmark.rows.length} rows saved to benchmark/corpus-synthetic-diacritics.json`);
  }
}

/** CLI runner */
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  console.log('═══════ RUNNING VIETNAMESE CORPUS PIPELINE ═══════');

  // Load domain corpus files + Wikipedia clean corpus
  const domainFiles = [
    path.join(DATA_DIR, 'corpus_banking_fintech.txt'),
    path.join(DATA_DIR, 'corpus_retail_ecommerce.txt'),
    path.join(DATA_DIR, 'corpus_telco_utilities_services.txt'),
    path.join(DATA_DIR, 'corpus-train.txt'),
    path.join(HERE, '..', 'dataset_raw', 'viwiki_clean_sentences.txt')
  ];

  const rawSentences = [];
  for (const fullPath of domainFiles) {
    if (existsSync(fullPath)) {
      const lines = readFileSync(fullPath, 'utf8').split(/\r?\n/);
      for (const line of lines) {
        const cleaned = cleanSentence(line);
        if (cleaned) rawSentences.push(cleaned);
      }
    }
  }

  const dedupedSentences = Array.from(new Set(rawSentences));
  console.log(`Ingested ${rawSentences.length} lines -> ${dedupedSentences.length} unique clean sentences.`);

  const pipeline = new VietnameseCorpusPipeline({
    minWordFreq: 1,
    maxLexiconSize: 20000,
  });

  const stats = pipeline.process(dedupedSentences);
  console.log(`Vocabulary stats: ${stats.uniqueWords} unigrams, ${stats.uniqueBigrams} bigrams, ${stats.uniqueTrigrams} trigrams.`);

  pipeline.exportToProject(dedupedSentences, stats);
}
