// ============================================================
// Deterministic synthetic missing-diacritic benchmark generator
// ============================================================
import { createHash } from 'node:crypto';
import { accentKey, hasVietnameseAccent } from '../src/normalizer.mjs';
import { tokenize } from '../src/tokenizer.mjs';

export const DiacriticMode = Object.freeze({
  ALL_STRIPPED: 'ALL_STRIPPED',
  SINGLE_TOKEN: 'SINGLE_TOKEN',
  FEW_TOKENS: 'FEW_TOKENS',
  MIXED: 'MIXED',
});

function seededRandom(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fingerprint(text) {
  return createHash('sha256').update(String(text).normalize('NFC').toLocaleLowerCase('vi-VN').replace(/\s+/gu, ' ').trim()).digest('hex');
}

/**
 * Strip accents from deterministic selected words and retain source metadata.
 */
export function generateDiacriticBenchmarkRow(
  cleanText,
  mode = DiacriticMode.MIXED,
  id = 'SYN_001',
  category = 'missing-diacritic',
  options = {},
) {
  const tokens = tokenize(cleanText);
  const accentedWordIndices = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token.type === 'WORD' && hasVietnameseAccent(token.original)) accentedWordIndices.push(i);
  }
  if (accentedWordIndices.length === 0) return null;

  const random = options.random ?? seededRandom(options.seed ?? 20260824);
  let targetIndices = new Set();
  switch (mode) {
    case DiacriticMode.ALL_STRIPPED:
      targetIndices = new Set(accentedWordIndices);
      break;
    case DiacriticMode.SINGLE_TOKEN:
      targetIndices.add(accentedWordIndices[Math.floor(random() * accentedWordIndices.length)]);
      break;
    case DiacriticMode.FEW_TOKENS: {
      const count = Math.min(accentedWordIndices.length, Math.floor(random() * 2) + 2);
      targetIndices = new Set([...accentedWordIndices].sort(() => random() - 0.5).slice(0, count));
      break;
    }
    case DiacriticMode.MIXED:
    default: {
      const ratio = 0.3 + random() * 0.4;
      for (const index of accentedWordIndices) {
        if (random() < ratio || targetIndices.size === 0) targetIndices.add(index);
      }
    }
  }

  let reconstructed = '';
  const expect = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (!targetIndices.has(i)) {
      reconstructed += token.original;
      continue;
    }
    const stripped = accentKey(token.original);
    const start = reconstructed.length;
    reconstructed += stripped;
    expect.push({
      ruleId: 'POSSIBLE_MISSING_DIACRITIC',
      value: stripped,
      suggestion: token.original.toLowerCase(),
      span: [start, reconstructed.length],
    });
  }
  return {
    id,
    category,
    mode: 'ACCENTED',
    degradationType: mode,
    text: reconstructed,
    groundTruth: cleanText,
    source: 'clean-source-heldout',
    sourceSplit: options.sourceSplit ?? 'test',
    sourceIndex: options.sourceIndex ?? null,
    sourceFingerprint: options.sourceFingerprint ?? fingerprint(cleanText),
    seed: options.seed ?? null,
    expect,
  };
}

/** Generate rows only from the caller-provided held-out clean source. */
export function generateSyntheticBenchmark(cleanSentences, targetCount = 200, options = {}) {
  const rows = [];
  const sources = [...new Set((cleanSentences ?? []).filter((sentence) => typeof sentence === 'string' && sentence.trim()))];
  if (sources.length === 0) {
    return {
      meta: { description: 'Synthetic benchmark has no held-out clean source rows', generatedCount: 0, seed: options.seed ?? 20260824, sourceSplit: options.sourceSplit ?? 'test' },
      rows,
    };
  }
  const modes = [DiacriticMode.ALL_STRIPPED, DiacriticMode.SINGLE_TOKEN, DiacriticMode.FEW_TOKENS, DiacriticMode.MIXED];
  const seed = options.seed ?? 20260824;
  let sequence = 1;
  for (let index = 0; rows.length < targetCount && index < sources.length * 8; index++) {
    const sourceIndex = index % sources.length;
    const random = seededRandom(seed + index);
    const row = generateDiacriticBenchmarkRow(
      sources[sourceIndex],
      modes[sequence % modes.length],
      `SYN_MD_${String(sequence).padStart(4, '0')}`,
      'missing-diacritic',
      { random, seed, sourceIndex, sourceSplit: options.sourceSplit ?? 'test', sourceFingerprint: fingerprint(sources[sourceIndex]) },
    );
    if (row && row.expect.length > 0) {
      rows.push(row);
      sequence++;
    }
  }
  return {
    meta: {
      description: 'Synthetic missing-diacritic benchmark from held-out clean source (source ground truth excluded from corpus-train)',
      generatedCount: rows.length,
      seed,
      sourceSplit: options.sourceSplit ?? 'test',
      sourceRows: sources.length,
      modesDistribution: Object.fromEntries(modes.map((mode) => [mode, rows.filter((row) => row.degradationType === mode).length])),
    },
    rows,
  };
}
