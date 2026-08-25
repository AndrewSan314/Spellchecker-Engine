// ============================================================
// Synthetic Missing Diacritics Benchmark Generator
// Implements 4 realistic degradation strategies from clean text:
// 1. ALL_STRIPPED: 100% stripped (classical unaccented text)
// 2. SINGLE_TOKEN: Exactly 1 token stripped (subtle missing accent)
// 3. FEW_TOKENS: 2-3 tokens stripped
// 4. MIXED: Realistic SMS typo pattern (random selective stripping)
// ============================================================
import { accentKey, hasVietnameseAccent } from '../src/normalizer.mjs';
import { tokenize } from '../src/tokenizer.mjs';

export const DiacriticMode = Object.freeze({
  ALL_STRIPPED: 'ALL_STRIPPED',
  SINGLE_TOKEN: 'SINGLE_TOKEN',
  FEW_TOKENS: 'FEW_TOKENS',
  MIXED: 'MIXED',
});

/**
 * Strips diacritics from selected words in a clean Vietnamese sentence
 * and returns a labeled benchmark row with exact expected issues.
 *
 * @param {string} cleanText Clean, accented Vietnamese sentence (ground truth)
 * @param {DiacriticMode} mode Degradation mode
 * @param {string} id Unique benchmark ID
 * @param {string} [category="missing-diacritic"]
 * @returns {object|null} Benchmark row { id, text, mode: "ACCENTED", expect: [...], groundTruth }
 */
export function generateDiacriticBenchmarkRow(cleanText, mode = DiacriticMode.MIXED, id = 'SYN_001', category = 'missing-diacritic') {
  const tokens = tokenize(cleanText);
  const accentedWordIndices = [];

  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.type === 'WORD' && hasVietnameseAccent(t.original)) {
      accentedWordIndices.push(i);
    }
  }

  if (accentedWordIndices.length === 0) return null; // No accents to strip

  let targetIndices = new Set();

  switch (mode) {
    case DiacriticMode.ALL_STRIPPED:
      targetIndices = new Set(accentedWordIndices);
      break;

    case DiacriticMode.SINGLE_TOKEN: {
      const chosen = accentedWordIndices[Math.floor(Math.random() * accentedWordIndices.length)];
      targetIndices.add(chosen);
      break;
    }

    case DiacriticMode.FEW_TOKENS: {
      const count = Math.min(accentedWordIndices.length, Math.floor(Math.random() * 2) + 2); // 2 or 3
      const shuffled = [...accentedWordIndices].sort(() => Math.random() - 0.5);
      targetIndices = new Set(shuffled.slice(0, count));
      break;
    }

    case DiacriticMode.MIXED:
    default: {
      // Pick 30% to 70% of accented words
      const ratio = 0.3 + Math.random() * 0.4;
      for (const idx of accentedWordIndices) {
        if (Math.random() < ratio || targetIndices.size === 0) {
          targetIndices.add(idx);
        }
      }
      break;
    }
  }

  let reconstructed = '';
  const expect = [];

  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (targetIndices.has(i)) {
      const stripped = accentKey(t.original);
      const start = reconstructed.length;
      reconstructed += stripped;
      const end = reconstructed.length;

      expect.push({
        ruleId: 'POSSIBLE_MISSING_DIACRITIC',
        value: stripped,
        suggestion: t.original.toLowerCase(),
        span: [start, end],
      });
    } else {
      reconstructed += t.original;
    }
  }

  return {
    id,
    category,
    mode: 'ACCENTED',
    degradationType: mode,
    text: reconstructed,
    groundTruth: cleanText,
    expect,
  };
}

/**
 * Batch generate synthetic dataset from a list of clean sentences.
 */
export function generateSyntheticBenchmark(cleanSentences, targetCount = 200) {
  const rows = [];
  const modes = [
    DiacriticMode.ALL_STRIPPED,
    DiacriticMode.SINGLE_TOKEN,
    DiacriticMode.FEW_TOKENS,
    DiacriticMode.MIXED,
  ];

  let seq = 1;
  let idx = 0;

  while (rows.length < targetCount && idx < cleanSentences.length * 4) {
    const sentence = cleanSentences[idx % cleanSentences.length];
    const mode = modes[seq % modes.length];
    const row = generateDiacriticBenchmarkRow(sentence, mode, `SYN_MD_${String(seq).padStart(4, '0')}`);
    if (row && row.expect.length > 0) {
      rows.push(row);
      seq++;
    }
    idx++;
  }

  return {
    meta: {
      description: 'Synthetically generated missing diacritics benchmark from clean Vietnamese corpus',
      generatedCount: rows.length,
      modesDistribution: {
        ALL_STRIPPED: rows.filter(r => r.degradationType === DiacriticMode.ALL_STRIPPED).length,
        SINGLE_TOKEN: rows.filter(r => r.degradationType === DiacriticMode.SINGLE_TOKEN).length,
        FEW_TOKENS: rows.filter(r => r.degradationType === DiacriticMode.FEW_TOKENS).length,
        MIXED: rows.filter(r => r.degradationType === DiacriticMode.MIXED).length,
      }
    },
    rows,
  };
}
