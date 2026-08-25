// ============================================================
// Spelling Benchmark Adapter — for VSEC & Viwiki-Spelling
// Formats external benchmark datasets into the ValidationEngine
// issue/benchmark contract for automated recall & precision testing.
// ============================================================
import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tokenize } from '../src/tokenizer.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BENCHMARK_DIR = path.join(HERE, '..', 'benchmark');

/**
 * Common Vietnamese typo mutation generator (simulating VSEC error distributions):
 * - Telex key mistakes: s/x, d/r/gi, ch/tr, c/k, ng/ngh, iu/iêu
 * - Keyboard adjacency typos on QWERTY: u/j, i/o, a/s, n/m
 * - Tone mark swaps / missing tone letters: f, s, r, x, j
 */
const TYPO_SUBSTITUTIONS = [
  { from: 'chào', to: 'chao' },
  { from: 'khách', to: 'khachs' },
  { from: 'hàng', to: 'hangf' },
  { from: 'vui', to: 'vuj' },
  { from: 'kiểm', to: 'kiemr' },
  { from: 'tra', to: 'cha' },
  { from: 'giảm', to: 'rảm' },
  { from: 'giá', to: 'dá' },
  { from: 'thông', to: 'thôgn' },
  { from: 'tin', to: 'tjn' },
  { from: 'đơn', to: 'doown' },
  { from: 'nhận', to: 'nhạn' },
  { from: 'chúc', to: 'chúsc' },
  { from: 'mừng', to: 'mừngg' },
  { from: 'ưu', to: 'uu' },
  { from: 'đãi', to: 'đải' },
  { from: 'thanh', to: 'than' },
  { from: 'toán', to: 'toasn' },
  { from: 'giao', to: 'diao' },
  { from: 'dịch', to: 'gịch' },
  { from: 'vụ', to: 'vuj' },
  { from: 'thẻ', to: 'ther' },
  { from: 'tín', to: 'tính' },
  { from: 'dụng', to: 'dụgn' },
  { from: 'tài', to: 'taif' },
  { from: 'khoản', to: 'khoanr' },
  { from: 'số', to: 'soos' },
  { from: 'tiền', to: 'tienf' },
  { from: 'hạn', to: 'hạnj' },
  { from: 'bảo', to: 'bảor' },
  { from: 'trì', to: 'chì' },
  { from: 'hotline', to: 'hotlin' },
  { from: 'voucher', to: 'vouchre' },
  { from: 'freeship', to: 'freship' }
];

/**
 * Generate VSEC-style spelling benchmark rows from clean sentences.
 */
export function generateVsecStyleSpellingBenchmark(cleanSentences, count = 100) {
  const rows = [];
  let seq = 1;

  for (let i = 0; i < cleanSentences.length && rows.length < count; i++) {
    const sentence = cleanSentences[i];
    let mutated = sentence;
    const expected = [];

    // Find any substitution matching words in this sentence
    for (const sub of TYPO_SUBSTITUTIONS) {
      const regex = new RegExp(`\\b${sub.from}\\b`, 'gi');
      if (regex.test(mutated)) {
        regex.lastIndex = 0;
        const match = regex.exec(mutated);
        if (match) {
          const start = match.index;
          mutated = mutated.slice(0, start) + sub.to + mutated.slice(start + match[0].length);
          const end = start + sub.to.length;
          expected.push({
            ruleId: 'POSSIBLE_SPELLING_ERROR',
            value: sub.to,
            suggestion: sub.from,
            span: [start, end],
          });
          break; // 1 typo per sentence for clean isolation
        }
      }
    }

    if (expected.length > 0) {
      rows.push({
        id: `VSEC_SIM_${String(seq).padStart(4, '0')}`,
        category: 'spelling-vsec',
        text: mutated,
        mode: 'ACCENTED',
        groundTruth: sentence,
        expect: expected,
        allowExtra: true,
      });
      seq++;
    }
  }

  return {
    meta: {
      description: 'VSEC-aligned Vietnamese spelling benchmark (mistyped & misspelled errors)',
      generatedCount: rows.length,
    },
    rows,
  };
}

/** CLI runner */
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  console.log('═══════ GENERATING VSEC / VIWIKI SPELLING BENCHMARK ═══════');
  const corpusPath = path.join(HERE, '..', 'src', 'data', 'corpus-train.txt');
  const sentences = readFileSync(corpusPath, 'utf8').split(/\r?\n/).filter(Boolean);

  const spellingBenchmark = generateVsecStyleSpellingBenchmark(sentences, 100);
  const outPath = path.join(BENCHMARK_DIR, 'corpus-spelling-vsec.json');
  writeFileSync(outPath, JSON.stringify(spellingBenchmark, null, 2), 'utf8');

  console.log(`✅ Saved ${spellingBenchmark.rows.length} spelling benchmark rows to ${outPath}`);
}
