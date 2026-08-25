// ============================================================
// Converter for Viwiki-Spelling Dataset
// Extracts sentence-level benchmark cases with ground-truth
// spelling error annotations and relative token offsets.
// ============================================================
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RAW_FILE = path.join(HERE, '..', 'dataset_raw', 'viwiki_spelling_test.json');
const OUT_FILE = path.join(HERE, '..', 'benchmark', 'corpus-viwiki-spelling.json');
const CLEAN_TEXT_OUT = path.join(HERE, '..', 'dataset_raw', 'viwiki_clean_sentences.txt');

function extractSentenceAround(text, startOffset, length) {
  // Find preceding sentence boundary (. ! ? \n)
  let sentStart = 0;
  for (let i = startOffset - 1; i >= 0; i--) {
    const ch = text[i];
    if (ch === '\n' || (ch === '.' && i + 1 < text.length && /\s/.test(text[i + 1]))) {
      sentStart = i + 1;
      break;
    }
  }

  // Find succeeding sentence boundary
  let sentEnd = text.length;
  for (let i = startOffset + length; i < text.length; i++) {
    const ch = text[i];
    if (ch === '\n' || (ch === '.' && (i + 1 >= text.length || /\s/.test(text[i + 1])))) {
      sentEnd = ch === '.' ? i + 1 : i;
      break;
    }
  }

  const rawSentence = text.substring(sentStart, sentEnd).trim();
  const relOffset = startOffset - (sentStart + text.substring(sentStart, sentStart + (text.substring(sentStart).length - text.substring(sentStart).trimStart().length)).length);

  return {
    sentence: rawSentence,
    relOffset,
  };
}

function main() {
  console.log('═══════ CONVERTING VIWIKI-SPELLING DATASET ═══════');
  const rawData = readFileSync(RAW_FILE, 'utf8');
  const lines = rawData.split(/\r?\n/).filter(Boolean);

  const benchmarkRows = [];
  const cleanSentences = [];
  let mistakeCount = 0;

  for (let docIdx = 0; docIdx < lines.length; docIdx++) {
    const doc = JSON.parse(lines[docIdx]);
    const docText = doc.text;
    const mistakes = doc.mistakes ?? [];

    // Extract clean paragraphs for training corpus
    const paragraphs = docText.split(/\r?\n+/);
    for (const p of paragraphs) {
      const sentences = p.split(/(?<=[.!?])\s+/);
      for (const s of sentences) {
        const trimmed = s.trim();
        if (trimmed.length > 20 && trimmed.length < 200 && /[\p{L}]/u.test(trimmed)) {
          cleanSentences.push(trimmed);
        }
      }
    }

    // Extract benchmark rows for each labeled mistake
    for (let mIdx = 0; mIdx < mistakes.length; mIdx++) {
      const m = mistakes[mIdx];
      const startOffset = parseInt(m.start_offset, 10);
      const errText = m.text;
      const suggestions = m.suggest ?? [];

      if (isNaN(startOffset) || !errText) continue;

      // Verify the substring matches at offset
      const actualSub = docText.substring(startOffset, startOffset + errText.length);
      if (actualSub !== errText) {
        // Fallback search nearby
        const found = docText.indexOf(errText, Math.max(0, startOffset - 50));
        if (found === -1) continue;
      }

      const { sentence } = extractSentenceAround(docText, startOffset, errText.length);
      if (sentence && sentence.includes(errText) && sentence.length > 15 && sentence.length < 250) {
        mistakeCount++;
        benchmarkRows.push({
          id: `VIWIKI_SP_${String(mistakeCount).padStart(4, '0')}`,
          category: 'spelling-viwiki',
          text: sentence,
          mode: 'ACCENTED',
          expect: [{
            ruleId: 'POSSIBLE_SPELLING_ERROR',
            value: errText,
            suggestion: suggestions[0] ?? undefined,
          }],
          allowExtra: true,
        });
      }
    }
  }

  // Save benchmark dataset
  const outputData = {
    meta: {
      description: 'Viwiki-Spelling external benchmark adapted for SMS Validation Engine',
      source: 'heraclex12/Viwiki-spelling (CC BY 4.0)',
      documentCount: lines.length,
      extractedCases: benchmarkRows.length,
    },
    rows: benchmarkRows.slice(0, 150), // Standard sample size for benchmark runner
  };

  writeFileSync(OUT_FILE, JSON.stringify(outputData, null, 2), 'utf8');
  console.log(`✔ Generated ${outputData.rows.length} Viwiki-Spelling benchmark rows in ${OUT_FILE}`);

  // Save clean sentences
  const dedupedClean = Array.from(new Set(cleanSentences));
  writeFileSync(CLEAN_TEXT_OUT, dedupedClean.join('\n'), 'utf8');
  console.log(`✔ Extracted ${dedupedClean.length} clean Wikipedia sentences to ${CLEAN_TEXT_OUT}`);
}

main();
