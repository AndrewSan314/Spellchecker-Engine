// ============================================================
// Viwiki-Spelling external benchmark converter
// Annotated documents are test-only. No clean sentences are exported.
// Multiple mistakes in one sentence become one spelling-scoped row.
// ============================================================
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const RAW_FILE = path.join(ROOT, 'dataset_raw', 'viwiki_spelling_test.json');
const OUT_FILE = path.join(ROOT, 'benchmark', 'corpus-viwiki-spelling.json');
const DEFAULT_MAX_ROWS = 150;

/**
 * Task 1 policy: suggestions identical to the mistake itself carry zero
 * correction signal ("ràng -> ràng"), so they are dropped at retention time.
 */
export const IDENTITY_SUGGESTION_POLICY = 'drop-identity-suggestions';

function extractSentenceAround(text, startOffset, length) {
  let sentStart = 0;
  for (let i = Math.max(0, startOffset - 1); i >= 0; i--) {
    const ch = text[i];
    if (ch === '\n' || (ch === '.' && i + 1 < text.length && /\s/.test(text[i + 1]))) {
      sentStart = i + 1;
      break;
    }
  }
  let sentEnd = text.length;
  for (let i = Math.max(0, startOffset + length); i < text.length; i++) {
    const ch = text[i];
    if (ch === '\n' || (ch === '.' && (i + 1 >= text.length || /\s/.test(text[i + 1])))) {
      sentEnd = ch === '.' ? i + 1 : i;
      break;
    }
  }
  const leading = text.slice(sentStart, sentEnd).match(/^\s*/u)?.[0].length ?? 0;
  return { sentence: text.slice(sentStart, sentEnd).trim(), sentenceStart: sentStart + leading };
}

function validMistake(docText, mistake) {
  const startOffset = Number.parseInt(mistake?.start_offset, 10);
  const value = typeof mistake?.text === 'string' ? mistake.text : '';
  if (!Number.isInteger(startOffset) || startOffset < 0 || !value) return null;
  // Task 1 fix: retain a defensive copy of suggestions so addMistake() can
  // merge them; identity suggestions are dropped per IDENTITY_SUGGESTION_POLICY.
  const suggestions = Array.isArray(mistake.suggest)
    ? mistake.suggest.filter(
        (s) => typeof s === 'string' && s.trim() && s !== value,
      )
    : [];
  if (docText.slice(startOffset, startOffset + value.length) === value) {
    return { startOffset, value, suggestions };
  }
  const nearby = docText.indexOf(value, Math.max(0, startOffset - 50));
  return nearby < 0 ? null : { startOffset: nearby, value, suggestions };
}

function addMistake(group, mistake, absoluteOffset) {
  const key = `${absoluteOffset}:${mistake.value}`;
  const existing = group.mistakes.get(key)
    ?? { value: mistake.value, startOffset: absoluteOffset, suggestions: [] };
  for (const suggestion of Array.isArray(mistake.suggestions) ? mistake.suggestions : []) {
    if (suggestion && !existing.suggestions.includes(suggestion)) existing.suggestions.push(suggestion);
  }
  group.mistakes.set(key, existing);
}

/**
 * Pure conversion over already-parsed documents — unit-testable without the
 * raw dataset file (plan Task 1 Step 1).
 */
export function convertDocuments(docs, { maxRows = DEFAULT_MAX_ROWS } = {}) {
  const groups = new Map();
  let sourceMistakes = 0;
  let skippedMistakes = 0;
  for (let docIndex = 0; docIndex < docs.length; docIndex++) {
    const doc = docs[docIndex];
    const text = String(doc.text ?? '');
    for (const rawMistake of doc.mistakes ?? []) {
      sourceMistakes++;
      const mistake = validMistake(text, rawMistake);
      if (!mistake) {
        skippedMistakes++;
        continue;
      }
      const extracted = extractSentenceAround(text, mistake.startOffset, mistake.value.length);
      const sentence = extracted.sentence;
      if (sentence.length <= 15 || sentence.length >= 250 || !sentence.includes(mistake.value)) {
        skippedMistakes++;
        continue;
      }
      const groupKey = `${docIndex}:${extracted.sentenceStart}:${sentence}`;
      const group = groups.get(groupKey) ?? { docIndex, sentence, sentenceStart: extracted.sentenceStart, mistakes: new Map() };
      addMistake(group, mistake, mistake.startOffset);
      groups.set(groupKey, group);
    }
  }

  const allRows = [...groups.values()].map((group, index) => {
    const expect = [...group.mistakes.values()]
      .sort((a, b) => a.startOffset - b.startOffset || a.value.localeCompare(b.value))
      .map((mistake) => {
        const relative = mistake.startOffset - group.sentenceStart;
        return {
          ruleId: 'POSSIBLE_SPELLING_ERROR',
          value: mistake.value,
          suggestion: mistake.suggestions[0] ?? undefined,
          suggestions: mistake.suggestions,
          span: [relative, relative + mistake.value.length],
        };
      });
    return {
      id: `VIWIKI_EXT_${String(index + 1).padStart(4, '0')}`,
      category: 'spelling-viwiki-external',
      source: 'Viwiki-Spelling',
      split: 'external-test',
      text: group.sentence,
      mode: 'ACCENTED',
      expect,
      fullyLabeledRuleIds: ['POSSIBLE_SPELLING_ERROR'],
      allowExtra: false,
    };
  }).filter((row) => row.expect.length > 0);
  const rows = allRows.slice(0, maxRows);
  return {
    meta: {
      description: 'Viwiki-Spelling external/OOD spelling benchmark; never used for training or dev',
      source: 'heraclex12/Viwiki-spelling (CC BY 4.0)',
      split: 'external-test',
      documentCount: docs.length,
      sourceMistakes,
      skippedMistakes,
      groupedRows: allRows.length,
      extractedCases: rows.reduce((count, row) => count + row.expect.length, 0),
      rowLimit: maxRows,
      fullyLabeledRuleIds: ['POSSIBLE_SPELLING_ERROR'],
      generatedFromCorpusTrain: false,
      cleanTrainingArtifact: null,
    },
    rows,
  };
}

/** fixture helper for tests: one document in, first converted row out */
export function convertDocumentFixture(doc) {
  const out = convertDocuments([doc], { maxRows: 10 });
  if (!out.rows.length) throw new Error('fixture produced no rows');
  return out.rows[0];
}

export function convertViwikiExternal({ maxRows = DEFAULT_MAX_ROWS } = {}) {
  const lines = readFileSync(RAW_FILE, 'utf8').split(/\r?\n/).filter(Boolean);
  const docs = lines.map((line) => JSON.parse(line));
  const output = convertDocuments(docs, { maxRows });
  output.meta.documentCount = lines.length;
  return output;
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  const requested = Number.parseInt(process.argv[2] ?? `${DEFAULT_MAX_ROWS}`, 10);
  const output = convertViwikiExternal({ maxRows: Number.isInteger(requested) && requested > 0 ? requested : DEFAULT_MAX_ROWS });
  writeFileSync(OUT_FILE, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Saved ${output.rows.length} grouped Viwiki external rows to ${OUT_FILE}`);
  console.log(`Expected labels: ${output.meta.extractedCases}; source mistakes: ${output.meta.sourceMistakes}`);
}
