#!/usr/bin/env node
// ============================================================
// Builds the in-repo Vietnamese SMS dataset from dataset_sms/templates.json.
//
// Why this exists: every number the engine reports today comes from VSEC
// (Wikipedia-style prose) whose error distribution overlaps its own training
// split by ~68% (tools/measure_error_channel_overlap.mjs). Nothing in the repo
// measured the engine on the domain it actually serves — brandname SMS. This
// generator produces that domain: clean messages for training a compact
// domain language model, and noisy messages with labelled correction pairs
// for evaluation.
//
// Properties that matter:
//   - DETERMINISTIC: same templates + same seed => byte-identical output.
//   - GROUP-SPLIT: a template group lands entirely in train, dev or test, so
//     no phrasing is shared across splits (the leakage the review flagged).
//   - SLOT-SAFE: errors are injected ONLY into the hand-written literal text,
//     never into slot values (URLs, money, dates, codes, phone numbers) —
//     those exist to exercise protected ranges and must stay intact.
//
// Usage:
//   node tools/build_sms_dataset.mjs            # write dataset_sms/ + benchmark/sms/
//   node tools/build_sms_dataset.mjs --seed 7   # different draw
// ============================================================
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { accentKey, hasVietnameseAccent } from '../src/normalizer.mjs';
import { LexiconService, AccentIndex } from '../src/lexical.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const OUT_DIR = path.join(ROOT, 'dataset_sms');
const BENCH_DIR = path.join(ROOT, 'benchmark', 'sms');

// renders per template — train groups get more text because the LM trains on
// them; dev/test stay small so evaluation is quick.
const RENDERS = { train: 40, dev: 16, test: 16 };

const PROFILE_WEIGHTS = [
  ['CLEAN', 20],
  ['ALL_UNACCENTED', 20],
  ['SOME_UNACCENTED', 20],
  ['TELEX', 10],
  ['WRONG_DIACRITIC', 10],
  ['TYPO', 10],
  ['BOUNDARY', 5],
  ['PUNCT', 5],
];

const RULE_FOR_PROFILE = {
  ALL_UNACCENTED: 'POSSIBLE_MISSING_DIACRITIC',
  SOME_UNACCENTED: 'POSSIBLE_MISSING_DIACRITIC',
  TELEX: 'POSSIBLE_SPELLING_ERROR',
  WRONG_DIACRITIC: 'POSSIBLE_SPELLING_ERROR',
  TYPO: 'POSSIBLE_SPELLING_ERROR',
  BOUNDARY: 'POSSIBLE_WORD_BOUNDARY_ERROR',
};

/** mulberry32 — same generator the existing synthetic tooling uses */
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

function sha256(text) {
  return createHash('sha256').update(text).digest('hex');
}

/** stable group -> split assignment (70/15/15) from the group name alone */
function splitOf(group) {
  const h = parseInt(sha256(`split:${group}`).slice(0, 8), 16) % 100;
  if (h < 70) return 'train';
  return h < 85 ? 'dev' : 'test';
}

// ---------- diacritic helpers ----------
const TONE_LETTER = { '̀': 'f', '́': 's', '̉': 'r', '̃': 'x', '̣': 'j' };
const TONE_MARKS = new Set(Object.keys(TONE_LETTER));

/** "hàng" -> {base:"hang", tone:"f"} keeping quality marks ("hưởng" -> "hương"/"r") */
function splitTone(word) {
  const nfd = word.normalize('NFD');
  let tone = null;
  let out = '';
  for (const ch of nfd) {
    if (TONE_MARKS.has(ch)) { tone = TONE_LETTER[ch]; continue; }
    out += ch;
  }
  return { base: out.normalize('NFC'), tone };
}

/** Telex leftover typo: "hàng" -> "hangf" (the engine's telexProven shape) */
function telexTypo(word) {
  const { base, tone } = splitTone(word);
  if (!tone) return null;
  return `${base}${tone}`;
}

function stripAccents(word) {
  return accentKey(word) === word.toLowerCase() ? null : accentKey(word);
}

/** preserve the original capitalisation pattern of the first letter */
function matchCase(original, replacement) {
  if (!original || !replacement) return replacement;
  if (original[0] === original[0].toLocaleUpperCase('vi-VN')
    && original[0] !== original[0].toLocaleLowerCase('vi-VN')) {
    return replacement[0].toLocaleUpperCase('vi-VN') + replacement.slice(1);
  }
  return replacement;
}

// ---------- rendering ----------
/**
 * Renders one template into { text, protectedSpans } where protectedSpans
 * cover every slot value (never touched by error injection).
 */
function render(template, slots, rnd) {
  let text = '';
  const protectedSpans = [];
  const re = /\{(\w+)\}/g;
  let last = 0;
  let m;
  while ((m = re.exec(template.text)) !== null) {
    text += template.text.slice(last, m.index);
    const pool = slots[m[1]];
    if (!pool) throw new Error(`${template.id}: unknown slot {${m[1]}}`);
    const value = pool[Math.floor(rnd() * pool.length) % pool.length];
    protectedSpans.push([text.length, text.length + value.length]);
    text += value;
    last = m.index + m[0].length;
  }
  text += template.text.slice(last);
  return { text, protectedSpans };
}

/** word tokens (letters only) outside every protected span */
function injectableWords(text, protectedSpans) {
  const words = [];
  const re = /[\p{L}\p{M}]+/gu;
  let m;
  while ((m = re.exec(text)) !== null) {
    const start = m.index;
    const end = start + m[0].length;
    if (protectedSpans.some(([s, e]) => start < e && end > s)) continue;
    if (m[0].length < 2) continue;
    words.push({ word: m[0], start, end });
  }
  return words;
}

/** apply edits (non-overlapping, sorted) and return the new text + pairs */
function applyEdits(text, edits) {
  const sorted = [...edits].sort((a, b) => a.start - b.start);
  let out = '';
  let last = 0;
  const pairs = [];
  for (const edit of sorted) {
    out += text.slice(last, edit.start);
    pairs.push({
      error: edit.to,
      correction: edit.from,
      start: out.length,
      end: out.length + edit.to.length,
    });
    out += edit.to;
    last = edit.end;
  }
  out += text.slice(last);
  return { text: out, pairs };
}

// ---------- error profiles ----------
function buildEdits(profile, text, protectedSpans, rnd, accentIndex) {
  const words = injectableWords(text, protectedSpans);
  const accented = words.filter((w) => hasVietnameseAccent(w.word));
  const pick = (list, n) => {
    const copy = [...list];
    const chosen = [];
    while (copy.length > 0 && chosen.length < n) {
      chosen.push(copy.splice(Math.floor(rnd() * copy.length), 1)[0]);
    }
    return chosen;
  };

  switch (profile) {
    case 'ALL_UNACCENTED': {
      return accented
        .map((w) => ({ ...w, from: w.word, to: stripAccents(w.word) }))
        .filter((e) => e.to && e.to !== e.from)
        .map((e) => ({ ...e, to: matchCase(e.from, e.to) }));
    }
    case 'SOME_UNACCENTED': {
      return pick(accented, 1 + Math.floor(rnd() * 3))
        .map((w) => ({ ...w, from: w.word, to: stripAccents(w.word) }))
        .filter((e) => e.to && e.to !== e.from)
        .map((e) => ({ ...e, to: matchCase(e.from, e.to) }));
    }
    case 'TELEX': {
      return pick(accented, 1 + Math.floor(rnd() * 2))
        .map((w) => ({ ...w, from: w.word, to: telexTypo(w.word.toLowerCase()) }))
        .filter((e) => e.to)
        .map((e) => ({ ...e, to: matchCase(e.from, e.to) }));
    }
    case 'WRONG_DIACRITIC': {
      // real-word same-key swap ("quý" -> "quỳ"): the ACCENTED_SAME_KEY class
      const edits = [];
      for (const w of pick(accented, 4)) {
        const twins = accentIndex.candidates(accentKey(w.word))
          .map((c) => c.word.toLowerCase())
          .filter((s) => s !== w.word.toLowerCase() && hasVietnameseAccent(s));
        if (twins.length === 0) continue;
        edits.push({ ...w, from: w.word, to: matchCase(w.word, twins[0]) });
        if (edits.length >= 1 + Math.floor(rnd() * 2)) break;
      }
      return edits;
    }
    case 'TYPO': {
      // adjacent transposition inside the word -> a non-word ("khách"->"khcáh")
      return pick(words.filter((w) => w.word.length >= 4), 1 + Math.floor(rnd() * 2))
        .map((w) => {
          const chars = [...w.word];
          const i = 1 + Math.floor(rnd() * (chars.length - 2));
          [chars[i], chars[i + 1]] = [chars[i + 1], chars[i]];
          return { ...w, from: w.word, to: chars.join('') };
        })
        .filter((e) => e.to !== e.from);
    }
    case 'BOUNDARY': {
      // merge two adjacent injectable words ("cảm ơn" -> "cảmơn")
      for (const i of [...words.keys()].sort(() => rnd() - 0.5)) {
        const a = words[i];
        const b = words[i + 1];
        if (!b || b.start !== a.end + 1 || text[a.end] !== ' ') continue;
        return [{
          start: a.start, end: b.end,
          from: `${a.word} ${b.word}`, to: `${a.word}${b.word}`,
        }];
      }
      return [];
    }
    default:
      return [];
  }
}

/** deterministic punctuation/whitespace damage with its own expectations */
function punctuationDamage(text, rnd) {
  const variants = [
    { find: /, /, to: ' , ', ruleId: 'WHITESPACE_BEFORE_PUNCTUATION' },
    { find: /\.$/, to: '..', ruleId: 'REPEATED_PUNCTUATION' },
    { find: /, /, to: ',', ruleId: 'MISSING_WHITESPACE_AFTER_PUNCTUATION' },
    { find: / /, to: '  ', ruleId: 'MULTIPLE_WHITESPACE' },
  ];
  const order = [...variants.keys()].sort(() => rnd() - 0.5);
  for (const i of order) {
    const v = variants[i];
    if (!v.find.test(text)) continue;
    return { text: text.replace(v.find, v.to), expect: [{ ruleId: v.ruleId }] };
  }
  return null;
}

// ---------- main ----------
function build({ seed = 20260903 } = {}) {
  const spec = JSON.parse(readFileSync(path.join(OUT_DIR, 'templates.json'), 'utf8'));
  const lexicon = LexiconService.load();
  const accentIndex = AccentIndex.build(lexicon);
  const rnd = seededRandom(seed);

  const rows = { train: [], dev: [], test: [] };
  const cleanText = { train: [], dev: [], test: [] };
  const profileCursor = [];
  for (const [name, weight] of PROFILE_WEIGHTS) {
    for (let i = 0; i < weight; i++) profileCursor.push(name);
  }

  for (const template of spec.templates) {
    const split = splitOf(template.group);
    const renders = RENDERS[split];
    for (let r = 0; r < renders; r++) {
      const { text: clean, protectedSpans } = render(template, spec.slots, rnd);
      cleanText[split].push(clean);
      const profile = profileCursor[Math.floor(rnd() * profileCursor.length)];
      const id = `${template.id}-${String(r).padStart(3, '0')}`;

      if (profile === 'CLEAN') {
        rows[split].push({
          id, group: template.group, domain: template.domain, split,
          profile, text: clean, clean, correction_pairs: [], expect: [],
        });
        continue;
      }
      if (profile === 'PUNCT') {
        const damaged = punctuationDamage(clean, rnd);
        if (!damaged) continue;
        rows[split].push({
          id, group: template.group, domain: template.domain, split,
          profile, text: damaged.text, clean, correction_pairs: [],
          expect: damaged.expect,
        });
        continue;
      }

      const edits = buildEdits(profile, clean, protectedSpans, rnd, accentIndex);
      if (edits.length === 0) continue;
      const { text, pairs } = applyEdits(clean, edits);
      const ruleId = RULE_FOR_PROFILE[profile];
      rows[split].push({
        id, group: template.group, domain: template.domain, split, profile,
        text, clean,
        correction_pairs: pairs,
        expect: pairs.map((p) => ({
          ruleId, value: p.error, suggestion: p.correction,
          positionStart: p.start, positionEnd: p.end,
        })),
      });
    }
  }

  mkdirSync(OUT_DIR, { recursive: true });
  mkdirSync(BENCH_DIR, { recursive: true });
  const manifest = {
    schema: 'sms-dataset-v1',
    generatedBy: 'tools/build_sms_dataset.mjs',
    seed,
    templates: spec.templates.length,
    groups: [...new Set(spec.templates.map((t) => t.group))].length,
    splitPolicy: 'by template group (sha256(group) % 100 -> 70/15/15); no group appears in two splits',
    counts: {},
    files: {},
  };

  for (const split of ['train', 'dev', 'test']) {
    const jsonl = `${rows[split].map((r) => JSON.stringify(r)).join('\n')}\n`;
    const file = path.join(OUT_DIR, `sms-${split}.jsonl`);
    writeFileSync(file, jsonl);
    manifest.counts[split] = {
      rows: rows[split].length,
      labels: rows[split].reduce((n, r) => n + r.expect.length, 0),
      cleanRows: rows[split].filter((r) => r.expect.length === 0).length,
      byProfile: rows[split].reduce((acc, r) => {
        acc[r.profile] = (acc[r.profile] ?? 0) + 1; return acc;
      }, {}),
    };
    manifest.files[path.basename(file)] = sha256(jsonl);
  }

  // TRAIN-ONLY clean corpus: the only text the domain LM is allowed to see.
  const trainCorpus = `${[...new Set(cleanText.train)].join('\n')}\n`;
  const corpusFile = path.join(OUT_DIR, 'sms-clean-train.txt');
  writeFileSync(corpusFile, trainCorpus);
  manifest.files['sms-clean-train.txt'] = sha256(trainCorpus);
  manifest.counts.trainCorpusLines = trainCorpus.trim().split('\n').length;

  // Benchmark shards (kept OUT of benchmark/ root so the tracked benchmark
  // totals stay comparable — benchmark/run-benchmark.mjs globs corpus*.json).
  for (const split of ['dev', 'test']) {
    const body = JSON.stringify({
      meta: {
        source: 'dataset_sms (synthetic, authored in-repo)',
        split,
        note: 'Rows carry fullyLabeledRuleIds so extras outside the labelled lanes are not counted as false positives.',
      },
      rows: rows[split].map((r) => ({
        id: r.id,
        category: `sms-${r.profile.toLowerCase().replace(/_/g, '-')}`,
        text: r.text,
        mode: 'ACCENTED',
        brand: 'VT_TENDOO',
        fullyLabeledRuleIds: ['POSSIBLE_MISSING_DIACRITIC', 'POSSIBLE_SPELLING_ERROR',
          'POSSIBLE_WORD_BOUNDARY_ERROR'],
        expect: r.expect,
      })),
    }, null, 2);
    const file = path.join(BENCH_DIR, `corpus-sms-${split}.json`);
    writeFileSync(file, `${body}\n`);
    manifest.files[`benchmark/sms/${path.basename(file)}`] = sha256(`${body}\n`);
  }

  const manifestBody = `${JSON.stringify(manifest, null, 2)}\n`;
  writeFileSync(path.join(OUT_DIR, 'manifest.json'), manifestBody);
  return manifest;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const seedArg = process.argv.indexOf('--seed');
  const manifest = build({ seed: seedArg > 0 ? Number(process.argv[seedArg + 1]) : undefined });
  console.log('SMS dataset built');
  console.log(`  templates=${manifest.templates} groups=${manifest.groups} seed=${manifest.seed}`);
  for (const split of ['train', 'dev', 'test']) {
    const c = manifest.counts[split];
    console.log(`  ${split.padEnd(5)} rows=${String(c.rows).padStart(4)} labels=${String(c.labels).padStart(4)}`
      + ` clean=${String(c.cleanRows).padStart(3)}  ${JSON.stringify(c.byProfile)}`);
  }
  console.log(`  train LM corpus lines=${manifest.counts.trainCorpusLines}`);
}

export { build, splitOf, telexTypo, splitTone };
