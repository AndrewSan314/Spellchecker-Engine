#!/usr/bin/env node
// ============================================================
// Builds the COMPACT "lite" serving profile: an SMS-domain language model
// AND the matching pruned lexicon.
//
// Problem it solves (local runs): src/data/lm-ngrams.tsv is 80 MB holding
// 200k unigrams + 1M bigrams + 2.76M trigrams. Node keeps those as Map<string>
// entries, so the engine costs ~6 s of cold start and ~830 MB RSS before it
// answers a single message — painful for local testing, and most of that mass
// is Wikipedia/news vocabulary an SMS validator never sees.
//
// What this builder does:
//   1. computes a DOMAIN vocabulary from the SMS training corpus, the three
//      in-repo domain corpora, and the accent twins of those words (candidate
//      surfaces must stay scorable, otherwise the ranker goes blind);
//   2. keeps the general model's REAL counts for those words only, capped by
//      frequency, dropping every n-gram whose tokens fall outside the domain;
//   3. overlays counts from dataset_sms/sms-clean-train.txt (TRAIN split only)
//      with a weight, so SMS collocations like "mã OTP" / "hạn thanh toán"
//      become attested;
//   4. writes the same TSV format the engine already validates, so this is a
//      drop-in artifact and not a second code path;
//   5. emits src/data/lexicon-sms.txt with EXACTLY the same vocabulary, so the
//      SymSpell deletion index (~200 MB over the 51k-word lexicon) shrinks
//      with it and the two artifacts can never disagree about which words
//      exist.
//
// Probabilities of KEPT words are preserved: pUni divides by
// (totalTokens + 0.5*vocabSize) and totalTokens (~1.3e9) dominates, so
// shrinking the vocabulary from 200k to ~60k moves pUni by <0.01%. Thresholds
// calibrated on the full artifact therefore stay meaningful.
//
// LEAKAGE: reads dataset_sms/sms-train.jsonl's clean corpus ONLY. dev/test
// never touch this file — test/test_sms_dataset.mjs asserts it.
//
// Usage:
//   node tools/build_sms_profile.mjs
//   node tools/build_sms_profile.mjs --max-bigrams 400000 --max-trigrams 500000
// ============================================================
import { createReadStream, readFileSync, writeFileSync, statSync, existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { accentKey } from '../src/normalizer.mjs';
import { LexiconService, AccentIndex } from '../src/lexical.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const DATA = path.join(ROOT, 'src', 'data');

const DEFAULTS = {
  source: path.join(DATA, 'lm-ngrams.tsv'),
  out: path.join(DATA, 'lm-ngrams.sms.tsv'),
  outLexicon: path.join(DATA, 'lexicon-sms.txt'),
  smsCorpus: path.join(ROOT, 'dataset_sms', 'sms-clean-train.txt'),
  maxUnigrams: 60_000,
  // Chosen on the SMS dev split (never test).
  //
  // The first cut used a per-ROW frequency cap and a smaller budget. It was
  // wrong twice over: it split candidate competitions apart (see the cap
  // below) and it left the model self-contradictory (see the back-off
  // repair). With those two fixed, the budget buys real quality:
  //   B/T        LM      RSS     cold   dev recall  precision
  //   120k/120k   4.3 MB  249 MB  0.7 s  0.624       0.963
  //   300k/300k  10.5 MB  309 MB  1.1 s  0.739       0.976   <- default
  //   500k/500k  20.3 MB  418 MB  1.5 s  0.734       0.965
  // 500k is not better than 300k: past that point the rows added are rare
  // n-grams that mostly add noise. 300k is the knee.
  maxBigrams: 300_000,
  maxTrigrams: 300_000,
  domainWeight: 120,
  // rivals retained per accent-stripped key (see the cap below): enough to
  // decide any realistic competition, small enough that long tails of a
  // frequent key cannot starve other groups
  maxRivalsPerGroup: 6,
  generalVocabTop: 20_000, // most frequent general words kept regardless
};

function tokenizeLine(line) {
  // same cleanup as NGramLanguageModel._loadRawText: promo codes like
  // "GIAM50K" must not leak phantom words into the model
  const cleaned = line.replace(/[0-9\p{L}][0-9\p{L}.-]*[0-9\p{L}]/gu,
    (m) => (/\d/.test(m) ? ' ' : m));
  return cleaned.normalize('NFC').toLowerCase().match(/[\p{L}\p{M}]+/gu) ?? [];
}

function readCorpusWords(files) {
  const lines = [];
  for (const file of files) {
    if (!existsSync(file)) continue;
    for (const raw of readFileSync(file, 'utf8').split(/\r?\n/)) {
      const line = raw.trim();
      if (line && !line.startsWith('#')) lines.push(line);
    }
  }
  return lines;
}

/** counts n-grams of a corpus, returns {uni, bi, tri, tokens} */
function countCorpus(lines, weight) {
  const uni = new Map();
  const bi = new Map();
  const tri = new Map();
  let tokens = 0;
  for (const line of lines) {
    const words = tokenizeLine(line);
    let p1 = null;
    let p2 = null;
    for (const w of words) {
      uni.set(w, (uni.get(w) ?? 0) + weight);
      tokens += weight;
      if (p1 !== null) {
        const k = `${p1} ${w}`;
        bi.set(k, (bi.get(k) ?? 0) + weight);
      }
      if (p2 !== null) {
        const k = `${p2} ${p1} ${w}`;
        tri.set(k, (tri.get(k) ?? 0) + weight);
      }
      p2 = p1;
      p1 = w;
    }
  }
  return { uni, bi, tri, tokens };
}

export async function buildSmsProfile(opts = {}) {
  const cfg = { ...DEFAULTS, ...opts };
  const t0 = Date.now();

  // ---------- 1. domain vocabulary ----------
  const domainCorpusFiles = [
    cfg.smsCorpus,
    path.join(DATA, 'corpus_banking_fintech.txt'),
    path.join(DATA, 'corpus_retail_ecommerce.txt'),
    path.join(DATA, 'corpus_telco_utilities_services.txt'),
    path.join(DATA, 'corpus-train.txt'),
  ];
  const domainLines = readCorpusWords(domainCorpusFiles);
  const smsLines = readCorpusWords([cfg.smsCorpus]);
  const domainWords = new Set();
  for (const line of domainLines) for (const w of tokenizeLine(line)) domainWords.add(w);

  // accent twins: a candidate surface the ranker may propose must remain
  // scorable, otherwise every rival collapses to the same smoothed floor.
  const lexicon = LexiconService.load(path.join(DATA, 'lexicon-built.txt'));
  const accentIndex = AccentIndex.build(lexicon);
  const keep = new Set(domainWords);
  for (const w of domainWords) {
    for (const cand of accentIndex.candidates(accentKey(w))) {
      keep.add(cand.word.toLowerCase());
    }
  }
  // Snapshot BEFORE the general top-N words are merged in: this is the set the
  // n-gram cap protects. It deliberately includes accent twins, so BOTH sides
  // of a candidate competition survive or fall together — see the cap below.
  const domainVocab = new Set(keep);
  const domainOnly = keep.size;

  // ---------- 2. stream the general artifact ----------
  const uni = new Map();
  const biRows = [];
  const triRows = [];
  let headerTokens = 0;
  let generalRank = 0;
  let seenU = 0;
  let seenB = 0;
  let seenT = 0;

  const rl = createInterface({
    input: createReadStream(cfg.source, { encoding: 'utf8' }),
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    if (!line) continue;
    if (line[0] === '#') {
      const m = /^#tokens=(\d+)$/.exec(line);
      if (m) headerTokens = Number(m[1]);
      continue;
    }
    const kind = line[0];
    const i2 = line.lastIndexOf('\t');
    const key = line.slice(2, i2);
    const count = Number(line.slice(i2 + 1));
    if (kind === 'U') {
      seenU += 1;
      generalRank += 1;
      // the artifact is frequency-sorted inside each section
      if (keep.has(key) || generalRank <= cfg.generalVocabTop) {
        keep.add(key);
        uni.set(key, count);
      }
    } else if (kind === 'B') {
      seenB += 1;
      const sp = key.indexOf(' ');
      if (uni.has(key.slice(0, sp)) && uni.has(key.slice(sp + 1))) {
        biRows.push([key, count]);
      }
    } else {
      seenT += 1;
      const a = key.indexOf(' ');
      const b = key.indexOf(' ', a + 1);
      if (uni.has(key.slice(0, a)) && uni.has(key.slice(a + 1, b)) && uni.has(key.slice(b + 1))) {
        triRows.push([key, count]);
      }
    }
  }

  // ---------- 3. cap by CANDIDATE GROUP, not by row ----------
  //
  // A per-row frequency cap is wrong for this model, and it produced a real
  // bug: "mã này" (count 63, ranked ~322,000th) was cut while its rival
  // "mà nay" (285) survived, so an unaccented "ma nay" was confidently
  // "corrected" to "mà nay" instead of "mã này". Dropping ONE SIDE of a
  // candidate competition does not make the model smaller, it makes it wrong
  // — the decoder then sees unanimous evidence for the only surviving spelling.
  //
  // So rows are grouped by their accent-stripped key ("ma nay" groups
  // mà nay / mã này / mã nay / mà này) and whole groups are kept or dropped
  // together, ranked by the group's strongest member. Groups touching the SMS
  // domain vocabulary go first. The engine therefore never sees a rigged vote.
  const byCount = (x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1);
  const allDomain = (key) => {
    for (const token of key.split(' ')) if (!domainVocab.has(token)) return false;
    return true;
  };
  const strippedKey = (key) => key.split(' ').map((t) => accentKey(t)).join(' ');
  const capped = (rows, limit) => {
    const groups = new Map();
    for (const row of rows) {
      const gk = strippedKey(row[0]);
      let group = groups.get(gk);
      if (!group) {
        group = { rows: [], max: 0, domain: false };
        groups.set(gk, group);
      }
      group.rows.push(row);
      if (row[1] > group.max) group.max = row[1];
      if (!group.domain && allDomain(row[0])) group.domain = true;
    }
    const ordered = [...groups.values()].sort((a, b) =>
      (b.domain ? 1 : 0) - (a.domain ? 1 : 0) || b.max - a.max);
    const kept = [];
    let split = 0;
    for (const group of ordered) {
      // Inside a group, only the strongest rivals matter: a competition is
      // decided between the top spellings, and the long tail of the same
      // stripped key adds bytes without changing any decision. Keeping whole
      // groups unbounded burns the budget on those tails (17,970 of 462,763
      // groups survived when we tried it) and costs far more coverage than
      // the tails are worth.
      group.rows.sort(byCount);
      const rivals = group.rows.slice(0, cfg.maxRivalsPerGroup);
      if (kept.length + rivals.length > limit) {
        // Budget exhausted: stop rather than admit half a competition.
        split += 1;
        continue;
      }
      for (const row of rivals) kept.push(row);
    }
    kept.sort(byCount);
    return {
      map: new Map(kept),
      groups: groups.size,
      groupsKept: groups.size - split,
      groupsDropped: split,
    };
  };
  const biCap = capped(biRows, cfg.maxBigrams);
  const triCap = capped(triRows, cfg.maxTrigrams);
  const bi = biCap.map;
  const tri = triCap.map;

  // ---------- 4. SMS domain overlay (train split only) ----------
  const overlay = countCorpus(smsLines, cfg.domainWeight);
  for (const [w, c] of overlay.uni) uni.set(w, (uni.get(w) ?? 0) + c);
  for (const [k, c] of overlay.bi) {
    const sp = k.indexOf(' ');
    if (!uni.has(k.slice(0, sp)) || !uni.has(k.slice(sp + 1))) continue;
    bi.set(k, (bi.get(k) ?? 0) + c);
  }
  for (const [k, c] of overlay.tri) {
    const a = k.indexOf(' ');
    const b = k.indexOf(' ', a + 1);
    if (!uni.has(k.slice(0, a)) || !uni.has(k.slice(a + 1, b)) || !uni.has(k.slice(b + 1))) continue;
    tri.set(k, (tri.get(k) ?? 0) + c);
  }

  // unigram cap AFTER the overlay so domain words are never evicted
  let unigramRows = [...uni.entries()].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1));
  if (unigramRows.length > cfg.maxUnigrams) {
    const domainKeep = unigramRows.filter(([w]) => keep.has(w));
    unigramRows = domainKeep.slice(0, cfg.maxUnigrams);
    const survivors = new Set(unigramRows.map(([w]) => w));
    for (const k of [...bi.keys()]) {
      const sp = k.indexOf(' ');
      if (!survivors.has(k.slice(0, sp)) || !survivors.has(k.slice(sp + 1))) bi.delete(k);
    }
    for (const k of [...tri.keys()]) {
      const a = k.indexOf(' ');
      const b = k.indexOf(' ', a + 1);
      if (!survivors.has(k.slice(0, a)) || !survivors.has(k.slice(a + 1, b))
        || !survivors.has(k.slice(b + 1))) tri.delete(k);
    }
  }

  // ---------- 5. back-off consistency ----------
  //
  // Every occurrence of the trigram (a b c) contains the bigrams (a b) and
  // (b c), so a correct count table satisfies
  //     c(a,b) >= sum_c c(a,b,c)   and   c(b,c) >= sum_a c(a,b,c).
  // The source artifact violates this: it caps bigrams and trigrams
  // independently, so "chia sẻ mã" survives with count 102 while the bigram
  // "sẻ mã" was pruned to nothing. The scorer then backs off P(mã | sẻ) to a
  // near-zero smoothed value and that single term drowns the trigram evidence
  // — which is exactly how "chia se ma nay" got "corrected" to "mà" instead
  // of "mã". A model may be small; it may not be self-contradictory.
  //
  // Rows added here are REQUIRED for the kept trigrams to be scorable, so
  // they are not subject to the cap.
  const requiredBigrams = new Map();
  const require = (key, count) => {
    requiredBigrams.set(key, (requiredBigrams.get(key) ?? 0) + count);
  };
  for (const [key, count] of tri) {
    const a = key.indexOf(' ');
    const b = key.indexOf(' ', a + 1);
    require(key.slice(0, b), count);
    require(key.slice(a + 1), count);
  }
  let repairedRows = 0;
  let addedRows = 0;
  for (const [key, needed] of requiredBigrams) {
    const sp = key.indexOf(' ');
    if (!uni.has(key.slice(0, sp)) || !uni.has(key.slice(sp + 1))) continue;
    const current = bi.get(key) ?? 0;
    if (current >= needed) continue;
    if (current === 0) addedRows += 1; else repairedRows += 1;
    bi.set(key, needed);
  }

  // ---------- 6. emit ----------
  const totalTokens = headerTokens + overlay.tokens;
  const out = [];
  out.push('#SMS-LM v1 (domain-pruned)');
  out.push(`#source=${path.basename(cfg.source)}`);
  out.push(`#tokens=${totalTokens}`);
  for (const [w, c] of unigramRows) out.push(`U\t${w}\t${c}`);
  for (const [k, c] of [...bi.entries()].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1))) {
    out.push(`B\t${k}\t${c}`);
  }
  for (const [k, c] of [...tri.entries()].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1))) {
    out.push(`T\t${k}\t${c}`);
  }
  const body = `${out.join('\n')}\n`;
  writeFileSync(cfg.out, body);

  const manifest = {
    schema: 'sms-lm-manifest-v1',
    generatedBy: 'tools/build_sms_profile.mjs',
    generatedAt: new Date().toISOString(),
    source: {
      file: path.basename(cfg.source),
      bytes: statSync(cfg.source).size,
      rows: { U: seenU, B: seenB, T: seenT },
      headerTokens,
    },
    domainCorpora: domainCorpusFiles.map((f) => path.basename(f)),
    smsTrainOnly: true,
    params: {
      maxUnigrams: cfg.maxUnigrams,
      maxBigrams: cfg.maxBigrams,
      maxRivalsPerGroup: cfg.maxRivalsPerGroup,
      maxTrigrams: cfg.maxTrigrams,
      domainWeight: cfg.domainWeight,
      generalVocabTop: cfg.generalVocabTop,
    },
    result: {
      file: path.basename(cfg.out),
      bytes: body.length,
      rows: { U: unigramRows.length, B: bi.size, T: tri.size },
      backoffRepair: { addedBigrams: addedRows, raisedBigrams: repairedRows },
      candidateGroups: {
        B: { total: biCap.groups, kept: biCap.groupsKept, dropped: biCap.groupsDropped },
        T: { total: triCap.groups, kept: triCap.groupsKept, dropped: triCap.groupsDropped },
      },
      domainVocab: domainOnly,
      totalTokens,
      sha256: createHash('sha256').update(body).digest('hex'),
      buildMs: Date.now() - t0,
    },
  };
  // ---------- 7. matching pruned lexicon ----------
  // Same vocabulary as the LM: a word the LM cannot score should not be a
  // candidate the ranker can propose, and vice versa. This also shrinks the
  // SymSpell deletion index, the second memory hog after the LM.
  const survivors = new Set(unigramRows.map(([w]) => w));
  const lexOut = ['# Built by tools/build_sms_profile.mjs — word<TAB>freq<TAB>type',
    `# vocabulary mirrors ${path.basename(cfg.out)}`];
  let lexKept = 0;
  for (const raw of readFileSync(path.join(DATA, 'lexicon-built.txt'), 'utf8').split(/\r?\n/)) {
    if (!raw || raw.startsWith('#')) continue;
    const [word, freq, type = 'A'] = raw.split('\t');
    if (!word || !survivors.has(word.toLowerCase())) continue;
    lexOut.push(`${word}\t${freq}\t${type}`);
    lexKept += 1;
  }
  const lexBody = `${lexOut.join('\n')}\n`;
  writeFileSync(cfg.outLexicon, lexBody);
  manifest.result.lexicon = {
    file: path.basename(cfg.outLexicon),
    bytes: lexBody.length,
    entries: lexKept,
    sha256: createHash('sha256').update(lexBody).digest('hex'),
  };

  writeFileSync(`${cfg.out}.manifest.json`, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

function parseArgs(argv) {
  const opts = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    const value = argv[i + 1];
    opts[key] = /^\d+$/.test(value) ? Number(value) : value;
  }
  return opts;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const manifest = await buildSmsProfile(parseArgs(process.argv));
  const { source, result } = manifest;
  console.log('SMS lite profile built');
  console.log(`  source ${source.file}: ${(source.bytes / 1e6).toFixed(1)} MB`
    + ` U=${source.rows.U} B=${source.rows.B} T=${source.rows.T}`);
  console.log(`  result ${result.file}: ${(result.bytes / 1e6).toFixed(1)} MB`
    + ` U=${result.rows.U} B=${result.rows.B} T=${result.rows.T}`);
  console.log(`  lexicon ${result.lexicon.file}: ${(result.lexicon.bytes / 1e6).toFixed(2)} MB`
    + ` entries=${result.lexicon.entries}`);
  console.log(`  domain vocabulary=${result.domainVocab}`
    + `  candidate groups kept: B ${result.candidateGroups.B.kept}/${result.candidateGroups.B.total}`
    + ` T ${result.candidateGroups.T.kept}/${result.candidateGroups.T.total}`);
  console.log(`  back-off repair: +${result.backoffRepair.addedBigrams} bigram rows added,`
    + ` ${result.backoffRepair.raisedBigrams} raised to match their trigrams`);
  console.log(`  buildMs=${result.buildMs}`);
  console.log(`  sha256=${result.sha256.slice(0, 16)}…`);
}
