// Task 5 Step 1 — candidate-oracle audit (spelling-engine-optimization plan).
// Reads ONE labeled VSEC split and reports where correct targets are lost in
// the candidate pipeline: accent-family coverage, SymSpell top-6/12/20 key
// coverage, surface-pruning losses, Telex losses, multi-token/unsupported.
//
// Split guard (plan): dev/train are tuning splits and always allowed;
// test/external-test open ONLY behind --final after configuration freeze.
//
// Usage:
//   node tools/audit_candidate_recall.mjs --split dev
//   node tools/audit_candidate_recall.mjs --split dev --out dataset_artifacts/evaluation/candidate-recall-dev.json
import { readFileSync, writeFileSync, renameSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createDefaultEngine } from '../src/engine.mjs';
import { accentKey } from '../src/normalizer.mjs';
import { applyTelexHints, damerauOsaDistance, maxEditDistanceFor } from '../src/language.mjs';
import { collectVsecExpectations } from './spelling_benchmark_adapter_vsec.mjs';
import {
  buildCorrectionCandidates, CORRECTION_LANES, selectDiverseShortlist,
} from '../src/rules/linguistic-rules.mjs';
import { classifyCorrectionRelation } from '../src/correction-taxonomy.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const VSEC_DIR = process.env.VSEC_SPLIT_DIR
  ? path.resolve(process.env.VSEC_SPLIT_DIR)
  : path.join(ROOT, 'dataset_artifacts', 'vsec');

function parseArgs(argv) {
  const flags = { split: 'dev', final: false, out: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--split') flags.split = argv[++i];
    else if (argv[i] === '--final') flags.final = true;
    else if (argv[i] === '--out') flags.out = argv[++i];
  }
  return flags;
}

/** atomic JSON write + sha256 manifest sidecar (plan rule 8) */
function writeArtifact(outPath, payload) {
  const dir = path.dirname(outPath);
  mkdirSync(dir, { recursive: true });
  const body = Buffer.from(JSON.stringify(payload, null, 2), 'utf8');
  const tmp = `${outPath}.tmp-${process.pid}`;
  writeFileSync(tmp, body);
  renameSync(tmp, outPath);
  const sha256 = createHash('sha256').update(body).digest('hex');
  writeFileSync(`${outPath}.manifest.json`, JSON.stringify({
    artifact: path.basename(outPath), bytes: body.length, sha256,
  }, null, 2));
  return sha256;
}

function readJsonl(filePath) {
  return readFileSync(filePath, 'utf8').split(/\r?\n/)
    .filter(Boolean).map((line) => JSON.parse(line));
}

function words(text) {
  const out = [];
  const re = /[\p{L}\p{M}][\p{L}\p{M}'’-]*/gu;
  for (const m of String(text).matchAll(re)) {
    out.push({ raw: m[0], start: m.index });
  }
  return out;
}

function main() {
  const flags = parseArgs(process.argv.slice(2));
  const TUNING_SPLITS = new Set(['dev', 'train']);
  if (!TUNING_SPLITS.has(flags.split)) {
    // plan: reject test/external input unless explicit final-report flag
    if (!flags.final) {
      console.error(`REFUSED: split "${flags.split}" is a held-out evaluation `
        + 'split. Tuning must use --split dev|train. To run the one-time '
        + 'post-freeze report pass --final explicitly.');
      process.exit(2);
    }
    console.error(`WARNING: --final report on "${flags.split}" — this usage `
      + 'counts against the single post-freeze external evaluation.');
  }

  const engine = createDefaultEngine();
  const { accentIndex, typoCandidateProvider, lexicon, configService } = engine.services;
  // the production typo-lane pool width comes from config — measure against
  // what the engine can actually emit, not a hard-coded cap
  const configuredKeyCap = configService.snapshot()
    .get('linguistic.spellingCandidateKeys') ?? 20;

  const inputPath = path.join(VSEC_DIR, `vsec-${flags.split}.jsonl`);
  const rows = readJsonl(inputPath);

  const stats = {
    split: flags.split,
    rows: rows.length,
    labeledCorrections: 0,
    evaluated: 0,
    tokenNotMatched: 0,
    accentFamilyCovered: 0,
    symspellTop6Keys: 0,
    symspellTop12Keys: 0,
    symspellTop20Keys: 0,
    losses: {
      sameKeyBeyondAccentCap: 0,     // same key but outside accent-index top-12
      surfacePruned: 0,             // key matched at 12 but target surface dropped
      editDistanceOutOfRange: 0,
      telexOnlyLoss: 0,
      multiTokenUnsupported: 0,
      other: 0,
    },
    // Task 3 (recall-improvement plan): per-lane observability. Oracle
    // coverage is reported SEPARATELY for each lane and for each owning
    // relation — never collapsed into a single number.
    relations: {},
    laneCoverage: Object.fromEntries(CORRECTION_LANES.map((l) => [l, {
      labelsBuilt: 0, targetInLane: 0,
    }])),
    ownedLaneCoverage: {},
  };
  const missExamples = [];

  const poolHas = (token, targetLower, keyCap) => {
    const maxDist = maxEditDistanceFor(token.length);
    const pool = typoCandidateProvider.generatePool(token, maxDist, { keyCap });
    return pool.entries.some((e) => e.word.toLowerCase() === targetLower);
  };

  for (const row of rows) {
    const expects = collectVsecExpectations(row).filter((e) => e.suggestions?.length);
    if (expects.length === 0) continue;
    const toks = words(row.text);
    for (const exp of expects) {
      stats.labeledCorrections++;
      const errLower = exp.value.toLowerCase();
      // VSEC annotations sometimes carry trailing punctuation inside the
      // correction ("tạo.", "thấy,") — normalize targets to word characters
      // before coverage checks (label hygiene, not candidate generation).
      const suggRaw = String(exp.suggestions[0])
        .replace(/^[^\p{L}\p{M}]+|[^\p{L}\p{M}]+$/gu, '');
      const targetLower = suggRaw.toLowerCase();
      const tok = toks.find((w) => w.raw.toLowerCase().normalize('NFC') === errLower)
        ?? toks.find((w) => accentKey(w.raw) === accentKey(errLower));
      if (!tok) { stats.tokenNotMatched++; continue; }
      stats.evaluated++;

      // ---- Task 3: per-lane candidate reachability ----------------------
      const relation = classifyCorrectionRelation(exp.value, suggRaw);
      stats.relations[relation] = (stats.relations[relation] ?? 0) + 1;
      const builderToken = { normalized: tok.raw.toLowerCase(), original: tok.raw };
      for (const lane of CORRECTION_LANES) {
        const built = buildCorrectionCandidates({
          token: builderToken, lane, services: engine.services,
          snap: configService.snapshot(),
        });
        if (!built.eligible) continue;
        const cov = stats.laneCoverage[lane];
        cov.labelsBuilt++;
        if (built.entries.some((c) => c.word.toLowerCase() === targetLower)) {
          cov.targetInLane++;
        }
      }
      const OWNED_LANE = {
        UNACCENTED_SAME_KEY: 'UNACCENTED_SAME_KEY',
        ACCENTED_SAME_KEY: 'ACCENTED_SAME_KEY',
        DIFFERENT_KEY_SINGLE_TOKEN: 'DIFFERENT_KEY_REAL_WORD',
      };
      const ownedLane = OWNED_LANE[relation];
      if (ownedLane && !/\s/.test(String(suggRaw))) {
        const built = buildCorrectionCandidates({
          token: builderToken, lane: ownedLane,
          services: engine.services, snap: configService.snapshot(),
        });
        const cov = stats.ownedLaneCoverage[ownedLane] ??= {
          total: 0, covered: 0,
        };
        cov.total++;
        if (built.eligible
          && built.entries.some((c) => c.word.toLowerCase() === targetLower)) {
          cov.covered++;
        }
        // Task 7: structural shortlist survival — does the target survive
        // selectDiverseShortlist under the WORST-CASE assumption of zero
        // context attestation? (Production probes real LM context, so this
        // is a lower bound on shortlist survival, not an upper bound.)
        if (built.eligible) {
          const sl = stats.shortlistCoverageNoContext ??= {};
          const s2 = sl[ownedLane] ??= { total: 0, survivedSize4: 0, survivedSize6: 0, survivedSize8: 0 };
          s2.total++;
          for (const size of [4, 6, 8]) {
            const picked = selectDiverseShortlist(built.entries, {
              attestOf: new Map(), famKey: accentKey(builderToken.normalized), size,
            });
            if (picked.some((c) => c.word.toLowerCase() === targetLower)) {
              s2[`survivedSize${size}`]++;
            }
          }
        }
      }

      const fam = accentIndex.candidates(accentKey(tok.raw))
        .some((c) => c.word.toLowerCase() === targetLower);
      if (fam) stats.accentFamilyCovered++;
      if (poolHas(tok.raw, targetLower, 6)) stats.symspellTop6Keys++;
      if (poolHas(tok.raw, targetLower, 12)) stats.symspellTop12Keys++;
      if (poolHas(tok.raw, targetLower, 20)) stats.symspellTop20Keys++;

      // Engine-oracle: a label is COVERED when any production lane could
      // emit it — the SymSpell pool (typo lane) OR the accent family
      // (missing-diacritic lane for unaccented tokens, same-key sibling
      // merge for accented ones).
      if (/\s/.test(String(suggRaw))) { stats.losses.multiTokenUnsupported++; continue; }
      if (fam || poolHas(tok.raw, targetLower, configuredKeyCap)) continue;

      // classify the loss
      const maxDist = maxEditDistanceFor(tok.length);
      const pool = typoCandidateProvider.generatePool(tok.raw, maxDist, { keyCap: configuredKeyCap });
      const targetKey = accentKey(targetLower);
      const inputKeys = pool.inputKeys;
      const bestDist = Math.min(...inputKeys.map((k) => damerauOsaDistance(k, targetKey)),
        Number.POSITIVE_INFINITY);
      let reason;
      if (inputKeys.includes(targetKey) || accentKey(tok.raw) === targetKey) {
        // same stripped key as the input but OUTSIDE the accent-index top-12
        // frequency cap -> genuinely unreachable this build
        reason = 'sameKeyBeyondAccentCap';
      } else if (bestDist <= maxDist && pool.matchedKeys.includes(targetKey)) {
        reason = 'surfacePruned';
      } else if (bestDist > maxDist) {
        reason = 'editDistanceOutOfRange';
      } else if (applyTelexHints(errLower) !== errLower
        || applyTelexHints(targetLower) !== targetLower) {
        reason = 'telexOnlyLoss';
      } else {
        reason = 'other';
      }
      stats.losses[reason]++;
      if (missExamples.length < 40) {
        missExamples.push({ id: row.id ?? undefined, value: exp.value, target: suggRaw, reason,
          freq: lexicon.frequency(targetLower) });
      }
    }
  }

  const pct = (n, d) => (d ? Math.round((1000 * n) / d) / 10 : 0);
  const laneRates = Object.fromEntries(Object.entries(stats.laneCoverage)
    .map(([lane, c]) => [lane, {
      ...c,
      coveragePct: pct(c.targetInLane, c.labelsBuilt),
    }]));
  const ownedRates = Object.fromEntries(Object.entries(stats.ownedLaneCoverage)
    .map(([lane, c]) => [lane, {
      ...c,
      coveragePct: pct(c.covered, c.total),
    }]));
  const shortlistRates = Object.fromEntries(
    Object.entries(stats.shortlistCoverageNoContext ?? {})
      .map(([lane, c]) => [lane, {
        ...c,
        survivalPct: pct(c.survivedSize4, c.total),
      }]));
  const report = {
    ...stats,
    rates: {
      accentFamilyPct: pct(stats.accentFamilyCovered, stats.evaluated),
      top6Pct: pct(stats.symspellTop6Keys, stats.evaluated),
      top12Pct: pct(stats.symspellTop12Keys, stats.evaluated),
      top20Pct: pct(stats.symspellTop20Keys, stats.evaluated),
      oracleCoveragePct: pct(
        stats.evaluated - Object.values(stats.losses).reduce((a, b) => a + b, 0),
        stats.evaluated),
    },
    laneCoverage: laneRates,
    ownedLaneCoverage: ownedRates,
    shortlistCoverageNoContext: shortlistRates,
    missExamples,
  };
  console.log(JSON.stringify(report, null, 2));
  if (flags.out) {
    const sha = writeArtifact(path.resolve(flags.out), report);
    console.error(`artifact written: ${flags.out} (sha256 ${sha.slice(0, 16)}…)`);
  }
}

main();
