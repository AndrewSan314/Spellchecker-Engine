// ============================================================
// Task 3 (tiny-attention-spelling-reranker-FIXED plan) — pure schema,
// leakage-safe split assignment and oracle@K selection rules for the
// attention ranking dataset. No I/O: everything here is unit-testable.
// ============================================================
import { createHash } from 'node:crypto';

export const ATTENTION_RANKING_SCHEMA = 'attention-ranking-v1';
export const ATTENTION_MESSAGES_SCHEMA = 'attention-messages-v1';
export const ATTENTION_SPLIT_SALT = 'attention-v2-20260825';
export const ATTENTION_SHORTLIST_CONFIG_SCHEMA = 'attention-shortlist-config-v1';

export const RANKING_LANES = Object.freeze([
  'ACCENTED_SAME_KEY',
  'UNACCENTED_SAME_KEY',
  'DIFFERENT_KEY_REAL_WORD',
  'UNKNOWN_TYPO',
  'CLEAN_KEEP',
  'HARD_NEGATIVE_KEEP',
]);

export const ROW_SOURCES = Object.freeze([
  'vsec-train',
  'clean-train',
  'synthetic-clean-train',
  'hard-negative-clean-train',
]);

export const KEEP_LANES = Object.freeze(['CLEAN_KEEP', 'HARD_NEGATIVE_KEEP']);

/** Candidate-count choices for the frozen diversity shortlist. */
export const K_CHOICES = Object.freeze([4, 6, 8]);
export const K_SELECTION_TOLERANCE = 0.005;

/**
 * Deterministic 00..99 bucket for a group: sha256(groupId + salt) rendered
 * as hex; the first byte pair ("00".."ff") mod 100 gives the bucket.
 * 00..79 train | 80..89 calibration | 90..99 internal-test.
 */
export function splitBucketForGroup(groupId, salt = ATTENTION_SPLIT_SALT) {
  const digest = createHash('sha256').update(`${groupId}${salt}`, 'utf8').digest('hex');
  return Number.parseInt(digest.slice(0, 2), 16) % 100;
}

export function splitNameForBucket(bucket) {
  if (bucket <= 79) return 'train';
  if (bucket <= 89) return 'calibration';
  return 'internal-test';
}

export function splitForGroup(groupId, salt = ATTENTION_SPLIT_SALT) {
  return splitNameForBucket(splitBucketForGroup(groupId, salt));
}

/** Normalized sentence hash used as groupId and deny-list identity. */
export function normalizedSentenceHash(text) {
  // normalizeForModel semantics without importing the tokenizer (keeps this
  // module dependency-light for tests): NFC + Unicode lowercase
  const norm = String(text ?? '').normalize('NFC').toLowerCase();
  return createHash('sha256').update(norm, 'utf8').digest('hex');
}

/**
 * Validate one ranking row. Throws with a precise reason on violation.
 * Structural rules enforced here:
 *   - option index 0 is always KEEP_ORIGINAL;
 *   - KEEP lanes always label index 0;
 *   - correction rows label an index within [1 .. candidates.length];
 *   - enums closed.
 */
export function assertRankingRowShape(row) {
  const fail = (msg) => { throw new Error(`ranking-row ${row?.id ?? '?'}: ${msg}`); };
  if (!row || row.recordType !== 'ranking-row') fail('recordType must be "ranking-row"');
  if (typeof row.id !== 'string' || !row.id) fail('id required');
  if (typeof row.groupId !== 'string' || !row.groupId) fail('groupId required');
  if (!ROW_SOURCES.includes(row.source)) fail(`unknown source ${row.source}`);
  if (!RANKING_LANES.includes(row.lane)) fail(`unknown lane ${row.lane}`);
  if (!Array.isArray(row.candidates)) fail('candidates must be an array');
  const options = 1 + row.candidates.length;
  if (options < 2 && !KEEP_LANES.includes(row.lane)) {
    fail('correction rows need at least one candidate');
  }
  if (!Number.isInteger(row.labelIndex) || row.labelIndex < 0 || row.labelIndex >= options) {
    fail(`labelIndex ${row.labelIndex} out of range for ${options} options`);
  }
  if (KEEP_LANES.includes(row.lane) && row.labelIndex !== 0) {
    fail('KEEP lanes must label index 0');
  }
  if (row.candidates.some((c) => c?.surface === 'KEEP_ORIGINAL')) {
    fail('KEEP_ORIGINAL must never appear as a candidate surface');
  }
  return true;
}

/** Group-disjointness across the three attention splits. */
export function assertNoGroupOverlap(splits) {
  const seen = new Map(); // groupId -> split name
  for (const [name, rows] of Object.entries(splits)) {
    for (const groupId of rows) {
      const g = typeof groupId === 'string' ? groupId : groupId?.groupId;
      if (seen.has(g)) {
        throw new Error(`group overlap: ${g} in both "${seen.get(g)}" and "${name}"`);
      }
      seen.set(g, name);
    }
  }
  return true;
}

/** Oracle@K for one row: was the gold correction inside the first K candidates? */
export function rowHitsAtK(row, k) {
  if (row.labelIndex === 0) return null; // KEEP rows carry no retrieval signal
  const targetIdx = row.labelIndex - 1;   // candidate index of the gold
  return targetIdx < k;
}

export function oracleAtK(rows, k) {
  let hits = 0;
  let counted = 0;
  for (const row of rows) {
    const hit = rowHitsAtK(row, k);
    if (hit === null) continue;
    counted++;
    if (hit) hits++;
  }
  return counted ? hits / counted : null;
}

/**
 * Calculate the three distinct oracle metrics with strict invariant checks:
 *   1. widePoolOracle       = widePoolHits / allAttempts
 *   2. shortlistRetention@K = shortlistHits / widePoolHits
 *   3. absoluteOracle@K     = shortlistHits / allAttempts
 */
export function calculateOracleMetrics({
  allAttempts,
  widePoolHits,
  shortlistHitsAt4,
  shortlistHitsAt6,
  shortlistHitsAt8,
}) {
  const count = (name, value, max) => {
    if (!Number.isInteger(value) || !Number.isFinite(value)) {
      throw new Error(`Invalid ${name}: count must be finite integer`);
    }
    if (value < 0 || (max != null && value > max)) {
      throw new Error(`Invalid ${name}: must be between 0 and ${max}`);
    }
    return value;
  };
  count('allAttempts', allAttempts);
  count('widePoolHits', widePoolHits, allAttempts);
  count('shortlistHitsAt4', shortlistHitsAt4, widePoolHits);
  count('shortlistHitsAt6', shortlistHitsAt6, widePoolHits);
  count('shortlistHitsAt8', shortlistHitsAt8, widePoolHits);
  if (!(shortlistHitsAt4 <= shortlistHitsAt6
    && shortlistHitsAt6 <= shortlistHitsAt8)) {
    throw new Error('Invariant violation: shortlist hits must be monotonic');
  }
  const widePoolOracle = allAttempts > 0 ? widePoolHits / allAttempts : 0;
  const shortlistRetentionAt4 = widePoolHits > 0 ? shortlistHitsAt4 / widePoolHits : 0;
  const shortlistRetentionAt6 = widePoolHits > 0 ? shortlistHitsAt6 / widePoolHits : 0;
  const shortlistRetentionAt8 = widePoolHits > 0 ? shortlistHitsAt8 / widePoolHits : 0;
  const absoluteOracleAt4 = allAttempts > 0 ? shortlistHitsAt4 / allAttempts : 0;
  const absoluteOracleAt6 = allAttempts > 0 ? shortlistHitsAt6 / allAttempts : 0;
  const absoluteOracleAt8 = allAttempts > 0 ? shortlistHitsAt8 / allAttempts : 0;

  if (absoluteOracleAt4 > widePoolOracle + 1e-6
    || absoluteOracleAt6 > widePoolOracle + 1e-6
    || absoluteOracleAt8 > widePoolOracle + 1e-6) {
    throw new Error('Invariant violation: absoluteOracle@K cannot exceed widePoolOracle');
  }
  if (shortlistRetentionAt4 > 1.0001
    || shortlistRetentionAt6 > 1.0001
    || shortlistRetentionAt8 > 1.0001) {
    throw new Error('Invariant violation: shortlist retention cannot exceed 100%');
  }

  return {
    allAttempts,
    widePoolHits,
    shortlistHits: {
      4: shortlistHitsAt4,
      6: shortlistHitsAt6,
      8: shortlistHitsAt8,
    },
    widePoolOracle,
    shortlistRetention: {
      4: shortlistRetentionAt4,
      6: shortlistRetentionAt6,
      8: shortlistRetentionAt8,
    },
    absoluteOracle: {
      4: absoluteOracleAt4,
      6: absoluteOracleAt6,
      8: absoluteOracleAt8,
    },
    // Keep numerators and denominators next to percentages so consumers can
    // audit the arithmetic instead of trusting a rounded ratio.
    numerators: {
      widePoolOracle: widePoolHits,
      shortlistRetention: {
        4: shortlistHitsAt4,
        6: shortlistHitsAt6,
        8: shortlistHitsAt8,
      },
      absoluteOracle: {
        4: shortlistHitsAt4,
        6: shortlistHitsAt6,
        8: shortlistHitsAt8,
      },
    },
    denominators: {
      widePoolOracle: allAttempts,
      shortlistRetention: {
        4: widePoolHits,
        6: widePoolHits,
        8: widePoolHits,
      },
      absoluteOracle: {
        4: allAttempts,
        6: allAttempts,
        8: allAttempts,
      },
    },
  };
}

/**
 * Freeze rule: select K in {4,6,8} prioritizing shortlist retention >= 90%
 * and smallest K within tolerance.
 */
export function selectShortlistK(metrics) {
  const ret8 = metrics?.shortlistRetention?.['8'] ?? metrics?.shortlistRetentionAt8 ?? metrics?.at8;
  const ret6 = metrics?.shortlistRetention?.['6'] ?? metrics?.shortlistRetentionAt6 ?? metrics?.at6;
  const ret4 = metrics?.shortlistRetention?.['4'] ?? metrics?.shortlistRetentionAt4 ?? metrics?.at4;
  const wide = metrics?.widePoolOracle ?? 0;

  if (ret8 == null || !Number.isFinite(ret8) || ret8 < 0 || ret8 > 1) {
    throw new Error('retention@8 required before freezing K and must be 0..1');
  }

  const threshold = ret8 - K_SELECTION_TOLERANCE;
  for (const [k, v] of [[4, ret4], [6, ret6], [8, ret8]]) {
    if (v != null && Number.isFinite(v) && v >= 0 && v <= 1 && v >= threshold) {
      return {
        k,
        reason: `smallest K in {4,6,8} with retention@K (${(v * 100).toFixed(2)}%) `
          + `>= retention@8 - 0.005 (${(threshold * 100).toFixed(2)}%); `
          + `wide-pool oracle ${(wide * 100).toFixed(2)}%`,
      };
    }
  }

  return {
    k: 8,
    reason: `frozen K=8 with retention@8 (${(ret8 * 100).toFixed(2)}%), wide-pool oracle ${(wide * 100).toFixed(2)}%`,
  };
}

const FORBIDDEN_MARKERS = Object.freeze([
  'dev', 'test', 'external-test', 'viwiki', 'heldout', 'evaluation',
]);

/**
 * Path guard for extraction inputs. Forbidden held-out markers are rejected
 * BEFORE any read. Allowed supervised sources are exactly
 * dataset_artifacts/vsec/vsec-train.jsonl and
 * dataset_artifacts/clean-source/clean-train.txt.
 */
export function assertAllowedTrainingSources(paths) {
  const out = [];
  for (const raw of paths ?? []) {
    const p = String(raw).toLowerCase().replace(/\\/g, '/');
    for (const marker of FORBIDDEN_MARKERS) {
      if (p.includes(marker)) {
        throw new Error(`forbidden training source "${raw}" (matches `
          + `"${marker}"): held-out data must never reach attention training`);
      }
    }
    const ok = p.endsWith('dataset_artifacts/vsec/vsec-train.jsonl')
      || p.endsWith('dataset_artifacts/clean-source/clean-train.txt')
      || p.endsWith('/vsec-train.jsonl')
      || p.endsWith('/clean-train.txt');
    if (!ok) {
      throw new Error(`training source "${raw}" is not an allowed supervised `
        + 'input (vsec-train.jsonl | clean-train.txt)');
    }
    out.push(String(raw));
  }
  return out;
}
