// ============================================================
// Task 1 (tiny-attention-spelling-reranker-FIXED plan) — evaluation
// provenance contract.
//
// Every decision-dump JSONL starts with a header record
// (recordType:"header") binding the file to exact source/config/artifact
// hashes, an effective-config fingerprint and a feature-contract version.
// Decision lines carry recordType:"decision". Consumers (threshold
// calibration, offline replay) refuse stale dumps whose hashes no longer
// match the repository state instead of silently replaying them.
// ============================================================
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export const EVALUATION_HEADER_VERSION = 1;

/** Canonical hash keys used by this repository's evaluators. Attention keys
 *  become mandatory only once those artifacts exist (Task 7/8). */
export const CANONICAL_HASH_KEYS = Object.freeze([
  'linguisticRulesDotMjs',
  'configDotMjs',
  'recallRerankerModelJson',
  'lmNgramsTsv',
  'attentionTokenizer', // optional until Task 7/8
  'attentionVocab',     // optional until Task 7/8
  'attentionModel',     // optional until Task 7/8
]);

/** Deterministic sha256 over canonical JSON (sorted object keys). Used for
 *  effective-config fingerprints so semantically equal configs collide. */
export function sha256OfJson(value) {
  const canonical = JSON.stringify(value, (k, v) => {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      return Object.fromEntries(
        Object.entries(v).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    return v;
  });
  return createHash('sha256').update(canonical, 'utf8').digest('hex');
}

function requireNonEmptyString(v, name) {
  if (typeof v !== 'string' || v.trim() === '') {
    throw new TypeError(`createEvaluationHeader: "${name}" must be a non-empty string`);
  }
  return v;
}

/**
 * Build the first JSONL line of an evaluation dump.
 * @param {object} p
 * @param {string} p.schema dataset schema name (e.g. "shadow-decisions-v2")
 * @param {string} p.split evaluated split ("dev" | "train" | internal names)
 * @param {Record<string,string|null>} p.hashes canonical source/artifact hashes
 * @param {{configHash?:string|null, featureContract?:string|null}} [p.config]
 * @param {string} p.createdBy tool identifier
 */
export function createEvaluationHeader({ schema, split, hashes = {}, config = {}, createdBy }) {
  requireNonEmptyString(schema, 'schema');
  requireNonEmptyString(split, 'split');
  requireNonEmptyString(createdBy, 'createdBy');
  return {
    recordType: 'header',
    headerVersion: EVALUATION_HEADER_VERSION,
    schema,
    split,
    hashes: { ...hashes },
    config: { ...config },
    createdBy,
    createdAt: new Date().toISOString(),
  };
}

/**
 * Validate a parsed first-line object against expectations. Only axes present
 * in `expected` are enforced; a null/undefined expected hash is skipped so
 * callers may pass their full canonical map with unknown entries nulled.
 * @returns true when valid; throws with a precise reason otherwise.
 */
export function validateEvaluationHeader(header, expected = {}) {
  if (!header || typeof header !== 'object' || header.recordType !== 'header') {
    throw new Error('not an evaluation JSONL header '
      + '(first line must carry recordType:"header")');
  }
  if (header.headerVersion !== EVALUATION_HEADER_VERSION) {
    throw new Error(`unsupported evaluation header version `
      + `${JSON.stringify(header.headerVersion)}`);
  }
  if (expected.schema != null && header.schema !== expected.schema) {
    throw new Error(`schema mismatch: dump has ${JSON.stringify(header.schema)}, `
      + `expected ${JSON.stringify(expected.schema)}`);
  }
  if (expected.split != null && header.split !== expected.split) {
    throw new Error(`split mismatch: dump has ${JSON.stringify(header.split)}, `
      + `expected ${JSON.stringify(expected.split)}`);
  }
  for (const [key, want] of Object.entries(expected.hashes ?? {})) {
    if (want == null) continue;
    const got = header.hashes?.[key];
    if (got == null) {
      throw new Error(`missing hash "${key}" in evaluation header `
        + '(stale or foreign dump — regenerate it)');
    }
    if (got !== want) {
      throw new Error(`hash mismatch for ${key}: repository changed since `
        + 'this dump was written — regenerate the dump before calibrating');
    }
  }
  if (expected.config?.configHash != null
    && header.config?.configHash !== expected.config.configHash) {
    throw new Error(`config hash mismatch: dump was produced under a different `
      + `effective configuration (${JSON.stringify(header.config?.configHash)})`);
  }
  if (expected.config?.featureContract != null
    && header.config?.featureContract !== expected.config.featureContract) {
    throw new Error(`feature-contract version mismatch: dump has `
      + `${JSON.stringify(header.config?.featureContract)}, expected `
      + `${JSON.stringify(expected.config.featureContract)}`);
  }
  return true;
}

/** Tag one record as a decision line; never relabels existing non-decision tags. */
export function tagDecisionRecord(record) {
  if (record?.recordType != null && record.recordType !== 'decision') {
    throw new Error(`refusing to relabel record with recordType:${record.recordType}`);
  }
  return record.recordType === 'decision' ? record : { ...record, recordType: 'decision' };
}

/** Serialize header + tagged records to the JSONL body (header first). */
export function serializeEvaluationJsonl(header, records) {
  validateEvaluationHeader(header, {});
  const lines = [JSON.stringify(header)];
  let count = 0;
  for (const r of records ?? []) {
    lines.push(JSON.stringify(tagDecisionRecord(r)));
    count++;
  }
  return { body: `${lines.join('\n')}\n`, count };
}

/** Write an evaluation JSONL file. Returns { count }. */
export function writeEvaluationJsonl(outPath, header, records) {
  const { body, count } = serializeEvaluationJsonl(header, records);
  writeFileSync(outPath, body, 'utf8');
  return { count };
}

/**
 * Read + parse an evaluation JSONL file. The FIRST line must be a valid
 * header (recordType:"header", supported version); stale header-less dumps
 * are refused here so downstream replay can never silently proceed.
 * @returns {{ header: object, records: object[] }}
 */
export function readEvaluationJsonl(path) {
  const raw = readFileSync(path, 'utf8');
  const lines = raw.split(/\r?\n/).filter((l) => l.trim() !== '');
  if (!lines.length) {
    throw new Error(`${path}: empty evaluation dump`);
  }
  let header;
  try {
    header = JSON.parse(lines[0]);
  } catch (err) {
    throw new Error(`${path}: first line is not JSON (${err.message})`);
  }
  validateEvaluationHeader(header, {});
  const records = [];
  for (let i = 1; i < lines.length; i++) {
    let rec;
    try {
      rec = JSON.parse(lines[i]);
    } catch (err) {
      throw new Error(`${path}: line ${i + 1} is not JSON (${err.message})`);
    }
    if (rec?.recordType != null && rec.recordType !== 'decision') {
      throw new Error(`${path}: line ${i + 1} carries unexpected `
        + `recordType:${rec.recordType}`);
    }
    records.push(rec);
  }
  return { header, records };
}
