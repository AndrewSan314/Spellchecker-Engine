// Deterministic train-only artifact for the context-proven error channel.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const norm = (s) => String(s ?? '').toLowerCase()
  .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
export function buildErrorChannel(rows, source = 'dataset_artifacts/vsec/vsec-train.jsonl') {
  const pairs = new Map(), correct = new Set();
  for (const row of rows) {
    for (const a of row.syllable_annotations ?? []) if (a?.is_correct !== false) correct.add(norm(a.syllable));
    for (const p of row.correction_pairs ?? []) {
      const source = norm(p.error), target = norm(p.correction); if (!source || !target) continue;
      const m = pairs.get(source) ?? new Map(); m.set(target, (m.get(target) ?? 0) + 1); pairs.set(source, m);
    }
  }
  const direct = [...pairs].flatMap(([source, targets]) => targets.size === 1 && !correct.has(source)
    ? [[source, { target: [...targets.keys()][0], count: [...targets.values()][0] }]] : []);
  const pairProof = Object.fromEntries([...pairs].map(([source, targets]) => [source, [...targets.keys()]]));
  const pairCounts = Object.fromEntries([...pairs].map(([source, targets]) => [source, Object.fromEntries(targets)]));
  const payload = { schema: 'vsec-error-channel-v1', source, pairProof, pairCounts, direct: Object.fromEntries(direct) };
  payload.sha256 = createHash('sha256').update(JSON.stringify({ pairProof, pairCounts, direct: payload.direct })).digest('hex');
  return payload;
}

if (process.argv[1]?.endsWith('build_error_channel.mjs')) {
  const input = 'dataset_artifacts/vsec/vsec-train.jsonl';
  const payload = buildErrorChannel(readFileSync(input, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse), input);
  writeFileSync('src/data/error-channel.json', `${JSON.stringify(payload)}\n`, 'utf8');
  console.log(`wrote ${Object.keys(payload.pairProof).length} pair proofs and ${Object.keys(payload.direct).length} direct entries`);
}
