import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
export class ErrorChannel {
  static fromPayload(raw) {
    if (raw.schema !== 'vsec-error-channel-v1' || !raw.pairProof || !raw.pairCounts || !raw.direct) throw new Error('invalid error-channel artifact');
    if (raw.sha256 !== createHash('sha256').update(JSON.stringify({ pairProof: raw.pairProof, pairCounts: raw.pairCounts, direct: raw.direct })).digest('hex')) throw new Error('error-channel hash mismatch');
    return Object.freeze({ enabled: true, lookup: (s) => raw.direct[String(s).toLowerCase()] ?? null,
      hasPair: (s, t) => (raw.pairProof[String(s).toLowerCase()] ?? []).includes(String(t).toLowerCase()),
      pairCount: (s, t) => raw.pairCounts[String(s).toLowerCase()]?.[String(t).toLowerCase()] ?? 0 });
  }

  static load(path) {
    try {
      const raw = JSON.parse(readFileSync(path, 'utf8'));
      return ErrorChannel.fromPayload(raw);
    } catch (err) {
      if (err?.code === 'ENOENT') return Object.freeze({ enabled: false, lookup: () => null, hasPair: () => false, pairCount: () => 0 });
      throw err;
    }
  }
}
