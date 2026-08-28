// Five-fold, message-disjoint check for train-derived channel rules.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { RuleIds } from '../src/core.mjs';
import { ErrorChannel } from '../src/error-channel.mjs';
import { linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';
import { extractContextEvidence } from '../src/context-evidence.mjs';
import { buildErrorChannel } from './build_error_channel.mjs';
import { vsecRowToBenchmark } from './spelling_benchmark_adapter_vsec.mjs';

const FOLDS = 5;
const argIndex = process.argv.indexOf('--shortlist-size');
const shortlistSize = argIndex >= 0 ? Number(process.argv[argIndex + 1]) : 4;
const minTokenIndex = process.argv.indexOf('--min-token-length');
const minTokenLength = minTokenIndex >= 0 ? Number(process.argv[minTokenIndex + 1]) : 2;
const unaccentedRealWordMode = process.argv.includes('--unaccented-real-word-active') ? 'ACTIVE' : 'OFF';
if (!Number.isInteger(shortlistSize) || shortlistSize < 1 || shortlistSize > 8) {
  throw new Error('--shortlist-size must be an integer in [1,8]');
}
if (!Number.isInteger(minTokenLength) || minTokenLength < 1 || minTokenLength > 3) {
  throw new Error('--min-token-length must be an integer in [1,3]');
}
const raw = readFileSync('dataset_artifacts/vsec/vsec-train.jsonl', 'utf8')
  .split(/\r?\n/).filter(Boolean).map(JSON.parse);
const foldOf = (row) => createHash('sha256').update(String(row.text)).digest()[0] % FOLDS;
const engine = createDefaultEngine();
engine.configService.reload({ linguistic: {
  spellingShortlistSize: shortlistSize,
  spellingMinTokenLength: minTokenLength,
  unaccentedRealWordMode,
} });
const folds = [];
let totalTp = 0, totalFp = 0, totalLabels = 0;
const fpBuckets = new Map();
const tpBuckets = new Map();
const directLowContext = new Map();

for (let fold = 0; fold < FOLDS; fold++) {
  const train = raw.filter((row) => foldOf(row) !== fold);
  const test = raw.filter((row) => foldOf(row) === fold);
  const channel = ErrorChannel.fromPayload(buildErrorChannel(train, `vsec-train-oof-fold-${fold}`));
  // Rules close over this shared service object, so both rule and global
  // verification use the fold-local immutable channel without reloading LM.
  engine.errorChannel = channel;
  engine.services.errorChannel = channel;
  let tp = 0, fp = 0, labels = 0;
  for (const [index, source] of test.entries()) {
    const row = vsecRowToBenchmark(source, index, `oof${fold}`);
    labels += row.expect.length;
    const ctx = new ValidationContext(row.text, 'ACCENTED', 'TENDOO');
    const issues = engine.validate(ctx).issues
      .filter((issue) => issue.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
        || issue.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
    const words = engine.documentBuilder.build(ctx).tokens.filter((token) => token.type === 'WORD');
    for (const issue of issues) {
      const target = issue.suggestions?.[0];
      const count = channel.pairCount(issue.value, target);
      const lane = issue.message.startsWith('Spelling suggestion:') ? 'direct' : issue.ruleId;
      const key = `${lane}|pair=${count}|dictionary=${engine.lexicon.contains(issue.value)}`;
      const hit = row.expect.some((expected) => linguisticCorrectionMatches(issue, expected));
      if (lane === 'direct' && count <= 2) {
        const idx = words.findIndex((word) => word.start === issue.start && word.end === issue.end);
        const ev = extractContextEvidence({ languageModel: engine.languageModel,
          words: words.map((word) => word.normalized), idx, candidateWord: target, originalWord: issue.value });
        const gap = ev.candidateAttestedWindows - ev.originalAttestedWindows;
        const bucket = directLowContext.get(gap) ?? { tp: 0, fp: 0 };
        bucket[hit ? 'tp' : 'fp']++;
        directLowContext.set(gap, bucket);
      }
      if (hit) {
        tp++;
        tpBuckets.set(key, (tpBuckets.get(key) ?? 0) + 1);
      } else {
        fp++;
        fpBuckets.set(key, (fpBuckets.get(key) ?? 0) + 1);
      }
    }
  }
  const precision = tp / (tp + fp);
  const recall = tp / labels;
  folds.push({ fold, rows: test.length, labels, tp, fp, precision, recall });
  totalTp += tp; totalFp += fp; totalLabels += labels;
}
console.log(JSON.stringify({ shortlistSize, minTokenLength, unaccentedRealWordMode, folds, aggregate: {
  tp: totalTp, fp: totalFp, labels: totalLabels,
  precision: totalTp / (totalTp + totalFp), recall: totalTp / totalLabels,
}, tpBuckets: Object.fromEntries([...tpBuckets].sort(([a], [b]) => a.localeCompare(b))),
  directLowContext: Object.fromEntries([...directLowContext].sort(([a], [b]) => Number(a) - Number(b))),
  fpBuckets: Object.fromEntries([...fpBuckets].sort(([a], [b]) => a.localeCompare(b))) }, null, 2));
