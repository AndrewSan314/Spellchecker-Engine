import { createDefaultEngine, ValidationContext } from '../src/engine.mjs';
import { RuleIds } from '../src/core.mjs';
import { adaptVsecSplit } from './spelling_benchmark_adapter_vsec.mjs';
import { linguisticCorrectionMatches } from '../src/correction-taxonomy.mjs';
import { extractContextEvidence } from '../src/context-evidence.mjs';

const split = process.argv[process.argv.indexOf('--split') + 1] ?? 'dev';
if (split === 'test' && !process.argv.includes('--final')) {
  throw new Error('REFUSED: held-out diagnostics require --final');
}

const engine = createDefaultEngine();
const buckets = new Map();
const tpBuckets = new Map();
const unknownContext = new Map();
const samples = [];
let tp = 0;
let fp = 0;
for (const row of adaptVsecSplit(split).rows) {
  const ctx = new ValidationContext(row.text, row.mode ?? 'ACCENTED', row.brand ?? row.brandname ?? 'TENDOO');
  const issues = engine.validate(ctx).issues
    .filter((i) => i.ruleId === RuleIds.POSSIBLE_SPELLING_ERROR
      || i.ruleId === RuleIds.POSSIBLE_MISSING_DIACRITIC);
  const words = engine.documentBuilder.build(ctx).tokens.filter((t) => t.type === 'WORD');
  for (const issue of issues) {
    const target = issue.suggestions?.[0];
    const count = engine.errorChannel.pairCount(issue.value, target);
    const lane = issue.message.startsWith('Spelling suggestion:') ? 'direct' : issue.ruleId;
    const key = `${lane}|pair=${count}|dictionary=${engine.lexicon.contains(issue.value)}`;
    const hit = row.expect.some((expected) => linguisticCorrectionMatches(issue, expected));
    if (lane === RuleIds.POSSIBLE_SPELLING_ERROR && count === 0
      && !engine.lexicon.contains(issue.value)) {
      const idx = words.findIndex((word) => word.start === issue.start && word.end === issue.end);
      const evidence = idx < 0 ? null : extractContextEvidence({
        languageModel: engine.languageModel, words: words.map((word) => word.normalized), idx,
        candidateWord: target, originalWord: words[idx].normalized,
      });
      const proof = evidence?.candidateAttestedWindows > evidence?.originalAttestedWindows;
      const contextKey = proof ? 'candidate-wins' : 'not-candidate-wins';
      const current = unknownContext.get(contextKey) ?? { tp: 0, fp: 0 };
      current[hit ? 'tp' : 'fp']++;
      unknownContext.set(contextKey, current);
    }
    if (hit) {
      tp++;
      tpBuckets.set(key, (tpBuckets.get(key) ?? 0) + 1);
      continue;
    }
    fp++;
    buckets.set(key, (buckets.get(key) ?? 0) + 1);
    if (samples.length < 100) samples.push({ id: row.id, value: issue.value, target, lane, count });
  }
}
console.log(JSON.stringify({ split, tp, fp,
  tpBuckets: Object.fromEntries([...tpBuckets].sort(([a], [b]) => a.localeCompare(b))),
  unknownContext: Object.fromEntries(unknownContext),
  buckets: Object.fromEntries([...buckets].sort(([a], [b]) => a.localeCompare(b))), samples }, null, 2));
