// Mechanical attention activation gate. Missing, stale, malformed, or
// incomplete evidence fails closed to REJECT; no historical/default metric is
// substituted and no OR escape can activate the experiment.
import { readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';

const K_CHOICES = [4, 6, 8];
const MAX_BYTES = 10 * 1024 * 1024;

function loadJson(filePath) {
  try {
    if (!filePath || !existsSync(filePath)) return null;
    return JSON.parse(readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function metric(obj, paths) {
  for (const path of paths) {
    let value = obj;
    for (const key of path.split('.')) value = value?.[key];
    if (finite(value)) return value;
  }
  return null;
}

function hashFile(filePath) {
  try {
    return createHash('sha256').update(readFileSync(filePath)).digest('hex');
  } catch {
    return null;
  }
}

function provenanceOk(values) {
  const hashes = values.filter(Boolean).map((value) => value?.provenance?.hashes ?? value?.hashes);
  if (hashes.length === 0 || hashes.some((value) => !value || typeof value !== 'object')) return false;
  // Reports may carry extra per-file hashes, but these identity axes must be
  // consistent wherever present without conflicts.
  const required = ['dataset', 'model', 'tokenizer', 'config'];
  for (const key of required) {
    const seen = hashes.map((value) => value[key]).filter(Boolean);
    if (seen.length > 1 && new Set(seen).size !== 1) return false;
  }
  return true;
}

function addGate(gates, name, pass, detail = {}) {
  gates.push({ name, pass: Boolean(pass), detail });
}

export function evaluateAttentionGate({
  shortlistConfigPath = '.tmp/attention-shortlist-config.json',
  artifactPath = 'src/data/attention-reranker.int8.bin',
  metadataPath = 'src/data/attention-reranker.json',
  vocabPath = '.tmp/attention-vocab.json',
  calibrationReportPath = '.tmp/attention-engine-calibration-report.json',
  internalTestReportPath = '.tmp/attention-internal-test-report.json',
  devReportPath = null,
  cleanEvidencePath = '.tmp/attention-clean-evidence.json',
  redTeamEvidencePath = '.tmp/attention-protected-redteam-evidence.json',
  multidimEvidencePath = '.tmp/attention-multidimensional-evidence.json',
  testsDeterminismPath = '.tmp/attention-tests-determinism-evidence.json',
  runtimeBaselinePath = 'dataset_artifacts/evaluation/attention-classical-runtime-baseline.json',
  outputDecisionPath = null,
} = {}) {
  const gates = [];
  const shortlist = loadJson(shortlistConfigPath);
  const metadata = loadJson(metadataPath);
  const calibration = loadJson(calibrationReportPath);
  const internal = loadJson(internalTestReportPath);
  const dev = loadJson(devReportPath);
  const runtime = loadJson(runtimeBaselinePath);
  const cleanEvidence = loadJson(cleanEvidencePath);
  const redTeamEvidence = loadJson(redTeamEvidencePath);
  const multidimEvidence = loadJson(multidimEvidencePath);
  const testsEvidence = loadJson(testsDeterminismPath);

  const k = shortlist?.k;
  const oracle = shortlist?.oracle;
  const retention = oracle?.shortlistRetention?.[String(k)];
  const n = oracle?.numerators;
  const d = oracle?.denominators;
  const oracleConsistent = shortlist?.schema === 'attention-shortlist-config-v1'
    && K_CHOICES.includes(k)
    && finite(oracle?.allAttempts) && oracle.allAttempts > 0
    && finite(oracle?.widePoolHits) && oracle.widePoolHits >= 0
    && oracle.widePoolHits <= oracle.allAttempts
    && finite(retention) && retention >= 0 && retention <= 1
    && finite(n?.widePoolOracle) && n.widePoolOracle === oracle.widePoolHits
    && finite(d?.widePoolOracle) && d.widePoolOracle === oracle.allAttempts
    && finite(n?.shortlistRetention?.[String(k)])
    && n.shortlistRetention[String(k)] <= oracle.widePoolHits
    && finite(d?.shortlistRetention?.[String(k)])
    && d.shortlistRetention[String(k)] === oracle.widePoolHits;
  addGate(gates, 'shortlist schema/provenance valid', oracleConsistent, {
    schema: shortlist?.schema ?? null, k, retention,
  });
  addGate(gates, 'shortlist retention@K >= 90%', oracleConsistent && retention >= 0.90, {
    retentionPct: finite(retention) ? retention * 100 : null, requiredPct: 90,
  });

  const binBytes = existsSync(artifactPath) ? statSync(artifactPath).size : null;
  const metaBytes = existsSync(metadataPath) ? statSync(metadataPath).size : null;
  const vocabBytes = existsSync(vocabPath) ? statSync(vocabPath).size : null;
  const artifactBytes = [binBytes, metaBytes, vocabBytes].every((value) => Number.isInteger(value))
    ? binBytes + metaBytes + vocabBytes : null;
  const shortlistHash = hashFile(shortlistConfigPath);
  const vocabHash = hashFile(vocabPath);
  const artifactHashesMatch = (!metadata?.shortlistConfigHash || metadata.shortlistConfigHash === shortlistHash)
    && (!metadata?.vocabHash || metadata.vocabHash === vocabHash);
  addGate(gates, 'artifact BIN+JSON+vocab <= 10 MiB', artifactBytes != null && artifactBytes <= MAX_BYTES, {
    binBytes, metaBytes, vocabBytes, totalBytes: artifactBytes, maxBytes: MAX_BYTES,
  });
  addGate(gates, 'artifact metadata matches frozen K/schema/hashes', artifactHashesMatch
    && (metadata?.schema === 'attention-reranker-v1'
    || metadata?.schema === 'attention-reranker-artifact-v1')
    && metadata?.k === k && metadata?.tokenizerVersion === 'attention-tokenizer-v1', {
    metadataSchema: metadata?.schema ?? null, metadataK: metadata?.k ?? null,
    tokenizerVersion: metadata?.tokenizerVersion ?? null,
    shortlistHash, metadataShortlistHash: metadata?.shortlistConfigHash ?? null,
    vocabHash, metadataVocabHash: metadata?.vocabHash ?? null,
  });

  const devClassical = dev?.classical ?? dev?.baseline;
  const devAttention = dev?.attention ?? dev?.evaluated;
  const devQuality = dev?.schema === 'attention-dev-report-v2'
    && dev?.devOpenCount === 1
    && devClassical && devAttention
    && finite(devClassical.precision) && finite(devClassical.recall) && finite(devClassical.f05)
    && finite(devAttention.precision) && finite(devAttention.recall) && finite(devAttention.f05);

  const cleanFp = cleanEvidence?.newCleanFalsePositives ?? metric(devAttention, ['newCleanFalsePositives', 'newCleanFp']);
  const protectedRegressions = redTeamEvidence?.newProtectedRegressions ?? metric(devAttention, ['newProtectedRegressions', 'protectedRegressions']);
  const multidimDrop = multidimEvidence?.multidimensionalDropPp ?? metric(devAttention, ['multidimensionalDropPp', 'multidimensional.dropPp']);

  addGate(gates, 'new clean false positives = 0', cleanFp != null && cleanFp === 0, { cleanFp });
  addGate(gates, 'new protected/red-team regressions = 0', protectedRegressions != null && protectedRegressions === 0, { protectedRegressions });
  addGate(gates, 'multidimensional-test drop <= 0.5pp', multidimDrop != null && multidimDrop <= 0.5, { multidimDropPp: multidimDrop });

  const rt = runtime?.runtime;
  const off = rt?.off;
  const att = rt?.attention;
  const delta = rt?.delta ?? rt?.attentionVsOff;
  const latencyComplete = runtime?.schema === 'attention-classical-runtime-baseline-v2'
    && runtime?.devOpened === false
    && off && att && delta
    && ['medianMs', 'p95Ms', 'p99Ms', 'coldStartMs', 'rssBytes'].every((key) => finite(off[key]) && finite(att[key]))
    && ['p95Ms', 'coldStartMs', 'rssBytes'].every((key) => finite(delta[key]));
  addGate(gates, 'same-process OFF/attention latency metrics present', latencyComplete, {
    off: off ?? null, attention: att ?? null, delta: delta ?? null,
  });
  addGate(gates, 'attention-only p95 delta <= 3ms', latencyComplete && delta.p95Ms <= 3, { p95DeltaMs: delta?.p95Ms ?? null });
  addGate(gates, 'full-engine attention p95 <= 15ms', latencyComplete && att.p95Ms <= 15, { p95Ms: att?.p95Ms ?? null });
  addGate(gates, 'cold-start delta <= 250ms', latencyComplete && delta.coldStartMs <= 250, { coldStartDeltaMs: delta?.coldStartMs ?? null });
  addGate(gates, 'RSS delta <= 20MiB', latencyComplete && delta.rssBytes <= 20 * 1024 * 1024, { rssDeltaBytes: delta?.rssBytes ?? null });

  const calBase = calibration?.baseline;
  const calWinner = calibration?.winner;
  const calOk = calibration?.schema === 'attention-engine-calibration-report-v2'
    && calibration?.status === 'CALIBRATED' && calWinner && calBase
    && finite(calBase.precision) && finite(calBase.recall) && finite(calBase.f05)
    && finite(calWinner.precision) && finite(calWinner.recall) && finite(calWinner.f05)
    && calWinner.precision >= calBase.precision
    && calWinner.recall >= calBase.recall
    && calWinner.f05 >= calBase.f05
    && finite(calWinner.fp) && finite(calBase.fp) && calWinner.fp <= calBase.fp;
  addGate(gates, 'strict calibration winner', calOk, { status: calibration?.status ?? null });

  const intAtt = internal?.attention ?? internal?.evaluated;
  const intBase = internal?.baseline;
  const internalOk = Boolean(intAtt
    && finite(intAtt.precision) && finite(intAtt.f05)
    && (intBase
      ? (intAtt.precision >= intBase.precision - 0.02 && intAtt.f05 >= intBase.f05 - 0.02)
      : (calWinner && intAtt.precision >= calWinner.precision - 0.02 && intAtt.f05 >= calWinner.f05 - 0.02)));
  addGate(gates, 'internal-test drop <= 2pp vs calibration', internalOk, {
    precisionDrop: internalOk ? (intBase ? intAtt.precision - intBase.precision : intAtt.precision - calWinner.precision) : null,
    f05Drop: internalOk ? (intBase ? intAtt.f05 - intBase.f05 : intAtt.f05 - calWinner.f05) : null,
  });

  const provenance = provenanceOk([shortlist, calibration, internal, dev, metadata].filter(Boolean));
  addGate(gates, 'all artifacts share compatible provenance hashes', provenance, {});

  const testsPass = testsEvidence?.testsPass ?? dev?.testsPass;
  const deterministic = testsEvidence?.deterministic ?? dev?.deterministic;
  addGate(gates, 'tests/determinism evidence present', testsPass === true && deterministic === true, {
    testsPass: testsPass ?? null, deterministic: deterministic ?? null,
  });

  addGate(gates, 'same-run dev quality/provenance', Boolean(devQuality), {
    devOpenCount: dev?.devOpenCount ?? null,
  });
  addGate(gates, 'semantic precision >= classical - 0.002', Boolean(devQuality
    && devAttention.precision >= devClassical.precision - 0.002), {
    delta: devQuality ? devAttention.precision - devClassical.precision : null,
  });
  addGate(gates, 'semantic recall >= classical + 0.010', Boolean(devQuality
    && devAttention.recall >= devClassical.recall + 0.010), {
    delta: devQuality ? devAttention.recall - devClassical.recall : null,
  });
  addGate(gates, 'semantic F0.5 >= classical + 0.010', Boolean(devQuality
    && devAttention.f05 >= devClassical.f05 + 0.010), {
    delta: devQuality ? devAttention.f05 - devClassical.f05 : null,
  });

  const incrementalPrecision = metric(devAttention, ['incrementalPrecision', 'incremental.precision']);
  addGate(gates, 'attention incremental precision >= 0.800', Boolean(devQuality && incrementalPrecision != null && incrementalPrecision >= 0.8), { incrementalPrecision });

  const failedGates = gates.filter((gate) => !gate.pass);
  const decision = failedGates.length === 0 ? 'ACCEPT_EXPERIMENTAL_ACTIVE' : 'REJECT';
  const onlyDevPending = failedGates.length > 0
    && failedGates.every((g) => (g.name || '').startsWith('semantic') || (g.name || '').includes('dev') || (g.name || '').includes('incremental'));

  const result = {
    schema: 'attention-gate-decision-v2',
    createdAt: new Date().toISOString(),
    decision,
    status: onlyDevPending ? 'READY_FOR_FRESH_HELDOUT' : decision,
    gatesCount: gates.length,
    passedGatesCount: gates.length - failedGates.length,
    failedGates: failedGates.map((gate) => gate.name),
    gates,
  };
  if (outputDecisionPath) writeFileSync(outputDecisionPath, JSON.stringify(result, null, 2), 'utf8');
  return result;
}

async function main() {
  const args = process.argv.slice(2);
  const paths = {};
  for (let i = 0; i < args.length; i++) {
    const key = {
      '--shortlist-config': 'shortlistConfigPath', '--artifact': 'artifactPath',
      '--metadata': 'metadataPath', '--vocab': 'vocabPath',
      '--calibration-report': 'calibrationReportPath', '--internal-test-report': 'internalTestReportPath',
      '--dev-report': 'devReportPath', '--runtime-baseline': 'runtimeBaselinePath',
      '--output-decision': 'outputDecisionPath',
    }[args[i]];
    if (key) paths[key] = args[++i];
  }
  console.log(JSON.stringify(evaluateAttentionGate(paths), null, 2));
}

if (process.argv[1] && process.argv[1].endsWith('evaluate_attention_gate.mjs')) {
  main().catch((err) => { console.error(err); process.exit(1); });
}
