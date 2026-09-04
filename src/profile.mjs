// ============================================================
// Serving profiles — which DATA ARTIFACTS the engine loads.
//
// The engine's cost is dominated by two artifacts, not by code:
//   lm-ngrams.tsv     80 MB  -> ~300-500 MB RSS, ~5 s cold start
//   lexicon-built.txt 51k words -> ~200 MB once SymSpell indexes deletions
//
// `full` is the published research configuration. `lite` is the local /
// demo configuration: an SMS-domain-pruned pair built by
// tools/build_sms_profile.mjs that keeps the SAME counts for the words it
// keeps, so calibrated thresholds keep their meaning.
//
// ONE environment variable decides it — ENGINE_PROFILE — and both artifacts
// move together, so the LM and the lexicon can never disagree about which
// words exist. `LM_ARTIFACT` / `LEXICON_ARTIFACT` stay as explicit escape
// hatches for tooling.
//
// Nothing here changes rule behaviour or thresholds; those live in
// src/config.mjs + config/spelling-tuning.json.
// ============================================================
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');

export const PROFILES = Object.freeze({
  full: Object.freeze({
    lm: 'lm-ngrams.tsv',
    lexicon: 'lexicon-built.txt',
    describe: 'published research configuration (80 MB LM, 51k-word lexicon)',
  }),
  lite: Object.freeze({
    lm: 'lm-ngrams.sms.tsv',
    lexicon: 'lexicon-sms.txt',
    describe: 'SMS-domain profile for local runs (built by tools/build_sms_profile.mjs)',
  }),
});

export const DEFAULT_PROFILE = 'full';

let warned = new Set();
function warnOnce(key, message) {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(message);
}

/** exported for tests — forget which warnings were already printed */
export function resetProfileWarnings() {
  warned = new Set();
}

export function profileName(env = process.env) {
  const wanted = env.ENGINE_PROFILE ?? DEFAULT_PROFILE;
  if (!PROFILES[wanted]) {
    throw new Error(`unknown ENGINE_PROFILE "${wanted}" `
      + `(known: ${Object.keys(PROFILES).join(', ')})`);
  }
  return wanted;
}

/**
 * @param {'lm'|'lexicon'} kind
 * @returns {{path: string, profile: string, requested: string}}
 *   A `lite` artifact that has not been built falls back to `full` with one
 *   warning — never silently, and never a crash on a fresh clone.
 */
export function resolveArtifact(kind, env = process.env) {
  const override = kind === 'lm' ? env.LM_ARTIFACT : env.LEXICON_ARTIFACT;
  if (override) return { path: override, profile: 'custom', requested: 'custom' };
  const requested = profileName(env);
  const file = PROFILES[requested][kind];
  const full = path.join(DATA_DIR, file);
  if (requested !== DEFAULT_PROFILE && !existsSync(full)) {
    warnOnce(`${requested}:${kind}`,
      `[profile] ENGINE_PROFILE=${requested} wants ${file}, which is missing — `
      + 'using the full artifact. Build it with: node tools/build_sms_profile.mjs');
    return {
      path: path.join(DATA_DIR, PROFILES[DEFAULT_PROFILE][kind]),
      profile: DEFAULT_PROFILE,
      requested,
    };
  }
  return { path: full, profile: requested, requested };
}
