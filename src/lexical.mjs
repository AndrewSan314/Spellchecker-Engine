// ============================================================
// Lexical layer — plan §14 / §15 / §16 / §23
//   LexiconService       dictionary lookups (immutable, in-memory)
//   WhitelistService     SYSTEM/BRAND whitelist
//   AbbreviationService  brandname-scoped abbreviation config
//   AccentIndex          reverse accent index (stripped -> candidates)
// All loaded once at startup into immutable maps. No DB per token (§13).
// ============================================================
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { dictKey, accentKey } from './normalizer.mjs';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');
const MAX_ACCENT_CANDIDATES_PER_TOKEN = 12; // plan §16.3

export class LexiconService {
  /** @param {Map<string,{word:string,freq:number,type:string,src:string}>} entries */
  constructor(entries) {
    this.entries = new Map(entries); // key = dictKey(word) — dấu KEPT (plan §15)
    Object.freeze(this);
  }
  /**
   * plan §10: corpus-backed frequency vs hand-assigned seed values.
   * 'curated'-only entries carry NO real corpus evidence — gates that
   * reason about language statistics must use realFrequency(), not the
   * placeholder number.
   */
  realFrequency(normalizedWord) {
    const e = this.find(normalizedWord);
    return e && e.src !== 'curated' ? e.freq : 0;
  }
  static fromEntries(entriesIterable) {
    return new LexiconService(entriesIterable);
  }
  static load(file = path.join(DATA_DIR, 'lexicon.txt')) {
    const map = new Map();
    const raw = readFileSync(file, 'utf8');
    for (const lineRaw of raw.split(/\r?\n/)) {
      const line = lineRaw.trim();
      if (!line || line.startsWith('#')) continue;
      const [word, freqStr, type = 'A'] = line.split('\t');
      if (!word) continue;
      const key = dictKey(word);
      const freq = Number.parseInt(freqStr, 10) || 1;
      const prev = map.get(key);
      // merge duplicates keeping max frequency
      if (!prev || freq > prev.freq) map.set(key, { word, freq, type });
    }
    return new LexiconService(map);
  }
  find(normalizedWord) {
    return this.entries.get(dictKey(normalizedWord)) ?? null;
  }
  contains(normalizedWord) {
    return this.entries.has(dictKey(normalizedWord));
  }
  frequency(normalizedWord) {
    return this.find(normalizedWord)?.freq ?? 0;
  }
  allWords() {
    return [...this.entries.values()];
  }
}

export class WhitelistService {
  constructor(terms) {
    this.terms = new Set(terms.map((t) => t.toLowerCase()));
    Object.freeze(this.terms);
    Object.freeze(this);
  }
  static load(file = path.join(DATA_DIR, 'whitelist.txt')) {
    const terms = readFileSync(file, 'utf8')
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith('#'));
    return new WhitelistService(terms);
  }
  isAllowed(token, _ctx) {
    // case-insensitive technical/brand terms (plan §23)
    return this.terms.has(token.toLowerCase());
  }
}

export class AbbreviationService {
  constructor(config) {
    this.config = config;
    this.entrySet = new Set(config.entries);
    Object.freeze(this.entrySet);
    Object.freeze(this);
  }
  static load(file = path.join(DATA_DIR, 'abbreviations.json')) {
    return new AbbreviationService(JSON.parse(readFileSync(file, 'utf8')));
  }
  isRuleEnabledFor(brandname) {
    return Boolean(brandname) && this.config.ruleEnabledForBrandname[brandname] === true;
  }
  isAbbreviation(token) {
    return this.entrySet.has(token); // exact whole-token match only
  }
}

/**
 * AccentIndex — plan §16.1..§16.3
 * stripped(lowercased) word -> [{word,freq}] sorted desc frequency, capped.
 */
export class AccentIndex {
  constructor(index) {
    this.index = index;
    Object.freeze(this.index);
    Object.freeze(this);
  }
  static build(lexicon) {
    const index = new Map();
    for (const { word, freq } of lexicon.allWords()) {
      const stripped = accentKey(word);
      if (!index.has(stripped)) index.set(stripped, []);
      index.get(stripped).push({ word, freq });
    }
    for (const list of index.values()) {
      list.sort((a, b) => b.freq - a.freq || a.word.localeCompare(b.word));
      list.length = Math.min(list.length, MAX_ACCENT_CANDIDATES_PER_TOKEN);
    }
    return new AccentIndex(index);
  }
  candidates(strippedLowerToken) {
    return this.index.get(strippedLowerToken) ?? [];
  }
}
