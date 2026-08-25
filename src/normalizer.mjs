// ============================================================
// VietnameseTextNormalizer — plan §7.1
// 3 distinct operations, never mixed:
//   normalizeNfc            -> analysis-only canonical form
//   lowerForLookup          -> dictionary key form (NFC + lowercase)
//   stripVietnameseDiacritics -> accent-stripped lookup form
// NOTE: đ/Đ do NOT decompose in NFD; explicit map required.
// Original text is never mutated; outputs are analysis data only.
// ============================================================

const COMBINING_MARK = /\p{M}/u;

/** NFD then drop combining marks + map đ/Đ explicitly (plan pseudocode §7.1) */
export function stripVietnameseDiacritics(input) {
  const nfd = input.normalize('NFD');
  let out = '';
  for (const ch of nfd) {
    if (COMBINING_MARK.test(ch)) continue;
    if (ch === 'đ') { out += 'd'; continue; }
    if (ch === 'Đ') { out += 'D'; continue; }
    out += ch;
  }
  return out;
}

export function normalizeNfc(value) {
  return value.normalize('NFC');
}

export function lowerForLookup(value) {
  return normalizeNfc(value).toLowerCase();
}

/** Dictionary primary key: NFC + lowercase (plan §15 — never strip dấu for the key). */
export function dictKey(word) {
  return lowerForLookup(word);
}

/** Accent index key: stripped + lowercase (plan §16.1). */
export function accentKey(word) {
  return stripVietnameseDiacritics(lowerForLookup(word));
}

/**
 * True when a single character is a Latin letter carrying at least one
 * combining mark (i.e. an accented letter), or is đ/Đ.
 * Used by ACCENT_CHARACTER_IN_NON_ACCENT_MODE (plan §11.1).
 */
const D_WITH_STROKE = new Set(['đ', 'Đ']);

export function isVietnameseAccentedChar(ch) {
  if (D_WITH_STROKE.has(ch)) return true;
  if (!/\p{L}/u.test(ch)) return false;
  const decomposed = ch.normalize('NFD');
  // base ASCII letter + >=1 combining mark => accented Latin letter
  return decomposed.length > 1 && COMBINING_MARK.test(decomposed.slice(1))
    && /[a-zA-Z]/.test(decomposed[0]);
}

/** True if string contains at least one accented character (incl. đ/Đ). */
export function hasVietnameseAccent(s) {
  for (const ch of s) {
    if (isVietnameseAccentedChar(ch)) return true;
  }
  return false;
}
