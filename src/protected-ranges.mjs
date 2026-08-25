// ============================================================
// Protected Range Detection — plan §8
// Goal: keep punctuation/spell rules away from URLs, emails,
// phones, dates/times, numbers/money, codes and placeholders.
// All detectors run on the ORIGINAL text and report UTF-16 spans.
// ============================================================
import { ProtectedRangeType } from './core.mjs';

/** plan §8.2 overlap priority (higher wins) */
const TYPE_PRIORITY = {
  [ProtectedRangeType.URL]: 90,
  [ProtectedRangeType.EMAIL]: 85,
  [ProtectedRangeType.PLACEHOLDER]: 80,
  [ProtectedRangeType.TRANSACTION_CODE]: 70,
  [ProtectedRangeType.PRODUCT_CODE]: 70,
  [ProtectedRangeType.PHONE]: 60,
  [ProtectedRangeType.DATE]: 50,
  [ProtectedRangeType.TIME]: 50,
  [ProtectedRangeType.MONEY]: 40,
  [ProtectedRangeType.NUMBER]: 30,
};

export class ProtectedRange {
  constructor(start, end, type) {
    this.start = start;
    this.end = end;
    this.type = type;
    Object.freeze(this);
  }
  overlaps(s, e) { return s < this.end && e > this.start; }   // plan §6.2
  contains(s, e) { return s >= this.start && e <= this.end; }
}

// ---------- individual detectors (plan §8.3..§8.9) ----------

// Scheme / www forms are unambiguous. Bare domains require a known TLD to
// avoid protecting dotted initialisms like "A.B.C" (plan §11.7 note).
const KNOWN_TLDS = [
  'com', 'net', 'org', 'gov', 'edu', 'info', 'io', 'co', 'me', 'tv', 'app', 'dev',
  'vn', 'com.vn', 'net.vn', 'org.vn', 'edu.vn', 'gov.vn',
];
const TLD_ALT = KNOWN_TLDS.map((t) => t.replace(/\./g, '\\.')).join('|');
const URL_RE = new RegExp(
  String.raw`(?:https?:\/\/|www\.)[^\s]+` +                       // scheme / www form
  String.raw`|[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)*\.(?:${TLD_ALT})\b(?:\/[^\s]*)?`,
  'giu',
);

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/gu;

// VN phone baseline (plan §8.5): 09xxxxxxxx / +84xxxxxxxxx / 84 xxx xxx xxx.
// Must NOT swallow dates ("12-10-2026") or short numbers => anchored prefixes.
const PHONE_RE = /(?:(?:\+|00)?84|0)(?:[\s.-]?\d){8,10}/gu;

// Strict calendar-ish date so "10.000" is not eaten as a date (day<=31, month<=12).
const DATE_RE = /\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])(?:[/.-](?:\d{4}|\d{2}))?\b/gu;
const TIME_RE = /\b\d{1,2}:\d{2}(?::\d{2})?\b/gu;

// Money / number with separators: 10.000 | 1,000,000 | 10,5 | 1.5 (plan §8.7)
// Lookarounds instead of \b so currency-suffixed forms ("1.500.000d") and
// letter neighbours don't break protection.
const NUMBER_RE = /(?<![\d.,])\d{1,3}(?:[.,]\d{3})+(?![\d.,])|(?<![\d.,])\d+[.,]\d+(?![\d.,])/gu;

// Codes: letters+digits mix, >=5 chars, >=2 digits — DH123456 VT001 ABC-2026 OTP123
// Deliberately narrow (plan §8.8: "không được regex quá rộng").
const CODE_RE = /\b[A-Za-z]{1,6}-\d{2,}[A-Za-z0-9]*\b|\b[A-Za-z]{2,6}\d{2,}[A-Za-z0-9]*\b/gu;

// Placeholders: {{customer_name}} ${otp} {amount} (plan §8.9)
const PLACEHOLDER_RE = /\{\{\s*[\w.]+\s*\}\}|\$\{[^}\s]+\}|\{[\w.]+\}/gu;

function collect(re, text, type, out) {
  re.lastIndex = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[0].length === 0) { re.lastIndex++; continue; }
    let end = m.index + m[0].length;
    // plan "Fix URL protected range": a URL match must stop at the REAL end
    // of the URL. Sentence punctuation glued to the URL must stay OUTSIDE
    // the protected span so whitespace rules can flag it.
    //   host region  (scheme..before first '/'): cut at , ! ? ; :
    //   path region  (after first '/'):          cut at , ; !   (? & = are
    //   legitimate query characters and are preserved).
    if (type === ProtectedRangeType.URL) {
      const s = m[0];
      const schemeEnd = s.startsWith('www.')
        ? 4
        : Math.max(s.indexOf('://') + 3, 0);
      const slash = s.indexOf('/', schemeEnd);
      const scanFrom = slash === -1 ? schemeEnd : slash;
      let cut = s.length;
      for (let k = scanFrom; k < s.length; k++) {
        const c = s[k];
        const bad = slash !== -1 && k >= slash
          ? ',;'.includes(c) || c === '!'
          : ',:;?!'.includes(c);
        if (bad) { cut = k; break; }
      }
      // trailing sentence dots ("…com." / "…com...") when what remains ends
      // alphanumerically; path ellipsis mid-URL is untouched.
      while (cut > schemeEnd && s[cut - 1] === '.'
        && /[A-Za-z0-9]/.test(s[cut - 2] ?? '')) cut--;
      end = m.index + Math.max(cut, schemeEnd + 1);
    }
    if (end - m.index <= 0) continue;
    out.push(new ProtectedRange(m.index, end, type));
  }
}

/**
 * CompositeProtectedRangeDetector — plan §8.1:
 * run all -> merge duplicates -> resolve overlaps by type priority -> sort.
 */
export function detectProtectedRanges(text) {
  const raw = [];
  collect(URL_RE, text, ProtectedRangeType.URL, raw);
  collect(EMAIL_RE, text, ProtectedRangeType.EMAIL, raw);
  collect(PHONE_RE, text, ProtectedRangeType.PHONE, raw);
  collect(DATE_RE, text, ProtectedRangeType.DATE, raw);
  collect(TIME_RE, text, ProtectedRangeType.TIME, raw);
  collect(NUMBER_RE, text, ProtectedRangeType.NUMBER, raw);
  collect(CODE_RE, text, ProtectedRangeType.PRODUCT_CODE, raw);
  collect(PLACEHOLDER_RE, text, ProtectedRangeType.PLACEHOLDER, raw);

  // exact-duplicate merge
  const seen = new Map();
  for (const r of raw) {
    const k = `${r.start}:${r.end}:${r.type}`;
    if (!seen.has(k)) seen.set(k, r);
  }

  // overlap resolution by priority, deterministic order
  const sorted = [...seen.values()].sort(
    (a, b) => a.start - b.start || a.end - b.end || TYPE_PRIORITY[b.type] - TYPE_PRIORITY[a.type],
  );
  const kept = [];
  for (const r of sorted) {
    const clash = kept.find((k) => k.overlaps(r.start, r.end));
    if (!clash) kept.push(r);
    else if (TYPE_PRIORITY[r.type] > TYPE_PRIORITY[clash.type]) {
      // replace lower-priority overlapping range (only when fully covering for simplicity)
      if (r.contains(clash.start, clash.end)) {
        kept.splice(kept.indexOf(clash), 1, r);
      }
    }
  }
  return kept.sort((a, b) => a.start - b.start || a.end - b.end);
}
