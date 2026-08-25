// ============================================================
// SmsTokenizer — plan §6.1 / §7
// Scans the ORIGINAL string only; every token keeps exact
// UTF-16 offsets so substring(start,end) === original.
// Token types: WORD | NUMBER | PUNCTUATION | WHITESPACE | OTHER
// ============================================================
import { TokenType } from './core.mjs';
import { dictKey } from './normalizer.mjs';

// Scan order matters: whitespace, letters(+marks for NFD safety), digits, everything else.
const TOKEN_RE = /(\s+)|([\p{L}\p{M}]+)|(\d+)|([^\s\p{L}\p{M}\d]+)/gu;

const PUNCT_SET = new Set([...'. , ! ? : ; - – — / \\ ( ) [ ] { } " \' ` ~ @ # $ % ^ & * + = | < >']);

function classifyOtherRun(run) {
  for (const ch of run) {
    if (!PUNCT_SET.has(ch)) return TokenType.OTHER; // emoji, symbols...
  }
  return TokenType.PUNCTUATION;
}

/**
 * @param {string} text original SMS content
 * @returns {Array<{original:string, normalized:string, start:number, end:number, type:TokenType}>}
 */
export function tokenize(text) {
  const tokens = [];
  TOKEN_RE.lastIndex = 0;
  let m;
  while ((m = TOKEN_RE.exec(text)) !== null) {
    const [raw, ws, word, num, other] = m;
    const start = m.index;
    const end = start + raw.length;
    let type;
    if (ws !== undefined) type = TokenType.WHITESPACE;
    else if (word !== undefined) type = TokenType.WORD;
    else if (num !== undefined) type = TokenType.NUMBER;
    else type = classifyOtherRun(raw);

    tokens.push({
      original: raw,
      // analysis form: NFC + lowercase; offsets NEVER come from this
      normalized: type === TokenType.WORD || type === TokenType.OTHER
        ? dictKey(raw)
        : raw.toLowerCase(),
      start,
      end,
      type,
    });
  }
  return tokens;
}

/** Convenience: WORD tokens only. */
export function wordTokens(tokens) {
  return tokens.filter((t) => t.type === TokenType.WORD);
}
