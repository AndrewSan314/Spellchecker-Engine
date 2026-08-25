# ============================================================================
# build_lm.py — Corpus -> CLEAN + DEDUP -> Lexicon + Language Model artifacts
#
# Sources (per architecture):
#   - binhvq/news-corpus      : parquet shards, single `text` column
#   - Vietnamese Wikipedia    : viwiki/*.txt crawled pages (81 MB)
#   - Domain seed (optional)  : src/data/corpus-train.txt, upweighted xDOMAIN_WEIGHT
#
# Outputs (into sms-validation-demo/src/data/):
#   - lm-ngrams.tsv     : pruned unigram/bigram/trigram counts (fast-path LM)
#   - lexicon-built.txt : word<TAB>freq (curated seed merged in, weighted)
#
# Cleaning mirrors the ENGINE's counting semantics:
#   NFC normalize -> drop mixed letter+digit clusters (GIAM50K lesson) ->
#   unicode-letters-only tokens -> lowercase lookup form.
# Dedup: exact-line hash across the WHOLE stream.
# Pruning keeps memory bounded: count==1 n-grams are dropped at every flush.
# ============================================================================
import glob
import hashlib
import html
import os
import re
import sys
import time

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWS_GLOB = os.path.join(ROOT, "..", "binhvq-news-corpus", "data", "*.parquet")
WIKI_DIR = os.path.join(ROOT, "..", "corpus.viwiki-master",
                        "corpus.viwiki-master", "viwiki")
DOMAIN_SEED = os.path.join(ROOT, "src", "data", "corpus-train.txt")
OUT_DIR = os.path.join(ROOT, "src", "data")

NEWS_SHARDS = int(os.environ.get("LM_NEWS_SHARDS", "2"))
DOMAIN_WEIGHT = int(os.environ.get("LM_DOMAIN_WEIGHT", "5"))
LEX_MIN_FREQ = int(os.environ.get("LM_LEX_MIN_FREQ", "40"))
BIG_MIN = 2
TRI_MIN = 3
MAX_VOCAB = 200_000
ROWGROUP_BATCH = 20_000  # rows per counting flush

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
TAG_RE = re.compile(r"<[^>]+>")
CODE_RE = re.compile(r"[0-9\w][0-9\w.\-]*[0-9\w]")
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # unicode letters only
# sentence boundary split BEFORE counting — n-grams must never span
# sentence edges ("...nhat. Ky..." must not create "nhat ky")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
MIN_SENT_TOKENS = 3


def clean_text(raw: str):
    """-> normalized, URL/tag-free text with code clusters blanked."""
    s = html.unescape(raw)
    s = TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = s.normalize("NFC") if hasattr(s, "normalize") else s
    # drop mixed letter+digit clusters BEFORE tokenizing (GIAM50K lesson)
    return CODE_RE.sub(lambda m: " " if any(ch.isdigit() for ch in m.group(0))
                       else m.group(0), s)


def iter_sentences(raw: str):
    """Yield (tokens, ok) per sentence; None-yields are skipped upstream."""
    body = clean_text(raw)
    for sent in SENT_SPLIT_RE.split(body):
        toks = TOKEN_RE.findall(sent.lower())
        if len(toks) < MIN_SENT_TOKENS:
            continue
        ascii_only = sum(1 for t in toks if t.isascii())
        if ascii_only > 0.5 * len(toks):
            continue  # not Vietnamese enough
        yield toks


class Counter3:
    def __init__(self):
        self.uni = {}
        self.bi = {}
        self.tri = {}
        self.total_tokens = 0
        self.seen_lines = set()
        self.kept_docs = 0
        self.kept_sentences = 0

    def add_sentence(self, toks, weight=1):
        u, b, t = self.uni, self.bi, self.tri
        p1 = p2 = None
        for w in toks:
            u[w] = u.get(w, 0) + weight
            self.total_tokens += weight
            if p1 is not None:
                k = f"{p1} {w}"
                b[k] = b.get(k, 0) + weight
            if p2 is not None:
                k = f"{p2} {p1} {w}"
                t[k] = t.get(k, 0) + weight
            p2, p1 = p1, w

    def add_document(self, raw, weight=1):
        """Dedup at DOCUMENT level; count per-SENTENCE inside it."""
        key = hashlib.md5(" ".join(raw.split()).encode("utf-8")).hexdigest()
        if key in self.seen_lines:
            return False
        self.seen_lines.add(key)
        n = 0
        for toks in iter_sentences(raw):
            self.add_sentence(toks, weight)
            n += 1
        if n:
            self.kept_docs += 1
            self.kept_sentences += n
        return n > 0

    def flush_prune(self):
        """Drop singleton entries to bound memory between batches."""
        for d, mn in ((self.bi, BIG_MIN), (self.tri, TRI_MIN)):
            dead = [k for k, v in d.items() if v < mn]
            for k in dead:
                del d[k]
        return len(dead_total(self)) if False else (
            len(self.bi), len(self.tri))


def dead_total(_):
    return 0


def iter_news(shards_limit):
    for i, path in enumerate(sorted(glob.glob(NEWS_GLOB))):
        if shards_limit and i >= shards_limit:
            break
        pf = pq.ParquetFile(path)
        buf = []
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=["text"]).column("text")
            for s in col.to_pylist():
                buf.append(s)
                if len(buf) >= ROWGROUP_BATCH:
                    yield from buf
                    buf.clear()
        yield from buf
        print(f"  [news] shard {i} done: {path}", file=sys.stderr)


def iter_wiki():
    for path in sorted(glob.glob(os.path.join(WIKI_DIR, "**", "*.txt"),
                                 recursive=True)):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    yield raw
        except OSError:
            continue


def main():
    t0 = time.time()
    c = Counter3()

    def feed(lines_iter, weight):
        added = 0
        for raw in lines_iter:
            if not isinstance(raw, str):
                continue
            if c.add_document(raw, weight):
                added += 1
            if c.kept_sentences and c.kept_sentences % 200_000 < 60:
                nb, nt = c.flush_prune()
                print(f"    ... {c.kept_sentences} sents | bi={nb:,} "
                      f"tri={nt:,}", file=sys.stderr)
        return added

    print("== source: news ==", file=sys.stderr)
    feed(iter_news(NEWS_SHARDS), 1)
    print(f"  after news: docs={c.kept_docs:,} sents={c.kept_sentences:,} "
          f"tok={c.total_tokens:,}", file=sys.stderr)

    print("== source: wikipedia ==", file=sys.stderr)
    feed(iter_wiki(), 1)

    print(f"== source: domain seed x{DOMAIN_WEIGHT} ==", file=sys.stderr)
    with open(DOMAIN_SEED, encoding="utf-8") as fh:
        feed(fh, DOMAIN_WEIGHT)

    c.flush_prune()
    # hard-prune singletons that survived within one batch window
    for d, mn in ((c.uni, 2),):
        for k in [k for k, v in d.items() if v < mn]:
            del d[k]

    # ---- vocabulary cap -------------------------------------------------
    if len(c.uni) > MAX_VOCAB:
        keep = set(sorted(c.uni, key=c.uni.get, reverse=True)[:MAX_VOCAB])
        c.uni = {k: v for k, v in c.uni.items() if k in keep}
        c.bi = {k: v for k, v in c.bi.items()
                if all(w in keep for w in k.split())}
        c.tri = {k: v for k, v in c.tri.items()
                 if all(w in keep for w in k.split())}

    # ---- write LM artifact ----------------------------------------------
    lm_path = os.path.join(OUT_DIR, "lm-ngrams.tsv")
    with open(lm_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("#SMS-LM v1\n")
        fh.write(f"#tokens={c.total_tokens}\n")
        for w, v in c.uni.items():
            fh.write(f"U\t{w}\t{v}\n")
        for k, v in c.bi.items():
            fh.write(f"B\t{k}\t{v}\n")
        for k, v in c.tri.items():
            fh.write(f"T\t{k}\t{v}\n")

    # ---- write lexicon ---------------------------------------------------
    lex_path = os.path.join(OUT_DIR, "lexicon-built.txt")
    n_lex = 0
    with open(lex_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Built by tools/build_lm.py — word<TAB>freq\n")
        for w, v in sorted(c.uni.items(), key=lambda kv: -kv[1]):
            if v >= LEX_MIN_FREQ:
                fh.write(f"{w}\t{v}\tA\n")
                n_lex += 1

    print(f"DONE in {time.time()-t0:.0f}s | docs={c.kept_docs:,} "
          f"| sents={c.kept_sentences:,} | tokens={c.total_tokens:,} "
          f"| uni={len(c.uni):,} | bi={len(c.bi):,} "
          f"| tri={len(c.tri):,} | lex={n_lex:,}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
