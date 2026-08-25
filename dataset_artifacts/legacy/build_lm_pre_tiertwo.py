#!/usr/bin/env python3
"""Build a leakage-audited, bounded-memory Vietnamese n-gram LM.

The builder never reads benchmark rows as training data.  It excludes exact
normalized sentences from the held-out clean source, synthetic ground truth,
Viwiki external benchmark, and VSEC dev/test text/corrected text before global
sentence deduplication and weighting.  N-grams are counted in bounded chunks
and merged with an external heap merge; the old LM is backed up before an
atomic replacement.

Configuration (environment or CLI):

    LM_NEWS_SHARDS=4       # selected deterministically from 17 downloaded shards
    LM_NEWS_MAX_ROWS=0     # 0 means all rows in selected shards
    LM_DOMAIN_WEIGHT=5
    LM_CHUNK_SENTENCES=100000

The default of four shards is explicit in the manifest; it is not the old
silent two-shard default.  All 17 shards total 19,370,237 rows (~4.2 GB), so
four shards are the conservative reproducible default for this workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import html
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT.parent / "binhvq-news-corpus" / "data"
WIKI_DIR = ROOT.parent / "corpus.viwiki-master" / "corpus.viwiki-master" / "viwiki"
DOMAIN_SEED = ROOT / "src" / "data" / "corpus-train.txt"
OUT_DIR = ROOT / "src" / "data"
ARTIFACT_DIR = ROOT / "dataset_artifacts" / "lm"
VSEC_SPLIT_DIR = ROOT / "dataset_artifacts" / "vsec"
BENCHMARK_DIR = ROOT / "benchmark"
CLEAN_SPLIT_DIR = ROOT / "dataset_artifacts" / "clean-source"

DEFAULT_NEWS_SHARDS = 4
DEFAULT_DOMAIN_WEIGHT = 5
DEFAULT_LEX_MIN_FREQ = 40
DEFAULT_BIG_MIN = 2
DEFAULT_TRI_MIN = 3
DEFAULT_MAX_VOCAB = 200_000
DEFAULT_CHUNK_SENTENCES = 100_000
DEFAULT_BATCH_ROWS = 20_000

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
TAG_RE = re.compile(r"<[^>]+>")
CODE_RE = re.compile(r"[0-9\w][0-9\w.\-]*[0-9\w]")
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
SPACE_RE = re.compile(r"\s+", re.UNICODE)
MIN_SENT_TOKENS = 3


def env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise SystemExit(f"{name} must be >= {minimum}, got {value}")
    return value


def normalize_sentence(value: object) -> str:
    """Canonical full-sentence key used for exclusions and deduplication."""

    text = unicodedata.normalize("NFC", html.unescape(str(value or "")))
    return SPACE_RE.sub(" ", text).strip().casefold()


def clean_text(raw: str) -> str:
    text = html.unescape(str(raw or ""))
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    return CODE_RE.sub(lambda match: " " if any(ch.isdigit() for ch in match.group(0)) else match.group(0), text)


def iter_sentences(raw: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (canonical full sentence, lowercase Vietnamese letter tokens)."""

    for sentence in SENT_SPLIT_RE.split(clean_text(raw)):
        canonical = normalize_sentence(sentence)
        tokens = TOKEN_RE.findall(canonical)
        if len(tokens) < MIN_SENT_TOKENS:
            continue
        ascii_only = sum(1 for token in tokens if token.isascii())
        if ascii_only > 0.5 * len(tokens):
            continue
        yield canonical, tokens


def sentence_fingerprint(value: str) -> bytes:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SeenSentenceStore:
    """Disk-backed exact dedup set, bounded in Python heap memory."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("CREATE TABLE seen (fingerprint BLOB PRIMARY KEY)")
        self.connection.commit()
        self.pending = 0

    def add(self, canonical: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen(fingerprint) VALUES (?)",
            (sentence_fingerprint(canonical),),
        )
        self.pending += 1
        if self.pending >= 10_000:
            self.connection.commit()
            self.pending = 0
        return cursor.rowcount == 1

    def close(self) -> None:
        if self.pending:
            self.connection.commit()
        self.connection.close()


class ChunkCounter:
    """Count n-grams in memory and spill sorted chunks to disk."""

    def __init__(self, directory: Path, chunk_sentences: int) -> None:
        self.directory = directory
        self.chunk_sentences = chunk_sentences
        self.uni: dict[str, int] = {}
        self.bi: dict[str, int] = {}
        self.tri: dict[str, int] = {}
        self.total_tokens = 0
        self.sentences = 0
        self.parts: list[Path] = []

    def add_sentence(self, tokens: list[str], weight: int) -> None:
        previous_one = previous_two = None
        for token in tokens:
            self.uni[token] = self.uni.get(token, 0) + weight
            self.total_tokens += weight
            if previous_one is not None:
                key = f"{previous_one} {token}"
                self.bi[key] = self.bi.get(key, 0) + weight
            if previous_two is not None:
                key = f"{previous_two} {previous_one} {token}"
                self.tri[key] = self.tri.get(key, 0) + weight
            previous_two, previous_one = previous_one, token
        self.sentences += 1
        if self.sentences % self.chunk_sentences == 0:
            self.flush()

    def flush(self) -> None:
        if not self.uni and not self.bi and not self.tri:
            return
        path = self.directory / f"part-{len(self.parts):06d}.tsv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for kind, values in (("U", self.uni), ("B", self.bi), ("T", self.tri)):
                for key in sorted(values):
                    handle.write(f"{kind}\t{key}\t{values[key]}\n")
        self.parts.append(path)
        self.uni.clear()
        self.bi.clear()
        self.tri.clear()


def iter_news(paths: list[Path], max_rows: int = 0) -> Iterator[tuple[Path, int, str]]:
    yielded = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        row_index = 0
        for row_group in range(parquet.num_row_groups):
            values = parquet.read_row_group(row_group, columns=["text"]).column("text").to_pylist()
            for raw in values:
                if max_rows and yielded >= max_rows:
                    return
                yielded += 1
                yield path, row_index, raw
                row_index += 1


def iter_wiki(paths: list[Path]) -> Iterator[tuple[Path, int, str]]:
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for index, raw in enumerate(handle):
                    yield path, index, raw
        except OSError:
            continue


def iter_domain(path: Path) -> Iterator[tuple[Path, int, str]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for index, raw in enumerate(handle):
            yield path, index, raw


def read_json_rows(path: Path) -> Iterator[Mapping[str, object]]:
    if not path.exists():
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    for row in value.get("rows", []) if isinstance(value, Mapping) else []:
        if isinstance(row, Mapping):
            yield row


def load_exclusions() -> tuple[set[str], dict[str, int], list[Path]]:
    """Load held-out canonical sentences and provenance paths."""

    sources: list[tuple[str, Path]] = []
    for name in ("clean-dev.txt", "clean-test.txt"):
        sources.append(("clean-source", CLEAN_SPLIT_DIR / name))
    sources.extend([
        ("synthetic", BENCHMARK_DIR / "corpus-synthetic-diacritics.json"),
        ("viwiki-external", BENCHMARK_DIR / "corpus-viwiki-spelling.json"),
        ("vsec-dev", VSEC_SPLIT_DIR / "vsec-dev.jsonl"),
        ("vsec-test", VSEC_SPLIT_DIR / "vsec-test.jsonl"),
    ])
    exclusions: set[str] = set()
    counts: Counter[str] = Counter()
    paths: list[Path] = []
    for category, path in sources:
        if not path.exists():
            continue
        paths.append(path)
        if path.suffix == ".txt":
            values = (line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines())
            for value in values:
                key = normalize_sentence(value)
                if key:
                    if key not in exclusions:
                        counts[category] += 1
                    exclusions.add(key)
            continue
        for row in read_json_rows(path):
            values = []
            for field in ("text", "groundTruth", "corrected_text"):
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    values.append(value)
            for value in values:
                key = normalize_sentence(value)
                if key:
                    if key not in exclusions:
                        counts[category] += 1
                    exclusions.add(key)
    return exclusions, dict(counts), paths


def sorted_news_paths(news_shards: int) -> list[Path]:
    paths = sorted(NEWS_DIR.glob("*.parquet"))
    if news_shards <= 0:
        return paths
    return paths[:news_shards]


def iter_part(path: Path) -> Iterator[tuple[str, str, int]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            kind, key, value = line.rstrip("\n").split("\t", 2)
            yield kind, key, int(value)


def merged_parts(parts: list[Path]) -> Iterator[tuple[str, str, int]]:
    streams = [iter_part(path) for path in parts]
    merged = heapq.merge(*streams, key=lambda item: (item[0], item[1]))
    current: tuple[str, str] | None = None
    total = 0
    for kind, key, value in merged:
        identity = (kind, key)
        if current is not None and identity != current:
            yield current[0], current[1], total
            total = 0
        current = identity
        total += value
    if current is not None:
        yield current[0], current[1], total


def write_outputs(counter: ChunkCounter, total_tokens: int, config: Mapping[str, int], temp_dir: Path) -> tuple[Path, Path, dict[str, int]]:
    counter.flush()
    min_uni = config["uni_min"]
    min_bigram = config["big_min"]
    min_trigram = config["tri_min"]
    max_vocab = config["max_vocab"]

    unigram = [(key, value) for kind, key, value in merged_parts(counter.parts) if kind == "U" and value >= min_uni]
    unigram.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab and len(unigram) > max_vocab:
        unigram = unigram[:max_vocab]
    vocab = {key for key, _ in unigram}

    lm_path = temp_dir / "lm-ngrams.tsv"
    with lm_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("#SMS-LM v1\n")
        handle.write(f"#tokens={total_tokens}\n")
        for key, value in unigram:
            handle.write(f"U\t{key}\t{value}\n")
        for kind, key, value in merged_parts(counter.parts):
            if kind == "B" and value >= min_bigram and all(token in vocab for token in key.split()):
                handle.write(f"B\t{key}\t{value}\n")
            elif kind == "T" and value >= min_trigram and all(token in vocab for token in key.split()):
                handle.write(f"T\t{key}\t{value}\n")

    lex_path = temp_dir / "lexicon-built.txt"
    with lex_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Built by tools/build_lm.py — word<TAB>freq\n")
        for key, value in unigram:
            if value >= config["lex_min_freq"]:
                handle.write(f"{key}\t{value}\tA\n")
    return lm_path, lex_path, {"unigrams": len(unigram), "lexicon": sum(value >= config["lex_min_freq"] for _, value in unigram)}


def validate_lm(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    total_tokens = None
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n")
        if first != "#SMS-LM v1":
            raise ValueError(f"invalid LM header: {first!r}")
        token_line = handle.readline().rstrip("\n")
        if not token_line.startswith("#tokens="):
            raise ValueError("LM token header missing")
        total_tokens = int(token_line.split("=", 1)[1])
        for line_number, line in enumerate(handle, 3):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3 or fields[0] not in {"U", "B", "T"} or not fields[1] or int(fields[2]) <= 0:
                raise ValueError(f"invalid LM line {line_number}")
            counts[fields[0]] += 1
    if total_tokens <= 0 or not counts["U"]:
        raise ValueError("LM has no positive token/unigram counts")
    return {"tokens": total_tokens, **dict(counts)}


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_manifest(config: Mapping[str, object], source_records: list[dict[str, object]], exclusion_counts: Mapping[str, int], stats: Mapping[str, object], lm_path: Path, lex_path: Path) -> dict[str, object]:
    return {
        "format": "sms-lm-manifest-v2",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "config": dict(config),
        "sourceFiles": source_records,
        "exclusions": {
            "categories": dict(exclusion_counts),
            "totalCanonicalSentences": stats["exclusion_set_size"],
            "filteredBySource": stats["filtered_by_source"],
            "postFilterExactIntersection": 0,
            "policy": "canonical full sentence (NFC/casefold/whitespace) before global deduplication and weighting",
        },
        "counts": stats,
        "outputs": {
            "lm": {"path": relative(lm_path), "bytes": lm_path.stat().st_size, "sha256": sha256_file(lm_path)},
            "lexicon": {"path": relative(lex_path), "bytes": lex_path.stat().st_size, "sha256": sha256_file(lex_path)},
        },
        "resourcePolicy": {
            "newsShardInventory": 17,
            "selectedNewsShards": config["news_shards"],
            "reason": "all 17 shards contain 19,370,237 rows (~4.2 GB); default four shards keeps build bounded and reproducible",
            "boundedMemory": "disk-backed sentence dedup + sorted n-gram chunks",
        },
    }


def replace_atomically(lm_tmp: Path, lex_tmp: Path, manifest: dict[str, object], lm_path: Path, lex_path: Path, manifest_path: Path) -> Path | None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = ARTIFACT_DIR / time.strftime("legacy-%Y%m%dT%H%M%SZ", time.gmtime())
    backups: list[tuple[Path, Path]] = []
    for target in (lm_path, lex_path, manifest_path):
        if target.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / target.name
            shutil.copy2(target, backup)
            backups.append((target, backup))
    manifest_tmp = manifest_path.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.replace(lm_tmp, lm_path)
        os.replace(lex_tmp, lex_path)
        os.replace(manifest_tmp, manifest_path)
    except Exception:
        for target, backup in backups:
            shutil.copy2(backup, target)
        if manifest_tmp.exists():
            manifest_tmp.unlink()
        raise
    return backup_dir if backups else None


def audit_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lm_path = ROOT / manifest["outputs"]["lm"]["path"]
    lex_path = ROOT / manifest["outputs"]["lexicon"]["path"]
    parsed = validate_lm(lm_path)
    return {
        "pass": sha256_file(lm_path) == manifest["outputs"]["lm"]["sha256"]
        and sha256_file(lex_path) == manifest["outputs"]["lexicon"]["sha256"]
        and parsed["tokens"] == manifest["counts"]["weightedTokens"],
        "manifest": str(path),
        "lm": parsed,
        "postFilterExactIntersection": manifest["exclusions"].get("postFilterExactIntersection"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-shards", type=int, default=env_int("LM_NEWS_SHARDS", DEFAULT_NEWS_SHARDS, 0))
    parser.add_argument("--max-news-rows", type=int, default=env_int("LM_NEWS_MAX_ROWS", 0, 0))
    parser.add_argument("--domain-weight", type=int, default=env_int("LM_DOMAIN_WEIGHT", DEFAULT_DOMAIN_WEIGHT, 1))
    parser.add_argument("--chunk-sentences", type=int, default=env_int("LM_CHUNK_SENTENCES", DEFAULT_CHUNK_SENTENCES, 1))
    parser.add_argument("--lex-min-freq", type=int, default=env_int("LM_LEX_MIN_FREQ", DEFAULT_LEX_MIN_FREQ, 1))
    parser.add_argument("--uni-min", type=int, default=env_int("LM_UNI_MIN", 2, 1))
    parser.add_argument("--big-min", type=int, default=env_int("LM_BIG_MIN", DEFAULT_BIG_MIN, 1))
    parser.add_argument("--tri-min", type=int, default=env_int("LM_TRI_MIN", DEFAULT_TRI_MIN, 1))
    parser.add_argument("--max-vocab", type=int, default=env_int("LM_MAX_VOCAB", DEFAULT_MAX_VOCAB, 0))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--manifest", type=Path, default=OUT_DIR / "lm-ngrams.manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.audit:
        print(json.dumps(audit_manifest(manifest_path), ensure_ascii=False, indent=2))
        return 0

    selected_news = sorted_news_paths(args.news_shards)
    wiki_paths = sorted(WIKI_DIR.rglob("*.txt"))
    exclusions, exclusion_counts, exclusion_paths = load_exclusions()
    temp_root = Path(tempfile.mkdtemp(prefix="sms-lm-build-", dir=str(ARTIFACT_DIR)))
    seen_path = temp_root / "seen.sqlite3"
    parts_dir = temp_root / "parts"
    parts_dir.mkdir()
    seen = SeenSentenceStore(seen_path)
    counter = ChunkCounter(parts_dir, args.chunk_sentences)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    filtered_by_source: Counter[str] = Counter()
    started = time.time()

    def feed(source: str, records: Iterable[tuple[Path, int, str]], weight: int) -> None:
        for path, row_index, raw in records:
            source_counts[source]["rawRows"] += 1
            if not isinstance(raw, str):
                source_counts[source]["invalidRows"] += 1
                continue
            for canonical, tokens in iter_sentences(raw):
                source_counts[source]["candidateSentences"] += 1
                if canonical in exclusions:
                    filtered_by_source[source] += 1
                    continue
                if not seen.add(canonical):
                    source_counts[source]["dedupedSentences"] += 1
                    continue
                counter.add_sentence(tokens, weight)
                source_counts[source]["keptSentences"] += 1
            if source == "news" and source_counts[source]["rawRows"] % 100_000 == 0:
                print(f"[news] rows={source_counts[source]['rawRows']:,} kept={source_counts[source]['keptSentences']:,} chunks={len(counter.parts)} elapsed={time.time()-started:.0f}s", file=sys.stderr)

    try:
        print(f"== source: news ({len(selected_news)} of 17 shards) ==", file=sys.stderr)
        feed("news", iter_news(selected_news, args.max_news_rows), 1)
        print(f"== source: wikipedia ({len(wiki_paths)} files) ==", file=sys.stderr)
        feed("wikipedia", iter_wiki(wiki_paths), 1)
        print(f"== source: domain seed x{args.domain_weight} ==", file=sys.stderr)
        feed("domain", iter_domain(DOMAIN_SEED), args.domain_weight)
        seen.close()

        config = {
            "news_shards": args.news_shards,
            "max_news_rows": args.max_news_rows,
            "domain_weight": args.domain_weight,
            "chunk_sentences": args.chunk_sentences,
            "lex_min_freq": args.lex_min_freq,
            "uni_min": args.uni_min,
            "big_min": args.big_min,
            "tri_min": args.tri_min,
            "max_vocab": args.max_vocab,
        }
        lm_tmp, lex_tmp, output_counts = write_outputs(counter, counter.total_tokens, config, temp_root)
        parsed = validate_lm(lm_tmp)
        source_paths = selected_news + wiki_paths + ([DOMAIN_SEED] if DOMAIN_SEED.exists() else [])
        source_records = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_paths]
        source_stats = {source: dict(counts) for source, counts in source_counts.items()}
        stats = {
            "sourceCounts": source_stats,
            "filtered_by_source": dict(filtered_by_source),
            "weightedTokens": counter.total_tokens,
            "keptSentences": counter.sentences,
            "dedupedSentences": sum(counts.get("dedupedSentences", 0) for counts in source_counts.values()),
            "exclusion_set_size": len(exclusions),
            "outputCounts": {**output_counts, **{key.lower(): value for key, value in parsed.items()}},
            "tempChunkCount": len(counter.parts),
            "elapsedSeconds": round(time.time() - started, 2),
        }
        manifest = build_manifest(config, source_records, exclusion_counts, stats, OUT_DIR / "lm-ngrams.tsv", OUT_DIR / "lexicon-built.txt")
        # Hashes in manifest must describe the staged files before replacement.
        manifest["outputs"]["lm"].update({"bytes": lm_tmp.stat().st_size, "sha256": sha256_file(lm_tmp)})
        manifest["outputs"]["lexicon"].update({"bytes": lex_tmp.stat().st_size, "sha256": sha256_file(lex_tmp)})
        backup = replace_atomically(lm_tmp, lex_tmp, manifest, OUT_DIR / "lm-ngrams.tsv", OUT_DIR / "lexicon-built.txt", manifest_path)
        print(json.dumps({"manifest": str(manifest_path), "backup": str(backup) if backup else None, "counts": stats, "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 0
    finally:
        try:
            seen.close()
        except Exception:
            pass
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
