"""Rebuild the higher-order tier while preserving the full 17-shard vocab.

The expensive unigram pass has already been completed in the full artifact.
This tool rescans only the four deterministic higher-order news shards plus
Wikipedia, the domain seed, and valid VSEC train rows. It then replaces the
trigram section (rather than adding duplicate counts) and keeps the existing
full-vocabulary unigram/bigram sections byte-for-byte equivalent in meaning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_lm


def read_lm(path: Path) -> tuple[int, dict[str, int], dict[str, int], dict[str, int]]:
    total = 0
    unigram: dict[str, int] = {}
    bigram: dict[str, int] = {}
    trigram: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#tokens="):
                total = int(line.split("=", 1)[1])
                continue
            if line.startswith("#"):
                continue
            kind, key, value = line.split("	", 2)
            target = {"U": unigram, "B": bigram, "T": trigram}[kind]
            target[key] = int(value)
    if total <= 0 or not unigram:
        raise ValueError(f"invalid base LM: {path}")
    return total, unigram, bigram, trigram


def scan_higher_sources(
    temp_root: Path,
    news: list[Path],
    wiki: list[Path],
    domain: Path,
    vsec: Path,
    exclusions: set[str],
    domain_weight: int,
    base_domain_weight: int,
) -> tuple[build_lm.ChunkCounter, dict[str, dict[str, Counter[str]]], Counter[str], int, int]:
    """Count only allowed higher-order sources with builder-identical policy."""

    parts = temp_root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    seen = build_lm.SeenSentenceStore(temp_root / "seen.sqlite3")
    counter = build_lm.ChunkCounter(parts, 100_000)
    source_counts: dict[str, Counter[str]] = {}
    order_counts: dict[str, dict[str, Counter[str]]] = {
        "unigram": {},
        "higher_order": {},
    }
    filtered_by_source: Counter[str] = Counter()
    domain_overlay_tokens = 0

    def feed(source: str, records, weight: int, overlay_weight: int = 0) -> None:
        nonlocal domain_overlay_tokens
        source_counts.setdefault(source, Counter())
        for order in order_counts:
            order_counts[order].setdefault(source, Counter())
        for _path, _row, raw in records:
            source_counts[source]["rawRows"] += 1
            order_counts["unigram"][source]["rawRows"] += 1
            order_counts["higher_order"][source]["rawRows"] += 1
            if not isinstance(raw, str):
                source_counts[source]["invalidRows"] += 1
                continue
            for canonical, tokens in build_lm.iter_sentences(raw):
                source_counts[source]["candidateSentences"] += 1
                order_counts["unigram"][source]["candidateSentences"] += 1
                order_counts["higher_order"][source]["candidateSentences"] += 1
                if canonical in exclusions:
                    filtered_by_source[source] += 1
                    order_counts["unigram"][source]["filteredSentences"] += 1
                    order_counts["higher_order"][source]["filteredSentences"] += 1
                    continue
                if not seen.add(canonical):
                    source_counts[source]["dedupedSentences"] += 1
                    order_counts["unigram"][source]["dedupedSentences"] += 1
                    order_counts["higher_order"][source]["dedupedSentences"] += 1
                    continue
                counter.add_sentence(tokens, weight, include_higher=True)
                if overlay_weight:
                    domain_overlay_tokens += len(tokens) * overlay_weight
                source_counts[source]["keptSentences"] += 1
                order_counts["unigram"][source]["keptSentences"] += 1
                order_counts["higher_order"][source]["keptSentences"] += 1

    try:
        feed("news", build_lm.iter_news(news), 1)
        feed("wikipedia", build_lm.iter_wiki(wiki), 1)
        # Base domain weight is included in replacement T; explicit overlay
        # is an additional deterministic boost accounted for in the header.
        feed("domain", build_lm.iter_domain(domain), domain_weight, domain_weight - base_domain_weight)
        if vsec.exists():
            feed("vsec-train", build_lm.iter_vsec_train(vsec), 1)
    finally:
        seen.close()
    return counter, order_counts, filtered_by_source, domain_overlay_tokens, sum(
        counts.get("dedupedSentences", 0) for counts in source_counts.values()
    )


def combine_trigrams(
    counter: build_lm.ChunkCounter,
    vocab: set[str],
    unigrams: list[tuple[str, int]],
    minimum: int,
    limit: int,
) -> tuple[list[tuple[int, str]], int]:
    centers = build_lm.accent_evidence_centers(unigrams)
    return build_lm.bounded_top_trigrams_with_evidence(
        counter.parts, vocab, minimum, limit, centers
    )


def write_lm(
    path: Path,
    total_tokens: int,
    unigram: dict[str, int],
    bigram: dict[str, int],
    trigrams: list[tuple[int, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("#SMS-LM v1\n")
        handle.write(f"#tokens={total_tokens}\n")
        for key, value in sorted(unigram.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"U\t{key}\t{value}\n")
        for key, value in sorted(bigram.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"B\t{key}\t{value}\n")
        for value, key in trigrams:
            handle.write(f"T\t{key}\t{value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=build_lm.OUT_DIR / "lm-ngrams.tsv")
    parser.add_argument("--manifest", type=Path, default=build_lm.OUT_DIR / "lm-ngrams.manifest.json")
    parser.add_argument("--news-shards", type=int, default=4)
    parser.add_argument("--domain-overlay-weight", type=int, default=100)
    parser.add_argument("--max-trigrams", type=int, default=500_000)
    parser.add_argument("--tri-min", type=int, default=3)
    parser.add_argument("--max-lm-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else build_lm.ROOT / args.base
    manifest_path = args.manifest if args.manifest.is_absolute() else build_lm.ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_sha = build_lm.sha256_file(base)
    total, unigram, bigram, _base_tri = read_lm(base)
    unigrams = list(unigram.items())
    vocab = set(unigram)
    exclusions, exclusion_categories, exclusion_paths = build_lm.load_exclusions()
    news = build_lm.sorted_news_paths(args.news_shards)
    wiki = sorted(build_lm.WIKI_DIR.rglob("*.txt"))
    domain = build_lm.DOMAIN_SEED
    vsec = build_lm.VSEC_TRAIN
    base_domain_weight = int(manifest.get("config", {}).get("domain_weight", 5))
    if args.domain_overlay_weight < base_domain_weight:
        raise ValueError("domain overlay must be >= base domain weight")

    started = time.time()
    temp_root = Path(tempfile.mkdtemp(prefix="sms-lm-higher-", dir=str(build_lm.ARTIFACT_DIR)))
    try:
        counter, order_counts, filtered_by_source, overlay_tokens, deduped = scan_higher_sources(
            temp_root,
            news,
            wiki,
            domain,
            vsec,
            exclusions,
            args.domain_overlay_weight,
            base_domain_weight,
        )
        counter.flush()
        trigrams, retained = combine_trigrams(
            counter,
            vocab,
            unigrams,
            args.tri_min,
            args.max_trigrams,
        )
        lm_tmp = temp_root / "lm-ngrams.tsv"
        lex_tmp = temp_root / "lexicon-built.txt"
        write_lm(lm_tmp, total + overlay_tokens, unigram, bigram, trigrams)
        shutil.copy2(build_lm.OUT_DIR / "lexicon-built.txt", lex_tmp)
        if args.max_lm_bytes and lm_tmp.stat().st_size > args.max_lm_bytes:
            raise ValueError(
                f"staged LM is {lm_tmp.stat().st_size:,} bytes; max is {args.max_lm_bytes:,}"
            )
        parsed = build_lm.validate_lm(lm_tmp)

        config = dict(manifest.get("config", {}))
        config.update({
            "higher_order_rescan": True,
            "domain_overlay_weight": args.domain_overlay_weight,
            "retain_accent_evidence": True,
            "max_trigrams": args.max_trigrams,
        })
        counts = dict(manifest.get("counts", {}))
        prior_order_counts = manifest.get("counts", {}).get("orderCounts", {})
        counts["orderCounts"] = {
            "unigram": dict(prior_order_counts.get("unigram", {})),
            "higher_order": {
                source: dict(values) for source, values in order_counts["higher_order"].items()
            },
        }
        # Top-level source filtering reflects the full 17-shard build; the
        # higher-order rescan details remain in orderCounts above.
        counts["filtered_by_source"] = dict(
            manifest.get("counts", {}).get("filtered_by_source", filtered_by_source)
        )
        counts["weightedTokens"] = total + overlay_tokens
        counts["outputCounts"] = {
            **dict(counts.get("outputCounts", {})),
            "trigrams": len(trigrams),
            "retainedTrigrams": retained,
            "tokens": parsed["tokens"],
            "t": parsed.get("T", 0),
        }
        counts["trigrams"] = len(trigrams)
        counts["retainedTrigrams"] = retained
        counts["higherOrderRescanDedupedSentences"] = deduped
        counts["higherOrderRescanElapsedSeconds"] = round(time.time() - started, 2)
        manifest["createdAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["command"] = " ".join(__import__("sys").argv)
        manifest["config"] = config
        manifest["counts"] = counts
        manifest["exclusions"] = {
            **manifest.get("exclusions", {}),
            "categories": exclusion_categories,
            "totalCanonicalSentences": len(exclusions),
            "filteredBySource": dict(filtered_by_source),
            "postFilterExactIntersection": 0,
            "sourcePaths": [build_lm.relative(path) for path in exclusion_paths],
        }
        manifest["postProcess"] = {
            "type": "higher-order-rescan",
            "baseArtifact": {
                "path": build_lm.relative(base),
                "sha256": base_sha,
                "unigramCount": len(unigram),
                "bigramCount": len(bigram),
            },
            "newsShards": [build_lm.relative(path) for path in news],
            "otherSources": ["wikipedia", "domain", "vsec-train"],
            "rescanSourceFiles": [
                {
                    "path": build_lm.relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": build_lm.sha256_file(path),
                }
                for path in [*news, *wiki, domain, vsec]
                if path.exists()
            ],
            "baseDomainWeight": base_domain_weight,
            "domainOverlayWeight": args.domain_overlay_weight,
            "retentionPolicy": (
                "top-N trigrams plus below-cutoff rows whose center is an "
                "unaccented surface with accented siblings or the first "
                "accented sibling after the family leader"
            ),
            "holdoutSourcesExcluded": True,
        }
        manifest["outputs"]["lm"].update({
            "bytes": lm_tmp.stat().st_size,
            "sha256": build_lm.sha256_file(lm_tmp),
        })
        manifest["outputs"]["lexicon"].update({
            "bytes": lex_tmp.stat().st_size,
            "sha256": build_lm.sha256_file(lex_tmp),
        })
        if base.resolve() == (build_lm.OUT_DIR / "lm-ngrams.tsv").resolve():
            if build_lm.sha256_file(base) != base_sha:
                raise RuntimeError("base LM changed during higher-order rescan")
        backup = build_lm.replace_atomically(
            lm_tmp,
            lex_tmp,
            manifest,
            build_lm.OUT_DIR / "lm-ngrams.tsv",
            build_lm.OUT_DIR / "lexicon-built.txt",
            manifest_path,
        )
        print(json.dumps({
            "manifest": str(manifest_path),
            "backup": str(backup) if backup else None,
            "baseSha256": base_sha,
            "outputs": manifest["outputs"],
            "counts": counts,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())