#!/usr/bin/env python3
"""Task 11 — deterministic TSV -> binary LM exporter (format VILM v1).

Spec: docs/lm-binary-format.md. Converts the AUDITED TSV artifact only; it
never rebuilds corpora. Output is byte-deterministic, atomic, and carries a
manifest sidecar with source/artifact SHA-256.

Usage:
    python tools/export_lm_binary.py [--in src/data/lm-ngrams.tsv] \
        [--out dataset_artifacts/lm/lm-ngrams.v1.bin] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import zlib

MAGIC = b"VILM"
VERSION = 1
SEC_VOCAB = 0
SEC_UNIGRAM = 1
SEC_BIGRAM = 2
SEC_TRIGRAM = 3
REC_FMT = {
    SEC_UNIGRAM: struct.Struct("<IQ"),
    SEC_BIGRAM: struct.Struct("<IIQ"),
    SEC_TRIGRAM: struct.Struct("<IIIQ"),
}
HDR = struct.Struct("<BIII")  # kind, rowCount, bytesLen, crc32


def _fail(msg: str) -> None:
    raise ValueError(msg)


def parse_tsv(path: str):
    """Strictly parse the audited TSV. Yields ('U'|'B'|'T', key, count)."""
    token_total = None
    seen = {"U": set(), "B": set(), "T": set()}
    last_rank = 0
    rank_of = {"U": 1, "B": 2, "T": 3}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("#tokens="):
                    if token_total is not None:
                        _fail(f"line {line_no}: duplicate #tokens header")
                    try:
                        token_total = int(line.split("=", 1)[1])
                    except ValueError:
                        _fail(f"line {line_no}: bad #tokens header")
                    if token_total <= 0:
                        _fail(f"line {line_no}: #tokens must be positive")
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                _fail(f"line {line_no}: expected <kind>\\t<key>\\t<count>")
            kind, key, count_str = parts
            rank = rank_of.get(kind)
            if rank is None:
                _fail(f"line {line_no}: invalid record kind {kind!r}")
            if rank < last_rank:
                _fail(f"line {line_no}: section order violation at {kind!r}")
            if key in seen[kind]:
                _fail(f"line {line_no}: duplicate key in {kind} section")
            try:
                count = int(count_str)
            except ValueError:
                _fail(f"line {line_no}: count must be an integer")
            if count <= 0:
                _fail(f"line {line_no}: count must be positive")
            seen[kind].add(key)
            last_rank = rank
            yield kind, key, count, token_total


def export_lm(tsv_path: str, out_path: str, force: bool = False) -> dict:
    """Convert the audited TSV into a VILM v1 binary + manifest sidecar."""
    if os.path.exists(out_path) and not force:
        raise RuntimeError(
            f"refusing to overwrite existing artifact {out_path} (use --force)")

    unigrams: list[tuple[str, int]] = []
    bigrams: list[tuple[str, str, int]] = []
    trigrams: list[tuple[str, str, str, int]] = []
    token_total = None
    for kind, key, count, tt in parse_tsv(tsv_path):
        token_total = token_total or tt
        if kind == "U":
            unigrams.append((key, count))
        elif kind == "B":
            words = key.split(" ")
            if len(words) != 2:
                _fail(f"bigram key must hold two words: {key!r}")
            bigrams.append((words[0], words[1], count))
        else:
            words = key.split(" ")
            if len(words) != 3:
                _fail(f"trigram key must hold three words: {key!r}")
            trigrams.append((words[0], words[1], words[2], count))
    if token_total is None:
        _fail("missing #tokens=<int> header")

    # vocabulary: every distinct word across all sections, UTF-8 byte order
    vocab_set = {w for w, _c in unigrams}
    vocab_set.update(w for a, b, _ in bigrams for w in (a, b))
    vocab_set.update(w for a, b, c, _ in trigrams for w in (a, b, c))
    vocab = sorted(vocab_set, key=lambda w: w.encode("utf-8"))
    word_id = {w: i for i, w in enumerate(vocab)}

    def ids(*words_):
        for w in words_:
            i = word_id.get(w)
            if i is None:  # cannot happen by construction
                _fail(f"internal: unknown word {w!r}")
        return tuple(word_id[w] for w in words_)

    uni_rows = sorted(
        ((*ids(w), c) for w, c in unigrams), key=lambda r: r[0])
    bi_rows = sorted(
        ((*ids(a, b), c) for a, b, c in bigrams), key=lambda r: (r[0], r[1]))
    tri_rows = sorted(
        ((*ids(a, b, c), c2) for a, b, c, c2 in trigrams),
        key=lambda r: (r[0], r[1], r[2]))

    # vocabulary payload: offsets table then concatenated UTF-8 bytes
    blob = bytearray()
    offsets = [0]
    for w in vocab:
        blob += w.encode("utf-8")
        offsets.append(len(blob))
    vocab_payload = struct.pack(f"<{len(offsets)}I", *offsets) + bytes(blob)

    def section_bytes(kind: int, row_count: int, length: int, crc: int,
                      payload: bytes) -> bytes:
        return HDR.pack(kind, row_count, length, crc) + payload

    out = bytearray(MAGIC)
    out += struct.pack("<II", VERSION, 1)      # version, flags(bit0=sorted)
    out += struct.pack("<Q", token_total)
    out += struct.pack("<I", len(vocab))
    out += struct.pack("<I", 4)                # sectionCount incl. VOCAB
    out += section_bytes(SEC_VOCAB, len(vocab), len(vocab_payload),
                         zlib.crc32(vocab_payload) & 0xFFFFFFFF,
                         vocab_payload)
    for kind, rows, fmt in (
        (SEC_UNIGRAM, uni_rows, REC_FMT[SEC_UNIGRAM]),
        (SEC_BIGRAM, bi_rows, REC_FMT[SEC_BIGRAM]),
        (SEC_TRIGRAM, tri_rows, REC_FMT[SEC_TRIGRAM]),
    ):
        payload = bytearray()
        for row in rows:
            payload += fmt.pack(*row)
        payload = bytes(payload)
        out += section_bytes(kind, len(rows), len(payload),
                             zlib.crc32(payload) & 0xFFFFFFFF, payload)

    body = bytes(out)
    sha256 = hashlib.sha256(body).hexdigest()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out_path)),
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    src_sha = hashlib.sha256(open(tsv_path, "rb").read()).hexdigest()
    try:
        source_tsv_ref = os.path.relpath(tsv_path).replace(os.sep, "/")
    except ValueError:
        # different drive on Windows — fall back to a normalized absolute path
        source_tsv_ref = os.path.abspath(tsv_path).replace(os.sep, "/")
    manifest = {
        "artifact": os.path.basename(out_path),
        "version": VERSION,
        "tokenTotal": token_total,
        "counts": {"U": len(unigrams), "B": len(bigrams), "T": len(trigrams)},
        "bytes": len(body),
        "sha256": sha256,
        "sourceTsv": source_tsv_ref,
        "sourceTsvSha256": src_sha,
    }
    with open(out_path + ".manifest.json", "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return manifest


def read_sections(path: str) -> dict:
    """Validate and decode a VILM file back to plain-Python structures."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != MAGIC:
        _fail("bad magic bytes (not a VILM file)")
    version, flags = struct.unpack_from("<II", raw, 4)
    if version != VERSION:
        _fail(f"unsupported VILM version {version}")
    token_total = struct.unpack_from("<Q", raw, 12)[0]
    vocab_count = struct.unpack_from("<I", raw, 20)[0]
    section_count = struct.unpack_from("<I", raw, 24)[0]
    pos = 28

    sections: dict[int, tuple[int, bytes]] = {}
    for _ in range(section_count):
        if pos + HDR.size > len(raw):
            _fail("truncated section header")
        kind, row_count, length, crc = HDR.unpack_from(raw, pos)
        pos += HDR.size
        if pos + length > len(raw):
            _fail(f"truncated section payload (kind={kind})")
        payload = raw[pos:pos + length]
        pos += length
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            _fail(f"crc32 mismatch in section kind={kind}")
        sections[kind] = (row_count, payload)
    if pos != len(raw):
        _fail("trailing bytes after final section")

    # vocabulary
    row_count, payload = sections[SEC_VOCAB]
    if row_count != 0 and row_count != vocab_count:
        _fail("vocab rowCount mismatch")
    need = (vocab_count + 1) * 4
    if len(payload) < need:
        _fail("vocab offsets truncated")
    offsets = struct.unpack_from(f"<{vocab_count + 1}I", payload, 0)
    blob = payload[need:]
    if offsets[-1] != len(blob):
        _fail("vocab string table length mismatch")
    vocab = [
        blob[offsets[i]:offsets[i + 1]].decode("utf-8")
        for i in range(vocab_count)
    ]

    def decode(kind: int, arity: int) -> dict:
        row_count_, payload_ = sections[kind]
        fmt = REC_FMT[kind]
        if len(payload_) != fmt.size * row_count_:
            _fail(f"section kind={kind} payload/rowCount mismatch")
        out_: dict = {}
        for off in range(0, len(payload_), fmt.size):
            fields = fmt.unpack_from(payload_, off)
            *ids_, cnt = fields
            for wid in ids_:
                if wid >= vocab_count:
                    _fail(f"wordId {wid} out of range in section kind={kind}")
            key = tuple(vocab[wid] for wid in ids_)
            if key in out_:
                _fail(f"duplicate decoded key {key!r}")
            out_[key] = cnt
        return out_

    data = {
        "version": version,
        "flags": flags,
        "tokenTotal": token_total,
        "vocabCount": vocab_count,
        "vocab": vocab,
        "unigrams": decode(SEC_UNIGRAM, 1),
        "bigrams": decode(SEC_BIGRAM, 2),
        "trigrams": decode(SEC_TRIGRAM, 3),
    }
    if len(data["unigrams"]) != vocab_count:
        _fail("unigram rows must cover the whole vocabulary")
    return data


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src",
                    default=os.path.join(root, "src", "data", "lm-ngrams.tsv"))
    ap.add_argument("--out", dest="out", default=os.path.join(
        root, "dataset_artifacts", "lm", "lm-ngrams.v1.bin"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    man = export_lm(args.src, args.out, force=args.force)
    print(json.dumps(man, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
