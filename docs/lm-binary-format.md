# LM Binary Format v1 (`VILM1`)

Deterministic, compact deployment format for the Vietnamese SMS trigram
language model (plan Task 11, spelling-engine-optimization).
Source of truth: `src/data/lm-ngrams.tsv` (built by `tools/build_lm.py`,
audited by `tools/audit_lm.py`). The exporter NEVER rebuilds corpora — it
converts the audited TSV and records its hash.

## Goals

- Load in Node without generating millions of line strings (plan Task 13).
- Word-ID + typed-array friendly layout for RSS <= 250 MB.
- Byte-deterministic output: same TSV -> identical bytes (sorted keys).
- Tamper-evident: magic, version, per-section CRC32, manifest SHA-256.

## File layout

All integers are **little-endian**, packed, no alignment padding.

```
offset  size  field
0       4     magic          ASCII "VILM"
4       4     version        uint32, currently 1
8       4     flags          uint32, bit0 set => counts section order is
                             ascending id tuples (always set by exporter v1)
12      8     tokenTotal     uint64 weighted token count (TSV #tokens header)
16      4     vocabCount     uint32 number of distinct words
20      4     sectionCount   uint32 number of sections that follow (=3)

then sectionCount sections, each:

  1     kind           uint8    1=UNIGRAM 2=BIGRAM 3=TRIGRAM
  4     rowCount       uint32
  4     bytesLen       uint32   payload byte length (excludes this header)
  4     crc32          uint32   CRC-32 (IEEE) of payload bytes
  ...   payload

Payloads:

  VOCAB STRING TABLE (implicit first — see below)
  UNIGRAM : rowCount x (uint32 wordId, uint64 count)
  BIGRAM  : rowCount x (uint32 w1, uint32 w2, uint64 count)
  TRIGRAM : rowCount x (uint32 w1, uint32 w2, uint32 w3, uint64 count)

The FIRST section (kind=0, not counted in sectionCount) is the vocabulary
string table:

  1     kind           uint8    0=VOCAB
  4     rowCount       uint32   = vocabCount
  4     bytesLen       uint32
  4     crc32          uint32
  ...   payload = offsets[(vocabCount+1)]xuint32 followed by UTF-8 bytes;
        word i occupies bytes offsets[i]..offsets[i+1]; wordId = index in
        lexicographic (byte-wise UTF-8) order.
```

## Determinism rules

- Vocabulary sorted by UTF-8 byte order of the NFC-lowercased key (the TSV
  key form); ids assigned 0..vocabCount-1 in that order.
- Unigram rows sorted ascending by wordId.
- Bigram rows sorted ascending lexicographically by (w1, w2).
- Trigram rows sorted ascending lexicographically by (w1, w2, w3).
- No timestamps, no environment data inside the file.

## Validation contract (loader AND exporter)

Reject (hard error, never fallback):

- wrong magic or unsupported version;
- truncated file / any section whose payload is shorter than declared;
- CRC32 mismatch on any section;
- row decode overrun (payload length != rowCount * recordSize);
- wordId >= vocabCount anywhere;
- duplicate keys within a section;
- tokenTotal/vocabCount inconsistencies vs section contents where checkable.

## Manifest sidecar (`<file>.manifest.json`)

```json
{
  "artifact": "lm-ngrams.v1.bin",
  "version": 1,
  "tokenTotal": 1325600394,
  "counts": { "U": 200000, "B": 1000000, "T": 2760041 },
  "bytes": <int>,
  "sha256": "<hex>",
  "sourceTsv": "src/data/lm-ngrams.tsv",
  "sourceTsvSha256": "<hex>",
  "createdAt": "<ISO-8601>"
}
```

## Exporter / reader

**Exporter** — `tools/export_lm_binary.py`

- Input defaults to `src/data/lm-ngrams.tsv`, output to
  `dataset_artifacts/lm/lm-ngrams.v1.bin`.
- Writes atomically (temp file + rename) and records the manifest sidecar.
- Refuses to overwrite an existing output unless `--force`.
- Validates every row while converting (tab structure, positive integer
  counts, unique keys, wordId bounds); any violation aborts before writing.

**Reader** — Node implementation lands in `src/lm/binary-backend.mjs`
(Task 12). The Python module also exposes `read_sections()` so round-trip
tests can verify exact count equivalence without Node.
