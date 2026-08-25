# Leakage-safe spelling data artifacts

Raw downloads stay outside this directory and are never rewritten.

## Rebuild deterministic VSEC splits

```powershell
python tools/split_spelling_datasets.py
python tools/split_spelling_datasets.py --audit --external-benchmark benchmark/corpus-viwiki-spelling.json
node tools/spelling_benchmark_adapter.mjs test
```

The default seed is `20260824`; VSEC is grouped by normalized noisy/corrected
text, annotation error template, and variable-masked template before an
80/10/10 whole-group assignment. Outputs are `vsec-{train,dev,test}.jsonl`,
`vsec-index.jsonl`, and `vsec-manifest.json`.

## Build clean source splits and held-out synthetic evaluation

```powershell
node tools/corpus_pipeline.mjs
```

The pipeline uses only domain seed files and writes
`clean-source/clean-{train,dev,test}.txt`. `src/data/corpus-train.txt` contains
the clean-train split; `benchmark/corpus-synthetic-diacritics.json` is mutated
only from clean-test and records source fingerprints/seed. Viwiki-Spelling is
converted separately as `benchmark/corpus-viwiki-spelling.json` and is always
`external-test`.

Fast audit:

```powershell
python -m unittest test.test_spelling_dataset_split -v
```
