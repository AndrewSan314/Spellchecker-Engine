# Benchmark label scope

VSEC and Viwiki rows annotate spelling errors only. They therefore use:

```json
"fullyLabeledRuleIds": ["POSSIBLE_SPELLING_ERROR"]
```

`benchmark/run-benchmark.mjs` counts unmatched remaining issues as false
positives only when their rule ID is in that list. The legacy
`"fullyLabeled": true` field remains supported and means every rule. This
keeps formatting/style diagnostics on spelling rows from being misreported as
annotation false positives while still detecting extra spelling issues.
