// Compatibility entry point for the real VSEC benchmark adapter.
// The former corpus-derived VSEC simulator was retired to
// spelling_benchmark_adapter_legacy.mjs and is not used by benchmarks.
export * from './spelling_benchmark_adapter_vsec.mjs';

if (process.argv[1] && process.argv[1].endsWith('spelling_benchmark_adapter.mjs')) {
  const { adaptVsecSplit } = await import('./spelling_benchmark_adapter_vsec.mjs');
  const { writeFileSync } = await import('node:fs');
  const path = await import('node:path');
  const { fileURLToPath } = await import('node:url');
  const here = path.dirname(fileURLToPath(import.meta.url));
  const split = process.argv[2] ?? 'test';
  const outPath = process.argv[3]
    ? path.resolve(process.argv[3])
    : path.join(here, '..', 'benchmark', `corpus-vsec-${split}.json`);
  const output = adaptVsecSplit(split);
  writeFileSync(outPath, JSON.stringify(output, null, 2), 'utf8');
  console.log(`Saved ${output.rows.length} real VSEC ${split} rows to ${outPath}`);
  console.log(`Expected labels: ${output.meta.expectedIssueCount}`);
}
