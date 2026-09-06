import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const args = Object.fromEntries(process.argv.slice(2).map((arg) => {
  const [key, ...rest] = arg.replace(/^--/, '').split('=');
  return [key, rest.join('=')];
}));

const dataset = JSON.parse(fs.readFileSync(path.resolve(args.dataset), 'utf8'));
const repo = path.resolve(args.repo);
const engineModule = await import(pathToFileURL(path.join(repo, 'src', 'engine.mjs')));
const { createDefaultEngine, ValidationContext } = engineModule;

const loadStart = performance.now();
const engine = createDefaultEngine();
const loadMs = performance.now() - loadStart;

for (const row of dataset.slice(0, 3)) {
  engine.validate(new ValidationContext(row.content, row.messageMode, row.brand ?? null));
}
console.log(JSON.stringify({
  __meta: true,
  loadMs,
  rssAfterLoad: process.memoryUsage().rss,
  warmupCount: 3,
  node: process.version,
}));

for (const row of dataset) {
  const start = performance.now();
  const result = engine.validate(new ValidationContext(
    row.content,
    row.messageMode,
    row.brand ?? null,
  ));
  const durationMs = performance.now() - start;
  console.log(JSON.stringify({
    id: row.id,
    category: row.category,
    domain: row.domain,
    input: row.content,
    messageMode: row.messageMode,
    groundTruth: {
      clean: row.clean,
      correctionPairs: row.expect ?? [],
      protectedSpans: row.protected_spans ?? [],
    },
    issues: result.issues.map((issue) => ({
      ruleId: issue.ruleId,
      severity: issue.severity,
      start: issue.start,
      end: issue.end,
      value: issue.value,
      suggestions: issue.suggestions ?? [],
      confidence: issue.confidence,
      message: issue.message,
    })),
    durationMs,
  }));
}
