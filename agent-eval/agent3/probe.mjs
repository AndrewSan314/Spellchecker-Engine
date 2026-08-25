// Probe harness — try candidate rows against the engine BEFORE finalizing the corpus.
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const engine = createDefaultEngine();

export function validate(text, mode = 'ACCENTED', brand = null) {
  const res = engine.validate(new ValidationContext(text, mode, brand));
  return res.issues.map((i) => ({
    ruleId: i.ruleId,
    sev: i.severity,
    start: i.start,
    end: i.end,
    value: i.value,
    conf: i.confidence,
    msg: i.message,
  }));
}

export function show(label, text, mode = 'ACCENTED', brand = null) {
  const issues = validate(text, mode, brand);
  console.log(`\n=== ${label} [${mode}${brand ? '/' + brand : ''}] ===`);
  console.log(JSON.stringify(text));
  if (issues.length === 0) {
    console.log('  CLEAN ✔');
  } else {
    for (const i of issues) {
      console.log(`  ✘ ${i.ruleId} @${i.start}-${i.end} "${i.value}"${i.conf != null ? ' conf=' + i.conf : ''}`);
      console.log(`     ${i.msg}`);
    }
  }
}
