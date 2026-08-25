import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const engine = createDefaultEngine();

function show(label, text, mode, brand, opts) {
  const ctx = new ValidationContext(text, mode, brand, null, opts);
  const res = engine.validate(ctx);
  console.log(`--- ${label} ---`);
  console.log(`  ctor ok, issues=${res.issues.length}`);
  for (const i of res.issues) {
    console.log(`   ${i.ruleId} [${i.start},${i.end}) value=${JSON.stringify(i.value)} sug=${JSON.stringify(i.suggestions)} conf=${i.confidence}`);
  }
}

show('A: Vui long kiem tra / ACCENTED / no brand', 'Vui long kiem tra', 'ACCENTED', null);
show('B: Vui long kiem tra / ACCENTED / VT_TENDOO', 'Vui long kiem tra', 'ACCENTED', 'VT_TENDOO');
show('C: L03 full / ACCENTED / VT_TENDOO', 'Vui long kiem tra va xac nhan thong tin', 'ACCENTED', 'VT_TENDOO');

// offset probe on a single word
const r1 = engine.validate(new ValidationContext('long', 'ACCENTED', 'VT_TENDOO'));
const r2 = engine.validate(new ValidationContext(' long', 'ACCENTED', 'VT_TENDOO'));
console.log('S01 original issues:', r1.issues.length, 'shifted issues:', r2.issues.length);
console.log('Array.isArray:', Array.isArray(r1.issues), Array.isArray(r2.issues));
