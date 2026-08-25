import { readFileSync } from 'node:fs';
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const eng = createDefaultEngine();
const vals = (t) => eng.validate(new ValidationContext(t, 'ACCENTED', 'VT_TENDOO'))
  .issues.filter((i) => i.ruleId === 'POSSIBLE_MISSING_DIACRITIC')
  .map((i) => i.value + '->' + i.suggestions[0]);

console.log('A "tot dep"           :', JSON.stringify(vals('tot dep')));
console.log('B "hom nay giam den"  :', JSON.stringify(vals('hom nay giam den')));
console.log('C "giam den 50%"      :', JSON.stringify(vals('giam den 50%')));
console.log('D "qua tang dep"      :', JSON.stringify(vals('qua tang dep')));
console.log('E candidates("truong"):',
  JSON.stringify(eng.accentIndex.candidates('truong').map((c) => c.word)));
console.log('F candidates("nghi")  :',
  JSON.stringify(eng.accentIndex.candidates('nghi').map((c) => c.word)));
console.log('G contains(giam/ban)  :', eng.lexicon.contains('giam'), eng.lexicon.contains('ban'));

const lines = readFileSync(new URL('../../src/data/lexicon.txt', import.meta.url), 'utf8')
  .split(/\r?\n/);
lines.forEach((l, i) => {
  if (/^(giam|ban|dep)\t/.test(l)) console.log('lexicon line', i + 1, ':', JSON.stringify(l));
});
