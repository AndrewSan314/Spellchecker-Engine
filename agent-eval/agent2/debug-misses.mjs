// Per-miss forensics: run the exact sentences, print the [pmd] traces.
import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const cases = [
  ['L05', 'Uu dai dac biet hom nay giam den 50%', 'giam'],
  ['L08', 'Xac nhan don hang so DH8823 trong hom nay', 'so'],
  ['L10', 'Chuc ban mot ngay tot dep', 'ban,dep'],
  ['L15', 'Gio mo cua 8h-22h hang ngay', 'hang'],
  ['M02', 'Quý khách vui lòng nhan qua tang tu Tendoo', 'tu'],
  ['M03', 'Xe giao hàng sẽ đến sau mot phut nua, mong anh thong cam', 'mot'],
];

const engine = createDefaultEngine();
for (const [id, text] of cases) {
  console.log(`\n===== ${id}: "${text}" =====`);
  process.env.SMS_VAL_DEBUG = '1';
  const res = engine.validate(new ValidationContext(text, 'ACCENTED', 'VT_TENDOO'));
  delete process.env.SMS_VAL_DEBUG;
  for (const i of res.issues) {
    if (i.ruleId === 'POSSIBLE_MISSING_DIACRITIC') {
      console.log(`   FIRED ${JSON.stringify(i.value)} -> ${JSON.stringify(i.suggestions)} conf=${i.confidence}`);
    }
  }
}
