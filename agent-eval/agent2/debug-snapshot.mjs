import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';
const engine = createDefaultEngine();
const cases = [
  ['L06', 'Gui tang ban qua tang sinh nhat'],
  ['L13', 'Thanh toan goi cuoc 3G toc do cao'],
];
for (const [id, text] of cases) {
  console.log(`===== ${id}: "${text}" =====`);
  process.env.SMS_VAL_DEBUG = '1';
  engine.validate(new ValidationContext(text, 'ACCENTED', 'VT_TENDOO'));
  delete process.env.SMS_VAL_DEBUG;
}
