import { createDefaultEngine, ValidationContext } from '../../src/engine.mjs';

const mk = () => createDefaultEngine();
const v = (eng, text, mode = 'ACCENTED', brand = 'VT_TENDOO') =>
  eng.validate(new ValidationContext(text, mode, brand)).issues;

// 1. fresh engine, L03
console.log('fresh L03:', v(mk(), 'Vui long kiem tra va xac nhan thong tin').length);

// 2. shared engine sequence
const e2 = mk();
console.log('seq A L01:', v(e2, 'Kinh chao quy khach hang').length);
console.log('seq B L03:', v(e2, 'Vui long kiem tra va xac nhan thong tin').length);

// 3. fresh engine: 'long' vs ' long'
const e3 = mk();
console.log("fresh 'long':", v(e3, 'long').length);
console.log("fresh ' long':", v(e3, ' long').length);
for (const i of v(e3, ' long')) console.log('   ', JSON.stringify(i));

// 4. reversed order on another fresh engine
const e4 = mk();
console.log("rev ' long':", v(e4, ' long').length);
console.log("rev 'long':", v(e4, 'long').length);

// 5. same engine, repeat identical input twice
const e5 = mk();
console.log('rep1 L03:', v(e5, 'Vui long kiem tra').length);
console.log('rep2 L03:', v(e5, 'Vui long kiem tra').length);
